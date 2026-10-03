r"""Latent diffusion (denoiser) training entry (Lightning).

Runs are identified by a run id and stored under
``artifacts/autoencoders/<ae_run>/latents/<dump>/denoisers/<runid>/0`` (the
fixed ``0`` matches the lap directory used by downstream scripts). Full
Lightning checkpoints live in ``<runpath>/lightning/``, while
``model_last.pth`` / ``model_best.pth`` / ``config.yaml`` / ``metadata.yaml``
keep the pre-Lightning artifact layout intact.

Usage (from the repository root, no torchrun needed):

    CUDA_VISIBLE_DEVICES=0 SWANLAB_MODE=disabled python -m appa.launchers den-train \
        ae_run=<runid>/0 latent_dump=<dump_id> train.update_steps=2000

Multi-GPU (DDP processes spawned by Lightning):

    APPA_DEVICES=4 python -m appa.launchers den-train
"""

import dask
import os
import torch

from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf

from appa.config import PATH_AE
from appa.config.hydra import compose
from appa.training.callbacks import (
    ArtifactCheckpointCallback,
    KillSwitchCallback,
    SnapshotLoggingCallback,
)
from appa.training.datamodules import LatentBlanketDataModule
from appa.training.entry import PRECISION_MAP, samples_to_batches, val_check_fraction
from appa.training.loggers import build_logger
from appa.utils.io import safe_load
from appa.utils.seeding import seed_everything
from experiments.diffusion.module import DenoiserTrainingModule

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "train.yaml")


def train(
    runid: str,
    cfg: DictConfig,
    resume=None,
    fork=None,
    devices=None,
):
    r"""Trains a latent diffusion model with PyTorch Lightning.

    Arguments:
        runid: The run id.
        cfg: The training configuration as a DictConfig.
        resume: An optional checkpoint path to resume from.
        fork: An optional run path whose model weights initialize the run.
        devices: The number of GPUs to use (default: ``APPA_DEVICES`` or all
            visible devices).
    """

    dask.config.set(scheduler="synchronous")
    torch.set_float32_matmul_precision("high")

    # Reproducibility (identical across ranks; disable with train.seed=null)
    seed_everything(cfg.train.get("seed", None) or -1)

    devices = (
        devices
        or int(os.environ.get("APPA_DEVICES", "0") or 0)
        or (torch.cuda.device_count() or 1)
    )
    world = max(devices, 1)

    assert cfg.train.batch_size_per_step % (cfg.train.batch_element_per_gpu * world) == 0, (
        "Batch size must be divisible by number of GPU * samples per GPU"
    )

    assert cfg.ae_run is not None, "Autoencoder run id and lap must be specified."
    assert cfg.latent_dump is not None, "Latent dump id must be specified."

    grad_acc = cfg.train.batch_size_per_step // (cfg.train.batch_element_per_gpu * world)

    ae_cfg = compose(config_file=PATH_AE / cfg.ae_run / "config.yaml")

    # Data / model
    dm = LatentBlanketDataModule(cfg, ae_cfg, world_size=world)
    module = DenoiserTrainingModule(
        cfg,
        ae_cfg,
        runid=runid,
        valid_dataset=dm.valid_dataset,
        snapshot_indices=dm.snapshot_indices,
    )

    latent_folder = PATH_AE / cfg.ae_run / "latents" / cfg.latent_dump

    if fork is not None:
        ckpt = safe_load(os.path.join(fork, "model_last.pth"))
        module.model.load_state_dict(ckpt)

    runpath = latent_folder / "denoisers" / runid / "0"
    runpath.mkdir(parents=True, exist_ok=True)

    # Tracker + recorded configuration
    config = OmegaConf.to_container(cfg, resolve=True)
    config["num_model_params"] = sum(p.numel() for p in module.model.parameters())
    config["path"] = str(runpath)
    config["ae_path"] = str(PATH_AE / cfg.ae_run)
    config["latent_dump_path"] = str(latent_folder)
    config["hardware"] = {
        "num_nodes": 1,
        "num_gpus": world,
        "num_cpus": len(os.sched_getaffinity(0)),
    }

    ae_id = cfg.ae_run.split("/")[-2]
    run_name = f"{ae_id} K{cfg.train.blanket_size} {cfg.backbone.name}"
    description = f"Denoiser {cfg.backbone.name} " + module.model.description()

    if fork is not None:
        description = f"Forked from {fork}.\n" + description

    logger = build_logger(
        cfg,
        runid=runid,
        project="appa_denoiser",
        run_name=run_name,
        description=description,
        config=config,
        save_dir=runpath,
    )

    if not (runpath / "config.yaml").exists():
        OmegaConf.save(OmegaConf.create(config), runpath / "config.yaml")

    # Restore the best validation score when resuming
    best = float("inf")
    metadata_path = runpath / "metadata.yaml"

    if metadata_path.exists():
        metadata = OmegaConf.load(metadata_path)
        best = float(metadata.get("best_val_loss", float("inf")))

    callbacks = [
        KillSwitchCallback(runpath),
        ArtifactCheckpointCallback(
            runpath=runpath,
            runid=runid,
            get_model=lambda m: m.model,
            every_n_train_batches=samples_to_batches(
                cfg.train.save_interval, cfg.train.batch_size_per_step, grad_acc
            ),
            monitor="valid/losses/mean",
            best_val_loss=best,
        ),
        SnapshotLoggingCallback(logger),
    ]

    valid_batches = samples_to_batches(
        cfg.valid.log_interval, cfg.train.batch_size_per_step, grad_acc
    )
    num_batches = max(len(dm.train_dataset) // world, 1)

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        strategy="ddp" if world > 1 else "auto",
        precision=PRECISION_MAP[cfg.train.precision],
        max_steps=cfg.train.update_steps,
        num_sanity_val_steps=0,
        val_check_interval=val_check_fraction(valid_batches, num_batches),
        log_every_n_steps=max(cfg.train.log_interval // cfg.train.batch_size_per_step, 1),
        # Lightning >= 2.6 treats logger=None as "default logger"; logger=False
        # is required to actually disable tracking.
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # Kill switch marker (remove the file to gracefully stop the run)
    open(runpath / ".running", "a").close()

    trainer.fit(module, datamodule=dm, ckpt_path=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", type=str, help="Hydra overrides.")
    parser.add_argument("--devices", type=int, default=None, help="Number of GPUs to use.")
    parser.add_argument(
        "--continue",
        dest="continue_",
        type=str,
        default=None,
        help="Resume the given denoiser run path (e.g. <ae_run>#<dump>#<runid>) "
        "from its last checkpoint.",
    )
    parser.add_argument(
        "--fork",
        type=str,
        default=None,
        help="Start a new run initialized from the weights of the given run path.",
    )

    args = parser.parse_args()

    import os

    from appa.training.entry import get_or_create_runid

    if args.continue_ and args.fork:
        raise SystemExit("Cannot continue and fork at the same time.")

    sep = "#"

    if args.continue_ or args.fork:
        ae_run, latent_dump, forkid = (args.continue_ or args.fork).split(sep)
        config_path = PATH_AE / ae_run / "latents" / latent_dump / "denoisers" / forkid / "0"
        config_path = config_path / "config.yaml"
    else:
        runid = get_or_create_runid()
        config_path = CONFIG_PATH

    cfg = compose(config_file=config_path, overrides=args.overrides)

    if args.continue_:
        runid = forkid
        runpath = PATH_AE / cfg.ae_run / "latents" / cfg.latent_dump / "denoisers" / runid / "0"
        resume = runpath / "lightning" / "last.ckpt"
        fork = None
    elif args.fork:
        runid = get_or_create_runid()
        resume = None
        fork = PATH_AE / ae_run / "latents" / latent_dump / "denoisers" / forkid / "0"
    else:
        resume = None
        fork = None

    os.environ["APPA_RUNID"] = runid

    print(f"Starting denoiser training with runid: {runid}")

    train(runid, cfg, resume=resume, fork=fork, devices=args.devices)
