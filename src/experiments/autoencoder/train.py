r"""Autoencoder training entry (Lightning).

Runs are identified by a run id and stored under ``artifacts/autoencoders/<runid>/0``
(the fixed ``0`` matches the lap directory used by downstream scripts). Full
Lightning checkpoints live in ``<runpath>/lightning/``, while
``model_last.pth`` / ``model_best.pth`` / ``config.yaml`` / ``metadata.yaml``
keep the pre-Lightning artifact layout intact.

Usage (from the repository root, no torchrun needed):

    CUDA_VISIBLE_DEVICES=0 SWANLAB_MODE=disabled python -m appa.launchers ae-train \
        train.update_steps=2000

Multi-GPU (DDP processes spawned by Lightning):

    APPA_DEVICES=4 python -m appa.launchers ae-train
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
from appa.training.datamodules import ERA5DataModule
from appa.training.entry import PRECISION_MAP, samples_to_batches, val_check_fraction
from appa.training.loggers import build_logger
from appa.training.optim import get_optimizer  # noqa: F401 (re-exported for convenience)
from appa.utils.io import safe_load
from appa.utils.seeding import seed_everything
from experiments.autoencoder.module import AETrainingModule

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "train.yaml")


def train(
    runid: str,
    cfg: DictConfig,
    resume=None,
    fork=None,
    devices=None,
):
    r"""Trains an autoencoder with PyTorch Lightning.

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

    grad_acc = cfg.train.batch_size_per_step // (cfg.train.batch_element_per_gpu * world)

    # Data / model
    dm = ERA5DataModule(cfg, world_size=world)
    module = AETrainingModule(cfg, runid=runid, quali_dataset=dm.quali_dataset)

    if fork is not None:
        ckpt = safe_load(os.path.join(fork, "model_last.pth"))
        module.ae.load_state_dict(ckpt, strict=False)

    runpath = PATH_AE / runid / "0"
    runpath.mkdir(parents=True, exist_ok=True)

    # Tracker + recorded configuration
    config = OmegaConf.to_container(cfg, resolve=True)
    config["num_model_params"] = sum(p.numel() for p in module.ae.parameters())
    config["path"] = str(runpath)
    config["hardware"] = {
        "num_nodes": 1,
        "num_gpus": world,
        "num_cpus": len(os.sched_getaffinity(0)),
    }
    config["ae"]["latent_shape"] = list(module.ae.latent_shape)

    run_name = f"{runid} {cfg.ae.name}"
    description = f"AE {cfg.ae.name} " + module.ae.description()

    if fork is not None:
        description = f"Forked from {fork}.\n" + description

    logger = build_logger(
        cfg,
        runid=runid,
        project="appa_ae",
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
            get_model=lambda m: m.ae,
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
        help="Resume the given runid (e.g. abc123) from its last checkpoint.",
    )
    parser.add_argument(
        "--fork",
        type=str,
        default=None,
        help="Start a new run initialized from the weights of the given runid.",
    )

    args = parser.parse_args()

    import os

    from appa.training.entry import get_or_create_runid

    if args.continue_ and args.fork:
        raise SystemExit("Cannot continue and fork at the same time.")

    if args.continue_:
        runid = args.continue_
        config_path = PATH_AE / runid / "0" / "config.yaml"
        resume = PATH_AE / runid / "0" / "lightning" / "last.ckpt"
    elif args.fork:
        runid = get_or_create_runid()
        config_path = PATH_AE / args.fork / "0" / "config.yaml"
        resume = None
    else:
        runid = get_or_create_runid()
        config_path = CONFIG_PATH
        resume = None

    os.environ["APPA_RUNID"] = runid

    cfg = compose(config_file=config_path, overrides=args.overrides)

    fork = PATH_AE / args.fork / "0" if args.fork else None

    print(f"Starting AE training with runid: {runid}")

    train(
        runid,
        cfg,
        resume=resume if args.continue_ else None,
        fork=fork,
        devices=args.devices,
    )
