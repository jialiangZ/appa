r"""Command-line launchers for local (no-Slurm) runs.

Usage with torchrun (from the repository root or anywhere):

    torchrun --standalone --nproc-per-node 1 -m appa.launchers ae-train [overrides...]
    torchrun --standalone --nproc-per-node 1 -m appa.launchers den-train [overrides...]
    torchrun --standalone --nproc-per-node 1 -m appa.launchers generate <model_path> [overrides...]

The same subcommands are available through the `appa` console script, but the
training entry points require a torchrun process group (DDP), so prefer
torchrun for `ae-train` and `den-train`.
"""

import sys

from pathlib import Path

from appa.config.hydra import compose

SRC_DIR = Path(__file__).resolve().parents[1]

CONFIGS = {
    "ae-train": SRC_DIR / "experiments" / "autoencoder" / "configs" / "train.yaml",
    "den-train": SRC_DIR / "experiments" / "diffusion" / "configs" / "train.yaml",
    "generate": SRC_DIR / "experiments" / "diffusion" / "configs" / "generate.yaml",
}

__all__ = [
    "ae_train",
    "den_train",
    "generate",
    "main",
]


def ae_train(overrides) -> None:
    r"""Launches a local autoencoder training run."""

    import wandb

    from experiments.autoencoder.train import train

    cfg = compose(config_file=CONFIGS["ae-train"], overrides=overrides)

    runid = wandb.util.generate_id()
    print(f"Starting local AE training with runid: {runid}")

    train(runid, cfg, fork_lap=0, lap=0)


def den_train(overrides) -> None:
    r"""Launches a local latent diffusion (denoiser) training run."""

    import wandb

    from experiments.diffusion.train import train

    cfg = compose(config_file=CONFIGS["den-train"], overrides=overrides)

    runid = wandb.util.generate_id()
    print(f"Starting local denoiser training with runid: {runid}")

    train(runid, cfg, fork_lap=0, lap=0)


def generate(argv) -> None:
    r"""Generates prior trajectories from a trained denoiser."""

    from pathlib import Path

    import torch
    from einops import rearrange
    from omegaconf import OmegaConf

    from appa.utils.io import safe_load, safe_save
    from experiments.diffusion.generate import generate_prior_trajectory

    model_path = Path(argv[0]).resolve()
    overrides = argv[1:]

    cfg = compose(config_file=CONFIGS["generate"], overrides=overrides)
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)

    # Derive the padded trajectory size the same way generate.py does.
    trajectory_sizes = [int(s) for s in str(cfg.trajectory_sizes).split(",")]
    start_dates = list(cfg.start_dates)

    for traj_size in trajectory_sizes:
        target_dir = model_path.parent.parent.parent.parent / "prior" / "local" / f"{traj_size}h"
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {traj_size}h trajectory -> {target_dir}", flush=True)
        all_traj, all_ts = [], []

        for date in start_dates:
            date_str, hour_str = str(date).split()
            generate_prior_trajectory(
                model_path=model_path,
                model_target=cfg.model_target,
                diffusion=cfg.diffusion,
                unpadded_trajectory_size=traj_size,
                padded_trajectory_size=traj_size,
                blanket_overlap=cfg.blanket_overlap,
                start_date=date_str,
                start_hour=int(hour_str.replace("h", "")),
                precision=cfg.precision,
                target_dir=target_dir,
            )

            traj_file = target_dir / "trajectory.pt"
            ts_file = target_dir / "timestamps.pt"
            if traj_file.exists():
                traj = safe_load(traj_file)
                ts = safe_load(ts_file)
                print(
                    f"  {date}: trajectory = {tuple(traj.shape)}, "
                    f"timestamps = {tuple(ts.shape)}",
                    flush=True,
                )
                all_traj.append(traj)
                all_ts.append(ts)

        if all_traj:
            stacked = rearrange(torch.cat(all_traj, 0), "(d s) ... -> d s ...", s=1)
            safe_save(stacked, target_dir / "trajectories.pt")
            safe_save(
                rearrange(torch.cat(all_ts, 0), "(d s) ... -> d s ...", s=1),
                target_dir / "timestamps.pt",
            )
            print(f"Saved to {target_dir}", flush=True)

        print("Done.", flush=True)


def main() -> None:
    r"""Console-script entry point dispatching on the first argument."""

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)

    command, args = sys.argv[1], sys.argv[2:]

    if command == "ae-train":
        ae_train(args)
    elif command == "den-train":
        den_train(args)
    elif command == "generate":
        generate(args)
    else:
        raise SystemExit(f"Unknown command {command!r}. See --help.")


if __name__ == "__main__":
    main()
