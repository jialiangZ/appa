r"""Local (no-dawgz) prior trajectory generation.

Usage:
    CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node 1 \
        run_generate.py <model_path> [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node 1 \
        run_generate.py artifacts/autoencoders/small_ae_v1/0/latents/small_v1/denoisers/917894i3/0 \
        diffusion.num_steps=64 trajectory_sizes=72 start_dates=2000-03-15
"""

import sys
from pathlib import Path

import torch
from einops import rearrange

from omegaconf import OmegaConf, open_dict

from appa.config.hydra import compose
from appa.utils.io import safe_load, safe_save
from experiments.diffusion.generate import generate_prior_trajectory

if __name__ == "__main__":
    model_path = Path(sys.argv[1]).resolve()
    overrides = sys.argv[2:]

    cfg = compose(config_file="src/experiments/diffusion/configs/generate.yaml", overrides=overrides)
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)

    # Derive the padded trajectory size the same way generate.py does.
    trajectory_sizes = [int(s) for s in str(cfg.trajectory_sizes).split(",")]
    start_dates = list(cfg.start_dates)
    num_samples = cfg.num_samples_per_date

    for traj_size in trajectory_sizes:
        target_dir = Path(model_path).parent.parent.parent.parent / "prior" / "local" / f"{traj_size}h"
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
                # Aggregate like generate.py does: stack over dates.
                all_traj.append(traj)
                all_ts.append(ts)

        if all_traj:
            stacked = rearrange(
                torch.cat(all_traj, 0),
                "(d s) ... -> d s ...",
                s=1,
            )
            safe_save(stacked, target_dir / "trajectories.pt")
            safe_save(rearrange(torch.cat(all_ts, 0), "(d s) ... -> d s ...", s=1),
                      target_dir / "timestamps.pt")
            print(f"Saved to {target_dir}", flush=True)

        print("Done.", flush=True)
