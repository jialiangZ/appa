r"""Script for all-at-once forecasting experiments."""

import copy
import gc
import shutil
import sys
import torch
import wandb

from datetime import datetime, timedelta
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.data.const import (
    DATASET_DATES_TEST,
)
from appa.data.datasets import LatentBlanketDataset
from appa.utils.dates import create_trajectory_timestamps
from appa.utils.math import str_to_ids
from appa.utils.io import safe_load, safe_save


def schedule_jobs(config, unique_id):
    target_dir = Path(config.model_path) / "persistence" / unique_id

    # Save config.yaml in target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_config = copy.deepcopy(config)
    saved_config.unique_id = unique_id
    OmegaConf.save(saved_config, target_dir / "config.yaml")

    # Launch job array of n samples for each window, according to a list of n timestamps
    assimilation_lengths = str(config.pop("assimilation_lengths"))
    lead_times = str(config.pop("lead_times"))
    blanket_size = compose(f"{config.model_path}/config.yaml").train.blanket_size

    num_samples_per_date = config.pop("num_samples_per_date")
    start_dates = config.pop("start_dates")
    start_dates = [start_date for start_date in start_dates for _ in range(num_samples_per_date)]
    start_dates = [
        (day, int(hour.split("h")[0])) for date in start_dates for day, hour in [date.split(" ")]
    ]
    start_dates, start_hours = zip(*start_dates)
    num_samples_per_traj_size = len(start_dates)

    for assimilation_length in str_to_ids(ids_str=assimilation_lengths):
        for lead_time in str_to_ids(lead_times):
            trajectory_size = max(blanket_size, assimilation_length + lead_time)
            run_name = f"{assimilation_length}_{lead_time}h"
            window_target_dir = target_dir / run_name

            for i in range(num_samples_per_traj_size):
                start_date = start_dates[i]
                start_hour = start_hours[i]

                unpadded_trajectory_size = assimilation_length + lead_time
                padded_trajectory_size = trajectory_size
                target_dir = window_target_dir / f"tmp_{i}"
                target_dir.mkdir(parents=True, exist_ok=True)

                # Offset the start date so start_date is the first predicted tate.
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
                start_date += timedelta(hours=start_hour)
                start_date -= timedelta(hours=assimilation_length)
                start_hour = int(start_date.hour)
                start_date = start_date.strftime("%Y-%m-%d")

                # Model and configs
                model_path = Path(config.model_path)
                latent_dir = model_path.parents[2]
                denoiser_cfg = compose(model_path / "config.yaml")
                trajectory_dt = denoiser_cfg.train.blanket_dt

                latent_gt_ds = LatentBlanketDataset(
                    path=latent_dir / config.latent_dump_name,
                    start_date=start_date,
                    start_hour=start_hour,
                    end_date=DATASET_DATES_TEST[-1],
                    blanket_size=1,
                    standardize=True,
                    stride=trajectory_dt,
                )

                blanket, _ = latent_gt_ds[0]
                blanket = blanket[-1:].repeat(padded_trajectory_size, 1, 1)[None]

                # Trim to assimilation window.
                sampled_traj = blanket[:, :unpadded_trajectory_size]
                safe_save(sampled_traj, target_dir / "trajectory.pt")

                timestamps = create_trajectory_timestamps(
                    start_date, start_hour, unpadded_trajectory_size, trajectory_dt
                )
                safe_save(
                    timestamps[None],
                    target_dir / "timestamps.pt",
                )

            trajectories = []
            timestamps = []
            for i in range(num_samples_per_traj_size):
                target_dir_traj = window_target_dir / f"tmp_{i}"
                trajectory = safe_load(target_dir_traj / "trajectory.pt")
                timestamp = safe_load(target_dir_traj / "timestamps.pt")

                trajectories.append(trajectory)
                timestamps.append(timestamp)

                shutil.rmtree(target_dir_traj)
                del trajectory, timestamp
                gc.collect()

            trajectories = torch.cat(trajectories)
            timestamps = torch.cat(timestamps)

            trajectories = rearrange(trajectories, "(d s) ... -> d s ...", s=num_samples_per_date)
            timestamps = rearrange(timestamps, "(d s) ... -> d s ...", s=num_samples_per_date)

            safe_save(trajectories, window_target_dir / "trajectories.pt")
            safe_save(timestamps, window_target_dir / "timestamps.pt")

            print("Saved to", window_target_dir)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python persistence.py [+id=<id>] model_path=... param1=A param2=B ...")
        return

    config = compose("configs/persistence.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" in config:
        unique_id = config.pop("id")
    else:
        unique_id = wandb.util.generate_id()

    schedule_jobs(config, unique_id)


if __name__ == "__main__":
    main()
