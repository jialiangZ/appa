r"""Script to save encoded trajectories in the forecast format."""

import copy
import sys
import torch
import wandb

from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.data.const import (
    DATASET_DATES_TEST,
)
from appa.data.datasets import LatentBlanketDataset
from appa.utils.dates import create_trajectory_timestamps
from appa.utils.math import str_to_ids
from appa.utils.io import safe_save


def schedule_jobs(config, unique_id):
    latent_path = Path(config.pop("latent_path"))
    target_dir = latent_path.parent / "trajectories" / unique_id

    # Save config.yaml in target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_config = copy.deepcopy(config)
    saved_config.unique_id = unique_id
    OmegaConf.save(saved_config, target_dir / "config.yaml")

    # Launch job array of n samples for each window, according to a list of n timestamps
    trajectory_sizes = str(config.pop("trajectory_sizes"))

    num_samples_per_date = config.pop("num_samples_per_date")
    start_dates = config.pop("start_dates")
    start_dates = [
        (day, int(hour.split("h")[0])) for date in start_dates for day, hour in [date.split(" ")]
    ]
    start_dates, start_hours = zip(*start_dates)

    trajectory_dt = config.trajectory_dt
    ae_config = compose(Path(latent_path).parent / "ae" / "config.yaml")

    OmegaConf.save({"trajectory_dt": trajectory_dt}, target_dir / "metadata.yaml")

    for trajectory_size in str_to_ids(ids_str=trajectory_sizes):
        states = []
        times = []

        trajectory_size_dir = target_dir / f"{trajectory_size}h"
        trajectory_size_dir.mkdir(parents=True, exist_ok=True)

        for start_date, start_hour in zip(start_dates, start_hours):
            ensemble_states = []
            ensemble_times = []

            latent_dataset = LatentBlanketDataset(
                path=Path(latent_path),
                start_date=start_date,
                end_date=DATASET_DATES_TEST[-1],
                start_hour=start_hour,
                noise_level=ae_config.ae.noise_level,
                stride=trajectory_dt,
                blanket_size=trajectory_size,
            )

            for _ in range(num_samples_per_date):
                ensemble_times.append(
                    create_trajectory_timestamps(
                        start_date, start_hour, trajectory_size, trajectory_dt
                    )
                )

                ensemble_states.append(latent_dataset[0][0])

            states.append(torch.stack(ensemble_states))
            times.append(torch.stack(ensemble_times))

        states = torch.stack(states)
        times = torch.stack(times)

        safe_save(states, trajectory_size_dir / "trajectories.pt")
        safe_save(times, trajectory_size_dir / "timestamps.pt")

        print(f"Saved trajectories to {trajectory_size_dir}.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python trajectories.py [+id=<id>] latent_path=... param1=A param2=B ...")
        return

    config = compose("configs/trajectories.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" in config:
        unique_id = config.pop("id")
    else:
        unique_id = wandb.util.generate_id()

    schedule_jobs(config, unique_id)


if __name__ == "__main__":
    main()
