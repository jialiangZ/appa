r"""Create an observation mask using the weather stations from the
NOAA Global Summary of the Day for 2025 or the WEATHER-5K paper.
"""

import sys

from pathlib import Path

from appa.config.hydra import compose
from appa.assimilation.observations import (
    create_observation_mask,
    process_raw_noaa_csv,
    process_weather_5k_metadata,
)


def process_11k(data_path: Path, process_path: Path, mask_path: Path):
    process_raw_noaa_csv(data_path, process_path)
    create_observation_mask(processed_data=process_path, save_path=mask_path)


def process_5k(data_path: Path, process_path: Path, mask_path: Path):
    process_weather_5k_metadata(data_path, process_path)
    create_observation_mask(processed_data=process_path, save_path=mask_path)


if __name__ == "__main__":
    config = compose("configs/ground_stations_masks.yaml", overrides=sys.argv[1:])

    raw_path_11k = Path(config.raw_path_11k)
    process_path_11k = Path(config.process_path_11k)
    mask_path_11k = Path(config.mask_path_11k)

    raw_path_5k = Path(config.raw_path_5k)
    process_path_5k = Path(config.process_path_5k)
    mask_path_5k = Path(config.mask_path_5k)

    if config.observation_model in ["11k", "both"]:
        process_11k(raw_path_11k, process_path_11k, mask_path_11k)

    if config.observation_model in ["5k", "both"]:
        process_5k(raw_path_5k, process_path_5k, mask_path_5k)
