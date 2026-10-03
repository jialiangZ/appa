r"""Lightning data modules for the autoencoder and denoiser training.

The datasets are the same as the ones used by the pre-Lightning scripts; the
data modules only wrap them. Distributed (sharding) samplers are left to
Lightning, which attaches them automatically when the trainer uses more than
one device.
"""

__all__ = [
    "ERA5DataModule",
    "LatentBlanketDataModule",
]

import os

from lightning.pytorch.core import LightningDataModule
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from appa.config import PATH_ERA5, PATH_STAT
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset, LatentBlanketDataset
from appa.data.transforms import StandardizeTransform

from ..config import PATH_AE


def _num_workers(world_size: int, cap: int = 16) -> int:
    r"""Returns a bounded number of dataloader workers per rank."""

    cpus = len(os.sched_getaffinity(0))

    return max(1, min(cpus // max(world_size, 1), cap))


class ERA5DataModule(LightningDataModule):
    r"""Data module for autoencoder training on ERA5.

    Provides the training and (quantitative) validation loaders. The
    qualitative validation snapshots are served by :py:`attr:`quali_dataset`
    and consumed directly by the snapshot logging callback (rank zero only).

    Arguments:
        cfg: The training configuration.
        world_size: The number of distributed processes.
    """

    def __init__(self, cfg: DictConfig, world_size: int = 1):
        super().__init__()

        self.cfg = cfg
        self.world_size = world_size

        pressure_levels = (
            SUB_PRESSURE_LEVELS if cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
        )
        self.pressure_levels = pressure_levels

        st = StandardizeTransform(
            PATH_STAT,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=pressure_levels,
        )

        cache = cfg.train.get("cache", "auto")

        splits = [
            DATASET_DATES_TRAINING,
            DATASET_DATES_VALIDATION,
            DATASET_DATES_VALIDATION,
        ]
        lnum_samples = [None, cfg.valid.n_valid_samples, cfg.valid.n_valid_snapshots]

        datasets = [
            ERA5Dataset(
                path=PATH_ERA5,
                start_date=start_date,
                end_date=end_date,
                num_samples=num_samples,
                transform=st,
                trajectory_size=cfg.train.trajectory_size,
                state_variables=ERA5_VARIABLES,
                context_variables=CONTEXT_VARIABLES,
                levels=pressure_levels,
                cache=cache,
            )
            for (start_date, end_date), num_samples in zip(splits, lnum_samples, strict=False)
        ]

        self.train_dataset, self.valid_dataset, self.quali_dataset = datasets

        # RAM-cached datasets index memory directly: no workers needed.
        self.num_workers = 0 if datasets[0].cache == "ram" else _num_workers(world_size)

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(
            dataset=self.train_dataset,
            batch_size=self.cfg.train.batch_element_per_gpu,
            num_workers=self.num_workers,
            shuffle=self.cfg.train.shuffle_training,
        )

    def val_dataloader(self) -> DataLoader:
        return get_dataloader(
            dataset=self.valid_dataset,
            batch_size=self.cfg.train.batch_element_per_gpu,
            num_workers=self.num_workers,
            shuffle=False,
        )


class LatentBlanketDataModule(LightningDataModule):
    r"""Data module for latent diffusion (denoiser) training.

    The quantitative validation loader iterates a fixed number of evenly
    spaced indices of the validation dataset (``valid.n_valid_samples``),
    mirroring the pre-Lightning behavior. The qualitative validation dates
    are exposed as :py:`attr:`snapshot_indices`.

    Arguments:
        cfg: The training configuration (holding ``ae_run`` and ``latent_dump``).
        ae_cfg: The configuration of the autoencoder run that generated the
            latents.
        world_size: The number of distributed processes.
    """

    def __init__(self, cfg: DictConfig, ae_cfg: DictConfig, world_size: int = 1):
        import torch

        super().__init__()

        self.cfg = cfg
        self.world_size = world_size

        latent_folder = Path(PATH_AE) / cfg.ae_run / "latents" / cfg.latent_dump
        latent_data = latent_folder / cfg.latent_data

        assert latent_data.exists(), (
            "Latent data must exist. Run the script scripts/run_latents.py to generate it."
        )

        if cfg.train.get("inject_ae_noise", False):
            noise_level = ae_cfg.ae.noise_level
        else:
            noise_level = 0.0

        def dates(section: str) -> tuple[str, str]:
            section_cfg = cfg.get(section, None)
            start = section_cfg.get("start_date", None) if section_cfg is not None else None
            end = section_cfg.get("end_date", None) if section_cfg is not None else None

            if section == "train":
                default_start, default_end = DATASET_DATES_TRAINING
            else:
                default_start, default_end = DATASET_DATES_VALIDATION

            return (
                start if start is not None else default_start,
                end if end is not None else default_end,
            )

        train_dates, valid_dates = dates("train"), dates("valid")

        self.train_dataset = LatentBlanketDataset(
            latent_data,
            train_dates[0],
            train_dates[1],
            cfg.train.blanket_size,
            standardize=cfg.train.standardize,
            stride=cfg.train.blanket_dt,
            noise_level=noise_level,
        )

        self.valid_dataset = LatentBlanketDataset(
            latent_data,
            valid_dates[0],
            valid_dates[1],
            cfg.train.blanket_size,
            standardize=cfg.train.standardize,
            stride=cfg.train.blanket_dt,
            noise_level=noise_level,
        )

        n_valid = cfg.valid.n_valid_samples
        quant_indices = torch.linspace(0, len(self.valid_dataset) - 1, n_valid).int().tolist()
        self.quant_dataset = Subset(self.valid_dataset, quant_indices)

        self.snapshot_indices = (
            torch.linspace(0, len(self.valid_dataset) - 1, cfg.valid.n_valid_dates).int().tolist()
        )

        self.num_workers = _num_workers(world_size)

    def train_dataloader(self) -> DataLoader:
        return get_dataloader(
            dataset=self.train_dataset,
            batch_size=self.cfg.train.batch_element_per_gpu,
            num_workers=self.num_workers,
            shuffle=self.cfg.train.shuffle_training,
        )

    def val_dataloader(self) -> DataLoader:
        return get_dataloader(
            dataset=self.quant_dataset,
            batch_size=self.cfg.train.batch_element_per_gpu,
            num_workers=self.num_workers,
            shuffle=False,
        )
