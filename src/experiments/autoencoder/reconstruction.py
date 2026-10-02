r"""Compute reconstruction errors for a trained autoencoder."""

import dask
import h5py
import shutil
import sys
import torch
import wandb
import warnings

from dawgz import after, job, schedule
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from torch import Tensor
from typing import Sequence

from appa.config import PATH_ERA5, PATH_MASK, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_RESOLUTION,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from appa.utils.io import load_auto_encoder


def compute_reconstruction_errors(
    model_path: Path,
    use_best: bool,
    data_path: Path,
    batch_size: int,
    start_date: str,
    end_date: str,
    num_bins: int,
    atm_levels: Sequence[int],
    rank: int = None,
    world_size: int = None,
) -> Tensor:
    r"""Computes the signed error histograms, squared errors, and returns a decoded sample.

    Arguments:
        autoencoder: Trained autoencoder used to encode.
        data_path: Path to the dataset.
        output_path: Path to save the final output .h5 file.
        batch_size: Number of samples processed at once.
        start_date: Start date in (YYYY-MM-DD) format.
        end_date: End date in (YYYY-MM-DD) format.
        rank: If not None, the rank of the current process.
        world_size: If not None, the total number of processes.

    Returns:
        signed_errors_histograms: Tensor of shape (num_samples, num_variables, num_bins) containing signed errors histograms.
        signed_errors_bin_edges: Tensor of shape (num_samples, num_variables, num_bins + 1) containing the bin edges for the signed errors.
        std_mse: Tensor of shape (num_samples, T, num_variables) containing squared errors over standardized data.
        gt_sample: Tensor of shape (T, num_variables, Lat, Lon) containing the ground-truth sample.
        pred_sample: Tensor of shape (T, num_variables, Lat, Lon) containing the predicted sample.
    """

    autoencoder = load_auto_encoder(
        model_path,
        model_name="model_best" if use_best else "model_last",
    )

    dask.config.set(scheduler="synchronous")

    st = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=atm_levels,
    )
    dataset = ERA5Dataset(
        path=data_path,
        start_date=start_date,
        end_date=end_date,
        num_samples=None,
        transform=st,
        trajectory_size=1,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=atm_levels,
    )

    dataloader = get_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=7,
        prefetch_factor=2,
        shuffle=False,
        rank=rank,
        world_size=world_size,
        drop_last_ddp=False,  # Ensure we see all samples. Truncated later.
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    std_mse = None
    signed_errors_histograms = None
    signed_errors_bin_edges = None
    error_idx = 0

    gt_sample, pred_sample = None, None

    # Bin ranges
    bin_ranges = []
    multipliers = []
    for sv in ERA5_SURFACE_VARIABLES:
        bin_ranges.append(tuple(config.bin_ranges[sv]))
        if sv in config.multipliers:
            multipliers.append(config.multipliers[sv])
        else:
            multipliers.append(1.0)
    for av in ERA5_ATMOSPHERIC_VARIABLES:
        bin_ranges += [tuple(config.bin_ranges[av])] * len(atm_levels)
        if av in config.multipliers:
            multipliers += [config.multipliers[av]] * len(atm_levels)
        else:
            multipliers += [1.0] * len(atm_levels)

    # For sea surface temperature, we need to mask the land.
    if "sea_surface_temperature" in ERA5_VARIABLES:
        import xarray as xr

        sst_idx = ERA5_VARIABLES.index("sea_surface_temperature")
        sea_mask_cpu = xr.open_zarr(PATH_MASK)["sea_surface_temperature_mask"].values
        sea_mask = torch.from_numpy(sea_mask_cpu).to(device)[None, None]
    else:
        sst_idx = None

    for state, context, date in dataloader:
        with torch.no_grad():
            state = rearrange(
                state.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            context = rearrange(
                context.to(device, non_blocking=True), "B T K Lat Lon -> (B T) (Lat Lon) K"
            )
            date = rearrange(date.to(device, non_blocking=True), "B T D -> (B T) D")
            _, state_pred = autoencoder(state, date, context)

            state = rearrange(
                state, "(B T) (Lat Lon) Z -> B T Z Lat Lon", B=batch_size, Lon=ERA5_RESOLUTION[0]
            )
            state_pred = rearrange(
                state_pred,
                "(B T) (Lat Lon) Z -> B T Z Lat Lon",
                B=batch_size,
                Lon=ERA5_RESOLUTION[0],
            )

            if gt_sample is None:
                gt_sample = state[0].cpu()
                pred_sample = state_pred[0].cpu()

            std_error = (state - state_pred) ** 2  # [B T Z Lat Lon]
            std_error = std_error.mean(dim=(-1, -2))  # [B T Z]
            if sst_idx is not None:
                # Only compute error over sea surface
                std_error[:, :, sst_idx] = torch.mean(
                    (state[:, :, sst_idx][sea_mask] - state_pred[:, :, sst_idx][sea_mask]) ** 2,
                    dim=-1,
                )
            state, _ = st.unstandardize(state.cpu())
            state_pred, _ = st.unstandardize(state_pred.cpu())

            signed_error = (state.cuda() - state_pred.cuda()).cpu()  # [B T Z Lat Lon]

            # Compute histogram for signed errors
            # TODO: Batch support for here.
            histograms = torch.empty((signed_error.shape[2], num_bins))
            bin_edges = torch.empty((signed_error.shape[2], num_bins + 1))
            for i in range(signed_error.shape[2]):
                var_signed_error = signed_error[0, 0, i] * multipliers[i]

                # Mask land for sea surface temperature
                if sst_idx is not None and i == sst_idx:
                    var_signed_error = var_signed_error[sea_mask_cpu]

                histogram, bins = torch.histogram(
                    input=var_signed_error, bins=num_bins, range=bin_ranges[i]
                )
                histograms[i] = histogram
                bin_edges[i] = bins

        if signed_errors_histograms is None:
            std_mse = torch.empty(len(dataloader), *std_error.shape[1:], device=device)
            signed_errors_histograms = torch.empty(
                len(dataloader), signed_error.shape[2], num_bins
            )
            signed_errors_bin_edges = torch.empty(
                len(dataloader), signed_error.shape[2], num_bins + 1
            )

        std_mse[error_idx : error_idx + signed_error.shape[0]] = std_error
        signed_errors_histograms[error_idx : error_idx + signed_error.shape[0]] = histograms
        signed_errors_bin_edges[error_idx : error_idx + signed_error.shape[0]] = bin_edges
        error_idx += signed_error.shape[0]

    return signed_errors_histograms, signed_errors_bin_edges, std_mse, gt_sample, pred_sample


if __name__ == "__main__":
    config = compose("configs/reconstruction.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" in config:
        unique_id = config.pop("id")
    else:
        unique_id = wandb.util.generate_id()

    data_path = config.data_path
    if data_path == "era5":
        data_path = PATH_ERA5

    start_date = config.start_date
    end_date = config.end_date

    model_path = Path(config.model_path)
    output_path = model_path / "reconstruction" / unique_id
    tmp_dir = output_path / "tmp"

    tmp_dir.mkdir(parents=True, exist_ok=True)

    use_best = config.checkpoint == "best"

    # Save AE checkpoint, its config, and metadata about the dump.
    (output_path / "ae").mkdir(parents=True, exist_ok=True)
    config_path = model_path / "config.yaml"
    shutil.copy(config_path, output_path / "ae")
    shutil.copy(
        model_path / f"model_{'best' if use_best else 'last'}.pth",
        output_path / "ae" / "model.pth",
    )
    OmegaConf.save(config, output_path / "config.yaml")

    ae_config = compose(config_path)
    if "sub_pressure_levels" not in ae_config.train:
        sub_pressure_levels = False
        warnings.warn(
            "sub_pressure_levels not found in config, using default value (False).",
            stacklevel=1,
        )
    else:
        sub_pressure_levels = ae_config.train.sub_pressure_levels

    if sub_pressure_levels:
        atm_levels = SUB_PRESSURE_LEVELS
    else:
        atm_levels = ERA5_PRESSURE_LEVELS

    @job(
        name="appa reconstruction (compute)",
        array=config.num_chunks,
        **config.hardware.latent_chunk,
    )
    def compute_and_save_latent_chunk(i: int):
        signed_errors_histograms, signed_errors_bin_edges, std_mse, gt_sample, pred_sample = (
            compute_reconstruction_errors(
                model_path,
                use_best,
                [Path(dp) for dp in data_path] if isinstance(data_path, list) else Path(data_path),
                config.batch_size,
                start_date,
                end_date,
                config.num_bins,
                atm_levels,
                rank=i,
                world_size=config.num_chunks,
            )
        )

        # Save errors
        with h5py.File(tmp_dir / f"{i}.h5", "w") as f:
            f.create_dataset("std_mse", data=std_mse.cpu().numpy())
            f.create_dataset(
                "signed_errors_histograms", data=signed_errors_histograms.cpu().numpy()
            )
            f.create_dataset("signed_errors_bin_edges", data=signed_errors_bin_edges.cpu().numpy())

        # Save sample
        with h5py.File(tmp_dir / f"sample_{i}.h5", "w") as f:
            f.create_dataset("gt", data=gt_sample.cpu().numpy())
            f.create_dataset("pred", data=pred_sample.cpu().numpy())

    @after(compute_and_save_latent_chunk)
    @job(name="appa reconstruction (agg)", **config.hardware.aggregate)
    def aggregate():
        # Save only one of the samples
        shutil.copy2(tmp_dir / "sample_0.h5", output_path / "sample.h5")

        # Aggregate and save errors
        len_dataset = len(
            ERA5Dataset(
                path=data_path,
                start_date=start_date,
                end_date=end_date,
                state_variables=ERA5_VARIABLES,  # TODO config
                context_variables=CONTEXT_VARIABLES,
                levels=atm_levels,
            )
        )
        std_mse = []
        signed_errors_histograms = []
        signed_errors_bin_edges = []
        for i in range(config.num_chunks):
            chunk_path = tmp_dir / f"{i}.h5"
            with h5py.File(chunk_path, "r") as f:
                std_mse.append(torch.tensor(f["std_mse"]))
                signed_errors_histograms.append(torch.tensor(f["signed_errors_histograms"]))
                signed_errors_bin_edges.append(torch.tensor(f["signed_errors_bin_edges"]))
        with h5py.File(output_path / "errors.h5", "w") as f:
            for k, v in zip(
                ("std_mse", "signed_errors_histograms", "signed_errors_bin_edges"),
                (std_mse, signed_errors_histograms, signed_errors_bin_edges),
            ):
                v = torch.stack(v)
                v = rearrange(v, "N C ... -> (C N) ...")
                v = v[:len_dataset]

                f.create_dataset(k, data=v.cpu().numpy())

        shutil.rmtree(tmp_dir)

    schedule(
        aggregate,
        name="appa reconstruction",
        export="ALL",
        account=config.hardware.account,
        backend=config.hardware.backend,
    )
