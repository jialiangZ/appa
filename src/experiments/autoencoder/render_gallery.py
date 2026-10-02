r"""Plots a gallery of GT, observations and predictions."""

import dawgz
import h5py
import numpy as np
import sys
import torch

from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import binary_dilation
from typing import Optional

from appa.config import PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TEST,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.datasets import ERA5Dataset
from appa.data.mappings import feature_idx_to_name
from appa.data.transforms import StandardizeTransform
from appa.diagnostics.const import CMAPS_SURF
from appa.diagnostics.plots import plot_image_grid
from appa.utils.io import load_auto_encoder, safe_load


def plot_galleries(config):
    path = Path(config.trajectory_dir)

    if "forecast" in str(path):
        assimilation_window = int(path.name.split("_")[0])
    else:
        assimilation_window = None  # No or all states are assimilated

    denoiser_cfg = compose(path.parents[2] / "config.yaml")
    latent_dir = path.parents[5]
    ae_cfg = compose(latent_dir / "ae" / "config.yaml")

    ae = load_auto_encoder(latent_dir / "ae", "model", eval_mode=True)
    ae.cuda()
    ae.requires_grad_(False)

    output_dir = path / "renders" / "galleries"
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel in config.channels:
        plot_gallery(
            trajectories_dir=path,
            channel=channel,
            date=config.date,
            indices_to_display=config.times,
            latent_dir=latent_dir,
            autoencoder=ae,
            denoiser_cfg=denoiser_cfg,
            ae_cfg=ae_cfg,
            output_dir=output_dir,
            cmap=config.cmap,
            show_observations=config.show_observations,
            show_both_masks=config.show_both_masks,
            assimilation_window=assimilation_window,
        )


def plot_gallery(
    trajectories_dir: Path,
    channel: int,
    date: int,
    indices_to_display: list,
    autoencoder: torch.nn.Module,
    latent_dir: Path,
    denoiser_cfg: dict,
    ae_cfg: dict,
    output_dir: Path,
    cmap: Optional[str] = None,
    show_observations: bool = True,
    show_both_masks: bool = False,
    assimilation_window: Optional[int] = None,
):
    r"""Plot a gallery of GT, observations, and predicted samples.

    Args:
        trajectories_dir: Directory containing the trajectories and possibly masks.
        channel: Index of the channel to plot.
        date: Date index to plot among the dates available in the trajectories.
        indices_to_display: List of time indices in the trajectory to display.
        decoder: Decoder module of the autoencoder.
        latent_dir: Directory containing the latent dump.
        denoiser_cfg: Configuration for the denoiser.
        ae_cfg: Configuration for the autoencoder.
        output_dir: Directory to save the gallery plot in.
        cmap: Colormap to use for the plots. If None, will use the default for the variable.
        show_observations: Whether to show observations in the gallery.
        show_both_masks: Whether to show both satellite and station masks in the observations.
        assimilation_window: Number of frames assimilated (e.g., forecasting).
    """
    blanket_dt = denoiser_cfg.train.blanket_dt

    ae_cfg = compose(latent_dir / "ae" / "config.yaml")

    stats_path = latent_dir / "stats.pth"
    latent_stats = safe_load(stats_path)
    latent_mean = latent_stats["mean"].cuda()
    latent_std = latent_stats["std"]
    latent_std = torch.sqrt(latent_std**2 + ae_cfg.ae.noise_level**2).cuda()

    indices_to_display = [config.times[0]] + [t for t in config.times[1:]]

    variable_name = feature_idx_to_name(channel)

    trajectories = torch.load(trajectories_dir / "trajectories.pt", weights_only=False)
    timestamps = torch.load(trajectories_dir / "timestamps.pt", weights_only=False)

    if len(trajectories.shape) == 4:
        num_dates = len(timestamps[:, 0].unique(dim=0))
        trajectories = rearrange(trajectories, "(d s) ... -> d s ...", d=num_dates)
        timestamps = rearrange(timestamps, "(d s) ... -> d s ...", d=num_dates)

    trajectories = trajectories[date]
    timestamps = timestamps[date]

    # Store states to display
    ground_truths = []
    observations = []
    predictions = [[]] * config.samples_per_date

    masks_file = trajectories_dir / "masks.h5"
    all_masks = h5py.File(trajectories_dir / "masks.h5", "r") if masks_file.exists() else None

    variables_and_levels = {
        "state_variables": ERA5_VARIABLES,
        "context_variables": CONTEXT_VARIABLES,
        "levels": SUB_PRESSURE_LEVELS
        if ae_cfg.train.sub_pressure_levels
        else ERA5_PRESSURE_LEVELS,
    }
    start_ts = timestamps[0, 0]
    st = StandardizeTransform(PATH_STAT, **variables_and_levels)
    era5 = ERA5Dataset(
        path=PATH_ERA5,
        start_date=f"{start_ts[0].item()}-{start_ts[1].item():02d}-{start_ts[2].item():02d}",
        start_hour=start_ts[3].item(),
        end_date=DATASET_DATES_TEST[-1],
        num_samples=None,
        transform=st,
        trajectory_size=1,
        trajectory_dt=blanket_dt,
        **variables_and_levels,
    )

    contexts_gt = []
    times_gt = []

    for idx in indices_to_display:
        x_gt, c_gt, t_gt = era5[idx]
        x_gt = rearrange(x_gt, "T Z Lat Lon -> T Lat Lon Z")
        c_gt = rearrange(c_gt, "b c h w -> b (h w) c")
        x_gt = x_gt[0, ..., channel]
        ground_truths.append(x_gt.cpu().numpy())
        contexts_gt.append(c_gt)
        times_gt.append(t_gt)

    masks = all_masks[f"masks_{date}"] if all_masks is not None else None

    if show_observations:
        for i in range(len(indices_to_display)):
            idx_to_display = indices_to_display[i]

            over_assim_window = (
                assimilation_window is not None and idx_to_display >= assimilation_window
            )
            mask_exists = (
                masks is None or f"mask_{idx_to_display}" in masks["satellite_masks"]["leo"]
            )

            if not mask_exists or over_assim_window:
                nan_obs = torch.full((721, 1440), np.nan, dtype=torch.float32).numpy()
                observations.append(nan_obs)
                continue

            if masks is None:
                observations.append(ground_truths[i])
            else:
                time_masks_sat = torch.tensor(
                    masks["satellite_masks"]["leo"][f"mask_{idx_to_display}"][()]
                )
                time_masks_stat = torch.tensor(
                    masks["stations_masks"]["weather11k"]["assimilation"][()]
                )

                mask_img = np.ones((1440 * 721, x_gt.shape[-1]), dtype=bool)

                if show_both_masks:
                    mask = torch.unique(torch.cat([time_masks_sat, time_masks_stat]))
                else:
                    if channel < len(ERA5_SURFACE_VARIABLES):
                        mask = time_masks_stat
                    else:
                        mask = time_masks_sat
                mask_img[mask] = False

                mask_img = rearrange(mask_img, "(Lat Lon) Z -> Lat Lon Z", Lat=721)

                dilation_struct = ndimage.generate_binary_structure(2, 15)
                dilated_array = binary_dilation(
                    input=np.logical_not(mask_img[..., 0]),
                    structure=dilation_struct,
                    iterations=2,
                )
                dilated_array = np.logical_not(dilated_array)

                x_masked = ground_truths[i].copy()
                x_masked[dilated_array] = np.nan

                observations.append(x_masked)

    times_titles = []
    for sample_id in range(config.samples_per_date):
        for i in range(len(indices_to_display)):
            blanket_element = (
                trajectories[sample_id, indices_to_display[i] : indices_to_display[i] + 1].cuda()
                * latent_std
                + latent_mean
            )
            if sample_id == 0:
                state_ts = timestamps[sample_id, indices_to_display[i]]
                times_titles.append(
                    f"{state_ts[0].item()}-{state_ts[1].item():02d}-{state_ts[2].item():02d} {state_ts[3].item():02d}:00"
                )
            with torch.no_grad():
                x_hat = autoencoder.decode(
                    blanket_element, times_gt[i : i + 1], contexts_gt[i : i + 1]
                )
            x_hat = rearrange(x_hat, "T (Lat Lon) Z  -> T Lat Lon Z", Lat=721).cpu()
            x_hat = x_hat[0, ..., channel]
            predictions[sample_id].append(x_hat.cpu().numpy())

    x = ground_truths
    if show_observations:
        x += observations
    for p in predictions:
        x += p

    if cmap is None:
        cmap = CMAPS_SURF[variable_name]

    vmins = x_gt.quantile(config.quantile)
    vmaxs = x_gt.quantile(1 - config.quantile)

    if config.show_time_offsets:
        x_titles = [times_titles[0]] + [f"+{t} hours" for t in config.times[1:]]
    else:
        x_titles = times_titles

    y_titles = ["Ground truth"]
    if show_observations:
        y_titles.append("Observation")
    y_titles += ["Sample"] * config.samples_per_date

    f = plot_image_grid(
        torch.from_numpy(np.array(x)),
        shape=(len(y_titles), 4),
        cmap=cmap,
        vmin=vmins.item(),
        vmax=vmaxs.item(),
        border=[0.03, 0.01, 1 - 0.01, 1 - 0.045],
        y_titles=y_titles,
        x_titles=x_titles,
        tex_font=False,
    )

    f.savefig(output_dir / f"{variable_name}.pdf")

    print("Saved to", output_dir / f"{variable_name}.pdf")


if __name__ == "__main__":
    config = compose("configs/render_gallery.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    hardware_cfg = config.pop("hardware")

    dawgz.schedule(
        dawgz.job(
            f=partial(plot_galleries, config=config),
            name="appa render (gallery)",
            cpus=hardware_cfg.cpus,
            gpus=hardware_cfg.gpus,
            ram=hardware_cfg.ram,
            time=hardware_cfg.time,
            partition=hardware_cfg.partition,
        ),
        name="appa render (gallery)",
        backend=hardware_cfg.backend,
        account=hardware_cfg.account,
        env=[
            "export OMP_NUM_THREADS=" + f"{hardware_cfg.cpus}",
            "export WANDB_SILENT=true",
            "export XDG_CACHE_HOME=$HOME/.cache",
            "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
        ],
    )
