r"""Decode latent trajectories and render them as videos."""

import h5py
import numpy as np
import os
import shutil
import sys
import torch

from dawgz import after, job, schedule
from einops import rearrange
from matplotlib import pyplot as plt
from os import PathLike
from pathlib import Path
from scipy import ndimage
from torch import Tensor
from torch.nn import Module
from typing import Optional

from appa.config import PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TEST,
    ERA5_PRESSURE_LEVELS,
    ERA5_RESOLUTION,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from appa.diagnostics.plots import plot_image_grid
from appa.utils.io import load_auto_encoder, safe_load

# TODO: Support multiple samples for a same date


def make_title(
    render_type: str,
    t: int,
    metadata: dict,
):
    r"""Makes a title for the plot.

    Args:
        render_type: Type of rendering.
        t: Time step.
        metadata: Metadata dictionary.

    Returns:
        str: Title for the plot.
    """
    if "forecast" in render_type:
        assimilation_length = metadata["assimilation_length"]
        return "Assimilated" if t < assimilation_length else "Forecasted"
    elif render_type == "reanalysis":
        return "Assimilated"
    elif render_type == "prior":
        return "Unconditional"
    else:
        raise ValueError(f"Unknown render type: {render_type}")


def render_mask(
    x_gt,
    masks_dict: dict,
    channels: list,
    t: int,
    mask_mode: str = "all",
):
    r"""Renders a masked ground-truth state.

    Args:
        x_gt: Ground truth state, with shape :math:`(Z, Lat, Lon)`.
        masks_dict: Dictionary containing the masks to render.
        channels: List of channels to render.
        t: Time step of the trajectory to render.
        mask_mode: Mask display mode. Options: "satellites", "stations", "all", "auto".

    Returns:
        Tensor: Masked ground-truth state, with shape :math:`(Z, Lat, Lon)`, where unobserved areas are NaN.

    """
    leo_sat = masks_dict["satellite_masks"]["leo"]

    # Check if the mask exists (assimilated vs forecasted states)
    if f"mask_{t}" in leo_sat:
        time_masks_sat = leo_sat[f"mask_{t}"][()]
    else:
        return torch.full(size=x_gt.shape, fill_value=torch.nan)

    time_masks_stat = torch.tensor(masks_dict["stations_masks"]["weather11k"]["assimilation"][()])

    lon_res, lat_res = ERA5_RESOLUTION

    masks = []
    for c in range(x_gt.shape[0]):
        mask_img = np.ones((lat_res * lon_res), dtype=bool)
        if mask_mode == "satellites":
            mask_img[time_masks_sat] = False
        elif mask_mode == "stations":
            mask_img[time_masks_stat] = False
        elif "all" in mask_mode or "both" in mask_mode:
            combined_mask = torch.unique(torch.cat([time_masks_sat, time_masks_stat]))
            mask_img[combined_mask] = False
        elif mask_mode == "auto":
            if channels[c] < len(ERA5_SURFACE_VARIABLES):
                mask_img[time_masks_stat] = False
            else:
                mask_img[time_masks_sat] = False
        else:
            raise ValueError(f"Unknown mask mode: {mask_mode}")

        mask_img = mask_img.reshape((lat_res, lon_res))

        dilation_struct = ndimage.generate_binary_structure(2, 15)

        dilated_array = ndimage.binary_dilation(
            input=np.logical_not(mask_img),
            structure=dilation_struct,
            iterations=2,
        )
        dilated_array = np.logical_not(dilated_array)
        dilated_array = torch.from_numpy(dilated_array)

        masks.append(dilated_array)

    x_masked = x_gt.clone()
    x_masked[torch.stack(masks)] = torch.nan

    return x_masked.float()


def render_trajectory(
    trajectory: Tensor,
    timestamps: Tensor,
    latent_dir: PathLike,
    autoencoder: Module,
    channels: list,
    output_frames_dir: PathLike,
    render_type: str,
    metadata: dict,
    dt: int = 1,
    masks: Optional[object] = None,
    show_gt: bool = True,
    show_obs: bool = True,
    mask_mode: str = "auto",
    starting_frame: int = 0,
    skip_existing: bool = False,
    # Render options
    tex_font: bool = True,
    num_columns: int = None,  # Defaults to the number of channels
    vmins: list = None,
    vmaxs: list = None,
):
    r"""Render a trajectory through a decoder.

    Args:
        trajectory: Latent trajectory of shape :math:`(T, N, C)`.
        timestamps: Timestamps of shape :math:`(T, 4)`.
        autoencoder: Autoencoder module.
        channels: List of channels to render.
        output_frames_dir: Output directory of the frames.
        render_type: Type of rendering: "forecast", "reanalysis", or "prior".
        show_gt: Whether to show the ground truth.
        show_obs: Whether to show the observed (possibly masked) state.
        mask_mode: Mask display mode. Options: "satellites", "stations", "all", "auto".
        starting_frame: Starting frame for the rendering.
        skip_existing: Whether to skip existing files.
        tex_font: Whether to use LaTeX font for the title.
        num_columns: Number of columns for the plot. Defaults to the number of channels.
    """
    ae_cfg = compose(latent_dir / "ae" / "config.yaml")

    start__ = timestamps[0].tolist()
    curr_date = f"{start__[0]:04d}-{start__[1]:02d}-{start__[2]:02d}"
    start_hour = start__[3]

    # Get the ground truth
    variables_levels = {
        "state_variables": ERA5_VARIABLES,
        "context_variables": CONTEXT_VARIABLES,
        "levels": SUB_PRESSURE_LEVELS
        if ae_cfg.train.sub_pressure_levels
        else ERA5_PRESSURE_LEVELS,
    }
    st = StandardizeTransform(PATH_STAT, **variables_levels)
    era5 = ERA5Dataset(
        path=PATH_ERA5,
        start_date=curr_date,
        start_hour=start_hour,
        end_date=DATASET_DATES_TEST[-1],
        transform=st,
        trajectory_dt=dt,
        **variables_levels,
    )

    stats_path = latent_dir / "stats.pth"
    latent_stats = safe_load(stats_path)
    latent_mean = latent_stats["mean"].cuda()
    latent_std = latent_stats["std"]
    latent_std = torch.sqrt(latent_std**2 + ae_cfg.ae.noise_level**2).cuda()

    num_rows = 1
    if show_gt:
        num_rows += 1
    if show_obs:
        num_rows += 1

    for i in range(trajectory.shape[0]):
        if skip_existing and (output_frames_dir / f"{starting_frame + i}.png").exists():
            print(f"Skipping {starting_frame + i}", flush=True)
            continue

        x_gt, c_gt, t_gt = era5[i]

        c_gt = rearrange(c_gt, "b c h w -> b (h w) c")

        # Unscale
        blanket_element = trajectory[None, i].cuda() * latent_std + latent_mean
        with torch.no_grad():
            x_hat = autoencoder.decode(blanket_element, t_gt, c_gt)
        x_hat = rearrange(x_hat, "T (Lat Lon) Z  -> T Z Lat Lon", Lat=ERA5_RESOLUTION[1]).cpu()

        x_hat = x_hat[0, channels]

        xs = []
        row_titles = []

        x_gt = x_gt[0, channels]

        if show_gt:
            xs.append(x_gt)
            row_titles.append("Ground Truth")

        if show_obs:
            x_masked = (
                render_mask(
                    x_gt=x_gt,
                    masks_dict=masks,
                    channels=channels,
                    t=i + starting_frame,
                    mask_mode=mask_mode,
                )
                if masks is not None
                else x_gt
            )
            xs.append(x_masked)
            row_titles.append("Observation")

        xs.append(x_hat)
        row_titles.append("Sample")

        x = torch.cat(xs)

        if num_columns < len(channels):
            assert len(channels) % num_columns == 0

            x = rearrange(x, "(N B K) ... -> (B N K) ...", N=num_rows, K=num_columns)

            row_titles *= len(channels) // num_columns

        f = plot_image_grid(
            x,
            shape=(len(x) // num_columns, num_columns),
            vmin=vmins,
            vmax=vmaxs,
            y_titles=row_titles,
            border=[0.02, 0.01, 1 - 0.01, 1 - 0.055],
            tex_font=tex_font,
        )

        curr_date = timestamps[i].tolist()
        curr_date = (
            f"{curr_date[0]:04d}-{curr_date[1]:02d}-{curr_date[2]:02d} {curr_date[3]:02d}:00"
        )
        state_title = make_title(render_type, starting_frame + i, metadata)
        title = f"State {starting_frame + i} - {curr_date} - {state_title}"
        if tex_font:
            title = r"\textbf{" + title + r"}"

        ax = f.axes[0]  # first axes
        renderer = f.canvas.get_renderer()

        # Get axes bounding box in display (pixel) coords
        bbox = ax.get_window_extent(renderer=renderer)

        # Convert 5 px above bbox into figure coordinates
        inv = f.transFigure.inverted()
        _, y_fig = inv.transform((bbox.x0 + bbox.width / 2, bbox.y1 + 2))

        f.text(
            0.5,
            y_fig,
            title,
            ha="center",
            va="bottom",
            fontsize=32,
        )

        f.savefig(output_frames_dir / f"{starting_frame + i}.png")
        plt.close(f)

        print(
            f"Frame {starting_frame + i}/{trajectory.shape[0] + starting_frame - 1} saved.",
            flush=True,
        )


def frames_to_video(
    frames_dir: PathLike,
    output_dir: PathLike,
    file_name: str,
    framerate: int,
    height: int = 1080,
    width: int = 1920,
):
    r"""Renders a video from frames.

    Args:
        frames_dir: Directory containing the frames.
        output_dir: Output directory for the video.
        file_name: Name of the output video file.
        framerate: Framerate of the video.
        height: Height of the video.
        width: Width of the video.
    """

    os.system(
        f"ffmpeg -y -framerate {framerate} -i {frames_dir}/%d.png -vf scale={width}:{height} "
        f"-c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p {output_dir}/{file_name}.mp4"
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python render.py [+id=<id>] trajectory_dir=... param1=A param2=B ...")
        return

    config = compose("configs/render.yaml", overrides=sys.argv[1:])

    render_type = config.render_type
    if render_type is None:
        if "forecast" in config.trajectory_dir:
            render_type = "forecast"
        elif "reanalysis" in config.trajectory_dir:
            render_type = "reanalysis"
        else:
            render_type = "prior"
        print("Automatically determined render type:", render_type)

    trajectory_dir = Path(config.trajectory_dir)
    renders_dir = trajectory_dir / "renders"
    if "id" in config:
        renders_dir = renders_dir / config.id
    renders_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if "forecast" in render_type:
        metadata["assimilation_length"] = int(trajectory_dir.name.split("_")[0])

    denoiser_dir = trajectory_dir.parent
    while denoiser_dir.parents[1].name != "denoisers":
        denoiser_dir = denoiser_dir.parent

    latent_dir = denoiser_dir.parents[2]

    show_gt = config.show_gt
    show_obs = config.show_obs
    if render_type == "prior":
        show_obs = False

    trajectories = safe_load(trajectory_dir / "trajectories.pt")
    timestamps = safe_load(trajectory_dir / "timestamps.pt")

    # Current strategy: last sample.  # TODO: Support multiple samples
    trajectories = trajectories[:, -1]
    timestamps = timestamps[:, -1]

    num_chunks = min(config.num_chunks, trajectories.shape[1])
    framerate = config.framerate

    denoiser_cfg = compose(denoiser_dir / "config.yaml")
    ae_cfg = compose(latent_dir / "ae" / "config.yaml")
    variables_levels = {
        "state_variables": ERA5_VARIABLES,
        "context_variables": CONTEXT_VARIABLES,
        "levels": SUB_PRESSURE_LEVELS
        if ae_cfg.train.sub_pressure_levels
        else ERA5_PRESSURE_LEVELS,
    }
    st = StandardizeTransform(PATH_STAT, **variables_levels)

    channels = config.channels
    quantile = config.quantile
    num_columns = config.num_columns or len(config.channels)

    jobs = []
    print(f"Loaded trajectories with shape {trajectories.shape}.")
    for j, (trajectory, timestamp) in enumerate(zip(trajectories, timestamps)):
        start_date = timestamp[0].tolist()

        # Load vmin/vmax from the first state of the trajectory.
        era5_traj = ERA5Dataset(
            path=PATH_ERA5,
            start_date=f"{start_date[0]:04d}-{start_date[1]:02d}-{start_date[2]:02d}",
            start_hour=start_date[3],
            end_date=DATASET_DATES_TEST[-1],
            transform=st,
            trajectory_dt=denoiser_cfg.train.blanket_dt,
            **variables_levels,
        )
        num_rows = 1
        if show_gt:
            num_rows += 1
        if show_obs:
            num_rows += 1
        x_gt_0, _, _ = era5_traj[0]
        x_gt_0 = x_gt_0[0, channels]
        vmins = [x_gt_0[c].quantile(quantile) for c in range(len(channels))] * num_rows
        vmaxs = [x_gt_0[c].quantile(1 - quantile) for c in range(len(channels))] * num_rows
        num_columns = len(channels) if num_columns is None else num_columns

        if num_columns < len(channels):
            assert len(channels) % num_columns == 0

            vmins = rearrange(vmins, "(N B K) ... -> (B N K) ...", N=num_rows, K=num_columns)
            vmaxs = rearrange(vmaxs, "(N B K) ... -> (B N K) ...", N=num_rows, K=num_columns)

        start_date = (
            f"{start_date[0]:04d}_{start_date[1]:02d}_{start_date[2]:02d}_{start_date[3]:02d}"
        )

        indices = torch.arange(trajectory.shape[0])
        indices = indices.tensor_split(num_chunks)
        indices = [ind[0] for ind in indices]

        frames_dir = renders_dir / "frames" / start_date
        frames_dir.mkdir(parents=True, exist_ok=True)

        @job(
            name="appa render",
            array=num_chunks,
            **config.hardware.render_chunk,
        )
        def render_frames_chunk(
            i: int,
            j: int = j,
            frames_dir: Path = frames_dir,
            offsets: list = indices,
            show_obs: bool = show_obs,
            vmins: list = vmins,
            vmaxs: list = vmaxs,
            num_columns: int = num_columns,
        ):
            traj = safe_load(trajectory_dir / "trajectories.pt")
            times = safe_load(trajectory_dir / "timestamps.pt")

            traj = traj[j, -1]
            times = times[j, -1]

            traj = traj.tensor_split(num_chunks)[i]
            times = times.tensor_split(num_chunks)[i]

            ae = load_auto_encoder(latent_dir / "ae", "model", device="cuda", eval_mode=True)
            ae.cuda()

            denoiser_cfg = compose(denoiser_dir / "config.yaml")
            trajectory_dt = denoiser_cfg.train.blanket_dt

            masks_file = trajectory_dir / "masks.h5"
            masks = None
            if show_obs and masks_file.exists():
                with h5py.File(masks_file, "r") as f:
                    masks = f[f"masks_{j}"]

            render_trajectory(
                trajectory=traj,
                timestamps=times,
                latent_dir=latent_dir,
                autoencoder=ae,
                channels=channels,
                output_frames_dir=frames_dir,
                render_type=render_type,
                metadata=metadata,
                dt=trajectory_dt,
                masks=masks,
                show_gt=show_gt,
                show_obs=show_obs,
                mask_mode=config.mask_display_mode,
                starting_frame=offsets[i],
                skip_existing=config.skip_existing,
                num_columns=num_columns,
                vmins=vmins,
                vmaxs=vmaxs,
                tex_font=config.tex_font,
            )

        @after(render_frames_chunk)
        @job(name="appa render (encode)", **config.hardware.encode_video)
        def encode_video(
            frames_dir: Path = frames_dir,
            start_date: str = start_date,
        ):
            frames_to_video(
                frames_dir,
                renders_dir,
                start_date,
                framerate=framerate,
            )

            if not config.keep_frames:
                shutil.rmtree(frames_dir)

        jobs.append(encode_video)

    schedule(
        *jobs,
        account=config.hardware.account,
        name="appa render",
        export="ALL",
        backend=config.hardware.backend,
    )


if __name__ == "__main__":
    main()
