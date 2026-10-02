import matplotlib.pyplot as plt
import numpy as np
import sys
import xarray as xr

from matplotlib.lines import Line2D
from os import PathLike
from pathlib import Path
from typing import Union

from appa.config.hydra import compose
from appa.data.const import ERA5_ATMOSPHERIC_VARIABLES
from appa.diagnostics.const import CMAPS_LINE_DEEPMIND, EARTH_RADIUS, PRETTY_VAR_NAMES


def plot_power_spectra(
    output_path: PathLike,
    path_latent: PathLike,
    path_samples: PathLike,
    path_gt: PathLike,
    levels: list[int],
    atmospheric_variables: Union[list[str], str],
    quantile: str,
    plot_params: dict,
    grid_params: dict,
    labels_params: dict,
    ae_style: dict,
    samples_style: dict,
    error_style: dict,
):
    ds_gt = xr.open_zarr(path_gt / "statistics.zarr").load()
    ds_latent = xr.open_zarr(path_latent / "statistics.zarr").load()
    ds_samples = xr.open_zarr(path_samples / "statistics.zarr").load()

    # Computing physical wavelengths
    k_values = 2 * np.pi * EARTH_RADIUS / np.arange(1, 722)

    if isinstance(atmospheric_variables, str) and atmospheric_variables == "era5":
        atmospheric_variables = ERA5_ATMOSPHERIC_VARIABLES

    fig, ax = plt.subplots(
        len(levels),
        len(atmospheric_variables),
        figsize=(12.8, 15),
        gridspec_kw={"height_ratios": [1] * len(levels), "wspace": 0.065, "hspace": 0.065},
        squeeze=False,
    )

    data_legend = [
        Line2D([0], [0], label="Ground truth", color="black", linestyle="solid"),
        Line2D([0], [0], label="AE reconstructions", **ae_style),
        Line2D([0], [0], label="Samples", **samples_style),
    ]

    fig.legend(
        handles=data_legend,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.075),
        frameon=False,
        fontsize=13,
    )

    # Define target wavelengths
    target_wavelengths = [1e4, 1e3, 1e2]

    # Convert target wavelengths to indices
    target_indices = [np.argmin(np.abs(k_values - tw)) for tw in target_wavelengths]

    for v_index, v in enumerate(atmospheric_variables):
        # Extracting data
        ds_gt_v, ds_ae_v, ds_de_v = ds_gt[v], ds_latent[v], ds_samples[v]

        # Getting line
        gt_line = ds_gt_v.sel(statistics=quantile, level=levels).values
        ae_line = ds_ae_v.sel(statistics=quantile, level=levels).values
        de_line = ds_de_v.sel(statistics=quantile, level=levels).values

        gt_q5 = ds_gt_v.sel(statistics="Q5", level=levels).values
        gt_q90 = ds_gt_v.sel(statistics="Q95", level=levels).values

        ae_q5 = ds_ae_v.sel(statistics="Q5", level=levels).values
        ae_q90 = ds_ae_v.sel(statistics="Q95", level=levels).values

        de_q5 = ds_de_v.sel(statistics="Q5", level=levels).values
        de_q90 = ds_de_v.sel(statistics="Q95", level=levels).values

        # Compute asymmetric error bars
        gt_lower_errors = np.array([gt_line[:, idx] - gt_q5[:, idx] for idx in target_indices])
        gt_upper_errors = np.array([gt_q90[:, idx] - gt_line[:, idx] for idx in target_indices])

        ae_lower_errors = np.array([ae_line[:, idx] - ae_q5[:, idx] for idx in target_indices])
        ae_upper_errors = np.array([ae_q90[:, idx] - ae_line[:, idx] for idx in target_indices])

        de_lower_errors = np.array([de_line[:, idx] - de_q5[:, idx] for idx in target_indices])
        de_upper_errors = np.array([de_q90[:, idx] - de_line[:, idx] for idx in target_indices])

        color = CMAPS_LINE_DEEPMIND[v]["color"]

        for l in range(0, len(levels)):
            # Current level index
            index_level = l

            # Adding grid
            ax[l, v_index].grid(**grid_params)

            # Plot Ground Truth
            ax[l, v_index].plot(
                k_values, gt_line[index_level], color=color, alpha=0.85, label=PRETTY_VAR_NAMES[v]
            )

            # Plot Autoencoder (AE)
            ax[l, v_index].plot(k_values, ae_line[index_level], **ae_style)

            # Plot Denoiser (DE)
            ax[l, v_index].plot(k_values, de_line[index_level], **samples_style)

            ax[l, v_index].errorbar(
                [k_values[idx] for idx in target_indices],  # X values
                [gt_line[index_level, idx] for idx in target_indices],  # Y values
                yerr=[
                    gt_lower_errors[:, index_level],
                    gt_upper_errors[:, index_level],
                ],  # Asymmetric error bars
                fmt="o",
                color=color,
                **error_style,
                label=None,
            )

            ax[l, v_index].errorbar(
                [k_values[idx] for idx in target_indices],  # X values
                [ae_line[index_level, idx] for idx in target_indices],  # Y values
                yerr=[
                    ae_lower_errors[:, index_level],
                    ae_upper_errors[:, index_level],
                ],  # Asymmetric error bars
                fmt="o",
                color=ae_style["color"],
                **error_style,
                label=None,
            )

            ax[l, v_index].errorbar(
                [k_values[idx] for idx in target_indices],  # X values
                [de_line[index_level, idx] for idx in target_indices],  # Y values
                yerr=[
                    de_lower_errors[:, index_level],
                    de_upper_errors[:, index_level],
                ],  # Asymmetric error bars
                fmt="o",
                color=samples_style["color"],
                **error_style,
                label=None,
            )

            ax[l, v_index].invert_xaxis()
            ax[l, v_index].set(**plot_params)

            if l == 0:  # first row
                # Write above plot
                ax[l, v_index].text(
                    0.5,
                    1.05,
                    PRETTY_VAR_NAMES[v],
                    transform=ax[l, v_index].transAxes,
                    fontsize=10,
                    ha="center",
                    fontweight="bold",
                    va="bottom",
                )

            if v_index == 4:  # last column
                # Write to the right of the plot rotated
                ax[l, v_index].text(
                    1.05,
                    0.5,
                    f"{levels[l]} hPa",
                    transform=ax[l, v_index].transAxes,
                    fontsize=12,
                    va="center",
                    ha="left",
                    rotation=-90,
                    fontweight="bold",
                )

            if l == 5:
                ax[l, v_index].set_xlabel(
                    r"Wavelength $\lambda$ [km]", ha="center", **labels_params
                )
                ax[l, v_index].tick_params(labelsize=10)
            if v_index == 0:
                ax[l, v_index].set_ylabel(r"Power Spectral Density", ha="center", **labels_params)
                ax[l, v_index].set_yticks([1e0, 1e-6, 1e-12])
                ax[l, v_index].tick_params(labelsize=10)
            if l != 5:
                ax[l, v_index].set_xticks([])
            if v_index != 0:
                ax[l, v_index].set_yticks([1e0, 1e-6, 1e-12])
                ax[l, v_index].tick_params(labelleft=False)  # Hide labels but keep ticks

    # Adjusting DPI
    plt.gcf().set_dpi(300)
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.show()


def main():
    config = compose("configs/power_spectra.yaml", overrides=sys.argv[1:])

    plot_power_spectra(
        output_path=Path(config.output_path),
        path_latent=Path(config.path_latent),
        path_samples=Path(config.path_samples),
        path_gt=Path(config.path_gt),
        levels=config.levels,
        atmospheric_variables=config.atmospheric_variables,
        quantile=config.quantile,
        plot_params=config.plot_params,
        grid_params=config.grid_params,
        labels_params=config.labels_params,
        ae_style=config.ae_style,
        samples_style=config.samples_style,
        error_style=config.error_style,
    )


if __name__ == "__main__":
    main()
