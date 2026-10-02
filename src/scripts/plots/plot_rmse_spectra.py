r"""Plots the per-variable RMSE of the AE reconstruction error."""

import h5py
import numpy as np
import sys
import xarray as xr

from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.data.const import (
    ATM_SURF_VARIABLE_MAPPINGS,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.diagnostics.const import CMAPS_LINE_DEEPMIND, EARTH_RADIUS, PRETTY_VAR_NAMES


def plot_power_spectra(
    fig,
    ax,
    path_latent,
    path_samples,
    path_gt,
    surface_variables,
    atmospheric_variables,
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

    if isinstance(surface_variables, str) and surface_variables == "era5":
        surface_variables = ERA5_SURFACE_VARIABLES
    if isinstance(atmospheric_variables, str) and atmospheric_variables == "era5":
        atmospheric_variables = ERA5_ATMOSPHERIC_VARIABLES

    data_legend = [
        Line2D([0], [0], label="Ground truth", color="black", linestyle="solid"),
        Line2D([0], [0], label="AE reconstructions", **ae_style),
        Line2D([0], [0], label="Samples", **samples_style),
    ]

    fig.legend(
        handles=data_legend,
        loc="lower left",
        bbox_to_anchor=(0.3195, 0.5025),
        frameon=True,
        fontsize=8.5,
    )

    # Define target wavelengths
    target_wavelengths = [1e4, 1e3, 1e2]

    # Convert target wavelengths to indices
    target_indices = [np.argmin(np.abs(k_values - tw)) for tw in target_wavelengths]

    for i_ in range(6):
        i = i_ % 3
        j = i_ // 3

        if i_ >= len(surface_variables):
            ax[j][i].axis("off")
            ax[j][i].set_visible(False)

            continue

        v = surface_variables[i_]

        # Extracting data
        ds_gt_v, ds_ae_v, ds_de_v = ds_gt[v], ds_latent[v], ds_samples[v]

        # Getting line
        gt_line = ds_gt_v.sel(statistics=quantile).values
        ae_line = ds_ae_v.sel(statistics=quantile).values
        de_line = ds_de_v.sel(statistics=quantile).values

        gt_q5 = ds_gt_v.sel(statistics="Q5").values
        gt_q90 = ds_gt_v.sel(statistics="Q95").values

        ae_q5 = ds_ae_v.sel(statistics="Q5").values
        ae_q90 = ds_ae_v.sel(statistics="Q95").values

        de_q5 = ds_de_v.sel(statistics="Q5").values
        de_q90 = ds_de_v.sel(statistics="Q95").values

        # Compute asymmetric error bars
        gt_lower_errors = [gt_line[idx] - gt_q5[idx] for idx in target_indices]
        gt_upper_errors = [gt_q90[idx] - gt_line[idx] for idx in target_indices]

        ae_lower_errors = [ae_line[idx] - ae_q5[idx] for idx in target_indices]
        ae_upper_errors = [ae_q90[idx] - ae_line[idx] for idx in target_indices]

        de_lower_errors = [de_line[idx] - de_q5[idx] for idx in target_indices]
        de_upper_errors = [de_q90[idx] - de_line[idx] for idx in target_indices]

        color = CMAPS_LINE_DEEPMIND[v]["color"]

        # Adding grid
        ax[j][i].grid(True, **grid_params)

        # Plot Ground Truth
        ax[j][i].plot(k_values, gt_line, color=color, alpha=0.85, label=PRETTY_VAR_NAMES[v])

        # Plot Autoencoder (AE)
        ax[j][i].plot(k_values, ae_line, **ae_style)

        # Plot Denoiser (DE)
        ax[j][i].plot(k_values, de_line, **samples_style)

        ax[j][i].errorbar(
            [k_values[idx] for idx in target_indices],  # X values
            [gt_line[idx] for idx in target_indices],  # Y values
            yerr=[gt_lower_errors, gt_upper_errors],  # Asymmetric error bars
            fmt="o",
            color=color,
            **error_style,
            label=None,
        )

        ax[j][i].errorbar(
            [k_values[idx] for idx in target_indices],  # X values
            [ae_line[idx] for idx in target_indices],  # Y values
            yerr=[ae_lower_errors, ae_upper_errors],  # Asymmetric error bars
            fmt="o",
            color=ae_style["color"],
            **error_style,
            label=None,
        )

        ax[j][i].errorbar(
            [k_values[idx] for idx in target_indices],  # X values
            [de_line[idx] for idx in target_indices],  # Y values
            yerr=[de_lower_errors, de_upper_errors],  # Asymmetric error bars
            fmt="o",
            color=samples_style["color"],
            **error_style,
            label=None,
        )

        ax[j][i].invert_xaxis()
        ax[j][i].set(**plot_params)

        # Formatting
        ax[j][i].legend(loc="upper right", fontsize=8)
        # ax[j][i].text(0.97, 0.87, "Surface", transform=ax[j][i].transAxes, **text_params)

        if j == 1:
            ax[j][i].set_xlabel(
                r"Wavelength $\lambda$ [km]", labelpad=5, ha="center", **labels_params
            )
            ax[j][i].tick_params(labelsize=10)
        else:
            ax[j][i].set_xticks([])

        if i == 2:
            ax[j][i].yaxis.set_label_position("right")
            ax[j][i].yaxis.tick_right()
            ax[j][i].set_ylabel(
                r"Power Spectral Density", ha="center", rotation=270, labelpad=15, **labels_params
            )
            ax[j][i].set_yticks([1e0, 1e-6, 1e-12])
            ax[j][i].tick_params(labelsize=10)
        else:
            # ax[j][i].set_yticks([1e0, 1e-6, 1e-12])
            # ax[j][i].tick_params(labelleft=False)  # Hide labels but keep ticks
            ax[j][i].set_yticklabels([])
            ax[j][i].set_yticks([1e0, 1e-6, 1e-12])
            ax[j][i].yaxis.grid(True)
            ax[j][i].tick_params(axis="y", which="both", length=0)


def plot_reconstruction_rmse(config):
    reconstruction_path = Path(config.reconstruction_path)
    output_dir = reconstruction_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(reconstruction_path / "errors.h5", "r") as f:
        std_rmse = np.sqrt(f["std_mse"][()][:, 0])
    mean_rmse = np.mean(std_rmse, axis=0)
    std_rmse = np.std(std_rmse, axis=0)

    ae_config = compose(Path(config.path_latent).parent / "ae" / "config.yaml")
    pressure_levels = (
        SUB_PRESSURE_LEVELS if ae_config.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )
    num_pressure_levels = len(pressure_levels)
    surface_pressure_level = (
        pressure_levels[-1] + 50
    )  # Add 50 artificially to have a sufficient gap in the plot.

    surface_variables = config.surface_variables
    atmospheric_variables = config.atmospheric_variables

    mean_rmse_dict, std_rmse_dict = {}, {}
    for atm_var in atmospheric_variables:
        var_idx = len(
            ERA5_SURFACE_VARIABLES
        ) + num_pressure_levels * ERA5_ATMOSPHERIC_VARIABLES.index(atm_var)

        mean_rmse_dict[atm_var] = {
            "atm": mean_rmse[var_idx : var_idx + num_pressure_levels],
        }
        std_rmse_dict[atm_var] = {
            "atm": std_rmse[var_idx : var_idx + num_pressure_levels],
        }

    for surf_var in surface_variables:
        var_idx = ERA5_SURFACE_VARIABLES.index(surf_var)

        associated_atm_var = ATM_SURF_VARIABLE_MAPPINGS.get(surf_var, None)

        if associated_atm_var is None or associated_atm_var not in ERA5_ATMOSPHERIC_VARIABLES:
            mean_rmse_dict[surf_var] = {
                "surface": mean_rmse[var_idx],
            }
            std_rmse_dict[surf_var] = {
                "surface": std_rmse[var_idx],
            }
        else:
            mean_rmse_dict[associated_atm_var]["surface"] = mean_rmse[var_idx]
            std_rmse_dict[associated_atm_var]["surface"] = std_rmse[var_idx]

    font_sizes = config.font_sizes

    fig = plt.figure(figsize=(12.8, 6.25))
    gs = gridspec.GridSpec(
        8,
        8,
        height_ratios=[1] * 8,
        wspace=0.065,
        hspace=0.25,
    )

    ax1 = plt.subplot(gs[:, :2])

    legend_handles = []
    for variable in mean_rmse_dict.keys():
        style = config.styles[variable]
        mean_rmse = mean_rmse_dict[variable]
        std_rmse = std_rmse_dict[variable]

        if "surface" in mean_rmse:
            ax1.scatter(
                mean_rmse["surface"],
                surface_pressure_level,
                label=PRETTY_VAR_NAMES.get(variable, variable),
                **style,
            )
            ax1.errorbar(
                mean_rmse["surface"],
                surface_pressure_level,
                xerr=std_rmse["surface"],
                fmt="none",
                capsize=5,
                capthick=1,
                elinewidth=1,
                **style,
            )

        if "atm" in mean_rmse:
            ax1.plot(
                mean_rmse["atm"],
                pressure_levels,
                label=PRETTY_VAR_NAMES.get(variable, variable),
                **style,
            )
            ax1.errorbar(
                mean_rmse["atm"],
                pressure_levels,
                xerr=std_rmse["atm"],
                fmt="none",
                capsize=5,
                capthick=1,
                elinewidth=1,
                **style,
            )

        legend_handles.append(
            Line2D([0], [0], label=PRETTY_VAR_NAMES.get(variable, variable), **style)
        )

    # Finalize axis appearance
    ax1.set_xlabel(
        "RMSE (standardized data)",
        fontsize=font_sizes["axis_label"],  # fontweight="bold"
    )
    ax1.set_ylabel(
        "Pressure level [hPa]", fontsize=font_sizes["axis_label"]
    )  # , fontweight="bold")
    ax1.set_xlim(config.ticks.error_min, config.ticks.error_max)
    ax1.set_xticks(config.ticks.error)
    selected_ticks = [surface_pressure_level] + config.ticks.levels
    selected_labels = ["Surface"] + [str(l) for l in config.ticks.levels]

    ax1.set_yticks(selected_ticks)
    ax1.set_yticklabels(selected_labels, fontsize=font_sizes["ticks"])

    ax1.tick_params(axis="x", labelsize=font_sizes["ticks"])
    ax1.tick_params(axis="y", labelsize=font_sizes["ticks"])
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.6)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=True,
        fontsize=font_sizes["legend"],
        ncol=4,
    )

    axes = [
        [
            plt.subplot(gs[:4, 2:4]),
            plt.subplot(gs[:4, 4:6]),
            plt.subplot(gs[:4, 6:8]),
        ],
        [
            plt.subplot(gs[4:, 2:4]),
            plt.subplot(gs[4:, 4:6]),
            plt.subplot(gs[4:, 6:8]),
        ],
    ]

    plot_power_spectra(
        fig,
        axes,
        path_latent=Path(config.path_latent),
        path_samples=Path(config.path_samples),
        path_gt=Path(config.path_gt),
        surface_variables=ERA5_SURFACE_VARIABLES,
        atmospheric_variables=ERA5_ATMOSPHERIC_VARIABLES,
        **config.power_spectra,
    )

    plt.savefig(
        output_dir / config.output_file,
        bbox_inches="tight",
        dpi=900,
    )
    print("Saved to", output_dir / config.output_file)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python rmse_spectra.py [+id=<id>] reconstruction_path=... param1=A param2=B ...")
        return

    config = compose("configs/rmse_spectra.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    plot_reconstruction_rmse(config)


if __name__ == "__main__":
    main()
