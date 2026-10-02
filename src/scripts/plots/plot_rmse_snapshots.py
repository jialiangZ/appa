r"""Plots the per-variable RMSE & reconstruction snapshots."""

import h5py
import matplotlib.patheffects as path_effects
import numpy as np
import sys

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
from appa.diagnostics.const import PRETTY_VAR_NAMES


def plot_rmse_snapshots(config):
    reconstruction_path = Path(config.reconstruction_path)
    output_dir = reconstruction_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(reconstruction_path / "errors.h5", "r") as f:
        std_rmse = np.sqrt(f["std_mse"][()][:, 0])
    mean_rmse = np.mean(std_rmse, axis=0)
    std_rmse = np.std(std_rmse, axis=0)

    ae_config = compose(reconstruction_path.parents[1] / "config.yaml")
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
        "RMSE (standardized data)", fontsize=font_sizes["axis_label"], fontweight="bold"
    )
    ax1.set_ylabel("Pressure level [hPa]", fontsize=font_sizes["axis_label"], fontweight="bold")
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

    gt_axes = [
        plt.subplot(gs[:2, 2:4]),
        plt.subplot(gs[:2, 4:6]),
        plt.subplot(gs[:2, 6:8]),
        plt.subplot(gs[4:6, 2:4]),
        plt.subplot(gs[4:6, 4:6]),
        plt.subplot(gs[4:6, 6:8]),
    ]
    pred_axes = [
        plt.subplot(gs[2:4, 2:4]),
        plt.subplot(gs[2:4, 4:6]),
        plt.subplot(gs[2:4, 6:8]),
        plt.subplot(gs[6:8, 2:4]),
        plt.subplot(gs[6:8, 4:6]),
        plt.subplot(gs[6:8, 6:8]),
    ]

    with h5py.File(reconstruction_path / "sample.h5", "r") as f:
        state = f["gt"][()]
        x_pred = f["pred"][()]

    # Plot GT & reconstructions
    for i, (ax_gt, ax_pred) in enumerate(zip(gt_axes, pred_axes)):
        if i >= len(config.snapshot_variables):
            ax_gt.axis("off")
            ax_pred.axis("off")
            continue

        variable_info = config.snapshot_variables[i]

        if variable_info.type == "surface":
            var_idx = ERA5_SURFACE_VARIABLES.index(variable_info.name)
        else:
            var_idx = len(
                ERA5_SURFACE_VARIABLES
            ) + num_pressure_levels * ERA5_ATMOSPHERIC_VARIABLES.index(variable_info.name)
            var_idx += pressure_levels.index(variable_info.level)

        gt_img = state[0, var_idx]
        pred_img = x_pred[0, var_idx]

        if config.center_europe:
            gt_img = np.roll(gt_img, shift=gt_img.shape[1] // 2, axis=1)
            pred_img = np.roll(pred_img, shift=pred_img.shape[1] // 2, axis=1)

        vmin, vmax = np.quantile(gt_img, (config.quantile_cmap, 1.0 - config.quantile_cmap))

        ax_gt.imshow(gt_img, cmap="jet", vmin=vmin, vmax=vmax, interpolation=None)
        ax_pred.imshow(pred_img, cmap="jet", vmin=vmin, vmax=vmax, interpolation=None)

        for ax in (ax_gt, ax_pred):
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        text = ax_gt.text(
            0.02,
            0.83,
            "ABCDEF"[i],
            transform=ax_gt.transAxes,
            fontsize=font_sizes["ticks"],
            verticalalignment="bottom",
            fontweight="bold",
            color="white",
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="black"),
            path_effects.Normal(),
        ])

    for ax in [gt_axes[2], gt_axes[5]]:
        ax.set_ylabel("Ground truth", fontsize=font_sizes["axis_label"], rotation=270, labelpad=15)
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_visible(True)

    for ax in [pred_axes[2], pred_axes[5]]:
        ax.set_ylabel(
            "Reconstruction", fontsize=font_sizes["axis_label"], rotation=270, labelpad=15
        )
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_visible(True)

    plt.savefig(
        output_dir / config.output_file,
        bbox_inches="tight",
        dpi=900,
    )
    print("Saved to", output_dir / config.output_file)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print("python rmse_snapshots.py [+id=<id>] reconstruction_path=... param1=A param2=B ...")
        return

    config = compose("configs/rmse_snapshots.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    plot_rmse_snapshots(config)


if __name__ == "__main__":
    main()
