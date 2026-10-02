r"""Plots the distribution of the AE reconstruction signed error."""

import h5py
import matplotlib
import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.data.const import (
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.diagnostics.const import PRETTY_VAR_NAMES, UNITS


def plot_reconstruction_distribution(config):
    reconstruction_path = Path(config.reconstruction_path)
    output_dir = reconstruction_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    ae_cfg = compose(reconstruction_path.parents[1] / "config.yaml")

    with h5py.File(reconstruction_path / "errors.h5", "r") as f:
        signed_errors_histograms = f["signed_errors_histograms"][()] / (
            1440 * 721
        )  # (N, Z, num_bins)
        signed_errors_bin_edges = f["signed_errors_bin_edges"][()]  # (N, Z, num_bins + 1)

    ylabel = "Fraction of pixels at error [%]"

    level_colors = config.level_colors

    pressure_levels = (
        SUB_PRESSURE_LEVELS if ae_cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )

    max_num_vars = max(len(config.surface_variables), len(config.atmosphere_variables))

    _, axs = plt.subplots(
        2,
        max_num_vars,
        figsize=(12.8, 5.13),
        gridspec_kw={"height_ratios": [1, 1], "wspace": 0.065, "hspace": 0.195},
        squeeze=False,
    )

    surface_variables = config.surface_variables
    atmosphere_variables = config.atmosphere_variables

    legend_entries = {}

    for i, var in enumerate(surface_variables):
        ax = axs[0, i]

        var_name = var.name

        var_idx = ERA5_SURFACE_VARIABLES.index(var_name)

        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=0)

        hist = signed_errors_histograms[:, var_idx]
        if "unit" in var:
            unit_name = var.unit
        else:
            unit_name = UNITS.get(var_name, var_name)
        unit_name = PRETTY_VAR_NAMES[var_name] + " [" + unit_name + "]"

        bin_edges = signed_errors_bin_edges[:, var_idx].mean(
            axis=0
        )  # Same bin edges for all samples

        mean_hist = np.mean(hist, axis=0) * 100
        std_hist = np.std(hist, axis=0) * 100
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        (line,) = ax.plot(
            bin_centers,
            mean_hist,
            label="Surface mean",
            linewidth=1.5,
            color="black",
        )
        ax.fill_between(
            bin_centers,
            mean_hist - std_hist,
            mean_hist + std_hist,
            color="gray",
            alpha=0.3,
            label="Surface std",
        )
        ax.annotate(
            unit_name,
            xy=(0.965, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            size=config.font_sizes.variable_names,
            bbox=dict(
                boxstyle="round",
                facecolor=(1, 1, 1, 0.65),
                edgecolor=(0, 0, 0, 1.0),
                linewidth=0.8,
            ),
        )
        if "Surface" not in legend_entries:
            legend_entries["Surface"] = line

    for i, var in enumerate(atmosphere_variables):
        ax = axs[1, i]

        var_name = var.name
        if "unit" in var:
            unit_name = var.unit
        else:
            unit_name = UNITS.get(var_name, var_name)
        unit_name = PRETTY_VAR_NAMES[var_name] + " [" + unit_name + "]"

        var_idx = len(ERA5_SURFACE_VARIABLES) + len(
            pressure_levels
        ) * ERA5_ATMOSPHERIC_VARIABLES.index(var_name)

        ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=0)
        ax.annotate(
            unit_name,
            xy=(0.965, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            size=config.font_sizes.variable_names,
            bbox=dict(
                boxstyle="round",
                facecolor=(1, 1, 1, 0.65),
                edgecolor=(0, 0, 0, 1.0),
                linewidth=0.8,
            ),
        )

        for l, level in enumerate(pressure_levels):
            hist = signed_errors_histograms[:, var_idx + l]
            bin_edges = signed_errors_bin_edges[:, var_idx + l].mean(
                axis=0
            )  # Same bin edges for all samples
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            (line,) = ax.plot(
                bin_centers,
                hist.mean(axis=0) * 100,
                label=f"{level}",
                linewidth=0.5,
                color=level_colors.get(level, "#000000"),
            )
            if level not in legend_entries:
                legend_entries[level] = line

    for i in (0, 1):
        axs[i, 0].set_ylabel(ylabel, fontsize=config.font_sizes.axis_labels)
        axs[i, 0].tick_params(axis="y", labelsize=config.font_sizes.tick_labels)

    axs = axs.flatten()

    for i, var in enumerate(surface_variables + atmosphere_variables):
        ax = axs[i]

        ax.set_ylim(1e-3, 100)
        ax.set_yscale("log")
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_useMathText(False)
        ax.set_xscale("linear")
        ax.tick_params(axis="x", labelsize=config.font_sizes.tick_labels)

        var_min, var_max = var.bin_range
        tick_positions = np.linspace(var_min, var_max, num=5)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(tick_positions))

        ax.grid(True, alpha=0.5, linestyle="-")

    axs[-1].axis("off")
    legend = axs[-1].legend(
        legend_entries.values(),
        legend_entries.keys(),
        title="Levels [hPa]",
        title_fontsize=config.font_sizes.level_legend,
        fontsize=config.font_sizes.level_legend,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.001, 1.025),
        frameon=True,
    )
    for (
        legend_handle
    ) in legend.legend_handles:  # Increase line width to be colorblindness-friendly
        legend_handle.set_linewidth(2.0)

    plt.savefig(
        output_dir / config.output_file,
        dpi=config.figure_dpi,
        transparent=True,
        bbox_inches="tight",
    )
    print("Saved to", output_dir / config.output_file)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(
            "python reconstruction_dist.py [+id=<id>] reconstruction_path=... param1=A param2=B ..."
        )
        return

    config = compose("configs/reconstruction_dist.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    plot_reconstruction_distribution(config)


if __name__ == "__main__":
    main()
