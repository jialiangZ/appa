import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sys
import xarray as xr

from matplotlib.lines import Line2D
from os import PathLike
from pathlib import Path

from appa.config.hydra import compose
from appa.data.const import ERA5_PRESSURE_LEVELS, SUB_PRESSURE_LEVELS

ALT_HISTOGRAMS_FILE = "altitude_histograms.npy"
ALT_EDGES_FILE = "altitude_histogram_edges.npy"
GEOBALANCE_HISTOGRAMS_FILE = "geobalance_histograms_{}.npy"
GEOBALANCE_HISTOGRAMS_EDGES_FILE = "geobalance_histogram_edges_{}.npy"


def plot_physical_consistency(
    output_path: PathLike,
    path_latent: PathLike,
    path_samples: PathLike,
    path_gt: PathLike,
    levels: list[int],
    quantiles: list[float],
    show_samples: bool,
    show_latent: bool,
    tick_fontsize: int,
    label_fontsize: int,
):
    altitude_paths = {
        "gt": path_gt / ALT_HISTOGRAMS_FILE,
        "latent": path_latent / ALT_HISTOGRAMS_FILE,
        "samples": path_samples / ALT_HISTOGRAMS_FILE,
        "edges": path_latent / ALT_EDGES_FILE,
    }

    total_num_levels = np.load(altitude_paths["gt"]).shape[0]
    total_levels = (
        SUB_PRESSURE_LEVELS
        if total_num_levels == len(SUB_PRESSURE_LEVELS)
        else ERA5_PRESSURE_LEVELS
    )

    altitude_data = {
        "edges": np.load(altitude_paths["edges"]),
    }
    for k in ("gt", "latent", "samples"):
        altitude_data[k] = [
            np.load(altitude_paths[k])[total_levels.index(level)] for level in levels
        ]

    geobalance_paths = {
        a: {
            "gt": path_gt / f"{GEOBALANCE_HISTOGRAMS_FILE.format(a)}",
            "latent": path_latent / f"{GEOBALANCE_HISTOGRAMS_FILE.format(a)}",
            "samples": path_samples / f"{GEOBALANCE_HISTOGRAMS_FILE.format(a)}",
            "edges": path_latent / f"{GEOBALANCE_HISTOGRAMS_EDGES_FILE.format(a)}",
        }
        for a in ("cos", "sin")
    }

    geobalance_data = {
        a: {
            "edges": np.load(geobalance_paths[a]["edges"]),
        }
        for a in ("cos", "sin")
    }
    for a in ("cos", "sin"):
        for k in ("gt", "latent", "samples"):
            geobalance_data[a][k] = [
                np.load(geobalance_paths[a][k])[total_levels.index(level)] for level in levels
            ]

    correlation_paths = {
        "gt": path_gt / "correlation.zarr",
        "latent": path_latent / "correlation.zarr",
        "samples": path_samples / "correlation.zarr",
    }

    correlation_data = {
        k: xr.open_zarr(v)["correlation"].load() for k, v in correlation_paths.items()
    }

    correlation_quantiles = {}
    for k, v in correlation_data.items():
        correlation_quantiles[k] = [np.quantile(v, q, axis=0) for q in quantiles]

    ERA5_HIST = {
        "alpha": 0.5,
        "color": "blue",
        "label": "Ground truth",
    }

    SAMPLES_HIST = {
        "alpha": 1,
        "color": "black",
        "label": "Samples",
        "linestyle": "dotted",
        "linewidth": 3,
    }

    ERA5_CORR = {
        "alpha": 0.4,
        "linewidth": 3,
        "label": ERA5_HIST["label"],
        "color": ERA5_HIST["color"],
    }

    SAMPLES_CORR = {
        "alpha": 1,
        "linewidth": 3,
        "linestyle": SAMPLES_HIST["linestyle"],
        "label": SAMPLES_HIST["label"],
        "color": SAMPLES_HIST["color"],
    }

    ERA5_ERROR = {
        "fmt": "o",
        "alpha": 0.5,
        "capsize": 5,
        "markersize": 5,
        "color": ERA5_HIST["color"],
    }

    SAMPLES_ERROR = {
        "fmt": "o",
        "alpha": 0.5,
        "capsize": ERA5_ERROR["capsize"],
        "markersize": ERA5_ERROR["markersize"],
        "color": SAMPLES_HIST["color"],
    }

    GRID_PARAMS = {
        "which": "both",
        "linestyle": "-",
        "linewidth": 0.5,
        "alpha": 0.25,
    }

    fig = plt.figure(figsize=(12.8, 6.25))
    n_rows = len(levels) * 4
    gs = gridspec.GridSpec(
        n_rows,
        n_rows,
        height_ratios=[1] * n_rows,
        wspace=0.065,
        hspace=0.25,
    )

    data_legend = [
        Line2D([0], [0], **ERA5_CORR),
        Line2D([0], [0], **SAMPLES_CORR),
    ]

    # n_levels for ∆H, n_levels for cos, and n_levels for sin, then 1 for correlation
    axes = [None for _ in range(len(levels) * 3 + 1)]
    for i in range(len(levels)):
        for k, j in enumerate(list(range(n_rows))[i :: len(levels)][:3]):
            axes[j] = plt.subplot(
                gs[
                    i * n_rows // len(levels) : (i + 1) * n_rows // len(levels),
                    k * len(levels) : (k + 1) * len(levels),
                ]
            )
    axes[-1] = plt.subplot(gs[:, 3 * len(levels) : 4 * len(levels)])

    for ax in axes:
        ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
        ax.xaxis.label.set_size(label_fontsize)
        ax.yaxis.label.set_size(label_fontsize)

    # ∆H
    for i in range(len(levels)):
        ax = axes[i]

        ax.bar(
            altitude_data["edges"][:-1],
            altitude_data["gt"][i],
            width=np.diff(altitude_data["edges"]),
            align="edge",
            **ERA5_HIST,
        )
        if show_samples:
            ax.step(
                altitude_data["edges"][:-1],
                altitude_data["samples"][i],
                where="post",
                **SAMPLES_HIST,
            )
        if show_latent:
            ax.step(
                altitude_data["edges"][:-1],
                altitude_data["latent"][i],
                where="post",
                **SAMPLES_HIST,
            )
        ax.set_xlim(-0.15, 0.15)
        ax.set_ylim(1e-1, 1e2)
        ax.set_yscale("log")
        ax.grid(True, **GRID_PARAMS)

    # Angle between the wind and the geopotential gradient
    for i, a in enumerate(("cos", "sin")):
        axes_ = axes[(i + 1) * len(levels) : (i + 2) * len(levels)]
        for j, ax in enumerate(axes_):
            ax.bar(
                geobalance_data[a]["edges"][:-1],
                geobalance_data[a]["gt"][j],
                width=np.diff(geobalance_data[a]["edges"]),
                **ERA5_HIST,
            )
            if show_samples:
                ax.step(
                    geobalance_data[a]["edges"][:-1],
                    geobalance_data[a]["samples"][j],
                    where="post",
                    **SAMPLES_HIST,
                )
            if show_latent:
                ax.step(
                    geobalance_data[a]["edges"][:-1],
                    geobalance_data[a]["latent"][j],
                    where="post",
                    **SAMPLES_HIST,
                )
            ax.set_xlim(-1, 1)
            ax.set_ylim(1e-1, 1e2)
            ax.set_yscale("log")
            ax.grid(True, **GRID_PARAMS)

    # Plot the median correlation (Q50) with the uncertainty band (Q5-Q95)
    axes[-1].plot(correlation_quantiles["gt"][1], total_levels, **ERA5_CORR)
    if show_samples:
        axes[-1].plot(correlation_quantiles["samples"][1], total_levels, **SAMPLES_CORR)
    if show_latent:
        axes[-1].plot(correlation_quantiles["latent"][1], total_levels, **SAMPLES_CORR)

    # Error bars (first and third quantiles)
    for k, show, plot_params in zip(
        ("gt", "latent", "samples"),
        (True, show_latent, show_samples),
        (ERA5_ERROR, SAMPLES_ERROR, SAMPLES_ERROR),
    ):
        if show:
            axes[-1].errorbar(
                correlation_quantiles[k][1],
                total_levels,
                xerr=[
                    correlation_quantiles[k][1] - correlation_quantiles[k][0],
                    correlation_quantiles[k][2] - correlation_quantiles[k][1],
                ],
                **plot_params,
            )

    # Settings
    axes[-1].set_xlim(0.2, 1)
    axes[-1].set_ylabel("Pressure level [hPa]", rotation=270, labelpad=10)
    axes[-1].set_xlabel("Pearson correlation coefficient [-]", labelpad=10)
    axes[-1].yaxis.set_label_position("right")
    axes[-1].yaxis.tick_right()
    axes[-1].invert_yaxis()
    axes[-1].grid(True, linestyle="--", alpha=0.6)

    for i in range(len(levels) - 1):
        for ax in axes[i :: len(levels)][:3]:
            ax.set_xticklabels([])
            ax.set_xticklabels([])
            ax.set_xticklabels([])

    for ax in axes[len(levels) : -1]:
        ax.set_yticklabels([])

    for i, ax in enumerate(axes[len(levels) - 1 :: 2][:3]):
        if i == 0:
            ax.set_xticks([-0.1, 0, 0.1])
        else:
            ax.set_xticks([-0.5, 0, 0.5])

    # Add "cos θ_diff" text to the top right of each subplot
    for ax in axes[: len(levels)]:
        ax.text(
            0.95,
            0.95,
            r"$\Delta H$",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )
    for ax in axes[len(levels) : 2 * len(levels)]:
        ax.text(
            0.95,
            0.95,
            r"$\cos (\theta)$",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )
    for ax in axes[2 * len(levels) : 3 * len(levels)]:
        ax.text(
            0.95,
            0.95,
            r"$\sin (\theta)$",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

    for i, level in enumerate(levels):
        for ax in axes[i :: len(levels)][:3]:
            ax.text(
                0.95,
                0.85,
                f"{level} [hPa]",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
            )

    fig.legend(
        handles=data_legend,
        loc="upper center",
        bbox_to_anchor=(0.22, 0.94),
        frameon=True,
        fontsize=label_fontsize,
        ncol=2,
    )

    plt.savefig(output_path, dpi=600)
    plt.show()


def main():
    config = compose("configs/physical_consistency.yaml", overrides=sys.argv[1:])

    plot_physical_consistency(
        output_path=Path(config.output_path),
        path_latent=Path(config.path_latent),
        path_samples=Path(config.path_samples),
        path_gt=Path(config.path_gt),
        levels=config.levels,
        quantiles=config.quantiles,
        show_samples=config.show_samples,
        show_latent=config.show_latent,
        tick_fontsize=config.tick_fontsize,
        label_fontsize=config.label_fontsize,
    )


if __name__ == "__main__":
    main()
