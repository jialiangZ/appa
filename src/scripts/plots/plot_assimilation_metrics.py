import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
import xarray as xr

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path

from appa.config.hydra import compose
from appa.utils.math import str_to_ids

warnings.filterwarnings("ignore")


def load_reanalysis_data(path, metrics, metrics_id, channels, levels, windows, filtering=False):
    output = {metric: {} for metric in metrics}

    group = "avg_ensembles" if filtering else "avg_time"

    for L in windows:
        d = xr.open_zarr(path / f"{L}h" / "evaluation" / metrics_id / "metrics.zarr", group=group)

        for metric in metrics:
            output[metric][L] = []
            for channel, level in zip(channels, levels):
                d_channel = d[channel]

                sel_dict = {"metric": metric}
                if level is not None:
                    sel_dict["level"] = level
                if channel == "sea_surface_temperature":
                    sel_dict["region"] = (
                        "land" if "land" in d_channel.region.values else "ocean"
                    )  # backward compatibility
                elif "region" in d_channel.dims:
                    sel_dict["region"] = "global"

                val = d_channel.sel(**sel_dict).values.tolist()

                output[metric][L] += [val[-1]] if filtering else [val]

    return output


def load_forecast_data(path, metrics, eval_id, channels, levels):
    r"""Loads forecasting data into a dictionary for some metrics, channels and levels.

    Args:
        path (Path): Path to the directory containing the forecasting data (e.g., .../forecast/id/2_72h).
        metrics (list): List of metrics to load.
        eval_id (str): ID of the evaluation run.
        channels (list): List of channels to load.
        levels (list): List of the levels corresponding to each channel to load.

    Returns:
        output (dict): Dictionary containing the loaded data for each metric and channel.
        dt (int): Hours between two states in the trajectory.
    """
    output = {metric: {} for metric in metrics}

    d = xr.open_zarr(path / "evaluation" / eval_id / "metrics.zarr", group="avg_ensembles")
    for channel, level in zip(channels, levels):
        d_channel = d[channel]
        for metric in metrics:
            sel_dict = {"metric": metric}
            if level is not None:
                sel_dict["level"] = level
            if channel == "sea_surface_temperature":
                sel_dict["region"] = (
                    "land" if "land" in d_channel.region.values else "ocean"
                )  # backward compatibility
            elif "region" in d_channel.dims:
                sel_dict["region"] = "global"

            val = d.sel(**sel_dict)[channel].values.tolist()
            output[metric][channel] = [val] if isinstance(val, float) else val

    denoiser_cfg = compose(path.parents[2] / "config.yaml")
    dt = denoiser_cfg.train.blanket_dt

    return output, dt


def plot_skill_crps(config):
    metrics = config.metrics
    line_width = config.line_width
    channel_multipliers = [c.multiplier if "multiplier" in c else 1 for c in config.channels]
    channels = [c.name for c in config.channels]
    channel_names = [c.display_name for c in config.channels]
    levels = [c.level if "level" in c else None for c in config.channels]

    reanalysis_path = Path(config.reanalysis.path)
    forecasting_obs_path = Path(config.forecasting.observational.path)
    forecasting_full_path = Path(config.forecasting.full.path)
    forecasting_persistence_path = Path(config.forecasting.persistence.path)
    forecasting_prior_path = Path(config.forecasting.prior.path)

    assimilation_window_sizes = str_to_ids(config.reanalysis.window_sizes)

    reanalysis_data = load_reanalysis_data(
        reanalysis_path,
        metrics,
        config.reanalysis.eval_id,
        channels,
        levels,
        assimilation_window_sizes,
    )
    filtering_data = load_reanalysis_data(
        reanalysis_path,
        metrics,
        config.reanalysis.eval_id,
        channels,
        levels,
        assimilation_window_sizes,
        filtering=True,
    )
    forecasting_obs_data, forecasting_obs_dt = load_forecast_data(
        forecasting_obs_path, metrics, config.forecasting.observational.eval_id, channels, levels
    )
    forecasting_full_data, forecasting_full_dt = load_forecast_data(
        forecasting_full_path, metrics, config.forecasting.full.eval_id, channels, levels
    )
    forecasting_persistence_data, forecasting_persistence_dt = load_forecast_data(
        forecasting_persistence_path,
        metrics,
        config.forecasting.persistence.eval_id,
        channels,
        levels,
    )
    forecasting_prior_data, forecasting_prior_dt = load_forecast_data(
        forecasting_prior_path, metrics, config.forecasting.prior.eval_id, channels, levels
    )

    fig = plt.figure(figsize=(6.4 * len(metrics), 2 * len(channels)))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    width_ratios = [1, 1, 1]
    for _ in range(len(metrics) - 1):
        width_ratios += [0.225, 1, 1, 1]
    gs = GridSpec(len(channels), len(width_ratios), figure=fig, width_ratios=width_ratios)
    axes = [
        [
            fig.add_subplot(gs[row, col if col < 3 else col + 1 if col < 6 else col + 2])
            for col in range(3 * len(metrics))
        ]
        for row in range(len(channels))
    ]

    title_positions = [[0.5], [0.31, 0.715], [0.245, 0.5115, 0.7775]][len(metrics) - 1]
    metrics_translations = {
        "crps": "CRPS",
        "skill": "Skill",
        "spread": "Spread",
        "spread_skill_ratio": "Spread-skill ratio",
    }
    for i, metric in enumerate(metrics):
        fig.text(
            title_positions[i],
            0.9,
            s=metrics_translations[metric],
            ha="center",
            va="bottom",
            fontsize=config.font_sizes.metric_titles,
            weight="bold",
        )

    font_size_ticklabels = config.font_sizes.tick_labels

    grid_params = {
        "visible": True,
        "axis": "both",
        "alpha": 0.6,
    }

    for i, c in enumerate(channels):
        last_row = i == len(channels) - 1

        for k, metric in enumerate(metrics):
            col_offset = k * 3

            channel_mult = channel_multipliers[i] if metric != "spread_skill_ratio" else 1.0

            # Reanalysis
            data_rea = [reanalysis_data[metric][L][i] for L in assimilation_window_sizes]
            data_rea = np.array(data_rea) * channel_mult

            axes[i][col_offset].plot(
                assimilation_window_sizes,
                data_rea,
                color=config.colors.aao_obs,
                linewidth=line_width,
            )

            ticks = config.reanalysis_filtering_ticks
            axes[i][col_offset].set_xticks(ticks)
            axes[i][col_offset].set_xticklabels(
                [
                    str(int(t / 24)) if config.reanalysis_filtering_ticks_show_days else t
                    for t in ticks
                ],
                fontsize=font_size_ticklabels,
            )
            axes[i][col_offset].grid(**grid_params)
            if last_row:
                axes[i][col_offset].set_xlabel(
                    "Assimilation length [days]", fontsize=font_size_ticklabels
                )
            else:
                axes[i][col_offset].tick_params(labelbottom=False)
                axes[i][col_offset].tick_params(axis="x", which="both", length=0)

            # Filtering
            data_filt = [filtering_data[metric][L][i] for L in assimilation_window_sizes]
            data_filt = np.array(data_filt) * channel_mult

            axes[i][col_offset + 1].plot(
                assimilation_window_sizes,
                data_filt,
                color=config.colors.aao_obs,
                linewidth=line_width,
            )

            axes[i][col_offset + 1].set_xticks(ticks)
            axes[i][col_offset + 1].set_xticklabels(
                [
                    str(int(t / 24)) if config.reanalysis_filtering_ticks_show_days else t
                    for t in ticks
                ],
                fontsize=font_size_ticklabels,
            )
            axes[i][col_offset + 1].grid(**grid_params)
            if last_row:
                axes[i][col_offset + 1].set_xlabel(
                    "Assimilation length [days]", fontsize=font_size_ticklabels
                )
            else:
                axes[i][col_offset + 1].tick_params(labelbottom=False)
                axes[i][col_offset + 1].tick_params(axis="x", which="both", length=0)
            axes[i][col_offset + 1].tick_params(axis="y", which="both", length=0)

            # Forecast (24h assimilation length)
            y_forecast_obs = np.array(forecasting_obs_data[metric][c]) * channel_mult
            y_forecast_full = np.array(forecasting_full_data[metric][c]) * channel_mult
            y_persistence = np.array(forecasting_persistence_data[metric][c]) * channel_mult
            y_prior = np.array(forecasting_prior_data[metric][c]) * channel_mult

            x_offset_forecast_obs = int(forecasting_obs_path.stem.split("_")[0])
            x_offset_forecast_full = int(forecasting_full_path.stem.split("_")[0])
            x_offset_forecast_persistence = int(forecasting_persistence_path.stem.split("_")[0])

            x_forecast_obs = forecasting_obs_dt * (
                np.arange(len(y_forecast_obs)) - x_offset_forecast_obs + 1
            )
            x_forecast_full = forecasting_full_dt * (
                np.arange(len(y_forecast_full)) - x_offset_forecast_full + 1
            )
            x_persistence = forecasting_persistence_dt * (
                np.arange(len(y_persistence)) - x_offset_forecast_persistence + 1
            )
            x_prior = forecasting_prior_dt * (np.arange(len(y_prior)) + 1)

            # Forecast from observations
            axes[i][col_offset + 2].plot(
                x_forecast_obs,
                y_forecast_obs,
                color=config.colors.aao_obs,
                linewidth=line_width,
            )
            # Forecast from full states
            axes[i][col_offset + 2].plot(
                x_forecast_full,
                y_forecast_full,
                color=config.colors.ar_full,
                linestyle="-",
                linewidth=line_width,
            )
            # Persistence model
            axes[i][col_offset + 2].plot(
                x_persistence,
                y_persistence,
                color=config.colors.persistence,
                linestyle="--",
                linewidth=line_width,
            )
            # Prior generated samples
            axes[i][col_offset + 2].plot(
                x_prior,
                y_prior,
                color=config.colors.prior,
                linestyle="--",
                linewidth=line_width,
            )

            # Baselines for skill
            if levels[i] is None:
                file_name = c
            else:
                file_name = f"{c}_{levels[i]}"
            if metric == "skill":
                for baseline in ("ifs", "graphdop"):
                    if not os.path.exists(f"baselines/{baseline}/{file_name}_x.npy"):
                        continue

                    x_ = np.load(f"baselines/{baseline}/{file_name}_x.npy")
                    y_ = np.load(f"baselines/{baseline}/{file_name}_y.npy")
                    # Prior generated samples
                    axes[i][col_offset + 2].plot(
                        x_ * 24,
                        y_,
                        color=config.colors[baseline],
                        linestyle="-",
                        linewidth=line_width,
                    )

            axes[i][col_offset + 2].axvline(x=0, color="silver", linestyle="dotted", linewidth=2.5)
            ticks = config.forecasting_ticks
            axes[i][col_offset + 2].set_xticks(ticks)
            axes[i][col_offset + 2].set_xticklabels(
                [str(int(t / 24)) if config.forecast_ticks_show_days else t for t in ticks],
                fontsize=font_size_ticklabels,
            )
            if last_row:
                axes[i][col_offset + 2].set_xlabel(
                    "Lead time [days]", fontsize=font_size_ticklabels
                )
            else:
                axes[i][col_offset + 2].tick_params(axis="x", which="both", length=0)
                axes[i][col_offset + 2].tick_params(labelbottom=False)
            axes[i][col_offset + 2].grid(**grid_params)
            axes[i][col_offset + 2].tick_params(axis="y", which="both", length=0)

            # If None, will be set to data, if not None, overrides data y_low.
            y_lows = [None, None, None, None, 0, None, None, None, None]
            diff_margin_low = 0.2
            diff_margin_high = 0.075
            y_low = min(
                data_rea.min(),
                data_filt.min(),
                y_forecast_obs.min(),
                y_forecast_full.min(),
                y_prior.min(),
            )
            y_high = max(
                data_rea.max(),
                data_filt.max(),
                y_forecast_obs.max(),
                y_forecast_full.max(),
                y_prior.max(),
            )

            if c == "2m_temperature" and metric == "skill":
                y_high = 3.5

            if y_lows[i] is not None:
                y_low = y_lows[i]
                axes[i][col_offset].set_ylim(y_low, y_high * (1 + diff_margin_high))
                axes[i][col_offset + 1].set_ylim(y_low, y_high * (1 + diff_margin_high))
                axes[i][col_offset + 2].set_ylim(y_low, y_high * (1 + diff_margin_high))
            else:
                axes[i][col_offset].set_ylim(
                    y_low * (1 - diff_margin_low), y_high * (1 + diff_margin_high)
                )
                axes[i][col_offset + 1].set_ylim(
                    y_low * (1 - diff_margin_low), y_high * (1 + diff_margin_high)
                )
                axes[i][col_offset + 2].set_ylim(
                    y_low * (1 - diff_margin_low), y_high * (1 + diff_margin_high)
                )

            axes[i][col_offset].set_yticklabels(
                labels=axes[i][col_offset].get_yticklabels(), fontsize=font_size_ticklabels
            )

            axes[i][col_offset + 1].tick_params(labelleft=False)
            axes[i][col_offset + 2].tick_params(labelleft=False)

            axes[i][col_offset + 2].set_xlim(*config.forecasting_xlim)

    font_size_task_titles = config.font_sizes.task_titles
    for i in range(len(metrics)):
        axes[0][i * 3].set_title("Reanalysis", fontsize=font_size_task_titles)
        axes[0][i * 3 + 1].set_title("Filtering", fontsize=font_size_task_titles)
        axes[0][i * 3 + 2].set_title("Forecast", fontsize=font_size_task_titles)

    for i in range(len(channels)):
        channel_name = channel_names[i]
        axes[i][0].set_ylabel(channel_name, fontsize=font_size_ticklabels + 1)
        axes[i][0].yaxis.set_label_coords(-0.225, 0.5)

    show_persistence = "skill" in metrics or "crps" in metrics
    show_baselines = ("skill" in metrics) and (
        "2m_temperature" in channels
        or "10m_u_component_of_wind" in channels
        or "10m_v_component_of_wind" in channels
        or ("temperature" in channels and levels[channels.index("temperature")] == 850)
    )
    handles = [
        Line2D([], [], color=config.colors.aao_obs, label="Assimilation (ours)", linewidth=2.5),
        Line2D(
            [],
            [],
            color=config.colors.aao_obs,
            label="Observational forecast (ours)",
            linewidth=2.5,
        ),
        Line2D(
            [],
            [],
            color=config.colors.ar_full,
            label="Full-state forecast (ours)",
            linewidth=2.5,
        ),
    ]
    if show_baselines:
        handles += [
            Line2D([], [], color=config.colors.ifs, label="IFS", linewidth=2.5),
            Line2D([], [], color=config.colors.graphdop, label="GraphDOP", linewidth=2.5),
        ]

    handles += [
        Line2D([], [], color=config.colors.prior, label="Prior", linestyle="--", linewidth=2.5),
    ]
    if show_persistence:
        handles.append(
            Line2D(
                [],
                [],
                color=config.colors.persistence,
                label="Persistence",
                linestyle="--",
                linewidth=2.5,
            )
        )

    leg1 = fig.legend(
        handles=handles[:3],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.09),
    )
    fig.legend(
        handles=handles[3:],
        loc="upper center",
        ncol=len(handles) - 3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.075),
    )
    fig.add_artist(leg1)

    out_path = Path(config.output_path)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path.absolute()}.")


def main():
    config = compose("configs/assimilation_metrics.yaml", overrides=sys.argv[1:])

    plot_skill_crps(config)


if __name__ == "__main__":
    main()
