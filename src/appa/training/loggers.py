r"""Experiment tracker factories (SwanLab / Weights & Biases).

The active tracker is selected with the ``tracker`` section of the training
configuration::

    tracker:
        name: swanlab   # swanlab | wandb | none
        entity: <wandb entity or swanlab workspace>

Passing ``name: none`` (or omitting the section) disables tracking entirely,
which is handy for local debugging (together with ``SWANLAB_MODE=disabled``
or ``WANDB_MODE=disabled``).
"""

__all__ = [
    "build_logger",
    "log_images",
    "logger_image",
]

from matplotlib.figure import Figure
from omegaconf import DictConfig
from pathlib import Path
from typing import Any


def build_logger(
    cfg: DictConfig,
    runid: str,
    project: str,
    run_name: str,
    description: str,
    config: dict,
    save_dir: str | Path | None = None,
):
    r"""Builds the experiment logger selected by :py:`cfg.tracker.name`.

    Arguments:
        cfg: The training configuration (holding an optional ``tracker`` section).
        runid: The run id, used for resuming an existing experiment.
        project: The tracker project name (e.g. ``appa_ae``).
        run_name: The human readable run name.
        description: A short description of the run (notes).
        config: The full configuration dictionary to record.
        save_dir: Directory where tracker files are stored.

    Returns:
        A Lightning logger instance, or :py:`None` if tracking is disabled.
    """

    tracker = cfg.get("tracker", None)
    name = str(tracker.get("name", "none")) if tracker is not None else "none"
    entity = str(tracker.get("entity", None)) if tracker is not None else None

    if name == "none":
        return None

    if name == "swanlab":
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        return SwanLabLogger(
            project=project,
            workspace=entity,
            experiment_name=run_name,
            description=description,
            config=config,
            id=runid,
        )

    if name == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(
            name=run_name,
            project=project,
            entity=entity,
            id=runid,
            group=f"train_{runid}",
            resume="allow",
            notes=description,
            config=config,
            save_dir=str(save_dir) if save_dir is not None else None,
        )

    raise ValueError(f"Unknown tracker {name!r}. Expected 'swanlab', 'wandb', or 'none'.")


def logger_backend(logger: Any) -> str | None:
    r"""Returns the backend name of a logger ('swanlab', 'wandb'), if any."""

    if logger is None:
        return None

    return type(logger).__name__.lower().replace("logger", "")


def logger_image(logger: Any, fig: Figure):
    r"""Wraps a matplotlib figure into the image type of the logger backend."""

    backend = logger_backend(logger)

    if backend == "swanlab":
        import swanlab

        return swanlab.Image(fig)

    if backend == "wandb":
        import wandb

        return wandb.Image(fig)

    return None


def log_images(logger: Any, metrics: dict) -> None:
    r"""Logs a dict of (image) metrics to the tracker experiment.

    Images must be wrapped with :py:`func:`logger_image`` first. Metrics are
    logged outside of the Lightning metric aggregation, which only supports
    scalars.
    """

    if logger is None or not metrics:
        return

    experiment = logger.experiment
    experiment.log(metrics)
