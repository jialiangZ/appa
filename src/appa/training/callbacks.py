r"""Lightning callbacks shared by the training scripts.

- :class:`KillSwitchCallback` gracefully stops a run when the ``.running``
  marker file is removed (same protocol as the pre-Lightning scripts).
- :class:`ArtifactCheckpointCallback` saves full Lightning checkpoints for
  resuming, and additionally exports the artifact layout (``model_last.pth``,
  ``model_best.pth``, ``metadata.yaml``) expected by the downstream scripts
  (latent dumps, generation, evaluation, ...).
- :class:`SnapshotLoggingCallback` asks the LightningModule to log its
  qualitative validation snapshots on global rank zero.
"""

__all__ = [
    "ArtifactCheckpointCallback",
    "KillSwitchCallback",
    "SnapshotLoggingCallback",
]

import torch

from collections.abc import Callable
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from pathlib import Path

from appa.utils.io import safe_save


class KillSwitchCallback(Callback):
    r"""Stops training gracefully when the ``.running`` marker is removed.

    Every rank polls the same (shared) filesystem path, so no collective is
    needed. Stopping is graceful: the last checkpoint can be resumed with
    ``ckpt_path=.../lightning/last.ckpt``.
    """

    def __init__(self, runpath: str | Path):
        super().__init__()

        self.marker = Path(runpath) / ".running"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not self.marker.is_file():
            trainer.should_stop = True


class ArtifactCheckpointCallback(Callback):
    r"""Saves checkpoints and exports the legacy artifact layout.

    On a sample-count cadence (converted to a number of training batches), the
    callback saves

    - ``<runpath>/lightning/last.ckpt``: a full Lightning checkpoint (model,
      optimizer, scheduler and step counters) used to resume training;
    - ``<runpath>/model_last.pth``: the raw model state dict, for downstream
      scripts;
    - ``<runpath>/metadata.yaml``: run metadata (last step, best validation).

    Whenever the monitored validation metric improves, the same is done for
    the "best" variants.

    Arguments:
        runpath: The artifact directory of the run (e.g. ``artifacts/autoencoders/<runid>/0``).
        runid: The run id.
        get_model: A callable returning the exported module from the
            LightningModule (e.g. ``lambda m: m.ae``).
        every_n_train_batches: Saving cadence, in training batches.
        monitor: The validation metric monitored for the "best" export.
    """

    def __init__(
        self,
        runpath: str | Path,
        runid: str,
        get_model: Callable,
        every_n_train_batches: int,
        monitor: str = "valid/losses/mean",
        best_val_loss: float = float("inf"),
    ):
        super().__init__()

        self.runpath = Path(runpath)
        self.runid = runid
        self.get_model = get_model
        self.every_n_train_batches = max(int(every_n_train_batches), 1)
        self.monitor = monitor
        self.best_val_loss = best_val_loss

        self.ckpt_dir = self.runpath / "lightning"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (batch_idx + 1) % self.every_n_train_batches != 0:
            return

        # save_checkpoint must be called on ALL ranks: distributed strategies
        # synchronize with a barrier/collective internally and only the global
        # zero rank writes the file.
        trainer.save_checkpoint(self.ckpt_dir / "last.ckpt")

        if trainer.is_global_zero:
            safe_save(self.get_model(pl_module).state_dict(), self.runpath / "model_last.pth")
            self._write_metadata(last_step_done=trainer.global_step - 1)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        # The monitored metric is logged with sync_dist=True, so every rank
        # reaches the same decision below and the collective stays balanced.
        score = trainer.callback_metrics.get(self.monitor, None)

        if score is None:
            return

        score = float(score)

        if score < self.best_val_loss:
            self.best_val_loss = score

            trainer.save_checkpoint(self.ckpt_dir / "best.ckpt")

            if trainer.is_global_zero:
                safe_save(self.get_model(pl_module).state_dict(), self.runpath / "model_best.pth")
                self._write_metadata(last_step_done=trainer.global_step - 1)

    def _write_metadata(self, last_step_done: int) -> None:
        with open(self.runpath / "metadata.yaml", "w") as f:
            OmegaConf.save(
                {
                    "last_step_done": last_step_done,
                    "best_val_loss": self.best_val_loss,
                    "lap": 0,
                    "runid": self.runid,
                },
                f,
            )


class SnapshotLoggingCallback(Callback):
    r"""Logs qualitative validation snapshots on global rank zero.

    The plotting logic lives on the LightningModule
    (:py:meth:`log_snapshots`), which knows how to interpret its own inputs;
    this callback only triggers it at the right time.
    """

    def __init__(self, logger):
        super().__init__()

        self.logger = logger

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        if self.logger is None:
            return

        with torch.no_grad():
            pl_module.log_snapshots(self.logger)
