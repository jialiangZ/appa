r"""Shared LightningModule base for appa training.

The base implements the training loop semantics of the pre-Lightning scripts:

- manual optimization with gradient accumulation (DDP sync disabled while
  accumulating, mirroring the original ``no_sync`` calls);
- ``safe_gd_step``-like behavior: optimizer updates are skipped when the
  gradients contain NaN/Inf values. With ``precision=16-mixed`` this is
  handled by the precision plugin's GradScaler; otherwise non-finite
  gradients are detached before stepping;
- the learning-rate scheduler is stepped once per optimizer update;
- train statistics (loss/grad-norm mean and std) are aggregated over a window
  of ``train.log_interval`` samples, like the original scripts;
- validation losses are aggregated as ``valid/losses/mean`` and
  ``valid/losses/std``.
"""

__all__ = [
    "AppaTrainingModule",
]

import time
import torch

from contextlib import nullcontext
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
from torch import Tensor

from .optim import grad_norm


class AppaTrainingModule(LightningModule):
    r"""Base class for the autoencoder and denoiser training modules.

    Subclasses must implement :py:meth:`compute_loss` (training),
    :py:meth:`compute_valid_loss` (validation) and
    :py:meth:`configure_optimizers`.

    Arguments:
        cfg: The training configuration.
        runid: The run id.
    """

    def __init__(self, cfg: DictConfig, runid: str = ""):
        super().__init__()

        self.cfg = cfg
        self.runid = runid
        self.automatic_optimization = False
        self.amp_dtype = getattr(torch, cfg.train.precision)

        self._losses = []
        self._grad_norms = []
        self._valid_losses = []
        self._steps_since_log = 0
        self._acc_start = None

    #
    # Optimization
    #

    @property
    def grad_acc(self) -> int:
        r"""The number of batches per optimizer step on this rank."""

        per_gpu = self.cfg.train.batch_element_per_gpu * max(self.trainer.world_size, 1)

        return self.cfg.train.batch_size_per_step // per_gpu

    def on_fit_start(self) -> None:
        bsp = self.cfg.train.batch_size_per_step
        per_gpu = self.cfg.train.batch_element_per_gpu * self.trainer.world_size

        assert bsp % per_gpu == 0, (
            "Batch size must be divisible by number of GPU * samples per GPU"
        )

    def compute_loss(self, batch) -> Tensor:
        raise NotImplementedError()

    def compute_valid_loss(self, batch) -> Tensor:
        return self.compute_loss(batch)

    def training_step(self, batch, batch_idx) -> Tensor:
        opt = self.optimizers()

        if batch_idx % self.grad_acc == 0:
            self._acc_start = time.time()

        is_last = (batch_idx + 1) % self.grad_acc == 0

        # Do not synchronize gradients across ranks while accumulating.
        ctx = nullcontext() if is_last else opt.toggle_model(sync_grad=False)

        with ctx:
            loss = self.compute_loss(batch)
            self.manual_backward(loss / self.grad_acc)

        self._losses.append(loss.detach())

        if is_last:
            dt = time.time() - self._acc_start
            self._take_optimizer_step(opt)
            self._log_train_window(dt)

        return loss.detach()

    def _take_optimizer_step(self, opt) -> None:
        r"""Steps the optimizer, skipping updates with non-finite gradients.

        With a mixed-precision plugin scaler (``16-mixed``), scaling,
        unscaling, NaN-skipping and scaler updating all happen inside
        ``opt.step()``. Without a scaler, non-finite gradients are detached
        (``set_to_none=True``) so that the underlying optimizer skips every
        parameter, leaving the model unchanged.
        """

        scaler = getattr(self.trainer.precision_plugin, "scaler", None)

        if scaler is None:
            raw = self.trainer.strategy.optimizers[0]
            norm = grad_norm(raw)

            if norm is not None:
                self._grad_norms.append(norm.detach())

                if not torch.isfinite(norm):
                    raw.zero_grad(set_to_none=True)

        opt.step()

        scheduler = self.lr_schedulers()

        if scheduler is not None:
            scheduler.step()

        # Match the original safe_gd_step: always clear gradients after a step.
        self.trainer.strategy.optimizers[0].zero_grad()

    def on_before_optimizer_step(self, optimizer) -> None:
        r"""Records the (unclipped, unscaled) gradient norm."""

        scaler = getattr(self.trainer.precision_plugin, "scaler", None)

        if scaler is not None:
            norm = grad_norm(optimizer)

            if norm is not None:
                self._grad_norms.append(norm.detach())

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        r"""Clips gradients with ``optim.grad_clip``, as the original scripts.

        Lightning does not apply ``Trainer(gradient_clip_val=...)`` in manual
        optimization, so clipping is driven by the configuration instead.
        """

        clip_val = self.cfg.optim.get("grad_clip", None) if self.cfg.get("optim", None) else None

        if clip_val:
            self.clip_gradients(optimizer, clip_val=clip_val)

    #
    # Logging
    #

    def _log_train_window(self, dt: float) -> None:
        r"""Logs train statistics every ``train.log_interval`` samples."""

        log_every = max(self.cfg.train.log_interval // self.cfg.train.batch_size_per_step, 1)

        self._steps_since_log += 1

        if self._steps_since_log < log_every:
            return

        self._steps_since_log = 0

        step = self.trainer.global_step
        losses = torch.stack(self._losses).float()

        log_kwargs = {
            "sync_dist": True,
            "reduce_fx": "mean",
            "on_step": True,
            "on_epoch": False,
        }

        self.log("train/losses/mean", losses.mean(), **log_kwargs)
        self.log("train/losses/std", losses.std(unbiased=False), **log_kwargs)
        self.log("loss", losses.mean(), prog_bar=True, **log_kwargs)

        if len(self._grad_norms) > 0:
            norms = torch.stack(self._grad_norms).float()
            self.log("train/grad_norm/mean", norms.mean(), **log_kwargs)
            self.log("train/grad_norm/std", norms.std(unbiased=False), **log_kwargs)

        lr = self.trainer.strategy.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, **log_kwargs)
        self.log("train/update_step_time", dt, **log_kwargs)
        self.log("train/samples_seen", self.cfg.train.batch_size_per_step * step, **log_kwargs)
        self.log("train/update_steps_done", step, **log_kwargs)

        self._losses = []
        self._grad_norms = []

    #
    # Validation
    #

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Tensor:
        loss = self.compute_valid_loss(batch)
        # Keep on-device: sync_dist logging performs a NCCL all_reduce on the
        # logged values, which does not support CPU tensors.
        self._valid_losses.append(loss.detach())

        return loss

    def on_validation_epoch_end(self) -> None:
        if len(self._valid_losses) == 0:
            return

        losses = torch.stack(self._valid_losses)

        self.log("valid/losses/mean", losses.mean(), sync_dist=True, reduce_fx="mean")

        if len(losses) > 1:
            self.log(
                "valid/losses/std", losses.std(unbiased=False), sync_dist=True, reduce_fx="mean"
            )

        self._valid_losses = []
        self._valid_extras()

    def _valid_extras(self) -> None:
        r"""Hook for subclass-specific validation metrics."""

    #
    # Misc
    #

    def log_snapshots(self, logger) -> None:
        r"""Logs qualitative validation snapshots (implemented by subclasses)."""
