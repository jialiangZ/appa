r"""Tests for the Lightning training components.

These tests are CPU-only and self-contained: they use a tiny synthetic
LightningModule/Dataset, so they do not require ERA5 data or GPUs.
"""

import pytest
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from appa.training.callbacks import (
    ArtifactCheckpointCallback,
    KillSwitchCallback,
    SnapshotLoggingCallback,
)
from appa.training.entry import (
    PRECISION_MAP,
    new_runid,
    samples_to_batches,
    val_check_fraction,
)
from appa.training.loggers import build_logger
from appa.training.modules import AppaTrainingModule
from appa.training.optim import grad_norm, has_finite_grads


class RandDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return torch.randn(4, 8)


class TinyModule(AppaTrainingModule):
    r"""A minimal module exercising the shared manual-optimization logic."""

    def __init__(self, cfg):
        super().__init__(cfg)

        self.lin = torch.nn.Linear(8, 8)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.lin.parameters(), lr=1e-3)

    def compute_loss(self, batch):
        return self.lin(batch).square().mean()


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "train": {
            "update_steps": 4,
            "batch_element_per_gpu": 2,
            "batch_size_per_step": 4,
            "precision": "float32",
            "log_interval": 4,
            "save_interval": 4,
        },
        "tracker": {"name": "none", "entity": None},
    })


def _trainer(**kwargs):
    from lightning.pytorch import Trainer

    defaults = {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "num_sanity_val_steps": 0,
    }
    defaults.update(kwargs)

    return Trainer(**defaults)


def test_entry_helpers():
    assert PRECISION_MAP["float32"] == "32-true"
    assert PRECISION_MAP["float16"] == "16-mixed"
    assert len(new_runid()) == 8

    # 8 samples seen per update step, 2 batches per update step
    assert samples_to_batches(16, 8, 2) == 4

    # Interval larger than an epoch clamps to 1.0
    assert val_check_fraction(100, 10) == 1.0
    assert val_check_fraction(5, 10) == 0.5


def test_build_logger_disabled(cfg):
    assert build_logger(cfg, "runid", "proj", "name", "desc", {}) is None


def test_grad_norm_utils():
    lin = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)

    assert grad_norm(opt) is None
    assert has_finite_grads(opt)

    lin(torch.randn(2, 4)).sum().backward()

    norm = grad_norm(opt)
    assert norm is not None and torch.isfinite(norm)

    # Non-finite gradients are detected
    lin.weight.grad[0, 0] = float("nan")
    assert not has_finite_grads(opt)


def test_manual_optimization_counts_and_zeroes(cfg):
    r"""One optimizer step per grad_acc batches; gradients cleared in between."""

    module = TinyModule(cfg)
    loader = DataLoader(RandDataset(), batch_size=None)
    trainer = _trainer(limit_train_batches=4, callbacks=[])

    trainer.fit(module, train_dataloaders=loader)

    # batch_size_per_step=4, batch_element_per_gpu=2 -> 2 batches per update
    assert trainer.global_step == 2

    for p in module.lin.parameters():
        assert p.grad is None


def test_nan_gradients_skip_update(cfg):
    r"""A non-finite gradient must leave the parameters unchanged."""

    module = TinyModule(cfg)

    original_step = module._take_optimizer_step

    def poison(opt):
        # Simulate a NaN gradient before stepping
        module.lin.weight.grad = torch.full_like(module.lin.weight, float("nan"))
        original_step(opt)

    module._take_optimizer_step = poison

    before = module.lin.weight.detach().clone()

    loader = DataLoader(RandDataset(), batch_size=None)
    trainer = _trainer(limit_train_batches=2)
    trainer.fit(module, train_dataloaders=loader)

    assert torch.equal(module.lin.weight.detach(), before)


def test_artifact_checkpoint_callback(cfg, tmp_path):
    r"""The callback exports both Lightning ckpts and the legacy layout."""

    from omegaconf import OmegaConf as OC

    module = TinyModule(cfg)
    loader = DataLoader(RandDataset(), batch_size=None)

    runpath = tmp_path / "runid" / "0"
    callback = ArtifactCheckpointCallback(
        runpath=runpath,
        runid="runid",
        get_model=lambda m: m.lin,
        every_n_train_batches=2,
        monitor="valid/losses/mean",
    )

    val_loader = DataLoader(RandDataset(), batch_size=None)
    trainer = _trainer(limit_train_batches=4, limit_val_batches=2, callbacks=[callback])

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=val_loader)

    assert (runpath / "lightning" / "last.ckpt").is_file()
    assert (runpath / "lightning" / "best.ckpt").is_file()
    assert (runpath / "model_last.pth").is_file()
    assert (runpath / "model_best.pth").is_file()

    metadata = OC.load(runpath / "metadata.yaml")
    assert metadata["runid"] == "runid"
    assert metadata["lap"] == 0
    assert metadata["last_step_done"] == trainer.global_step - 1
    assert metadata["best_val_loss"] < float("inf")

    # Exported weights match the module
    state = torch.load(runpath / "model_last.pth", weights_only=True)
    assert set(state.keys()) == set(module.lin.state_dict().keys())


def test_resume_from_lightning_checkpoint(cfg, tmp_path):
    r"""A resumed run continues from the restored step count.

    Note: the checkpoint stores the mid-epoch position, so the resumed run
    must use the same dataloader (the first fit is bounded by ``max_steps``,
    not by a batch limit, to keep the loaders identical).
    """

    runpath = tmp_path / "runid" / "0"
    callback = ArtifactCheckpointCallback(
        runpath=runpath,
        runid="runid",
        get_model=lambda m: m.lin,
        every_n_train_batches=1,
        monitor="valid/losses/mean",
    )

    loader = DataLoader(RandDataset(), batch_size=None)

    module = TinyModule(cfg)
    trainer = _trainer(max_epochs=1, limit_train_batches=4, max_steps=1, callbacks=[callback])
    trainer.fit(module, train_dataloaders=loader)
    steps_after_fit = trainer.global_step
    weights_after_fit = module.lin.weight.detach().clone()

    assert steps_after_fit == 1

    # Resume: one more update (batches 2-3 of the epoch)
    module2 = TinyModule(cfg)
    trainer2 = _trainer(
        max_epochs=1,
        limit_train_batches=4,
        max_steps=2,
        callbacks=[
            ArtifactCheckpointCallback(
                runpath=runpath,
                runid="runid",
                get_model=lambda m: m.lin,
                every_n_train_batches=1,
                monitor="valid/losses/mean",
            )
        ],
    )
    trainer2.fit(
        module2,
        train_dataloaders=loader,
        ckpt_path=runpath / "lightning" / "last.ckpt",
    )

    assert trainer2.global_step == 2
    assert not torch.equal(module2.lin.weight.detach(), weights_after_fit)


def test_kill_switch_callback(cfg, tmp_path):
    r"""Removing the .running marker stops the run gracefully."""

    from lightning.pytorch.callbacks import Callback

    runpath = tmp_path / "run"
    runpath.mkdir(parents=True)
    marker = runpath / ".running"
    marker.touch()

    module = TinyModule(cfg)
    loader = DataLoader(RandDataset(), batch_size=None)

    class RemoveMarker(Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == 1:
                marker.unlink()

    trainer = _trainer(
        limit_train_batches=4,
        callbacks=[KillSwitchCallback(runpath), RemoveMarker()],
    )
    trainer.fit(module, train_dataloaders=loader)

    # Stopped early: fewer than the 2 optimizer updates of 4 batches
    assert trainer.global_step <= 1


def test_snapshot_callback_skips_without_logger():
    r"""Without a tracker, snapshot logging is a no-op (and must not crash)."""

    class Dummy:
        def log_snapshots(self, logger):
            raise AssertionError("should not be called")

    from lightning.pytorch import Trainer

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=False,
    )
    callback = SnapshotLoggingCallback(None)
    # Emulate the guards without running a full fit loop
    trainer.sanity_checking = False
    callback.on_validation_end(trainer, Dummy())
