r"""Tests for the optimizers."""

import pytest
import torch

from omegaconf import OmegaConf

from appa.training.optim import get_optimizer


@pytest.mark.parametrize(
    "optim_name",
    [
        "adamw",
        "soap",
        "psgd",
    ],
)
def test_optims(optim_name):
    dummy_nn = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.Linear(10, 10),
    )

    optim_cfg_path = f"src/experiments/autoencoder/configs/optim/{optim_name}.yaml"

    cfg = OmegaConf.load(optim_cfg_path)

    assert cfg.optimizer == optim_name

    get_optimizer(
        params=dummy_nn.parameters(),
        update_steps=1000,
        **cfg,
    )
