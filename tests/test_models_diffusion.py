r"""Tests for the appa.models.diffusion module."""

import pytest
import torch
import torch.nn as nn

from torch import Tensor
from typing import Any

from appa.models.diffusion import Denoiser, DenoiserLoss, LogLinearSchedule


class DummyBackbone(nn.Module):
    def __init__(self, in_sz: int):
        super().__init__()

        self.l1 = nn.Linear(in_sz, in_sz)
        self.l2 = nn.Linear(1, in_sz)

    def forward(self, x: Tensor, t: Tensor, context: Any = None):
        t = t.unsqueeze(-1)

        return self.l1(x) + self.l2(t)


@pytest.mark.parametrize("features", [5])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("context", [True, False])
def test_Denoiser(features: int, batch_size: int, context: bool):
    net = DummyBackbone(features)

    d = Denoiser(net)
    l = DenoiserLoss(d)
    s = LogLinearSchedule()

    x = torch.randn(batch_size, features)
    t = torch.rand(batch_size)

    if context:
        loss = l(x, s(t), context=torch.rand(1))
    else:
        loss = l(x, s(t))

    assert loss.requires_grad
