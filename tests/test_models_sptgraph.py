r"""Tests for the appa.models.diffusion module with SpatioTemporalGraphDiT."""

import pytest
import torch
import torch.nn as nn

from functools import partial

from appa.models.diffusion import Denoiser, DenoiserLoss, LogLinearSchedule, TrajectoryDenoiser
from appa.models.nn.sptgraph import SpatioTemporalGraphDiT, create_spatio_temporal_edges
from appa.models.sampling import DDIMSampler, DDPMSampler


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("T", [10])
@pytest.mark.parametrize("channels", [8])
@pytest.mark.parametrize("full_attention", [True, False])
def test_Denoiser(batch_size: int, N: int, T: int, channels: int, full_attention: bool):
    # Create spatial edges (e.g., connect nodes in a ring)
    if full_attention:
        spatial_edges = None
    else:
        spatial_edges = torch.tensor([[i, (i + 1) % N] for i in range(N)], dtype=torch.long)

    grid = torch.randn(N, 2)
    backbone = SpatioTemporalGraphDiT(
        input_grid=grid,
        input_channels=channels,
        hidden_channels=channels // 2,
        hidden_blocks=2,
        self_edges=spatial_edges,
        cross_edges=spatial_edges,
    )

    denoiser = Denoiser(backbone)
    loss_fn = DenoiserLoss(denoiser)
    noise_schedule = LogLinearSchedule()

    x = torch.randn(batch_size, T * N * channels)
    # random date, at most 12 for month
    date = torch.randint(1, 13, size=(batch_size, T, 4))

    t = torch.rand(batch_size)
    sigma_t = noise_schedule(t)

    loss = loss_fn(x, sigma_t, date=date)

    assert loss.requires_grad

    loss.backward()

    for name, p in denoiser.named_parameters():
        if p.grad is not None:
            assert p.grad.norm().item() > 0, f"No gradient for parameter {name}"

    print(f"Loss: {loss.item()}")


@pytest.mark.parametrize("sampler_class", [DDIMSampler, DDPMSampler])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("T", [12])
@pytest.mark.parametrize("channels", [8])
@pytest.mark.parametrize("trajectory", [True, False])
def test_samplers(
    sampler_class: nn.Module, batch_size: int, N: int, T: int, channels: int, trajectory: bool
):
    # Create spatial edges (e.g., connect nodes in a ring)
    spatial_edges = torch.tensor([[i, (i + 1) % N] for i in range(N)], dtype=torch.long)

    grid = torch.randn(N, 2)
    backbone = SpatioTemporalGraphDiT(
        input_grid=grid,
        input_channels=channels,
        hidden_channels=channels // 2,
        hidden_blocks=2,
        self_edges=spatial_edges,
        cross_edges=spatial_edges,
    )
    if trajectory:
        backbone = TrajectoryDenoiser(
            backbone, blanket_size=7, blanket_stride=5, state_size=N * channels
        )

    # random date, at most 12 for month
    date = torch.randint(1, 13, size=(batch_size, T, 4))
    denoiser = Denoiser(backbone)
    denoiser = partial(denoiser, date=date)
    sampler = sampler_class(denoiser, schedule=LogLinearSchedule())

    x_noisy = torch.randn(batch_size, T * N * channels)

    x_sampled = sampler(x_noisy)

    assert x_sampled.shape == x_noisy.shape


@pytest.mark.parametrize("trajectory_length", [5, 8])
@pytest.mark.parametrize("blanket_size", [5])
@pytest.mark.parametrize("stride", [1, 3])
def test_trajectory_wrapper(trajectory_length: int, blanket_size: int, stride: int):
    N = 12
    channels = 8
    batch_size = 2
    spatial_edges = torch.tensor([[i, (i + 1) % N] for i in range(N)], dtype=torch.long)

    grid = torch.randn(N, 2)
    backbone = SpatioTemporalGraphDiT(
        input_grid=grid,
        input_channels=channels,
        hidden_channels=channels // 2,
        hidden_blocks=2,
        self_edges=spatial_edges,
        cross_edges=spatial_edges,
    )
    backbone = TrajectoryDenoiser(
        backbone, blanket_size=blanket_size, blanket_stride=stride, state_size=N * channels
    )

    date = torch.randint(1, 13, size=(batch_size, trajectory_length, 4))

    x = torch.randn(batch_size, trajectory_length * N * channels)

    x_ = backbone(x, torch.rand(1), date)

    assert x_.shape == x.shape


@pytest.mark.parametrize(
    "num_nodes, blanket_size, self_edges, cross_edges, expected_output",
    [
        (
            3,
            3,
            torch.tensor([[0, 0], [1, 1]]),
            torch.empty((0, 2)),
            torch.tensor([[0, 0], [1, 1], [3, 3], [4, 4], [6, 6], [7, 7]]),
        ),
        (
            3,
            2,
            torch.tensor(
                [[0, 1], [1, 2], [2, 0]],
            ),
            torch.tensor([[0, 0], [2, 2]]),
            torch.tensor([
                [0, 1],
                [1, 2],
                [2, 0],
                [3, 4],
                [4, 5],
                [5, 3],
                [0, 3],
                [2, 5],
                [3, 0],
                [5, 2],
            ]),
        ),
    ],
)
def test_spatio_temporal_edges(num_nodes, blanket_size, self_edges, cross_edges, expected_output):
    output = create_spatio_temporal_edges(num_nodes, self_edges, cross_edges, blanket_size)
    assert torch.equal(output, expected_output), f"""
    Failed for blanket_size={blanket_size}, num_node={num_nodes}.
    Expected:
    {expected_output}
    Got:
    {output}
    """
