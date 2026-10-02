r"""Tests for the appa.utils.grids module."""

import pytest
import torch

from numpy import deg2rad

from appa.utils.grids import (
    ORG_vertices_per_lat,
    create_edges,
    create_icosphere,
    create_N320,
    create_ORG,
    latlon_to_xyz,
    xyz_to_latlon,
)


def test_latlon_xyz():
    grid = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    grid = grid / grid.norm(dim=-1, keepdim=True)

    latlon = xyz_to_latlon(grid)
    assert torch.allclose(
        latlon, torch.tensor([[0.0, 0.0], [0.0, 90.0], [90.0, 0.0], [35.264389, 45.0]]), atol=1e-6
    )

    grid2 = latlon_to_xyz(latlon)
    assert torch.allclose(grid, grid2, atol=1e-6)


def test_ORG_vertices_per_latitude():
    with pytest.raises(AssertionError):
        ORG_vertices_per_lat(0)
    assert (
        ORG_vertices_per_lat(1) == 20
    )  # At the pole (latitude_idx = 1), there should be 20 points
    assert ORG_vertices_per_lat(2) == 24  # For latitude_idx = 2, there should be 24 points
    assert ORG_vertices_per_lat(100) == 416  # For latitude_idx = 100, there should be 416 points


def test_create_ORG():
    number = lambda x: 2 * 4 * x * (x + 9)
    result = create_ORG(10)
    expected_points = sum(ORG_vertices_per_lat(i + 1) for i in range(10))
    assert result.shape == (expected_points * 2, 2)
    assert number(10) == result.shape[0] * 2

    result = create_ORG(96)

    expected_points = 40320  # AIFS paper
    assert result.shape == (expected_points, 2)
    assert number(96) == result.shape[0] * 2


def test_create_N320():
    grid = create_N320()

    assert grid.shape[0] == 721 * 1440
    assert grid.shape[1] == 2


def test_create_icosphere():
    grid, _ = create_icosphere(subdivisions=0)
    assert grid.shape == (12, 2)

    grid, _ = create_icosphere(subdivisions=1)
    assert grid.shape == (42, 2)

    grid, _ = create_icosphere(subdivisions=2)
    assert grid.shape == (162, 2)

    grid, _ = create_icosphere(subdivisions=3)
    assert grid.shape == (642, 2)

    grid, _ = create_icosphere(subdivisions=4)
    assert grid.shape == (2562, 2)

    grid, _ = create_icosphere(subdivisions=5)
    assert grid.shape == (10242, 2)

    grid, mm = create_icosphere(subdivisions=6)
    assert grid.shape == (40962, 2)  # as in GraphCast
    assert mm.shape[0] - grid.shape[0] == 327660  # multi-mesh edges reported in GraphCast


def test_edges():
    centers = torch.tensor([[0.0, 0.0]])
    grid = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, -6.0], [10.0, 10.0]])

    edges = create_edges(centers, grid, max_arc=deg2rad(2.0))
    expected_edges = torch.tensor([[0, 0]])

    assert torch.allclose(edges, expected_edges)

    edges = create_edges(centers, grid, max_arc=deg2rad(5.0))
    expected_edges = torch.tensor([[0, 0], [0, 1]])

    assert torch.allclose(edges, expected_edges)

    edges = create_edges(centers, grid, max_arc=deg2rad(7.0))
    expected_edges = torch.tensor([[0, 0], [0, 1], [0, 2]])

    assert torch.allclose(edges, expected_edges)

    edges = create_edges(centers, grid, neighbors=2)

    assert edges.shape[0] == 2 * centers.shape[0]

    centers = torch.tensor([[0.0, 0.0], [1.0, 2.0]])

    edges = create_edges(centers, grid, neighbors=1)

    assert edges.shape[0] == centers.shape[0]
