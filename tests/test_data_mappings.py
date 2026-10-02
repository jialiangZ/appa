r"""Tests for the appa.data.mappings module."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from appa.data.const import ERA5_PRESSURE_LEVELS, SUB_PRESSURE_LEVELS
from appa.data.mappings import (
    extract_level_indices,
    tensor_to_xarray,
)


@pytest.fixture
def fake_era5_dataset(tmp_path):
    time = pd.date_range("1999-01-01", periods=12, freq="h")
    latitude = np.array([-10.0, -10.25], dtype=np.float32)
    longitude = np.array([300.0, 300.2], dtype=np.float32)
    level = np.array(ERA5_PRESSURE_LEVELS, dtype=np.int64)

    temp_2m = np.ones((len(time), len(latitude), len(longitude))) * 1
    geopotential = np.ones((len(time), len(level), len(latitude), len(longitude))) * 2
    temperature = np.ones((len(time), len(level), len(latitude), len(longitude))) * 3

    ds = xr.Dataset(
        {
            "2m_temperature": (["time", "latitude", "longitude"], temp_2m),
            "geopotential": (["time", "level", "latitude", "longitude"], geopotential),
            "temperature": (["time", "level", "latitude", "longitude"], temperature),
        },
        coords={
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
            "level": level,
        },
    )

    zarr_path = tmp_path / "fake_era5.zarr"
    ds.to_zarr(zarr_path, mode="w")
    return zarr_path


def expected_mappings_for(sublevels):
    r"""Builds the expected variable->(start, end) mapping for a level count.

    Deriving the expectations from the constant keeps the test correct for any
    level configuration (e.g. the 13-level low-res repro or the 37-level setup).
    """
    n = len(sublevels)
    return [
        {"2m_temperature": (0, 1), "geopotential": (1, 1 + n), "temperature": (1 + n, 1 + 2 * n)},
        {"geopotential": (0, n), "temperature": (n, 2 * n)},
        {"geopotential": (0, n), "2m_temperature": (n, n + 1), "temperature": (n + 1, 2 * n + 1)},
    ]


@pytest.mark.parametrize(
    "sublevels",
    [
        ERA5_PRESSURE_LEVELS,
        SUB_PRESSURE_LEVELS,
    ],
)
def test_extract_level_indices(fake_era5_dataset, sublevels):
    expected_mappings = expected_mappings_for(sublevels)
    mappings = [
        extract_level_indices(path=fake_era5_dataset, variables=vars, sub_levels=sublevels)
        for vars in [
            ["2m_temperature", "geopotential", "temperature"],
            ["geopotential", "temperature"],
            ["geopotential", "2m_temperature", "temperature"],
        ]
    ]

    for i, expected in enumerate(expected_mappings):
        assert mappings[i] == expected, f"Mismatch for sublevels={sublevels}, case {i}"


def test_tensor_to_xarray(fake_era5_dataset):
    dataset = (
        xr
        .open_zarr(fake_era5_dataset)
        .to_stacked_array(new_dim="z_total", sample_dims=("time", "longitude", "latitude"))
        .transpose("time", "z_total", ...)
    )
    dataset = torch.as_tensor(dataset.load().data)

    xarray = tensor_to_xarray(
        dataset,
        variables=["2m_temperature", "geopotential", "temperature"],
        path=fake_era5_dataset,
        sub_levels=ERA5_PRESSURE_LEVELS,
    )

    v1 = torch.from_numpy(xarray["2m_temperature"].data)
    v2 = torch.from_numpy(xarray["geopotential"].data)
    v3 = torch.from_numpy(xarray["temperature"].data)
    assert torch.allclose(v1, torch.ones_like(v1))
    assert torch.allclose(v2, torch.ones_like(v2) * 2)
    assert torch.allclose(v3, torch.ones_like(v3) * 3)
