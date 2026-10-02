r"""Tests for the appa.data.dataset module."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from torch.utils.data import DataLoader

from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform


@pytest.fixture
def fake_era5_dataset(tmp_path):
    time = pd.date_range("1999-01-01", periods=18, freq="h")
    latitude = np.array([-10.0, -10.25], dtype=np.float32)
    longitude = np.array([300.0, 300.2], dtype=np.float32)
    level = np.array([1, 2], dtype=np.int64)

    temp_2m = 295 + 2 * np.random.randn(len(time), len(latitude), len(longitude))
    geopotential = 4.67e5 + 1e4 * np.random.randn(
        len(time), len(level), len(latitude), len(longitude)
    )
    temperature = 260 + 5 * np.random.randn(len(time), len(level), len(latitude), len(longitude))

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


@pytest.fixture
def fake_statistics_dataset(fake_era5_dataset, tmp_path):
    dataset = xr.open_zarr(fake_era5_dataset)

    dataset_mean = dataset.mean(dim=["time", "latitude", "longitude"])
    dataset_std = dataset.std(dim=["time", "latitude", "longitude"])

    dataset_mean = dataset_mean.expand_dims(statistic=["mean"])
    dataset_std = dataset_std.expand_dims(statistic=["std"])
    dataset_mean = dataset_mean.expand_dims(surface=[0])
    dataset_std = dataset_std.expand_dims(surface=[0])

    dataset_complete = xr.concat([dataset_mean, dataset_std], dim="statistic")

    output_path = tmp_path / "fake_statistics.zarr"
    dataset_complete.to_zarr(output_path, mode="w")

    return output_path


@pytest.mark.parametrize(
    "trajectory_size, batch_size, state_variables, context_variables, levels, expected_shapes",
    [
        (1, 2, None, None, None, [5, 0]),
        (1, 2, ["2m_temperature", "geopotential"], None, [1], [2, 0]),
        (1, 2, ["2m_temperature", "temperature"], ["geopotential"], [1, 2], [3, 2]),
        (5, 4, None, None, None, [5, 0]),
        (5, 4, ["2m_temperature", "geopotential"], None, [2], [2, 0]),
        (5, 4, ["2m_temperature", "geopotential"], ["temperature"], [1, 2], [3, 2]),
    ],
)
def test_era5_dataset(
    fake_era5_dataset,
    trajectory_size,
    batch_size,
    state_variables,
    context_variables,
    levels,
    expected_shapes,
):
    # Valid
    ds = ERA5Dataset(
        path=fake_era5_dataset,
        start_date="1999-01-01",
        end_date="1999-01-01",
        state_variables=state_variables,
        context_variables=context_variables,
        levels=levels,
        trajectory_dt=2,
        trajectory_size=trajectory_size,
    )

    print("BATCH SIZE", batch_size)

    print("A", ds[0][0].shape)
    print("B", ds[1][0].shape)
    print("C", ds[2][0].shape)
    print("D", ds[3][0].shape)

    samples, context, time = next(iter(DataLoader(ds, batch_size=batch_size)))

    # Expected numbers of variables
    num_state_vars, num_context_vars = expected_shapes

    print(num_state_vars, num_context_vars)
    print("Samples shape:", samples.shape)

    assert len(ds) == ds.dataset.time.size - trajectory_size + 1
    assert samples.shape == (batch_size, trajectory_size, num_state_vars, 2, 2)
    assert context.shape == (batch_size, trajectory_size, num_context_vars, 2, 2)
    assert time.shape == (batch_size, trajectory_size, 4)

    # Invalid
    with pytest.raises(ValueError):
        ERA5Dataset(
            path=fake_era5_dataset,
            start_date="1999-01-01",
            end_date="1999-01-32",
        )


def test_era5_dataset_cache_equivalence(fake_era5_dataset, fake_statistics_dataset):
    r"""RAM-cached and lazy datasets must produce identical samples."""

    tf_std = StandardizeTransform(
        path=fake_statistics_dataset,
        state_variables=["2m_temperature", "temperature"],
        context_variables=["geopotential"],
        levels=[1, 2],
    )

    common = dict(
        path=fake_era5_dataset,
        start_date="1999-01-01",
        end_date="1999-01-01",
        trajectory_size=3,
        trajectory_dt=2,
        state_variables=["2m_temperature", "temperature"],
        context_variables=["geopotential"],
        levels=[1, 2],
        transform=tf_std,
    )

    ds_lazy = ERA5Dataset(cache="none", **common)
    ds_ram = ERA5Dataset(cache="ram", **common)

    assert ds_lazy.cache == "none"
    assert ds_ram.cache == "ram"

    assert len(ds_lazy) == len(ds_ram)

    for idx in [0, 1, 5]:
        state_l, context_l, time_l = ds_lazy[idx]
        state_r, context_r, time_r = ds_ram[idx]

        assert state_l.shape == state_r.shape
        assert context_l.shape == context_r.shape
        assert time_l.shape == time_r.shape

        assert torch.allclose(state_l, state_r, equal_nan=True)
        assert torch.allclose(context_l, context_r, equal_nan=True)
        assert torch.equal(time_l, time_r)

    # "auto" resolves to "ram" for a small dataset
    ds_auto = ERA5Dataset(cache="auto", **common)
    assert ds_auto.cache == "ram"

    # Mutating a cached sample must not corrupt the cache
    state_before, _, _ = ds_ram[0]
    state_before.fill_(0.0)
    state_after, _, _ = ds_ram[0]
    assert not torch.allclose(state_before, state_after)


def test_era5_dataset_incomplete_trajectory_raises(fake_era5_dataset):
    r"""Both cache modes reject incomplete trajectories identically."""

    common = dict(
        path=fake_era5_dataset,
        start_date="1999-01-01",
        end_date="1999-01-01",
        trajectory_size=5,
        trajectory_dt=4,
    )

    for cache in ["none", "ram"]:
        ds = ERA5Dataset(cache=cache, **common)
        with pytest.raises(ValueError):
            _ = ds[len(ds) - 1]


def test_StandardizeTransform(fake_era5_dataset, fake_statistics_dataset):
    tf_std = StandardizeTransform(path=fake_statistics_dataset)

    dataset = ERA5Dataset(
        path=fake_era5_dataset,
        start_date="1999-01-01",
        end_date="1999-01-01",
        trajectory_size=1,
        transform=None,
    )

    dataset_tf = ERA5Dataset(
        path=fake_era5_dataset,
        start_date="1999-01-01",
        end_date="1999-01-01",
        trajectory_size=1,
        transform=tf_std,
    )

    samples, context, time = next(iter(DataLoader(dataset, batch_size=12)))
    samples_tf, context_tf, time_tf = next(iter(DataLoader(dataset_tf, batch_size=12)))

    print("Samples shape:", samples.shape)
    print("Transformed samples shape:", samples_tf.shape)

    assert not torch.equal(samples, samples_tf), (
        "Transformed sample should differ from the original."
    )
    assert torch.equal(time, time_tf), (
        "Time features should remain unchanged after transformation."
    )

    mean = torch.mean(samples_tf, dim=(0, 2, 3))
    std = torch.std(samples_tf, dim=(0, 2, 3))

    assert torch.all((mean >= -1) & (mean <= 1)), (
        "Mean values should be within [-1, 1] with high probability."
    )
    assert torch.all((std >= 0) & (std <= 2)), (
        "Std values should be within [0, 2] with high probability."
    )
