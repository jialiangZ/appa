r"""Tools to create mappings between data representations."""

import functools
import re
import torch
import xarray as xr

from omegaconf import ListConfig
from pathlib import Path
from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple

from appa.config import PATH_ERA5
from appa.data.const import (
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    SUB_PRESSURE_LEVELS,
)


def get_sublevel_indices(sub_levels: Sequence[int]) -> Sequence[int]:
    """Extract indices of a subset of all available atmospheric levels.

    Arguments:
        sub_levels: List of subset pressure levels in [Pa].

    Returns:
        List of indices.
    """

    return [k for k, l in enumerate(ERA5_PRESSURE_LEVELS) if l in sub_levels]


def extract_level_indices(
    variables: Optional[Sequence[str]] = None,
    path: Optional[Path] = PATH_ERA5,
    sub_levels: Optional[Sequence[int]] = ERA5_PRESSURE_LEVELS,
) -> Dict[str, Tuple[int, int]]:
    r"""Determine the starting and ending indices of variables.

    Illustration:
        Assuming a tensor where each variable is stacked along the 'level'
        dimension (see :class:`ERA5Dataset`), this function determines the
        starting and ending indices of each variable in the stacked dimension.

    Arguments:
        variables: Variable present in the stacked torch tensor of interest.
        path: Path to the Zarr dataset.
        sub_levels: Subset of atmospheric levels to extract.

    Returns:
        The mapping dictionary between each variable (key) and its corresponding level indices (values).
    """

    if isinstance(path, (list, ListConfig)):
        data_ls = []
        for pth in path:
            data_ls.append(xr.open_zarr(pth, chunks={}))
        dataset = xr.concat(data_ls, dim="time")
    else:
        dataset = xr.open_zarr(path)

    if variables is not None:
        dataset = dataset[variables]

    level_count = len(sub_levels)

    idx_start, mapping = 0, {}
    for v in dataset:
        idx_end = idx_start + (level_count if "level" in dataset[v].dims else 1)
        mapping[v] = (idx_start, idx_end)
        idx_start = idx_end
    return mapping


def tensor_to_xarray(
    x: Tensor,
    variables: Optional[Sequence[str]] = None,
    path: Optional[Path] = PATH_ERA5,
    sub_levels: Optional[Sequence[int]] = None,
    roll: bool = True,
) -> xr.Dataset:
    r"""Convert weather data to an xarray dataset.

    Arguments:
        x: Tensor of weather data (*, Z, X, Y).
        variables: Variables present in the data.
        path: Path to the Zarr dataset.
        sub_levels: Subset of atmospheric levels to extract.
        roll: Whether to center Europe on the projected data.
    """
    # Broadcasting input shape to (B, T, Z, X, Y)
    while x.ndim < 5:
        x = x.unsqueeze(dim=0)

    data_arrays = []
    extracted_indices = extract_level_indices(variables, path, sub_levels=sub_levels)

    for v, (idx_start, idx_end) in extracted_indices.items():
        idx_range = torch.arange(idx_start, idx_end)

        is_atm_variable = len(idx_range) > 1

        data = x[:, :, idx_range]

        if roll:
            data = data.roll(shifts=720, dims=-1)

        if is_atm_variable:
            expected_levels = data.shape[2]
            sub_levels = list(range(expected_levels))
            coords = {"level": sub_levels}
            dims = ("batch", "trajectory", "level", "latitude", "longitude")
        else:
            coords = {}
            dims = ("batch", "trajectory", "latitude", "longitude")
            data = data.squeeze(dim=2)  # Remove level dimension

        # Ensure data has the right shape
        data_array = xr.DataArray(
            data=data,
            dims=dims,
            name=v,
            coords=coords,
        )

        data_arrays.append(data_array)

    return xr.merge(data_arrays)


@functools.lru_cache(maxsize=None)
def feature_idx_to_name(
    feature_idx: int,
    sublevels: bool = True,
    prettify: bool = False,
) -> str:
    r"""Returns the variable name given its feature index.

    Arguments:
        feature_idx: The feature index.
        sublevels: Whether to use sub pressure levels.
        prettify: Whether to return a human-readable name ("_" -> " " + Upper case).

    Returns:
        The variable name.
    """

    levels = SUB_PRESSURE_LEVELS if sublevels else ERA5_PRESSURE_LEVELS
    num_levels = len(levels)
    if feature_idx < len(ERA5_SURFACE_VARIABLES):
        feature_name = ERA5_SURFACE_VARIABLES[feature_idx]
        level = 0
    else:
        feature_idx -= len(ERA5_SURFACE_VARIABLES)
        feature_name = ERA5_ATMOSPHERIC_VARIABLES[feature_idx // num_levels]
        level = levels[feature_idx % num_levels]

    if prettify:
        feature_name = feature_name.replace("_", " ")
        feature_name = re.sub(
            r"\b(?!m\b|u\b|v\b|of\b)(\w)", lambda m: m.group(1).upper(), feature_name
        )
        if level > 0:
            feature_name += f" ({level}hPa)"
    elif level > 0:
        feature_name += f"_{level}hPa"

    return feature_name
