r"""Global paths and configuration helpers."""

__all__ = [
    "compose",
    "get_paths",
    "get_resolution",
    "DATA_REGISTRY",
]

import os

from pathlib import Path
from typing import Dict, Optional, Tuple

from .hydra import compose

PROJECT = Path("/home/jlzhang/Documents/appa")

PATH_ARTIFACTS = PROJECT / "artifacts"
PATH_AE = PATH_ARTIFACTS / "autoencoders"

# Data registry per (longitude, latitude) resolution. Switch the active
# resolution with the APPA_RESOLUTION environment variable (e.g. "240x121" or
# "1440x721") or by passing `resolution=` to `get_paths`.
DATA_REGISTRY: Dict[Tuple[int, int], Dict[str, Path]] = {
    (240, 121): {
        "era5": PATH_ARTIFACTS / "data" / "era5_2017-1h-240x121-std.zarr",
        "stat": PATH_ARTIFACTS / "data" / "stats_era5_2017-1h-240x121.zarr",
        "mask": PATH_ARTIFACTS / "data" / "masks_era5_2017-1h-240x121.zarr",
    },
    (1440, 721): {
        "era5": PATH_ARTIFACTS / "data" / "era5_2000-1h-1440x721.zarr",
        "stat": PATH_ARTIFACTS / "data" / "stats_era5_2000-1h-1440x721.zarr",
        "mask": PATH_ARTIFACTS / "data" / "masks_era5_1993-2021-1h-1440x721.zarr",
    },
}

DEFAULT_RESOLUTION = (240, 121)


def get_resolution() -> Tuple[int, int]:
    r"""Returns the active data resolution.

    The resolution is read from the APPA_RESOLUTION environment variable
    (format "WxH", e.g. "240x121") and defaults to 240x121 (the low-res repro
    setup). Raises ValueError for malformed or unregistered resolutions.
    """

    raw = os.environ.get("APPA_RESOLUTION", "240x121")
    normalized = raw.lower().replace("×", "x").strip()
    try:
        lon, lat = normalized.split("x")
        resolution = (int(lon), int(lat))
    except ValueError:
        raise ValueError(
            f"Invalid APPA_RESOLUTION={raw!r}. Expected the 'WxH' format, e.g. '240x121'."
        ) from None

    if resolution not in DATA_REGISTRY:
        available = ", ".join(f"{w}x{h}" for w, h in sorted(DATA_REGISTRY))
        raise ValueError(
            f"No data registered for resolution {resolution}. Available: {available}."
        )

    return resolution


def get_paths(resolution: Optional[Tuple[int, int]] = None) -> Dict[str, Path]:
    r"""Returns the data paths {"era5", "stat", "mask"} for the given resolution.

    Arguments:
        resolution: (longitude, latitude) pair. If None, the active resolution
            from `get_resolution` is used.
    """

    if resolution is None:
        resolution = get_resolution()

    return dict(DATA_REGISTRY[resolution])


# Backwards-compatible dynamic aliases. Resolved lazily so that the
# APPA_RESOLUTION environment variable is respected even if it is set after
# this module is imported.
_DYNAMIC_PATH_ALIASES = {
    "PATH_ERA5": "era5",
    "PATH_STAT": "stat",
    "PATH_MASK": "mask",
}


def __getattr__(name: str):
    if name in _DYNAMIC_PATH_ALIASES:
        return get_paths()[_DYNAMIC_PATH_ALIASES[name]]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
