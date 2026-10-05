r"""Aggregate leftover power-spectra chunks (spectral_tmp_*) into final zarrs.

Useful when the dawgz chain dies between the chunk jobs and the aggregate job
— the per-chunk results are already computed and only need merging.

Usage:
    .venv/bin/python tools/aggregate_spectra.py <out_folder> <pressure_levels csv>
"""

import shutil
import sys

from pathlib import Path

import numpy as np
import xarray as xr

from appa.data.const import ERA5_VARIABLES

statistics = ["mean", "var", "Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]


def aggregate(out_folder: Path, num_chunks: int, pressure_levels, config_save=None) -> None:
    r"""Merges spectral_tmp_<rank> chunk results into the final zarr stores."""

    out_folder = Path(out_folder)

    ds = xr.open_mfdataset(
        [out_folder / f"spectral_tmp_{rank}" / "power_spectra.zarr" for rank in range(num_chunks)],
        concat_dim="time",
        combine="nested",
        engine="zarr",
        parallel=True,
    ).sortby("time")

    stat_dict = {stat: {} for stat in statistics}
    data_arrays = []

    for v in ERA5_VARIABLES:
        data = ds[v].sel(statistics="spectra").load().values
        dimension = len(data.shape)

        for stat in statistics:
            if stat == "mean":
                stat_dict[stat][v] = data.mean(axis=0)
            elif stat == "var":
                stat_dict[stat][v] = data.var(axis=0)
            else:
                stat_dict[stat][v] = np.quantile(data, float(stat[1:]) / 100, axis=0)

            if dimension == 2:
                stat_dict[stat][v] = stat_dict[stat][v][np.newaxis, :]

        ds_array = xr.DataArray(
            np.stack([stat_dict[stat][v] for stat in statistics]),
            dims=("statistics", "level", "coefficients"),
            name=v,
        )
        if len(ds_array.level) == 1:
            ds_array = ds_array.squeeze("level")
        data_arrays.append(ds_array)

    stats_ds = xr.merge(data_arrays)
    stats_ds["statistics"] = statistics
    stats_ds["level"] = pressure_levels
    stats_ds["coefficients"] = np.arange(ds.sizes.get("coefficients", 121))
    stats_ds.to_zarr(out_folder / "statistics.zarr", mode="w")

    for var in ds.variables:
        ds[var].encoding.clear()
    ds.to_zarr(out_folder / "power_spectra.zarr", mode="w", consolidated=True)

    for rank in range(num_chunks):
        shutil.rmtree(out_folder / f"spectral_tmp_{rank}")

    print(f"Aggregated {num_chunks} chunks -> {out_folder}")


def main():
    out_folder = Path(sys.argv[1])
    num_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    pressure_levels = [int(s) for s in sys.argv[3].split(",")]
    aggregate(out_folder, num_chunks, pressure_levels)


if __name__ == "__main__":
    main()
