r"""Benchmark and sanity-check ERA5Dataset loading modes.

Compares the lazy Zarr path (`cache="none"`) against the in-memory cache
(`cache="ram"`) on the real ERA5 data, replicating the autoencoder training
configuration (variables, levels, standardization). Verifies numerical
equivalence and reports throughput.

Usage:
    .venv/bin/python scripts/bench_dataloader.py [--days 3] [--samples 64]
"""

import argparse
import sys
import time
from contextlib import contextmanager

import torch

from appa.config import PATH_ERA5, PATH_STAT
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TRAINING,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
)
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from datetime import datetime, timedelta


@contextmanager
def timer(label: str, results: dict):
    start = time.perf_counter()
    yield
    results[label] = time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="Days of data to load.")
    parser.add_argument("--samples", type=int, default=64, help="Samples to benchmark.")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="DataLoader batch size (per GPU)."
    )
    args = parser.parse_args()

    start = datetime.strptime(DATASET_DATES_TRAINING[0], "%Y-%m-%d")
    end = start + timedelta(days=args.days - 1)
    start_date, end_date = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    st = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=ERA5_PRESSURE_LEVELS,
    )

    common = dict(
        path=PATH_ERA5,
        start_date=start_date,
        end_date=end_date,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=ERA5_PRESSURE_LEVELS,
        transform=st,
    )

    timings: dict = {}
    step_mb = None

    # --- Lazy path -----------------------------------------------------------
    with timer("lazy_init", timings):
        ds_lazy = ERA5Dataset(cache="none", **common)

    with timer("lazy_sequential", timings):
        for i in range(args.samples):
            state, context, time_feats = ds_lazy[i]
    step_mb = state.element_size() * state.numel() / 1e6

    # --- RAM path ------------------------------------------------------------
    with timer("ram_init", timings):
        ds_ram = ERA5Dataset(cache="ram", **common)

    with timer("ram_sequential", timings):
        for i in range(args.samples):
            state_r, context_r, time_r = ds_ram[i]

    # --- Equivalence ---------------------------------------------------------
    n_check = min(8, len(ds_lazy))
    max_err = 0.0
    for i in range(n_check):
        s_l, c_l, t_l = ds_lazy[i]
        s_r, c_r, t_r = ds_ram[i]
        assert s_l.shape == s_r.shape, f"shape mismatch at {i}: {s_l.shape} vs {s_r.shape}"
        assert torch.equal(t_l, t_r), f"time feature mismatch at {i}"
        max_err = max(max_err, (s_l - s_r).abs().max().item())
        max_err = max(max_err, (c_l - c_r).abs().max().item())

    # --- DataLoader throughput ----------------------------------------------
    from appa.data.dataloaders import get_dataloader

    n_dl = min(args.samples, len(ds_lazy) - 1)
    with timer("lazy_dataloader", timings):
        loader = get_dataloader(ds_lazy, batch_size=args.batch_size, num_workers=4)
        for i, _ in enumerate(loader):
            if i + 1 >= n_dl:
                break

    with timer("ram_dataloader", timings):
        loader = get_dataloader(ds_ram, batch_size=args.batch_size, num_workers=0)
        for i, _ in enumerate(loader):
            if i + 1 >= n_dl:
                break

    # --- Report --------------------------------------------------------------
    n_time = ds_lazy.dataset.time.size
    print()
    print("=" * 72)
    print(f"ERA5Dataset loading benchmark  ({start_date} .. {end_date}, {n_time} steps)")
    print(f"sample tensor: {tuple(state.shape)}, {state_mb_str(step_mb)}, batch={args.batch_size}")
    print("=" * 72)
    rows = [
        ("init (total)", "lazy_init", "ram_init", f"{n_time} steps materialized"),
        ("__getitem__ x%d" % args.samples, "lazy_sequential", "ram_sequential", "sequential"),
        ("DataLoader x%d" % n_dl, "lazy_dataloader", "ram_dataloader", f"workers 4 vs 0"),
    ]
    print(f"{'phase':<24}{'lazy (s)':>12}{'ram (s)':>12}{'speedup':>10}  notes")
    for label, lazy_key, ram_key, note in rows:
        lazy_t, ram_t = timings[lazy_key], timings[ram_key]
        print(f"{label:<24}{lazy_t:>12.3f}{ram_t:>12.3f}{lazy_t / max(ram_t, 1e-9):>9.1f}x  {note}")

    lazy_per = timings["lazy_sequential"] / args.samples * 1e3
    ram_per = timings["ram_sequential"] / args.samples * 1e3
    print(f"\nper-sample __getitem__: lazy {lazy_per:.2f} ms | ram {ram_per:.3f} ms")
    print(f"max |lazy - ram| over {n_check} checked samples: {max_err:.3e}")

    if max_err > 1e-3:
        print("WARNING: cached and lazy samples differ beyond float tolerance!")
        sys.exit(1)

    print("\nOK: cache modes are numerically equivalent.")


def state_mb_str(step_mb: float) -> str:
    return f"{step_mb:.1f} MB/state"


if __name__ == "__main__":
    main()
