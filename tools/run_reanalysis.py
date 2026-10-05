r"""Local (no-dawgz) reanalysis entry point.

Runs the reanalysis assimilation serially in a single process, bypassing the
dawgz workflow (whose async backend executes jobs in one process and conflicts
with the global torch.distributed / hydra state).

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node 1 \
        tools/run_reanalysis.py model_path=<denoiser lap> [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node 1 \
        tools/run_reanalysis.py \
        model_path=artifacts/autoencoders/b6vxfhdu/0/latents/lr240/denoisers/ipvnvlpe/0 \
        model_target=best
"""

import os
import sys

from pathlib import Path

import numpy as np
import torch

from einops import rearrange

from omegaconf import OmegaConf

OBS_DIR = Path(__file__).resolve().parents[1] / "src" / "experiments" / "observations"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Resolve the model path against the invocation directory BEFORE chdir.
    argv_model = next((a for a in sys.argv[1:] if a.startswith("model_path=")), None)
    if argv_model is not None:
        raw = argv_model.split("=", 1)[1]
        abs_model = (repo_root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
        sys.argv[sys.argv.index(argv_model)] = f"model_path={abs_model}"

    os.chdir(OBS_DIR)  # ground_masks/ and configs/ are cwd-relative

    from appa.config.hydra import compose
    from appa.utils.io import safe_load, safe_save
    from experiments.observations.reanalysis import reanalysis

    config = compose("configs/reanalysis.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    model_path = Path(str(config.model_path)).resolve()
    target_root = model_path / "reanalysis" / "local"
    target_root.mkdir(parents=True, exist_ok=True)

    denoiser_cfg = compose(model_path / "config.yaml")
    blanket_size = denoiser_cfg.train.blanket_size
    overlap = config.blanket_overlap
    stride = blanket_size - overlap

    lengths = [int(s) for s in str(config.assimilation_lengths).split(",")]
    num_samples = int(config.num_samples_per_date)
    start_dates = list(config.start_dates)

    for length in lengths:
        trajectory_size = max(blanket_size, length)
        while (trajectory_size - blanket_size) % stride != 0:
            trajectory_size += 1

        window_dir = target_root / f"{length}h"
        window_dir.mkdir(parents=True, exist_ok=True)

        for i, date in enumerate(d for d in start_dates for _ in range(num_samples)):
            day, hour = str(date).split()
            tmp = window_dir / f"tmp_{i}"
            tmp.mkdir(parents=True, exist_ok=True)

            print(
                f"[reanalysis {length}h] sample {i + 1}/{len(start_dates) * num_samples}",
                flush=True,
            )
            job_kwargs = {
                k: v
                for k, v in config.items()
                if k
                not in ("hardware", "assimilation_lengths", "num_samples_per_date", "start_dates")
            }
            reanalysis(
                assimilation_length=length,
                padded_trajectory_size=trajectory_size,
                start_date=day,
                start_hour=int(hour.split("h")[0]),
                target_dir=tmp,
                split_seed=np.random.randint(0, 2**32 - 1),
                **job_kwargs,
            )

        # Aggregate tmp_* into (dates, samples) tensors, like the dawgz job.
        trajectories, timestamps = [], []
        for i in range(len(start_dates) * num_samples):
            tmp = window_dir / f"tmp_{i}"
            trajectories.append(safe_load(tmp / "trajectory.pt"))
            timestamps.append(safe_load(tmp / "timestamps.pt"))
            import shutil

            shutil.move(str(tmp / "masks.h5"), str(window_dir / f"masks_{i}.h5"))
            shutil.rmtree(tmp)

        trajectories = rearrange(torch.cat(trajectories), "(d s) ... -> d s ...", s=num_samples)
        timestamps = rearrange(torch.cat(timestamps), "(d s) ... -> d s ...", s=num_samples)
        safe_save(trajectories, window_dir / "trajectories.pt")
        safe_save(timestamps, window_dir / "timestamps.pt")
        print(
            f"[reanalysis {length}h] saved {tuple(trajectories.shape)} -> {window_dir}", flush=True
        )


if __name__ == "__main__":
    main()
