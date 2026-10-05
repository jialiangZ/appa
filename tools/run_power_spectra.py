r"""Serial (no-dawgz) power spectra computation.

Bypasses the dawgz async backend, whose thread-pool execution of chunk jobs
deadlocks with forked DataLoader workers. Chunks are computed one after
another in the main process, then aggregated.

Usage:
    .venv/bin/python tools/run_power_spectra.py ae|denoiser|gt
"""

import sys

from pathlib import Path

import torch

from omegaconf import open_dict

ROOT = Path("/home/jlzhang/Documents/appa")
LATENT_DIR = ROOT / "artifacts/autoencoders/b6vxfhdu/0/latents/lr240"
DENOISER_TRAJ = LATENT_DIR.parent / "prior/local/24h/trajectories.pt"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from appa.config import PATH_ERA5, PATH_STAT  # noqa: E402
from appa.config.hydra import compose  # noqa: E402
from appa.data.const import (  # noqa: E402
    DATASET_DATES_TEST,
    ERA5_PRESSURE_LEVELS,
    ERA5_VARIABLES,
)
from appa.data.dataloaders import get_dataloader  # noqa: E402
from appa.data.transforms import StandardizeTransform  # noqa: E402
from appa.utils.io import load_auto_encoder, safe_load  # noqa: E402

from experiments.physics.power_spectra import (  # noqa: E402
    compute_power_spectra,
    load_ae_dataset,
    load_denoiser_dataset,
    load_gt_dataset,
)
from aggregate_spectra import aggregate  # noqa: E402


def run(data_type: str) -> None:
    data_split = "test"
    standardize = True
    num_samples_per_date = 3

    if data_type == "denoiser":
        path = DENOISER_TRAJ
        out_folder = path.parent
    elif data_type == "ae":
        path = LATENT_DIR
        out_folder = LATENT_DIR / "physical_analysis"
    else:  # gt
        path = LATENT_DIR  # unused for gt, kept for uniformity
        out_folder = PATH_ERA5.parent / "physical_analysis"

    print(f"[power-spectra {data_type}] -> {out_folder}", flush=True)

    path_ae = LATENT_DIR / "ae"
    cfg_ae = compose(path_ae / "config.yaml")

    latent_stats = safe_load(LATENT_DIR / "stats.pth")
    latent_mean = latent_stats["mean"]
    latent_std = torch.sqrt(latent_stats["std"] ** 2 + cfg_ae.ae.noise_level**2)

    st = StandardizeTransform(
        PATH_STAT, state_variables=ERA5_VARIABLES, levels=ERA5_PRESSURE_LEVELS
    )
    era5_mean, era5_std = st.state_mean, st.state_std

    if data_type == "denoiser":
        dataset = load_denoiser_dataset(path, num_samples_per_date)
    elif data_type == "ae":
        dataset = load_ae_dataset(path / "era5.h5", DATASET_DATES_TEST)
    else:
        dataset = load_gt_dataset(DATASET_DATES_TEST, ERA5_PRESSURE_LEVELS, st)

    num_chunks = 10
    while len(dataset) % num_chunks != 0:
        num_chunks -= 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open_dict(cfg_ae):
        cfg_ae.ae.checkpointing = True
        cfg_ae.ae.noise_level = 0.0

    ae = load_auto_encoder(path_ae, "model", eval_mode=True)
    ae.decoder = ae.decoder.to(device)

    for rank in range(num_chunks):
        chunk_folder = out_folder / f"spectral_tmp_{rank}"
        chunk_folder.mkdir(parents=True, exist_ok=True)

        gt_dataset = load_gt_dataset(
            DATASET_DATES_TEST, ERA5_PRESSURE_LEVELS, st if standardize else None
        )
        gt_dataloader = get_dataloader(
            gt_dataset, rank=rank, world_size=num_chunks, num_workers=4, prefetch_factor=2
        )

        if data_type in ("denoiser", "ae"):
            if data_type == "denoiser":
                latent_dataset = load_denoiser_dataset(path, num_samples_per_date)
            else:
                latent_dataset = load_ae_dataset(path / "era5.h5", DATASET_DATES_TEST)
            latent_dataloader = get_dataloader(
                latent_dataset, rank=rank, world_size=num_chunks, num_workers=4, prefetch_factor=2
            )
        else:
            latent_dataloader = None

        print(f"[power-spectra {data_type}] chunk {rank + 1}/{num_chunks}", flush=True)

        compute_power_spectra(
            gt_dataloader,
            latent_dataloader,
            chunk_folder,
            ERA5_PRESSURE_LEVELS,
            data_type,
            ae,
            era5_mean,
            era5_std,
            standardize,
            latent_mean=latent_mean if data_type == "denoiser" else None,
            latent_std=latent_std if data_type == "denoiser" else None,
        )

    aggregate(out_folder, num_chunks, ERA5_PRESSURE_LEVELS)
    print(f"[power-spectra {data_type}] DONE", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ae")
