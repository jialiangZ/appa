r"""Evaluate reanalysis (assimilated) trajectories against ERA5.

Decodes the reanalysis outputs (tools/run_reanalysis.py) and scores
per-variable RMSE against ERA5, in physical units, alongside two references:
the unconditional prior samples and the two-truth-days baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/eval_reanalysis.py
"""

from pathlib import Path

import torch

from einops import rearrange

from appa.config import PATH_ERA5, PATH_STAT
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
)
from appa.data.datasets import ERA5Dataset
from appa.data.transforms import StandardizeTransform
from appa.utils.io import load_auto_encoder, safe_load

LATENT_DIR = Path("artifacts/autoencoders/b6vxfhdu/0/latents/lr240")
DENOISER_DIR = LATENT_DIR / "denoisers" / "ipvnvlpe" / "0"
PRIOR_TRAJ = LATENT_DIR.parent / "prior" / "local" / "24h" / "trajectories.pt"
START = "2017-04-01"


@torch.no_grad()
def decode(ae, traj, ts, latent_mean, latent_std, device):
    r"""Decodes (T, N, C) latents to physical space (T, N_latlon, Z)."""

    out = []
    for t in range(traj.shape[0]):
        z = traj[t].to(device) * latent_std + latent_mean
        out.append(ae.decode(z, ts[t : t + 1].to(device), None).cpu())
    return torch.stack(out, 0)


def main():
    device = torch.device("cuda")

    ae = load_auto_encoder(LATENT_DIR / "ae", "model", device=device, eval_mode=True).to(device)
    ae.requires_grad_(False)

    stats = safe_load(LATENT_DIR / "stats.pth")
    latent_mean, latent_std = stats["mean"].to(device), stats["std"].to(device)

    st = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=ERA5_PRESSURE_LEVELS,
    )
    ch_mean, ch_std = st.state_mean.squeeze(), st.state_std.squeeze()

    def truth_hours(start, T):
        ds = ERA5Dataset(
            PATH_ERA5,
            start,
            start,
            state_variables=ERA5_VARIABLES,
            context_variables=CONTEXT_VARIABLES,
            levels=ERA5_PRESSURE_LEVELS,
            transform=st,
            trajectory_size=T,
            cache="none",
        )
        y, _, _ = ds[0]
        return rearrange(y, "T Z H W -> T (H W) Z")

    def phys_rmse(x_std, y_std):
        a = x_std * ch_std + ch_mean
        b = y_std * ch_std + ch_mean
        return (a - b).pow(2).mean(dim=(0, 1)).sqrt()

    surf = {v: i for i, v in enumerate(ERA5_SURFACE_VARIABLES)}

    print("=" * 78)
    print("Reanalysis (obs-conditioned) vs prior vs baseline — physical-unit RMSE")
    print("=" * 78)

    # Baseline: truth vs another truth day (24h), full-trajectory reference.
    t1 = truth_hours(START, 24)
    t2 = truth_hours("2017-04-15", 24)
    base = phys_rmse(t2, t1)

    # Prior (unconditional): 4 samples of START, 24h.
    prior = safe_load(PRIOR_TRAJ)[0]  # (S, T, N, C) — first date
    prior_ts = safe_load(PRIOR_TRAJ.parent / "timestamps.pt")[0, 0]  # (T, 4)
    prior_rmse = None
    for s in range(prior.shape[0]):
        x = decode(ae, prior[s], prior_ts, latent_mean, latent_std, device)
        r = phys_rmse(x, t1)
        prior_rmse = r if prior_rmse is None else prior_rmse + r
    prior_rmse /= prior.shape[0]

    header = f"{'window':>6} | {'2m_temp':>8} {'10m_u':>7} {'10m_v':>7} {'MSLP':>8} | {'ALL71':>8}  (avg over 2 samples)"
    print(header)
    print("-" * 78)

    def row(label, r):
        print(
            f"{label:>6} | {r[surf['2m_temperature']]:>8.3f} "
            f"{r[surf['10m_u_component_of_wind']]:>7.3f} "
            f"{r[surf['10m_v_component_of_wind']]:>7.3f} "
            f"{r[surf['mean_sea_level_pressure']]:>8.1f} | {r.mean():>8.3f}"
        )

    print("[references, 24h]")
    row("base", base)
    row("prior", prior_rmse)
    print("[reanalysis: assimilation window = trajectory length]")

    for L in [1, 2, 4, 8, 16]:
        traj = safe_load(DENOISER_DIR / "reanalysis" / "local" / f"{L}h" / "trajectories.pt")[
            0
        ]  # (S, T, N, C)
        ts = safe_load(DENOISER_DIR / "reanalysis" / "local" / f"{L}h" / "timestamps.pt")[0, 0]

        y = truth_hours(START, L)
        r = None
        for s in range(traj.shape[0]):
            x = decode(ae, traj[s], ts, latent_mean, latent_std, device)
            rr = phys_rmse(x, y)
            r = rr if r is None else r + rr
        r /= traj.shape[0]
        row(f"{L}h", r)

    print(
        "\n注: base = 两个真值日互差（无条件可达下界参照）; prior = 无条件采样;\n"
        "     reanalysis 各窗口在同化窗口内由观测驱动重建, 应显著优于 prior。"
    )


if __name__ == "__main__":
    main()
