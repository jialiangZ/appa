r"""Decode the lr240 prior trajectory to physical space and score against ERA5.

Usage:
    .venv/bin/python tools/decode_lr240.py [traj_path] [--device cuda]

Defaults to the 24h prior trajectory generated from the lr240 denoiser
(ipvnvlpe). Reports per-variable RMSE in physical units after
un-standardization, plus the climatology-spread ratio for context.
"""

import argparse
import sys

from pathlib import Path

import torch

from einops import rearrange

from appa.config import PATH_ERA5, PATH_STAT
from appa.data.const import CONTEXT_VARIABLES, ERA5_PRESSURE_LEVELS, ERA5_VARIABLES
from appa.data.datasets import ERA5Dataset
from appa.data.mappings import feature_idx_to_name
from appa.data.transforms import StandardizeTransform
from appa.utils.io import load_auto_encoder, safe_load

LATENT_DIR = Path("artifacts/autoencoders/b6vxfhdu/0/latents/lr240")
DEFAULT_TRAJ = LATENT_DIR.parent / "prior" / "local" / "24h" / "trajectories.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_path", nargs="?", default=DEFAULT_TRAJ)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load artifacts ------------------------------------------------------
    ae = load_auto_encoder(LATENT_DIR / "ae", "model", device=device, eval_mode=True).to(device)
    ae.requires_grad_(False)

    stats = safe_load(LATENT_DIR / "stats.pth")
    latent_mean, latent_std = stats["mean"].to(device), stats["std"].to(device)

    traj = safe_load(args.traj_path)  # (D, S, T, N, C)
    ts = safe_load(Path(args.traj_path).parent / "timestamps.pt")  # (D, S, T, 4)

    # Flatten (date, sample) into one batch axis and keep the first trajectory.
    traj = traj.reshape(-1, *traj.shape[-3:])[:1]  # (1, T, N, C)
    ts = ts.reshape(-1, *ts.shape[-2:])[:1]  # (1, T, 4)

    print(f"轨迹: {tuple(traj.shape)}  时间戳: {ts[0, 0].tolist()} -> {ts[0, -1].tolist()}")

    # --- Decode all hours ----------------------------------------------------
    print("\n解码中...")
    decoded = []
    for t in range(traj.shape[1]):
        z = traj[0, t].to(device) * latent_std + latent_mean
        x = ae.decode(z, ts[0, t : t + 1].to(device), None)
        decoded.append(x.cpu())
    x = torch.stack(decoded, 0)  # (T, N, Z) standardized space
    print(f"物理空间 (标准化): {tuple(x.shape)}")

    # --- Ground truth --------------------------------------------------------
    y0 = ts[0, 0].tolist()
    start = f"{y0[0]:04d}-{y0[1]:02d}-{y0[2]:02d}"

    st = StandardizeTransform(
        PATH_STAT,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=ERA5_PRESSURE_LEVELS,
    )
    ds = ERA5Dataset(
        PATH_ERA5,
        start,
        start,
        state_variables=ERA5_VARIABLES,
        context_variables=CONTEXT_VARIABLES,
        levels=ERA5_PRESSURE_LEVELS,
        trajectory_size=traj.shape[1],
        transform=st,
        cache="none",
    )
    truth, _, _ = ds[0]  # (T, Z, Lat, Lon)
    truth = rearrange(truth, "T Z H W -> T (H W) Z")
    assert truth.shape == x.shape, f"{truth.shape} vs {x.shape}"

    # --- Un-standardize to physical units -------------------------------------
    mean = st.state_mean.squeeze()  # (Z,)
    std = st.state_std.squeeze()  # (Z,)
    x_phys = x * std + mean
    t_phys = truth * std + mean

    # --- Per-variable RMSE -----------------------------------------------------
    errors = (x_phys - t_phys).pow(2).mean(dim=(0, 1)).sqrt()  # (Z,)
    spread = (t_phys - t_phys.mean(dim=0, keepdim=True)).abs().mean(dim=(0, 1))

    print("\n" + "=" * 66)
    print(f"逐变量 RMSE（物理单位，{start} 起 {traj.shape[1]} 小时）")
    print("=" * 66)
    print(f"{'variable':<28}{'RMSE':>12}{'truth scale':>14}{'ratio':>9}")

    groups = {}
    for c in range(errors.shape[0]):
        name = feature_idx_to_name(c, sublevels=False)
        short = name.split("_")[0] if "hPa" in name else name
        groups.setdefault(short, []).append((name, errors[c], spread[c]))

    for short, items in groups.items():
        rmse = torch.stack([e for _, e, _ in items]).mean()
        sc = torch.stack([s for _, _, s in items]).mean()
        label = short if len(items) == 1 else f"{short} ({len(items)} levels avg)"
        print(f"{label:<28}{rmse:>12.4f}{sc:>14.4f}{rmse / sc:>9.2f}")

    total = errors.mean()
    print("-" * 66)
    print(f"{'ALL (71 channels)':<28}{total:>12.4f}")

    out = Path("/tmp/opencode/decoded_lr240.pt")
    torch.save(x_phys, out)
    print(f"\n已保存反标准化解码结果: {out}")


if __name__ == "__main__":
    main()
