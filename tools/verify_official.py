r"""Decode official-model latents and check physical plausibility."""

from pathlib import Path

import torch
from einops import rearrange

from appa.config import PATH_STAT
from appa.data.const import CONTEXT_VARIABLES, ERA5_VARIABLES, SUB_PRESSURE_LEVELS
from appa.data.transforms import StandardizeTransform
from appa.utils.io import load_auto_encoder, safe_load

latent_dir = Path("artifacts/autoencoders/workshop/0/latents/workshop")
ae = load_auto_encoder(latent_dir / "ae", "model", device="cuda", eval_mode=True).cuda()
ae.requires_grad_(False)
stats = safe_load(latent_dir / "stats.pth")
mean, std = stats["mean"].cuda(), stats["std"].cuda()

traj = safe_load(latent_dir.parent / "prior" / "local" / "24h" / "trajectory.pt")
ts = safe_load(latent_dir.parent / "prior" / "local" / "24h" / "timestamps.pt")
print(f"潜轨迹: {tuple(traj.shape)}  标准化空间 std={traj.std():.3f} mean={traj.mean():.4f}")

# Decode t=0 and t=12 (two moments), in physical units.
st = StandardizeTransform(
    PATH_STAT,
    state_variables=ERA5_VARIABLES,
    context_variables=CONTEXT_VARIABLES,
    levels=SUB_PRESSURE_LEVELS,
)

# Context: official config has 6 context channels; zeros after standardization = mean value.
ctx = torch.zeros(1, 0 if not st.context_mean is None else 0)

for t in [0, 12, 23]:
    z = traj[0, t].cuda() * std + mean
    tstamp = ts[0, 0, t : t + 1].cuda()
    # timestamp must be float for the model
    with torch.no_grad():
        x = ae.decode(z, tstamp.float(), None)  # (1, 1038240, 71)
    x = x.float().cpu()  # (1038240, 71)

    # Unstandardize physical fields (state_mean/std shape: (1,71,1,1))
    m = st.state_mean.reshape(-1)  # (71,)
    s = st.state_std.reshape(-1)
    x_phys = x * s + m  # (1038240, 71) broadcast

    names = [
        "2m_temperature",
        "10m_u_wind",
        "10m_v_wind",
        "MSLP",
        "SST",
        "precip",
        "temperature",
        "u_wind",
        "v_wind",
        "geopotential",
        "specific_humidity",
    ]
    print(f"\n--- t={t}h 物理量纲检查 (global mean) ---")
    for i, n in enumerate(names):
        if n == "specific_humidity":
            # 13 levels: show 500hPa (index 8 of SUB levels: [50,100,150,200,250,300,400,500,...] idx=7)
            v = x_phys[:, 6 + 7].mean()
            print(f"  {n:20s} @500hPa: {v:.4f} kg/kg  (典型 ~1e-3)")
        elif n in ("temperature", "u_wind", "v_wind", "geopotential"):
            v_lo = x_phys[:, 6 + (6 - 1) * 13 + 0].mean() if False else None
        else:
            v = x_phys[:, i].mean()
            print(f"  {n:20s}: {v:10.3f}")
    if t == 0:
        t2m = x_phys[:, 0]
        print(f"\n  2m_temperature range: {t2m.min():.1f} .. {t2m.max():.1f} K (期望 ~200-320)")
