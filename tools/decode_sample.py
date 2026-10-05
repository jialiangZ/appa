r"""Decode generated latents to physical space and compare against ERA5 ground truth."""

import sys
from pathlib import Path

import torch
from einops import rearrange

from appa.data.const import ERA5_SURFACE_VARIABLES
from appa.data.datasets import ERA5Dataset
from appa.utils.io import load_auto_encoder, safe_load

model_path = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("artifacts/autoencoders/small_ae_v1/0/latents/small_v1/denoisers/917894i3/0")
)
latent_dir = model_path.parents[2]

ae = load_auto_encoder(latent_dir / "ae", "model", device="cuda", eval_mode=True).cuda()
ae.requires_grad_(False)
stats = safe_load(latent_dir / "stats.pth")
latent_mean, latent_std = stats["mean"], stats["std"]

traj = safe_load(model_path.parents[3] / "prior" / "local" / "24h" / "trajectory.pt")
ts = safe_load(model_path.parents[3] / "prior" / "local" / "24h" / "timestamps.pt")

print(f"潜空间轨迹: {tuple(traj.shape)}")
print(f"时间戳范围: {ts[0, 0].tolist()} -> {ts[0, -1].tolist()}")

# Decode all 24 hours in batches.
print("\n解码中...")
decoded = []
for t in range(traj.shape[1]):
    z = traj[0, t].cuda() * latent_std.cuda() + latent_mean.cuda()
    x = ae.decode(z, ts[0, 0, t : t + 1].cuda(), None)
    decoded.append(x.cpu())
    if (t + 1) % 8 == 0:
        print(f"  {t + 1}/24")

x = torch.stack(decoded, 0)  # (24, Lat*Lon, Z)
print(f"物理空间: {tuple(x.shape)}")

# Ground truth
y, m, d = [int(v) for v in ts[0, 0, 0, :3].tolist()]
start = f"{y:04d}-{m:02d}-{d:02d}"
from appa.config import PATH_STAT
from appa.data.const import CONTEXT_VARIABLES, ERA5_VARIABLES, ERA5_PRESSURE_LEVELS
from appa.data.transforms import StandardizeTransform

st = StandardizeTransform(
    PATH_STAT,
    state_variables=ERA5_VARIABLES,
    context_variables=CONTEXT_VARIABLES,
    levels=ERA5_PRESSURE_LEVELS,
)
ds = ERA5Dataset(
    "artifacts/data/era5_2000-1h-1440x721.zarr",
    start,
    start,
    state_variables=ERA5_VARIABLES,
    context_variables=CONTEXT_VARIABLES,
    levels=ERA5_PRESSURE_LEVELS,
    trajectory_size=24,
    transform=st,
)
# _load_sub_trajectory returns (T, Z, Lat, Lon); flatten to (T, Lat*Lon, Z)
truth, _, _ = ds[0]
truth = rearrange(truth, "T Z H W -> T (H W) Z")
assert truth.shape == x.shape, f"{truth.shape} vs {x.shape}"

print("\n" + "=" * 60)
print("数值对比 (Z=4 通道)")
print("=" * 60)
print(f"{'channel':<26}{'sample mean':>14}{'sample std':>13}{'truth mean':>13}{'RMSE':>10}")
for c in range(x.shape[-1]):
    name = ERA5_SURFACE_VARIABLES[c] if c < len(ERA5_SURFACE_VARIABLES) else f"level_{c}"
    s, t = x[..., c], truth[..., c]
    rmse = torch.sqrt(((s - t) ** 2).mean())
    print(f"{name:<26}{s.mean():>14.4f}{s.std():>13.4f}{t.mean():>13.4f}{rmse:>10.4f}")

print("\n(样本已标准化: mean≈0, std≈1 表示与训练分布一致)")
torch.save(x, "/tmp/opencode/decoded_sample.pt")
print("\n已保存 /tmp/opencode/decoded_sample.pt")
