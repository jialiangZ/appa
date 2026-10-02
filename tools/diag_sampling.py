r"""Diagnose NaN in denoiser sampling under torchrun (distributed=True)."""

import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from appa.config.hydra import compose
from appa.utils.dates import create_trajectory_timestamps
from appa.models.diffusion import Denoiser, TrajectoryDenoiser, create_schedule
from appa.models.sampling import DDIMSampler
from appa.utils.io import load_denoiser

dist.init_process_group(backend="nccl")
rank = dist.get_rank()

mp = Path("artifacts/autoencoders/small_ae_v1/0/latents/small_v1/denoisers/917894i3/0")
cfg = compose(mp / "config.yaml")
sched = create_schedule(cfg.train, device="cuda")
bb = load_denoiser(mp, best=True, overrides={"checkpointing": True}).backbone.cuda()
bb.requires_grad_(False)

K = cfg.train.blanket_size
D = 23 * 47 * 32
T = 24

den = TrajectoryDenoiser(
    Denoiser(bb).cuda(),
    blanket_size=K,
    blanket_stride=12,
    state_size=D,
    distributed=True,
    pass_blanket_ids=False,
)
den = partial(den, date=create_trajectory_timestamps("2000-03-15", 0, T)[None].cuda())

x1 = torch.randn(1, T * D).cuda() * sched.sigma_tmax().cuda()
if rank == 0:
    print(f"x1: std={x1.std().item():.2f} NaN={torch.isnan(x1).sum().item()}", flush=True)

s = DDIMSampler(denoiser=den, steps=4, schedule=sched, silent=True)
xt = x1
dt = torch.as_tensor(1 / 4, device="cuda")

for i, t in enumerate(torch.linspace(sched.t_max, dt, 4, device="cuda")):
    sigma_t = sched(t)
    # Probe the denoiser alone at this sigma.
    with torch.no_grad():
        out = den(xt, sigma_t)
    if rank == 0:
        print(
            f"step {i}: sigma_t={sigma_t.item():.3f} "
            f"denoiser NaN={torch.isnan(out).sum().item()}/{out.numel()} "
            f"std={torch.nan_to_num(out).std().item():.4f}",
            flush=True,
        )
    xt = s.step(xt, t, t - dt)

dist.destroy_process_group()
