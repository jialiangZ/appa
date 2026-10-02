r"""Local (no-Slurm) launcher for latent diffusion (denoiser) training.

Usage:
    CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled torchrun \
        --standalone --nproc-per-node 1 run_den_train.py \
        ae_run=<runid>/0 latent_dump=<dump_id> [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled torchrun --standalone \
        --nproc-per-node 1 run_den_train.py \
        ae_run=abc123/0 latent_dump=default \
        train.update_steps=2000 train.batch_size_per_step=8
"""

import sys
import wandb

from appa.config.hydra import compose
from experiments.diffusion.train import train

if __name__ == "__main__":
    cfg = compose(
        config_file="src/experiments/diffusion/configs/train.yaml",
        overrides=sys.argv[1:],
    )

    runid = wandb.util.generate_id()
    print(f"Starting local denoiser training with runid: {runid}")

    train(runid, cfg, fork_lap=0, lap=0)
