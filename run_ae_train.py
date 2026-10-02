r"""Local (no-Slurm) launcher for autoencoder training.

Usage:
    CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled torchrun \
        --standalone --nproc-per-node 1 run_ae_train.py [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled torchrun --standalone \
        --nproc-per-node 1 run_ae_train.py train.update_steps=2000
"""

import sys
import wandb

from appa.config.hydra import compose
from experiments.autoencoder.train import train

if __name__ == "__main__":
    cfg = compose(
        config_file="src/experiments/autoencoder/configs/train.yaml",
        overrides=sys.argv[1:],
    )

    runid = wandb.util.generate_id()
    print(f"Starting local AE training with runid: {runid}")

    train(runid, cfg, fork_lap=0, lap=0)
