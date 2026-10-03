r"""Local launcher for latent diffusion (denoiser) training (thin wrapper).

Usage:
    CUDA_VISIBLE_DEVICES=3 SWANLAB_MODE=disabled python run_den_train.py \
        ae_run=<runid>/0 latent_dump=<dump_id> [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 SWANLAB_MODE=disabled python run_den_train.py \
        ae_run=abc123/0 latent_dump=default \
        train.update_steps=2000 train.batch_size_per_step=8
"""

from appa.launchers import den_train

if __name__ == "__main__":
    import sys

    den_train(sys.argv[1:])
