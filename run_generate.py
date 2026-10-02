r"""Local (no-dawgz) prior trajectory generation (thin wrapper).

Usage:
    CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node 1 \
        run_generate.py <model_path> [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc-per-node 1 \
        run_generate.py artifacts/autoencoders/small_ae_v1/0/latents/small_v1/denoisers/917894i3/0 \
        diffusion.num_steps=64 trajectory_sizes=72 start_dates=2000-03-15
"""

from appa.launchers import generate

if __name__ == "__main__":
    import sys

    generate(sys.argv[1:])
