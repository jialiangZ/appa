r"""Local launcher for autoencoder training (thin wrapper).

Usage:
    CUDA_VISIBLE_DEVICES=3 SWANLAB_MODE=disabled python run_ae_train.py [overrides...]

Example:
    CUDA_VISIBLE_DEVICES=3 SWANLAB_MODE=disabled python run_ae_train.py \
        train.update_steps=2000
"""

from appa.launchers import ae_train

if __name__ == "__main__":
    import sys

    ae_train(sys.argv[1:])
