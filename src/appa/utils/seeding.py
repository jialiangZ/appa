r"""Reproducibility helpers."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "seed_everything",
]


def seed_everything(seed: int) -> None:
    r"""Seeds all random number generators for reproducibility.

    The same seed is applied on every rank so that model initialization stays
    identical across DDP processes. Rank-specific randomness (data shuffling,
    dropout realizations) is handled deterministically by the distributed
    sampler and the CUDA RNG state.

    Arguments:
        seed: The base seed. Use a negative value to disable seeding.
    """

    if seed is None or seed < 0:
        logger.info("Random seeding disabled (seed=%s).", seed)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Seeded random generators with seed=%d.", seed)
