r"""Helpers shared by the training entry scripts."""

__all__ = [
    "PRECISION_MAP",
    "get_or_create_runid",
    "new_runid",
    "samples_to_batches",
    "val_check_fraction",
]

import os

PRECISION_MAP = {
    "float32": "32-true",
    "float16": "16-mixed",
    "bfloat16": "bf16-mixed",
}


def new_runid() -> str:
    r"""Generates a new run id (8 hex characters, like wandb run ids)."""

    import uuid

    return uuid.uuid4().hex[:8]


def get_or_create_runid() -> str:
    r"""Returns the run id of the current training invocation.

    The id is cached in the ``APPA_RUNID`` environment variable so that the
    DDP subprocesses spawned by Lightning (which re-execute the entry
    script) all agree on the same run id.
    """

    runid = os.environ.get("APPA_RUNID")

    if not runid:
        runid = new_runid()
        os.environ["APPA_RUNID"] = runid

    return runid


def samples_to_batches(samples: int, batch_size_per_step: int, grad_acc: int) -> int:
    r"""Converts a sample-count interval into a training-batch count.

    Arguments:
        samples: The interval, counted in samples seen.
        batch_size_per_step: The effective batch size of one optimizer step.
        grad_acc: The number of batches per optimizer step on this rank.
    """

    updates = max(samples // batch_size_per_step, 1)

    return max(updates * grad_acc, 1)


def val_check_fraction(interval_batches: int, num_batches: int) -> float:
    r"""Returns a fraction-of-epoch validation cadence for Lightning.

    Lightning accepts :py:`val_check_interval` as a number of batches, which
    must not exceed the epoch length. Since the original scripts used an
    (unbounded) infinite data cycle, intervals larger than an epoch are
    clamped to one validation per epoch.
    """

    fraction = interval_batches / max(num_batches, 1)

    return min(max(fraction, 1e-6), 1.0)
