r"""Data loaders."""

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Optional


def get_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last_ddp: bool = True,
    pin_memory: bool = True,
    num_workers: int = 0,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    **kwargs: Any,
) -> DataLoader:
    r"""Returns a possibly-distributed data loader for a given dataset.

    The sampler is created only if the rank is provided, and can be
    accessed with `dataloader.sampler`.

    Workers are kept alive between epochs (persistent_workers) and prefetch
    `prefetch_factor` batches each by default, avoiding the per-epoch worker
    restart cost. For RAM-cached datasets (`ERA5Dataset(cache="ram")`),
    `num_workers=0` is usually fastest since `__getitem__` is a pure tensor
    slice.

    Arguments:
        dataset: A Torch dataset.
        shuffle: Whether to shuffle the dataset or not.
        rank: The rank of the current process for distributed training. Default to None,
              corresponding to non distributed loader without sampler.
        world_size: The total number of processes for distributed training.
        drop_last_ddp: If true, adds indices to make the dataset length divisible by world_size.
                       If false, will ignore the last elements so that the dataset length is divisible.
        pin_memory: Pin the memory for faster move to the GPU.
        num_workers: Number of worker processes. 0 loads in the main process.
        persistent_workers: Whether to keep workers alive between epochs. Defaults to
                            true when num_workers > 0, false otherwise.
        prefetch_factor: Number of batches loaded in advance by each worker. Defaults
                         to 4 when num_workers > 0 (PyTorch default is 2).
        kwargs: Additional arguments passed to the DataLoader.
    """

    if rank is None:
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset=dataset,
            drop_last=drop_last_ddp,
            shuffle=shuffle,
            rank=rank,
            num_replicas=world_size,
        )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs = dict(kwargs)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 4

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        **loader_kwargs,
    )

    return dataloader
