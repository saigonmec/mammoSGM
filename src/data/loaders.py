"""DataLoader construction for both pipelines."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from .augment import reseed_transform
from .based_data import CancerImageDataset
from .metadata import LABEL_COL, get_weighted_sampler
from .patch_dataset import CancerPatchDataset


def _worker_init_fn(worker_id: int) -> None:
    """Every worker gets its own numpy/python/albumentations RNG stream (derived from the
    torch seed, hence reproducible); otherwise forked workers repeat the same augmentations."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    if info is not None:
        reseed_transform(getattr(info.dataset, "transform", None), seed)


def build_loaders(
    kind: str,
    train_df,
    val_df,
    test_df,
    data_folder: str,
    img_size,
    num_classes: int,
    batch_size: int = 16,
    num_workers: int = 2,
    balance: str = "sampler",
    seed: int = 42,
    pin_memory: bool = False,
    patch_kwargs: dict | None = None,
    legacy: bool = False,
) -> dict:
    """kind: 'based' | 'patch'. Returns {'train'?, 'val'?, 'test'?} (a split is skipped if its
    dataframe is None). balance='sampler' -> class-balanced WeightedRandomSampler on train."""

    def make_dataset(df, train: bool):
        if kind == "based":
            return CancerImageDataset(df, data_folder, img_size, train=train, legacy=legacy)
        return CancerPatchDataset(df, data_folder, img_size, train=train, legacy=legacy, **(patch_kwargs or {}))

    def make_loader(df, train: bool):
        ds = make_dataset(df, train)
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = None
        if train and balance == "sampler":
            sampler = get_weighted_sampler(df[LABEL_COL].values, num_classes, generator=g)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(train and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=train and len(ds) > batch_size,  # BN needs >1 sample in a batch
            generator=g if train else None,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    loaders = {}
    if train_df is not None:
        loaders["train"] = make_loader(train_df, True)
    if val_df is not None:
        loaders["val"] = make_loader(val_df, False)
    if test_df is not None:
        loaders["test"] = make_loader(test_df, False)
    return loaders
