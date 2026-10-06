"""Patch (MIL) dataset. The strip geometry / preprocessing lives in patches.py (shared with
Grad-CAM and deploy/); this module only adds file loading, augmentation and labels."""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from .augment import get_train_augmentation
from .imaging import load_rgb
from .metadata import LABEL_COL
from .patches import PatchSample, _legacy_boxes, patch_boxes, prepare_patches  # noqa: F401  (re-exported)


class CancerPatchDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder: str,
        img_size,
        num_patches: int = 3,
        overlap_ratio: float = 0.2,
        local_scale: float = 1.0,
        rotate_landscape: bool = True,
        train: bool = False,
        legacy: bool = False,
    ):
        if data_folder is None:
            raise ValueError("data_folder must not be None")
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.num_patches = int(num_patches)
        self.overlap_ratio = float(overlap_ratio)
        self.local_scale = float(local_scale)
        self.rotate_landscape = bool(rotate_landscape)
        self.legacy = bool(legacy)
        self.transform = get_train_augmentation(rotate90=False, transpose=False) if train else None
        self._links = self.df["link"].tolist()
        self._labels = self.df[LABEL_COL].astype(int).tolist()

    def __len__(self):
        return len(self._links)

    def __getitem__(self, idx):
        img = load_rgb(os.path.join(self.data_folder, self._links[idx]))
        sample = prepare_patches(
            img,
            self.img_size,
            self.num_patches,
            self.overlap_ratio,
            self.local_scale,
            self.rotate_landscape,
            augment=self.transform,
            legacy=self.legacy,
        )
        return sample.tensor, torch.tensor(self._labels[idx], dtype=torch.long)
