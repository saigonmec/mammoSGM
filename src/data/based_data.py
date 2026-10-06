"""Dataset for the plain (single image -> class) models."""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from .augment import get_train_augmentation
from .imaging import load_rgb, normalize_image, resize_to
from .metadata import LABEL_COL


class CancerImageDataset(Dataset):
    def __init__(self, df, data_folder: str, img_size, train: bool = False, legacy: bool = False):
        if data_folder is None:
            raise ValueError("data_folder must not be None")
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.legacy = bool(legacy)
        self.transform = get_train_augmentation() if train else None
        self._links = self.df["link"].tolist()
        self._labels = self.df[LABEL_COL].astype(int).tolist()

    def __len__(self):
        return len(self._links)

    def __getitem__(self, idx):
        img = load_rgb(os.path.join(self.data_folder, self._links[idx]))
        img = resize_to(img, *self.img_size, legacy=self.legacy)  # same resize as eval / Grad-CAM
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return normalize_image(img), torch.tensor(self._labels[idx], dtype=torch.long)
