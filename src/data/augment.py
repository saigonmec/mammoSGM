"""Training augmentations (albumentations >= 2.0). Output stays uint8 HWC; normalisation
happens later in `imaging.normalize_image` so train / eval / Grad-CAM share one path."""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2


def get_train_augmentation(rotate90: bool = True, transpose: bool = True) -> A.Compose:
    """Same augmentation recipe as the original project.

    For the MIL (patch) pipeline use `rotate90=False, transpose=False`: the image is
    portrait and cut into *vertical* strips, so rotating/transposing it would change what
    a "patch" means (the original code undid this with a rotate_if_landscape hack,
    which made Transpose just a duplicate of VerticalFlip).
    """
    augs = [
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LANCZOS4},
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.95, 0.95),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=0.8,
                ),
            ],
            p=0.125,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.01, 0.08),
            hole_width_range=(0.01, 0.15),
            fill=0,
            p=0.1,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if transpose:
        augs.append(A.Transpose(p=0.5))
    if rotate90:
        augs.append(A.RandomRotate90(p=0.5))
    augs += [
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-30, 30), shear=(-10, 10), p=0.3),
        A.ElasticTransform(alpha=1, sigma=20, p=0.1),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Equalize(p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2),
        A.GaussNoise(p=0.1),
    ]
    return A.Compose(augs)


def reseed_transform(transform, seed: int) -> None:
    """Give a (forked) DataLoader worker its own augmentation RNG stream."""
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(int(seed) % (2**32))
