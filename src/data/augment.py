import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_base_augmentations(enable_rotate90=True):
    aug_list = [
        A.OneOf(
            [
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.75, 0.75),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LANCZOS4,
                    },
                    p=0.1,
                ),
                A.Downscale(
                    scale_range=(0.95, 0.95),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
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
        A.Transpose(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.1),
            rotate=(-30, 30),
            shear=(-10, 10),
            p=0.3,
        ),
        A.ElasticTransform(alpha=1, sigma=20, p=0.1),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Equalize(p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        A.UnsharpMask(
            blur_limit=(3, 5), sigma_limit=(1.0, 2.0), alpha=(0.1, 0.3), p=0.2
        ),
        A.GaussNoise(p=0.1),
    ]
    if enable_rotate90:
        aug_list.insert(5, A.RandomRotate90(p=0.5))  # Insert after VerticalFlip
    return aug_list


def get_train_augmentation(
    height, width, extra_aug=None, resize_first=True, enable_rotate90=True
):
    aug_list = []
    if resize_first:
        aug_list.append(A.Resize(height, width))
    aug_list += get_base_augmentations(enable_rotate90=enable_rotate90)
    # if not resize_first:
    #    aug_list.append(A.Resize(height, width))
    if extra_aug:
        aug_list += extra_aug
    aug_list += [
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(aug_list)


def get_test_augmentation(height, width, extra_aug=None, resize_first=True):
    aug_list = []
    if resize_first:
        aug_list.append(A.Resize(height, width))
    if extra_aug:
        aug_list += extra_aug
    aug_list += [
        A.Normalize([0.5] * 3, [0.5] * 3),
        ToTensorV2(),
    ]
    return A.Compose(aug_list)
