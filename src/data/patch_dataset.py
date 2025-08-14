import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .augment import get_train_augmentation, get_test_augmentation
from .dataloader import (
    get_image_size_from_config,
    get_num_patches_from_config,
    get_weighted_sampler,
)
from .based_data import get_num_workers
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import multiprocessing
import random


def split_image_into_patches(image, num_patches=2, patch_size=None, overlap_ratio=0.2):
    """
    Split an image into vertical patches with optional overlap.

    Args:
        image (PIL.Image, np.ndarray, or torch.Tensor): Input image.
        num_patches (int): Number of patches to split.
        patch_size (tuple or None): Target patch size (height, width). If None, calculated automatically.
        overlap_ratio (float): Overlap ratio between consecutive patches.

    Returns:
        list of torch.Tensor: List of patch tensors (C, H, W).
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    else:
        image = np.array(image)

    # Đảm bảo image là uint8 và trong range [0, 255]
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]
    if patch_size is None:
        patch_height = height // num_patches
        patch_size = (patch_height, width)
    else:
        patch_height = patch_size[0]
    step = int(patch_height * (1 - overlap_ratio))
    patches = []

    # --- Tắt debug ---
    # print(
    #     f"[DEBUG] Splitting image shape {image.shape}, dtype {image.dtype}, range [{image.min()}, {image.max()}]"
    # )

    for i in range(num_patches):
        if i == num_patches - 1:
            start_h = height - patch_height
            end_h = height
        else:
            start_h = i * step
            end_h = start_h + patch_height
        start_h = max(0, start_h)
        end_h = min(height, end_h)

        patch = image[start_h:end_h, :, :]
        # print(
        #     f"[DEBUG] Patch {i}: shape {patch.shape}, range [{patch.min()}, {patch.max()}]"
        # )

        if patch.shape[0] != patch_height:
            pad_h = patch_height - patch.shape[0]
            patch = np.pad(patch, ((0, pad_h), (0, 0), (0, 0)), mode="edge")

        patch = cv2.resize(
            patch, (patch_size[1], patch_size[0]), interpolation=cv2.INTER_LINEAR
        )

        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        # print(
        #     f"[DEBUG] Patch {i} after tensor: range [{patch_tensor.min().item()}, {patch_tensor.max().item()}]"
        # )
        patches.append(patch_tensor)
    return patches


def compute_required_img_shape(img_size, num_patches, overlap_ratio=0.2):
    """
    Compute the required original image shape so that after splitting into patches (with overlap),
    each patch can be resized to img_size.

    Args:
        img_size (tuple): (height, width) of the final patch.
        num_patches (int): Number of patches.
        overlap_ratio (float): Overlap ratio between patches.

    Returns:
        tuple: (required_height, required_width)
    """
    patch_height, patch_width = img_size
    step = int(patch_height * (1 - overlap_ratio))
    required_height = step * (num_patches - 1) + patch_height
    required_width = patch_width
    return (required_height, required_width)


class CancerPatchDataset(Dataset):
    def __init__(
        self,
        df,
        data_folder,
        transform=None,
        num_patches=2,
        augment_before_split=True,
        config_path="config/config.yaml",
        img_size=None,
        overlap_ratio=0.2,
    ):
        """
        Dataset for splitting images into patches for MIL training.

        Args:
            df (pd.DataFrame): DataFrame with image paths and labels.
            data_folder (str): Root directory containing images.
            transform (albumentations.Compose or None): Augmentation pipeline.
            num_patches (int): Number of patches per image.
            augment_before_split (bool): Whether to augment before splitting.
            config_path (str): Path to config file.
            img_size (tuple or None): Target patch size (height, width).
            overlap_ratio (float): Overlap ratio between patches.
        """
        if data_folder is None:
            raise ValueError(
                "data_folder must not be None. Please provide the data folder path."
            )
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform = transform
        self.num_patches = num_patches
        if img_size is None:
            self.img_size = get_image_size_from_config(config_path)
        else:
            self.img_size = img_size
        self.augment_before_split = augment_before_split
        self.overlap_ratio = overlap_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            patch_tensors (torch.Tensor): Tensor of shape (num_patches+1, C, H, W).
            label (int): Image label.
        """
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])

        # Ensure img_size is (height, width) and both are int > 0
        img_size_hw = self.img_size
        if isinstance(img_size_hw, (list, tuple)):
            if len(img_size_hw) >= 2:
                h, w = int(img_size_hw[0]), int(img_size_hw[1])
            else:
                h = w = 448  # fallback
        else:
            h = w = int(img_size_hw) if img_size_hw else 448

        # Ensure positive values
        if h <= 0 or w <= 0:
            h = w = 448

        # Store original image for later use as full patch
        original_image = np.array(image)

        # --- Resize image to required shape before augmentation ---
        # required_shape = compute_required_img_shape(
        #    (h, w), self.num_patches, self.overlap_ratio
        # )
        # image = image.resize(
        #    (required_shape[1], required_shape[0]), resample=Image.BILINEAR
        # )

        # Apply augmentation to the resized image (no resize in this step)
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented["image"]
        else:
            image = np.array(image)

        # Đảm bảo ảnh đúng chiều (chiều cao >= chiều rộng) trước khi chia patches
        if isinstance(image, torch.Tensor):
            img_for_shape = image.permute(1, 2, 0).numpy()
        else:
            img_for_shape = image
        if img_for_shape.shape[0] < img_for_shape.shape[1]:
            # Nếu chiều cao < chiều rộng, xoay lại cho đúng (90 độ)
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()
                image = np.rot90(image).copy()  # copy() để tránh stride âm
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = np.rot90(image).copy()  # copy() để tránh stride âm
        # Split the augmented image into patches
        patches = split_image_into_patches(
            image, self.num_patches, overlap_ratio=self.overlap_ratio
        )
        patch_tensors = []

        # Create normalization and tensor conversion pipeline
        normalize_and_tensorize = A.Compose(
            [A.Normalize(mean=[0.5] * 3, std=[0.5] * 3), ToTensorV2()]
        )

        # Resize each patch to the standard size and convert to tensor
        for patch in patches:
            patch_np = patch.permute(1, 2, 0).numpy()
            if patch_np.max() > 1.0:
                patch_np = np.clip(patch_np, 0, 255).astype(np.uint8)
            else:
                patch_np = (patch_np * 255).astype(np.uint8)

            patch_np = cv2.resize(patch_np, (w, h))
            # print(
            #     f"[DEBUG] Patch before normalize: shape {patch_np.shape}, range [{patch_np.min()}, {patch_np.max()}]"
            # )

            patch_tensor = normalize_and_tensorize(image=patch_np)["image"]
            patch_tensors.append(patch_tensor)

        # Add full AUGMENTED image as the last patch (resize to img_size, normalize, to tensor)
        # Use augmented image (after transform), not the original image
        augmented_image = image

        # Xử lý chuyển đổi tensor/array cho full image
        if isinstance(augmented_image, torch.Tensor):
            # Nếu là tensor (C, H, W), chuyển về numpy (H, W, C)
            if augmented_image.dim() == 3:
                augmented_image = augmented_image.permute(1, 2, 0).numpy()
            else:
                augmented_image = augmented_image.numpy()
        else:
            augmented_image = np.array(augmented_image)

        # Đảm bảo là uint8 và shape đúng (H, W, C)
        if augmented_image.ndim == 3 and augmented_image.shape[2] == 3:
            # Đã đúng format (H, W, C)
            if augmented_image.dtype != np.uint8:
                if augmented_image.max() <= 1.0:
                    augmented_image = (augmented_image * 255).astype(np.uint8)
                else:
                    augmented_image = augmented_image.astype(np.uint8)

        full_img_resized = cv2.resize(augmented_image, (w, h))
        full_img_tensor = normalize_and_tensorize(image=full_img_resized)["image"]
        patch_tensors.append(full_img_tensor)

        patch_tensors = torch.stack(patch_tensors)

        # --- DEBUG: Lưu từng patch riêng biệt để kiểm tra ---
        # if idx < 3:
        #     print(
        #         f"[DEBUG] patch_tensors shape: {patch_tensors.shape}, dtype: {patch_tensors.dtype}, min: {patch_tensors.min().item()}, max: {patch_tensors.max().item()}"
        #     )
        #     debug_patches = patch_tensors.unsqueeze(0)  # (1, N, C, H, W)
        #     debug_labels = torch.tensor([label])
        #     save_random_batch_patches(
        #         [(debug_patches, debug_labels)], save_path=f"debug_patch_{idx}.png"
        #     )
        return patch_tensors, label


def get_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    num_patches=None,
    num_workers=None,
    pin_memory=True,
    img_size=None,
    overlap_ratio=0.2,
):
    if num_workers is None:
        num_workers = get_num_workers()
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    num_patches = get_num_patches_from_config(config_path, num_patches)
    # Ensure img_size is a tuple (height, width) with int values
    if isinstance(img_size, (list, tuple)):
        height, width = img_size
        if height is None or width is None:
            height = width = 448
        height, width = int(height), int(width)
    else:
        height = width = int(img_size)
    if height is None or width is None:
        height = width = 448
    # Only pass height, width to get_train_augmentation if both are not None
    if height is not None and width is not None:
        train_transform = get_train_augmentation(height, width, resize_first=False)
        test_transform = get_test_augmentation(height, width, resize_first=False)
    else:
        # fallback: no resize if height/width are not valid
        train_transform = get_train_augmentation(448, 448, resize_first=False)
        test_transform = get_test_augmentation(448, 448, resize_first=False)
    # Use the same img_size for both train and test datasets
    train_dataset = CancerPatchDataset(
        train_df,
        data_folder,
        train_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
        img_size=(height, width),
        overlap_ratio=overlap_ratio,
    )
    test_dataset = CancerPatchDataset(
        test_df,
        data_folder,
        test_transform,
        num_patches,
        augment_before_split=True,
        config_path=config_path,
        img_size=(height, width),
        overlap_ratio=overlap_ratio,
    )
    sampler = get_weighted_sampler(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- Lưu random batch mỗi khi load dataset ---
    try:
        # save_random_batch_patches(train_loader, save_path="random_batch_on_load.png")
        pass
    except Exception as e:
        print(f"[WARNING] Không thể lưu random batch patch khi load dataset: {e}")

    return train_loader, test_loader


def save_random_batch_patches(
    dataloader, save_path="random_batch_patches.png", mean=[0.5] * 3, std=[0.5] * 3
):
    """
    Lưu một batch ngẫu nhiên từ dataloader thành ảnh PNG.
    Mỗi hàng là các patch của cùng một ảnh (batch_size hàng, mỗi hàng num_patches+1 ảnh).
    Ảnh được lưu ở dạng greyscale (chỉ lấy channel 0).
    """
    import torchvision.utils as vutils

    # Lấy random một batch
    batch = None
    for i, (patches, labels) in enumerate(dataloader):
        if random.random() < 0.5 or batch is None:
            batch = (patches, labels)
        if i > 10:  # chỉ duyệt tối đa 10 batch đầu
            break
    if batch is None:
        print("Không tìm thấy batch nào trong dataloader.")
        return
    patches, labels = batch  # patches: (B, num_patches+1, C, H, W)
    B, N, C, H, W = patches.shape

    # Unnormalize về [0, 1] để lưu ảnh greyscale
    def unnormalize_grey(img):
        # img: (C, H, W), chỉ lấy channel 0
        img0 = img[0].clone() * std[0] + mean[0]
        return img0.clamp(0, 1).unsqueeze(0)  # (1, H, W)

    # Ghép các patch của mỗi ảnh thành một hàng (greyscale)
    rows = []
    for i in range(B):
        patch_imgs = [unnormalize_grey(patches[i, j]) for j in range(N)]  # (1, H, W)
        row = torch.cat(patch_imgs, dim=2)  # (1, H, N*W)
        rows.append(row)
    grid = torch.cat(rows, dim=1)  # (1, B*H, N*W)

    # Lưu ảnh greyscale
    vutils.save_image(grid, save_path)
    print(f"Đã lưu random batch patch grid (greyscale) vào {save_path}")
