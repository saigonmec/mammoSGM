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
    Đầu ra cùng kiểu và shape như đầu vào (PIL, np.ndarray, torch.Tensor).
    Patch cuối cùng luôn lấy từ đáy lên, không pad thêm chiều cao.
    """
    # Ghi nhớ kiểu và shape đầu vào
    input_type = type(image)
    orig_shape = None
    if isinstance(image, torch.Tensor):
        orig_shape = image.shape  # (C, H, W)
        image = image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        orig_shape = image.shape

    # Đảm bảo (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 3 and image.shape[-1] != 3:
        if image.shape[1] == 3:
            image = np.transpose(image, (0, 2, 1))

    height, width = image.shape[:2]
    patch_height = height // num_patches
    step = int(patch_height * (1 - overlap_ratio))
    if num_patches == 1 or step <= 0:
        starts = [0]
    else:
        starts = [i * step for i in range(num_patches - 1)]
        starts.append(height - patch_height)

    patches = []
    for i, start_h in enumerate(starts):
        end_h = start_h + patch_height
        # Patch cuối cùng luôn lấy từ đáy lên
        if i == num_patches - 1:
            start_h = height - patch_height
            end_h = height
        patch = image[start_h:end_h, :, :]
        patches.append(patch)

    # Chuyển lại về kiểu và shape đầu vào nếu cần
    if input_type is torch.Tensor:
        # (H, W, C) -> (C, H, W)
        patches = [torch.from_numpy(p).permute(2, 0, 1).float() for p in patches]
    elif input_type is Image.Image:
        patches = [Image.fromarray(p) for p in patches]
    # Nếu là np.ndarray thì giữ nguyên

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
        # Load image and label
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        label = int(self.df.loc[idx, "cancer"])

        # Nếu ảnh rộng hơn dài thì xoay lại cho đúng orientation
        image = rotate_if_landscape(image)

        # Resize ảnh về đúng chiều rộng img_size trước khi augment, chiều cao giữ nguyên tỉ lệ
        w_target = (
            self.img_size[1]
            if isinstance(self.img_size, (tuple, list))
            else self.img_size
        )
        orig_w, orig_h = image.size
        if orig_w != w_target:
            new_h = int(orig_h * w_target / orig_w)
            image = image.resize((w_target, new_h), Image.BILINEAR)

        # Convert to numpy for processing
        image_np = np.array(image)

        # Apply augmentation if available
        if self.transform:
            augmented = self.transform(image=image_np)
            image = augmented["image"]
            # Xoay nếu cần
            image = rotate_if_landscape(image)
        else:
            image = image_np

        # Bỏ bước xoay 90 độ, giữ nguyên orientation
        patches = split_image_into_patches(
            image, self.num_patches, overlap_ratio=self.overlap_ratio
        )

        # Debug: in thông tin patch
        # for i, patch in enumerate(patches):
        #    print(
        #        f"[DEBUG] Patch {i}: type={type(patch)}, shape={getattr(patch, 'shape', None)}"
        #    )

        patch_tensors = []
        h, w = (
            self.img_size
            if isinstance(self.img_size, tuple)
            else (self.img_size, self.img_size)
        )

        # Các patch đã được transform (bao gồm Normalize + ToTensorV2) nếu augment ở đầu vào
        # Nếu image là torch.Tensor thì patch cũng là torch.Tensor, chỉ cần resize về (h, w)
        for patch in patches:
            if isinstance(patch, torch.Tensor):
                patch_np = patch.permute(1, 2, 0).numpy()
            else:
                patch_np = patch
            patch_np = cv2.resize(patch_np, (w, h))
            # Chuyển lại về tensor, không cần normalize nữa
            patch_tensor = torch.from_numpy(patch_np).permute(2, 0, 1).float()
            patch_tensors.append(patch_tensor)

        # Add full augmented image as the last patch
        if isinstance(image, torch.Tensor):
            augmented_image = image.permute(1, 2, 0).numpy()
        else:
            augmented_image = np.array(image)
        full_img_resized = cv2.resize(augmented_image, (w, h))
        full_img_tensor = torch.from_numpy(full_img_resized).permute(2, 0, 1).float()
        patch_tensors.append(full_img_tensor)

        patch_tensors = torch.stack(patch_tensors)

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
        train_transform = get_train_augmentation(
            height, width, resize_first=False, extra_aug=None, enable_rotate90=False
        )
        test_transform = get_test_augmentation(height, width, resize_first=False)
    else:
        # fallback: no resize if height/width are not valid
        train_transform = get_train_augmentation(
            448, 448, resize_first=False, extra_aug=None, enable_rotate90=False
        )
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

    # Lấy random một batch đầu tiên (hoặc batch đầu nếu không shuffle)
    batch = None
    for patches, labels in dataloader:
        batch = (patches, labels)
        break
    if batch is None:
        print("Không tìm thấy batch nào trong dataloader.")
        return
    patches, labels = batch  # patches: (B, num_patches+1, C, H, W)
    B, N, C, H, W = patches.shape

    # Unnormalize về [0, 1] để lưu ảnh greyscale
    def unnormalize_grey(img):
        # img: (C, H, W), chỉ lấy channel 0
        img0 = img[0].detach().cpu() * std[0] + mean[0]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 1e-8)  # scale về [0,1]
        return img0.unsqueeze(0)  # (1, H, W)

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


def rotate_if_landscape(image):
    """
    Xoay ảnh 90 độ nếu chiều rộng lớn hơn chiều cao.
    Hỗ trợ input là PIL.Image, np.ndarray, torch.Tensor.
    """
    # Chuyển sang numpy nếu cần
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    elif isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image

    if image_np.shape[1] > image_np.shape[0]:
        image_np = np.rot90(image_np).copy()  # copy() để tránh stride âm
        # Nếu đầu vào là PIL.Image thì trả về PIL.Image
        if isinstance(image, Image.Image):
            return Image.fromarray(image_np)
        # Nếu đầu vào là torch.Tensor thì trả về torch.Tensor
        if isinstance(image, torch.Tensor):
            return torch.from_numpy(image_np).permute(2, 0, 1).float()
        # Nếu đầu vào là numpy thì trả về numpy
        return image_np
    else:
        return image
