import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .augment import get_train_augmentation, get_test_augmentation
from .dataloader import get_image_size_from_config, get_weighted_sampler
import multiprocessing


def get_num_workers():
    # Trả về số worker tối thiểu là 2, tối đa là 4
    try:
        cpu_count = multiprocessing.cpu_count()
        return max(2, min(6, cpu_count))
    except Exception:
        return 2


class CancerImageDataset(Dataset):
    def __init__(self, df, data_folder, transform=None):
        if data_folder is None:
            raise ValueError(
                "data_folder must not be None. Please provide the data folder path."
            )
        self.df = df.reset_index(drop=True)
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.df.loc[idx, "link"])
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        label = int(self.df.loc[idx, "cancer"])
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def get_dataloaders(
    train_df,
    test_df,
    data_folder,
    batch_size=16,
    config_path="config/config.yaml",
    img_size=None,
    num_workers=None,
    pin_memory=True,
):
    if num_workers is None:
        num_workers = get_num_workers()
    if img_size is None:
        img_size = get_image_size_from_config(config_path)
    if isinstance(img_size, (list, tuple)):
        height, width = int(img_size[0]), int(img_size[1])
    else:
        height = width = int(img_size)
    train_transform = get_train_augmentation(height, width, resize_first=True)
    test_transform = get_test_augmentation(height, width, resize_first=True)
    train_dataset = CancerImageDataset(train_df, data_folder, train_transform)
    test_dataset = CancerImageDataset(test_df, data_folder, test_transform)
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
    return train_loader, test_loader
