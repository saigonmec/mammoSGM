import os
import pandas as pd
import numpy as np
import yaml
from torch.utils.data import WeightedRandomSampler


def print_dataset_stats2(train_df, test_df, name="Dataset"):
    total = len(train_df) + len(test_df)
    print(f"=== {name} Statistics ===")
    print(f"Total samples: {total}")
    print(f"  - Train: {len(train_df)} ({len(train_df) / total:.1%})")
    print(f"  - Test:  {len(test_df)} ({len(test_df) / total:.1%})")
    if "patient_id" in train_df.columns or "patient_id" in test_df.columns:
        n_train_patients = (
            train_df["patient_id"].nunique()
            if "patient_id" in train_df.columns
            else "N/A"
        )
        n_test_patients = (
            test_df["patient_id"].nunique()
            if "patient_id" in test_df.columns
            else "N/A"
        )
        print(f"Unique patients (train): {n_train_patients}")
        print(f"Unique patients (test):  {n_test_patients}")
    print("\nLabel distribution (cancer):")
    train_counts = train_df["cancer"].value_counts().sort_index()
    test_counts = test_df["cancer"].value_counts().sort_index()
    all_labels = sorted(set(train_counts.index).union(set(test_counts.index)))
    print("Label | Train  | Test | Total  | % of total")
    for label in all_labels:
        n_train = train_counts.get(label, 0)
        n_test = test_counts.get(label, 0)
        n_total = n_train + n_test
        percent = n_total / total * 100
        print(
            f"  {label}   |  {n_train}   |  {n_test}  |  {n_total}   |  {percent:.1f}%"
        )
    print("=========================")


def _resolve_config_path(config_path):
    # Nếu là đường dẫn tuyệt đối hoặc file tồn tại, dùng trực tiếp
    if os.path.isabs(config_path) or os.path.exists(config_path):
        return os.path.abspath(config_path)
    # Nếu chỉ truyền tên file, tìm trong thư mục config cùng cấp với src
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    config_dir_path = os.path.join(root_dir, "config")
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"
    config_path = os.path.join(config_dir_path, config_path)
    return os.path.abspath(config_path)


def get_image_size_from_config(config_path="config/config.yaml"):
    config_path = _resolve_config_path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    img_size = config.get("image_size", 448)
    if isinstance(img_size, (list, tuple)):
        return tuple(img_size)
    return (img_size, img_size)


def get_target_column_from_config(config_path="config/config.yaml"):
    config_path = _resolve_config_path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("target_column", "cancer")


def get_num_patches_from_config(config_path="config/config.yaml", num_patches=None):
    if num_patches is not None:
        return num_patches
    config_path = _resolve_config_path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        config = config["config"]
    return config.get("num_patches", 2)


def get_model_key(
    config_path="config/config.yaml", base_model_key="dataset", num_patches=None
):
    num_patches = get_num_patches_from_config(config_path, num_patches)
    patch_suffix = f"p{num_patches}"
    return f"{base_model_key}_{patch_suffix}"


def load_metadata(data_folder, config_path="config/config.yaml"):
    target_column = get_target_column_from_config(config_path)
    metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = df.drop_duplicates(subset=["link"])
    # Filter NaN in target label column
    df = df[df[target_column].notna()]
    if not np.issubdtype(df[target_column].dtype, np.number):
        df["target_label"], class_names = pd.factorize(df[target_column])
        for idx, name in enumerate(class_names):
            print(f"Factorize: {idx} -> {name}")
        label_col = "target_label"
    else:
        label_col = target_column
        unique_vals = df[target_column].unique()
        # Nếu tất cả giá trị là số nguyên (dù kiểu float), chuyển về int
        if np.all(np.mod(unique_vals, 1) == 0):
            df[target_column] = df[target_column].astype(int)
            class_names = sorted([int(x) for x in unique_vals])
        else:
            class_names = sorted(unique_vals)
        class_names = sorted(df[target_column].unique())
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    train_df["cancer"] = train_df[label_col]
    test_df["cancer"] = test_df[label_col]

    print_dataset_stats2(train_df, test_df, name="Dataset")
    # Nếu có nhiều hơn 2 class, in ra danh sách class
    if train_df[label_col].nunique() > 2:
        print(
            "Detected multi-class classification. Classes:",
            list(class_names),
        )
    return train_df, test_df, class_names


def get_weighted_sampler(train_df, label_col="cancer"):
    class_counts = train_df[label_col].value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[train_df.iloc[i][label_col]] for i in range(total_samples)]
    sampler = WeightedRandomSampler(
        weights=weights, num_samples=total_samples, replacement=True
    )
    return sampler
