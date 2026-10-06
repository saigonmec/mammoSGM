import os
import sys

# ===== FORCE RELOAD ALL GRADCAM MODULES =====
# Remove ALL cached gradcam modules before importing
modules_to_reload = [
    "src.gradcam.gradcam_utils_patch",
    "src.gradcam.gradcam_utils_based",
    "src.gradcam",
]
for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]
        # print(f"🔄 Removed cached module: {module_name}")

# Now import fresh modules
import torch
from torch import nn
import pandas as pd
import argparse
from PIL import Image
import numpy as np

from src.gradcam.gradcam_utils_based import pre_gradcam, post_gradcam
from src.gradcam.gradcam_utils_patch import pre_mil_gradcam, split_image_into_patches


def load_data_bbx3(data_folder):
    metadata_path = os.path.join(data_folder, "metadata2.csv")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(data_folder, "metadata.csv")
    df = pd.read_csv(metadata_path)
    # Create bbx column from x, y, width, height columns
    df["bbx"] = df[["x", "y", "width", "height"]].apply(
        lambda row: [row["x"], row["y"], row["width"], row["height"]], axis=1
    )
    # Group all bbx by image_id into a list
    bbx_grouped = (
        df.groupby("image_id")["bbx"]
        .apply(list)
        .reset_index()
        .rename(columns={"bbx": "bbx_list"})
    )
    df = df.merge(bbx_grouped, on="image_id", how="left")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    return train_df, test_df


def load_full_model(
    model_path: str,
) -> tuple[
    nn.Module,
    tuple[int, int],
    str | None,
    str | None,
    dict[str, list[float]] | None,
    int | None,
    str | None,
]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()

    input_size = checkpoint.get("input_size", (448, 448))
    model_name = checkpoint.get("model_name", None)
    gradcam_layer = checkpoint.get("gradcam_layer", None)
    normalize = checkpoint.get("normalize", None)
    num_patches = checkpoint.get("num_patches", None)
    arch_type = checkpoint.get("arch_type", None)

    return (
        model,
        input_size,
        model_name,
        gradcam_layer,
        normalize,
        num_patches,
        arch_type,
    )


def print_gradcam_info(
    model_out,
    input_tensor,
    img,
    target_layer,
    class_idx,
    pred_class,
    prob_class,
    bbx_list,
    gt,
    image_path=None,
):
    print("Model name:", type(model_out).__name__)
    print("Input tensor shape:", input_tensor.shape)
    print("Image:", img)
    if image_path is not None:
        print("Image file:", os.path.basename(image_path))
    print("Original image size:", img.size if hasattr(img, "size") else "N/A")
    print("Target layer:", target_layer)
    print("Class idx (input):", class_idx)
    print("Predicted class:", pred_class)
    print("Predicted prob:", prob_class)
    print("Bounding box list:", bbx_list)
    print("Ground truth:", gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument(
        "--random", type=int, default=42, help="Sample index to use (default: 42)"
    )
    args = parser.parse_args()

    # Load model and metadata
    model_tuple = load_full_model(args.pretrained_model_path)
    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        model_tuple
    )

    # Get test_df from load_data_bbx3
    _, test_df = load_data_bbx3(args.dataset_folder)

    # Select sample index from --random argument
    sample_idx = args.random
    if sample_idx >= len(test_df):
        sample_idx = 0  # fallback to first if out of range

    # Get image path, bounding box, ground truth from test_df
    image_path = os.path.join(args.dataset_folder, test_df.iloc[sample_idx]["link"])
    bbx_list = (
        test_df.iloc[sample_idx]["bbx_list"] if "bbx_list" in test_df.columns else None
    )
    gt = test_df.iloc[sample_idx]["cancer"] if "cancer" in test_df.columns else None

    # Prepare input for GradCAM
    if arch_type == "based":
        (
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
        ) = pre_gradcam(model_tuple[:5], image_path, target_layer=None, class_idx=None)
        print_gradcam_info(
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
            bbx_list,
            gt,
            image_path=image_path,
        )
        from src.gradcam.gradcam_utils_based import (
            gradcam,
            gradcam_plus_plus,
            post_gradcam,
        )

        gradcam_map = gradcam(model_out, input_tensor, target_layer, class_idx)
        # gradcam_map = gradcam_plus_plus(model_out, input_tensor, target_layer, class_idx)
        post_gradcam(
            gradcam_map,
            img,
            bbx_list=bbx_list,
            option=5,
            blend_alpha=0.5,
            pred=pred_class,
            prob=prob_class,
            gt_label=gt,
        )
    else:
        # Patch/MIL model
        result = pre_mil_gradcam(model_tuple, image_path)
        (
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
        ) = result

        # Store ORIGINAL image size before any processing
        original_img_size = img.size  # (width, height) in PIL format

        print_gradcam_info(
            model_out,
            input_tensor,
            img,
            target_layer,
            class_idx,
            pred_class,
            prob_class,
            bbx_list,
            gt,
            image_path=image_path,
        )

        from src.gradcam.gradcam_utils_patch import mil_gradcam

        # Import post_gradcam từ based
        from src.gradcam.gradcam_utils_based import post_gradcam

        gradcam_map = mil_gradcam(model_out, input_tensor, target_layer, class_idx)

        print(f"\nGradCAM shape: {gradcam_map.shape}, dtype: {gradcam_map.dtype}")

        # Visualize each patch with its own GradCAM
        if gradcam_map.ndim == 3:
            # Patch/MIL model - multiple heatmaps (one per patch)
            num_patches_result = gradcam_map.shape[0]

            # Get model metadata
            _, input_size_meta, _, _, _, num_patches_meta, arch_type_meta = model_tuple

            # Check if has global image
            has_global = (
                "v4" in arch_type_meta.lower() or "global" in arch_type_meta.lower()
            )

            patch_images = split_image_into_patches(
                img, num_patches_meta, input_size_meta, add_global=has_global
            )

            print(
                f"\nVisualizing {num_patches_result} patches{' (including 1 global image)' if has_global else ''}:"
            )
            print(f"  Created {len(patch_images)} patch images for visualization")
            print(
                f"  Original image size: {original_img_size[0]}x{original_img_size[1]} (W×H)"
            )

            # Verify consistency
            if num_patches_result != len(patch_images):
                print(
                    f"  ⚠️ WARNING: GradCAM has {num_patches_result} heatmaps but we have {len(patch_images)} patch images!"
                )
                print(
                    f"     Using min({num_patches_result}, {len(patch_images)}) for visualization"
                )
                num_to_visualize = min(num_patches_result, len(patch_images))
            else:
                num_to_visualize = num_patches_result

            # Separate local patches and global patch
            if has_global:
                patch_heatmaps = gradcam_map[:-1]
                patch_imgs = patch_images[:-1]
                global_heatmap = gradcam_map[-1]
                global_img = patch_images[-1]
            else:
                patch_heatmaps = gradcam_map
                patch_imgs = patch_images
                global_heatmap = None
                global_img = None

            # Visualize each local patch individually
            for patch_idx in range(len(patch_heatmaps)):
                patch_cam = patch_heatmaps[patch_idx]
                patch_img = patch_imgs[patch_idx]
                if isinstance(patch_img, np.ndarray):
                    patch_img = Image.fromarray(patch_img)
                pred_str = f"Patch {patch_idx + 1}: {pred_class}"
                post_gradcam(
                    patch_cam,
                    patch_img,
                    bbx_list=None,
                    option=5,
                    blend_alpha=0.5,
                    pred=pred_str,
                    prob=prob_class,
                    gt_label=None,
                )

            # --- Calculate patch positions based on split_image_into_patches logic ---
            # Get patch vertical positions (start_h, end_h) for each patch
            # This logic is copied from split_image_into_patches
            img_np = np.array(img)
            height, width = img_np.shape[:2]
            num_patches_actual = len(patch_heatmaps)
            overlap_ratio = 0.2
            patch_height = height // num_patches_meta
            step = int(patch_height * (1 - overlap_ratio))
            if num_patches_meta == 1 or step <= 0:
                starts = [0]
            else:
                starts = [i * step for i in range(num_patches_meta - 1)]
                starts.append(height - patch_height)
            patch_positions = []
            for i, start_h in enumerate(starts[:num_patches_actual]):
                end_h = start_h + patch_height
                if i == num_patches_meta - 1:
                    start_h = height - patch_height
                    end_h = height
                patch_positions.append((start_h, end_h))

            # --- Merge all local patch heatmaps vertically (top-down, max overlap) ---
            combined_heatmap_max = np.zeros((height, width), dtype=np.uint8)
            for idx, cam in enumerate(patch_heatmaps):
                # Resize patch heatmap to match patch region in original image
                start_h, end_h = patch_positions[idx]
                patch_h = end_h - start_h
                cam_img = Image.fromarray(cam).resize(
                    (width, patch_h), Image.Resampling.BILINEAR
                )
                cam_np = np.array(cam_img)
                # Use maximum for overlap
                combined_heatmap_max[start_h:end_h, :] = np.maximum(
                    combined_heatmap_max[start_h:end_h, :], cam_np
                )

            combined_heatmap_img = Image.fromarray(combined_heatmap_max).resize(
                original_img_size, Image.Resampling.BILINEAR
            )
            combined_heatmap_np = np.array(combined_heatmap_img)

            print("\n=== Combined Patch Heatmap (Top-down, max overlap) ===")
            post_gradcam(
                combined_heatmap_np,
                img,
                bbx_list=None,
                option=5,
                blend_alpha=0.5,
                pred=f"Combined Patch: {pred_class}",
                prob=prob_class,
                gt_label=None,
            )

            # Visualize the global patch last (if present)
            if has_global and global_heatmap is not None and global_img is not None:
                if isinstance(global_img, np.ndarray):
                    global_img = Image.fromarray(global_img)
                pred_str = f"Global: {pred_class}"
                print("\n=== Global Image ===")
                post_gradcam(
                    global_heatmap,
                    global_img,
                    bbx_list=None,
                    option=5,
                    blend_alpha=0.5,
                    pred=pred_str,
                    prob=prob_class,
                    gt_label=None,
                )
        else:
            # Standard model - single heatmap
            post_gradcam(
                gradcam_map,
                img,
                bbx_list=bbx_list,
                option=5,
                blend_alpha=0.5,
                pred=pred_class,
                prob=prob_class,
                gt_label=gt,
            )


if __name__ == "__main__":
    main()
