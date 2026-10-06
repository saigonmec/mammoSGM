"""GradCAM and GradCAM++ implementation utilities."""

import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import math
# from src.data.patch_dataset import split_image_into_patches


def split_image_into_patches(
    img: Image.Image,
    num_patches: int,
    patch_size: tuple[int, int] = None,
    add_global: bool = False,
) -> list[Image.Image]:
    """
    Split an image into vertical patches with optional overlap.
    Optionally add a global (full resized) image as the last patch.
    """
    overlap_ratio = 0.2

    # Remember input type
    input_type = type(img)
    original_img = img

    # Convert to numpy for processing
    if isinstance(img, torch.Tensor):
        image = img.permute(1, 2, 0).cpu().numpy()
    elif isinstance(img, Image.Image):
        image = np.array(img)
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")

    # Ensure (H, W, C) format
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
        # Last patch always taken from bottom up
        if i == num_patches - 1:
            start_h = height - patch_height
            end_h = height
        patch = image[start_h:end_h, :, :]
        patches.append(patch)

    # Add global image as last patch if requested
    if add_global:
        if isinstance(original_img, Image.Image):
            # For global patch, maintain aspect ratio
            if patch_size is not None:
                # Calculate new dimensions maintaining aspect ratio
                orig_w, orig_h = original_img.size
                ratio = orig_h / orig_w
                new_w = patch_size[1]  # Use patch width
                new_h = int(new_w * ratio)  # Calculate height to maintain ratio
                global_patch = original_img.resize(
                    (new_w, new_h), Image.Resampling.BILINEAR
                )
            else:
                global_patch = original_img
            patches.append(np.array(global_patch))
        else:
            patches.append(image)

    # Convert back to original input type
    if input_type is torch.Tensor:
        patches = [torch.from_numpy(p).permute(2, 0, 1).float() for p in patches]
    elif input_type is Image.Image:
        patches = [Image.fromarray(p) for p in patches]
    # If np.ndarray, keep as is

    return patches


def pre_mil_gradcam(
    model_tuple: tuple[
        nn.Module,
        tuple[int, int],
        str | None,
        str | None,
        dict[str, list[float]] | None,
    ],
    image_path: str,
    target_layer: str | None = None,
    class_idx: int | None = None,
) -> tuple[nn.Module, torch.Tensor, Image.Image, str, int | None, int, float]:
    # Unpack model and info
    model, input_size, model_name, gradcam_layer, normalize, num_patches, arch_type = (
        model_tuple
    )
    model.eval()

    # Use gradcam_layer if available, else default to 'layer4'
    if target_layer is None:
        target_layer = gradcam_layer if gradcam_layer is not None else "layer4"

    # Use input_size directly as resize_size (expected to be a tuple like (448, 448))
    resize_size = input_size

    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = Image.open(image_path).convert("RGB")

    # Check if this is a MIL/patch-based model
    is_mil_model = num_patches is not None and arch_type is not None

    if is_mil_model:
        # Check if arch_type requires global image (like mil_v4)
        has_global = "v4" in arch_type.lower() or "global" in arch_type.lower()

        # For MIL models: split image into patches (with global if needed)
        patches = split_image_into_patches(
            img, num_patches, resize_size, add_global=has_global
        )

        # print(
        #    f"DEBUG: Split into {len(patches)} patches (num_patches={num_patches}, add_global={has_global})"
        # )

        # Preprocess each patch - ensure each patch is resized to input_size
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),  # Resize each patch to input_size
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        patch_tensors = [preprocess(patch) for patch in patches]

        # Stack patches: shape [N, 3, H, W] or [N+1, 3, H, W] if has_global
        patches_tensor = torch.stack(patch_tensors, dim=0)
        # print(f"DEBUG: Stacked patches tensor shape: {patches_tensor.shape}")

        # Add batch dimension: [1, N, 3, H, W] or [1, N+1, 3, H, W]
        input_tensor = patches_tensor.unsqueeze(0)
        # print(f"DEBUG: Final input_tensor shape: {input_tensor.shape}")
    else:
        # For standard models: single image preprocessing
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize["mean"], std=normalize["std"])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )
        input_tensor = preprocess(img).unsqueeze(0)

    # Predict class probabilities
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        # If class_idx is not specified, use the highest predicted class
        pred_class = (
            class_idx if class_idx is not None else int(output.argmax(dim=1).item())
        )
        pred_prob = probs[pred_class]

    return model, input_tensor, img, target_layer, class_idx, pred_class, pred_prob


def mil_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str,
    class_idx: int | None = None,
) -> ndarray:
    """Generate GradCAM for MIL/patch-based models."""
    activations_list = []
    gradients_list = []

    def forward_hook(module, input, output):
        activations_list.append(output.detach().clone())

    def backward_hook(module, grad_in, grad_out):
        gradients_list.append(grad_out[0].detach().clone())

    layer = dict([*model.named_modules()])[target_layer]
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    num_input_patches = input_tensor.shape[1] if input_tensor.dim() == 5 else 1

    print(f"\n{'=' * 60}")
    print(f"DEBUG mil_gradcam:")
    print(f"  Input patches: {num_input_patches}")
    print(f"  Captured {len(activations_list)} activation groups")

    # Concatenate all activations and gradients from multiple forward passes
    all_acts = []
    all_grads = []
    for acts, grads in zip(activations_list, gradients_list):
        print(f"  Act shape: {acts.shape}, Grad shape: {grads.shape}")
        # If 4D [B, C, H, W], treat as [B, C, H, W] where B=num_patches
        if acts.dim() == 4:
            all_acts.append(acts)
            all_grads.append(grads)
        elif acts.dim() == 5:  # [1, N, C, H, W]
            all_acts.append(acts[0])
            all_grads.append(grads[0])

    # Concatenate along batch dimension
    if len(all_acts) > 1:
        acts = torch.cat(all_acts, dim=0)  # [total_patches, C, H, W]
        grads = torch.cat(all_grads, dim=0)
        print(f"  Concatenated: acts {acts.shape}, grads {grads.shape}")
    else:
        acts = all_acts[0]
        grads = all_grads[0]

    # Ensure we have exactly num_input_patches
    if acts.shape[0] < num_input_patches:
        # Pad with duplicates of last
        repeats_needed = num_input_patches - acts.shape[0]
        acts = torch.cat([acts] + [acts[-1:]] * repeats_needed, dim=0)
        grads = torch.cat([grads] + [grads[-1:]] * repeats_needed, dim=0)
        print(f"  Padded to {num_input_patches} patches")
    elif acts.shape[0] > num_input_patches:
        acts = acts[:num_input_patches]
        grads = grads[:num_input_patches]
        print(f"  Truncated to {num_input_patches} patches")

    # Compute CAM for each patch
    cams = []
    for i in range(num_input_patches):
        g = grads[i]  # [C, H, W]
        a = acts[i]  # [C, H, W]
        weights = g.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * a).sum(dim=0)  # [H, W]
        cam = torch.relu(cam)
        cams.append(cam)
    cam = torch.stack(cams, dim=0)  # [N, H, W]

    print(f"  Output CAM shape: {cam.shape}")
    print(f"{'=' * 60}\n")

    # Normalize
    cam = cam.cpu().numpy()
    normalized_cams = []
    for i in range(cam.shape[0]):
        c = cam[i]
        c_min, c_max = c.min(), c.max()
        if c_max > c_min:
            c = (c - c_min) / (c_max - c_min)
        else:
            c = np.zeros_like(c)
        normalized_cams.append(np.uint8(c * 255))
    cam = np.stack(normalized_cams, axis=0)

    handle_f.remove()
    handle_b.remove()

    return cam
