"""Single place that turns an RGB image + checkpoint settings into the model input.
Used by Grad-CAM and by any batch inference script, so they cannot drift from each other."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.factory import is_mil
from src.utils.common import parse_img_size

from .imaging import foreground_crop, prepare_based_tensor
from .patches import PatchSample, prepare_patches


@dataclass
class ModelInput:
    x: torch.Tensor  # (1, 3, H, W)  or  (1, N+1, 3, H, W) for MIL
    shown: np.ndarray  # uint8 image the geometry refers to (after optional crop / rotation), full resolution
    ps: PatchSample | None  # MIL only: strip boxes etc.
    crop_offset: tuple  # (x0, y0) of the foreground crop in the original image
    width_before_rotation: int  # width of the (cropped) image before the optional 90deg rotation


def build_input(img: np.ndarray, settings: dict, crop_foreground: bool = False) -> ModelInput:
    legacy = bool(settings.get("legacy_preprocess"))
    img_size = parse_img_size(settings["img_size"])
    offset = (0, 0)
    if crop_foreground:
        img, (x0, y0, _, _) = foreground_crop(img)
        offset = (int(x0), int(y0))
    width = img.shape[1]
    if is_mil(settings["arch_type"]):
        ps = prepare_patches(img, img_size, int(settings["num_patches"]), float(settings["overlap_ratio"]),
                             float(settings["local_scale"]), bool(settings["rotate_landscape"]),
                             augment=None, legacy=legacy)
        return ModelInput(ps.tensor[None], ps.image, ps, offset, width)
    return ModelInput(prepare_based_tensor(img, img_size, legacy)[None], img, None, offset, width)


@torch.inference_mode()
def predict_probs(model, settings: dict, img: np.ndarray, device, crop_foreground: bool = False) -> np.ndarray:
    """-> softmax probabilities (C,) for one uint8 RGB image."""
    inp = build_input(img, settings, crop_foreground)
    model.eval()
    return torch.softmax(model(inp.x.to(device)).float(), dim=1)[0].cpu().numpy()
