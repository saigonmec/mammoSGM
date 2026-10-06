"""Image I/O + the *single* preprocessing path shared by datasets, Grad-CAM and deploy/.

Everything works on uint8 HWC numpy arrays until `normalize_image`, so training,
evaluation and visualisation can never drift apart. Dependency-free on purpose: deploy/
vendors this file verbatim (tools/sync_deploy.py).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def as_rgb_uint8(img) -> np.ndarray:
    """PIL image or numpy array (H, W) / (H, W, 1|3|4), uint8 / uint16 / int / float -> uint8 RGB (H, W, 3).
    16-bit data is rescaled by 65535 (by 255 if its max is <= 255); PIL's convert('RGB') would silently
    saturate it to white. Float arrays must already be in [0, 255]."""
    if isinstance(img, Image.Image):
        if img.mode not in ("I;16", "I;16B", "I;16L", "I"):
            return np.asarray(img.convert("RGB"))
        img = np.asarray(img)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] != 3):
        raise ValueError(f"expected an (H, W) or (H, W, 3) image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.integer):
            a = a / (65535.0 if a.max() > 255 else 255.0) * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr)


def load_rgb(path: str) -> np.ndarray:
    """Load an image file as uint8 RGB (see as_rgb_uint8 for 16-bit handling)."""
    try:
        return as_rgb_uint8(Image.open(path))
    except Exception as e:
        raise RuntimeError(f"Failed to read image '{path}': {e}") from e


def _interp(src_hw, dst_hw) -> int:
    shrinking = dst_hw[0] * dst_hw[1] < src_hw[0] * src_hw[1]
    return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR


def resize_to(img: np.ndarray, height: int, width: int, legacy: bool = False) -> np.ndarray:
    """Resize to exactly (height, width); INTER_AREA when shrinking (no aliasing).
    legacy=True reproduces src_legacy (plain INTER_LINEAR, which aliases on large downscales)."""
    if img.shape[0] == height and img.shape[1] == width:
        return img
    interp = cv2.INTER_LINEAR if legacy else _interp(img.shape[:2], (height, width))
    return cv2.resize(img, (width, height), interpolation=interp)


def resize_to_width(img: np.ndarray, width: int, legacy: bool = False) -> np.ndarray:
    """Resize keeping aspect ratio so that the new width == `width`."""
    h, w = img.shape[:2]
    if w == width:
        return img
    if legacy:  # src_legacy: PIL bilinear (antialiased), height truncated with int()
        return np.asarray(Image.fromarray(img).resize((width, int(h * width / w)), Image.BILINEAR))
    return resize_to(img, max(1, int(round(h * width / w))), width)


def rotate_if_landscape(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rotate 90deg (counter-clockwise) if W > H. Returns (image, rotated)."""
    if img.shape[1] > img.shape[0]:
        return np.ascontiguousarray(np.rot90(img)), True
    return img, False


def rotate_box_ccw(box, orig_w: int):
    """Map an [x, y, w, h] box from an image of width `orig_w` to the same image after
    np.rot90 (counter-clockwise): point (x, y) -> (y, orig_w - 1 - x)."""
    x, y, w, h = box
    return [y, orig_w - x - w, h, w]


def foreground_crop(img: np.ndarray, thresh: float = 0.02, min_occupancy: float = 0.005, margin: float = 0.01):
    """Tight bounding box of the breast in a full-field mammogram with a (near-)black background.
    A row/column belongs to the foreground if more than `min_occupancy` of its pixels exceed
    `thresh`*255 (ignores isolated specks / markers). Returns (crop, (x0, y0, x1, y1))."""
    gray = img.max(axis=2) if img.ndim == 3 else img
    fg = gray > thresh * 255
    rows = np.where(fg.mean(axis=1) > min_occupancy)[0]
    cols = np.where(fg.mean(axis=0) > min_occupancy)[0]
    H, W = gray.shape
    if len(rows) == 0 or len(cols) == 0:
        return img, (0, 0, W, H)
    my, mx = int(margin * H), int(margin * W)
    y0, y1 = max(0, rows[0] - my), min(H, rows[-1] + 1 + my)
    x0, x1 = max(0, cols[0] - mx), min(W, cols[-1] + 1 + mx)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> float32 CHW, (x/255 - mean) / std."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def prepare_based_tensor(img: np.ndarray, img_size, legacy: bool = False) -> torch.Tensor:
    """Eval-time preprocessing of the plain (non-MIL) models."""
    h, w = img_size
    return normalize_image(resize_to(img, h, w, legacy))
