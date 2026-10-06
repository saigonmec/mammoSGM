"""Strip geometry and the eval-time MIL preprocessing (pure numpy / cv2 / torch).

N overlapping vertical strips + the whole image as the last item: a sample is a tensor of shape
(N+1, 3, H, W), items [0..N-1] = local strips top -> bottom, item N = global (full) image.
`prepare_patches` is the single code path used by the dataset, Grad-CAM and deploy/.
Dependency-free on purpose: deploy/ vendors this file verbatim (tools/sync_deploy.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .imaging import normalize_image, resize_to, resize_to_width, rotate_if_landscape


def _legacy_boxes(height: int, num_patches: int, overlap_ratio: float):
    """Strip geometry of src_legacy (kept to reproduce weights trained with it): the last strip is
    pinned to the bottom edge whatever the stride, so 3+ strips leave a gap and 2 strips don't overlap."""
    ph = height // num_patches
    step = int(ph * (1 - overlap_ratio))
    if num_patches == 1 or step <= 0:
        return [(height - ph, height)] if num_patches == 1 else [(0, ph)]
    starts = [i * step for i in range(num_patches - 1)] + [height - ph]
    return [(s, s + ph) if i < num_patches - 1 else (height - ph, height) for i, s in enumerate(starts)]


def patch_boxes(height: int, num_patches: int, overlap_ratio: float = 0.2, legacy: bool = False):
    """[(y0, y1)] for `num_patches` equally sized strips covering the whole height.

    Strip height is chosen so that consecutive strips overlap by (at least) `overlap_ratio` of their
    height and the first/last strips touch the top/bottom edge. (The original code forced
    the last strip to the bottom edge regardless of the stride, which left an unseen gap
    for 3+ patches and zero overlap for 2.)
    """
    if num_patches < 1:
        raise ValueError("num_patches must be >= 1")
    if legacy:
        return _legacy_boxes(height, num_patches, overlap_ratio)
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if num_patches == 1:
        return [(0, height)]
    ph = height / (1.0 + (num_patches - 1) * (1.0 - overlap_ratio))
    ph = int(min(max(math.ceil(ph - 1e-9), 1), height))  # ceil: rounding down could leave 1px gaps
    starts = np.round(np.linspace(0, height - ph, num_patches)).astype(int)
    return [(int(s), int(s) + ph) for s in starts]


@dataclass
class PatchSample:
    tensor: torch.Tensor  # (N+1, 3, H, W), local patches first, global image last
    boxes: list  # [(y0, y1)] of the N local strips in `work_hw` coordinates
    work_hw: tuple  # (H, W) of the image the strips were cut from
    rotated: bool  # image was rotated 90deg (landscape -> portrait)
    image: np.ndarray  # uint8 RGB image after the optional rotation (original resolution)


def prepare_patches(
    img: np.ndarray,
    img_size,
    num_patches: int,
    overlap_ratio: float = 0.2,
    local_scale: float = 1.0,
    rotate_landscape: bool = True,
    augment=None,
    legacy: bool = False,
) -> PatchSample:
    """uint8 RGB image -> PatchSample.

    The image is first resized to width `img_w * local_scale` (aspect kept). Strips are cut
    from that working image and each is resized to (img_h, img_w); the global item is the
    whole working image resized the same way. With local_scale > 1 the local patches carry
    more detail than the global view (with 1.0 they are just crops of the same pixels).
    """
    h, w = int(img_size[0]), int(img_size[1])
    rotated = False
    if rotate_landscape:
        img, rotated = rotate_if_landscape(img)
    work = resize_to_width(img, max(1, int(round(w * local_scale))), legacy)
    if augment is not None:
        work = augment(image=work)["image"]
    work_hw = work.shape[:2]
    boxes = patch_boxes(work_hw[0], num_patches, overlap_ratio, legacy)
    items = [normalize_image(resize_to(work[y0:y1], h, w, legacy)) for y0, y1 in boxes]
    items.append(normalize_image(resize_to(work, h, w, legacy)))
    return PatchSample(torch.stack(items), boxes, tuple(work_hw), rotated, img)
