"""Heat-map post-processing and figures (numpy / cv2 for the maths, matplotlib for figures)."""

from __future__ import annotations

import cv2
import numpy as np


def otsu_threshold(u8: np.ndarray) -> int:
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(levels * hist) / np.maximum(w0, 1e-12)
    m1 = (np.sum(levels * hist) - np.cumsum(levels * hist)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    between[(w0 == 0) | (w1 == 0)] = -1
    return int(np.argmax(between))


def normalize_cams(cams: np.ndarray, per_map: bool = False) -> np.ndarray:
    """Raw CAMs (K, h, w) -> [0, 1]. per_map=False keeps relative strength across maps (one shared
    maximum); per_map=True stretches each map on its own (every map then looks 'hot')."""
    cams = np.maximum(np.asarray(cams, np.float32), 0)
    if per_map:
        out = np.zeros_like(cams)
        for i, c in enumerate(cams):
            lo, hi = c.min(), c.max()
            out[i] = (c - lo) / (hi - lo) if hi > lo else 0
        return out
    hi = cams.max()
    return cams / hi if hi > 0 else np.zeros_like(cams)


def resize_map(cam01: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(cam01.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def stitch_local_cams(cams01, boxes, display_hw):
    """Paste the per-strip CAMs back into one full-image map. Where strips overlap they are blended
    with complementary linear ramps (no visible seam, and the zero-padding artefacts every CNN shows at
    the border of a crop get ~0 weight). boxes: [(y0, y1)] in display coordinates; gaps stay 0."""
    H, W = display_hw
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    n = len(boxes)
    for i, (cam, (y0, y1)) in enumerate(zip(cams01, boxes)):
        c0, c1 = max(0, y0), min(H, y1)
        if c1 <= c0:
            continue
        h = c1 - c0
        w = np.ones(h, np.float32)
        if i > 0:  # overlap with the previous strip: ramp up
            ov = min(max(boxes[i - 1][1] - y0, 0), h)
            if ov:
                w[:ov] = (np.arange(ov) + 0.5) / ov
        if i < n - 1:  # overlap with the next strip: ramp down
            ov = min(max(y1 - boxes[i + 1][0], 0), h)
            if ov:
                w[h - ov:] = np.minimum(w[h - ov:], 1.0 - (np.arange(ov) + 0.5) / ov)
        num[c0:c1] += w[:, None] * resize_map(cam, h, W)
        den[c0:c1] += w[:, None]
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def overlay(img_u8: np.ndarray, cam01: np.ndarray, alpha: float = 0.5, otsu: bool = False) -> np.ndarray:
    """Blend a [0,1] heat-map (same HxW as the image) onto an RGB uint8 image."""
    cam_u8 = np.uint8(np.clip(cam01, 0, 1) * 255)
    color = cv2.cvtColor(cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB).astype(np.float32)
    base = img_u8.astype(np.float32)
    blended = (1 - alpha) * base + alpha * color
    if otsu:
        mask = (cam_u8 > otsu_threshold(cam_u8))[..., None]
        blended = np.where(mask, blended, base)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _plt(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _draw_boxes(ax, boxes, color="lime"):
    from matplotlib.patches import Rectangle

    for x, y, w, h in boxes or []:
        ax.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"))


def _title(gt, pred, prob):
    t = "Grad-CAM"
    if gt is not None:
        t += f" | GT: {gt}"
    if pred is not None:
        t += f" | Pred: {pred}"
    if prob is not None:
        t += f" ({prob * 100:.1f}%)"
    return t


def _finish(fig, plt, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def render_based(img, cam01, save_path=None, show=False, option=5, alpha=0.5, bbx_list=None,
                 pred=None, prob=None, gt=None):
    """option: 1 overlay | 2 image+heatmap | 3 +blend | 4 +blend(Otsu) | 5 image+heatmap+blend+blend(Otsu)."""
    if option not in (1, 2, 3, 4, 5):
        raise ValueError("option must be 1..5")
    plt = _plt(show)
    H, W = img.shape[:2]
    cam = resize_map(cam01, H, W)
    heat = overlay(np.zeros_like(img), cam, alpha=1.0)
    panels = {
        1: [("overlay", overlay(img, cam, alpha))],
        2: [("image", img), ("heatmap", heat)],
        3: [("image", img), ("heatmap", heat), ("blend", overlay(img, cam, alpha))],
        4: [("image", img), ("heatmap", heat), ("blend (Otsu)", overlay(img, cam, alpha, otsu=True))],
        5: [("image", img), ("heatmap", heat), ("blend", overlay(img, cam, alpha)),
            ("blend (Otsu)", overlay(img, cam, alpha, otsu=True))],
    }[option]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5), squeeze=False)
    for ax, (name, im) in zip(axes[0], panels):
        ax.imshow(im)
        ax.set_title(_title(gt, pred, prob) if name in ("image", "overlay") else name, fontsize=9)
        ax.axis("off")
        if name in ("image", "overlay"):
            _draw_boxes(ax, bbx_list)
    _finish(fig, plt, save_path, show)


def render_mil(img, boxes_disp, local_cams01, global_cam01, attn=None, save_path=None, show=False,
               alpha=0.5, bbx_list=None, pred=None, prob=None, gt=None, fusion_weights=None, combined01=None):
    """Top row: image | stitched local heat-map | combined map (or the local map as Otsu mask) | global-image heat-map.
    Bottom row: every local strip with its own CAM and MIL attention weight."""
    plt = _plt(show)
    H, W = img.shape[:2]
    stitched = stitch_local_cams(local_cams01, boxes_disp, (H, W))
    n = len(boxes_disp)
    ncols = max(4, n)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 11), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(_title(gt, pred, prob), fontsize=9)
    _draw_boxes(axes[0, 0], bbx_list)
    for (y0, y1) in boxes_disp:  # show where the strips are
        axes[0, 0].axhline(y0, color="yellow", lw=0.8, alpha=0.7)
        axes[0, 0].axhline(y1, color="yellow", lw=0.8, alpha=0.7)
    axes[0, 0].set_xlim(-0.5, W - 0.5)  # keep the image panel the same size as its neighbours
    axes[0, 0].set_ylim(H - 0.5, -0.5)
    axes[0, 1].imshow(overlay(img, stitched, alpha))
    fw = "" if fusion_weights is None else f" | fusion weight local={float(fusion_weights[0]):.2f}"
    axes[0, 1].set_title(f"local CAMs (stitched){fw}", fontsize=9)
    if combined01 is not None:  # the reference map of gradcam_mil: local + global on one scale
        axes[0, 2].imshow(overlay(img, combined01, alpha))
        axes[0, 2].set_title("COMBINED map (local + global)", fontsize=9)
    else:
        axes[0, 2].imshow(overlay(img, stitched, alpha, otsu=True))
        axes[0, 2].set_title("local CAMs (Otsu)", fontsize=9)
    gcam = resize_map(global_cam01, H, W)
    axes[0, 3].imshow(overlay(img, gcam, alpha))
    fwg = "" if fusion_weights is None else f" | fusion weight global={float(fusion_weights[1]):.2f}"
    axes[0, 3].set_title(f"global-image CAM{fwg}", fontsize=9)
    for i, (y0, y1) in enumerate(boxes_disp):
        crop = img[max(0, y0):min(H, y1)]
        cam = resize_map(local_cams01[i], crop.shape[0], crop.shape[1])
        axes[1, i].imshow(overlay(crop, cam, alpha))
        a = "" if attn is None else f"  attn={float(attn[i]):.2f}"
        axes[1, i].set_title(f"patch {i + 1}{a}", fontsize=9)
    _finish(fig, plt, save_path, show)
