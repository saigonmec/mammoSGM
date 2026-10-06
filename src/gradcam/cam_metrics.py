"""Numbers for comparing two heat-maps, or a heat-map with lesion boxes. numpy / cv2 only.

All maps are 2-D float arrays; `similarity` resizes the second one to the first one's shape.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.metrics import _rankdata, roc_auc


def _clean(m: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _shrink(m: np.ndarray, size: int) -> np.ndarray:
    h, w = m.shape
    s = size / max(h, w)
    if s >= 1:
        return m
    return cv2.resize(m, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _corr(a: np.ndarray, b: np.ndarray):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None  # undefined for a constant map
    return float(np.corrcoef(a, b)[0, 1])


def _topk_mask(m: np.ndarray, pct: float) -> np.ndarray:
    k = max(1, int(round(m.size * pct / 100.0)))
    thr = np.partition(m.ravel(), -k)[-k]
    return m >= thr


def _centroid(m: np.ndarray):
    m = np.maximum(m, 0)
    s = m.sum()
    if s <= 0:
        return 0.5, 0.5
    ys, xs = np.mgrid[: m.shape[0], : m.shape[1]]
    return float((m * xs).sum() / s / m.shape[1]), float((m * ys).sum() / s / m.shape[0])


def similarity(a: np.ndarray, b: np.ndarray, size: int = 128) -> dict:
    """How alike two heat-maps are (on a common grid of at most `size` px).

    pearson / spearman : linear / rank correlation of all pixels (None if a map is constant)
    cosine             : cosine similarity of the non-negative parts
    iou_top10 / top20  : overlap of the 10% / 20% hottest pixels
    centroid_dist      : distance between intensity-weighted centroids, in units of the image diagonal
    """
    a = _clean(a)
    b = cv2.resize(_clean(b), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) if a.shape != b.shape else _clean(b)
    a, b = _shrink(a, size), _shrink(b, size)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    fa, fb = a.ravel(), b.ravel()
    out = {"pearson": _corr(fa, fb), "spearman": _corr(_rankdata(fa), _rankdata(fb))}
    na, nb = np.linalg.norm(np.maximum(fa, 0)), np.linalg.norm(np.maximum(fb, 0))
    out["cosine"] = float(np.dot(np.maximum(fa, 0), np.maximum(fb, 0)) / (na * nb)) if na > 0 and nb > 0 else None
    for pct in (10, 20):
        ma, mb = _topk_mask(a, pct), _topk_mask(b, pct)
        out[f"iou_top{pct}"] = float((ma & mb).sum() / max((ma | mb).sum(), 1))
    (xa, ya), (xb, yb) = _centroid(a), _centroid(b)
    out["centroid_dist"] = float(np.hypot(xa - xb, ya - yb) / np.hypot(1.0, 1.0))
    return out


def boxes_to_mask(boxes_xywh, shape) -> np.ndarray:
    """[[x, y, w, h], ...] in the pixel coordinates of a map of `shape` (H, W) -> bool mask."""
    H, W = shape
    mask = np.zeros((H, W), bool)
    for x, y, w, h in boxes_xywh or []:
        x0, y0 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
        x1, y1 = int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def localization(heatmap: np.ndarray, boxes_xywh, size: int = 256) -> dict | None:
    """Does the heat-map point at the annotated lesion(s)?

    energy_in_box : share of the (non-negative) heat that falls inside the boxes
    area_frac     : share of the image covered by the boxes (energy_in_box of a uniform map)
    energy_ratio  : energy_in_box / area_frac  (>1 = better than chance)
    pointing_game : 1 if the hottest pixel lies inside a box
    top10_hit     : share of the 10% hottest pixels that lie inside a box
    auc           : ROC-AUC of the heat as a pixel-wise lesion detector (0.5 = chance)
    Returns None when there are no boxes (or they cover nothing / everything).
    """
    m = np.maximum(_clean(heatmap), 0)
    mask = boxes_to_mask(boxes_xywh, m.shape)
    if size and max(m.shape) > size:  # score on a smaller grid; boxes were built at full size
        s = size / max(m.shape)
        new = (max(1, int(round(m.shape[1] * s))), max(1, int(round(m.shape[0] * s))))
        m = cv2.resize(m, new, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), new, interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask.sum() == 0 or mask.all():
        return None
    total = float(m.sum())
    inside = float(m[mask].sum()) / total if total > 0 else 0.0
    area = float(mask.mean())
    top = _topk_mask(m, 10)
    return {
        "energy_in_box": inside,
        "area_frac": area,
        "energy_ratio": inside / area,
        "pointing_game": float(mask.ravel()[int(np.argmax(m))]),
        "top10_hit": float(mask[top].mean()),
        "auc": roc_auc(mask.ravel(), m.ravel()),
    }
