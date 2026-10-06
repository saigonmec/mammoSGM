"""MammoModel: everything needed to run a trained model — only this folder + the weight file.

    from deploy import MammoModel
    m = MammoModel("best.pth")                    # device: cuda if available, else cpu
    r = m.predict("breast_crop.png")              # path, PIL image or numpy array (8/16-bit, gray or RGB)
    r["positive_prob"], r["pred"], r["label"]
    r = m.predict(full_image, roi=(x0, y0, x1, y1), explain=True)   # roi from the breast detector
    r["heatmap"]                                  # float32 (H, W) in [0, 1], same size as the input image

All settings (architecture, input size, strips, preprocessing, threshold, class names) are read from the
weight file (`meta`). The model code and preprocessing are verbatim copies of src/ (deploy/core), so the
predictions are the ones measured during training (tests/test_deploy.py checks this).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from .core.data.imaging import MEAN, STD, as_rgb_uint8, foreground_crop, load_rgb, prepare_based_tensor
from .core.data.patches import prepare_patches
from .core.gradcam.cam import CAMExtractor, default_target_layer
from .core.gradcam.viz import normalize_cams, resize_map, stitch_local_cams
from .core.models.checkpoint import build_model_from_checkpoint, read_checkpoint, validate_meta
from .core.models.factory import is_mil

BLACK_LEVEL = 8  # pixels darker than this count as background in the "is this cropped?" check
MAX_BACKGROUND = 0.5  # a breast ROI crop is mostly tissue; a full-field mammogram is mostly black


class MammoModel:
    def __init__(self, weights: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        _, meta = read_checkpoint(weights)
        validate_meta(meta)  # raises if a setting that changes the model / preprocessing is missing
        if [float(v) for v in meta["mean"]] != list(MEAN) or [float(v) for v in meta["std"]] != list(STD):
            raise ValueError(f"weight was trained with mean/std {meta['mean']}/{meta['std']}, deploy normalises "
                             f"with {MEAN}/{STD}: re-sync deploy/ (tools/sync_deploy.py)")
        self.model, self.settings = build_model_from_checkpoint(weights, None, self.device)
        self.weights = weights
        self.mil = is_mil(self.settings["arch_type"])
        self.num_classes = int(self.settings["num_classes"])
        d = self.settings.get("deploy") or {}
        if not d:
            warnings.warn("weight has no 'deploy' block: decision threshold 0.5 / generic class names assumed "
                          "(export it with `python -m src.export`)")
        self.positive_class = d.get("positive_class", 1 if self.num_classes == 2 else None)
        self.threshold = d.get("decision_threshold", 0.5 if self.num_classes == 2 else None)
        names = d.get("label_names") or [str(c) for c in (self.settings.get("class_names") or range(self.num_classes))]
        if len(names) != self.num_classes:
            raise ValueError(f"{len(names)} label names for {self.num_classes} classes")
        self.label_names = list(names)
        self._check_versions(d.get("versions") or {})

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        s = self.settings
        out = {k: s.get(k) for k in ("arch_type", "model_type", "num_classes", "img_size", "num_patches",
                                     "overlap_ratio", "local_scale", "rotate_landscape", "legacy_preprocess",
                                     "fusion", "model_kwargs", "target_column", "epoch", "monitor", "best_score")}
        out.update(weights=os.path.basename(self.weights), label_names=self.label_names,
                   positive_class=self.positive_class, decision_threshold=self.threshold, device=str(self.device),
                   deploy=s.get("deploy"))
        return out

    @staticmethod
    def _check_versions(saved: dict) -> None:
        import timm
        import torchvision

        now = {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__}
        for k, v in now.items():
            if saved.get(k) and saved[k].split(".")[:2] != v.split(".")[:2]:
                warnings.warn(f"{k} {v} differs from the training version {saved[k]}: re-validate the outputs")

    # ------------------------------------------------------------------ input
    @staticmethod
    def _load(image) -> np.ndarray:
        if isinstance(image, (str, os.PathLike)):
            return load_rgb(str(image))
        return as_rgb_uint8(image)

    @staticmethod
    def _crop(img: np.ndarray, roi, crop: str):
        H, W = img.shape[:2]
        if roi is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in roi)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise ValueError(f"roi {roi} is empty / too small for an image of {W}x{H}")
            return img[y0:y1, x0:x1], (x0, y0)
        if crop == "foreground":
            c, (x0, y0, _, _) = foreground_crop(img)
            return c, (int(x0), int(y0))
        if crop not in (None, "none"):
            raise ValueError("crop must be 'none' or 'foreground'")
        if (img.max(axis=2) < BLACK_LEVEL).mean() > MAX_BACKGROUND:
            warnings.warn("input looks like a full-field mammogram (mostly black background) but the model expects "
                          "a breast ROI crop as in training: pass roi=(x0, y0, x1, y1) from the breast detector "
                          "(or crop='foreground' as an approximation)")
        return img, (0, 0)

    def _model_input(self, img: np.ndarray):
        s = self.settings
        size = (int(s["img_size"][0]), int(s["img_size"][1]))
        legacy = bool(s["legacy_preprocess"])
        if self.mil:
            ps = prepare_patches(img, size, int(s["num_patches"]), float(s["overlap_ratio"]), float(s["local_scale"]),
                                 bool(s["rotate_landscape"]), augment=None, legacy=legacy)
            return ps.tensor[None], ps
        return prepare_based_tensor(img, size, legacy)[None], None

    def _decide(self, probs: np.ndarray) -> int:
        if self.positive_class is not None and self.num_classes == 2:
            pc = int(self.positive_class)
            return pc if probs[pc] > self.threshold else 1 - pc
        return int(np.argmax(probs))

    # ------------------------------------------------------------------ public API
    def predict(self, image, roi=None, crop: str = "none", explain: bool = False, explain_class: int | None = None) -> dict:
        """image: path / PIL / numpy. roi: (x0, y0, x1, y1) breast box in image pixels (optional).
        explain=True adds "heatmap" (Grad-CAM of `explain_class`, default the positive class, in the input
        image's pixel grid, 0 outside the roi); MIL models also get "details" and V4a an "attention" map."""
        full = self._load(image)
        img, (ox, oy) = self._crop(full, roi, crop)
        x, ps = self._model_input(img)
        x = x.to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        pred = self._decide(probs)
        out = {
            "probs": {name: float(p) for name, p in zip(self.label_names, probs)},
            "positive_prob": None if self.positive_class is None else float(probs[int(self.positive_class)]),
            "pred": pred,
            "label": self.label_names[pred],
            "threshold": self.threshold,
            "roi": (ox, oy, ox + img.shape[1], oy + img.shape[0]),
        }
        if explain:
            cls = explain_class if explain_class is not None else (
                int(self.positive_class) if self.positive_class is not None else pred)
            maps = self._explain_mil(x, ps, cls) if self.mil else self._explain_based(x, img.shape[:2], cls)
            for k, v in maps.items():
                if isinstance(v, np.ndarray) and v.ndim == 2:  # paste into the input image's pixel grid
                    canvas = np.zeros(full.shape[:2], np.float32)
                    canvas[oy:oy + v.shape[0], ox:ox + v.shape[1]] = v
                    out[k] = canvas
                else:
                    out[k] = v
            out["explained_class"] = self.label_names[cls]
        return out

    # ------------------------------------------------------------------ explanations
    def _explain_based(self, x, hw, cls) -> dict:
        with CAMExtractor(self.model, default_target_layer(self.model)) as ex:
            res = ex(x, class_idx=cls)
        cam = normalize_cams(res.cams, per_map=True)[0]  # same as src/gradcam/gradcam_run.explain_image
        return {"heatmap": resize_map(cam, hw[0], hw[1])}

    def _explain_mil(self, x, ps, cls) -> dict:
        """The reference MIL Grad-CAM of src/gradcam/gradcam_mil.py (strips + global image on one scale,
        attribution per unit area, feather-blended strips), at full resolution, un-rotated."""
        n = len(ps.boxes)
        Hd, Wd = ps.image.shape[:2]
        sy = Hd / ps.work_hw[0]
        boxes = [(int(round(y0 * sy)), int(round(y1 * sy))) for y0, y1 in ps.boxes]
        with torch.no_grad():
            _, aux = self.model(x, return_aux=True)
        with CAMExtractor(self.model, default_target_layer(self.model)) as ex:
            res = ex(x, class_idx=cls)
        hc, wc = res.cams.shape[-2:]
        frac = np.array([max(y1 - y0, 1) / Hd for y0, y1 in boxes], np.float32)
        dens_local = res.cams[:n] * (hc * wc) / frac[:, None, None]
        dens_global = res.cams[n] * (hc * wc)
        combined = stitch_local_cams(dens_local, boxes, (Hd, Wd)) + resize_map(dens_global, Hd, Wd)
        combined = np.maximum(combined, 0)
        combined = combined / combined.max() if combined.max() > 0 else np.zeros_like(combined)

        def unrotate(m):
            return np.ascontiguousarray(np.rot90(m, k=-1)) if ps.rotated else m

        loc, glo = float(res.cams[:n].sum()), float(res.cams[n].sum())
        out = {
            "heatmap": unrotate(combined.astype(np.float32)),
            "details": {
                "strip_attention": [float(a) for a in aux["attn"][0].cpu()],
                "fusion_weights": None if aux.get("fusion_weights") is None else [float(v) for v in aux["fusion_weights"][0].cpu()],
                "local_share_of_heat": loc / (loc + glo) if loc + glo > 0 else None,
            },
        }
        ac = aux.get("attn_cells")
        if ac is not None and ac.dim() == 4:  # V4a: the model's own cell attention, per unit area
            a = ac[0].float().cpu().numpy()
            att = stitch_local_cams(a * a.shape[1] * a.shape[2] / frac[:, None, None], boxes, (Hd, Wd))
            out["attention"] = unrotate((att / att.max() if att.max() > 0 else att).astype(np.float32))
        return out
