"""Self-describing checkpoints.

A checkpoint is {"state_dict": ..., "meta": {...}} where `meta` holds everything needed to
rebuild the model and its preprocessing (arch_type, model_type, num_classes, class_names,
img_size, num_patches, overlap_ratio, local_scale, fusion, ...). Test / Grad-CAM therefore
need only the .pth file. Legacy raw state_dicts (from src_legacy) still load; their settings
must then be passed explicitly.

Loading is strict and raises on mismatch: the original code printed a warning and carried
on with random weights, which silently produced metrics for an untrained model.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .factory import get_model

DEPLOY_FORMAT = "mammosgm-deploy-1"

# keys that define the model / preprocessing; a checkpoint's meta overrides the CLI for these
STRUCTURE_KEYS = (
    "arch_type", "model_type", "num_classes", "fusion",
    "img_size", "num_patches", "overlap_ratio", "local_scale", "rotate_landscape", "legacy_preprocess",
)


def deploy_info(class_names, decision_threshold: float = 0.5, threshold_rule: str = "argmax (p > 0.5)",
                label_names=None, **extra) -> dict:
    """The `meta["deploy"]` block: everything a deployment needs besides the model settings.
    Binary models: positive class = index 1, predicted positive iff p[1] > decision_threshold
    (0.5 reproduces argmax, i.e. the metrics reported by training). Multi-class: argmax."""
    import datetime

    import cv2
    import numpy
    import timm
    import torchvision

    n = len(class_names)
    return {
        "format": DEPLOY_FORMAT,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "positive_class": 1 if n == 2 else None,
        "decision_threshold": float(decision_threshold) if n == 2 else None,
        "threshold_rule": threshold_rule if n == 2 else "argmax",
        "label_names": list(label_names) if label_names else [str(c) for c in class_names],
        "input": "breast ROI crop, the same kind of image as the training data (YOLO breast crop)",
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "timm": str(timm.__version__), "numpy": str(numpy.__version__), "opencv": str(cv2.__version__)},
        **extra,
    }


_REQUIRED = ("arch_type", "model_type", "num_classes", "img_size", "legacy_preprocess", "mean", "std")
_REQUIRED_MIL = ("num_patches", "overlap_ratio", "local_scale", "rotate_landscape")


def validate_meta(meta: dict | None) -> None:
    """Raise if a checkpoint lacks a setting that changes the model or its preprocessing."""
    if not meta:
        raise ValueError("checkpoint has no embedded settings (old bare state_dict): convert it with "
                         "`python -m src.models.legacy` and/or export it with `python -m src.export`")
    missing = [k for k in _REQUIRED if meta.get(k) is None]
    arch = str(meta.get("arch_type", ""))
    if arch.startswith("mil"):
        missing += [k for k in _REQUIRED_MIL if meta.get(k) is None]
    if arch == "mil_v4" and not meta.get("fusion"):
        missing.append("fusion")
    if arch == "mil_v4a" and meta.get("model_kwargs") is None:
        missing.append("model_kwargs")
    if missing:
        raise ValueError(f"checkpoint settings incomplete, missing {missing}; re-export it with `python -m src.export`")


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def plain(obj):
    """Meta -> plain python (dict / list / str / int / float / bool / None) so that the file loads with
    torch.load(weights_only=True). Catches e.g. torch.__version__ (a str subclass), numpy scalars, tuples."""
    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, str):
        return "" + obj  # str subclass -> exact str
    if hasattr(obj, "item") and getattr(obj, "ndim", 0) == 0:  # numpy / torch scalar
        return plain(obj.item())
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    raise TypeError(f"meta value of type {type(obj).__name__} is not allowed in a checkpoint: {obj!r}")


def save_checkpoint(path: str, model: nn.Module, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in _unwrap(model).state_dict().items()}
    torch.save({"state_dict": sd, "meta": plain(meta)}, path)


def read_checkpoint(path: str):
    """-> (state_dict, meta | None)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta")
    return obj, None  # legacy: bare state_dict


def load_weights(model: nn.Module, state_dict: dict, source: str = "checkpoint") -> None:
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    try:
        result = _unwrap(model).load_state_dict(state_dict, strict=False)
    except RuntimeError as e:  # e.g. shape mismatch (wrong backbone) raises even with strict=False
        raise RuntimeError(
            f"{source} does not match the model ({str(e).splitlines()[0]} ...).\n"
            "Check --arch_type / --model_type / --fusion / --num_classes."
        ) from e
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{source} does not match the model.\n"
            f"  missing keys   ({len(result.missing_keys)}): {result.missing_keys[:6]}\n"
            f"  unexpected keys({len(result.unexpected_keys)}): {result.unexpected_keys[:6]}\n"
            "Check --arch_type / --model_type / --fusion / --num_classes."
        )


def build_model_from_checkpoint(path: str, overrides: dict | None = None, device="cpu"):
    """-> (model in eval mode on `device`, settings dict).

    `settings` = overrides, then the checkpoint's meta on top for STRUCTURE_KEYS (and any
    other stored key). Required for legacy checkpoints: arch_type, model_type, num_classes.
    """
    state_dict, meta = read_checkpoint(path)
    settings = {k: v for k, v in (overrides or {}).items() if v is not None}
    if meta:
        settings.update(meta)
    else:
        print(f"[WARN] '{path}' has no embedded settings (legacy checkpoint); using the given arguments.")
    for key in ("arch_type", "model_type", "num_classes"):
        if settings.get(key) is None:
            raise ValueError(f"Cannot rebuild the model: '{key}' is unknown (legacy checkpoint? pass it explicitly)")
    model = get_model(
        settings["arch_type"],
        settings["model_type"],
        int(settings["num_classes"]),
        pretrained=False,  # weights come from the checkpoint; no ImageNet download needed
        fusion=settings.get("fusion") or "fuse",
        **(settings.get("model_kwargs") or {}),
    )
    load_weights(model, state_dict, source=path)
    return model.to(device).eval(), settings
