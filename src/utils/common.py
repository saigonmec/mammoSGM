"""Small shared helpers: config loading, seeding, device selection, size parsing."""

from __future__ import annotations

import gc
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = SRC_DIR / "config"


def load_config(name: str = "config.yaml") -> dict:
    """Load a YAML config. `name` may be a path or a file name inside src/config."""
    path = Path(name)
    if not path.exists():
        fname = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
        path = CONFIG_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {name} (also looked in {CONFIG_DIR})")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if isinstance(cfg, dict) and isinstance(cfg.get("config"), dict):
        cfg = cfg["config"]
    return cfg


def parse_img_size(val, default=None):
    """None | 448 | "448" | "224x320" | (h, w)  ->  (h, w)."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        h, w = val
        return int(h), int(w)
    if isinstance(val, (int, float)):
        s = int(val)
        return s, s
    s = str(val).lower().replace(" ", "")
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    s = int(s)
    return s, s


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str | None = None) -> torch.device:
    if pref in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def num_workers_default() -> int:
    return max(2, min(6, os.cpu_count() or 2))
