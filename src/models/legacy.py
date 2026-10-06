"""Import weights saved by src_legacy (`*_full.pth`: a pickled nn.Module, or the dict written by
its `run_gradcam`) as new-format checkpoints.

Unpickling can execute arbitrary code, so the file is read with an *allow-listed* unpickler
(torch / torchvision / timm / collections / src_legacy.models only). Classes saved as `src.*` are
mapped to `src_legacy.*`, i.e. the code that produced the weights. Only the state_dict is kept; the
model is rebuilt with the new code, and the conversion is verified numerically.

    python -m src.models.legacy --src tests/sampleSGM --out tests/sampleSGM/converted
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import types

import torch

from .backbone import SUPPORTED
from .checkpoint import load_weights, save_checkpoint
from .factory import get_model

_TORCH_GLOBALS = re.compile(r"^(\w*Storage|Size|device|Tensor|Parameter|dtype|float\d*|double|half|bfloat16|"
                            r"u?int\d*|long|short|bool|complex\d*)$")
_NUMPY_NAMES = {"dtype", "ndarray", "scalar", "_reconstruct", "_frombuffer"}
_BUILTINS = {"set", "frozenset", "tuple", "list", "dict", "int", "float", "str", "bool", "slice", "range",
             "complex", "bytearray", "bytes", "object"}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "src" or module.startswith("src."):
            module = "src_legacy" + module[3:]  # the code that created these files
        ok = (
            (module == "torch" and _TORCH_GLOBALS.match(name))
            or module.startswith(("torch._utils", "torch.storage", "torch._tensor", "torch.nn", "torchvision.models",
                                  "torchvision.ops", "timm.", "src_legacy.models"))
            or (module == "collections" and name in ("OrderedDict", "defaultdict"))
            or (module == "functools" and name == "partial")
            or (module == "argparse" and name == "Namespace")
            or (module in ("builtins", "__builtin__") and name in _BUILTINS)
            or (module.startswith("numpy") and name in _NUMPY_NAMES)
            or module == "_codecs"
        )
        if not ok:
            raise pickle.UnpicklingError(f"blocked global in checkpoint: {module}.{name}")
        return super().find_class(module, name)


_restricted_pickle = types.ModuleType("restricted_pickle")
_restricted_pickle.Unpickler = _RestrictedUnpickler
_restricted_pickle.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()
_restricted_pickle.dump, _restricted_pickle.dumps, _restricted_pickle.loads = pickle.dump, pickle.dumps, pickle.loads
_restricted_pickle.UnpicklingError, _restricted_pickle.HIGHEST_PROTOCOL = pickle.UnpicklingError, pickle.HIGHEST_PROTOCOL
_restricted_pickle.DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL


def load_legacy_object(path: str):
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle)


def _guess_model_type(name: str) -> str:
    for key in sorted(SUPPORTED, key=len, reverse=True):  # longest first: convnextv2_tiny before convnextv2...
        if re.search(rf"(^|_){re.escape(key)}(_|$)", name):
            return key
    raise ValueError(f"cannot infer the backbone from '{name}'")


def convert_legacy(path: str, out_path: str, model_type: str | None = None, verify: bool = True) -> dict:
    """-> meta of the written checkpoint."""
    obj = load_legacy_object(path)
    info = obj if isinstance(obj, dict) and "model" in obj else {}
    old = info.get("model", obj)
    if not isinstance(old, torch.nn.Module):
        raise ValueError(f"{path}: expected a pickled nn.Module or a dict with a 'model' entry, got {type(old)}")
    old.eval()
    stem = os.path.basename(path)
    cls = type(old).__name__

    if cls == "MILClassifierV4":
        arch, fusion = "mil_v4", getattr(old, "fusion", "fuse")
        num_classes = int(old.num_classes)
        m = re.search(r"_p(\d+)(?:_|\.|$)", stem)
        num_patches = int(m.group(1)) if m else info.get("num_patches")
        if num_patches is None:
            raise ValueError(f"{path}: number of patches unknown (no '_pN' in the name)")
    elif cls in ("ResNet", "ConvNeXt", "RegNet", "EfficientNet", "MaxxVit", "Eva", "VisionTransformer", "SwinTransformerV2"):
        arch, fusion, num_patches = "based", None, None
        num_classes = int(next(p for n, p in reversed(list(old.named_parameters())) if p.ndim == 2).shape[0])
    else:
        raise ValueError(f"{path}: architecture '{cls}' is not supported by the new code (only based / MILv4)")

    model_type = model_type or _guess_model_type(stem)
    size = re.search(r"(\d+)x(\d+)", stem)
    img_size = [int(size.group(1)), int(size.group(2))] if size else list(info.get("input_size", (448, 448)))

    new = get_model(arch, model_type, num_classes, pretrained=False, fusion=fusion or "fuse")
    load_weights(new, old.state_dict(), source=path)  # strict
    new.eval()
    if verify:  # the converted model must reproduce the original outputs exactly
        shape = (2, num_patches + 1, 3, 64, 64) if arch == "mil_v4" else (2, 3, 64, 64)
        x = torch.randn(*shape)
        with torch.no_grad():
            diff = (old(x) - new(x)).abs().max().item()
        if diff > 1e-4:
            raise RuntimeError(f"{path}: converted model differs from the original (max |diff| = {diff:.2e})")

    meta = dict(
        version=2, arch_type=arch, model_type=model_type, num_classes=num_classes, class_names=None,
        img_size=img_size, fusion=fusion, num_patches=num_patches, overlap_ratio=0.2, local_scale=1.0,
        rotate_landscape=True, legacy_preprocess=True, source=stem,
    )
    save_checkpoint(out_path, new, meta)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="a *_full.pth file or a folder searched recursively")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--model_type", help="force the backbone instead of parsing it from the file name")
    a = p.parse_args(argv)
    files = [a.src] if os.path.isfile(a.src) else sorted(glob.glob(os.path.join(a.src, "**", "*_full.pth"), recursive=True))
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        out = os.path.join(a.out, os.path.basename(f).replace("_full.pth", ".pth"))
        try:
            m = convert_legacy(f, out, a.model_type)
            print(f"OK   {os.path.basename(f)} -> {m['arch_type']} {m['model_type']} {m['img_size']} "
                  f"patches={m['num_patches']} classes={m['num_classes']}")
        except Exception as e:
            print(f"SKIP {os.path.basename(f)}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


if __name__ == "__main__":
    main()
