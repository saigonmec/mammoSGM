"""Model factory: arch_type -> nn.Module.

arch_type:
  "based"  : backbone + linear head on one resized image          (input (B, 3, H, W))
  "mil_v4" : attention-MIL over local patches + global image      (input (B, N+1, 3, H, W))
  "mil_v4a": GMIC with fixed strips: feature-map cells as instances, saliency top-t pooling,
             cell attention, deep supervision, branch dropout (see mil_py.py)
"""

from __future__ import annotations

from .backbone import get_backbone
from .mil_py import MILClassifierV4, MILClassifierV4a

ARCH_TYPES = ("based", "mil_v4", "mil_v4a")


def is_mil(arch_type: str) -> bool:
    return arch_type.startswith("mil")


def get_model(
    arch_type: str,
    model_type: str,
    num_classes: int,
    pretrained: bool = True,
    fusion: str = "fuse",
    **mil_kwargs,
):
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2 (got {num_classes}); training uses cross-entropy")
    if arch_type == "based":
        model, _ = get_backbone(model_type, num_classes=num_classes, pretrained=pretrained)
        return model
    if arch_type == "mil_v4":
        backbone, feature_dim = get_backbone(model_type, num_classes=0, pretrained=pretrained)
        return MILClassifierV4(backbone, feature_dim, num_classes=num_classes, fusion=fusion, **mil_kwargs)
    if arch_type == "mil_v4a":
        backbone, feature_dim = get_backbone(model_type, num_classes=0, pretrained=pretrained, feature_maps=True)
        return MILClassifierV4a(backbone, feature_dim, num_classes=num_classes, **mil_kwargs)
    raise ValueError(f"Unsupported arch_type '{arch_type}'. Choose from {ARCH_TYPES}")
