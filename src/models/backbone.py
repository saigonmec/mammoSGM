"""Backbone registry (torchvision ResNets + timm models).

`get_backbone(model_type, num_classes, feature_maps)`:
  num_classes == 0 -> feature extractor returning a pooled (B, feature_dim) vector (MILv4)
  num_classes  > 0 -> full classifier with a fresh linear head (the plain "based" model)
  feature_maps=True -> the last feature map (B, C, h, w) for CNNs / tokens (B, T, C) for ViTs (MILv4a,
                       which treats every cell as a MIL instance); module names of the CNN body are kept
                       (`layer4`, `body.stages.3`) so Grad-CAM finds its target layer.
"""

from __future__ import annotations

import timm
import torch.nn as nn
import torchvision.models as tvm

# model_type -> (torchvision constructor name, weights enum name)
TORCHVISION = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
    "resnext50": ("resnext50_32x4d", "ResNeXt50_32X4D_Weights"),
}

# model_type -> (timm name, extra create_model kwargs)
TIMM = {
    "resnest50": ("resnest50d", {}),
    "resnest101": ("resnest101e", {}),
    "resnest50s2": ("resnest50d_4s2x40d", {}),
    "regnety": ("regnety_080_tv", {}),
    "convnextv2_tiny": ("convnextv2_tiny.fcmae_ft_in22k_in1k", {}),
    "convnextv2base": ("convnextv2_base.fcmae_ft_in22k_in1k", {}),
    "efficientnetv2": ("efficientnetv2_rw_m.agc_in1k", {}),
    "efficientnetv2s": ("efficientnetv2_rw_s.ra2_in1k", {}),
    "maxvit_tiny": ("maxvit_tiny_tf_224.in1k", {}),  # fixed 224x224 input
    "maxvit_small": ("maxvit_small_tf_224.in1k", {}),
    "maxvit_base": ("maxvit_base_tf_224.in1k", {}),
    "eva02_small": ("eva02_small_patch14_224.mim_in22k", {"dynamic_img_size": True}),
    "eva02_base": ("eva02_base_patch14_448.mim_in22k_ft_in1k", {"dynamic_img_size": True}),
    "vit_small": ("vit_small_patch14_reg4_dinov2.lvd142m", {"dynamic_img_size": True}),
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", {"dynamic_img_size": True}),
    "dinov2_base": ("vit_base_patch14_dinov2.lvd142m", {"dynamic_img_size": True}),
    "swinv2_tiny": ("swinv2_tiny_window8_256.ms_in1k", {}),
    "swinv2_small": ("swinv2_small_window8_256.ms_in1k", {}),
    "swinv2_base": ("swinv2_base_window8_256.ms_in1k", {}),
    "mambaout_tiny": ("mambaout_tiny.in1k", {}),
}

SUPPORTED = sorted(list(TORCHVISION) + list(TIMM))


class ResNetFeatureMaps(nn.Module):
    """torchvision ResNet without avgpool/fc: forward -> (B, C, H/32, W/32). Keeps the layer names."""

    def __init__(self, r: nn.Module):
        super().__init__()
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(r, name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class TimmFeatureMaps(nn.Module):
    """timm model restricted to forward_features: (B, C, h, w) for CNNs, tokens (B, T, C) for ViTs."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.body = m

    def forward(self, x):
        return self.body.forward_features(x)


def get_backbone(model_type: str, num_classes: int = 0, pretrained: bool = True, feature_maps: bool = False):
    """-> (model, feature_dim)."""
    if feature_maps and num_classes:
        raise ValueError("feature_maps=True returns features, not class logits (use num_classes=0)")
    if model_type in TORCHVISION:
        ctor, weights_name = TORCHVISION[model_type]
        weights = getattr(tvm, weights_name).IMAGENET1K_V1 if pretrained else None
        model = getattr(tvm, ctor)(weights=weights)
        feature_dim = model.fc.in_features
        if feature_maps:
            return ResNetFeatureMaps(model), feature_dim
        model.fc = nn.Identity() if num_classes == 0 else nn.Linear(feature_dim, num_classes)
        return model, feature_dim
    if model_type in TIMM:
        name, kwargs = TIMM[model_type]
        # num_classes=0 drops the classifier but keeps pooling / pre-logits norm, which is
        # exactly what the original per-model head surgery produced.
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)
        if feature_maps:
            return TimmFeatureMaps(model), int(model.num_features)  # channels of forward_features
        # head_hidden_size == output dim of the pooled/pre-logits features (differs from num_features
        # for models with an MLP head such as MambaOut); older timm only has num_features
        return model, int(getattr(model, "head_hidden_size", None) or model.num_features)
    raise ValueError(f"Unsupported model_type '{model_type}'. Supported: {', '.join(SUPPORTED)}")
