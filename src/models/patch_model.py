import torch
import torch.nn as nn
from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_fastervit_backbone,
    get_dino_backbone,
)
from .patch_architectures import (
    PatchResNet,
    PatchTransformerClassifier,
    TokenMixerClassifier,
    PatchGlobalLocalClassifier,
    PatchGlobalLocalTokenMixerClassifier,
)
from .mil_py import (
    MILClassifier,
    MILClassifierV2,
    MILClassifierV3,
    MILClassifierV4,  # added
    MILClassifierV5,  # added
    MILClassifierV6,  # added
    MILClassifierV7,  # added v7
    MILClassifierV8,  # added v8
    MILClassifierV9,  # added v9
    MILClassifierV10,  # added v10
)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_patch_model(
    model_type="resnet50", num_classes=2, num_patches=2, arch_type="patch_resnet"
):
    if model_type in ["resnet50", "resnet101", "resnext50"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
    elif model_type in [
        "resnest50",
        "resnest101",
        "resnest50s2",
        "regnety",
        "convnextv2",
        "convnextv2_tiny",
        "efficientnetv2",
    ]:
        backbone, feature_dim = get_timm_backbone(model_type)
    elif model_type == "fastervit":
        backbone, feature_dim = get_fastervit_backbone()
    elif model_type == "dinov2":
        backbone, feature_dim = get_dino_backbone()
        # Nếu muốn dùng kiến trúc đặc biệt cho dinov2, có thể bổ sung ở đây
        # ...existing code...
    else:
        raise ValueError("Unsupported model_type for patch model")

    if arch_type == "patch_resnet":
        return PatchResNet(backbone, feature_dim, num_classes, num_patches)
    elif arch_type == "patch_transformer":
        return PatchTransformerClassifier(
            backbone, feature_dim, num_classes, num_patches
        )
    elif arch_type == "token_mixer":
        return TokenMixerClassifier(backbone, feature_dim, num_classes, num_patches)
    elif arch_type == "global_local":
        return PatchGlobalLocalClassifier(
            backbone, feature_dim, num_classes, num_patches
        )
    elif arch_type == "global_local_token":
        return PatchGlobalLocalTokenMixerClassifier(
            backbone, feature_dim, num_classes, num_patches
        )
    elif arch_type == "mil":
        return MILClassifier(backbone, feature_dim, num_classes, num_patches)
    elif arch_type == "mil_v2":
        return MILClassifierV2(backbone, feature_dim, num_classes)
    elif arch_type == "mil_v3":
        return MILClassifierV3(
            base_model_local=backbone,
            base_model_global=backbone,
            local_dim=feature_dim,
            global_dim=feature_dim,
            num_classes=num_classes,
        )
    elif arch_type == "mil_v4":
        return MILClassifierV4(
            # base_model_local=backbone,
            # base_model_global=backbone,
            base_model=backbone,
            feature_dim=feature_dim,
            # local_dim=feature_dim,
            # global_dim=feature_dim,
            num_classes=num_classes,
        )
    elif arch_type == "mil_v5":
        return MILClassifierV5(
            base_model_local=backbone,
            base_model_global=backbone,
            local_dim=feature_dim,
            global_dim=feature_dim,
            num_classes=num_classes,
        )
    elif arch_type == "mil_v6":
        return MILClassifierV6(
            base_model_local=backbone,
            base_model_global=backbone,
            local_dim=feature_dim,
            global_dim=feature_dim,
            num_classes=num_classes,
        )
    elif arch_type == "mil_v7":
        return MILClassifierV7(
            base_model=backbone,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout=0.1,
            fusion_dim=512,
        )
    elif arch_type == "mil_v8":
        return MILClassifierV8(
            base_model=backbone,
            feature_dim=feature_dim,
            num_classes=num_classes,
            top_k=5,
            fusion_dim=512,
            dropout=0.3,
        )
    elif arch_type == "mil_v9":
        return MILClassifierV9(
            base_model=backbone,
            feature_dim=feature_dim,
            num_classes=num_classes,
            # max_patches=num_patches,
        )
    elif arch_type == "mil_v10":
        return MILClassifierV10(
            base_model=backbone,
            feature_dim=feature_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported arch_type: {arch_type}")
