from .backbone import (
    get_resnet_backbone,
    get_timm_backbone,
    get_fastervit_backbone,
    get_dino_backbone,
)
from .head import get_linear_head
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="dinov2")


def get_based_model(model_type="resnet50", num_classes=2):
    if model_type in ["resnet50", "resnet101", "resnext50"]:
        backbone, feature_dim = get_resnet_backbone(model_type)
        # Replace the head with a linear classifier
        model = backbone
        model.fc = get_linear_head(feature_dim, num_classes)
    elif model_type in [
        "resnest50",
        "resnest101",
        "resnest50s2",
        "regnety",
        "convnextv2base",
        "convnextv2_tiny",
        "efficientnetv2",
        "efficientnetv2s",
        "maxvit_tiny",
        "maxvit_small",
        "maxvit_base",  # thêm maxvit_base
        "eva02_small",  # sửa lại đúng tên model_type
        "eva02_base",  # thêm eva02_base
        "vit_small",  # thêm model mới
        "swinv2_tiny",
        "swinv2_base",
        "swinv2_small",
    ]:
        model, feature_dim = get_timm_backbone(model_type)
        # Replace the head with a linear classifier for all timm backbones
        if hasattr(model, "fc"):
            model.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head") and hasattr(model.head, "fc"):
            model.head.fc = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "head"):
            model.head = get_linear_head(feature_dim, num_classes)
        elif hasattr(model, "classifier"):
            model.classifier = get_linear_head(feature_dim, num_classes)
        else:
            raise ValueError("Unknown head structure for timm backbone")
    elif model_type == "fastervit":
        model, feature_dim = get_fastervit_backbone()
        model.head = get_linear_head(feature_dim, num_classes)
    elif model_type in [
        "dinov2",
        "dinov2_base",
        "dinov2_small",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
    ]:
        # Map "dinov2" to default dinov2_vitb14
        dino_type = "dinov2_s" if model_type == "dinov2" else model_type
        transformer, feature_dim = get_dino_backbone(dino_type)

        class DinoVisionTransformerClassifier(nn.Module):
            def __init__(self, transformer, feature_dim, num_classes):
                super().__init__()
                self.transformer = transformer
                self.feature_dim = feature_dim
                self.classifier = get_linear_head(feature_dim, num_classes)

            def forward(self, x):
                x = self.transformer(x)
                # Some DINOv3 backbones may not have .norm, check before using
                # if hasattr(self.transformer, "norm"):
                #    x = self.transformer.norm(x)
                x = self.classifier(x)
                return x

        model = DinoVisionTransformerClassifier(transformer, feature_dim, num_classes)
        # Freeze backbone only if using dinov3
        if model_type.startswith("dinov3"):
            for param in model.transformer.parameters():
                # param.requires_grad = False
                param.requires_grad = True
    else:
        raise ValueError("Unsupported model_type for base model")
    return model
