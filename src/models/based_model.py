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
        "convnextv2",
        "convnextv2_tiny",
        "efficientnetv2",
    ]:
        model, _ = get_timm_backbone(model_type)
        # timm backbone already has num_classes argument in create_model
        model.default_cfg["num_classes"] = num_classes
    elif model_type == "fastervit":
        model, feature_dim = get_fastervit_backbone()
        model.head = get_linear_head(feature_dim, num_classes)
    elif model_type == "dinov2":
        transformer, feature_dim = get_dino_backbone()

        class DinoVisionTransformerClassifier(nn.Module):
            def __init__(self, transformer, feature_dim, num_classes):
                super().__init__()
                self.transformer = transformer
                self.classifier = get_linear_head(feature_dim, num_classes)

            def forward(self, x):
                x = self.transformer(x)
                x = self.transformer.norm(x)
                x = self.classifier(x)
                return x

        model = DinoVisionTransformerClassifier(transformer, feature_dim, num_classes)
    else:
        raise ValueError("Unsupported model_type for base model")
    return model
