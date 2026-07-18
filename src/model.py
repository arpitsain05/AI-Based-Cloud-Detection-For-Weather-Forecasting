import torch.nn as nn
from torchvision import models

try:
    from src.config import MODEL_NAME, NUM_CLASSES
except ImportError:
    from config import MODEL_NAME, NUM_CLASSES


def build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True, dropout=0.2, freeze_backbone=False):
    """
    Build an EfficientNet-B0 transfer-learning model with a project-specific head.
    """
    if pretrained:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=True)
    else:
        try:
            model = models.efficientnet_b0(weights=None)
        except TypeError:
            model = models.efficientnet_b0(pretrained=False)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


try:
    from src.swin_model import build_swin_transformer
except ImportError:
    from swin_model import build_swin_transformer


def build_model(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    pretrained=True,
    dropout=0.2,
    freeze_backbone=False,
):
    """
    Factory for supported model backbones.
    """
    normalized_name = model_name.lower()
    if "swin" in normalized_name:
        return build_swin_transformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    elif normalized_name == "efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: efficientnet_b0, swin_transformer")


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
