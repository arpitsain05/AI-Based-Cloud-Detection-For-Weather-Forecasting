import torch.nn as nn
from torchvision import models

try:
    from src.config import NUM_CLASSES
except ImportError:
    from config import NUM_CLASSES


def build_swin_transformer(num_classes=NUM_CLASSES, pretrained=True, dropout=0.2, freeze_backbone=False):
    """
    Build a Swin Transformer (swin_t) transfer-learning model.
    """
    if pretrained:
        try:
            weights = models.Swin_T_Weights.DEFAULT
            model = models.swin_t(weights=weights)
        except AttributeError:
            model = models.swin_t(pretrained=True)
    else:
        try:
            model = models.swin_t(weights=None)
        except TypeError:
            model = models.swin_t(pretrained=False)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model
