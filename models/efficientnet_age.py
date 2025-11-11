from __future__ import annotations

import torch.nn as nn
from torchvision import models

EFFICIENTNET_IMG_SIZES = {
    "b0": 224,
    "b1": 240,
    "b2": 260,
    "b3": 300,
    "b4": 380,
    "b5": 456,
    "b6": 528,
    "b7": 600,
}


def get_default_efficientnet_weights(variant: str):
    """Resolve the torchvision weights enum for the requested EfficientNet variant."""
    weights_enum_name = f"EfficientNet_{variant.upper()}_Weights"
    weights_enum = getattr(models, weights_enum_name, None)
    if weights_enum is None:
        return None
    default_weights = getattr(weights_enum, "DEFAULT", None)
    if default_weights is not None:
        return default_weights
    try:
        return next(iter(weights_enum))
    except TypeError:
        return None


class EfficientNetAgeRegressor(nn.Module):
    """EfficientNet backbone that predicts age mean/log-variance pairs."""

    def __init__(self, variant: str):
        super().__init__()
        variant = variant.lower()
        if variant not in EFFICIENTNET_IMG_SIZES:
            raise ValueError(f"Unsupported EfficientNet variant '{variant}'.")

        model_name = f"efficientnet_{variant}"
        if not hasattr(models, model_name):
            raise ValueError(f"torchvision.models does not provide '{model_name}'.")

        backbone_builder = getattr(models, model_name)

        weights = get_default_efficientnet_weights(variant)
        try:
            if weights is not None:
                backbone = backbone_builder(weights=weights)
            else:
                backbone = backbone_builder(pretrained=True)
        except TypeError:
            backbone = backbone_builder(pretrained=True)

        self.backbone = backbone
        self.variant = variant
        # Replace the classifier to output mean and log-variance (2 values)
        if isinstance(self.backbone.classifier, nn.Sequential) and len(self.backbone.classifier) >= 2:
            in_feats = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_feats, 2)
        else:
            # Fallback: handle unexpected classifier structure
            in_feats = getattr(self.backbone.classifier, "in_features", None)
            if in_feats is None:
                raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")
            self.backbone.classifier = nn.Linear(in_feats, 2)

    def forward(self, x):
        preds = self.backbone(x)
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        mean, log_var = preds.chunk(2, dim=1)
        return mean.squeeze(1), log_var.squeeze(1)
