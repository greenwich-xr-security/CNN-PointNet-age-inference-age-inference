from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

from .efficientnet_age import get_default_efficientnet_weights

try:
    from open3d.ml.torch.models import PointNet2
except Exception as exc:  # noqa: BLE001
    PointNet2 = None
    _POINTNET2_IMPORT_ERROR = exc
else:
    _POINTNET2_IMPORT_ERROR = None


def _build_resnet_encoder(backbone: str, latent_dim: int, pretrained: bool = True) -> nn.Module:
    """Create a ResNet encoder that outputs a latent vector."""
    backbone = backbone.lower()
    if backbone not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError(f"Unsupported RGB backbone '{backbone}'.")
    builder = getattr(models, backbone)

    # Handle torchvision weight enums while keeping backward compatibility.
    weights = None
    weights_attr = f"{backbone.upper()}_Weights"
    weights_enum = getattr(models, weights_attr, None)
    if weights_enum is not None:
        weights = getattr(weights_enum, "DEFAULT", None)

    try:
        cnn = builder(weights=weights if pretrained else None)
    except TypeError:
        cnn = builder(pretrained=pretrained)

    in_feats = cnn.fc.in_features
    cnn.fc = nn.Linear(in_feats, latent_dim)
    return cnn


def _build_efficientnet_encoder(variant: str, latent_dim: int, pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet encoder with a latent projection."""
    variant = variant.lower()
    model_name = f"efficientnet_{variant}"
    if not hasattr(models, model_name):
        raise ValueError(f"torchvision.models does not provide '{model_name}'.")
    builder = getattr(models, model_name)

    weights = get_default_efficientnet_weights(variant) if pretrained else None
    try:
        effnet = builder(weights=weights if pretrained else None)
    except TypeError:
        effnet = builder(pretrained=pretrained)

    if isinstance(effnet.classifier, nn.Sequential) and len(effnet.classifier) >= 2:
        in_feats = effnet.classifier[-1].in_features
        effnet.classifier[-1] = nn.Linear(in_feats, latent_dim)
    else:
        in_feats = getattr(effnet.classifier, "in_features", None)
        if in_feats is None:
            raise RuntimeError(f"Unexpected EfficientNet-{variant.upper()} classifier structure")
        effnet.classifier = nn.Linear(in_feats, latent_dim)
    return effnet


class RgbEncoder(nn.Module):
    """RGB encoder using a ResNet or EfficientNet backbone with a projection head."""

    def __init__(self, backbone: str = "resnet18", latent_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.latent_dim = latent_dim
        backbone_lower = backbone.lower()
        if backbone_lower.startswith("efficientnet_"):
            variant = backbone_lower.split("efficientnet_")[1]
            self.backbone = _build_efficientnet_encoder(variant, latent_dim, pretrained=pretrained)
        else:
            self.backbone = _build_resnet_encoder(backbone, latent_dim, pretrained=pretrained)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder wrapper from Open3D that outputs a latent vector."""

    def __init__(
        self,
        latent_dim: int = 256,
        *,
        pointnet_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if PointNet2 is None:
            raise ImportError(
                "open3d PointNet2 unavailable. Install open3d>=0.17 and ensure open3d.ml is importable."
            ) from _POINTNET2_IMPORT_ERROR

        kwargs = {"in_channels": 0, "num_classes": latent_dim, "use_xyz": True}
        if pointnet_kwargs:
            kwargs.update(pointnet_kwargs)
        self.model = PointNet2(**kwargs)
        self.latent_dim = latent_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: Tensor of shape (B, N, 3) or (B, 3, N).
        Returns: Tensor of shape (B, latent_dim).
        """
        if points.dim() != 3:
            raise ValueError(f"Point cloud tensor must be 3D, got shape {points.shape}.")
        if points.shape[-1] != 3 and points.shape[1] != 3:
            raise ValueError(f"Expected XYZ coordinates in last or channel dim, got shape {points.shape}.")

        if points.shape[1] == 3:
            points = points.transpose(1, 2)  # (B, 3, N)

        logits = self.model(points)
        if isinstance(logits, dict):
            # Open3D returns dict for some tasks; prefer logits/cls_logits keys.
            logits = logits.get("logits", logits.get("cls_logits", None))
        if logits is None:
            raise RuntimeError("PointNet2 forward returned None; check input formatting.")
        return logits


class ProbabilisticRegressionHead(nn.Module):
    """Outputs mean and log-variance for probabilistic regression."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor):
        preds = self.net(x)
        mean, log_var = preds.chunk(2, dim=1)
        return mean.squeeze(1), log_var.squeeze(1)


@dataclass
class FusionConfig:
    use_rgb: bool = True
    use_point_cloud: bool = True
    rgb_backbone: str = "resnet18"  # supports resnet18/34/50 or efficientnet_b0..b7
    rgb_latent_dim: int = 256
    pc_latent_dim: int = 256
    pretrained_rgb: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.1
    pointnet_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if not (self.use_rgb or self.use_point_cloud):
            raise ValueError("At least one modality must be enabled (use_rgb or use_point_cloud).")


class MultimodalAgeRegressor(nn.Module):
    """
    Combines RGB and/or point cloud encoders, concatenates latents, and predicts mean/log-variance.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.config.validate()

        input_dim = 0
        if config.use_rgb:
            self.rgb_encoder = RgbEncoder(
                backbone=config.rgb_backbone,
                latent_dim=config.rgb_latent_dim,
                pretrained=config.pretrained_rgb,
            )
            input_dim += config.rgb_latent_dim
        else:
            self.rgb_encoder = None

        if config.use_point_cloud:
            self.pc_encoder = PointNet2Encoder(
                latent_dim=config.pc_latent_dim,
                pointnet_kwargs=config.pointnet_kwargs,
            )
            input_dim += config.pc_latent_dim
        else:
            self.pc_encoder = None

        self.head = ProbabilisticRegressionHead(
            input_dim=input_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.head_dropout,
        )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
    ):
        latents = []
        if self.rgb_encoder is not None:
            if images is None:
                raise ValueError("RGB encoder enabled but images not provided.")
            latents.append(self.rgb_encoder(images))

        if self.pc_encoder is not None:
            if points is None:
                raise ValueError("Point cloud encoder enabled but points not provided.")
            latents.append(self.pc_encoder(points))

        fused = latents[0] if len(latents) == 1 else torch.cat(latents, dim=1)
        return self.head(fused)


def build_age_model(
    use_rgb: bool = True,
    use_point_cloud: bool = True,
    **kwargs,
) -> MultimodalAgeRegressor:
    """
    Convenience builder for MultimodalAgeRegressor.

    kwargs can override FusionConfig fields, e.g. rgb_backbone, rgb_latent_dim, pc_latent_dim.
    """
    cfg = FusionConfig(use_rgb=use_rgb, use_point_cloud=use_point_cloud, **kwargs)
    return MultimodalAgeRegressor(cfg)
