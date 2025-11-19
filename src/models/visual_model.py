"""
Visual Feature Extraction Model
Uses RegNet (or ResNet as alternative) for extracting visual features from images
"""

import torch
import torch.nn as nn
from typing import Union, List, Tuple
from PIL import Image
import numpy as np
from torchvision.models import ( resnet50, ResNet50_Weights)


class VisualFeatureExtractor(nn.Module):
    """
    Visual feature extraction using pre-trained CNN (ResNet50)
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        feature_dim: int = 512,
        freeze_backbone: bool = True,
        use_pretrained: bool = True,
        device: str | None = None,
    ):

        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.model_name = model_name
        self.use_pretrained = use_pretrained

        # Load backbone and discover its output feature
        self.backbone, in_feats = self._load_backbone(model_name, use_pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Projection head to desired feature_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(in_feats, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
        )

        self.to(self.device)

    def _load_backbone(self, model_name: str, use_pretrained: bool) -> Tuple[nn.Module, int]:

        name = model_name.lower()
        weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        m = resnet50(weights=weights)
        in_feats = m.fc.in_features
        m.fc = nn.Identity()
        return m, in_feats

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # backbone forward returns flattened vectors when fc is Identity (resnet/regnet)
        feats = self.backbone(images)
        if feats.dim() == 4:
            # e.g., VGG path before classifier flatten
            feats = torch.flatten(feats, 1)
        feats = self.feature_projection(feats)
        return feats

    def extract_features_from_path(self, image_path: str) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = self.preprocess_image(image_path).unsqueeze(0).to(self.device)
            return self.forward(x).squeeze(0)

    def extract_features_batch(self, image_paths: List[str]) -> torch.Tensor:
        self.eval()
        tensors = []
        for p in image_paths:
            try:
                tensors.append(self.preprocess_image(p))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))
        if not tensors:
            return torch.empty(0, self.feature_dim)
        batch = torch.stack(tensors, 0).to(self.device)
        with torch.no_grad():
            return self.forward(batch).cpu()


class VisualSentimentModel(nn.Module):
    """Feature extractor + classifier head for visual sentiment."""

    def __init__(
        self,
        num_classes: int = 3,
        feature_dim: int = 512,
        model_name: str = "resnet50",
        freeze_backbone: bool = True,
        use_pretrained: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes

        self.feature_extractor = VisualFeatureExtractor(
            model_name=model_name,
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone,
            use_pretrained=use_pretrained,
            device=self.device,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(images)
        return self.classifier(feats)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.argmax(self.forward(images), dim=1)

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(images), dim=1)

