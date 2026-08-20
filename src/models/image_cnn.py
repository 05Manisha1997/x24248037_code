from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_image_model(architecture: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "cnn":
        return SimpleCNN(num_classes=num_classes)

    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if architecture == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


def get_image_backbone_dim(architecture: str) -> int:
    architecture = architecture.lower()
    dims = {
        "cnn": 256,
        "resnet18": 512,
        "resnet50": 2048,
        "efficientnet_b0": 1280,
        "vgg16": 512 * 7 * 7,
        "mobilenet_v2": 1280,
    }
    if architecture not in dims:
        raise ValueError(f"Unknown architecture: {architecture}")
    return dims[architecture]


class ImageEncoder(nn.Module):

    def __init__(self, architecture: str = "resnet18", embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        self.architecture = architecture.lower()
        backbone_dim = get_image_backbone_dim(self.architecture)
        base = build_image_model(self.architecture, num_classes=2, pretrained=pretrained)

        if self.architecture == "cnn":
            self.features = base.features
            self.pool = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, backbone_dim))
        elif self.architecture in ("efficientnet_b0", "mobilenet_v2"):
            self.features = base.features
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        elif self.architecture == "vgg16":
            self.features = base.features
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.Flatten())
        else:
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.proj = nn.Sequential(nn.Linear(backbone_dim, embed_dim), nn.ReLU())
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.pool(self.features(x))
        return self.proj(feats)


class ImageFeatureExtractor(nn.Module):
    
    def __init__(self, architecture: str = "resnet18"):
        super().__init__()
        base = build_image_model(architecture, num_classes=512)
        if architecture.lower() == "cnn":
            self.backbone = base.features
            self.proj = nn.Linear(128 * 4 * 4, 512)
        else:
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return self.proj(feats)
