from __future__ import annotations

import torch
import torch.nn as nn


class EarlyFusionClassifier(nn.Module):

    def __init__(self, input_dims: list[int], hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        total = sum(input_dims)
        self.net = nn.Sequential(
            nn.Linear(total, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat(embeddings, dim=-1))


class LateFusionClassifier(nn.Module):

    def __init__(self, input_dims: list[int], num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, num_classes),
                )
                for d in input_dims
            ]
        )
        self.weights = nn.Parameter(torch.zeros(len(input_dims)))

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([head(emb) for head, emb in zip(self.heads, embeddings)], dim=0)
        w = torch.softmax(self.weights, dim=0)
        return (logits * w.view(-1, 1, 1)).sum(dim=0)


class AttentionFusionClassifier(nn.Module):

    def __init__(self, input_dims: list[int], hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in input_dims])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        projected = torch.stack([proj(emb) for proj, emb in zip(self.projections, embeddings)], dim=1)
        attended, _ = self.attention(projected, projected, projected)
        pooled = attended.mean(dim=1)
        return self.classifier(pooled)


def build_fusion_model(method: str, input_dims: list[int], hidden_dim: int = 256, num_classes: int = 2) -> nn.Module:
    method = method.lower()
    if method == "early":
        return EarlyFusionClassifier(input_dims, hidden_dim=hidden_dim, num_classes=num_classes)
    if method == "late":
        return LateFusionClassifier(input_dims, num_classes=num_classes, hidden_dim=max(32, hidden_dim // 4))
    if method == "attention":
        return AttentionFusionClassifier(input_dims, hidden_dim=hidden_dim, num_classes=num_classes)
    raise ValueError(f"Unknown fusion method: {method}")
