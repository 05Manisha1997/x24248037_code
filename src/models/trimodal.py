from __future__ import annotations

import torch
import torch.nn as nn

from src.models.fusion import build_fusion_model
from src.models.image_cnn import ImageEncoder
from src.models.video_lstm import VideoAudioLSTM


class TrimodalFusionNetwork(nn.Module):
    
    def __init__(
        self,
        text_dim: int,
        image_architecture: str = "resnet18",
        video_dim: int = 171,
        audio_dim: int = 128,
        video_hidden: int = 128,
        video_layers: int = 2,
        embed_dim: int = 128,
        fusion_method: str = "attention",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_method = fusion_method.lower()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.image_encoder = ImageEncoder(image_architecture, embed_dim=embed_dim)
        self.video_backbone = VideoAudioLSTM(
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_dim=video_hidden,
            num_layers=video_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.video_proj = nn.Sequential(
            nn.Linear(self.video_backbone.video_encoder.out_dim + self.video_backbone.audio_encoder.out_dim, embed_dim),
            nn.ReLU(),
        )
        self.fusion = build_fusion_model(
            self.fusion_method,
            [embed_dim, embed_dim, embed_dim],
            hidden_dim=embed_dim * 2,
            num_classes=num_classes,
        )
        self.embed_dim = embed_dim

    def encode_modalities(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_emb = self.text_encoder(text)
        image_emb = self.image_encoder(image)
        video_emb = self.video_proj(self.video_backbone.encode(video, audio))
        return text_emb, image_emb, video_emb

    def forward(
        self,
        text: torch.Tensor,
        image: torch.Tensor,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encode_modalities(text, image, video, audio)
        return self.fusion(*embeddings)


def build_trimodal_model(
    fusion_method: str,
    text_dim: int,
    image_architecture: str = "resnet18",
    **kwargs,
) -> TrimodalFusionNetwork:
    return TrimodalFusionNetwork(
        text_dim=text_dim,
        image_architecture=image_architecture,
        fusion_method=fusion_method,
        **kwargs,
    )
