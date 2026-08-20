from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LMVDSequenceDataset(Dataset):

    def __init__(
        self,
        sample_ids: list[int],
        labels: list[int],
        data_dir: Path,
        max_video: int = 915,
        max_audio: int = 186,
        cache_only: bool = True,
    ):
        from src.data.loaders import load_lmvd_sequence

        self.sample_ids = sample_ids
        self.labels = labels
        self.data_dir = Path(data_dir)
        self.max_video = max_video
        self.max_audio = max_audio
        self.cache_only = cache_only
        self._load = load_lmvd_sequence

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        video, audio = self._load(
            self.sample_ids[idx],
            self.data_dir,
            self.max_video,
            self.max_audio,
            cache_only=self.cache_only,
        )
        return {
            "video": torch.from_numpy(video),
            "audio": torch.from_numpy(audio),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class ModalityLSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x.abs().sum(dim=-1) > 0).float()
        lengths = mask.sum(dim=1).clamp(min=1).cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.dropout(h)


class VideoAudioLSTM(nn.Module):
   
    def __init__(
        self,
        video_dim: int = 171,
        audio_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion: str = "concat",
    ):
        super().__init__()
        self.video_encoder = ModalityLSTM(video_dim, hidden_dim, num_layers, dropout)
        self.audio_encoder = ModalityLSTM(audio_dim, hidden_dim, num_layers, dropout)
        self.fusion = fusion
        fused_dim = self.video_encoder.out_dim + self.audio_encoder.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v_emb = self.video_encoder(video)
        a_emb = self.audio_encoder(audio)
        fused = torch.cat([v_emb, a_emb], dim=-1)
        return self.classifier(fused)

    def encode(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v_emb = self.video_encoder(video)
        a_emb = self.audio_encoder(audio)
        return torch.cat([v_emb, a_emb], dim=-1)


class VisualOnlyLSTM(nn.Module):
    
    def __init__(
        self,
        video_dim: int = 171,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ModalityLSTM(video_dim, hidden_dim, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.embed_dim = self.encoder.out_dim

    def forward(self, video: torch.Tensor, audio: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.encode(video, audio))

    def encode(self, video: torch.Tensor, audio: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(video)


class AudioOnlyLSTM(nn.Module):

    def __init__(
        self,
        audio_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ModalityLSTM(audio_dim, hidden_dim, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.embed_dim = self.encoder.out_dim

    def forward(self, video: torch.Tensor | None, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(video, audio))

    def encode(self, video: torch.Tensor | None, audio: torch.Tensor) -> torch.Tensor:
        return self.encoder(audio)


class VideoAudioGRU(nn.Module):

    def __init__(
        self,
        video_dim: int = 171,
        audio_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.video_gru = nn.GRU(
            video_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.audio_gru = nn.GRU(
            audio_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        fused_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.embed_dim = fused_dim

    def _encode_stream(self, gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
        mask = (x.abs().sum(dim=-1) > 0).float()
        lengths = mask.sum(dim=1).clamp(min=1).cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = gru(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(video, audio))

    def encode(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        v_emb = self._encode_stream(self.video_gru, video)
        a_emb = self._encode_stream(self.audio_gru, audio)
        return self.dropout(torch.cat([v_emb, a_emb], dim=-1))


class AudioConv1D(nn.Module):

    def __init__(
        self,
        audio_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.embed_dim = hidden_dim

    def forward(self, video: torch.Tensor | None, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(video, audio))

    def encode(self, video: torch.Tensor | None, audio: torch.Tensor) -> torch.Tensor:
        x = audio.transpose(1, 2)
        h = self.convs(x).squeeze(-1)
        return self.dropout(h)


def build_video_model(
    architecture: str,
    video_dim: int = 171,
    audio_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    architecture = architecture.lower()
    builders = {
        "video_audio_lstm": lambda: VideoAudioLSTM(
            video_dim, audio_dim, hidden_dim, num_layers, num_classes, dropout,
        ),
        "visual_lstm": lambda: VisualOnlyLSTM(video_dim, hidden_dim, num_layers, num_classes, dropout),
        "audio_lstm": lambda: AudioOnlyLSTM(audio_dim, hidden_dim, num_layers, num_classes, dropout),
        "video_audio_gru": lambda: VideoAudioGRU(
            video_dim, audio_dim, hidden_dim, num_layers, num_classes, dropout,
        ),
        "audio_cnn1d": lambda: AudioConv1D(audio_dim, hidden_dim, num_classes, dropout),
    }
    if architecture not in builders:
        raise ValueError(f"Unknown video architecture: {architecture}. Choose from {list(builders)}")
    return builders[architecture]()


def train_video_lstm(
    model: VideoAudioLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, list[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(video, audio)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history["train_loss"].append(total_loss / max(len(train_loader), 1))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)
                preds = model(video, audio).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        history["val_acc"].append(correct / max(total, 1))
        print(
            f"    epoch {epoch + 1}/{epochs}: "
            f"loss={history['train_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}",
            flush=True,
        )

    return history
