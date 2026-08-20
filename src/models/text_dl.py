from __future__ import annotations

import torch
import torch.nn as nn


class TextMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.embed_dim = hidden_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TextCNN(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        channels: int = 64,
        seq_len: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.expand = nn.Linear(input_dim, seq_len * input_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, channels, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, num_classes)
        self.embed_dim = channels * 3

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seq = self.expand(x).view(batch, self.seq_len, self.input_dim).transpose(1, 2)
        feats = [torch.relu(conv(seq)).max(dim=2).values for conv in self.convs]
        return self.dropout(torch.cat(feats, dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._features(x)


class TextBiLSTM(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        seq_len: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.expand = nn.Linear(input_dim, seq_len * input_dim)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.embed_dim = hidden_dim * 2

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seq = self.expand(x).view(batch, self.seq_len, -1)
        _, (hidden, _) = self.lstm(seq)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dropout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)


class TextGRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        seq_len: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.expand = nn.Linear(input_dim, seq_len * input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.embed_dim = hidden_dim * 2

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seq = self.expand(x).view(batch, self.seq_len, -1)
        _, hidden = self.gru(seq)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dropout(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)


def build_text_dl_model(
    architecture: str,
    input_dim: int,
    num_classes: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.3,
) -> nn.Module:
    architecture = architecture.lower()
    builders = {
        "text_mlp": lambda: TextMLP(input_dim, hidden_dim, num_classes, dropout),
        "text_cnn": lambda: TextCNN(input_dim, num_classes, dropout=dropout),
        "text_bilstm": lambda: TextBiLSTM(input_dim, hidden_dim // 2, num_classes=num_classes, dropout=dropout),
        "text_gru": lambda: TextGRU(input_dim, hidden_dim // 2, num_classes=num_classes, dropout=dropout),
    }
    if architecture not in builders:
        raise ValueError(f"Unknown text DL architecture: {architecture}. Choose from {list(builders)}")
    return builders[architecture]()
