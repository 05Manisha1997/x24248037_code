"""Shared PyTorch training utilities for thesis deep learning experiments."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    return metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    forward_fn: Callable,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = forward_fn(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    forward_fn: Callable,
) -> dict[str, float]:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        labels = batch["label"].to(device)
        logits = forward_fn(model, batch, device)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy() if probs.shape[1] == 2 else probs.max(dim=1).values.cpu().numpy())
    return _compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def class_weights_from_loader(loader: DataLoader, num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights from a DataLoader of dict batches with 'label'."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for batch in loader:
        labels = batch["label"].numpy() if hasattr(batch["label"], "numpy") else np.asarray(batch["label"])
        for c in range(num_classes):
            counts[c] += np.sum(labels == c)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    forward_fn: Callable,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | None = None,
    class_weight: bool = True,
    selection_metric: str = "f1_macro",
) -> dict[str, list[float] | dict[str, float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = None
    if class_weight:
        weight = class_weights_from_loader(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    history: dict[str, list[float] | dict[str, float]] = {"train_loss": [], "val_acc": []}

    best_metrics: dict[str, float] = {selection_metric: -1.0, "accuracy": 0.0}
    for _ in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device, forward_fn)
        metrics = evaluate_model(model, val_loader, device, forward_fn)
        history["train_loss"].append(loss)
        history["val_acc"].append(metrics["accuracy"])
        score = metrics.get(selection_metric, metrics.get("accuracy", 0.0))
        if score >= best_metrics.get(selection_metric, -1.0):
            best_metrics = metrics

    history["best"] = best_metrics
    return history


def forward_text(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return model(batch["text"].to(device))


def forward_image(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return model(batch["image"].to(device))


def forward_video(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return model(batch["video"].to(device), batch["audio"].to(device))


def forward_trimodal(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return model(
        batch["text"].to(device),
        batch["image"].to(device),
        batch["video"].to(device),
        batch["audio"].to(device),
    )
