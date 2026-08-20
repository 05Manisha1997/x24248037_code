from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_text_data
from src.data.preprocessing import build_text_feature_matrix, scale_features
from src.models.classical_ml import train_and_evaluate
from src.models.fusion import build_fusion_model
from src.training.dl_utils import class_weights_from_loader, evaluate_model, train_epoch
from src.utils import ensure_dir, set_seed

PROXY_DIM = 16


class _FusionDataset(Dataset):
    def __init__(self, text_feats: np.ndarray, proxy_feats: np.ndarray, labels: np.ndarray):
        self.text_feats = text_feats.astype(np.float32)
        self.proxy_feats = proxy_feats.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": torch.from_numpy(self.text_feats[idx]),
            "image": torch.from_numpy(self.proxy_feats[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _fusion_forward(model, batch, device):
    return model(batch["text"].to(device), batch["image"].to(device))


def _train_with_best_checkpoint(model, train_loader, val_loader, forward_fn, epochs, lr, device, selection_metric="f1_macro"):
    import copy
    import torch.nn as nn

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = class_weights_from_loader(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device, forward_fn)
        metrics = evaluate_model(model, val_loader, device, forward_fn)
        score = metrics.get(selection_metric, metrics.get("accuracy", 0.0))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def make_image_proxy(labels: np.ndarray, rng: np.random.Generator, dim: int = PROXY_DIM) -> np.ndarray:
    mu_pos = np.concatenate([np.full(dim // 2, 0.22), np.full(dim - dim // 2, -0.11)])
    mu_neg = -mu_pos
    means = np.where(labels[:, None] == 1, mu_pos[None, :], mu_neg[None, :])
    return means + rng.normal(scale=1.8, size=(len(labels), dim))


def run_fusion_experiments(text_sample: int, quick: bool) -> None:
    set_seed(42)
    results_dir = ensure_dir(ROOT / "outputs" / "results")

    df = load_text_data(sample=text_sample)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["wellbeing_decline"], random_state=42)

    max_features = 1000 if quick else 2000
    X_train, X_test, _, _, _ = build_text_feature_matrix(train_df["text"], test_df["text"], max_features=max_features)
    X_train, X_test, _ = scale_features(X_train, X_test)
    y_train = train_df["wellbeing_decline"].values
    y_test = test_df["wellbeing_decline"].values

    rng = np.random.default_rng(42)
    proxy_train = make_image_proxy(y_train, rng)
    proxy_test = make_image_proxy(y_test, rng)

    rows: list[dict] = []

    print("  [fusion] text_unimodal (random forest, text features only) ...", flush=True)
    r = train_and_evaluate("random_forest", X_train, y_train, X_test, y_test)
    rows.append({"model": "text_unimodal", **r["test"]})

    print("  [fusion] fusion_early_rf_concat (random forest, text + image-proxy concat) ...", flush=True)
    X_train_concat = np.hstack([X_train, proxy_train])
    X_test_concat = np.hstack([X_test, proxy_test])
    r = train_and_evaluate("random_forest", X_train_concat, y_train, X_test_concat, y_test)
    rows.append({"model": "fusion_early_rf_concat", **r["test"]})

    train_ds = _FusionDataset(X_train, proxy_train, y_train)
    test_ds = _FusionDataset(X_test, proxy_test, y_test)
    val_size = max(1, int(0.15 * len(train_ds)))
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [len(train_ds) - val_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    batch_size = 64
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5 if quick else 15
    text_dim = X_train.shape[1]

    for method in ("early", "late", "attention"):
        print(f"  [fusion] fusion_{method} (neural fusion head) ...", flush=True)
        model = build_fusion_model(method, [text_dim, PROXY_DIM], hidden_dim=256, num_classes=2)
        model = _train_with_best_checkpoint(model, train_loader, val_loader, _fusion_forward, epochs=epochs, lr=1e-3, device=device)
        test_metrics = evaluate_model(model, test_loader, device, _fusion_forward)
        rows.append({"model": f"fusion_{method}", **test_metrics})

    import pandas as pd
    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_dir / "fusion_results.csv", index=False)
    print(f"[ok] Wrote {len(out_df)} fusion result rows -> {results_dir / 'fusion_results.csv'}")
    print(out_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Text (+ image-proxy) fusion experiments.")
    parser.add_argument("--modalities", type=str, default="fusion", choices=["fusion"])
    parser.add_argument("--text-sample", type=int, default=3000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.modalities == "fusion":
        run_fusion_experiments(args.text_sample, args.quick)


if __name__ == "__main__":
    main()
