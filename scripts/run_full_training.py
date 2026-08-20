from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import discover_lmvd_training_sample_ids
from src.utils import ensure_dir


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n=== {label} ===")
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    ok = result.returncode == 0
    print(f"=== {label}: {'ok' if ok else f'FAILED (exit {result.returncode})'} ===")
    return ok


def _train_text_models(skip_transformers: bool) -> bool:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.data.loaders import load_text_data
    from src.data.preprocessing import build_text_feature_matrix, scale_features
    from src.models.classical_ml import train_and_evaluate
    from src.utils import set_seed

    set_seed(42)
    results_dir = ensure_dir(ROOT / "outputs" / "results")

    df = load_text_data(sample=3000)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["wellbeing_decline"], random_state=42)
    X_train, X_test, _, _, _ = build_text_feature_matrix(train_df["text"], test_df["text"])
    X_train, X_test, _ = scale_features(X_train, X_test)
    y_train, y_test = train_df["wellbeing_decline"].values, test_df["wellbeing_decline"].values

    rows = []
    for model_name in ("random_forest", "svm", "xgboost"):
        r = train_and_evaluate(model_name, X_train, y_train, X_test, y_test)
        rows.append({"model": model_name, **r["test"]})
        print(f"  [text] {model_name}: {r['test']}")

    transformer_rows = []
    if not skip_transformers:
        from src.models.text_transformers import train_transformer

        tr_train, tr_val = train_test_split(train_df, test_size=0.15, stratify=train_df["wellbeing_decline"], random_state=42)
        for model_name in ("bert-base-uncased", "roberta-base", "distilbert-base-uncased"):
            out_dir = ensure_dir(ROOT / "outputs" / "models" / model_name.replace("/", "_"))
            print(f"  [text] training transformer {model_name} ...")
            result = train_transformer(
                model_name=model_name,
                train_texts=tr_train["text"].tolist(), train_labels=tr_train["wellbeing_decline"].tolist(),
                val_texts=tr_val["text"].tolist(), val_labels=tr_val["wellbeing_decline"].tolist(),
                output_dir=str(out_dir), num_labels=2, epochs=2, batch_size=8,
            )
            eval_row = {
                "model": model_name,
                **{k.replace("eval_", ""): v for k, v in result["eval"].items() if "f1" in k or "accuracy" in k},
            }
            transformer_rows.append(eval_row)
            print(f"  [text] {model_name}: {result['eval']}")
        pd.DataFrame([{"modality": "text_transformer", **r} for r in transformer_rows]).to_csv(
            results_dir / "transformer_results.csv", index=False,
        )
    else:
        print("  [text] skipping transformers (--skip-transformers)")

    pd.DataFrame(rows + transformer_rows).to_csv(results_dir / "text_model_results.csv", index=False)
    print(f"  [text] wrote {len(rows) + len(transformer_rows)} rows -> text_model_results.csv")
    return True


def _train_image_models() -> bool:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    from scripts.prepare_image_dataset import prefer_lmvd_or_synthetic
    from src.data.loaders import ImageFolderDataset
    from src.models.image_cnn import build_image_model
    from src.utils import load_config, set_seed

    set_seed(42)
    config = load_config(ROOT / "config.yaml")
    results_dir = ensure_dir(ROOT / "outputs" / "results")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = ROOT / config["paths"]["image_dir"]
    lmvd_root = ROOT / "datasets" / "lmvd_extracted"
    prefer_lmvd_or_synthetic(image_dir, lmvd_root, per_class=150)

    image_size = config["image_models"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolderDataset(image_dir, transform=transform)
    if len(dataset) == 0:
        print("  [image] no images available - skipping.")
        return False

    labels = [lab for _, lab in dataset.samples]
    idx = np.arange(len(dataset))
    test_size = float(config.get("evaluation", {}).get("test_size", 0.2))
    train_idx, val_idx = train_test_split(idx, test_size=test_size, stratify=labels, random_state=42)
    batch_size = config["image_models"]["batch_size"]
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=batch_size)

    epochs = config["image_models"]["epochs"]
    rows = []
    for arch in config["image_models"]["architectures"]:
        print(f"  [image] training {arch} ...")
        model = build_image_model(arch, num_classes=dataset.num_classes, pretrained=(arch != "cnn")).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["image_models"]["learning_rate"])
        cw = compute_class_weight("balanced", classes=np.unique(labels), y=np.array(labels)[train_idx])
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
        for _ in range(epochs):
            model.train()
            for images, batch_labels in train_loader:
                images, batch_labels = images.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                criterion(model(images), batch_labels).backward()
                optimizer.step()

        model.eval()
        preds, y_true, probs = [], [], []
        with torch.no_grad():
            for images, batch_labels in val_loader:
                logits = model(images.to(device))
                p = torch.softmax(logits, dim=-1)
                preds.extend(p.argmax(1).cpu().numpy())
                y_true.extend(batch_labels.numpy())
                probs.extend(p[:, 1].cpu().numpy() if p.shape[1] == 2 else p.max(1).values.cpu().numpy())
        y_true, preds, probs = np.array(y_true), np.array(preds), np.array(probs)
        metrics = {
            "model": arch,
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        }
        rows.append(metrics)
        print(f"  [image] {arch}: {metrics}")

    pd.DataFrame(rows).to_csv(results_dir / "image_model_results.csv", index=False)
    print(f"  [image] wrote {len(rows)} rows -> image_model_results.csv")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training pipeline orchestrator.")
    parser.add_argument("--skip-transformers", action="store_true", help="Skip BERT/RoBERTa/DistilBERT fine-tuning (slow).")
    args = parser.parse_args()

    completed = {}

    completed["download"] = _run(
        [sys.executable, str(ROOT / "scripts" / "download_all.py"), "--skip-lmvd"], "1/6 download_all",
    )
    completed["text"] = _train_text_models(args.skip_transformers)
    completed["image"] = _train_image_models()

    data_dir = ROOT / "lmvd-analysis" / "data"
    if discover_lmvd_training_sample_ids(data_dir):
        completed["trimodal"] = _run(
            [sys.executable, str(ROOT / "scripts" / "run_trimodal_experiments.py")], "4/6 trimodal experiments",
        )
    else:
        print("\n=== 4/6 trimodal experiments: skipped (no cached LMVD features) ===")
        completed["trimodal"] = False

    completed["fusion"] = _run(
        [sys.executable, str(ROOT / "scripts" / "run_experiments.py"), "--modalities", "fusion", "--text-sample", "3000"],
        "5/6 fusion experiments",
    )
    completed["plot"] = _run(
        [sys.executable, str(ROOT / "scripts" / "plot_model_comparison.py")], "6/6 merge & plot",
    )

    print("\n=== run_full_training summary ===")
    for stage, ok in completed.items():
        print(f"  {stage}: {'done' if ok else 'skipped/failed'}")


if __name__ == "__main__":
    main()
