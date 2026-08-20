from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import plot_model_comparison
from src.utils import ensure_dir

METRIC_COLS = ["accuracy", "f1_weighted", "f1_macro", "roc_auc"]


def _metrics_from_row(row: pd.Series, prefix: str = "") -> dict:
    out = {}
    for col in METRIC_COLS:
        key = f"{prefix}{col}"
        out[col] = float(row[key]) if key in row and pd.notna(row[key]) else float("nan")
    return out


def _rows_text(results_dir: Path) -> list[dict]:
    path = results_dir / "text_model_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"modality": "text_dl", "model": r["model"], "source": path.name, "tier": "unimodal DL", **_metrics_from_row(r)}
        for _, r in df.iterrows()
    ]


def _rows_transformer(results_dir: Path) -> list[dict]:
    path = results_dir / "transformer_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"modality": "text_transformer", "model": r["model"], "source": path.name, "tier": "unimodal DL", **_metrics_from_row(r)}
        for _, r in df.iterrows()
    ]


def _rows_image(results_dir: Path) -> list[dict]:
    path = results_dir / "image_model_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"modality": "image_dl", "model": r["model"], "source": path.name, "tier": "unimodal DL", **_metrics_from_row(r)}
        for _, r in df.iterrows()
    ]


def _rows_fusion(results_dir: Path) -> list[dict]:
    path = results_dir / "fusion_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        tier = "proxy unimodal" if r["model"] == "text_unimodal" else "proxy fusion (text + synthetic proxy embedding)"
        rows.append({"modality": "fusion_proxy", "model": r["model"], "source": path.name, "tier": tier, **_metrics_from_row(r)})
    return rows


def _rows_trimodal(results_dir: Path) -> list[dict]:
    path = results_dir / "trimodal_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    tier_by_modality = {
        "text_dl": "unimodal DL (LMVD)",
        "image_dl": "unimodal DL (LMVD)",
        "video_dl": "unimodal DL (LMVD)",
        "bimodal": "bimodal fusion",
        "trimodal": "trimodal fusion",
    }
    rows = []
    for _, r in df.iterrows():
        modality = r["modality"]
        rows.append({
            "modality": modality, "model": r["model"], "source": path.name,
            "tier": tier_by_modality.get(modality, modality),
            **_metrics_from_row(r, prefix="test_"),
        })
    return rows


def _rows_model_comparison(results_dir: Path) -> list[dict]:
    path = results_dir / "model_comparison.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {"modality": r["modality"], "model": r["model"], "source": path.name, "tier": "unimodal / classical", **_metrics_from_row(r, prefix="test_")}
        for _, r in df.iterrows()
    ]


def _best_row(df: pd.DataFrame, mask: pd.Series, tier_label: str) -> dict | None:
    subset = df[mask].dropna(subset=["accuracy"])
    if subset.empty:
        return None
    best = subset.sort_values("accuracy", ascending=False).iloc[0]
    return {
        "tier": tier_label, "modality": best["modality"], "model": best["model"],
        "accuracy": best["accuracy"], "f1_macro": best["f1_macro"],
        "f1_weighted": best["f1_weighted"], "roc_auc": best["roc_auc"],
    }


def main() -> None:
    results_dir = ensure_dir(ROOT / "outputs" / "results")

    rows: list[dict] = []
    rows += _rows_text(results_dir)
    rows += _rows_transformer(results_dir)
    rows += _rows_image(results_dir)
    rows += _rows_fusion(results_dir)
    rows += _rows_trimodal(results_dir)
    rows += _rows_model_comparison(results_dir)

    if not rows:
        print("[skip] No per-modality result CSVs found under outputs/results/ - nothing to merge.")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["modality"].astype(str) + " / " + df["model"].astype(str)
    df = df[["modality", "model", "accuracy", "f1_weighted", "f1_macro", "roc_auc", "source", "tier", "label"]]
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    df.to_csv(results_dir / "all_models_comparison.csv", index=False)
    print(f"[ok] all_models_comparison.csv: {len(df)} rows from {df['source'].nunique()} source files")

    plot_model_comparison(df, metric="f1_weighted", output_path=results_dir / "all_models_comparison.png")

    fusion_df = df[df["modality"] == "fusion_proxy"]
    if not fusion_df.empty:
        plot_model_comparison(fusion_df, metric="accuracy", output_path=results_dir / "fusion_strategy_comparison.png")

    summary_rows = [
        r for r in (
            _best_row(df, df["modality"] == "fusion_proxy", "text"),
            _best_row(df, (df["modality"] == "fusion_proxy") & (df["model"] != "text_unimodal"), "fusion"),
            _best_row(df, df["modality"] == "trimodal", "trimodal"),
        ) if r is not None
    ]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(results_dir / "unimodal_vs_fusion.csv", index=False)

    files_counts = {}
    for src in df["source"].unique():
        files_counts[src] = int((df["source"] == src).sum())

    report = {
        "generated_at": datetime.now().isoformat(),
        "files": files_counts,
        "model_counts": files_counts,
        "top_15": df.head(15)[["modality", "model", "accuracy"]].to_dict(orient="records"),
        "total_models": len(df),
        "mean_accuracy": float(df["accuracy"].dropna().mean()) if df["accuracy"].notna().any() else None,
    }
    with (results_dir / "full_training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"[ok] full_training_report.json: {report['total_models']} models, "
          f"mean accuracy {report['mean_accuracy']:.4f}" if report["mean_accuracy"] else "[ok] full_training_report.json written")


if __name__ == "__main__":
    main()
