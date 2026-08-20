from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def results_to_dataframe(all_results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in all_results:
        row = {
            "modality": r.get("modality", ""),
            "model": r.get("model", r.get("model_name", "")),
            "task": r.get("task", "classification"),
        }
        if "test" in r:
            row.update({f"test_{k}": v for k, v in r["test"].items()})
        if "cv" in r:
            row["cv_mean"] = r["cv"].get("cv_mean")
            row["cv_std"] = r["cv"].get("cv_std")
        if "eval" in r:
            row.update({f"eval_{k.replace('eval_', '')}": v for k, v in r["eval"].items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_model_comparison(df: pd.DataFrame, metric: str = "test_f1_weighted", output_path: Path | None = None) -> None:
    if df.empty:
        return
    # Prefer requested metric; fall back safely through common aliases
    candidates = [
        metric,
        "test_f1_macro",
        "test_f1_weighted",
        "f1_macro",
        "f1_weighted",
        "test_accuracy",
        "accuracy",
        "test_roc_auc",
        "roc_auc",
    ]
    chosen = next((c for c in candidates if c in df.columns and df[c].notna().any()), None)
    if chosen is None:
        return
    metric = chosen
    plot_df = df.dropna(subset=[metric]).sort_values(metric, ascending=True)
    if plot_df.empty:
        return
    # Ensure modality/model columns exist for labels
    if "modality" not in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df["modality"] = "model"
    if "model" not in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df["model"] = plot_df.index.astype(str)
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.4)))
    modalities = plot_df["modality"].astype(str).unique()
    colors = sns.color_palette("muted", max(len(modalities), 1))
    palette = dict(zip(modalities, colors))
    labels = plot_df["modality"].astype(str) + " / " + plot_df["model"].astype(str)
    ax.barh(
        labels,
        plot_df[metric],
        color=[palette[m] for m in plot_df["modality"].astype(str)],
    )
    ax.set_xlabel(metric)
    ax.set_title("Model Comparison Across Modalities")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paradox_scatter(df: pd.DataFrame, output_path: Path | None = None) -> None:
    if "aosp_composite" not in df.columns:
        return
    y_col = "wellbeing_decline" if "wellbeing_decline" in df.columns else "distress_score"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42), x="aosp_composite", y=y_col, alpha=0.3, ax=ax)
    ax.set_xlabel("Achievement Self-Presentation (AOSP) Composite")
    ax.set_ylabel("Well-Being Decline Indicator")
    ax.set_title("The Idealised Self-Presentation Paradox")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    output_path: Path | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> np.ndarray:
    """Compute and (optionally) save a confusion matrix heatmap. Returns the raw counts matrix."""
    cm = confusion_matrix(y_true, y_pred)
    display = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        display,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def save_classification_report(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


def save_results_report(all_results: list[dict], paradox_stats: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(all_results)
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    with (output_dir / "paradox_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(paradox_stats, handle, indent=2)
    plot_model_comparison(df, output_path=output_dir / "model_comparison.png")
    print(f"Results saved to {output_dir}")
