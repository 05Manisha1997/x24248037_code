"""Load text and image data from the ``datasets/`` folder."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
THESIS_DATASETS = PROJECT_DIR.parent / "datasets"
LOCAL_DATASETS = PROJECT_DIR / "datasets"

WELLBEING_DECLINE_LABELS = {
    "Depression",
    "Anxiety",
    "Stress",
    "Suicidal",
    "Bipolar",
    "Personality disorder",
}


def datasets_dir() -> Path:
    for candidate in (LOCAL_DATASETS, THESIS_DATASETS):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find datasets folder. Expected ../datasets or ./datasets"
    )


def load_text_data(sample: int | None = 10_000, add_aosp: bool = True) -> pd.DataFrame:
    """Load social-media text posts and mental-health status labels."""
    path = datasets_dir() / "unstructured_data.csv"
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"statement": "text"})
    df["wellbeing_decline"] = df["status"].isin(WELLBEING_DECLINE_LABELS).astype(int)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    if add_aosp:
        thesis_root = PROJECT_DIR.parent
        if str(thesis_root) not in sys.path:
            sys.path.insert(0, str(thesis_root))
        from src.features.achievement_lexicon import extract_aosp_dataframe

        aosp = extract_aosp_dataframe(df["text"])
        df = pd.concat([df, aosp], axis=1)

    return df


def load_image_index() -> pd.DataFrame:
    """Index image files under datasets/images/{class}/*."""
    image_dir = datasets_dir() / "images"
    rows: list[dict] = []
    if not image_dir.exists():
        return pd.DataFrame(columns=["path", "class", "filename"])

    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for path in class_dir.glob(ext):
                rows.append(
                    {
                        "path": str(path),
                        "class": class_dir.name,
                        "filename": path.name,
                    }
                )
    return pd.DataFrame(rows)
