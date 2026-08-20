from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.achievement_lexicon import extract_aosp_dataframe
from src.utils import project_root

WELLBEING_DECLINE_LABELS = {"Depression", "Anxiety", "Stress", "Suicidal", "Bipolar", "Personality disorder"}
LMVD_REPO = "helang818/LMVD"
LMVD_RAW = f"https://raw.githubusercontent.com/{LMVD_REPO}/main"


def load_structured_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or project_root() / "datasets" / "structured_data.csv")
    df = pd.read_csv(path)
    df["high_distress"] = ((df["GAD_7_Score"] >= 10) | (df["PHQ_9_Score"] >= 10)).astype(int)
    df["life_satisfaction_proxy"] = 21 - df["GAD_7_Score"] - df["PHQ_9_Score"]  # inverted distress
    df["wellbeing_decline"] = df["high_distress"]
    return df


def load_text_data(path: str | Path | None = None, sample: int | None = None) -> pd.DataFrame:
    path = Path(path or project_root() / "datasets" / "unstructured_data.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"statement": "text"})
    df["wellbeing_decline"] = df["status"].isin(WELLBEING_DECLINE_LABELS).astype(int)
    df["life_satisfaction_proxy"] = 1 - df["wellbeing_decline"]  # binary proxy
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    aosp = extract_aosp_dataframe(df["text"])
    return pd.concat([df.reset_index(drop=True), aosp], axis=1)


def load_lmvd_labels(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or project_root() / "lmvd-analysis" / "data")
    summary = data_dir / "lmvd_labels_summary.csv"
    if summary.exists():
        return pd.read_csv(summary)

    label_dir = data_dir / "label"
    if not any(label_dir.glob("*_Depression.csv")):
        _download_lmvd_labels(label_dir, data_dir / "label.zip")

    records = []
    for csv_path in sorted(label_dir.glob("*_Depression.csv"), key=lambda p: int(p.stem.split("_")[0])):
        sample_id = int(csv_path.stem.split("_")[0])
        label_df = pd.read_csv(csv_path)
        label = int(label_df.columns[0])
        records.append({
            "sample_id": sample_id,
            "label": label,
            "class": "Depression" if label == 1 else "Normal",
            "wellbeing_decline": label,
        })

    labels_df = pd.DataFrame(records)
    train_df, temp = train_test_split(labels_df, test_size=0.3, stratify=labels_df["label"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=2 / 3, stratify=temp["label"], random_state=42)
    labels_df["split"] = "unassigned"
    labels_df.loc[labels_df["sample_id"].isin(train_df["sample_id"]), "split"] = "train"
    labels_df.loc[labels_df["sample_id"].isin(val_df["sample_id"]), "split"] = "val"
    labels_df.loc[labels_df["sample_id"].isin(test_df["sample_id"]), "split"] = "test"
    labels_df.to_csv(summary, index=False)
    return labels_df


def _download_lmvd_labels(label_dir: Path, zip_path: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{LMVD_RAW}/label/label.zip") as response:
        zip_path.write_bytes(response.read())
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(label_dir)


def _lmvd_zip_path(data_dir: Path) -> Path:
    return data_dir / "LMVD_Feature.zip"


def lmvd_zip_available(data_dir: Path) -> bool:
    zip_path = _lmvd_zip_path(data_dir)
    if not zip_path.exists():
        return False
    return zip_path.stat().st_size >= 15_500_000_000


def _lmvd_feature_dirs(data_dir: Path, modality: str) -> list[Path]:
    if modality == "video":
        return [data_dir / "tcnfeature"]
    return [data_dir / "audiofeature", data_dir / "Audio_feature"]


def discover_lmvd_features(data_dir: Path, modality: str) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for folder in _lmvd_feature_dirs(data_dir, modality):
        if not folder.exists():
            continue
        for path in folder.glob("*.npy"):
            mapping[int(path.stem)] = path
    return mapping


def discover_lmvd_sample_ids(data_dir: Path) -> set[int]:
    ids: set[int] = set()
    ids.update(discover_lmvd_features(data_dir, "video"))

    video_dir = data_dir / "Video_feature"
    if video_dir.exists():
        for path in video_dir.glob("*.csv"):
            ids.add(int(path.stem))

    zip_path = _lmvd_zip_path(data_dir)
    if lmvd_zip_available(data_dir):
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.namelist():
                    if member.startswith("tcnfeature/") and member.endswith(".npy"):
                        ids.add(int(Path(member).stem))
                    elif member.startswith("Video_feature/") and member.endswith(".csv"):
                        ids.add(int(Path(member).stem))
        except zipfile.BadZipFile:
            pass
    return ids


def discover_lmvd_training_sample_ids(
    data_dir: Path,
    min_cached: int = 100,
) -> set[int]:

    cached = set(discover_lmvd_features(data_dir, "video"))
    if len(cached) >= min_cached:
        return cached
    return discover_lmvd_sample_ids(data_dir)


def lmvd_features_available(data_dir: Path) -> bool:
    return bool(discover_lmvd_sample_ids(data_dir))


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _openface_row_to_tcn(row: pd.Series) -> np.ndarray:
    vec = np.zeros(171, dtype=np.float32)
    au_cols = sorted([c for c in row.index if c.startswith("AU") and c.endswith("_r")])[:17]
    for i, col in enumerate(au_cols):
        vec[i] = float(row[col]) if pd.notna(row[col]) else 0.0

    xs = sorted([c for c in row.index if c.startswith("x_")], key=lambda name: int(name.split("_")[1]))
    ys = sorted([c for c in row.index if c.startswith("y_")], key=lambda name: int(name.split("_")[1]))
    for i, (xc, yc) in enumerate(zip(xs, ys)):
        vec[17 + 2 * i] = float(row[xc]) if pd.notna(row[xc]) else 0.0
        vec[17 + 2 * i + 1] = float(row[yc]) if pd.notna(row[yc]) else 0.0

    for i, key in enumerate(
        ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]
    ):
        if key in row.index:
            vec[153 + i] = float(row[key]) if pd.notna(row[key]) else 0.0

    for i, key in enumerate(["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]):
        if key in row.index:
            vec[165 + i] = float(row[key]) if pd.notna(row[key]) else 0.0
    return vec


def _video_csv_path(data_dir: Path, sample_id: int) -> Path | None:
    for name in (f"{sample_id}.csv", f"{sample_id:03d}.csv"):
        path = data_dir / "Video_feature" / name
        if path.exists():
            return path
    return None


def _openface_usecols(name: str) -> bool:
    name = name.strip()
    if name.startswith("AU") and name.endswith("_r"):
        return True
    if name.startswith("x_") or name.startswith("y_"):
        return True
    if name.startswith("pose_"):
        return True
    return name in {
        "gaze_0_x", "gaze_0_y", "gaze_0_z",
        "gaze_1_x", "gaze_1_y", "gaze_1_z",
        "gaze_angle_x", "gaze_angle_y",
    }


def load_lmvd_video_from_csv(
    sample_id: int,
    data_dir: Path,
    max_video: int = 915,
) -> np.ndarray:
    video = np.zeros((max_video, 171), dtype=np.float32)
    path = _video_csv_path(data_dir, sample_id)
    if path is None:
        return video
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in header.columns if _openface_usecols(str(c))]
    df = _strip_columns(pd.read_csv(path, usecols=usecols))
    length = min(len(df), max_video)
    for i in range(length):
        video[i] = _openface_row_to_tcn(df.iloc[i])
    return video


def cache_lmvd_tcn_npy(sample_id: int, data_dir: Path, max_video: int = 915) -> Path | None:
    out_dir = data_dir / "tcnfeature"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}.npy"
    if out_path.exists():
        return out_path
    video = load_lmvd_video_from_csv(sample_id, data_dir, max_video)
    if not np.any(video):
        return None
    np.save(out_path, video)
    return out_path


def _load_npy_from_zip(zip_path: Path, member: str) -> np.ndarray | None:
    import io
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if member not in archive.namelist():
                return None
            with archive.open(member) as handle:
                return np.load(io.BytesIO(handle.read())).astype(np.float32)
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


def load_lmvd_sequence(
    sample_id: int,
    data_dir: Path,
    max_video: int = 915,
    max_audio: int = 186,
    *,
    cache_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:

    video = np.zeros((max_video, 171), dtype=np.float32)
    audio = np.zeros((max_audio, 128), dtype=np.float32)

    video_candidates = [
        data_dir / "tcnfeature" / f"{sample_id}.npy",
        data_dir / "tcnfeature" / f"{sample_id:03d}.npy",
    ]
    audio_candidates = [
        data_dir / "audiofeature" / f"{sample_id}.npy",
        data_dir / "audiofeature" / f"{sample_id:03d}.npy",
        data_dir / "Audio_feature" / f"{sample_id}.npy",
        data_dir / "Audio_feature" / f"{sample_id:03d}.npy",
    ]

    for path in video_candidates:
        if path.exists():
            v = np.load(path).astype(np.float32)
            length = min(len(v), max_video)
            video[:length] = v[:length]
            break

    if not cache_only and not np.any(video):
        csv_video = load_lmvd_video_from_csv(sample_id, data_dir, max_video)
        if np.any(csv_video):
            video = csv_video

    for path in audio_candidates:
        if path.exists():
            a = np.load(path).astype(np.float32)
            length = min(len(a), max_audio)
            audio[:length] = a[:length]
            break

    # Fallback: read directly from zip (saves ~16 GB disk space)
    if not cache_only and (not np.any(video) or not np.any(audio)):
        zip_path = data_dir / "LMVD_Feature.zip"
        if zip_path.exists():
            for stem in (str(sample_id), f"{sample_id:03d}"):
                if not np.any(video):
                    arr = _load_npy_from_zip(zip_path, f"tcnfeature/{stem}.npy")
                    if arr is not None:
                        length = min(len(arr), max_video)
                        video[:length] = arr[:length]
                if not np.any(audio):
                    arr = _load_npy_from_zip(zip_path, f"audiofeature/{stem}.npy")
                    if arr is not None:
                        length = min(len(arr), max_audio)
                        audio[:length] = arr[:length]

    return video, audio


class ImageFolderDataset:
    """Load images from datasets/images/{class_name}/*.jpg|png structure."""

    def __init__(self, image_dir: Path, transform=None):
        from PIL import Image

        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        image_dir = Path(image_dir)
        if not image_dir.exists():
            return
        classes = sorted(d.name for d in image_dir.iterdir() if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for class_name, idx in self.class_to_idx.items():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for path in (image_dir / class_name).glob(ext):
                    self.samples.append((path, idx))
        self._Image = Image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = self._Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
