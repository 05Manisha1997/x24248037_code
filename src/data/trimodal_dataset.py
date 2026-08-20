"""LMVD trimodal dataset: text (AOSP + expression), image, video/audio sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.data.loaders import load_lmvd_sequence
from src.features.achievement_lexicon import extract_aosp_features

FACE_EDGES = [
    (list(range(0, 17)), False),
    (list(range(17, 22)), False),
    (list(range(22, 27)), False),
    (list(range(27, 31)), False),
    (list(range(31, 36)), False),
    (list(range(36, 42)), True),
    (list(range(42, 48)), True),
    (list(range(48, 60)), True),
    (list(range(60, 68)), True),
]

TEXT_FEATURE_NAMES = [
    "competence_score", "success_score", "status_score", "idealisation_score",
    "distress_score", "social_comparison_score", "aosp_composite", "paradox_index",
    "word_count", "exclamation_density", "hashtag_count", "emoji_density",
    "au_mean", "au_std", "au_max", "gaze_mean", "gaze_std", "pose_mean", "pose_std",
]


def parse_visual_frame(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "au": frame[0:17],
        "landmarks": frame[17:153].reshape(68, 2),
        "gaze": frame[153:165],
        "pose": frame[165:171],
    }


def _normalize_landmarks(landmarks: np.ndarray, size: int) -> np.ndarray:
    pts = np.asarray(landmarks, dtype=float)
    if np.allclose(pts, 0):
        return np.zeros((68, 2), dtype=float)
    pts = pts - pts.min(axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    span[span == 0] = 1.0
    return pts / span * (size - 24) + 12


def render_landmark_image(landmarks: np.ndarray, size: int = 224) -> Image.Image:
    img = Image.new("RGB", (size, size), (248, 248, 252))
    draw = ImageDraw.Draw(img)
    pts = _normalize_landmarks(landmarks, size)
    if np.allclose(pts, 0):
        return img

    for part, closed in FACE_EDGES:
        coords = [(float(pts[i, 0]), float(pts[i, 1])) for i in part]
        draw.line(coords, fill=(46, 94, 170), width=2)
        if closed and len(coords) > 1:
            draw.line([coords[-1], coords[0]], fill=(46, 94, 170), width=2)

    for x, y in pts:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(224, 122, 95))
    return img


def expression_stats_from_video(video: np.ndarray) -> np.ndarray:
    valid = video[np.any(video != 0, axis=1)]
    if len(valid) == 0:
        return np.zeros(7, dtype=np.float32)
    au = valid[:, 0:17]
    gaze = valid[:, 153:165]
    pose = valid[:, 165:171]
    return np.array(
        [au.mean(), au.std(), au.max(), gaze.mean(), gaze.std(), pose.mean(), pose.std()],
        dtype=np.float32,
    )


def build_vlog_caption(video: np.ndarray) -> str:
    """Expression-derived caption — does NOT use the depression label (avoids leakage)."""
    stats = expression_stats_from_video(video)
    au_mean = float(stats[0])
    au_std = float(stats[1])
    if au_mean < 0.15:
        base = "quiet flat expression calm reserved day sharing thoughts"
    elif au_mean > 0.35:
        base = "expressive energetic intense emotional moment talking openly"
    else:
        base = "normal day reflecting on work life friends and goals"
    if au_std > 0.2:
        base += " varying facial cues shifting mood"
    else:
        base += " steady expression throughout"
    return base


def extract_trimodal_text_features(label: int, video: np.ndarray) -> np.ndarray:
    # label kept in signature for call-site compatibility; unused for captioning
    _ = label
    caption = build_vlog_caption(video)
    aosp = extract_aosp_features(caption)
    aosp_vec = np.array(
        [
            aosp.competence_score, aosp.success_score, aosp.status_score,
            aosp.idealisation_score, aosp.distress_score, aosp.social_comparison_score,
            aosp.aosp_composite, aosp.paradox_index, float(aosp.word_count),
            aosp.exclamation_density, float(aosp.hashtag_count), aosp.emoji_density,
        ],
        dtype=np.float32,
    )
    expr = expression_stats_from_video(video)
    return np.concatenate([aosp_vec, expr])


def find_lmvd_image(sample_id: int, label: int, image_root: Path) -> Path | None:
    class_name = "Depression" if label == 1 else "Normal"
    for folder in (image_root / class_name, image_root / class_name.lower()):
        if not folder.exists():
            continue
        for pattern in (f"{sample_id:04d}_*.png", f"{sample_id}_*.png", f"{sample_id:03d}_*.png"):
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]
    return None


class LMVDTrimodalDataset(Dataset):

    def __init__(
        self,
        sample_ids: list[int],
        labels: list[int],
        data_dir: Path,
        image_root: Path | None = None,
        max_video: int = 915,
        max_audio: int = 186,
        image_size: int = 224,
        image_transform=None,
        cache_only: bool = True,
    ):
        self.sample_ids = sample_ids
        self.labels = labels
        self.data_dir = Path(data_dir)
        self.image_root = Path(image_root) if image_root else None
        self.max_video = max_video
        self.max_audio = max_audio
        self.image_size = image_size
        self.image_transform = image_transform
        self.cache_only = cache_only

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]
        video, audio = load_lmvd_sequence(
            sample_id,
            self.data_dir,
            self.max_video,
            self.max_audio,
            cache_only=self.cache_only,
        )

        text_feats = extract_trimodal_text_features(label, video)
        image = self._load_image(sample_id, label, video)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "text": torch.from_numpy(text_feats),
            "image": image,
            "video": torch.from_numpy(video),
            "audio": torch.from_numpy(audio),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
        }

    def _load_image(self, sample_id: int, label: int, video: np.ndarray) -> torch.Tensor | Image.Image:
        if self.image_root is not None:
            path = find_lmvd_image(sample_id, label, self.image_root)
            if path is not None:
                return Image.open(path).convert("RGB").resize((self.image_size, self.image_size))

        valid_len = max(1, int(np.any(video != 0, axis=1).sum()))
        mid = valid_len // 2
        landmarks = parse_visual_frame(video[mid])["landmarks"]
        return render_landmark_image(landmarks, self.image_size)


def trimodal_text_dim() -> int:
    return len(TEXT_FEATURE_NAMES)
