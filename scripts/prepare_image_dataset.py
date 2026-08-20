from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.trimodal_dataset import parse_visual_frame, render_landmark_image

CLASSES = ["normal", "depression"]

# Coarse 68-point mean face shape (normalized 0-1 coordinate space), grouped to
# match FACE_EDGES in src.data.trimodal_dataset (jaw, brows, nose, eyes, mouth).
def _mean_face_landmarks() -> np.ndarray:
    pts = np.zeros((68, 2), dtype=float)

    # Jaw line (0-16): arc from left ear to right ear through the chin.
    jaw_angles = np.linspace(200, 340, 17)
    for i, angle in enumerate(jaw_angles):
        rad = np.deg2rad(angle)
        pts[i] = [0.5 + 0.45 * np.cos(rad), 0.45 + 0.55 * np.sin(rad)]

    # Eyebrows (17-21 left, 22-26 right).
    for i, t in enumerate(np.linspace(0, 1, 5)):
        pts[17 + i] = [0.22 + 0.18 * t, 0.32 - 0.05 * np.sin(np.pi * t)]
        pts[22 + i] = [0.60 + 0.18 * t, 0.27 + 0.05 * np.sin(np.pi * t)]

    # Nose bridge (27-30) then lower nose (31-35).
    for i, t in enumerate(np.linspace(0, 1, 4)):
        pts[27 + i] = [0.5, 0.35 + 0.18 * t]
    for i, t in enumerate(np.linspace(-1, 1, 5)):
        pts[31 + i] = [0.5 + 0.08 * t, 0.55 + 0.02 * (1 - abs(t))]

    # Eyes (36-41 left hexagon, 42-47 right hexagon).
    for i, angle in enumerate(np.linspace(0, 300, 6)):
        rad = np.deg2rad(angle)
        pts[36 + i] = [0.32 + 0.07 * np.cos(rad), 0.40 + 0.035 * np.sin(rad)]
        pts[42 + i] = [0.68 + 0.07 * np.cos(rad), 0.40 + 0.035 * np.sin(rad)]

    # Outer mouth (48-59, 12 points) and inner mouth (60-67, 8 points).
    for i, angle in enumerate(np.linspace(0, 330, 12)):
        rad = np.deg2rad(angle)
        pts[48 + i] = [0.5 + 0.14 * np.cos(rad), 0.75 + 0.05 * np.sin(rad)]
    for i, angle in enumerate(np.linspace(0, 315, 8)):
        rad = np.deg2rad(angle)
        pts[60 + i] = [0.5 + 0.08 * np.cos(rad), 0.75 + 0.03 * np.sin(rad)]

    return pts


_BASE_LANDMARKS = _mean_face_landmarks()


def synthetic_landmarks(label: int, rng: np.random.Generator) -> np.ndarray:
    pts = _BASE_LANDMARKS.copy()

    mouth_idx = list(range(48, 68))
    brow_idx = list(range(17, 27))
    curvature_mean = -0.022 if label == 1 else 0.011
    curvature = rng.normal(loc=curvature_mean, scale=0.028)
    pts[mouth_idx, 1] += curvature * np.array(
        [1 - abs(2 * i / (len(mouth_idx) - 1) - 1) for i in range(len(mouth_idx))]
    )
    brow_mean = 0.010 if label == 1 else -0.007
    pts[brow_idx, 1] += rng.normal(loc=brow_mean, scale=0.015)

    jitter = rng.normal(scale=0.035, size=pts.shape)
    pts = pts + jitter

    # small global affine variation (scale/rotation) for visual diversity
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.92, 1.08)
    rad = np.deg2rad(angle)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    center = pts.mean(axis=0)
    pts = (pts - center) @ rot.T * scale + center

    return pts


def _find_real_lmvd_images(lmvd_root: Path) -> dict[str, Path]:
    found = {}
    for class_name in CLASSES:
        candidates = [
            lmvd_root / "images" / class_name,
            lmvd_root / "images" / class_name.capitalize(),
            lmvd_root / class_name,
            lmvd_root / class_name.capitalize(),
        ]
        for cand in candidates:
            if cand.exists() and any(cand.glob("*.png")) or (cand.exists() and any(cand.glob("*.jpg"))):
                found[class_name] = cand
                break
    return found


def _render_real_lmvd_images(image_dir: Path, per_class: int) -> bool:
    from src.data.loaders import load_lmvd_labels, load_lmvd_sequence

    from scripts.extract_lmvd_modalities import extract_and_cache_samples

    data_dir = ROOT / "lmvd-analysis" / "data"
    zip_path = data_dir / "LMVD_Feature.zip"
    if not (zip_path.exists() and zip_path.stat().st_size > 15_000_000_000):
        return False

    labels_df = load_lmvd_labels(data_dir)
    made_any = False

    for label, class_name in enumerate(CLASSES):
        dst_dir = image_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        existing = list(dst_dir.glob("lmvd_*.png"))
        if len(existing) >= per_class:
            made_any = True
            continue

        candidate_ids = labels_df.loc[labels_df["label"] == label, "sample_id"].tolist()
        candidate_ids = candidate_ids[: per_class * 2]  # extra headroom for extraction misses
        print(f"[real] {class_name}: extracting/caching up to {len(candidate_ids)} LMVD samples ...")
        cached_ids = extract_and_cache_samples(candidate_ids, data_dir)

        n_written = 0
        for sample_id in cached_ids:
            if n_written >= per_class:
                break
            video, _ = load_lmvd_sequence(sample_id, data_dir, cache_only=True)
            valid_len = max(1, int(np.any(video != 0, axis=1).sum()))
            if valid_len <= 1:
                continue
            mid = valid_len // 2
            landmarks = parse_visual_frame(video[mid])["landmarks"]
            img = render_landmark_image(landmarks, size=224)
            img.save(dst_dir / f"lmvd_{sample_id}.png")
            n_written += 1

        print(f"[real] {class_name}: rendered {n_written} real LMVD landmark images -> {dst_dir}")
        made_any = made_any or n_written > 0

    return made_any


def prefer_lmvd_or_synthetic(image_dir: Path, lmvd_root: Path, per_class: int = 150) -> None:
    image_dir = Path(image_dir)
    real_sources = _find_real_lmvd_images(lmvd_root)

    if real_sources:
        import shutil
        for class_name, src_dir in real_sources.items():
            dst_dir = image_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            existing = list(dst_dir.glob("*.png")) + list(dst_dir.glob("*.jpg"))
            if len(existing) >= per_class:
                continue
            frames = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
            for frame in frames[:per_class]:
                shutil.copy2(frame, dst_dir / frame.name)
            print(f"[real] {class_name}: copied {min(len(frames), per_class)} frames from {src_dir}")
        return

    if _render_real_lmvd_images(image_dir, per_class):
        return

    rng = np.random.default_rng(42)
    for label, class_name in enumerate(CLASSES):
        dst_dir = image_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        existing = list(dst_dir.glob("*.png"))
        if len(existing) >= per_class:
            print(f"[synthetic] {class_name}: already has {len(existing)} images, skipping")
            continue
        n_needed = per_class - len(existing)
        for i in range(n_needed):
            landmarks = synthetic_landmarks(label, rng)
            img = render_landmark_image(landmarks, size=224)
            img.save(dst_dir / f"synthetic_{len(existing) + i:04d}.png")
        print(f"[synthetic] {class_name}: generated {n_needed} images -> {dst_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate datasets/images/ for image-model training.")
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--lmvd-root", type=str, default=None)
    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else ROOT / "datasets" / "images"
    lmvd_root = Path(args.lmvd_root) if args.lmvd_root else ROOT / "datasets" / "lmvd_extracted"
    prefer_lmvd_or_synthetic(image_dir, lmvd_root, per_class=args.per_class)


if __name__ == "__main__":
    main()
