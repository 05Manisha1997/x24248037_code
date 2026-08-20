from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import _openface_row_to_tcn, _openface_usecols, _strip_columns, load_lmvd_labels, load_lmvd_sequence
from src.utils import ensure_dir, save_json


def extract_and_cache_samples(sample_ids: list[int], data_dir: Path, max_video: int = 915) -> list[int]:
    zip_path = data_dir / "LMVD_Feature.zip"
    tcn_dir = ensure_dir(data_dir / "tcnfeature")
    audio_dir = ensure_dir(data_dir / "Audio_feature")

    ready = []
    already_cached = [
        sid for sid in sample_ids
        if (tcn_dir / f"{sid}.npy").exists() and (audio_dir / f"{sid:03d}.npy").exists()
    ]
    remaining = [sid for sid in sample_ids if sid not in already_cached]
    ready.extend(already_cached)

    if not remaining:
        return ready
    if not zip_path.exists():
        return ready

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

        def _find_member(prefix: str, ext: str) -> str | None:
            # The zip mixes two zero-padding widths across id ranges (an
            # artifact of how the LMVD dump was assembled in two batches):
            # 3-digit for lower ids, 4-digit for higher ones. Try both plus
            # the bare id rather than assuming one convention.
            for stem in (f"{sid:03d}", f"{sid:04d}", str(sid)):
                candidate = f"{prefix}/{stem}{ext}"
                if candidate in names:
                    return candidate
            return None

        for sid in remaining:
            tcn_path = tcn_dir / f"{sid}.npy"
            audio_path = audio_dir / f"{sid:03d}.npy"
            video_ok = tcn_path.exists()
            audio_ok = audio_path.exists()

            if not audio_ok:
                member = _find_member("Audio_feature", ".npy")
                if member:
                    audio_path.write_bytes(zf.read(member))
                    audio_ok = True

            if not video_ok:
                member = _find_member("Video_feature", ".csv")
                if member:
                    with zf.open(member) as handle:
                        raw = handle.read()
                    header = pd.read_csv(io.BytesIO(raw), nrows=0)
                    usecols = [c for c in header.columns if _openface_usecols(str(c))]
                    df = _strip_columns(pd.read_csv(io.BytesIO(raw), usecols=usecols))
                    video = np.zeros((max_video, 171), dtype=np.float32)
                    length = min(len(df), max_video)
                    for i in range(length):
                        video[i] = _openface_row_to_tcn(df.iloc[i])
                    if np.any(video):
                        np.save(tcn_path, video)
                        video_ok = True

            if video_ok and audio_ok:
                ready.append(sid)

    return ready


def _sample_alignment(sample_id: int, data_dir: Path) -> tuple[float, float] | None:
    video, audio = load_lmvd_sequence(sample_id, data_dir, cache_only=True)
    valid_video = video[np.any(video != 0, axis=1)]
    valid_audio = audio[np.any(audio != 0, axis=1)]
    if len(valid_video) == 0 or len(valid_audio) == 0:
        return None
    au_mean = float(np.abs(valid_video[:, 0:17]).mean())
    audio_energy = float(np.sqrt((valid_audio ** 2).mean()))
    return au_mean, audio_energy


def main() -> None:
    parser = argparse.ArgumentParser(description="AU-vs-audio-energy alignment analysis for LMVD samples.")
    parser.add_argument("--max-samples", type=int, default=150)
    args = parser.parse_args()

    data_dir = ROOT / "lmvd-analysis" / "data"
    results_dir = ensure_dir(ROOT / "outputs" / "results")
    out_path = results_dir / "av_expression_audio_alignment.json"
    zip_path = data_dir / "LMVD_Feature.zip"

    if not zip_path.exists():
        save_json(
            {
                "n_samples": 0,
                "note": "LMVD_Feature.zip not found. Run lmvd-analysis/download_lmvd_features.py first "
                        "(~16GB download).",
            },
            out_path,
        )
        print(f"[skip] No LMVD feature zip - wrote placeholder to {out_path}")
        return

    labels_df = load_lmvd_labels(data_dir)
    label_by_id = dict(zip(labels_df["sample_id"], labels_df["label"]))

    # Spread the requested sample budget evenly across classes so the
    # alignment analysis isn't accidentally dominated by one label.
    wanted = []
    for label in (0, 1):
        class_ids = labels_df.loc[labels_df["label"] == label, "sample_id"].tolist()
        wanted.extend(class_ids[: args.max_samples // 2 + 1])
    wanted = sorted(set(wanted))[: args.max_samples]

    print(f"[info] extracting/caching up to {len(wanted)} samples from {zip_path.name} ...")
    sample_ids = extract_and_cache_samples(wanted, data_dir)
    print(f"[info] {len(sample_ids)}/{len(wanted)} samples cached (video+audio)")

    au_vals, audio_vals, classes = [], [], []
    for sid in sample_ids:
        result = _sample_alignment(sid, data_dir)
        if result is None:
            continue
        au_mean, audio_energy = result
        au_vals.append(au_mean)
        audio_vals.append(audio_energy)
        classes.append(int(label_by_id.get(sid, 0)))

    if len(au_vals) < 3:
        save_json({"n_samples": len(au_vals), "note": "Too few cached samples for correlation analysis."}, out_path)
        print(f"[skip] Only {len(au_vals)} usable samples - wrote placeholder to {out_path}")
        return

    au_arr, audio_arr, class_arr = np.array(au_vals), np.array(audio_vals), np.array(classes)
    overall_r = float(np.corrcoef(au_arr, audio_arr)[0, 1]) if len(au_arr) > 2 else 0.0

    by_class = {}
    for label, name in ((0, "normal"), (1, "depression")):
        mask = class_arr == label
        n = int(mask.sum())
        if n < 2:
            by_class[name] = {"n": n, "au_mean": None, "audio_energy_mean": None, "pearson_r": None}
            continue
        r = float(np.corrcoef(au_arr[mask], audio_arr[mask])[0, 1]) if n > 2 else 0.0
        by_class[name] = {
            "n": n,
            "au_mean": float(au_arr[mask].mean()),
            "audio_energy_mean": float(audio_arr[mask].mean()),
            "pearson_r": r,
        }

    report = {
        "n_samples": len(au_vals),
        "pearson_au_vs_audio_energy": overall_r,
        "au_intensity_mean": float(au_arr.mean()),
        "audio_energy_mean": float(audio_arr.mean()),
        "by_class": by_class,
        "raw_media": {
            "wav_count": 0,
            "transcript_count": 0,
            "sample_wav": None,
            "note": "Whisper STT recommended as future work once raw audio is available; not required for this VGGish<->AU alignment demo.",
        },
        "method": "Per-sample mean |AU| from TCN/OpenFace channels [0:17] vs RMS magnitude of VGGish frames as audio-energy proxy.",
        "limitations": [
            "No raw wav/transcripts in the current LMVD feature dump used here.",
            "VGGish embeddings are not literal loudness; treat correlation as exploratory.",
            "Expression captions in trimodal are lexicon-derived, not ASR transcripts.",
        ],
    }
    save_json(report, out_path)
    print(f"[ok] AV alignment over {len(au_vals)} samples -> {out_path}")
    print(f"     overall pearson r = {overall_r:.4f}")


if __name__ == "__main__":
    main()
