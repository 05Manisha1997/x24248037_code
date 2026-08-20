from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import project_root


def _check_file(path: Path, label: str) -> bool:
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"[ok] {label}: {path} ({size_kb:.1f} KB)")
        return True
    print(f"[missing] {label}: {path}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify/download raw datasets for the pipeline.")
    parser.add_argument(
        "--skip-lmvd", action="store_true",
        help="Skip LMVD label download (video/audio features are handled separately by "
             "lmvd-analysis/download_lmvd_features.py, which is a much larger ~16GB download).",
    )
    args = parser.parse_args()

    root = project_root()
    ok = True
    ok &= _check_file(root / "datasets" / "structured_data.csv", "structured survey data")
    ok &= _check_file(root / "datasets" / "unstructured_data.csv", "unstructured text data")

    if not args.skip_lmvd:
        from src.data.loaders import load_lmvd_labels
        labels = load_lmvd_labels(root / "lmvd-analysis" / "data")
        print(f"[ok] LMVD labels: {len(labels)} samples")
    else:
        print("[skip] LMVD label download (--skip-lmvd)")

    print("All core datasets present." if ok else "Some datasets are missing - see messages above.")


if __name__ == "__main__":
    main()
