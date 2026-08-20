"""Download and extract LMVD_Feature.zip from Figshare (~16 GB)."""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

FIGSHARE_URL = "https://ndownloader.figshare.com/files/45872469"
DATA_DIR = Path(__file__).resolve().parent / "data"
ZIP_PATH = DATA_DIR / "LMVD_Feature.zip"
LOG_PATH = DATA_DIR / "download.log"
CHUNK_BYTES = 4 * 1024 * 1024


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def feature_dirs_ready() -> bool:
    video_dir = DATA_DIR / "tcnfeature"
    audio_dir = DATA_DIR / "audiofeature"
    return (
        video_dir.exists()
        and audio_dir.exists()
        and any(video_dir.glob("*.npy"))
        and any(audio_dir.glob("*.npy"))
    )


def zip_is_complete() -> bool:
    """Check download completeness by size (skip testzip — too memory-heavy for 16 GB)."""
    if not ZIP_PATH.exists():
        return False
    size = ZIP_PATH.stat().st_size
    return size >= 15_500_000_000  # ~16.32 GB expected


def download_zip() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0

    headers: dict[str, str] = {}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        log(f"Resuming download from {downloaded / 1e9:.2f} GB")

    request = Request(FIGSHARE_URL, headers=headers)
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        if downloaded and status == 200:
            log("Server did not resume; restarting download")
            downloaded = 0
            mode = "wb"
            total = int(response.headers.get("Content-Length", 0))
        elif status == 206:
            mode = "ab"
            total = downloaded + int(response.headers.get("Content-Length", 0))
        else:
            mode = "wb"
            total = int(response.headers.get("Content-Length", 0))

        log(f"Downloading to {ZIP_PATH} (target ~{total / 1e9:.2f} GB)")
        last_report = time.time()
        with ZIP_PATH.open(mode) as handle:
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 10:
                    pct = 100 * downloaded / total if total else 0
                    log(f"  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)")
                    last_report = now

    log(f"Download complete: {downloaded / 1e9:.2f} GB")
    if not zip_is_complete():
        log("Download incomplete — re-run to resume.")
        return 1
    return 0


def extract_zip() -> None:
    log(f"Extracting {ZIP_PATH.name} into {DATA_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        members = archive.namelist()
        total = len(members)
        for i, member in enumerate(members, start=1):
            archive.extract(member, DATA_DIR)
            if i % 200 == 0 or i == total:
                log(f"  extracted {i}/{total} files")
    video_count = len(list((DATA_DIR / "tcnfeature").glob("*.npy"))) if (DATA_DIR / "tcnfeature").exists() else 0
    audio_count = len(list((DATA_DIR / "audiofeature").glob("*.npy"))) if (DATA_DIR / "audiofeature").exists() else 0
    log(f"Extraction done. tcnfeature={video_count} npy, audiofeature={audio_count} npy")


def main() -> int:
    if feature_dirs_ready():
        log("Feature folders already present. Nothing to do.")
        return 0

    if not zip_is_complete():
        try:
            if download_zip() != 0:
                return 1
        except Exception as exc:
            log(f"Download failed: {exc}")
            return 1

    if feature_dirs_ready():
        log("Ready. Feature folders present.")
        return 0

    if zip_is_complete():
        log("Zip complete. Skipping full extraction (training reads TCN/audio from zip + Audio_feature/).")
        return 0

    if not feature_dirs_ready():
        try:
            extract_zip()
        except Exception as exc:
            log(f"Extraction failed: {exc}")
            return 1

    if feature_dirs_ready():
        log("Ready. Re-run lmvd_analysis.ipynb Section 6 to preview landmarks and audio.")
        return 0

    log("Extraction finished but expected .npy folders were not found. Check data/ layout.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
