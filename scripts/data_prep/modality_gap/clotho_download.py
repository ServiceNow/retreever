#!/usr/bin/env python3
"""
Download Clotho v2.1 dataset (text-audio retrieval).

Clotho has 3 splits:
  - development  (~3840 clips, 5 captions each) → used as train
  - validation   (~1046 clips, 5 captions each) → used as val
  - evaluation   (~1045 clips, 5 captions each) → used as test

Audio is downloaded as 7z archives from Zenodo. Captions are CSVs.
No YouTube download needed — all audio is directly hosted on Zenodo.

Output structure:
    ./data/clotho/
    ├── raw/
    │   ├── clotho_audio_development.7z
    │   ├── clotho_audio_validation.7z
    │   ├── clotho_audio_evaluation.7z
    │   ├── clotho_captions_development.csv
    │   ├── clotho_captions_validation.csv
    │   └── clotho_captions_evaluation.csv
    └── audio/
        ├── development/   <- extracted .wav files
        ├── validation/
        └── evaluation/

Usage:
    pip install requests tqdm py7zr
    python clotho_download.py --output_dir ./data/clotho
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests

# ============================================================================
# CONFIG — Clotho v2.1 on Zenodo (record 4783391)
# ============================================================================

ZENODO_RECORD = "https://zenodo.org/record/4783391/files"

DOWNLOAD_FILES = {
    # audio archives
    "clotho_audio_development.7z":  f"{ZENODO_RECORD}/clotho_audio_development.7z",
    "clotho_audio_validation.7z":   f"{ZENODO_RECORD}/clotho_audio_validation.7z",
    "clotho_audio_evaluation.7z":   f"{ZENODO_RECORD}/clotho_audio_evaluation.7z",
    # caption CSVs
    "clotho_captions_development.csv": f"{ZENODO_RECORD}/clotho_captions_development.csv",
    "clotho_captions_validation.csv":  f"{ZENODO_RECORD}/clotho_captions_validation.csv",
    "clotho_captions_evaluation.csv":  f"{ZENODO_RECORD}/clotho_captions_evaluation.csv",
    # metadata CSVs (optional but useful)
    "clotho_metadata_development.csv": f"{ZENODO_RECORD}/clotho_metadata_development.csv",
    "clotho_metadata_validation.csv":  f"{ZENODO_RECORD}/clotho_metadata_validation.csv",
    "clotho_metadata_evaluation.csv":  f"{ZENODO_RECORD}/clotho_metadata_evaluation.csv",
}

DEFAULT_OUTPUT_DIR = "./data/clotho"

# ============================================================================


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download with resume support and progress bar."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ Already exists: {dest.name}")
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_pos = tmp.stat().st_size if tmp.exists() else 0

    print(f"  Downloading {desc or dest.name} ...")
    if resume_pos:
        print(f"    Resuming from {resume_pos / (1024**2):.1f} MB")

    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    try:
        r = requests.get(url, stream=True, timeout=120, headers=headers)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_pos
        mode = "ab" if resume_pos else "wb"
        with open(tmp, mode) as f, tqdm(total=total, initial=resume_pos, unit="B",
                                         unit_scale=True, leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)
        print(f"  ✓ Saved: {dest.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def extract_7z(archive: Path, dest_dir: Path) -> bool:
    """Extract a .7z archive using py7zr (preferred) or system 7z."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try py7zr first
    try:
        import py7zr
        print(f"  Extracting {archive.name} with py7zr ...")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=dest_dir)
        print(f"  ✓ Extracted to {dest_dir}")
        return True
    except ImportError:
        pass

    # Fall back to system 7z
    try:
        result = subprocess.run(
            ["7z", "x", str(archive), f"-o{dest_dir}", "-y"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"  ✓ Extracted to {dest_dir}")
            return True
        else:
            print(f"  ✗ 7z extraction failed: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("  ✗ Neither py7zr nor system 7z found.")
        print("    Install with: pip install py7zr  OR  apt-get install p7zip-full")
        return False


def count_wav_files(directory: Path) -> int:
    return len(list(directory.rglob("*.wav"))) if directory.exists() else 0


def main():
    parser = argparse.ArgumentParser(
        description="Download Clotho v2.1 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clotho_download.py --output_dir ./data/clotho
  python clotho_download.py --no_extract   # download archives but don't extract
        """,
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no_extract", action="store_true",
                        help="Download archives but skip extraction")
    parser.add_argument("--extract_only", action="store_true",
                        help="Skip download, only extract already-downloaded archives")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    audio_dir = output_dir / "audio"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CLOTHO v2.1 DOWNLOAD")
    print("=" * 80)
    print(f"Output: {output_dir}")
    print("=" * 80)

    # 1. Download all files
    if not args.extract_only:
        print("\nDownloading files from Zenodo ...")
        results = {}
        for filename, url in DOWNLOAD_FILES.items():
            dest = raw_dir / filename
            results[filename] = download_file(url, dest, desc=filename)

        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"\n⚠ {len(failed)} file(s) failed to download:")
            for f in failed:
                print(f"  - {f}")

    # 2. Extract audio archives
    if not args.no_extract:
        print("\nExtracting audio archives ...")
        for split in ["development", "validation", "evaluation"]:
            archive = raw_dir / f"clotho_audio_{split}.7z"
            split_audio_dir = audio_dir / split

            existing = count_wav_files(split_audio_dir)
            if existing > 0:
                print(f"  ✓ {split}: {existing} WAV files already extracted")
                continue

            if not archive.exists():
                print(f"  ✗ Archive not found: {archive}")
                continue

            extract_7z(archive, split_audio_dir)

    # 3. Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for split in ["development", "validation", "evaluation"]:
        n_wav = count_wav_files(audio_dir / split)
        n_csv = 1 if (raw_dir / f"clotho_captions_{split}.csv").exists() else 0
        print(f"  {split:15s}: {n_wav:5d} WAV files  |  captions CSV: {'✓' if n_csv else '✗'}")

    print(f"\nNext step:")
    print(f"  python prepare_clotho.py --data_dir {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
