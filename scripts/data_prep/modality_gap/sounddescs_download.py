#!/usr/bin/env python3
"""
Download SoundDescs dataset (text-audio retrieval).

SoundDescs has ~32,979 audio clips with text descriptions, sourced from the
BBC Sound Effects library (https://sound-effects.bbcrewind.co.uk/).

Audio: downloaded as .wav.zip archives from BBC Rewind servers.
Text:  descriptions loaded from HuggingFace (CLAPv2/SoundDescs).

The download links are from the audio-retrieval-benchmark repo:
    https://github.com/akoepke/audio-retrieval-benchmark

NOT in LAION-Audio-630K training data -> clean OOD eval for CLAP.
(LAION-Audio-630K uses BBC Sound Effects but with a DIFFERENT set of
captions/processing. SoundDescs has its own human-written descriptions.)

Output structure:
    ./data/sounddescs/
    ├── raw/
    │   └── download_links_renamed.txt
    ├── zip_audios/            <- intermediate zips (cleaned after extraction)
    ├── audio/                 <- extracted WAV files
    └── metadata/
        └── metadata.jsonl     <- {audio_id, audio_path, text, raw_text}

Usage:
    pip install requests tqdm datasets wget
    python sounddescs_download.py --output_dir ./data/sounddescs
    python sounddescs_download.py --output_dir ./data/sounddescs --workers 2
"""

from __future__ import annotations

import json
import os
import argparse
import zipfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# ============================================================================
# CONFIG
# ============================================================================

DOWNLOAD_LINKS_URL = (
    "https://raw.githubusercontent.com/akoepke/audio-retrieval-benchmark/"
    "main/sounddescs_data/download_links_renamed.txt"
)

DEFAULT_OUTPUT_DIR = "./data/sounddescs"
HF_CACHE_DIR = "./hf_cache"

# ============================================================================


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download with resume support and progress bar."""
    if dest.exists() and dest.stat().st_size > 0:
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    try:
        r = requests.get(url, stream=True, timeout=120, headers=headers)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_pos
        mode = "ab" if resume_pos else "wb"
        with open(tmp, mode) as f, tqdm(total=total, initial=resume_pos, unit="B",
                                         unit_scale=True, desc=desc, leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)
        return True
    except Exception:
        return False


def download_one_audio(entry: str, zip_dir: Path) -> tuple[bool, str]:
    """Download one BBC audio zip. Entry is a full URL line from download_links_renamed.txt."""
    # Extract audio_id from the URL: ...rename=bbc_king-pengu_nhu0503004
    try:
        if "rename=" in entry:
            audio_id = entry.split("rename=")[1].strip()
        else:
            # fallback: extract from zip path
            audio_id = entry.split("/zip/")[1].split(".wav.zip")[0].lower()
    except (IndexError, AttributeError):
        return False, f"bad_url: {entry[:80]}"

    dest_dir = zip_dir / audio_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(dest_dir.glob("*.zip"))
    if existing:
        return True, audio_id

    dest_file = dest_dir / f"{audio_id}.wav.zip"
    try:
        r = requests.get(entry.strip(), timeout=120)
        r.raise_for_status()
        with open(dest_file, "wb") as f:
            f.write(r.content)
        return True, audio_id
    except Exception as e:
        return False, f"{audio_id}: {e}"


def unzip_one(zip_dir: Path, audio_id: str, audio_dir: Path) -> bool:
    """Unzip one audio file from zip_audios/<audio_id>/ to audio/<audio_id>.wav."""
    out_path = audio_dir / f"{audio_id}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    id_dir = zip_dir / audio_id
    zips = list(id_dir.glob("*.zip"))
    if not zips:
        return False

    try:
        with zipfile.ZipFile(zips[0], "r") as zf:
            names = zf.namelist()
            wav_names = [n for n in names if n.lower().endswith(".wav")]
            if not wav_names:
                return False
            # Extract the first wav file
            wav_data = zf.read(wav_names[0])
            with open(out_path, "wb") as f:
                f.write(wav_data)
        return True
    except (zipfile.BadZipFile, Exception):
        return False


def load_hf_descriptions(hf_cache_dir: str | None) -> dict[str, dict]:
    """Load text descriptions from CLAPv2/SoundDescs on HuggingFace.

    Returns dict keyed by audio filename stem -> {text, raw_text, audio_len}.
    """
    from datasets import load_dataset
    print("  Loading descriptions from HuggingFace (CLAPv2/SoundDescs) ...")
    ds = load_dataset("CLAPv2/SoundDescs", cache_dir=hf_cache_dir)

    desc_map = {}
    for split in ds:
        for row in ds[split]:
            idx = row.get("index", "")
            desc_map[idx] = {
                "text":      row.get("text", ""),
                "raw_text":  row.get("raw_text", []),
                "audio_len": row.get("audio_len", 0.0),
                "split":     split,
            }
    print(f"  Loaded {len(desc_map)} descriptions across {list(ds.keys())}")
    return desc_map


def main():
    parser = argparse.ArgumentParser(
        description="Download SoundDescs dataset (BBC Sound Effects + text descriptions)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sounddescs_download.py --output_dir ./data/sounddescs
  python sounddescs_download.py --workers 2         # BBC limits concurrency
  python sounddescs_download.py --limit 100         # download only first 100
  python sounddescs_download.py --extract_only      # just unzip existing archives
  python sounddescs_download.py --metadata_only     # just download HF descriptions
        """,
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel download workers (BBC may block >2)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Download only first N audio files (0=all)")
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--metadata_only", action="store_true",
                        help="Only download HF descriptions, skip audio")
    parser.add_argument("--no_extract", action="store_true")
    parser.add_argument("--keep_zips", action="store_true",
                        help="Keep zip_audios/ after extraction")
    parser.add_argument("--hf_cache_dir", default=HF_CACHE_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    zip_dir = output_dir / "zip_audios"
    audio_dir = output_dir / "audio"
    meta_dir = output_dir / "metadata"
    for d in (raw_dir, zip_dir, audio_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SOUNDDESCS DOWNLOAD")
    print("=" * 80)
    print(f"Output:  {output_dir}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    # 1. Download the links file from GitHub
    links_path = raw_dir / "download_links_renamed.txt"
    if not links_path.exists():
        print("\n[1/4] Downloading link list from audio-retrieval-benchmark ...")
        download_file(DOWNLOAD_LINKS_URL, links_path, desc="download_links_renamed.txt")
    else:
        print(f"\n[1/4] Link list already present: {links_path}")

    with open(links_path) as f:
        all_links = [line.strip() for line in f if line.strip()]
    print(f"  Total audio links: {len(all_links)}")

    if args.limit > 0:
        all_links = all_links[:args.limit]
        print(f"  Limited to first {args.limit}")

    # 2. Download audio zips from BBC
    if not args.extract_only and not args.metadata_only:
        print(f"\n[2/4] Downloading audio from BBC Rewind ({len(all_links)} files) ...")
        stats = {"ok": 0, "fail": 0}
        lock = threading.Lock()

        def worker(entry):
            ok, msg = download_one_audio(entry, zip_dir)
            with lock:
                if ok:
                    stats["ok"] += 1
                else:
                    stats["fail"] += 1
            return ok, msg

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, e): e for e in all_links}
            with tqdm(total=len(futures), desc="Downloading", unit="clip") as pbar:
                for fut in as_completed(futures):
                    pbar.update(1)

        print(f"  Downloaded: {stats['ok']}  |  Failed: {stats['fail']}")

    # 3. Extract zips to audio/
    if not args.no_extract and not args.metadata_only:
        print(f"\n[3/4] Extracting audio ...")
        audio_ids = [d.name for d in zip_dir.iterdir() if d.is_dir()]
        extracted = 0
        failed = 0
        for aid in tqdm(audio_ids, desc="Extracting", unit="clip"):
            if unzip_one(zip_dir, aid, audio_dir):
                extracted += 1
            else:
                failed += 1
        print(f"  Extracted: {extracted}  |  Failed: {failed}")

        if not args.keep_zips and extracted > 0:
            print("  Cleaning up zip_audios/ ...")
            import shutil
            shutil.rmtree(zip_dir, ignore_errors=True)

    # 4. Download text descriptions from HuggingFace and build metadata
    print(f"\n[4/4] Loading text descriptions from HuggingFace ...")
    desc_map = load_hf_descriptions(args.hf_cache_dir)

    # Match audio files to descriptions
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"  Audio files on disk: {len(wav_files)}")

    records = []
    matched = 0
    for wav in wav_files:
        audio_id = wav.stem
        desc = desc_map.get(audio_id, {})
        records.append({
            "audio_id":  audio_id,
            "audio_path": f"audio/{wav.name}",
            "text":       desc.get("text", ""),
            "raw_text":   desc.get("raw_text", []),
            "audio_len":  desc.get("audio_len", 0.0),
            "split":      desc.get("split", ""),
        })
        if desc.get("text"):
            matched += 1

    meta_path = meta_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Metadata saved: {meta_path}")
    print(f"  Matched audio+text: {matched} / {len(records)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Audio files: {len(wav_files)}")
    print(f"  Descriptions: {len(desc_map)}")
    print(f"  Matched:      {matched}")
    print(f"\nNext step:")
    print(f"  python prepare_sounddescs.py --data_dir {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
