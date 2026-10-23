#!/usr/bin/env python3
"""
Download AudioCaps audio from YouTube via yt-dlp.

Metadata is fetched from HuggingFace (d0rj/audiocaps), which mirrors the
official AudioCaps repo (github.com/cdjkim/audiocaps).  Audio is not bundled
in that dataset, so each 10-second clip is downloaded from YouTube.

Dataset: ~46k clips with train / validation / test splits.
  - Train: 1 caption per clip
  - Validation/Test: ~5 captions per clip (all kept during prepare step)

Output structure:
    ./data/audiocaps/
    ├── audio/
    │   ├── train/          <- 16kHz mono WAV files
    │   ├── validation/
    │   └── test/
    └── metadata/
        ├── train.jsonl
        ├── validation.jsonl
        └── test.jsonl

Usage:
    pip install datasets yt-dlp tqdm
    python audiocaps_download.py
    python audiocaps_download.py --workers 48
    python audiocaps_download.py --splits validation test   # resume missing splits
    python audiocaps_download.py --max_per_split 200        # quick test
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "./data/audiocaps"
DEFAULT_HF_ID = "d0rj/audiocaps"
DEFAULT_WORKERS = 32
SAMPLE_RATE = 16000
CLIP_DURATION = 10.0   # AudioCaps clips are always 10 seconds
MAX_RETRIES = 3
RETRY_SLEEP = 5

def _find_ytdlp() -> str:
    found = shutil.which("yt-dlp")
    if found:
        return found
    alt = Path(sys.executable).parent / "yt-dlp"
    if alt.exists():
        return str(alt)
    return "yt-dlp"

YT_DLP = _find_ytdlp()

COOKIE_ARGS: list[str] = []

HF_SPLITS = ["train", "validation", "test"]

# Map HF split name -> output directory/file name
SPLIT_OUT_NAME = {"train": "train", "validation": "val", "test": "test"}


def clip_filename(ytid: str, start_s: float) -> str:
    return f"{ytid}_{int(round(start_s * 1000)):07d}.wav"


def download_clip(
    ytid: str,
    start_s: float,
    out_path: Path,
    sample_rate: int = SAMPLE_RATE,
    max_retries: int = MAX_RETRIES,
) -> tuple[bool, str]:
    """Download a single 10-second clip using yt-dlp."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, str(out_path)

    end_s = start_s + CLIP_DURATION
    url = f"https://www.youtube.com/watch?v={ytid}"
    tmp_stem = str(out_path.with_suffix("")) + ".part"

    cmd = [
        YT_DLP, "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--download-sections", f"*{start_s:.3f}-{end_s:.3f}",
        "--force-keyframes-at-cuts",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "--no-playlist", "--quiet", "--no-warnings",
        "--socket-timeout", "60",
        "--retries", "3",
        *COOKIE_ARGS,
        "-o", tmp_stem,
        url,
    ]

    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            written = Path(tmp_stem + ".wav")
            if written.exists() and written.stat().st_size > 0:
                written.rename(out_path)
                return True, str(out_path)

            if result.returncode != 0:
                err = result.stderr.strip()[:300]
                if "Sign in" in err or "bot" in err.lower() or ("confirm" in err and "youtube" in err.lower()):
                    return False, f"bot_blocked: {ytid}"
                if any(x in err for x in ["Video unavailable", "Private video", "removed", "not available"]):
                    return False, f"unavailable: {ytid}"
                if attempt < max_retries - 1:
                    time.sleep(RETRY_SLEEP * (attempt + 1))
                    continue
                return False, f"yt-dlp failed (rc={result.returncode}): {err}"
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(RETRY_SLEEP)
                continue
            return False, f"timeout: {ytid}"
        except Exception as e:
            return False, f"error: {e}"

    return False, f"all {max_retries} retries failed: {ytid}"


def process_split(
    rows: list[dict],
    split: str,
    audio_dir: Path,
    workers: int,
    dry_run: bool,
) -> list[dict]:
    """Download clips for one split; return metadata list for downloaded clips."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    ok_records: list[dict] = []
    fail_log: list[str] = []
    lock = __import__("threading").Lock()

    def worker(row: dict) -> dict | None:
        ytid = row["youtube_id"]
        start_s = float(row["start_time"])
        fname = clip_filename(ytid, start_s)
        out_path = audio_dir / fname

        if dry_run:
            return {**row, "audio_path": f"audio/{split}/{fname}"}

        success, msg = download_clip(ytid, start_s, out_path)
        if success:
            return {**row, "audio_path": f"audio/{split}/{fname}", "split": split}
        else:
            with lock:
                fail_log.append(msg)
            return None

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in rows:
            futures.append(pool.submit(worker, row))

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {split}"):
            result = fut.result()
            if result is not None:
                ok_records.append(result)

    if fail_log:
        print(f"    ⚠ {len(fail_log)} failed (unavailable/timeout)")
    print(f"    ✓ {len(ok_records)}/{len(rows)} clips downloaded")
    return ok_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AudioCaps via yt-dlp (metadata from d0rj/audiocaps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audiocaps_download.py
  python audiocaps_download.py --workers 48
  python audiocaps_download.py --splits validation test
  python audiocaps_download.py --max_per_split 200   # quick test
        """,
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf_dataset_id", default=DEFAULT_HF_ID,
                        help=f"HuggingFace metadata dataset (default: {DEFAULT_HF_ID})")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--splits", nargs="+", default=HF_SPLITS,
                        choices=HF_SPLITS)
    parser.add_argument("--max_per_split", type=int, default=None,
                        help="Limit clips per split (for testing)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Write metadata only, skip yt-dlp")
    parser.add_argument("--cookies", default=None, metavar="FILE",
                        help="Netscape cookies.txt for YouTube authentication")
    parser.add_argument("--cookies-from-browser", default=None, dest="cookies_from_browser",
                        metavar="BROWSER",
                        help="Extract cookies from browser, e.g. chrome, firefox")
    args = parser.parse_args()

    global COOKIE_ARGS
    if args.cookies:
        COOKIE_ARGS = ["--cookies", args.cookies]
        print(f"Using cookies file: {args.cookies}")
    elif args.cookies_from_browser:
        COOKIE_ARGS = ["--cookies-from-browser", args.cookies_from_browser]
        print(f"Extracting cookies from browser: {args.cookies_from_browser}")
    else:
        print("⚠  No --cookies / --cookies-from-browser supplied.")
        print("   YouTube now requires authentication for many downloads.")
        print("   See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp")

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUDIOCAPS DOWNLOAD (metadata: d0rj/audiocaps  |  audio: yt-dlp)")
    print("=" * 80)
    print(f"Output:  {output_dir}")
    print(f"Workers: {args.workers}")
    print("=" * 80)

    from datasets import load_dataset
    print(f"\nLoading metadata from '{args.hf_dataset_id}' ...")
    ds = load_dataset(args.hf_dataset_id, cache_dir=args.hf_cache_dir)
    print(f"✓ Loaded: {ds}")

    for split in args.splits:
        hf_split = ds[split]
        rows = [dict(row) for row in hf_split]
        if args.max_per_split:
            rows = rows[: args.max_per_split]

        out_split = SPLIT_OUT_NAME[split]
        print(f"\n  {split} -> {out_split}: {len(rows)} clips")
        audio_dir = output_dir / "audio" / out_split
        records = process_split(rows, out_split, audio_dir, args.workers, args.dry_run)

        meta_path = metadata_dir / f"{out_split}.jsonl"
        with open(meta_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"    ✓ Metadata saved: {meta_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python prepare_audiocaps.py --data_dir {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
