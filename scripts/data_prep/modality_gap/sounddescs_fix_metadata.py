#!/usr/bin/env python3
"""
Fix SoundDescs metadata — re-match BBC audio filenames to CLAPv2/SoundDescs
descriptions from HuggingFace.

Problem: sounddescs_download.py matched by exact audio_id == HF index, but
the naming conventions differ. This script inspects both, finds the mapping,
and rewrites metadata.jsonl with correct text descriptions.

Usage:
    python fix_sounddescs_metadata.py
    python fix_sounddescs_metadata.py --data_dir ./data/sounddescs
    python fix_sounddescs_metadata.py --dry_run   # show match stats without writing
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict

DEFAULT_DATA_DIR = "./data/sounddescs"
HF_CACHE_DIR = "./hf_cache"


def load_hf_descriptions(hf_cache_dir: str) -> dict:
    """Load CLAPv2/SoundDescs from HF and return raw dataset + index map."""
    from datasets import load_dataset
    print("Loading HF dataset CLAPv2/SoundDescs ...")
    ds = load_dataset("CLAPv2/SoundDescs", cache_dir=hf_cache_dir)

    all_rows = []
    for split in ds:
        for row in ds[split]:
            row_dict = dict(row)
            row_dict["hf_split"] = split
            all_rows.append(row_dict)

    print(f"  Total HF rows: {len(all_rows)}")
    if all_rows:
        sample = all_rows[0]
        print(f"  Sample HF index: '{sample.get('index', 'N/A')}'")
        print(f"  Sample HF text:  '{str(sample.get('text', ''))[:80]}'")
        print(f"  HF columns: {list(sample.keys())}")

    return ds, all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Fix SoundDescs metadata by re-matching HF descriptions"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--hf_cache_dir", default=HF_CACHE_DIR)
    parser.add_argument("--dry_run", action="store_true",
                        help="Show match stats without writing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    audio_dir = data_dir / "audio"
    meta_dir = data_dir / "metadata"

    print("=" * 80)
    print("FIX SOUNDDESCS METADATA")
    print("=" * 80)

    # 1. Get list of audio files on disk
    wav_files = sorted(audio_dir.glob("*.wav"))
    audio_stems = {w.stem for w in wav_files}
    print(f"\nAudio files on disk: {len(wav_files)}")
    if wav_files:
        print(f"  Sample filenames:")
        for w in wav_files[:5]:
            print(f"    {w.stem}")

    # 2. Load HF descriptions
    ds, all_rows = load_hf_descriptions(args.hf_cache_dir)

    # 3. Build multiple index maps for matching
    # Try exact match first, then normalized versions
    by_exact = {}
    by_lower = {}
    by_stripped = {}  # remove common prefixes

    for row in all_rows:
        idx = row.get("index", "")
        if not idx:
            continue
        by_exact[idx] = row
        by_lower[idx.lower()] = row
        # Try stripping common prefix patterns
        for prefix in ["bbc_", "BBC_"]:
            if idx.startswith(prefix):
                by_stripped[idx[len(prefix):].lower()] = row

    print(f"\n  HF index samples:")
    for row in all_rows[:10]:
        print(f"    '{row.get('index', '')}'")

    # 4. Try matching strategies
    strategies = [
        ("exact",          lambda stem: by_exact.get(stem)),
        ("lowercase",      lambda stem: by_lower.get(stem.lower())),
        ("strip_bbc",      lambda stem: by_exact.get(stem.replace("bbc_", "").replace("BBC_", ""))),
        ("hf_add_bbc",     lambda stem: by_exact.get(f"bbc_{stem}")),
        ("stem_in_hf",     lambda stem: next((r for idx, r in by_exact.items() if stem in idx), None)),
        ("hf_in_stem",     lambda stem: next((r for idx, r in by_exact.items() if idx in stem), None)),
        ("lower_stripped", lambda stem: by_stripped.get(stem.lower().replace("bbc_", ""))),
    ]

    print(f"\n  Trying match strategies on first 100 audio files:")
    for name, fn in strategies:
        matched = sum(1 for s in list(audio_stems)[:100] if fn(s) is not None)
        print(f"    {name:20s}: {matched}/100")

    # 5. Find the best strategy
    best_name = None
    best_fn = None
    best_count = 0
    for name, fn in strategies:
        matched = sum(1 for s in list(audio_stems)[:100] if fn(s) is not None)
        if matched > best_count:
            best_count = matched
            best_name = name
            best_fn = fn

    if best_count == 0:
        # Try a more aggressive approach: match by numeric suffix or partial overlap
        print("\n  No strategy matched. Trying fuzzy approaches...")

        # Build reverse map: for each HF row, extract potential matching keys
        hf_by_parts = {}
        for row in all_rows:
            idx = row.get("index", "")
            # Extract just alphanumeric parts
            parts = idx.lower().replace("-", "_").replace(" ", "_")
            hf_by_parts[parts] = row

        # Try matching audio stems after normalization
        def normalize(s):
            return s.lower().replace("-", "_").replace(" ", "_")

        norm_fn = lambda stem: hf_by_parts.get(normalize(stem))
        matched = sum(1 for s in list(audio_stems)[:100] if norm_fn(s) is not None)
        print(f"    normalized:          {matched}/100")

        if matched > best_count:
            best_count = matched
            best_name = "normalized"
            best_fn = norm_fn

    if best_count == 0:
        print("\n  ERROR: Could not find any matching strategy.")
        print("  This likely means the HF index format is completely different")
        print("  from the BBC audio filenames. Manual investigation needed.")
        print(f"\n  Audio stems sample: {list(audio_stems)[:5]}")
        print(f"  HF index sample:   {[r.get('index','') for r in all_rows[:5]]}")
        return

    print(f"\n  Best strategy: {best_name} ({best_count}/100 matched)")

    # 6. Apply best strategy to all files
    print(f"\n  Matching all {len(wav_files)} audio files ...")
    records = []
    matched = 0
    unmatched = 0

    for wav in wav_files:
        audio_id = wav.stem
        row = best_fn(audio_id)

        text = ""
        raw_text = []
        audio_len = 0.0
        split = ""

        if row and row.get("text", "").strip():
            text = row["text"]
            raw_text = row.get("raw_text", [])
            audio_len = row.get("audio_len", 0.0)
            split = row.get("hf_split", row.get("split", ""))
            matched += 1
        else:
            unmatched += 1

        records.append({
            "audio_id": audio_id,
            "audio_path": f"audio/{wav.name}",
            "text": text,
            "raw_text": raw_text,
            "audio_len": audio_len,
            "split": split,
        })

    print(f"\n  Results:")
    print(f"    Matched (with text): {matched}")
    print(f"    Unmatched:           {unmatched}")
    print(f"    Total:               {len(records)}")

    if args.dry_run:
        print("\n  DRY RUN — not writing metadata")
        return

    # 7. Write fixed metadata
    meta_path = meta_dir / "metadata.jsonl"
    backup_path = meta_dir / "metadata.jsonl.bak"

    if meta_path.exists():
        import shutil
        shutil.copy2(meta_path, backup_path)
        print(f"\n  Backed up old metadata to {backup_path}")

    with open(meta_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Written fixed metadata: {meta_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  python prepare_sounddescs.py --data_dir {data_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
