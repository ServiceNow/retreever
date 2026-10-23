#!/usr/bin/env python3
"""
Convert downloaded Clotho v2.1 dataset to ReTreever format for text-audio retrieval.

Run clotho_download.py first.

Clotho structure:
  - development  (~3840 clips, 5 captions each) -> train
  - validation   (~1046 clips, 5 captions each) -> val
  - evaluation   (~1045 clips, 5 captions each) -> test

One sample per (clip, caption) pair — mirrors COCO's 5 captions per image.
  question    = one of the 5 human-written captions
  context     = relative path to .wav file
  context_uid = unique int per clip (5 samples share the same context_uid)

Output: HuggingFace DatasetDict {train, val, test, cuid2text}
saved at <data_dir>/dataset/

Usage:
    python prepare_clotho.py --data_dir ./data/clotho
    python prepare_clotho.py --data_dir ./data/clotho --verify_only
    python prepare_clotho.py --data_dir ./data/clotho --stats
"""

from __future__ import annotations
import csv
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict


# ============================================================================
# HELPERS
# ============================================================================

SPLIT_MAP = {
    # clotho audio subfolder -> output split name
    "development": "train",
    "validation":  "val",
    "evaluation":  "test",
}

CAPTION_COLS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]


def load_clotho_csv(csv_path: Path) -> list[dict]:
    """
    Parse a Clotho captions CSV.

    Columns: file_name, caption_1, caption_2, caption_3, caption_4, caption_5
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ============================================================================
# CONVERSION
# ============================================================================

def convert_clotho_to_retreever(data_dir: Path) -> DatasetDict | None:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    audio_base = data_dir / "audio"

    print("=" * 80)
    print("CLOTHO v2.1 -> RETREEVER FORMAT")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("=" * 80)

    uid_counter = [0]
    uid_to_meta: dict[int, dict] = {}
    all_splits: dict[str, list[dict]] = {}

    for clotho_split, output_split in SPLIT_MAP.items():
        csv_path = raw_dir / f"clotho_captions_{clotho_split}.csv"
        audio_dir = audio_base / clotho_split

        if not csv_path.exists():
            print(f"  ✗ CSV not found: {csv_path}")
            return None
        if not audio_dir.exists():
            print(f"  ✗ Audio dir not found: {audio_dir}")
            return None

        rows = load_clotho_csv(csv_path)
        print(f"\n  {clotho_split} -> {output_split}: {len(rows)} clips in CSV")

        samples = []
        missing = 0

        for row in rows:
            fname = row["file_name"]
            audio_path = audio_dir / fname
            if not audio_path.exists():
                missing += 1
                continue

            # Relative path from data_dir root
            rel_path = f"audio/{clotho_split}/{fname}"

            uid = uid_counter[0]
            uid_counter[0] += 1
            uid_to_meta[uid] = {
                "context_uid": uid,
                "context": rel_path,
                "file_name": fname,
            }

            captions = [row[c].strip() for c in CAPTION_COLS if row.get(c, "").strip()]

            for cap_idx, caption in enumerate(captions):
                samples.append({
                    "context":         rel_path,
                    "contexts_list":   [rel_path],
                    "titles_list":     [caption],
                    "useful_contexts": [1],
                    "question":        caption,
                    "answer":          fname,
                    "sample_idx":      len(samples),
                    "dataset":         "clotho",
                    "context_uid":     uid,
                    # Clotho-specific
                    "file_name":       fname,
                    "caption_idx":     cap_idx,
                })

        if missing:
            print(f"    ⚠ {missing} clips missing from disk, skipped")
        print(f"    ✓ {len(samples)} (clip, caption) pairs from {len(rows) - missing} clips")

        all_splits[output_split] = samples

    # cuid2text
    cuid2text = sorted(uid_to_meta.values(), key=lambda x: x["context_uid"])

    dataset_dict = DatasetDict({
        "train":     Dataset.from_list(all_splits["train"]),
        "val":       Dataset.from_list(all_splits["val"]),
        "test":      Dataset.from_list(all_splits["test"]),
        "cuid2text": Dataset.from_list(cuid2text),
    })

    print(f"\n{dataset_dict}")
    return dataset_dict


# ============================================================================
# VERIFY / STATS
# ============================================================================

def verify_dataset(data_dir: Path):
    from datasets import load_from_disk
    data_dir = Path(data_dir)
    print("\n" + "=" * 80)
    print("VERIFYING DATASET")
    print("=" * 80)
    try:
        ds = load_from_disk(str(data_dir / "dataset"))
        print(ds)
        for split in ["train", "val", "test"]:
            s = ds[split]
            print(f"\n  {split}: {len(s)} samples")
            if len(s) > 0:
                ex = s[0]
                print(f"    question: {ex['question'][:80]}")
                print(f"    context:  {ex['context']}")
                ok = (data_dir / ex["context"]).exists()
                print(f"    file exists: {'✓' if ok else '✗'}")
        print(f"\n  cuid2text: {len(ds['cuid2text'])} unique clips")
        print("\n✓ Verification complete!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def show_statistics(data_dir: Path):
    from datasets import load_from_disk
    data_dir = Path(data_dir)
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    try:
        ds = load_from_disk(str(data_dir / "dataset"))
        for split in ["train", "val", "test"]:
            s = ds[split]
            n_clips = len(set(s["context_uid"]))
            n_caps = len(s) / max(n_clips, 1)
            print(f"\n  {split}: {len(s):,} samples  |  {n_clips:,} clips  |  {n_caps:.1f} captions/clip")
        print(f"\n  Total unique clips: {len(ds['cuid2text']):,}")
    except Exception as e:
        print(f"Could not load statistics: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert Clotho v2.1 to ReTreever format")
    parser.add_argument("--data_dir", default="./data/clotho")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.verify_only:
        verify_dataset(data_dir)
        return
    if args.stats:
        show_statistics(data_dir)
        return

    ds = convert_clotho_to_retreever(data_dir)
    if ds is None:
        print("Conversion failed.")
        return

    out = data_dir / "dataset"
    print(f"\nSaving dataset to {out} ...")
    ds.save_to_disk(str(out))
    print("✓ Saved")

    verify_dataset(data_dir)
    show_statistics(data_dir)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\n  Layout:")
    print(f"    {data_dir}/")
    print(f"    ├── audio/development|validation|evaluation/")
    print(f"    ├── raw/   (CSVs + 7z archives)")
    print(f"    └── dataset/ (train / val / test / cuid2text)")
    print(f"\n  Load with:")
    print(f"    from datasets import load_from_disk")
    print(f"    ds = load_from_disk('{out}')")
    print("=" * 80)


if __name__ == "__main__":
    main()
