#!/usr/bin/env python3
"""
Convert downloaded AudioCaps dataset to ReTreever format for text-audio retrieval.

Run audiocaps_download.py first.

AudioCaps structure after download:
    ./data/audiocaps/
    ├── audio/
    │   ├── train/      <youtube_id>_<start_ms>.wav
    │   ├── val/
    │   └── test/
    └── metadata/
        ├── train.jsonl    {audiocap_id, youtube_id, start_time, caption, audio_path}
        ├── val.jsonl
        └── test.jsonl

Train: 1 caption per clip (1 sample per clip).
Val/Test: multiple captions per clip (grouped by audio_path -> one sample per
caption, like COCO's 5 captions per image). Clips sharing the same audio_path
share a context_uid.

Usage:
    python prepare_audiocaps.py --data_dir ./data/audiocaps
    python prepare_audiocaps.py --data_dir ./data/audiocaps --verify_only
    python prepare_audiocaps.py --data_dir ./data/audiocaps --stats
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict


# ============================================================================
# HELPERS
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# CONVERSION
# ============================================================================

def convert_audiocaps_to_retreever(data_dir: Path) -> DatasetDict | None:
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "metadata"

    print("=" * 80)
    print("AUDIOCAPS -> RETREEVER FORMAT")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("=" * 80)

    for split in ["train", "val", "test"]:
        if not (metadata_dir / f"{split}.jsonl").exists():
            print(f"  ✗ Missing: {metadata_dir / split}.jsonl")
            return None

    uid_counter = [0]
    # Map audio_path -> context_uid (same clip -> same uid across all captions)
    path_to_uid: dict[str, int] = {}
    uid_to_meta: dict[int, dict] = {}
    all_splits: dict[str, list[dict]] = {}

    for split in ["train", "val", "test"]:
        records = load_jsonl(metadata_dir / f"{split}.jsonl")
        print(f"\n  {split}: {len(records)} records in metadata")

        samples = []
        missing = 0

        for rec in tqdm(records, desc=f"  Building {split}"):
            audio_path = rec["audio_path"]   # e.g. "audio/train/XyZ_0010000.wav"
            full_path = data_dir / audio_path

            if not full_path.exists():
                missing += 1
                continue

            # Assign uid per unique audio clip
            if audio_path not in path_to_uid:
                uid = uid_counter[0]
                uid_counter[0] += 1
                path_to_uid[audio_path] = uid
                uid_to_meta[uid] = {
                    "context_uid":  uid,
                    "context":      audio_path,
                    "youtube_id":   rec["youtube_id"],
                    "start_time":   rec["start_time"],
                }
            uid = path_to_uid[audio_path]

            samples.append({
                "context":         audio_path,
                "contexts_list":   [audio_path],
                "titles_list":     [rec["caption"]],
                "useful_contexts": [1],
                "question":        rec["caption"],
                "answer":          rec["youtube_id"],
                "sample_idx":      len(samples),
                "dataset":         "audiocaps",
                "context_uid":     uid,
                # AudioCaps-specific
                "audiocap_id":     str(rec["audiocap_id"]),
                "youtube_id":      rec["youtube_id"],
                "start_time":      float(rec["start_time"]),
            })

        if missing:
            print(f"    ⚠ {missing} clips missing on disk, skipped")
        unique_clips = len(set(s["context_uid"] for s in samples))
        print(f"    ✓ {len(samples)} samples from {unique_clips} unique clips")
        all_splits[split] = samples

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
                ok = (data_dir / ex["context"]).exists()
                print(f"    context:  {ex['context']}  ({'✓' if ok else '✗ MISSING'})")
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
            avg_caps = len(s) / max(n_clips, 1)
            print(f"\n  {split}: {len(s):,} samples  |  {n_clips:,} clips  |  {avg_caps:.1f} captions/clip")
        print(f"\n  Total unique clips: {len(ds['cuid2text']):,}")
    except Exception as e:
        print(f"Could not load stats: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert AudioCaps to ReTreever format")
    parser.add_argument("--data_dir", default="./data/audiocaps")
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

    ds = convert_audiocaps_to_retreever(data_dir)
    if ds is None:
        print("Conversion failed.")
        return

    out = data_dir / "dataset"
    print(f"\nSaving to {out} ...")
    ds.save_to_disk(str(out))
    print("✓ Saved")

    verify_dataset(data_dir)
    show_statistics(data_dir)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\n  Load with:")
    print(f"    from datasets import load_from_disk")
    print(f"    ds = load_from_disk('{out}')")
    print("=" * 80)


if __name__ == "__main__":
    main()
