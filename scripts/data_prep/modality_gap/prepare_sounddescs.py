#!/usr/bin/env python3
"""
Convert downloaded SoundDescs dataset to ReTreever format for text-audio retrieval.

Run sounddescs_download.py first.

SoundDescs structure after download:
    ./data/sounddescs/
    ├── audio/
    │   └── <audio_id>.wav
    └── metadata/
        └── metadata.jsonl   {audio_id, audio_path, text, raw_text, audio_len, split}

SoundDescs has train/val/test splits from HuggingFace. We preserve those.
One caption per clip (the processed text description).

  question    = text description
  context     = relative path to .wav
  context_uid = unique int per clip

Output: HuggingFace DatasetDict {train, val, test, cuid2text}
saved at <data_dir>/dataset/

Usage:
    python prepare_sounddescs.py --data_dir ./data/sounddescs
    python prepare_sounddescs.py --data_dir ./data/sounddescs --verify_only
    python prepare_sounddescs.py --data_dir ./data/sounddescs --stats
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict


# ============================================================================
# HELPERS
# ============================================================================

def load_metadata(meta_path: Path) -> list[dict]:
    records = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================================
# CONVERSION
# ============================================================================

def convert_sounddescs_to_retreever(data_dir: Path) -> DatasetDict | None:
    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio"
    meta_path = data_dir / "metadata" / "metadata.jsonl"

    print("=" * 80)
    print("SOUNDDESCS -> RETREEVER FORMAT")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("=" * 80)

    if not meta_path.exists():
        print(f"  Metadata not found: {meta_path}")
        return None

    entries = load_metadata(meta_path)
    print(f"\nMetadata entries: {len(entries)}")

    # Filter to entries with audio on disk and a text caption
    present = []
    for e in entries:
        wav = data_dir / e["audio_path"]
        if wav.exists() and wav.stat().st_size > 0 and e.get("text", "").strip():
            present.append(e)

    missing = len(entries) - len(present)
    if missing:
        print(f"  {missing} entries without audio or caption, skipped")
    print(f"  {len(present)} entries available")

    if not present:
        print("ERROR: No valid entries found.")
        return None

    # Group by HF split (train/validation/test)
    split_map = {"train": "train", "validation": "val", "test": "test"}
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for e in present:
        hf_split = e.get("split", "train")
        target = split_map.get(hf_split, "train")
        by_split[target].append(e)

    uid_to_meta: dict[int, dict] = {}
    all_splits: dict[str, list[dict]] = {}
    uid_counter = 0

    for split_name in ["train", "val", "test"]:
        samples = []
        for e in by_split[split_name]:
            rel_path = e["audio_path"]
            uid = uid_counter
            uid_counter += 1

            uid_to_meta[uid] = {
                "context_uid": uid,
                "context":     rel_path,
                "audio_id":    e["audio_id"],
            }

            samples.append({
                "context":         rel_path,
                "contexts_list":   [rel_path],
                "titles_list":     [e["text"]],
                "useful_contexts": [1],
                "question":        e["text"],
                "answer":          e["audio_id"],
                "sample_idx":      len(samples),
                "dataset":         "sounddescs",
                "context_uid":     uid,
                # SoundDescs-specific
                "audio_id":        e["audio_id"],
                "audio_len":       e.get("audio_len", 0.0),
            })

        print(f"  {split_name}: {len(samples)} clips")
        all_splits[split_name] = samples

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
                print(f"    file exists: {'yes' if ok else 'MISSING'}")
        print(f"\n  cuid2text: {len(ds['cuid2text'])} unique clips")
        print("\nVerification complete!")
    except Exception as e:
        print(f"Failed: {e}")
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
            avg_len = sum(len(q.split()) for q in s["question"]) / max(len(s), 1)
            print(f"\n  {split}: {len(s):,} clips  |  avg caption length: {avg_len:.0f} words")
        print(f"\n  Total clips: {len(ds['cuid2text']):,}")
    except Exception as e:
        print(f"Could not load stats: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert SoundDescs to ReTreever format")
    parser.add_argument("--data_dir", default="./data/sounddescs")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--stats",       action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.verify_only:
        verify_dataset(data_dir)
        return
    if args.stats:
        show_statistics(data_dir)
        return

    ds = convert_sounddescs_to_retreever(data_dir)
    if ds is None:
        print("Conversion failed.")
        return

    out = data_dir / "dataset"
    print(f"\nSaving to {out} ...")
    ds.save_to_disk(str(out))
    print("Saved")

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
