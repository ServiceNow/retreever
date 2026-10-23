#!/usr/bin/env python3
"""Prepare an augmented audio dataset in ReTreever format.

Supports:
    --dataset clotho  (default)  Clotho v2.1 — captions from CSVs, split subdirs
    --dataset macs               MACS — captions from YAML, flat dir, random split

Reads the augmentation_manifest.json (produced by clotho_augment.py) and builds
a HuggingFace DatasetDict where every original clip is expanded to 10 versions
(original + 9 augmentations).

Each sample carries three extra fields compared with the original dataset:

    source_uid  — context_uid of the *original* (aug_index=0) version of the
                  clip; shared by all 10 augmented variants of the same recording.
    aug_type    — string label, e.g. "reverb", "original"
    aug_index   — int 0–9

Identifier contract
--------------------
    aug_index == 0  →  context_uid == source_uid   (original)
    aug_index  > 0  →  context_uid  ≠ source_uid   (augmented copy)

This lets downstream code group-by source_uid to find all 10 audio versions
that share the same ground-truth captions, enabling audio-audio and
audio-text retrieval evaluation.

Output
------
    {data_dir}/dataset_augmented/   — HF DatasetDict {train, val, test, cuid2text}

Usage
------
    # Clotho (default)
    python prepare_clotho_augmented.py --data_dir ./data/clotho

    # MACS
    python prepare_clotho_augmented.py --dataset macs --data_dir ./data/macs
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPLIT_MAP = {
    "development": "train",
    "validation":  "val",
    "evaluation":  "test",
}

CAPTION_COLS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_clotho_csv(csv_path: Path) -> list[dict]:
    """Parse a Clotho captions CSV (file_name + 5 caption columns)."""
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_manifest(manifest_path: Path) -> dict[str, list[dict]]:
    """Load augmentation_manifest.json.

    Returns:
        dict mapping source_filename → list of aug entries sorted by aug_index.
    """
    with open(manifest_path) as f:
        entries: list[dict] = json.load(f)

    by_source: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_source[entry["source_filename"]].append(entry)

    for fname in by_source:
        by_source[fname].sort(key=lambda x: x["aug_index"])

    return dict(by_source)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_clotho_augmented(data_dir: Path) -> DatasetDict | None:
    data_dir      = Path(data_dir)
    raw_dir       = data_dir / "raw"
    audio_aug_dir = data_dir / "audio_augmented"
    manifest_path = data_dir / "augmentation_manifest.json"

    print("=" * 80)
    print("CLOTHO v2.1 AUGMENTED -> RETREEVER FORMAT")
    print("=" * 80)
    print(f"Data directory : {data_dir}")
    print("=" * 80)

    if not manifest_path.exists():
        print(f"ERROR: augmentation_manifest.json not found: {manifest_path}")
        print("Run clotho_augment.py first.")
        return None

    manifest = load_manifest(manifest_path)
    n_sources  = len(manifest)
    n_entries  = sum(len(v) for v in manifest.values())
    print(f"Manifest loaded: {n_entries:,} entries across {n_sources:,} source files")

    uid_counter: list[int] = [0]
    uid_to_meta: dict[int, dict] = {}
    all_splits:  dict[str, list[dict]] = {}

    for clotho_split, output_split in SPLIT_MAP.items():
        csv_path      = raw_dir       / f"clotho_captions_{clotho_split}.csv"
        split_aug_dir = audio_aug_dir / clotho_split

        if not csv_path.exists():
            print(f"  ✗ CSV not found: {csv_path}")
            return None
        if not split_aug_dir.exists():
            print(f"  ✗ Augmented audio dir not found: {split_aug_dir}")
            print("    Run clotho_augment.py first.")
            return None

        rows = load_clotho_csv(csv_path)
        print(f"\n  {clotho_split} -> {output_split}: {len(rows)} original clips in CSV")

        samples: list[dict] = []
        n_missing_manifest  = 0
        n_missing_disk      = 0

        for row in rows:
            fname    = row["file_name"]
            captions = [row[c].strip() for c in CAPTION_COLS if row.get(c, "").strip()]

            aug_entries = manifest.get(fname)
            if aug_entries is None:
                n_missing_manifest += 1
                continue

            # source_uid = context_uid assigned to the original (aug_index == 0) version.
            # We reserve it first so aug_index-0's context_uid == source_uid.
            source_uid = uid_counter[0]
            uid_counter[0] += 1

            for aug_entry in aug_entries:
                aug_index  = aug_entry["aug_index"]
                aug_type   = aug_entry["aug_type"]
                aug_fname  = aug_entry["aug_filename"]
                aug_params = aug_entry["aug_params"]

                aug_path = split_aug_dir / aug_fname
                if not aug_path.exists():
                    n_missing_disk += 1
                    continue

                # Original keeps context_uid == source_uid; augmented copies get new UIDs.
                if aug_index == 0:
                    context_uid = source_uid
                else:
                    context_uid = uid_counter[0]
                    uid_counter[0] += 1

                rel_path = f"audio_augmented/{clotho_split}/{aug_fname}"

                uid_to_meta[context_uid] = {
                    "context_uid": context_uid,
                    "source_uid":  source_uid,
                    "context":     rel_path,
                    "file_name":   fname,
                    "aug_fname":   aug_fname,
                    "aug_type":    aug_type,
                    "aug_index":   aug_index,
                }

                for cap_idx, caption in enumerate(captions):
                    samples.append({
                        # Core ReTreever fields
                        "context":         rel_path,
                        "contexts_list":   [rel_path],
                        "titles_list":     [caption],
                        "useful_contexts": [1],
                        "question":        caption,
                        "answer":          fname,
                        "sample_idx":      len(samples),
                        "dataset":         "clotho_augmented",
                        "context_uid":     context_uid,
                        # Augmentation provenance
                        "source_uid":      source_uid,
                        "aug_type":        aug_type,
                        "aug_index":       aug_index,
                        # Clotho-specific
                        "file_name":       fname,
                        "caption_idx":     cap_idx,
                    })

        if n_missing_manifest:
            print(f"    ⚠  {n_missing_manifest} clips missing from manifest — skipped")
        if n_missing_disk:
            print(f"    ⚠  {n_missing_disk} augmented audio files missing on disk — skipped")

        n_audio_versions = len({s["context_uid"] for s in samples})
        n_orig_clips     = len({s["source_uid"]  for s in samples})
        print(
            f"    ✓ {len(samples):,} samples  |  "
            f"{n_audio_versions:,} audio versions  |  "
            f"{n_orig_clips:,} original clips  |  "
            f"~{len(samples) / max(n_orig_clips, 1):.0f} samples/original-clip"
        )

        all_splits[output_split] = samples

    # cuid2text: one entry per unique audio version (context_uid), sorted
    cuid2text = sorted(uid_to_meta.values(), key=lambda x: x["context_uid"])

    dataset_dict = DatasetDict({
        "train":     Dataset.from_list(all_splits["train"]),
        "val":       Dataset.from_list(all_splits["val"]),
        "test":      Dataset.from_list(all_splits["test"]),
        "cuid2text": Dataset.from_list(cuid2text),
    })

    print(f"\n{dataset_dict}")
    return dataset_dict


# ---------------------------------------------------------------------------
# MACS augmented conversion
# ---------------------------------------------------------------------------

def load_macs_yaml(yaml_path: Path) -> list[dict]:
    """Parse MACS.yaml → list of {filename, annotations: [{sentence, ...}]}."""
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["files"]


def convert_macs_augmented(
    data_dir: Path,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict | None:
    data_dir      = Path(data_dir)
    audio_aug_dir = data_dir / "audio_augmented"
    manifest_path = data_dir / "augmentation_manifest.json"
    yaml_path     = data_dir / "raw" / "MACS.yaml"

    print("=" * 80)
    print("MACS AUGMENTED -> RETREEVER FORMAT")
    print("=" * 80)
    print(f"Data directory : {data_dir}")
    print(f"Split ratio    : {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")
    print("=" * 80)

    if not manifest_path.exists():
        print(f"ERROR: augmentation_manifest.json not found: {manifest_path}")
        print("Run clotho_augment.py --dataset macs first.")
        return None
    if not yaml_path.exists():
        print(f"ERROR: MACS.yaml not found: {yaml_path}")
        return None

    manifest = load_manifest(manifest_path)
    n_sources = len(manifest)
    n_entries = sum(len(v) for v in manifest.values())
    print(f"Manifest loaded: {n_entries:,} entries across {n_sources:,} source files")

    # Load MACS annotations
    macs_entries = load_macs_yaml(yaml_path)
    print(f"MACS annotations: {len(macs_entries)} files")

    # Filter to entries present in both manifest and augmented audio
    present = []
    for entry in macs_entries:
        fname = entry["filename"]
        if fname in manifest:
            # Check at least the original aug exists on disk
            aug0 = manifest[fname][0]
            if (audio_aug_dir / aug0["aug_filename"]).exists():
                present.append(entry)

    missing = len(macs_entries) - len(present)
    if missing:
        print(f"  {missing} clips missing from manifest/disk, skipped")
    print(f"  {len(present)} clips with augmented audio available")

    if not present:
        print("ERROR: No valid entries found.")
        return None

    # Random split (MACS has no official splits)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(present))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    split_indices = {
        "train": indices[:n_train],
        "val":   indices[n_train:n_train + n_val],
        "test":  indices[n_train + n_val:],
    }

    uid_counter: list[int] = [0]
    uid_to_meta: dict[int, dict] = {}
    all_splits: dict[str, list[dict]] = {}

    for output_split, idxs in split_indices.items():
        samples: list[dict] = []
        n_missing_disk = 0

        for i in idxs:
            entry = present[i]
            fname = entry["filename"]
            captions = [a["sentence"].strip() for a in entry["annotations"]
                        if a.get("sentence", "").strip()]

            aug_entries = manifest[fname]

            # Reserve source_uid for the original
            source_uid = uid_counter[0]
            uid_counter[0] += 1

            for aug_entry in aug_entries:
                aug_index = aug_entry["aug_index"]
                aug_type  = aug_entry["aug_type"]
                aug_fname = aug_entry["aug_filename"]

                aug_path = audio_aug_dir / aug_fname
                if not aug_path.exists():
                    n_missing_disk += 1
                    continue

                if aug_index == 0:
                    context_uid = source_uid
                else:
                    context_uid = uid_counter[0]
                    uid_counter[0] += 1

                # MACS augmented audio is flat (no split subdirs)
                rel_path = f"audio_augmented/{aug_fname}"

                uid_to_meta[context_uid] = {
                    "context_uid": context_uid,
                    "source_uid":  source_uid,
                    "context":     rel_path,
                    "file_name":   fname,
                    "aug_fname":   aug_fname,
                    "aug_type":    aug_type,
                    "aug_index":   aug_index,
                }

                for cap_idx, caption in enumerate(captions):
                    samples.append({
                        # Core ReTreever fields
                        "context":         rel_path,
                        "contexts_list":   [rel_path],
                        "titles_list":     [caption],
                        "useful_contexts": [1],
                        "question":        caption,
                        "answer":          fname,
                        "sample_idx":      len(samples),
                        "dataset":         "macs_augmented",
                        "context_uid":     context_uid,
                        # Augmentation provenance
                        "source_uid":      source_uid,
                        "aug_type":        aug_type,
                        "aug_index":       aug_index,
                        # MACS-specific
                        "file_name":       fname,
                        "caption_idx":     cap_idx,
                    })

        if n_missing_disk:
            print(f"    {n_missing_disk} augmented audio files missing on disk")

        n_audio_versions = len({s["context_uid"] for s in samples})
        n_orig_clips     = len({s["source_uid"]  for s in samples})
        print(
            f"  {output_split}: {len(samples):,} samples  |  "
            f"{n_audio_versions:,} audio versions  |  "
            f"{n_orig_clips:,} original clips"
        )
        all_splits[output_split] = samples

    cuid2text = sorted(uid_to_meta.values(), key=lambda x: x["context_uid"])

    dataset_dict = DatasetDict({
        "train":     Dataset.from_list(all_splits["train"]),
        "val":       Dataset.from_list(all_splits["val"]),
        "test":      Dataset.from_list(all_splits["test"]),
        "cuid2text": Dataset.from_list(cuid2text),
    })

    print(f"\n{dataset_dict}")
    return dataset_dict


# ---------------------------------------------------------------------------
# Verify / Stats
# ---------------------------------------------------------------------------

def verify_dataset(data_dir: Path) -> None:
    from datasets import load_from_disk
    data_dir = Path(data_dir)
    print("\n" + "=" * 80)
    print("VERIFYING AUGMENTED DATASET")
    print("=" * 80)
    try:
        ds = load_from_disk(str(data_dir / "dataset_augmented"))
        print(ds)
        for split in ["train", "val", "test"]:
            s = ds[split]
            print(f"\n  {split}: {len(s):,} samples")
            if len(s) > 0:
                ex = s[0]
                print(f"    question   : {ex['question'][:80]}")
                print(f"    context    : {ex['context']}")
                print(f"    aug_type   : {ex['aug_type']}")
                print(f"    context_uid: {ex['context_uid']}")
                print(f"    source_uid : {ex['source_uid']}")
                ok = (data_dir / ex["context"]).exists()
                print(f"    file exists: {'✓' if ok else '✗'}")
        print(f"\n  cuid2text: {len(ds['cuid2text']):,} unique audio versions")
        print("\n✓ Verification complete!")
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def show_statistics(data_dir: Path) -> None:
    from datasets import load_from_disk
    data_dir = Path(data_dir)
    print("\n" + "=" * 80)
    print("AUGMENTED DATASET STATISTICS")
    print("=" * 80)
    try:
        ds = load_from_disk(str(data_dir / "dataset_augmented"))
        for split in ["train", "val", "test"]:
            s = ds[split]
            n_audio    = len(set(s["context_uid"]))
            n_sources  = len(set(s["source_uid"]))
            aug_types  = sorted(set(s["aug_type"]))
            print(f"\n  {split}:")
            print(f"    {len(s):,} samples  |  {n_audio:,} audio versions  |  {n_sources:,} original clips")
            print(f"    ~{len(s) / max(n_audio, 1):.1f} captions/audio-version  |  "
                  f"~{n_audio / max(n_sources, 1):.1f} audio-versions/original-clip")
            print(f"    aug types: {aug_types}")
        print(f"\n  Total unique audio versions (cuid2text): {len(ds['cuid2text']):,}")
    except Exception as e:
        print(f"Could not load statistics: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert augmented audio dataset to ReTreever format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="clotho", choices=["clotho", "macs"],
                        help="Which dataset to prepare augmented version for.")
    parser.add_argument("--data_dir", default=None,
                        help="Root data directory. Defaults to dataset-specific path.")
    parser.add_argument("--train_ratio", type=float, default=0.90,
                        help="Train split ratio (MACS only; Clotho uses official splits).")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Val split ratio (MACS only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting (MACS only).")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--stats",       action="store_true")
    args = parser.parse_args()

    default_dirs = {"clotho": "./data/clotho", "macs": "./data/macs"}
    data_dir = Path(args.data_dir) if args.data_dir else Path(default_dirs[args.dataset])

    if args.verify_only:
        verify_dataset(data_dir)
        return
    if args.stats:
        show_statistics(data_dir)
        return

    if args.dataset == "clotho":
        ds = convert_clotho_augmented(data_dir)
    elif args.dataset == "macs":
        ds = convert_macs_augmented(
            data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if ds is None:
        print("Conversion failed.")
        return

    out = data_dir / "dataset_augmented"
    print(f"\nSaving dataset to {out} ...")
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
