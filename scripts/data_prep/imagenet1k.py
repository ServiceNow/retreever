"""Prepare ImageNet-1K for ReTreever training.

The ReTreever ImageNet dataset loader (``retreever/data/imagenet_dataset.py``)
uses ``torchvision.datasets.ImageFolder``, which expects a directory layout of::

    {out_dir}/train/{wnid}/*.JPEG
    {out_dir}/val/{wnid}/*.JPEG

where ``{wnid}`` is the WordNet ID of the class (e.g. ``n01440764`` for tench).

This script downloads ImageNet-1K from Hugging Face (``ILSVRC/imagenet-1k``)
and writes it to disk in that layout.

NOTES
-----
- ``ILSVRC/imagenet-1k`` is *gated*: you must first accept the license at
  https://huggingface.co/datasets/ILSVRC/imagenet-1k and ``huggingface-cli login``.
- The full dataset is ~150 GB on disk after extraction. Use --max-per-class
  for a quick smoke test.

USAGE
-----
::

    python -m scripts.data_prep.imagenet1k --out-dir /path/to/imagenet1k
    # quick smoke test (100 images per class):
    python -m scripts.data_prep.imagenet1k --out-dir /tmp/imagenet1k_tiny --max-per-class 100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Output root directory")
    parser.add_argument("--hf-repo", default="ILSVRC/imagenet-1k", help="HF dataset repo id")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="HF splits to download. Will be renamed train->train, validation->val.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on number of images written per class (for smoke testing).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF datasets cache directory. Defaults to the standard HF cache.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map HF split names to the layout the retreever loader expects.
    split_rename = {"train": "train", "validation": "val", "val": "val", "test": "test"}

    for hf_split in args.splits:
        target_split = split_rename.get(hf_split, hf_split)
        split_dir = out_dir / target_split
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Streaming {args.hf_repo} split={hf_split} -> {split_dir} ===")
        ds = load_dataset(args.hf_repo, split=hf_split, cache_dir=args.cache_dir, streaming=True)

        # Use the class label feature names from the (non-streaming) info to get WNIDs.
        # For ImageNet-1K, names look like "tench, Tinca tinca". We use the integer
        # label as the folder name to stay independent of any label-string scheme.
        # If WNIDs are preferred, replace `cls_name` with the entry from a wnids.txt.
        counts: dict[int, int] = {}
        for example in tqdm(ds, desc=hf_split):
            label = example["label"]
            if args.max_per_class is not None and counts.get(label, 0) >= args.max_per_class:
                continue

            class_dir = split_dir / f"class_{label:04d}"
            class_dir.mkdir(parents=True, exist_ok=True)

            idx = counts.get(label, 0)
            counts[label] = idx + 1

            image = example["image"]
            # HF returns PIL.Image; coerce to RGB JPEG.
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(class_dir / f"{idx:07d}.JPEG", format="JPEG", quality=95)

        n_classes = len(counts)
        n_images = sum(counts.values())
        print(f"  -> wrote {n_images} images across {n_classes} classes to {split_dir}")

    print("\nDone. Point your config at:")
    print(f"  dataset: imagenet1k")
    print(f"  KNOWN_DATASETS['imagenet1k'] = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
