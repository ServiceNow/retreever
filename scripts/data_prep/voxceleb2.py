"""Prepare VoxCeleb2 for ReTreever training.

The ReTreever VoxCeleb dataset loader
(``retreever/data/voxceleb_dataset.py``) expects::

    {out_dir}/train/{speaker_id}/*.m4a
    {out_dir}/val/{speaker_id}/*.m4a

VoxCeleb2 is *not* freely distributed on Hugging Face. You must register at
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html and obtain:

  - vox2_dev_aac.zip   (development partition, train speakers)
  - vox2_test_aac.zip  (test partition, held-out speakers)

After extracting, the original layout is::

    aac/{speaker_id}/{video_id}/00001.m4a

This script flattens it to the layout the ReTreever loader expects.

USAGE
-----
::

    # 1. download the official zips, then extract them somewhere, e.g.:
    unzip vox2_dev_aac.zip -d /raw/voxceleb2/dev/
    unzip vox2_test_aac.zip -d /raw/voxceleb2/test/

    # 2. convert to retreever layout (creates symlinks by default; pass --copy
    #    to make actual file copies)
    python -m scripts.data_prep.voxceleb2 \
        --train-src /raw/voxceleb2/dev/aac \
        --val-src   /raw/voxceleb2/test/aac \
        --out-dir   /path/to/voxceleb2
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm


def convert_split(src: Path, dst: Path, copy: bool, ext: str) -> tuple[int, int]:
    """Flatten {src}/{speaker}/{video}/*.{ext} -> {dst}/{speaker}/*.{ext}."""
    dst.mkdir(parents=True, exist_ok=True)
    n_speakers = 0
    n_clips = 0
    speaker_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    for speaker_dir in tqdm(speaker_dirs, desc=f"speakers in {src.name}"):
        target_speaker_dir = dst / speaker_dir.name
        target_speaker_dir.mkdir(parents=True, exist_ok=True)
        n_speakers += 1
        for clip_path in speaker_dir.rglob(f"*.{ext}"):
            # Flatten "{video_id}/{clip}.m4a" -> "{video_id}__{clip}.m4a" so we
            # keep uniqueness across videos for the same speaker.
            rel = clip_path.relative_to(speaker_dir)
            flat_name = "__".join(rel.parts)
            target = target_speaker_dir / flat_name
            if target.exists():
                continue
            if copy:
                shutil.copy2(clip_path, target)
            else:
                os.symlink(clip_path.resolve(), target)
            n_clips += 1
    return n_speakers, n_clips


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train-src", required=True, help="Path to the extracted dev partition 'aac' directory.")
    parser.add_argument("--val-src", required=True, help="Path to the extracted test partition 'aac' directory.")
    parser.add_argument("--out-dir", required=True, help="Output root directory.")
    parser.add_argument(
        "--ext",
        default="m4a",
        help="Audio file extension. Default 'm4a' matches the VoxCeleb2 release.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks. Symlinks are much faster.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for split, src in [("train", args.train_src), ("val", args.val_src)]:
        src_path = Path(src)
        if not src_path.is_dir():
            raise SystemExit(f"Source for {split!r} does not exist: {src_path}")
        n_speakers, n_clips = convert_split(src_path, out_dir / split, args.copy, args.ext)
        print(f"[{split}] {n_speakers} speakers, {n_clips} clips -> {out_dir / split}")

    print("\nDone. Point your config at:")
    print(f"  dataset: voxceleb2")
    print(f"  KNOWN_DATASETS['voxceleb2'] = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
