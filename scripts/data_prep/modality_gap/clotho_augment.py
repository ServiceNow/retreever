#!/usr/bin/env python3
"""Generate augmented versions of audio clips for text-audio retrieval datasets.

Supports:
    --dataset clotho      (default)  Clotho v2.1 — audio in split subdirs
    --dataset macs                   MACS — flat audio/ directory
    --dataset sounddescs             SoundDescs (BBC) — flat audio/ directory

For every WAV, produces 9 augmented copies (aug_index 1-9)
plus a symlink of the original as aug_index 0, writing all results into
audio_augmented/.  A JSON manifest records full provenance.

Augmentation index map
-----------------------
    0  original          symlink to original WAV
    1  time_stretch_slow rate = 0.85
    2  time_stretch_fast rate = 1.15
    3  pitch_shift_up    +2 semitones
    4  pitch_shift_down  -2 semitones
    5  noise_low         white noise, SNR = 30 dB
    6  noise_high        white noise, SNR = 15 dB
    7  reverb            synthetic exponential-decay RIR (rt60 = 0.4 s)
    8  lowpass           low-pass filter, cutoff = 3000 Hz
    9  highpass          high-pass filter, cutoff =  300 Hz

Output layout (Clotho)
-----------------------
    {data_dir}/audio_augmented/{development,validation,evaluation}/

Output layout (MACS)
---------------------
    {data_dir}/audio_augmented/   (flat)

Requires: audiomentations, soundfile, scipy  (numpy already present)

Usage
------
    # Clotho (default)
    python clotho_augment.py --data_dir ./data/clotho
    python clotho_augment.py --data_dir ./data/clotho --splits development

    # MACS
    python clotho_augment.py --dataset macs --data_dir ./data/macs

    # SoundDescs (4 augmentations = 5 versions per clip)
    python clotho_augment.py --dataset sounddescs --aug_indices 0 1 3 5 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import soundfile as sf
from audiomentations import (
    AddGaussianSNR,
    HighPassFilter,
    LowPassFilter,
    PitchShift,
    TimeStretch,
)

CLOTHO_SPLITS = ["development", "validation", "evaluation"]

# Per-dataset configuration: native sample rate and directory layout
DATASET_CONFIGS = {
    "clotho": {
        "default_data_dir": "./data/clotho",
        "source_sample_rate": 44100,
        "has_split_subdirs": True,
        "splits": ["development", "validation", "evaluation"],
    },
    "macs": {
        "default_data_dir": "./data/macs",
        "source_sample_rate": 48000,
        "has_split_subdirs": False,
        "splits": ["all"],  # flat directory, single pseudo-split
    },
    "sounddescs": {
        "default_data_dir": "./data/sounddescs",
        "source_sample_rate": None,  # variable rates — read from file
        "has_split_subdirs": False,
        "splits": ["all"],  # flat audio/ directory
    },
}


# ---------------------------------------------------------------------------
# Augmentation definitions
# (aug_index, aug_type, aug_params_dict, transform_factory | None)
# transform_factory is None for aug_index=0 (symlink) and aug_index=7 (reverb)
# ---------------------------------------------------------------------------
AUG_DEFS = [
    (0, "original",          {},                         None),
    (1, "time_stretch_slow", {"rate": 0.85},             lambda: TimeStretch(min_rate=0.85, max_rate=0.85, p=1.0)),
    (2, "time_stretch_fast", {"rate": 1.15},             lambda: TimeStretch(min_rate=1.15, max_rate=1.15, p=1.0)),
    (3, "pitch_shift_up",    {"semitones": 2},           lambda: PitchShift(min_semitones=2, max_semitones=2, p=1.0)),
    (4, "pitch_shift_down",  {"semitones": -2},          lambda: PitchShift(min_semitones=-2, max_semitones=-2, p=1.0)),
    (5, "noise_low",         {"snr_db": 30},             lambda: AddGaussianSNR(min_snr_db=30, max_snr_db=30, p=1.0)),
    (6, "noise_high",        {"snr_db": 15},             lambda: AddGaussianSNR(min_snr_db=15, max_snr_db=15, p=1.0)),
    (7, "reverb",            {"rt60": 0.4},              None),   # custom — see apply_reverb()
    (8, "lowpass",           {"cutoff_hz": 3000},        lambda: LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=3000, p=1.0)),
    (9, "highpass",          {"cutoff_hz": 300},         lambda: HighPassFilter(min_cutoff_freq=300, max_cutoff_freq=300, p=1.0)),
]


# ---------------------------------------------------------------------------
# Reverb helper
# ---------------------------------------------------------------------------

def apply_reverb(samples: np.ndarray, sample_rate: int, rt60: float = 0.4) -> np.ndarray:
    """Apply a synthetic reverb via an exponential-decay impulse response.

    rt60 : reverberation time in seconds (time to decay 60 dB).
    The result is a 50 % dry / 50 % wet mix, peak-normalised to avoid clipping.
    """
    from scipy.signal import fftconvolve

    n_rir = int(rt60 * sample_rate)
    t     = np.arange(n_rir) / sample_rate
    # ln(10^-3) / rt60  →  -60 dB decay over rt60 seconds
    decay_rate = np.log(1e-3) / rt60
    rir = np.exp(decay_rate * t).astype(np.float32)

    # Brief pre-delay (~5 ms) to separate direct sound from reflections
    predelay = int(0.005 * sample_rate)
    rir[:predelay] *= 0.1

    # Add a handful of early reflections
    rng = np.random.default_rng(42)
    for _ in range(8):
        delay = int(rng.uniform(0.01, 0.08) * sample_rate)
        amp   = float(rng.uniform(0.1, 0.5))
        if delay < n_rir:
            rir[delay] += amp * float(np.exp(decay_rate * delay / sample_rate))

    # Normalise RIR
    peak_rir = np.max(np.abs(rir))
    if peak_rir > 1e-8:
        rir /= peak_rir

    reverbed = fftconvolve(samples, rir)[: len(samples)]

    # 50/50 dry–wet mix
    out  = 0.5 * samples + 0.5 * reverbed.astype(np.float32)
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    src_path: Path,
    dst_dir:  Path,
    stem:     str,
    split:    str,
    overwrite: bool,
    aug_indices: set[int] | None = None,
) -> list[dict]:
    """Produce output files for *src_path* and return manifest rows.

    aug_indices: if provided, only produce these aug_index values.
                 None means all (0-9).
    """
    # Load once — float32 mono
    samples, sr = sf.read(str(src_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    manifest_rows: list[dict] = []

    for aug_index, aug_type, aug_params, transform_factory in AUG_DEFS:
        if aug_indices is not None and aug_index not in aug_indices:
            continue
        out_fname = f"{stem}__aug{aug_index}.wav"
        out_path  = dst_dir / out_fname

        manifest_rows.append({
            "aug_filename":    out_fname,
            "source_filename": src_path.name,
            "split":           split,
            "aug_index":       aug_index,
            "aug_type":        aug_type,
            "aug_params":      aug_params,
        })

        if out_path.exists() and not overwrite:
            continue

        # ---- aug_index 0: symlink to original ----
        if aug_index == 0:
            if out_path.exists() or out_path.is_symlink():
                out_path.unlink()
            out_path.symlink_to(src_path.resolve())
            continue

        # ---- aug_index 7: custom reverb ----
        if aug_type == "reverb":
            aug_samples = apply_reverb(samples, sr)
        else:
            transform   = transform_factory()
            aug_samples = transform(samples=samples, sample_rate=sr)

        sf.write(str(out_path), aug_samples, sr, subtype="PCM_16")

    return manifest_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented versions of audio clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="clotho",
        choices=list(DATASET_CONFIGS.keys()),
        help="Which dataset to augment.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Root data directory (must contain an audio/ subfolder). "
             "Defaults to dataset-specific path if not provided.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Which splits to augment (Clotho only; ignored for flat-dir datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate and overwrite existing augmented files.",
    )
    parser.add_argument(
        "--aug_indices",
        nargs="+",
        type=int,
        default=None,
        help="Which aug indices to generate (default: all 0-9). "
             "E.g. --aug_indices 0 1 3 5 7 for 5 versions "
             "(original + time_stretch_slow + pitch_shift_up + noise_low + reverb).",
    )
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]
    data_dir      = Path(args.data_dir) if args.data_dir else Path(ds_cfg["default_data_dir"])
    audio_dir     = data_dir / "audio"
    aug_base_dir  = data_dir / "audio_augmented"
    manifest_path = data_dir / "augmentation_manifest.json"

    if not audio_dir.exists():
        print(f"ERROR: audio directory not found: {audio_dir}")
        sys.exit(1)

    # Resolve splits
    if ds_cfg["has_split_subdirs"]:
        splits = args.splits or ds_cfg["splits"]
    else:
        splits = ["all"]  # flat directory

    print("=" * 80)
    print(f"AUDIO AUGMENTATION  ({args.dataset.upper()})")
    print("=" * 80)
    print(f"  dataset    : {args.dataset}")
    print(f"  data_dir   : {data_dir}")
    print(f"  layout     : {'split subdirs' if ds_cfg['has_split_subdirs'] else 'flat'}")
    print(f"  overwrite  : {args.overwrite}")
    print(f"  versions   : 10 per clip  (1 original symlink + 9 augmented)")
    print("=" * 80)

    all_manifest: list[dict] = []
    t_start = time.time()

    for split in splits:
        if ds_cfg["has_split_subdirs"]:
            split_src_dir = audio_dir    / split
            split_dst_dir = aug_base_dir / split
        else:
            # Flat layout (e.g. MACS): audio/*.wav -> audio_augmented/*.wav
            split_src_dir = audio_dir
            split_dst_dir = aug_base_dir

        if not split_src_dir.exists():
            print(f"\n  WARNING: {split_src_dir} not found — skipping.")
            continue

        split_dst_dir.mkdir(parents=True, exist_ok=True)
        wav_files = sorted(split_src_dir.glob("*.wav"))
        print(f"\n  {split}: {len(wav_files)} clips")

        for i, src_path in tqdm(enumerate(wav_files)):
            rows = process_file(
                src_path=src_path,
                dst_dir=split_dst_dir,
                stem=src_path.stem,
                split=split,
                overwrite=args.overwrite,
            )
            all_manifest.extend(rows)

            if (i + 1) % 200 == 0 or (i + 1) == len(wav_files):
                elapsed = time.time() - t_start
                print(f"    [{i+1}/{len(wav_files)}] {elapsed:.0f}s elapsed")

        n_clips = len(wav_files)
        print(f"  {split}: {n_clips} clips -> {n_clips * 10} files in {split_dst_dir}")

    # Write consolidated manifest
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nManifest written: {manifest_path}  ({len(all_manifest):,} entries)")
    print(f"  Total time : {elapsed:.1f}s")
    print(f"\nNext step:")
    print(f"  python prepare_clotho_augmented.py --dataset {args.dataset} --data_dir {data_dir}")


if __name__ == "__main__":
    main()
