#!/usr/bin/env python3
"""Cache the CLAP mel features of every audio clip of a prepared text-audio dataset.

Training and evaluation read these tensors instead of decoding waveforms, which is several times faster. For every
unique ``context_uid`` of the HF DatasetDict at ``<data_dir>/<dataset_subdir>`` the script writes
``<output_dir>/<context_uid>.pt`` holding ``{input_features: (C, F, T), is_longer: bool}``, exactly what
``ClapFeatureExtractor`` produces for a 48 kHz, mono, 30-second (centre-cropped or zero-padded) clip.

    python preprocess_clap_features.py --data_dir ./data/clotho                              # -> ./data/clotho_clap-fused-ta
    python preprocess_clap_features.py --data_dir ./data/clotho --dataset_subdir dataset_augmented --name clotho_augmented
    python preprocess_clap_features.py --data_dir ./data/sounddescs
    python preprocess_clap_features.py --data_dir ./data/audiocaps --source_sample_rate 16000

The default output directory, ``<parent of data_dir>/<name>_<encoder_type>`` with ``name`` = the data directory's
name, is where scripts/train.py and experiments/modality_gap/evaluate.py look for the features.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio

CLAP_MODEL_NAMES = {"clap-fused-ta": "laion/clap-htsat-fused", "clap-unfused-ta": "laion/clap-htsat-unfused"}
TARGET_SAMPLE_RATE = 48000  # what CLAP expects


def load_audio(path: Path, resampler, max_samples: int) -> torch.Tensor:
    """Load, resample, mix down to mono and centre-crop / zero-pad to ``max_samples``; (1, max_samples)."""
    try:
        waveform, _ = torchaudio.load(str(path))
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return torch.zeros(1, max_samples)
    if resampler is not None:
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    n = waveform.shape[1]
    if n > max_samples:
        start = (n - max_samples) // 2
        waveform = waveform[:, start:start + max_samples]
    elif n < max_samples:
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - n))
    return waveform


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="dataset root (holds the HF DatasetDict and the audio files)")
    p.add_argument("--dataset_subdir", default="dataset", help="HF DatasetDict sub-directory; 'dataset_augmented' for clotho_augmented")
    p.add_argument("--name", default=None, help="dataset name for the output directory (default: the data_dir name)")
    p.add_argument("--encoder_type", default="clap-fused-ta", choices=list(CLAP_MODEL_NAMES))
    p.add_argument("--source_sample_rate", type=int, default=44100, help="native rate of the files (Clotho/SoundDescs 44100, AudioCaps 16000)")
    p.add_argument("--output_dir", default=None, help="default: <parent of data_dir>/<name>_<encoder_type>")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max_duration", type=float, default=30.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    name = args.name or data_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / f"{name}_{args.encoder_type}"
    dataset_dir = data_dir / args.dataset_subdir
    if not dataset_dir.exists():
        sys.exit(f"ERROR: no HF dataset at {dataset_dir}; run the prepare_*.py script first")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"features: {CLAP_MODEL_NAMES[args.encoder_type]}  {dataset_dir} -> {output_dir}  splits={args.splits}")

    import datasets as hf_datasets
    from transformers import ClapFeatureExtractor

    dataset_dict = hf_datasets.load_from_disk(str(dataset_dir))
    uid_to_relpath = {}
    for split in args.splits:
        if split not in dataset_dict:
            print(f"  WARNING: split '{split}' not in the dataset, skipping")
            continue
        for row in dataset_dict[split]:
            uid_to_relpath.setdefault(row["context_uid"], row["context"])
    print(f"{len(uid_to_relpath)} unique clips")

    extractor = ClapFeatureExtractor.from_pretrained(CLAP_MODEL_NAMES[args.encoder_type], cache_dir=args.hf_cache_dir)
    resampler = None if args.source_sample_rate == TARGET_SAMPLE_RATE else torchaudio.transforms.Resample(args.source_sample_rate, TARGET_SAMPLE_RATE)
    max_samples = int(TARGET_SAMPLE_RATE * args.max_duration)

    n_done = n_skip = n_error = 0
    t0 = time.time()
    batch_uids, batch_waves = [], []

    def flush():
        nonlocal n_done, n_error
        if not batch_uids:
            return
        try:
            feats = extractor([w.squeeze(0).float().numpy() for w in batch_waves], sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            for i, uid in enumerate(batch_uids):
                torch.save({"input_features": feats["input_features"][i].clone(), "is_longer": feats["is_longer"][i].bool().clone()},
                           output_dir / f"{uid}.pt")
                n_done += 1
        except Exception as e:
            print(f"  ERROR on batch starting at uid {batch_uids[0]}: {e}")
            n_error += len(batch_uids)
        batch_uids.clear()
        batch_waves.clear()

    for i, uid in enumerate(sorted(uid_to_relpath)):
        if (output_dir / f"{uid}.pt").exists() and not args.overwrite:
            n_skip += 1
            continue
        batch_uids.append(uid)
        batch_waves.append(load_audio(data_dir / uid_to_relpath[uid], resampler, max_samples))
        if len(batch_uids) >= args.batch_size:
            flush()
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(uid_to_relpath)}] {n_done} done, {n_skip} skipped, {n_error} errors, {time.time() - t0:.0f}s")
    flush()
    print(f"done: {n_done} written, {n_skip} skipped, {n_error} errors, {time.time() - t0:.0f}s -> {output_dir}")


if __name__ == "__main__":
    main()
