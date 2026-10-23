from torch.utils.data import Dataset
import torch
import torchaudio
from pathlib import Path
from typing import Optional


class TextAudioRetrievalDataset(Dataset):
    """
    Dataset for text-audio retrieval (e.g., Clotho).

    Mirrors TextImageRetrievalDataset: queries are text captions, contexts are
    audio waveform tensors loaded from disk.

    The HuggingFace dataset is expected to have (at minimum):
        - 'context'     : relative path to audio file, e.g. "audio/development/foo.wav"
        - 'question'    : text caption (the query)
        - 'context_uid' : integer unique id for each distinct audio file
        - 'sample_idx'  : integer sample index
        - 'dataset'     : dataset name string

    Audio files in Clotho are recorded at 44100 Hz.  This class resamples them
    to ``target_sample_rate`` (default 48000 Hz, what CLAP requires) using a
    resampler built **once** in ``__init__`` so the transform is not
    re-instantiated on every clip.
    """

    def __init__(
        self,
        data,                               # HuggingFace dataset split
        audio_base_dir: str = None,         # Root directory that resolves relative paths
        subset_size: Optional[int] = None,
        source_sample_rate: int = 44100,    # Native sample rate of the source files (Clotho=44100)
        target_sample_rate: int = 48000,    # Rate the encoder expects (CLAP=48000)
        max_duration: float = 30.0,         # Clips longer than this are centre-cropped
        preprocessed_dir: Optional[str] = None,  # If set, load pre-computed .pt files from
                                                  # this directory instead of raw waveforms.
                                                  # Each file is named {context_uid}.pt and
                                                  # contains {input_features, is_longer}.
    ):
        """
        Args:
            data: HuggingFace dataset split containing text-audio pairs.
            audio_base_dir: Root directory used to resolve relative audio paths
                stored in the ``context`` column, e.g. "/data/clotho".
            subset_size: If given, truncate the split to this many examples.
            source_sample_rate: Native sample rate of the source audio files.
                Clotho is 44100 Hz.
            target_sample_rate: Sample rate the encoder requires.  Audio is
                resampled from ``source_sample_rate`` to this value using a
                resampler built once in ``__init__``.
            max_duration: Maximum clip length in seconds.  Longer clips are
                centre-cropped; shorter ones are zero-padded.
        """
        self.data = data
        self.audio_base_dir = Path(audio_base_dir) if audio_base_dir else None
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.max_samples = int(target_sample_rate * max_duration)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # Build resampler once — zero overhead per clip if rates already match
        if source_sample_rate != target_sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=source_sample_rate, new_freq=target_sample_rate
            )
        else:
            self.resampler = None

        if subset_size is not None:
            self.data = self.data.select(range(min(subset_size, len(self.data))))

        print(
            f"TextAudioRetrievalDataset: {len(self.data)} samples | "
            f"{source_sample_rate} Hz -> {target_sample_rate} Hz | max_duration={max_duration}s | "
            f"preprocessed_dir={preprocessed_dir or 'none (raw waveforms)'}"
        )

    def __len__(self):
        return len(self.data)

    def _load_audio(self, relative_path: str) -> torch.Tensor:
        """Load, resample (if needed), mono-ify and pad/crop an audio file.

        The resampler (if source != target rate) is pre-built in __init__
        so there is no per-call construction overhead.

        Returns:
            Tensor of shape (1, max_samples) at ``self.target_sample_rate``.
        """
        if self.audio_base_dir is not None:
            full_path = self.audio_base_dir / relative_path
        else:
            full_path = Path(relative_path)

        try:
            waveform, sr = torchaudio.load(str(full_path))
        except Exception as e:
            print(f"Error loading audio {full_path}: {e}")
            return torch.zeros(1, self.max_samples)

        # Resample using the pre-built resampler
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Centre-crop or zero-pad to max_samples (at target_sample_rate)
        n = waveform.shape[1]
        if n > self.max_samples:
            start = (n - self.max_samples) // 2
            waveform = waveform[:, start : start + self.max_samples]
        elif n < self.max_samples:
            pad = self.max_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform  # shape: (1, max_samples)

    def __getitem__(self, idx):
        example = self.data[idx]

        if self.preprocessed_dir is not None:
            # ---------------------------------------------------------------
            # Fast path: load pre-computed mel spectrogram from disk.
            # The preprocessing script (preprocess_clotho.py) runs once and
            # saves {context_uid}.pt containing {input_features, is_longer}.
            # ---------------------------------------------------------------
            uid = example["context_uid"]
            pt_path = self.preprocessed_dir / f"{uid}.pt"
            precomputed = torch.load(pt_path, map_location="cpu", weights_only=True)
            return {
                "question":       example["question"],
                "input_features": precomputed["input_features"],  # (C, F, T)
                "is_longer":      precomputed["is_longer"].bool(), # scalar bool
                "context_uid":    uid,
                "sample_idx":     example["sample_idx"],
                "dataset":        example.get("dataset", ""),
                "answer":         example.get("answer", ""),
                "file_name":      example.get("file_name", ""),
            }

        # Legacy path: return raw waveform.
        audio_path = example["context"]  # relative path, e.g. "audio/development/foo.wav"
        waveform = self._load_audio(audio_path)  # (1, max_samples)
        return {
            "question": example["question"],    # text caption → the query
            "context_audio": waveform,           # (1, max_samples) float tensor
            "context_uid": example["context_uid"],
            "sample_idx": example["sample_idx"],
            "dataset": example["dataset"],
            "answer": example.get("answer", ""),
            "file_name": example.get("file_name", ""),
        }


# ---------------------------------------------------------------------------
# Smoke test
