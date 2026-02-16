from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torchaudio


class VoxCeleb2RetrievalDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 split='train', 
                 num_contexts=100, 
                 seed=42, 
                 subset=None,
                 for_eval=False,
                 sample_rate=16000,
                 max_duration=10.0,
                 audio_ext='wav'):
        """
        VoxCeleb2 dataset for speaker recognition/retrieval.
        
        Args:
            data_dir: Root directory of prepared VoxCeleb2 dataset
            split: 'train' or 'val'
            num_contexts: Number of context samples per query
            seed: Random seed
            subset: If specified, only use this many speakers (for quick testing)
            for_eval: If True, return only one context per query (itself)
            sample_rate: Target sample rate for audio
            max_duration: Maximum duration in seconds (clips will be truncated/padded)
        """
        self.data_dir = Path(data_dir) / split
        self.num_contexts = num_contexts
        self.for_eval = for_eval
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        self.audio_ext = audio_ext 
        
        # Collect all audio files and their speaker labels
        print("Loading audio file paths...")
        self.audio_files = []
        self.labels = []
        self.speaker_to_label = {}
        
        speaker_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # If subset requested, sample speakers
        if subset is not None:
            np.random.seed(seed)
            speaker_dirs = np.random.choice(speaker_dirs, size=min(subset, len(speaker_dirs)), replace=False)
            speaker_dirs = sorted(speaker_dirs)
        
        for label_idx, speaker_dir in enumerate(tqdm(speaker_dirs, desc="Scanning speakers")):
            speaker_id = speaker_dir.name
            self.speaker_to_label[speaker_id] = label_idx
            
            audio_files = list(speaker_dir.glob(f'*.{self.audio_ext}'))
            for audio_file in audio_files:
                self.audio_files.append(str(audio_file))
                self.labels.append(label_idx)
        
        print(f"Loaded {len(self.audio_files)} audio files from {len(speaker_dirs)} speakers")
        
        # Group by speaker (class)
        print("Grouping audio files by speaker...")
        self.speaker_to_indices = {}
        for idx, label in enumerate(tqdm(self.labels, desc="Building speaker index")):
            if label not in self.speaker_to_indices:
                self.speaker_to_indices[label] = []
            self.speaker_to_indices[label].append(idx)
        
        # Pre-sample contexts for each query
        print(f"Pre-sampling {num_contexts} contexts per query...")
        np.random.seed(seed)
        self.query_contexts = []
        
        self.valid_indices = list(range(len(self.audio_files)))
        
        for idx in tqdm(self.valid_indices, desc="Sampling contexts"):
            if self.for_eval:
                context_indices = [idx]
            else:
                label = self.labels[idx]
                class_indices = [i for i in self.speaker_to_indices[label] if i != idx]
                
                if len(class_indices) >= self.num_contexts:
                    context_indices = np.random.choice(
                        class_indices, 
                        size=self.num_contexts, 
                        replace=False
                    ).tolist()
                else:
                    context_indices = class_indices.copy()
            
            self.query_contexts.append(context_indices)
        
        # Build cumulative sum for fast indexing
        self.num_contexts_per_query = [len(contexts) for contexts in self.query_contexts]
        self.cumsum = np.cumsum([0] + self.num_contexts_per_query)
        
        print(f"Dataset ready: {len(self.audio_files)} queries -> {self.cumsum[-1]} total pairs")
        print(f"Sample rate: {self.sample_rate} Hz, Max duration: {self.max_duration}s")
    
    def _load_and_process_audio(self, audio_path):
        """
        Load audio file and process to consistent format.
        
        Returns:
            waveform: Tensor of shape (1, max_samples) or (2, max_samples) for stereo
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Truncate or pad to max_samples
        if waveform.shape[1] > self.max_samples:
            # Truncate (take center segment)
            start = (waveform.shape[1] - self.max_samples) // 2
            waveform = waveform[:, start:start + self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            # Pad with zeros
            pad_amount = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        
        return waveform
    
    def __len__(self):
        return self.cumsum[-1]
    
    def __getitem__(self, idx):
        # Map flat index to (query_idx, context_offset)
        query_pos = np.searchsorted(self.cumsum[1:], idx, side='right')
        context_offset = idx - self.cumsum[query_pos]
        
        # Get actual query and context index
        query_idx = self.valid_indices[query_pos]
        context_idx = self.query_contexts[query_pos][context_offset]
        
        # Load audio files
        query_audio = self._load_and_process_audio(self.audio_files[query_idx])
        context_audio = self._load_and_process_audio(self.audio_files[context_idx])
        label = self.labels[query_idx]
        
        return {
            'query_idx': query_idx,
            'query_audio': query_audio,
            'context_idx': context_idx,
            'context_audio': context_audio,
            'label': label,
            'context_uid': context_idx,
        }


if __name__ == "__main__":
    import torch
    
    dataset = VoxCeleb2RetrievalDataset(
        data_dir='/mnt/dssk/data_rw/esc50', 
        split='train',
        num_contexts=100,
        # subset=10,  # Only use 10 speakers for testing
        for_eval=False,
        sample_rate=16000,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of speakers: {len(dataset.speaker_to_indices)}")
    
    # Test a few samples
    for i in [0, 10, 100]:
        if i < len(dataset):
            sample = dataset[i]
            print(f"\nPair {i}:")
            print(f"  Query idx: {sample['query_idx']}, Context idx: {sample['context_idx']}")
            print(f"  Label: {sample['label']}")
            print(f"  Query audio shape: {sample['query_audio'].shape}")
            print(f"  Context audio shape: {sample['context_audio'].shape}")
            print(f"  Audio dtype: {sample['query_audio'].dtype}")