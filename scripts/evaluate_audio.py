#!/usr/bin/env python3
"""
Evaluation script for audio retrieval models (ReTreever).

Evaluates trained ReTreever checkpoints on audio retrieval datasets:
- VoxCeleb2 (audio → audio retrieval, same speaker = relevant)

Computes standard retrieval metrics:
- Hit@k (k=1, 5, 10)
- NDCG@k (k=10, 100)
- MAP@k (k=10, 100)

Usage:
    python scripts/evaluate_audio.py --model_ckpt checkpoints/voxceleb_model.pt \
                                      --model_cfg checkpoints/config.yaml \
                                      --dataset voxceleb2 \
                                      --data_dir /path/to/voxceleb2 \
                                      --output_dir ./eval_results \
                                      --device cuda

    # For custom batch sizes or number of queries
    python scripts/evaluate_audio.py --model_ckpt checkpoints/model.pt \
                                      --model_cfg config.yaml \
                                      --dataset voxceleb2 \
                                      --data_dir /path/to/voxceleb2 \
                                      --batch_size 64 \
                                      --max_queries 5000 \
                                      --device cuda
"""

import os
import sys
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import faiss
from tqdm import tqdm

from retreever import config
from retreever.models.retreever import load_from_ckpt

# Try to import audio dependencies
try:
    from torch.utils.data import DataLoader, Dataset
    import torchaudio
except ImportError:
    print("WARNING: torchaudio not found. Install with: pip install torchaudio")
    sys.exit(1)


def compute_metrics(predictions: List[List[int]], 
                   references: List[List[int]], 
                   k_values: List[int] = [1, 5, 10, 100]) -> Dict[str, float]:
    """
    Compute retrieval metrics: Hit@k, NDCG@k, MAP@k.
    
    Args:
        predictions: List of lists, where predictions[i] is the ranked list of retrieved context IDs for query i
        references: List of lists, where references[i] contains the relevant context IDs for query i
        k_values: List of k values to compute metrics for
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    for k in k_values:
        # Hit@k - any relevant item in top-k
        hits = 0
        for pred, ref in zip(predictions, references):
            top_k = pred[:k]
            if any(item in ref for item in top_k):
                hits += 1
        metrics[f'Hit@{k}'] = hits / len(predictions) if predictions else 0.0
        
        # NDCG@k
        ndcg_sum = 0.0
        for pred, ref in zip(predictions, references):
            ref_set = set(ref)
            dcg = sum(
                1 / math.log2(i + 2)
                for i, p in enumerate(pred[:k])
                if p in ref_set
            )
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(ref), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sum += ndcg
        metrics[f'NDCG@{k}'] = ndcg_sum / len(predictions) if predictions else 0.0
        
        # MAP@k
        ap_sum = 0.0
        for pred, ref in zip(predictions, references):
            ref_set = set(ref)
            num_hits = 0
            sum_precisions = 0.0
            
            for i in range(min(k, len(pred))):
                if pred[i] in ref_set:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    sum_precisions += precision_at_i
            
            num_relevant = min(len(ref), k)
            if num_relevant > 0:
                ap_sum += sum_precisions / num_relevant
                
        metrics[f'MAP@{k}'] = ap_sum / len(predictions) if predictions else 0.0
    
    return metrics


class VoxCeleb2Dataset(Dataset):
    """Simple VoxCeleb2 dataset for retrieval evaluation."""
    
    def __init__(self, root_dir: str, split: str = 'test', max_length: int = 160000):
        """
        Args:
            root_dir: Path to VoxCeleb2 directory
            split: 'test' or 'dev'
            max_length: Maximum audio length in samples (default: 10 seconds at 16kHz)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_length = max_length
        
        # Find all audio files
        split_dir = self.root_dir / split
        self.audio_files = []
        self.speaker_ids = []
        
        # VoxCeleb2 structure: split/speaker_id/video_id/utterance_id.wav
        for speaker_dir in sorted(split_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            
            for video_dir in speaker_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                    
                for audio_file in video_dir.glob('*.wav'):
                    self.audio_files.append(audio_file)
                    self.speaker_ids.append(speaker_id)
        
        # Build speaker map: speaker_id -> list of sample indices
        self.speaker_map = defaultdict(list)
        for idx, speaker_id in enumerate(self.speaker_ids):
            self.speaker_map[speaker_id].append(idx)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Truncate or pad to max_length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.squeeze(0), speaker_id


class AudioRetrievalEvaluator:
    """Evaluator for audio retrieval tasks."""
    
    def __init__(self, 
                 model_ckpt: str,
                 model_cfg: str,
                 dataset_name: str,
                 data_dir: str,
                 output_dir: str,
                 device: str = 'cuda',
                 batch_size: int = 64,
                 max_queries: Optional[int] = None,
                 max_contexts: Optional[int] = None):
        
        self.model_ckpt = model_ckpt
        self.model_cfg = model_cfg
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.batch_size = batch_size
        self.max_queries = max_queries
        self.max_contexts = max_contexts
        
        # Load model
        self.load_model()
        
        # Load dataset
        self.load_dataset()
        
    def load_model(self):
        """Load ReTreever model from checkpoint."""
        print(f"Loading model from {self.model_ckpt}...")
        
        # Load using retreever's load_from_ckpt function
        self.model, self.cfg = load_from_ckpt(
            self.model_ckpt,
            self.model_cfg,
            cache_dir=config.HF_CACHE_DIR,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        
    def load_dataset(self):
        """Load audio dataset (VoxCeleb2 test set)."""
        print(f"Loading {self.dataset_name} dataset from {self.data_dir}...")
        
        if self.dataset_name == "voxceleb2":
            self.dataset = VoxCeleb2Dataset(self.data_dir, split='test')
            self.speaker_map = self.dataset.speaker_map
            
            print(f"✓ Loaded {len(self.dataset)} audio samples from {len(self.speaker_map)} speakers")
            
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not yet supported")
        
    def encode_audio(self, mode: str = 'context') -> np.ndarray:
        """
        Encode all audio samples using the model.
        
        Args:
            mode: 'context' or 'query' (uses appropriate encoder)
            
        Returns:
            embeddings: numpy array of shape (n_samples, emb_dim)
        """
        print(f"Encoding audio as {mode}...")
        
        # Limit samples if requested
        dataset = self.dataset
        if mode == 'query' and self.max_queries:
            indices = list(range(min(self.max_queries, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)
        elif mode == 'context' and self.max_contexts:
            indices = list(range(min(self.max_contexts, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda batch: ([x[0] for x in batch], [x[1] for x in batch])
        )
        
        embeddings = []
        
        with torch.no_grad():
            for waveforms, _ in tqdm(dataloader, desc=f"Encoding {mode}"):
                # Stack waveforms into batch
                batch = torch.stack(waveforms).to(self.device)
                
                # Get embeddings from appropriate encoder
                if mode == 'context':
                    emb = self.model.context_encoder(batch)
                else:  # query
                    emb = self.model.query_encoder(batch)
                
                # Apply projection if it exists
                if mode == 'context' and self.model.context_projection is not None:
                    emb = self.model.context_projection(emb)
                elif mode == 'query' and self.model.query_projection is not None:
                    emb = self.model.query_projection(emb)
                
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"✓ Encoded {len(embeddings)} audio samples to {embeddings.shape[1]}-dim vectors")
        
        return embeddings
    
    def build_index(self, context_embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        print("Building FAISS index...")
        
        dim = context_embeddings.shape[1]
        
        # Use L2 distance (can be changed to inner product if embeddings are normalized)
        index = faiss.IndexFlatL2(dim)
        
        # Add context embeddings
        index.add(context_embeddings.astype('float32'))
        
        print(f"✓ Built FAISS index with {index.ntotal} vectors")
        
        return index
    
    def search(self, index: faiss.Index, query_embeddings: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors.
        
        Returns:
            scores: array of shape (n_queries, k)
            indices: array of shape (n_queries, k)
        """
        print(f"Searching top-{k} neighbors...")
        
        scores, indices = index.search(query_embeddings.astype('float32'), k)
        
        print(f"✓ Retrieved top-{k} for {len(query_embeddings)} queries")
        
        return scores, indices
    
    def get_ground_truth(self, query_indices: List[int]) -> List[List[int]]:
        """
        Get ground truth relevant items for each query.
        For VoxCeleb2: all audio samples from same speaker are relevant, excluding self.
        
        Args:
            query_indices: List of query sample indices
            
        Returns:
            List of lists of relevant context indices
        """
        ground_truth = []
        
        for query_idx in query_indices:
            # Get speaker ID for this query
            speaker_id = self.dataset.speaker_ids[query_idx]
            
            # All samples from same speaker are relevant (excluding self)
            relevant = [idx for idx in self.speaker_map[speaker_id] if idx != query_idx]
            ground_truth.append(relevant)
        
        return ground_truth
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print("=" * 80)
        print(f"EVALUATING {self.dataset_name.upper()} RETRIEVAL")
        print("=" * 80)
        
        # Encode all audio as contexts
        context_embeddings = self.encode_audio(mode='context')
        
        # Encode audio as queries (may be subset)
        query_embeddings = self.encode_audio(mode='query')
        n_queries = len(query_embeddings)
        
        # Determine query indices
        if self.max_queries:
            query_indices = list(range(min(self.max_queries, len(self.dataset))))
        else:
            query_indices = list(range(len(self.dataset)))
        
        # Build FAISS index
        index = self.build_index(context_embeddings)
        
        # Search
        k = 100  # retrieve top-100
        scores, indices = self.search(index, query_embeddings, k=k)
        
        # Get ground truth
        ground_truth = self.get_ground_truth(query_indices)
        
        # Convert indices to predictions list
        predictions = indices.tolist()
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(predictions, ground_truth, k_values=[1, 5, 10, 100])
        
        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:20s}: {value:.4f}")
        
        # Save results
        results = {
            'dataset': self.dataset_name,
            'model_ckpt': str(self.model_ckpt),
            'model_cfg': str(self.model_cfg),
            'n_queries': n_queries,
            'n_contexts': len(context_embeddings),
            'metrics': metrics,
        }
        
        output_file = self.output_dir / f'{self.dataset_name}_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ReTreever on audio retrieval tasks"
    )
    
    parser.add_argument(
        '--model_ckpt',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--model_cfg',
        type=str,
        required=True,
        help='Path to model config (.yaml file)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='voxceleb2',
        choices=['voxceleb2'],
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory (should contain test/ subdirectory for VoxCeleb2)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eval_results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for encoding'
    )
    
    parser.add_argument(
        '--max_queries',
        type=int,
        default=None,
        help='Maximum number of queries to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--max_contexts',
        type=int,
        default=None,
        help='Maximum number of contexts to index (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = AudioRetrievalEvaluator(
        model_ckpt=args.model_ckpt,
        model_cfg=args.model_cfg,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        max_contexts=args.max_contexts,
    )
    
    evaluator.evaluate()


if __name__ == '__main__':
    main()
