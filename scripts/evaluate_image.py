#!/usr/bin/env python3
"""
Evaluation script for image retrieval models (ReTreever).

Evaluates trained ReTreever checkpoints on image retrieval datasets:
- ImageNet-1K (image → image retrieval, same class = relevant)

Computes standard retrieval metrics:
- Hit@k (k=1, 5, 10)
- NDCG@k (k=10, 100)
- MAP@k (k=10, 100)

Usage:
    python scripts/evaluate_image.py --model_ckpt checkpoints/imagenet_model.pt \
                                      --model_cfg checkpoints/config.yaml \
                                      --dataset imagenet \
                                      --data_dir /path/to/imagenet1k \
                                      --output_dir ./eval_results \
                                      --device cuda

    # For custom batch sizes or number of queries
    python scripts/evaluate_image.py --model_ckpt checkpoints/model.pt \
                                      --model_cfg config.yaml \
                                      --dataset imagenet \
                                      --data_dir /path/to/imagenet \
                                      --batch_size 128 \
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

# Try to import vision dependencies
try:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:
    print("WARNING: torchvision not found. Install with: pip install torchvision")
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


class ImageRetrievalEvaluator:
    """Evaluator for image retrieval tasks."""
    
    def __init__(self, 
                 model_ckpt: str,
                 model_cfg: str,
                 dataset_name: str,
                 data_dir: str,
                 output_dir: str,
                 device: str = 'cuda',
                 batch_size: int = 128,
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
        """Load image dataset (ImageNet validation set)."""
        print(f"Loading {self.dataset_name} dataset from {self.data_dir}...")
        
        if self.dataset_name == "imagenet":
            # ImageNet validation set - standard transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            val_dir = os.path.join(self.data_dir, 'val')
            self.dataset = datasets.ImageFolder(val_dir, transform=transform)
            
            # Build label map: class_idx -> list of sample indices
            self.label_map = defaultdict(list)
            for idx, (_, label) in enumerate(self.dataset.imgs):
                self.label_map[label].append(idx)
                
            print(f"✓ Loaded {len(self.dataset)} images from {len(self.label_map)} classes")
            
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not yet supported")
        
    def encode_images(self, mode: str = 'context') -> np.ndarray:
        """
        Encode all images using the model.
        
        Args:
            mode: 'context' or 'query' (uses appropriate encoder)
            
        Returns:
            embeddings: numpy array of shape (n_images, emb_dim)
        """
        print(f"Encoding images as {mode}...")
        
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
            pin_memory=True
        )
        
        embeddings = []
        
        with torch.no_grad():
            for batch, _ in tqdm(dataloader, desc=f"Encoding {mode}"):
                batch = batch.to(self.device)
                
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
        print(f"✓ Encoded {len(embeddings)} images to {embeddings.shape[1]}-dim vectors")
        
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
        For ImageNet: all images with same class are relevant, excluding self.
        
        Args:
            query_indices: List of query sample indices
            
        Returns:
            List of lists of relevant context indices
        """
        ground_truth = []
        
        for query_idx in query_indices:
            # Get label for this query
            _, label = self.dataset.imgs[query_idx]
            
            # All samples with same label are relevant (excluding self)
            relevant = [idx for idx in self.label_map[label] if idx != query_idx]
            ground_truth.append(relevant)
        
        return ground_truth
    
    def evaluate(self):
        """Run full evaluation pipeline."""
        print("=" * 80)
        print(f"EVALUATING {self.dataset_name.upper()} RETRIEVAL")
        print("=" * 80)
        
        # Encode all images as contexts
        context_embeddings = self.encode_images(mode='context')
        
        # Encode images as queries (may be subset)
        query_embeddings = self.encode_images(mode='query')
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
        description="Evaluate ReTreever on image retrieval tasks"
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
        default='imagenet',
        choices=['imagenet'],
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory (should contain val/ subdirectory for ImageNet)'
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
        default=128,
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
    evaluator = ImageRetrievalEvaluator(
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
