#!/usr/bin/env python3
"""
Evaluation script for text retrieval models (ReTreever).

Evaluates trained ReTreever checkpoints on text retrieval datasets:
- Natural Questions (NQ)
- HotpotQA
- MS MARCO (optional)

Usage:
    python scripts/evaluate_text.py \\
        --model_ckpt path/to/checkpoint.pt \\
        --model_cfg path/to/config.yaml \\
        --dataset nq \\
        --output_dir ./results

Based on research-dssk/comprehensive_evaluation.py
"""

import argparse
import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from retreever import config
from retreever.models.retreever import load_from_ckpt


def compute_metrics(predictions: List[List], references: List[List], k_values: List[int]) -> Dict:
    """Compute retrieval metrics at various k values.
    
    Args:
        predictions: List of ranked document IDs per query
        references: List of relevant document IDs per query  
        k_values: K values to compute metrics for
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    for k in k_values:
        # Hit@k (Recall@k for single positive)
        hits = 0
        for pred, ref in zip(predictions, references):
            top_k = pred[:k]
            if any(r in top_k for r in ref):
                hits += 1
        metrics[f'hit@{k}'] = hits / len(predictions) if predictions else 0.0
        
        # NDCG@k
        total_ndcg = 0.0
        for pred, ref in zip(predictions, references):
            dcg = sum(
                1 / math.log2(i + 2)
                for i, p in enumerate(pred[:k])
                if p in ref
            )
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(ref), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg
        metrics[f'ndcg@{k}'] = total_ndcg / len(predictions) if predictions else 0.0
        
        # MAP@k
        total_ap = 0.0
        for pred, ref in zip(predictions, references):
            if len(ref) == 0:
                continue
            top_k_pred = pred[:k]
            num_hits = 0
            sum_precisions = 0.0
            for i, pred_id in enumerate(top_k_pred):
                if pred_id in ref:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    sum_precisions += precision_at_i
            num_relevant = min(len(ref), k)
            ap = sum_precisions / num_relevant if num_relevant > 0 else 0.0
            total_ap += ap
        metrics[f'map@{k}'] = total_ap / len(predictions) if predictions else 0.0
    
    return metrics


class TextRetrievalEvaluator:
    """Evaluator for text retrieval models."""
    
    def __init__(
        self,
        model_ckpt: str,
        model_cfg: str,
        dataset_name: str,
        output_dir: str,
        k_values: Optional[List[int]] = None,
        batch_size: int = 64,
        index_type: str = "faiss",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize evaluator.
        
        Args:
            model_ckpt: Path to model checkpoint (.pt file)
            model_cfg: Path to model config (.yaml file)
            dataset_name: Dataset to evaluate on ('nq', 'hotpotqa', etc.)
            output_dir: Directory to save results
            k_values: K values for metrics (default: [1, 10, 100])
            batch_size: Batch size for encoding
            index_type: Indexing strategy ('faiss' or 'faiss_multi')
            device: Device to run on
        """
        self.model_ckpt = model_ckpt
        self.model_cfg = model_cfg
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_values = k_values or [1, 10, 100]
        self.batch_size = batch_size
        self.index_type = index_type
        self.device = device
        
        print("=" * 80)
        print(f"TEXT RETRIEVAL EVALUATION")
        print("=" * 80)
        print(f"Model: {model_ckpt}")
        print(f"Dataset: {dataset_name}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"K values: {self.k_values}")
        print("=" * 80)
        print()
        
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
        
        print(f"✓ Model loaded successfully")
        print(f"  Tree type: {self.model.tree_type}")
        print(f"  Tree depth: {getattr(self.model, 'tree_depth', 'N/A')}")
        print(f"  Encoder: {self.model.encoder_type}")
        print()
        
    def load_dataset(self):
        """Load evaluation dataset.
        
        Expected format:
        - queries: List of query texts
        - contexts: List of context texts  
        - qrels: Dict[query_id, List[context_id]] - relevance judgments
        """
        print(f"Loading dataset: {self.dataset_name}...")
        
        # TODO: Implement dataset loading based on your data format
        # For now, using placeholder
        
        # Example structure (replace with actual loading logic):
        self.queries = []  # List of query texts
        self.contexts = []  # List of context texts
        self.qrels = {}  # {query_id: [relevant_context_ids]}
        
        # Placeholder: You would load from your actual dataset
        raise NotImplementedError(
            f"Dataset loading not implemented for {self.dataset_name}. "
            "Please implement load_dataset() method with your data loading logic."
        )
        
        print(f"✓ Dataset loaded")
        print(f"  # Queries: {len(self.queries)}")
        print(f"  # Contexts: {len(self.contexts)}")
        print()
        
    def encode_contexts(self) -> torch.Tensor:
        """Encode all contexts."""
        print("Encoding contexts...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.contexts), self.batch_size)):
                batch = self.contexts[i:i + self.batch_size]
                
                # Tokenize
                # TODO: Implement based on your model's tokenizer
                
                embeddings = None  # Get from model
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def encode_queries(self) -> torch.Tensor:
        """Encode all queries."""
        print("Encoding queries...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.queries), self.batch_size)):
                batch = self.queries[i:i + self.batch_size]
                
                # Tokenize and encode
                # TODO: Implement based on your model's tokenizer
                
                embeddings = None  # Get from model
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def build_index(self, context_embeddings: torch.Tensor):
        """Build FAISS index from context embeddings."""
        print(f"Building {self.index_type} index...")
        
        # Convert to numpy
        embeddings_np = context_embeddings.numpy()
        
        # Build index
        if self.index_type == "faiss":
            self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
            self.index.add(embeddings_np)
        else:
            raise NotImplementedError(f"Index type {self.index_type} not implemented")
        
        print(f"✓ Index built with {self.index.ntotal} vectors")
        print()
    
    def search(self, query_embeddings: torch.Tensor) -> List[List[int]]:
        """Search for top-k contexts for each query."""
        print("Searching...")
        
        query_np = query_embeddings.numpy()
        k = max(self.k_values)
        
        _, indices = self.index.search(query_np, k)
        
        return indices.tolist()
    
    def evaluate(self) -> Dict:
        """Run full evaluation pipeline."""
        print()
        print("=" * 80)
        print("RUNNING EVALUATION")
        print("=" * 80)
        print()
        
        # Encode
        context_embeddings = self.encode_contexts()
        query_embeddings = self.encode_queries()
        
        # Build index
        self.build_index(context_embeddings)
        
        # Search
        predictions = self.search(query_embeddings)
        
        # Prepare references
        references = [self.qrels.get(i, []) for i in range(len(self.queries))]
        
        # Compute metrics
        metrics = compute_metrics(predictions, references, self.k_values)
        
        # Print results
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:15s}: {value:.4f}")
        print("=" * 80)
        print()
        
        # Save results
        results = {
            'model': self.model_ckpt,
            'dataset': self.dataset_name,
            'metrics': metrics,
            'config': {
                'batch_size': self.batch_size,
                'index_type': self.index_type,
                'k_values': self.k_values,
            }
        }
        
        output_file = self.output_dir / f"eval_results_{self.dataset_name}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate text retrieval models")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to model config")
    parser.add_argument("--dataset", type=str, required=True, choices=["nq", "hotpotqa", "msmarco"], help="Dataset to evaluate on")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--index_type", type=str, default="faiss", choices=["faiss", "faiss_multi"], help="Index type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
    evaluator = TextRetrievalEvaluator(
        model_ckpt=args.model_ckpt,
        model_cfg=args.model_cfg,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        index_type=args.index_type,
        device=args.device,
    )
    
    metrics = evaluator.evaluate()
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
