"""Evaluator for retrieval models."""

import numpy as np
import re
import torch
from collections import defaultdict

from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Optional, Dict, List

from retreever.evaluation.metrics import HitK, NDCGK, MAPK, RecallK, MRR


def extract_k(metric_str: str, prefix: str) -> int:
    """
    Given a string starting with a metric prefix (e.g., "hit@" or "ndcg@")
    and ending with a number, extracts and returns the number.
    
    Args:
        metric_str: Metric string like "hit@10"
        prefix: Prefix like "hit@"
        
    Returns:
        The k value (integer)
    """
    if not metric_str.startswith(prefix):
        raise ValueError(f"The string {metric_str} must start with '{prefix}'")

    match = re.search(rf"{prefix}(\d+)$", metric_str)
    if not match:
        raise ValueError(f"The string {metric_str} must end with an integer after '{prefix}'")

    return int(match.group(1))


def load_metric(name: str, args=None):
    """
    Given a metric name, instantiate the right class.
    Note that metrics can be passed as "hit@k", "ndcg@k", etc.
    where k is replaced with any integer.
    
    Args:
        name: Metric name like "hit@10", "ndcg@5", etc.
        args: Optional args object with parameters
        
    Returns:
        Instantiated metric object
    """
    metric_mapping = {
        "hit@": HitK,
        "ndcg@": NDCGK,
        "map@": MAPK,
        "recall@": RecallK,
        "mrr": MRR,
    }

    for prefix, metric_class in metric_mapping.items():
        if name.startswith(prefix):
            if args is None:
                args = OmegaConf.create({})
            args.k = extract_k(name, prefix)
            return metric_class(name, args=args)
    
    # Special case for MRR (no @k)
    if name == "mrr":
        return MRR(name, args=args)
    
    raise ValueError(f"{name} is not a valid metric.")


class RetEvaluator:
    """
    Evaluator for retrieval models.
    
    Builds an index from contexts, runs retrieval on queries, and computes
    standard retrieval metrics like Hit@K, NDCG@K, mAP@K, Recall@K, and MRR.
    """
    
    def __init__(
        self,
        ks: List[int] = [1, 3, 10, 50, 100],
        index_min_relevance: float = 0.0001,
        additional_ctxs_per_device: int = 0,
        additional_ctxs: int = 0,
        **tracked_metrics,
    ):
        """
        Initialize the evaluator.
        
        Args:
            ks: List of k values for metrics (e.g., [1, 3, 10])
            index_min_relevance: Minimum relevance score for indexing contexts
            additional_ctxs_per_device: Number of additional (distractor) contexts per device
            additional_ctxs: Total additional contexts across all devices
            **tracked_metrics: Additional metrics to track (e.g., tree-specific metrics)
        """
        self.hitks = [HitK(name=f"hit@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        self.ndcgs = [NDCGK(name=f"NDCG@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        self.mapks = [MAPK(name=f"mAP@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        self.recallks = [RecallK(name=f"recall@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        
        self.tracked_metrics = tracked_metrics
        self.index_min_relevance = index_min_relevance
        self.ks = ks
        self.additional_ctxs_per_device = additional_ctxs_per_device
        self.additional_ctxs = additional_ctxs
        
        # For multi-label classification tasks
        self.indexed_label_to_uids = defaultdict(set)

    def __call__(
        self, 
        model, 
        data_loader, 
        additional_ctxs_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Args:
            model: The retrieval model to evaluate
            data_loader: DataLoader with (query, context) pairs to evaluate
            additional_ctxs_loader: Optional DataLoader with additional distractor contexts
            
        Returns:
            Dictionary of metric names to values
        """
        model.eval()
        model.reset_index()
        
        # Clear the label to uids mapping for fresh indexing
        self.indexed_label_to_uids.clear()
        
        # Determine device
        device = next(model.parameters()).device
        
        with torch.no_grad():
            predictions = []
            targets = []
            metric_values = {metric_name: [] for metric_name in self.tracked_metrics.keys()}
            
            # Phase 1: Build index with ground-truth contexts
            for batch in tqdm(data_loader, desc="Indexing gt contexts"):
                device_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                
                model.index_ctxs(
                    context_ids=device_batch["context_ids"],
                    context_attn_mask=device_batch["context_attn_mask"],
                    context_names=device_batch["context_uid"],
                    threshold=self.index_min_relevance,
                )
                
                # Track label to uids mapping for multi-label tasks
                if "label" in device_batch:
                    for uid, label in zip(device_batch["context_uid"], device_batch["label"]):
                        self.indexed_label_to_uids[label.item()].add(uid.item())
            
            # Phase 2: Add additional distractorcontexts if provided
            if additional_ctxs_loader is not None:
                total_additional_contexts = max(
                    self.additional_ctxs_per_device, 
                    self.additional_ctxs
                )
                max_batches = total_additional_contexts // additional_ctxs_loader.batch_size
                
                total = min(len(additional_ctxs_loader), max_batches)
                for i, batch in enumerate(
                    tqdm(additional_ctxs_loader, desc="Indexing distractor contexts", total=total)
                ):
                    if i >= max_batches:
                        break
                    
                    device_batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()
                    }
                    
                    model.index_ctxs(
                        context_ids=device_batch["context_ids"],
                        context_attn_mask=device_batch["context_attn_mask"],
                        context_names=device_batch["context_uid"],
                        threshold=self.index_min_relevance,
                    )
                    
                    # Track label to uids mapping
                    if "label" in device_batch:
                        for uid, label in zip(device_batch["context_uid"], device_batch["label"]):
                            self.indexed_label_to_uids[label.item()].add(uid.item())
            
            # Phase 3: Run retrieval evaluation
            for batch in tqdm(data_loader, desc="Retrieval evaluation"):
                device_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                
                # Retrieve top-k contexts for each query
                context_idxs = model.top_contexts(
                    question_ids=device_batch["question_ids"],
                    question_attn_mask=device_batch["question_attn_mask"],
                    k=max(self.ks) + 1,  # +1 to handle self-retrieval in classification tasks
                )

                if "label" in device_batch:
                    # Multi-label classification: filter out self-retrieval
                    predictions.extend([
                        [uid for uid in idxs.keys() if uid != query_uid.item()]
                        for idxs, query_uid in zip(context_idxs, device_batch["context_uid"])
                    ])
                    targets.extend([label.item() for label in device_batch["label"]])
                else:
                    # Standard retrieval: each query has one ground-truth context
                    targets.extend([[uid.item()] for uid in device_batch["context_uid"]])
                    predictions.extend([list(idxs.keys()) for idxs in context_idxs])
                
                # Compute additional tracked metrics (e.g., tree-specific metrics)
                if self.tracked_metrics:
                    model_outputs = model(**device_batch, return_loss=False)
                    
                    for metric_name, metric_call in self.tracked_metrics.items():
                        # Set local loss to True since no distributed gather
                        if hasattr(metric_call, 'local_loss'):
                            metric_call.local_loss = True
                        
                        metric_values[metric_name].append(metric_call(*model_outputs).cpu())
            
            # Convert label IDs to lists of UIDs for multi-label tasks
            if "label" in batch:
                targets = [
                    list(self.indexed_label_to_uids[label])
                    for label in targets
                ]
            
            # Phase 4: Compute final metrics
            metrics = {}
            
            # Standard retrieval metrics
            for i, k in enumerate(self.ks):
                metrics[f"eval/hit@{k}"] = self.hitks[i](
                    [preds[:k] for preds in predictions], targets
                )
                metrics[f"eval/NDCG@{k}"] = self.ndcgs[i](
                    [preds[:k] for preds in predictions], targets
                )
                metrics[f"eval/mAP@{k}"] = self.mapks[i](
                    [preds[:k] for preds in predictions], targets
                )
                metrics[f"eval/recall@{k}"] = self.recallks[i](
                    [preds[:k] for preds in predictions], targets
                )
            
            # Additional tracked metrics (tree-specific, etc.)
            for metric_name, values in metric_values.items():
                metrics[f"eval/{metric_name}"] = np.mean(
                    [v.item() if torch.is_tensor(v) else v for v in values]
                )
            
            return metrics
