import numpy as np
import re
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

from retreever.evaluation.metrics import HitK, NDCGK, MAPK
from collections import defaultdict


def extract_k(metric_str: str, prefix: str):
    """
    Given a string starting with a metric prefix (e.g., "hit@" or "ndcg@")
    and ending with a number, extracts and returns the number.
    """

    if not metric_str.startswith(prefix):
        raise ValueError(f"The string {metric_str} must start with '{prefix}'")

    match = re.search(rf"{prefix}(\d+)$", metric_str)
    if not match:
        raise ValueError(f"The string {metric_str} must end with an integer after '{prefix}'")

    return int(match.group(1))


def load_metric(name, file_name=None, args=None):
    """
    Given a metric name, instantiate the right class.
    Note that the hit metric can be passed as "hit@k"
    where k is replaced with any integer.
    """

    metric_mapping = {
        "hit@": HitK,
        "ndcg@": NDCGK,
    }

    for prefix, _ in metric_mapping.items():
        if name.startswith(prefix):
            args.k = extract_k(name, prefix)
            matched_prefix = prefix
            break
    else:
        raise ValueError(f"{name} is not a valid metric.")

    return metric_mapping[matched_prefix](name, file_name=file_name, args=args)


class RetEvaluator:
    def __init__(
        self,
        ks=[1, 3, 10, 50, 100],
        index_min_relevance=0.0001,
        additional_ctxs_per_device=0,
        additional_ctxs=0,
        **tracked_metrics,
    ):
        self.hitks = [HitK(name=f"hit@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        self.ndcgs = [NDCGK(name=f"NDCG@{k}", args=OmegaConf.create({"k": k})) for k in ks]
        self.mapks = [MAPK(name=f"mAP@{k}", args=OmegaConf.create({"k": k})) for k in ks]  # ADD THIS

        self.tracked_metrics = tracked_metrics
        self.index_min_relevance = index_min_relevance
        self.ks = ks
        self.additional_ctxs_per_device = additional_ctxs_per_device
        self.additional_ctxs = additional_ctxs
        
        self.indexed_label_to_uids = defaultdict(set)

        

    def __call__(self, model, data_loader, additional_ctxs_loader=None) -> float:
        """Compute generation metrics."""
        
        model.eval()
        model.reset_index()
        
        # clear the label to uids mapping for fresh indexing
        self.indexed_label_to_uids.clear()
        
        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            predictions = []
            targets = []
            metric_values = {metric_name: [] for metric_name in self.tracked_metrics.keys()}
            
            # Build index with ground-truth contexts
            for batch in tqdm(data_loader, desc="Indexing gt contexts"):
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                model.index_ctxs(
                    context_ids=device_batch["context_ids"],
                    context_attn_mask=device_batch["context_attn_mask"],
                    context_names=device_batch["context_uid"],
                    threshold=self.index_min_relevance,
                )
                
                # track label to uids mapping if label information present
                if "label" in device_batch:
                    for uid, label in zip(device_batch["context_uid"], device_batch["label"]):
                        self.indexed_label_to_uids[label.item()].add(uid.item())
            
            # Add additional contexts if provided
            if additional_ctxs_loader is not None:
                total_additional_contexts = self.additional_ctxs_per_device
                max_batches = total_additional_contexts // additional_ctxs_loader.batch_size
                
                total = min(len(additional_ctxs_loader), max_batches)
                for i, batch in enumerate(tqdm(additional_ctxs_loader, desc="Indexing irrelevant contexts", total=total)):
                    if i >= max_batches:
                        break
                    
                    device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    
                    model.index_ctxs(
                        context_ids=device_batch["context_ids"],
                        context_attn_mask=device_batch["context_attn_mask"],
                        context_names=device_batch["context_uid"],
                        threshold=self.index_min_relevance,
                    )
                    
                    # track label to uids mapping if label information present
                    if "label" in device_batch:
                        for uid, label in zip(device_batch["context_uid"], device_batch["label"]):
                            self.indexed_label_to_uids[label.item()].add(uid.item())
            
            
            # Run evaluation
            for batch in tqdm(data_loader, desc="Ret evaluation"):
                device_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                context_idxs = model.top_contexts(
                    question_ids=device_batch["question_ids"],
                    question_attn_mask=device_batch["question_attn_mask"],
                    k=max(self.ks) + 1, # One more as for label based tasks, we want to avoid counting retrieving the query itself
                )

                if "label" in device_batch:
                    # Classification mode: multiple positives per query
                    # targets.extend([
                    #     list(self.indexed_label_to_uids[label.item()])
                    #     for label in device_batch["label"]
                    # ])
                    predictions.extend([
                        [uid for uid in idxs.keys() if uid != query_uid.item()]
                        for idxs, query_uid in zip(context_idxs, device_batch["context_uid"])
                    ])
                    targets.extend([label.item() for label in device_batch["label"]])
                else:
                    targets.extend([[i.item()] for i in device_batch["context_uid"]])
                    predictions.extend([list(idxs.keys()) for idxs in context_idxs])
                
                # Pass all required arguments to model.forward
                model_outputs = model(**device_batch, return_loss=False)
                
                for metric_name, metric_call in self.tracked_metrics.items():
                    # Set local loss to True since no distributed gather is happening
                    if hasattr(metric_call, 'local_loss'):
                        metric_call.local_loss = True
                    
                    metric_values[metric_name].append(metric_call(*model_outputs).cpu())
                    
            if "label" in device_batch:
                # Classification mode: multiple positives per query
                print("Gathering positive labels")
                targets = [
                    list(self.indexed_label_to_uids[label])
                    for label in targets
                ]
            
            # Compute metrics
            metrics = {
                **{
                    f"eval/hit@{k}": self.hitks[i]([preds[:k] for preds in predictions], targets)
                    for i, k in enumerate(self.ks)
                },
                **{
                    f"eval/NDCG@{k}": self.ndcgs[i]([preds[:k] for preds in predictions], targets)
                    for i, k in enumerate(self.ks)
                },
                **{
                    f"eval/mAP@{k}": self.mapks[i]([preds[:k] for preds in predictions], targets)
                    for i, k in enumerate(self.ks)
                },
            }

            metrics |= {
                f"eval/{metric_name}": np.mean(values) for metric_name, values in metric_values.items()
            }
            
            return metrics
