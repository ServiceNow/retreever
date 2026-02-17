"""Retrieval evaluation metrics for ReTreever."""

from typing import List
import math
from abc import ABC, abstractmethod


class Metric(ABC):
    """Base class for evaluation metrics."""
    
    def __init__(self, name, **kwargs):
        self.name = name
        
    @abstractmethod
    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()


class HitK(Metric):
    """Hit@K metric - fraction of queries where at least one correct answer appears in top-k."""
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.args = kwargs.get("args", None)

        # Check if "k" is one of the arguments
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ValueError("You need to specify an integer value for k for the Hit@k metric")

    def _compute(self, predicted_ids: List, gt_ids: List):
        """
        Function to compute hit@k for a single instance.
        Returns a number between 0 and 1 indicating the fraction of
        the predicted values that are found in the gt_ids.
        1 indicates that *all* of the predicted ids are in the gt_ids.
        """
        top_k_predicted = predicted_ids[: self.k]  # Take the top-k predicted IDs

        # Count the number of predicted ids that are also in the gt list
        number_of_hits = len([item for item in top_k_predicted if item in gt_ids])
        # Return a partial hit depending on how many of the predicted ids are in the gt list
        return number_of_hits / len(gt_ids)

    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """
        Predictions and references are lists of lists.
        In each sample, we have a list of predictions and a list of gt references.
        Returns the hit value (total hits per number of samples)
        """
        # Compute hit@k metric over the entire dataset
        total_hits = 0
        total_instances = len(predictions)

        # Compute the hits for each sample
        for pred, ref in zip(predictions, references):
            total_hits += self._compute(pred, ref)

        return total_hits / total_instances


class NDCGK(Metric):
    """
    Normalized Discounted Cumulative Gain (NDCG) @ k metric.

    NDCG@k is a measure of ranking quality that evaluates how well the predicted
    ranking of items matches the ground truth. It is calculated as:

        NDCG@k = DCG@k / IDCG@k

    where:
    - DCG@k (Discounted Cumulative Gain) is defined as:
        DCG@k = Σ (rel_i / log2(i + 2)) for i = 1 to k
      rel_i is 1 if the i-th predicted item is in the ground truth, otherwise 0.
      'i' is the rank of the item in the predicted list (0-based indexing).

    - IDCG@k (Ideal Discounted Cumulative Gain) is the maximum possible DCG@k
      assuming an ideal ranking where all relevant items appear in the top positions.

    NDCG@k ranges from 0 to 1:
    - 1 indicates a perfect ranking where all relevant items are in the top positions.
    - 0 indicates no relevant items in the top-k predictions.

    Args:
        name (str): The name of the metric.
        args (dict): A dictionary containing the value of k (e.g., {"k": 5}).

    Raises:
        ValueError: If the value of k is not specified or is invalid.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.args = kwargs.get("args", None)

        # Check if a "k" is specified
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ValueError("You need to specify an integer value for k for the NDCG@k metric")

    def _dcg(self, predicted_ids: List, gt_ids: List):
        """
        Compute Discounted Cumulative Gain (DCG) for a single instance.
        DCG is the sum of relevant items' relevance scores, discounted by their rank.
        """
        dcg = sum(
            1 / math.log2(i + 2)
            for i, pred in enumerate(predicted_ids[: self.k])
            if pred in gt_ids
        )
        return dcg

    def _idcg(self, gt_ids: List):
        """
        Compute Ideal Discounted Cumulative Gain (IDCG) for the ground-truth IDs.
        IDCG is the maximum possible DCG if the ground truth items are ranked perfectly.
        """
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gt_ids), self.k)))
        return idcg

    def _compute(self, predicted_ids: List, gt_ids: List):
        """
        Compute NDCG@k for a single instance.
        NDCG is DCG divided by IDCG. Returns a number between 0 and 1.
        """
        dcg = self._dcg(predicted_ids, gt_ids)
        idcg = self._idcg(gt_ids)
        return dcg / idcg if idcg > 0 else 0.0

    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """
        Predictions and references are lists of lists.
        Computes the NDCG@k metric for the entire dataset.
        """
        total_ndcg = 0.0
        total_instances = len(predictions)

        for pred, ref in zip(predictions, references):
            total_ndcg += self._compute(pred, ref)

        return total_ndcg / total_instances if total_instances > 0 else 0.0


class RecallK(Metric):
    """Recall@K metric - proportion of relevant items retrieved in top-k."""
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        
        self.args = kwargs.get("args", None)
        
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ValueError("You need to specify an integer value for k for the Recall@k metric")
    
    def _compute(self, predicted_ids: List, gt_ids: List):
        """Compute Recall@K for a single instance."""
        if len(gt_ids) == 0:
            return 0.0
        
        top_k_predicted = predicted_ids[:self.k]
        num_relevant_retrieved = len([item for item in top_k_predicted if item in gt_ids])
        
        return num_relevant_retrieved / len(gt_ids)
    
    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """Compute Recall@K over the dataset."""
        total_recall = 0.0
        total_instances = len(predictions)
        
        for pred, ref in zip(predictions, references):
            total_recall += self._compute(pred, ref)
        
        return total_recall / total_instances if total_instances > 0 else 0.0


class MAPK(Metric):
    """
    Mean Average Precision @ K
    
    For each query:
    - AP@K = (1/min(num_positives, k)) * Σ(P@i * rel(i)) for i=1 to k
    where P@i is precision at position i, rel(i) is 1 if item at position i is relevant
    
    mAP@K is the mean of AP@K across all queries.
    """
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.args = kwargs.get("args", None)
        
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ValueError("You need to specify an integer value for k for the mAP@k metric")
    
    def _compute(self, predicted_ids: List, gt_ids: List):
        """Compute Average Precision @ K for a single query."""
        if len(gt_ids) == 0:
            return 0.0
        
        top_k_predicted = predicted_ids[:self.k]
        
        num_hits = 0
        sum_precisions = 0.0
        
        for i, pred_id in enumerate(top_k_predicted):
            if pred_id in gt_ids:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                sum_precisions += precision_at_i
        
        # Normalize by min(number of positives, k)
        num_relevant = min(len(gt_ids), self.k)
        
        return sum_precisions / num_relevant if num_relevant > 0 else 0.0
    
    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """Compute mean Average Precision @ K across all queries."""
        total_ap = 0.0
        total_instances = len(predictions)
        
        for pred, ref in zip(predictions, references):
            total_ap += self._compute(pred, ref)
        
        return total_ap / total_instances if total_instances > 0 else 0.0


class MRR(Metric):
    """Mean Reciprocal Rank - average of reciprocal ranks of the first relevant item."""
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.args = kwargs.get("args", None)
        
        # Optional k parameter to limit search depth
        self.k = self.args.k if self.args and hasattr(self.args, "k") else None
    
    def _compute(self, predicted_ids: List, gt_ids: List):
        """Compute reciprocal rank for a single query."""
        if len(gt_ids) == 0:
            return 0.0
        
        search_list = predicted_ids[:self.k] if self.k is not None else predicted_ids
        
        for i, pred_id in enumerate(search_list):
            if pred_id in gt_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """Compute Mean Reciprocal Rank over the dataset."""
        total_rr = 0.0
        total_instances = len(predictions)
        
        for pred, ref in zip(predictions, references):
            total_rr += self._compute(pred, ref)
        
        return total_rr / total_instances if total_instances > 0 else 0.0
