from retreever.evaluation import Metric
from typing import List
import math


class HitK(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.args = kwargs.get("args", None)

        # Check if an "k" is one of the arguments
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ("You need to specify an integer value for k for the Hit@k metric")

    def _compute(self, predicted_ids: List, gt_ids: List):
        """
        Function to compute hit@k for a single instance.
        Returns a  number between 0 and 1 indicating the fraction of
        the predicted values that found in the gt_ids.
        1 indicates that *all* of the predicted ids are in the gt_ids.
        """
        top_k_predicted = predicted_ids[: self.k]  # Take the top-k predicted IDs
        # Assumes we have a list of gt indices

        # If any one of the predicted indices are in the gt list, we get a 1
        # ## hit = any(item in gt_ids for item in top_k_predicted)
        # ## return 1 if hit else 0

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
    This class computes the Normalized Discounted Cumulative Gain (NDCG) @ k metric.

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
    

class MAPK(Metric):
    """
    Mean Average Precision @ K
    
    For each query:
    - AP@K = (1/min(num_positives, k)) * Σ(P@i * rel(i)) for i=1 to k
    where P@i is precision at position i, rel(i) is 1 if item at position i is relevant
    
    mAP@K is the mean of AP@K across all queries.
    
    This is better than Hit@K or Recall@K for multi-positive scenarios because:
    - It rewards early retrieval of positives
    - It accounts for the number of positives retrieved relative to k
    - It's more interpretable than NDCG
    """
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.args = kwargs.get("args", None)
        
        if self.args and hasattr(self.args, "k"):
            self.k = self.args.k
        else:
            raise ValueError("You need to specify an integer value for k for the mAP@k metric")
    
    def _compute(self, predicted_ids: List, gt_ids: List):
        """
        Compute Average Precision @ K for a single query.
        
        Args:
            predicted_ids: List of predicted item IDs (ranked)
            gt_ids: List of ground truth relevant item IDs
            
        Returns:
            Average Precision @ K (float between 0 and 1)
        """
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
        # This makes AP@K comparable across queries with different numbers of positives
        num_relevant = min(len(gt_ids), self.k)
        
        return sum_precisions / num_relevant if num_relevant > 0 else 0.0
    
    def __call__(self, predictions: List[List], references: List[List], questions=None, ids=None):
        """
        Compute mean Average Precision @ K across all queries.
        """
        total_ap = 0.0
        total_instances = len(predictions)
        
        for pred, ref in zip(predictions, references):
            total_ap += self._compute(pred, ref)
        
        return total_ap / total_instances if total_instances > 0 else 0.0
