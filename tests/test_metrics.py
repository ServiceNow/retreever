"""Unit tests for retrieval metrics."""

import pytest
import torch
from omegaconf import OmegaConf

from retreever.evaluation.metrics import HitK, NDCGK, RecallK, MAPK, MRR


class TestMetrics:
    """Test retrieval metrics."""
    
    def setup_method(self):
        """Setup test data."""
        # Predictions: list of lists of predicted IDs
        self.predictions = [
            [1, 2, 3, 4, 5],  # Query 1
            [10, 11, 2, 13, 14],  # Query 2
            [20, 1, 22, 23, 24],  # Query 3
        ]
        
        # References: list of lists of ground truth IDs
        self.references = [
            [1, 6],  # Query 1 has 2 relevant items
            [2],  # Query 2 has 1 relevant item
            [25],  # Query 3 has 1 relevant item (not in top-5)
        ]
    
    def test_hitk(self):
        """Test Hit@K metric."""
        args = OmegaConf.create({"k": 5})
        metric = HitK(name="hit@5", args=args)
        
        score = metric(self.predictions, self.references)
        
        # Query 1: 1/2 = 0.5
        # Query 2: 1/1 = 1.0
        # Query 3: 0/1 = 0.0
        # Average: (0.5 + 1.0 + 0.0) / 3 = 0.5
        assert abs(score - 0.5) < 1e-6
    
    def test_ndcgk(self):
        """Test NDCG@K metric."""
        args = OmegaConf.create({"k": 5})
        metric = NDCGK(name="ndcg@5", args=args)
        
        score = metric(self.predictions, self.references)
        
        # NDCG should be between 0 and 1
        assert 0 <= score <= 1
    
    def test_recallk(self):
        """Test Recall@K metric."""
        args = OmegaConf.create({"k": 5})
        metric = RecallK(name="recall@5", args=args)
        
        score = metric(self.predictions, self.references)
        
        # Same as HitK for our test data
        assert abs(score - 0.5) < 1e-6
    
    def test_mapk(self):
        """Test mAP@K metric."""
        args = OmegaConf.create({"k": 5})
        metric = MAPK(name="map@5", args=args)
        
        score = metric(self.predictions, self.references)
        
        # MAP should be between 0 and 1
        assert 0 <= score <= 1
    
    def test_mrr(self):
        """Test MRR metric."""
        args = OmegaConf.create({})
        metric = MRR(name="mrr", args=args)
        
        score = metric(self.predictions, self.references)
        
        # Query 1: 1/1 = 1.0 (first position)
        # Query 2: 1/3 = 0.333 (third position)
        # Query 3: 0 (not found)
        # Average: (1.0 + 0.333 + 0) / 3 ≈ 0.444
        assert abs(score - 0.444) < 0.01
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [[1], [2], [3]]
        references = [[1], [2], [3]]
        
        args = OmegaConf.create({"k": 1})
        
        # All metrics should be 1.0 for perfect predictions
        assert HitK(name="hit@1", args=args)(predictions, references) == 1.0
        assert NDCGK(name="ndcg@1", args=args)(predictions, references) == 1.0
        assert RecallK(name="recall@1", args=args)(predictions, references) == 1.0
    
    def test_empty_predictions(self):
        """Test metrics with no relevant items retrieved."""
        predictions = [[10, 11, 12], [20, 21, 22]]
        references = [[1], [2]]
        
        args = OmegaConf.create({"k": 3})
        
        # All metrics should be 0.0 when nothing is retrieved
        assert HitK(name="hit@3", args=args)(predictions, references) == 0.0
        assert NDCGK(name="ndcg@3", args=args)(predictions, references) == 0.0
        assert RecallK(name="recall@3", args=args)(predictions, references) == 0.0
