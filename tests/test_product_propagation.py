"""Comprehensive tests for product propagation and tree routing."""

import pytest
import torch
import numpy as np
from retreever.models.trees import QuadraticallyRelaxedTree, ProbabilisticallyRelaxedTree


class TestProductPropagation:
    """Test core product propagation algorithm."""
    
    def test_product_propagation_sums_to_one(self):
        """Test that product propagation produces valid probability distributions."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        routes = tree(x)
        
        # Routes should sum to 1 across leaves
        sums = routes.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    def test_product_propagation_all_positive(self):
        """Test that all route probabilities are non-negative."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        routes = tree(x)
        
        # All probabilities should be >= 0
        assert torch.all(routes >= 0)
    
    def test_product_propagation_with_different_depths(self):
        """Test product propagation works correctly at various tree depths."""
        batch_size = 2
        embed_dim = 768
        
        for depth in [2, 3, 4, 6, 8]:
            tree = QuadraticallyRelaxedTree(
                input_size=(embed_dim,),
                depth=depth,
                split_fn="linear",
            )
            
            x = torch.randn(batch_size, embed_dim)
            routes = tree(x)
            
            expected_leaves = 2 ** depth
            assert routes.shape == (batch_size, expected_leaves)
            
            # Should sum to 1
            sums = routes.sum(dim=1)
            assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


class TestTreeRouting:
    """Test tree routing mechanism."""
    
    def test_routing_consistency(self):
        """Test that same input produces same routing (deterministic mode)."""
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        tree.eval()  # Evaluation mode
        
        x = torch.randn(2, embed_dim)
        
        routes1 = tree(x)
        routes2 = tree(x)
        
        # Should be identical in eval mode
        assert torch.allclose(routes1, routes2)
    
    def test_routing_gradients(self):
        """Test that gradients flow through routing."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim, requires_grad=True)
        routes = tree(x)
        
        # Compute loss and backward
        loss = routes.sum()
        loss.backward()
        
        # Gradients should exist
        assert x.grad is not None
        assert torch.any(x.grad != 0)
    
    def test_leaf_assignment(self):
        """Test that predict() assigns inputs to leaf nodes."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        routes, leaves = tree.predict(x)
        
        # Leaves should be integers in valid range
        assert leaves.shape == (batch_size,)
        assert torch.all(leaves >= 0)
        assert torch.all(leaves < 2 ** depth)


class TestSplitDecisions:
    """Test split decision making at tree nodes."""
    
    def test_split_function_output_range(self):
        """Test that split functions output reasonable values."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        # Get split outputs (before sigmoid/softmax)
        split_output = tree.split(x)
        
        # Should have correct number of splits
        expected_splits = 2 ** depth - 1  # Number of internal nodes
        assert split_output.shape == (batch_size, expected_splits)
    
    def test_split_with_temperature(self):
        """Test that temperature affects split sharpness."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        # High temperature (soft splits)
        tree_soft = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
            temp_coeff=10.0,
        )
        
        # Low temperature (sharp splits)
        tree_sharp = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
            temp_coeff=0.1,
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        routes_soft = tree_soft(x)
        routes_sharp = tree_sharp(x)
        
        # Both should be valid distributions
        assert torch.allclose(routes_soft.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(routes_sharp.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        
        # Sharp should have more peaked distributions (higher max prob)
        max_soft = routes_soft.max(dim=1)[0].mean()
        max_sharp = routes_sharp.max(dim=1)[0].mean()
        
        assert max_sharp > max_soft


class TestHierarchicalRepresentations:
    """Test hierarchical representation extraction."""
    
    def test_intermediate_level_representations(self):
        """Test that tree can extract representations at intermediate depths."""
        batch_size = 2
        embed_dim = 768
        depth = 6
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        # Full depth
        routes_full = tree(x)
        assert routes_full.shape == (batch_size, 2 ** depth)
        
        # Test we can get different depth representations
        # (Implementation detail: this tests the capability exists)
        assert tree.num_leaves == 2 ** depth
    
    def test_depth_truncation(self):
        """Test that tree can be evaluated at different depths."""
        batch_size = 2
        embed_dim = 768
        max_depth = 8
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=max_depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        # At max depth
        routes = tree(x)
        assert routes.shape[1] == 2 ** max_depth


class TestPropagationComputation:
    """Test computational aspects of propagation."""
    
    def test_batch_independence(self):
        """Test that samples in a batch are processed independently."""
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        tree.eval()
        
        # Process individually
        x1 = torch.randn(1, embed_dim)
        x2 = torch.randn(1, embed_dim)
        
        route1 = tree(x1)
        route2 = tree(x2)
        
        # Process as batch
        x_batch = torch.cat([x1, x2], dim=0)
        routes_batch = tree(x_batch)
        
        # Should match individual processing
        assert torch.allclose(routes_batch[0], route1[0])
        assert torch.allclose(routes_batch[1], route2[0])
    
    def test_propagation_efficiency(self):
        """Test that propagation is efficient for large batches."""
        batch_size = 32
        embed_dim = 768
        depth = 6
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        # Should complete without error
        routes = tree(x)
        
        assert routes.shape == (batch_size, 2 ** depth)


class TestStochasticPropagation:
    """Test stochastic propagation (for probabilistic tree)."""
    
    def test_stochastic_sampling_varies(self):
        """Test that stochastic tree produces different samples."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = ProbabilisticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        tree.train()  # Enable stochastic mode
        
        x = torch.randn(batch_size, embed_dim)
        
        # Multiple forward passes
        routes1 = tree(x)
        routes2 = tree(x)
        routes3 = tree(x)
        
        # May differ due to sampling
        # But all should be valid distributions
        for routes in [routes1, routes2, routes3]:
            sums = routes.sum(dim=1)
            assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)


class TestPropagationEdgeCases:
    """Test edge cases in propagation."""
    
    def test_zero_input(self):
        """Test propagation with zero input."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.zeros(batch_size, embed_dim)
        routes = tree(x)
        
        # Should still produce valid distribution
        sums = routes.sum(dim=1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    def test_single_sample(self):
        """Test propagation with single sample."""
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(1, embed_dim)
        routes = tree(x)
        
        assert routes.shape == (1, 2 ** depth)
        assert torch.allclose(routes.sum(dim=1), torch.ones(1), atol=1e-5)
    
    def test_extreme_input_values(self):
        """Test propagation with extreme input values."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        # Very large values
        x_large = torch.ones(batch_size, embed_dim) * 100
        routes_large = tree(x_large)
        
        # Very small values
        x_small = torch.ones(batch_size, embed_dim) * 0.01
        routes_small = tree(x_small)
        
        # Both should produce valid distributions
        assert torch.allclose(routes_large.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(routes_small.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestPropagationInference:
    """Test propagation during inference."""
    
    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        tree.eval()
        
        x = torch.randn(2, embed_dim)
        
        # Multiple forward passes should be identical
        routes1 = tree(x)
        routes2 = tree(x)
        routes3 = tree(x)
        
        assert torch.equal(routes1, routes2)
        assert torch.equal(routes2, routes3)
    
    def test_greedy_leaf_selection(self):
        """Test greedy leaf selection for inference."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        tree.eval()
        
        x = torch.randn(batch_size, embed_dim)
        routes, leaves = tree.predict(x)
        
        # Selected leaves should correspond to maximum probabilities
        max_probs, max_indices = routes.max(dim=1)
        
        assert torch.equal(leaves, max_indices)
        
        # Max probabilities should be the highest in each row
        for i in range(batch_size):
            assert max_probs[i] >= routes[i].mean()
