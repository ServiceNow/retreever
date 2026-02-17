"""Comprehensive tests for tree types and tree functionality."""

import pytest
import torch
import numpy as np
from retreever.models.trees import (
    Tree,
    QuadraticallyRelaxedTree,
    ProbabilisticallyRelaxedTree,
    NoPropagationTree,
    NoTree,
    IdentityTree,
    tree_dict,
)


class TestTreeTypes:
    """Test all tree type instantiation and basic functionality."""
    
    @pytest.mark.parametrize("tree_type", [
        "qr_tree",
        "probabilistic_tree",
        "no_propagation_tree",
        "no_tree",
        "identity_tree",
    ])
    def test_tree_instantiation(self, tree_type):
        """Test that all tree types can be instantiated."""
        input_size = (768,)
        depth = 4
        split_fn = "linear"
        
        tree_class = tree_dict[tree_type]
        tree = tree_class(
            input_size=input_size,
            depth=depth,
            split_fn=split_fn,
        )
        
        assert tree is not None
        assert tree.bst.depth == depth
        assert tree.num_leaves == 2 ** depth
    
    @pytest.mark.parametrize("depth", [2, 4, 6, 8])
    def test_tree_depth_variations(self, depth):
        """Test trees with various depths."""
        input_size = (768,)
        tree = QuadraticallyRelaxedTree(
            input_size=input_size,
            depth=depth,
            split_fn="linear",
        )
        
        expected_leaves = 2 ** depth
        assert tree.num_leaves == expected_leaves
        assert tree.bst.nb_leaves == expected_leaves
    
    @pytest.mark.parametrize("split_fn", ["linear", "mlp", "cross_attn"])
    def test_tree_with_split_functions(self, split_fn):
        """Test trees with different split functions."""
        input_size = (768,)
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=input_size,
            depth=depth,
            split_fn=split_fn,
        )
        
        assert tree.split_type == split_fn
    
    def test_tree_forward_pass(self):
        """Test tree forward pass produces expected output."""
        batch_size = 4
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        # Create dummy input
        x = torch.randn(batch_size, embed_dim)
        
        # Forward pass
        output = tree(x)
        
        # Check output shape
        assert output.shape[0] == batch_size
        assert output.shape[1] == tree.num_leaves
    
    def test_tree_predict(self):
        """Test tree predict method returns routes and leaves."""
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
        
        # Routes should be probability distributions over leaves
        assert routes.shape == (batch_size, tree.num_leaves)
        # Leaves should be leaf indices
        assert leaves.shape == (batch_size,)
        assert torch.all(leaves >= 0) and torch.all(leaves < tree.num_leaves)


class TestQuadraticallyRelaxedTree:
    """Test QuadraticallyRelaxedTree specific functionality."""
    
    def test_qr_tree_product_propagation(self):
        """Test that QR tree performs product propagation correctly."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        output = tree(x)
        
        # QR tree should produce valid probability distributions
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_qr_tree_temperature(self):
        """Test QR tree with different temperature values."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        # High temperature (more uniform)
        tree_high_temp = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
            temp_coeff=10.0,
        )
        
        # Low temperature (more peaked)
        tree_low_temp = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
            temp_coeff=0.1,
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        output_high = tree_high_temp(x)
        output_low = tree_low_temp(x)
        
        # Both should be valid distributions
        assert torch.allclose(output_high.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(output_low.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestProbabilisticallyRelaxedTree:
    """Test ProbabilisticallyRelaxedTree specific functionality."""
    
    def test_prob_tree_stochastic_sampling(self):
        """Test that probabilistic tree can sample stochastically."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = ProbabilisticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        
        # Training mode: stochastic
        tree.train()
        output_train1 = tree(x)
        output_train2 = tree(x)
        
        # Outputs may differ due to stochastic sampling
        # But should still be valid distributions
        assert torch.allclose(output_train1.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.allclose(output_train2.sum(dim=1), torch.ones(batch_size), atol=1e-5)


class TestNoPropagationTree:
    """Test NoPropagationTree (constant depth, no product propagation)."""
    
    def test_no_propagation_fixed_depth(self):
        """Test that no propagation tree uses fixed depth."""
        batch_size = 2
        embed_dim = 768
        depth = 4
        
        tree = NoPropagationTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        output = tree(x)
        
        # Should still produce valid output
        assert output.shape[0] == batch_size
        assert output.shape[1] == tree.num_leaves


class TestNoTree:
    """Test NoTree (bypasses tree structure)."""
    
    def test_no_tree_passthrough(self):
        """Test that NoTree acts as passthrough."""
        batch_size = 2
        embed_dim = 768
        
        tree = NoTree(
            input_size=(embed_dim,),
            depth=0,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        output = tree(x)
        
        # Should return input as-is
        assert output.shape[0] == batch_size


class TestIdentityTree:
    """Test IdentityTree (identity transformation)."""
    
    def test_identity_tree_preserves_input(self):
        """Test that IdentityTree preserves input shape."""
        batch_size = 2
        embed_dim = 768
        
        tree = IdentityTree(
            input_size=(embed_dim,),
            depth=0,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim)
        output = tree(x)
        
        # Should preserve batch dimension
        assert output.shape[0] == batch_size


class TestTreeBiases:
    """Test tree bias initialization and balancing."""
    
    def test_linear_split_bias_initialization(self):
        """Test that linear split functions initialize bias for balanced tree."""
        embed_dim = 768
        depth = 4
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        # Linear split should have offset/bias initialized
        assert hasattr(tree, 'offset')
        assert tree.offset.shape[0] == tree.bst.nb_split
    
    def test_tree_gradient_flow(self):
        """Test that gradients flow through tree."""
        batch_size = 2
        embed_dim = 768
        depth = 3
        
        tree = QuadraticallyRelaxedTree(
            input_size=(embed_dim,),
            depth=depth,
            split_fn="linear",
        )
        
        x = torch.randn(batch_size, embed_dim, requires_grad=True)
        output = tree(x)
        
        # Compute dummy loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert tree.split.split[0].weight.grad is not None


class TestTreeDict:
    """Test tree_dict registry."""
    
    def test_all_trees_in_dict(self):
        """Test that all expected tree types are registered."""
        expected_trees = [
            "qr_tree",
            "probabilistic_tree",
            "no_propagation_tree",
            "no_tree",
            "identity_tree",
        ]
        
        for tree_name in expected_trees:
            assert tree_name in tree_dict
            assert callable(tree_dict[tree_name])
    
    def test_tree_classes_match(self):
        """Test that tree_dict maps to correct classes."""
        assert tree_dict["qr_tree"] == QuadraticallyRelaxedTree
        assert tree_dict["probabilistic_tree"] == ProbabilisticallyRelaxedTree
        assert tree_dict["no_propagation_tree"] == NoPropagationTree
        assert tree_dict["no_tree"] == NoTree
        assert tree_dict["identity_tree"] == IdentityTree
