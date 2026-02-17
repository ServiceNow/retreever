"""Comprehensive tests for split functions."""

import pytest
import torch
import numpy as np
from retreever.models.split_functions import (
    LinearSplit,
    MLPSplit,
    CrossAttentionSplit,
    split_dict,
)


class TestSplitFunctionTypes:
    """Test all split function types."""
    
    @pytest.mark.parametrize("split_type", ["linear", "mlp", "cross_attn"])
    def test_split_function_instantiation(self, split_type):
        """Test that all split functions can be instantiated."""
        in_size = (768,)
        nb_splits = 16
        tree_depth = 4
        
        split_class = split_dict[split_type]
        split_fn = split_class(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=tree_depth,
        )
        
        assert split_fn is not None
    
    @pytest.mark.parametrize("split_type", ["linear", "mlp", "cross_attn"])
    def test_split_function_forward(self, split_type):
        """Test forward pass for all split functions."""
        batch_size = 4
        in_size = (768,)
        nb_splits = 16
        tree_depth = 4
        
        split_class = split_dict[split_type]
        split_fn = split_class(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=tree_depth,
        )
        
        x = torch.randn(batch_size, 768)
        output = split_fn(x)
        
        assert output.shape == (batch_size, nb_splits)


class TestLinearSplit:
    """Test LinearSplit functionality."""
    
    def test_linear_split_basic(self):
        """Test basic linear split functionality."""
        in_size = (768,)
        nb_splits = 16
        
        split = LinearSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        batch_size = 4
        x = torch.randn(batch_size, 768)
        output = split(x)
        
        assert output.shape == (batch_size, nb_splits)
    
    def test_linear_split_different_input_sizes(self):
        """Test linear split with various input sizes."""
        nb_splits = 16
        
        # Test with different embedding dimensions
        for emb_dim in [384, 768, 1024, 2048]:
            split = LinearSplit(
                in_size=(emb_dim,),
                nb_splits=nb_splits,
                tree_depth=4,
            )
            
            x = torch.randn(2, emb_dim)
            output = split(x)
            
            assert output.shape == (2, nb_splits)
    
    def test_linear_split_parameter_initialization(self):
        """Test that linear split parameters are initialized correctly."""
        in_size = (768,)
        nb_splits = 16
        
        split = LinearSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        # Check that _init_params can be called
        split._init_params(start_idx=0)
        
        # Check that weights exist
        for layer in split.split:
            if hasattr(layer, 'weight'):
                assert layer.weight is not None
    
    def test_linear_split_gradients(self):
        """Test gradient flow through linear split."""
        in_size = (768,)
        nb_splits = 16
        
        split = LinearSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        x = torch.randn(2, 768, requires_grad=True)
        output = split(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestMLPSplit:
    """Test MLPSplit functionality."""
    
    def test_mlp_split_basic(self):
        """Test basic MLP split functionality."""
        in_size = (768,)
        nb_splits = 16
        
        split = MLPSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        batch_size = 4
        x = torch.randn(batch_size, 768)
        output = split(x)
        
        assert output.shape == (batch_size, nb_splits)
    
    def test_mlp_split_has_hidden_layers(self):
        """Test that MLP split has multiple layers."""
        in_size = (768,)
        nb_splits = 16
        
        split = MLPSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        # MLP should have more than one linear layer
        linear_layers = [m for m in split.split if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) > 1
    
    def test_mlp_split_with_hidden_dim(self):
        """Test MLP split with custom hidden dimension."""
        in_size = (768,)
        nb_splits = 16
        hidden_dim = 512
        
        split = MLPSplit(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
            hidden_size=hidden_dim,
        )
        
        x = torch.randn(2, 768)
        output = split(x)
        
        assert output.shape == (2, nb_splits)
    
    def test_mlp_split_nonlinearity(self):
        """Test that MLP split includes nonlinear activations."""
        in_size = (768,)
        nb_splits = 16
        
        split = MLPSplit(in_size=in_size, nb_splits=nb_splits, tree_depth=4)
        
        # Check for activation functions
        has_nonlinearity = any(
            isinstance(m, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh))
            for m in split.split
        )
        assert has_nonlinearity


class TestCrossAttentionSplit:
    """Test CrossAttentionSplit functionality."""
    
    def test_cross_attn_split_basic(self):
        """Test basic cross-attention split functionality."""
        in_size = (768,)
        nb_splits = 16
        
        split = CrossAttentionSplit(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 768)
        output = split(x)
        
        assert output.shape == (batch_size, nb_splits)
    
    def test_cross_attn_with_node_embeddings(self):
        """Test cross-attention split with node embedding strategy."""
        in_size = (768,)
        nb_splits = 16
        embedding_dim = 768
        
        split = CrossAttentionSplit(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
            embedding_dim=embedding_dim,
            num_embeddings_per_node=1,
        )
        
        x = torch.randn(2, 768)
        output = split(x)
        
        assert output.shape == (2, nb_splits)
    
    def test_cross_attn_with_multiple_embeddings_per_node(self):
        """Test cross-attention with multiple embeddings per node."""
        in_size = (768,)
        nb_splits = 16
        
        split = CrossAttentionSplit(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
            embedding_dim=768,
            num_embeddings_per_node=4,
        )
        
        x = torch.randn(2, 768)
        output = split(x)
        
        assert output.shape == (2, nb_splits)
    
    def test_cross_attn_with_sequence_input(self):
        """Test cross-attention with sequence input (tokens)."""
        in_size = (768,)
        nb_splits = 16
        
        split = CrossAttentionSplit(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
        )
        
        # Sequence input: (batch, seq_len, hidden_dim)
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 768)
        
        # Reshape for split function
        x_flat = x.mean(dim=1)  # Average pooling
        output = split(x_flat)
        
        assert output.shape == (batch_size, nb_splits)


class TestSplitFunctionDict:
    """Test split_dict registry."""
    
    def test_all_splits_in_dict(self):
        """Test that all expected split functions are registered."""
        expected_splits = ["linear", "mlp", "cross_attn"]
        
        for split_name in expected_splits:
            assert split_name in split_dict
            assert callable(split_dict[split_name])
    
    def test_split_classes_match(self):
        """Test that split_dict maps to correct classes."""
        assert split_dict["linear"] == LinearSplit
        assert split_dict["mlp"] == MLPSplit
        assert split_dict["cross_attn"] == CrossAttentionSplit


class TestSplitFunctionDifferentDepths:
    """Test split functions with various tree depths."""
    
    @pytest.mark.parametrize("depth", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("split_type", ["linear", "mlp", "cross_attn"])
    def test_split_with_varying_depths(self, depth, split_type):
        """Test split functions work with different tree depths."""
        in_size = (768,)
        nb_splits = 2 ** depth - 1  # Internal nodes
        
        split_class = split_dict[split_type]
        split_fn = split_class(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=depth,
        )
        
        x = torch.randn(2, 768)
        output = split_fn(x)
        
        assert output.shape == (2, nb_splits)


class TestSplitFunctionBatching:
    """Test split functions with different batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32])
    @pytest.mark.parametrize("split_type", ["linear", "mlp"])
    def test_split_with_varying_batch_sizes(self, batch_size, split_type):
        """Test split functions handle various batch sizes."""
        in_size = (768,)
        nb_splits = 16
        
        split_class = split_dict[split_type]
        split_fn = split_class(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
        )
        
        x = torch.randn(batch_size, 768)
        output = split_fn(x)
        
        assert output.shape == (batch_size, nb_splits)


class TestSplitFunctionReinitialization:
    """Test parameter reinitialization for split functions."""
    
    @pytest.mark.parametrize("split_type", ["linear", "mlp"])
    def test_partial_reinitialization(self, split_type):
        """Test that split functions can reinitialize from a given index."""
        in_size = (768,)
        nb_splits = 16
        
        split_class = split_dict[split_type]
        split_fn = split_class(
            in_size=in_size,
            nb_splits=nb_splits,
            tree_depth=4,
        )
        
        # Get initial weights
        initial_weights = [
            p.clone() for p in split_fn.parameters()
        ]
        
        # Reinitialize
        split_fn._init_params(start_idx=0)
        
        # Weights should be different after reinitialization
        new_weights = list(split_fn.parameters())
        
        # At least some weights should have changed
        weights_changed = False
        for init_w, new_w in zip(initial_weights, new_weights):
            if not torch.allclose(init_w, new_w):
                weights_changed = True
                break
        
        assert weights_changed
