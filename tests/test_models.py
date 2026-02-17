"""Comprehensive tests for ReTreever and MRL models."""

import pytest
import torch
from retreever.models.retreever import ReTreever
from retreever.models.mrl import MRL


class TestReTreeverInstantiation:
    """Test ReTreever model instantiation."""
    
    def test_basic_instantiation(self):
        """Test basic ReTreever instantiation."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None
        assert hasattr(model, 'query_encoder')
        assert hasattr(model, 'context_encoder')
        assert hasattr(model, 'query_tree')
        assert hasattr(model, 'context_tree')
    
    @pytest.mark.parametrize("tree_type", [
        "qr_tree",
        "probabilistic_tree",
        "no_propagation_tree",
    ])
    def test_different_tree_types(self, tree_type):
        """Test ReTreever with different tree types."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type=tree_type,
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "bge",
        "resnet50",
        "dinov2-large",
        "clip-vit-large-patch14",
    ])
    def test_different_encoders(self, encoder_type):
        """Test ReTreever with different encoder types."""
        model = ReTreever(
            loss=None,
            encoder_type=encoder_type,
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None


class TestReTreeverForward:
    """Test ReTreever forward pass."""
    
    def test_text_forward(self):
        """Test forward pass with text inputs."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        batch_size = 2
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        query_routes, query_leaves = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        assert query_routes is not None
        assert query_leaves is not None
        assert query_routes.shape[0] == batch_size
    
    def test_image_forward(self):
        """Test forward pass with image inputs."""
        model = ReTreever(
            loss=None,
            encoder_type="resnet50",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        batch_size = 2
        query_images = {"pixel_values": torch.randn(batch_size, 3, 224, 224)}
        context_images = {"pixel_values": torch.randn(batch_size, 3, 224, 224)}
        
        query_routes, query_leaves = model(
            **query_images,
            **{"context_" + k: v for k, v in context_images.items()},
        )
        
        assert query_routes is not None
        assert query_leaves is not None


class TestReTreeverConfiguration:
    """Test ReTreever configuration options."""
    
    def test_dual_model(self):
        """Test dual model configuration."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=True,
            cache_dir=None,
        )
        
        # Encoders should be different
        assert model.query_encoder is not model.context_encoder
    
    @pytest.mark.parametrize("depth", [2, 4, 6, 8, 10])
    def test_different_depths(self, depth):
        """Test ReTreever with different tree depths."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=depth,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model.query_tree.bst.depth == depth
        assert model.query_tree.num_leaves == 2 ** depth
    
    @pytest.mark.parametrize("split_fn", ["linear", "mlp", "cross_attn"])
    def test_different_split_functions(self, split_fn):
        """Test ReTreever with different split functions."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn=split_fn,
            dual_model=False,
            cache_dir=None,
        )
        
        assert model.query_tree.split_type == split_fn


class TestReTreeverWithFinetuning:
    """Test ReTreever with encoder finetuning."""
    
    @pytest.mark.parametrize("strategy", [
        "last_layer",
        "linear",
        "mlp",
        "shared_mlp_zero_init",
    ])
    def test_with_finetune_strategy(self, strategy):
        """Test ReTreever with finetuning strategies."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy=strategy,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None


class TestMRLModel:
    """Test MRL (Matryoshka Representation Learning) model."""
    
    def test_mrl_instantiation(self):
        """Test MRL model instantiation."""
        model = MRL(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            cache_dir=None,
        )
        
        assert model is not None
        assert hasattr(model, 'query_encoder')
        assert hasattr(model, 'context_encoder')
    
    def test_mrl_forward(self):
        """Test MRL forward pass."""
        model = MRL(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            cache_dir=None,
        )
        
        batch_size = 2
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        output = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        assert output is not None
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "bge",
        "resnet50",
    ])
    def test_mrl_different_encoders(self, encoder_type):
        """Test MRL with different encoder types."""
        model = MRL(
            loss=None,
            encoder_type=encoder_type,
            freeze_encoder=True,
            cache_dir=None,
        )
        
        assert model is not None


class TestModelGradients:
    """Test gradient flow through models."""
    
    def test_retreever_gradients(self):
        """Test gradient flow through ReTreever."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="mlp",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        batch_size = 2
        seq_len = 16
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        query_routes, _ = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        loss = query_routes.sum()
        loss.backward()
        
        # Check tree has gradients
        has_gradients = False
        for param in model.query_tree.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients


class TestModelEvaluation:
    """Test model evaluation mode."""
    
    def test_retreever_eval_mode(self):
        """Test ReTreever in evaluation mode."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        model.eval()
        
        batch_size = 2
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        # Should be deterministic in eval mode
        with torch.no_grad():
            routes1, _ = model(query_ids, query_mask, context_ids, context_mask)
            routes2, _ = model(query_ids, query_mask, context_ids, context_mask)
        
        assert torch.equal(routes1, routes2)


class TestModelBatching:
    """Test model with different batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_retreever_batch_sizes(self, batch_size):
        """Test ReTreever with various batch sizes."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        query_routes, query_leaves = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        assert query_routes.shape[0] == batch_size
        assert query_leaves.shape[0] == batch_size


class TestModelComponents:
    """Test individual model components."""
    
    def test_encoder_component(self):
        """Test that encoders are properly initialized."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model.query_encoder.emb_size > 0
        assert model.context_encoder.emb_size > 0
    
    def test_tree_component(self):
        """Test that trees are properly initialized."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model.query_tree.num_leaves == 2 ** 4
        assert model.context_tree.num_leaves == 2 ** 4


class TestModelParameterCount:
    """Test model parameter counts."""
    
    def test_parameter_sharing(self):
        """Test parameter sharing in single model mode."""
        model_shared = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        model_dual = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=True,
            cache_dir=None,
        )
        
        # Count total parameters
        params_shared = sum(p.numel() for p in model_shared.parameters())
        params_dual = sum(p.numel() for p in model_dual.parameters())
        
        # Dual model should have more parameters (separate encoders)
        assert params_dual > params_shared


class TestModelSerialization:
    """Test model save/load capability."""
    
    def test_state_dict(self):
        """Test that model state_dict can be extracted."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        state_dict = model.state_dict()
        
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0


class TestModelArchitectureVariations:
    """Test various architectural configurations."""
    
    def test_token_level_encoder(self):
        """Test model with token-level encoder outputs."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            encoder_token_level=True,
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None
    
    def test_encoder_normalization(self):
        """Test model with encoder normalization."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            encoder_normalize=True,
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None
