"""Comprehensive tests for encoder finetuning strategies."""

import pytest
import torch
from retreever.models.retreever import ReTreever
from omegaconf import OmegaConf


class TestFinetuneStrategyInstantiation:
    """Test that all finetuning strategies can be instantiated."""
    
    @pytest.mark.parametrize("strategy", [
        "last_layer",
        "linear",
        "linear_zero_init",
        "mlp",
        "mlp_no_residual",
        "mlp_zero_init",
        "shared_mlp_zero_init",
        "shared_mlp_zero_init_norm",
        "shared_linear_zero_init_norm",
        "pre_norm_mlp",
        "mrl",
        "bottleneck",
        "adapter",
        "bitfit",
        "layernorm",
    ])
    def test_strategy_instantiation(self, strategy):
        """Test that model can be created with each finetuning strategy."""
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
        assert model.query_encoder is not None


class TestLastLayerFinetuning:
    """Test last_layer finetuning strategy."""
    
    def test_last_layer_unfreezes_final_layer(self):
        """Test that last_layer strategy only unfreezes the final layer."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="last_layer",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        # Most encoder parameters should be frozen
        frozen_count = 0
        trainable_count = 0
        
        for param in model.query_encoder.parameters():
            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1
        
        # Should have some frozen and some trainable
        assert frozen_count > 0
        assert trainable_count > 0
        # Most should be frozen
        assert frozen_count > trainable_count


class TestAdapterStrategies:
    """Test adapter-based finetuning strategies."""
    
    @pytest.mark.parametrize("strategy", [
        "linear",
        "linear_zero_init",
        "mlp",
        "mlp_no_residual",
        "mlp_zero_init",
    ])
    def test_adapter_creates_projection(self, strategy):
        """Test that adapter strategies create projection layers."""
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
        
        # Should have projection layers
        assert hasattr(model, 'query_projection')
        assert hasattr(model, 'context_projection')
        assert model.query_projection is not None
        assert model.context_projection is not None
    
    @pytest.mark.parametrize("strategy", [
        "linear",
        "mlp",
    ])
    def test_adapter_forward_pass(self, strategy):
        """Test forward pass with adapter strategies."""
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
        
        # Create dummy text inputs
        batch_size = 2
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        query_routes, query_leaves = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        assert query_routes is not None
        assert query_leaves is not None


class TestSharedAdapterStrategies:
    """Test shared adapter strategies."""
    
    @pytest.mark.parametrize("strategy", [
        "shared_mlp_zero_init",
        "shared_mlp_zero_init_norm",
        "shared_linear_zero_init_norm",
    ])
    def test_shared_adapter_same_instance(self, strategy):
        """Test that shared strategies use same adapter instance for query and context."""
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
        
        # Query and context projection should be same object
        assert model.query_projection is model.context_projection
    
    def test_shared_adapter_parameter_tying(self):
        """Test that shared adapter reduces parameter count."""
        model_shared = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="shared_mlp_zero_init",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        model_separate = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="mlp_zero_init",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        # Count parameters in projections
        shared_params = sum(p.numel() for p in model_shared.query_projection.parameters())
        separate_params = sum(
            p.numel() for p in model_separate.query_projection.parameters()
        ) + sum(
            p.numel() for p in model_separate.context_projection.parameters()
        )
        
        # Shared should have fewer parameters
        assert shared_params < separate_params


class TestSpecializedAdapters:
    """Test specialized adapter strategies."""
    
    def test_mrl_adapter(self):
        """Test MRL adapter strategy."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="mrl",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert hasattr(model, 'query_projection')
        # MRL adapter should be shared
        assert model.query_projection is model.context_projection
    
    def test_bottleneck_adapter(self):
        """Test bottleneck adapter strategy."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="bottleneck",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert hasattr(model, 'query_projection')
        assert model.query_projection is not None
    
    def test_adapter_with_bottleneck_dim(self):
        """Test adapter strategy with custom bottleneck dimension."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="adapter",
            adapter_bottleneck_dim=128,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert hasattr(model, 'query_projection')
        # Bottleneck adapters should have reduced dimensionality
        assert model.query_projection is not None


class TestParameterEfficientStrategies:
    """Test parameter-efficient finetuning strategies (BitFit, LayerNorm)."""
    
    def test_bitfit_only_bias_trainable(self):
        """Test that BitFit only makes bias parameters trainable."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="bitfit",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        # Check encoder parameters
        trainable_params = []
        for name, param in model.query_encoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        # All trainable params should have 'bias' in name
        assert all('bias' in name.lower() for name in trainable_params)
        assert len(trainable_params) > 0
    
    def test_layernorm_only_ln_trainable(self):
        """Test that LayerNorm strategy only makes LayerNorm parameters trainable."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="layernorm",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        # Check encoder parameters
        trainable_params = []
        for name, param in model.query_encoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        # All trainable params should have 'LayerNorm' or 'layer_norm' in name
        assert all(
            'layernorm' in name.lower() or 'layer_norm' in name.lower()
            for name in trainable_params
        )
        assert len(trainable_params) > 0


class TestFinetuneStrategyWithDifferentEncoders:
    """Test finetuning strategies work with different encoder types."""
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "bge",
        "resnet50",
        "dinov2-large",
    ])
    def test_strategy_with_different_encoders(self, encoder_type):
        """Test that finetuning works with text and image encoders."""
        model = ReTreever(
            loss=None,
            encoder_type=encoder_type,
            freeze_encoder=False,
            encoder_finetune_strategy="mlp",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        
        assert model is not None
        assert hasattr(model, 'query_projection')


class TestFreezeEncoderWithoutStrategy:
    """Test that freeze_encoder=False without strategy raises error."""
    
    def test_no_strategy_raises_error(self):
        """Test that missing strategy with freeze_encoder=False raises error."""
        with pytest.raises(ValueError, match="no encoder_finetune_strategy specified"):
            model = ReTreever(
                loss=None,
                encoder_type="distilbert",
                freeze_encoder=False,
                encoder_finetune_strategy="none",
                tree_type="qr_tree",
                tree_depth=4,
                tree_split_fn="linear",
                dual_model=False,
                cache_dir=None,
            )


class TestFinetuneStrategyGradientFlow:
    """Test gradient flow through finetuning strategies."""
    
    @pytest.mark.parametrize("strategy", [
        "linear",
        "mlp",
        "shared_mlp_zero_init",
    ])
    def test_adapter_gradient_flow(self, strategy):
        """Test that gradients flow through adapter layers."""
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
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 32
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        query_routes, query_leaves = model(
            query_ids,
            query_mask,
            context_ids,
            context_mask,
        )
        
        # Compute loss and backward
        loss = query_routes.sum()
        loss.backward()
        
        # Check adapter has gradients
        has_gradients = False
        for param in model.query_projection.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients


class TestDualModelWithFinetuning:
    """Test finetuning strategies with dual model setup."""
    
    def test_dual_model_separate_adapters(self):
        """Test that dual model creates separate adapters."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="mlp",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=True,
            cache_dir=None,
        )
        
        # Encoders should be different
        assert model.query_encoder is not model.context_encoder
        
        # Adapters should also be different (not shared in non-shared strategy)
        assert model.query_projection is not model.context_projection
    
    def test_dual_model_shared_adapter(self):
        """Test that dual model with shared strategy shares adapters."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy="shared_mlp_zero_init",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=True,
            cache_dir=None,
        )
        
        # Adapters should be shared even with dual encoders
        assert model.query_projection is model.context_projection
