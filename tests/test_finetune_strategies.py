"""Tests for supported encoder finetuning strategies.

Supported strategies:
- shared_mlp_zero_init_norm
- shared_linear_zero_init_norm
- mrl
"""

import pytest
import torch
from retreever.models.retreever import ReTreever
from retreever.models.adapters import get_adapter, ADAPTER_REGISTRY


SUPPORTED_STRATEGIES = [
    "shared_mlp_zero_init_norm",
    "shared_linear_zero_init_norm",
    "mrl",
]


class TestFinetuneStrategyInstantiation:
    """Test that all supported finetuning strategies can be instantiated."""

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_strategy_instantiation(self, strategy):
        """Test that model can be created with each supported finetuning strategy."""
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

    def test_unsupported_strategy_raises_error(self):
        """Test that unsupported strategies raise a ValueError."""
        with pytest.raises(ValueError, match="Unknown encoder_finetune_strategy"):
            ReTreever(
                loss=None,
                encoder_type="distilbert",
                freeze_encoder=False,
                encoder_finetune_strategy="some_old_unsupported_strategy",
                tree_type="qr_tree",
                tree_depth=4,
                tree_split_fn="linear",
                dual_model=False,
                cache_dir=None,
            )

    def test_none_strategy_raises_error(self):
        """Test that 'none' strategy with freeze_encoder=False raises error."""
        with pytest.raises(ValueError, match="no encoder_finetune_strategy specified"):
            ReTreever(
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


class TestSharedAdapterStrategies:
    """Test that all supported strategies produce shared adapters."""

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_shared_adapter_same_instance(self, strategy):
        """Test that all supported strategies share the adapter between query and context."""
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
        assert model.query_projection is model.context_projection

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_adapter_creates_projection(self, strategy):
        """Test that all strategies create projection layers."""
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
        assert hasattr(model, 'query_projection')
        assert hasattr(model, 'context_projection')
        assert model.query_projection is not None
        assert model.context_projection is not None

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_encoder_frozen(self, strategy):
        """Test that encoder parameters are frozen when using adapter strategies."""
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
        for param in model.query_encoder.parameters():
            assert not param.requires_grad

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_adapter_has_trainable_params(self, strategy):
        """Test that adapter has trainable parameters."""
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
        trainable_adapter_params = sum(
            p.numel() for p in model.query_projection.parameters()
            if p.requires_grad
        )
        assert trainable_adapter_params > 0


class TestAdapterModule:
    """Test the adapters.py module directly."""

    def test_adapter_registry_has_supported_strategies(self):
        """Test that ADAPTER_REGISTRY contains all supported strategies."""
        for strategy in SUPPORTED_STRATEGIES:
            assert strategy in ADAPTER_REGISTRY

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_get_adapter_factory(self, strategy):
        """Test that get_adapter creates a valid adapter."""
        adapter = get_adapter(strategy=strategy, input_dim=768, output_dim=768)
        assert adapter is not None
        assert isinstance(adapter, torch.nn.Module)

    def test_get_adapter_invalid_strategy_raises(self):
        """Test that get_adapter raises ValueError for invalid strategy."""
        with pytest.raises(ValueError):
            get_adapter(strategy="invalid_strategy", input_dim=768, output_dim=768)

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_adapter_forward_pass(self, strategy):
        """Test that adapter can do a forward pass."""
        adapter = get_adapter(strategy=strategy, input_dim=768, output_dim=768)
        x = torch.randn(4, 768)
        out = adapter(x)
        assert out is not None
        assert out.shape[0] == 4  # Batch size preserved


class TestFreezeEncoderTrue:
    """Test behavior with freeze_encoder=True (no adapter needed)."""

    def test_freeze_encoder_no_strategy_needed(self):
        """Test that freeze_encoder=True works without an adapter strategy."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=True,
            encoder_finetune_strategy="none",
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=False,
            cache_dir=None,
        )
        assert model is not None
        for param in model.query_encoder.parameters():
            assert not param.requires_grad
        assert model.query_projection is None
        assert model.context_projection is None


class TestFinetuneStrategyGradientFlow:
    """Test gradient flow through supported finetuning strategies."""

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
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

        batch_size = 2
        seq_len = 16
        query_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_mask = torch.ones(batch_size, seq_len)
        context_ids = torch.randint(0, 1000, (batch_size, seq_len))
        context_mask = torch.ones(batch_size, seq_len)

        query_routes, query_leaves = model(
            query_ids, query_mask, context_ids, context_mask,
        )
        loss = query_routes.sum()
        loss.backward()

        has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.query_projection.parameters()
        )
        assert has_gradients


class TestDualModelWithFinetuning:
    """Test finetuning strategies with dual model setup."""

    @pytest.mark.parametrize("strategy", SUPPORTED_STRATEGIES)
    def test_dual_model_shared_adapter(self, strategy):
        """Test that dual model with shared strategy shares adapters."""
        model = ReTreever(
            loss=None,
            encoder_type="distilbert",
            freeze_encoder=False,
            encoder_finetune_strategy=strategy,
            tree_type="qr_tree",
            tree_depth=4,
            tree_split_fn="linear",
            dual_model=True,
            cache_dir=None,
        )
        # Adapters should be shared even with dual encoders
        assert model.query_projection is model.context_projection
        # Trees should be separate in dual mode
        assert model.query_tree is not model.context_tree
