"""Unit tests for encoders."""

import pytest
import torch
from retreever.models.encoders import get_encoders


class TestEncoders:
    """Test encoder functionality."""
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "dinov2-large",
        "resnet50",
        "clip-vit-large-patch14",
        "ast",
    ])
    def test_encoder_initialization(self, encoder_type):
        """Test that encoders initialize correctly."""
        query_encoder, context_encoder = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            dual_model=False,
        )
        assert query_encoder is not None
        assert context_encoder is not None
    
    def test_text_encoder_forward(self):
        """Test text encoder forward pass."""
        query_encoder, context_encoder = get_encoders(
            encoder_type="distilbert",
            cache_dir=None,
        )
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        output = query_encoder(input_ids, attention_mask=attention_mask)
        
        assert output.shape[0] == batch_size
        assert len(output.shape) in [2, 3]  # (batch, dim) or (batch, seq, dim)
    
    def test_image_encoder_forward(self):
        """Test image encoder forward pass."""
        query_encoder, context_encoder = get_encoders(
            encoder_type="resnet50",
            cache_dir=None,
        )
        
        # Create dummy image
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        inputs = {"pixel_values": dummy_image}
        
        # Forward pass
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
    
    def test_dual_model(self):
        """Test that dual model creates separate encoders."""
        query_encoder, context_encoder = get_encoders(
            encoder_type="distilbert",
            cache_dir=None,
            dual_model=True,
        )
        
        # Check that they are different objects
        assert query_encoder is not context_encoder


class TestEncoderDimensions:
    """Test encoder output dimensions."""
    
    @pytest.mark.parametrize("encoder_type,expected_dim", [
        ("distilbert", 768),
        ("dinov2-large", 1024),
        ("resnet50", 2048),
    ])
    def test_encoder_dimensions(self, encoder_type, expected_dim):
        """Test that encoders output expected dimensions."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
        )
        
        # Check embedding dimension
        assert query_encoder.emb_size == expected_dim
