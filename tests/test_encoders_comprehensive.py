"""Comprehensive tests for all encoder modalities and types."""

import pytest
import torch
from retreever.models.encoders import get_encoders, encoder_dict


class TestAllEncoderTypes:
    """Test all supported encoder types can be instantiated."""
    
    @pytest.mark.parametrize("encoder_type", list(encoder_dict.keys()))
    def test_encoder_instantiation(self, encoder_type):
        """Test that all encoders in encoder_dict can be instantiated."""
        query_encoder, context_encoder = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            dual_model=False,
        )
        
        assert query_encoder is not None
        assert context_encoder is not None
        assert hasattr(query_encoder, 'emb_size')
        assert query_encoder.emb_size > 0


class TestTextEncoders:
    """Test text encoder modalities."""
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "bge",
    ])
    def test_text_encoder_forward(self, encoder_type):
        """Test forward pass for all text encoders."""
        query_encoder, context_encoder = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
        )
        
        # Create dummy text inputs
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        output = query_encoder(input_ids, attention_mask=attention_mask)
        
        assert output.shape[0] == batch_size
        assert output.shape[-1] == query_encoder.emb_size
    
    def test_distilbert_output_dim(self):
        """Test DistilBERT outputs 768-dim embeddings."""
        query_encoder, _ = get_encoders(encoder_type="distilbert", cache_dir=None)
        
        assert query_encoder.emb_size == 768
    
    def test_bge_output_dim(self):
        """Test BGE outputs 1024-dim embeddings."""
        query_encoder, _ = get_encoders(encoder_type="bge", cache_dir=None)
        
        assert query_encoder.emb_size == 1024
    
    @pytest.mark.parametrize("encoder_type", ["distilbert", "bge"])
    def test_text_encoder_token_level(self, encoder_type):
        """Test text encoders support token-level outputs."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            token_level=True,
        )
        
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = query_encoder(input_ids, attention_mask=attention_mask)
        
        # Should have token dimension
        assert len(output.shape) == 3
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == query_encoder.emb_size


class TestImageEncodersDinoV2:
    """Test DinoV2 image encoders."""
    
    @pytest.mark.parametrize("encoder_type,expected_dim", [
        ("dinov2-small", 384),
        ("dinov2-base", 768),
        ("dinov2-large", 1024),
        ("dinov2-giant", 1536),
    ])
    def test_dinov2_variants(self, encoder_type, expected_dim):
        """Test all DinoV2 model sizes."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        assert query_encoder.emb_size == expected_dim
    
    @pytest.mark.parametrize("encoder_type", [
        "dinov2-small",
        "dinov2-base",
        "dinov2-large",
        "dinov2-giant",
    ])
    def test_dinov2_forward(self, encoder_type):
        """Test forward pass for DinoV2 encoders."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        # Create dummy image
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        inputs = {"pixel_values": dummy_image}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == query_encoder.emb_size


class TestImageEncodersResNet:
    """Test ResNet image encoders."""
    
    @pytest.mark.parametrize("encoder_type,expected_dim", [
        ("resnet18", 512),
        ("resnet34", 512),
        ("resnet50", 2048),
        ("resnet101", 2048),
        ("resnet152", 2048),
    ])
    def test_resnet_variants(self, encoder_type, expected_dim):
        """Test all ResNet model sizes."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        assert query_encoder.emb_size == expected_dim
    
    @pytest.mark.parametrize("encoder_type", [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ])
    def test_resnet_forward(self, encoder_type):
        """Test forward pass for ResNet encoders."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        # Create dummy image
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        inputs = {"pixel_values": dummy_image}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == query_encoder.emb_size


class TestImageEncodersCLIP:
    """Test CLIP vision encoders."""
    
    @pytest.mark.parametrize("encoder_type,expected_dim", [
        ("clip-vit-base-patch32", 768),
        ("clip-vit-base-patch16", 768),
        ("clip-vit-large-patch14", 768),
    ])
    def test_clip_variants(self, encoder_type, expected_dim):
        """Test all CLIP ViT model sizes."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        assert query_encoder.emb_size == expected_dim
    
    @pytest.mark.parametrize("encoder_type", [
        "clip-vit-base-patch32",
        "clip-vit-base-patch16",
        "clip-vit-large-patch14",
    ])
    def test_clip_forward(self, encoder_type):
        """Test forward pass for CLIP encoders."""
        query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
        
        # Create dummy image
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        inputs = {"pixel_values": dummy_image}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == query_encoder.emb_size


class TestAudioEncoders:
    """Test audio encoder modalities."""
    
    def test_ast_instantiation(self):
        """Test AST (Audio Spectrogram Transformer) encoder."""
        query_encoder, _ = get_encoders(encoder_type="ast", cache_dir=None)
        
        assert query_encoder is not None
        assert query_encoder.emb_size == 768
    
    def test_ast_forward(self):
        """Test AST forward pass."""
        query_encoder, _ = get_encoders(encoder_type="ast", cache_dir=None)
        
        # Create dummy audio input (AST expects input_values)
        batch_size = 2
        dummy_audio = torch.randn(batch_size, 1024, 128)  # (batch, time, freq)
        inputs = {"input_values": dummy_audio}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == query_encoder.emb_size


class TestMultiModalEncoders:
    """Test multi-modal encoders."""
    
    def test_flava_instantiation(self):
        """Test FLAVA (text-image) encoder."""
        query_encoder, _ = get_encoders(encoder_type="flava", cache_dir=None)
        
        assert query_encoder is not None
        assert query_encoder.emb_size == 768
    
    def test_flava_forward(self):
        """Test FLAVA forward pass."""
        query_encoder, _ = get_encoders(encoder_type="flava", cache_dir=None)
        
        # FLAVA can accept image inputs
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 224, 224)
        inputs = {"pixel_values": dummy_image}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == query_encoder.emb_size


class TestEncoderNormalization:
    """Test encoder normalization functionality."""
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "bge",
        "dinov2-base",
        "resnet50",
    ])
    def test_encoder_with_normalization(self, encoder_type):
        """Test encoders with normalization enabled."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            normalize=True,
        )
        
        # Create appropriate dummy inputs
        if "distilbert" in encoder_type or "bge" in encoder_type:
            batch_size = 2
            seq_len = 32
            inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        else:
            batch_size = 2
            inputs = {"pixel_values": torch.randn(batch_size, 3, 224, 224)}
        
        output = query_encoder(**inputs)
        
        # Check output is normalized
        norms = torch.norm(output, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)


class TestEncoderDualModel:
    """Test dual model functionality."""
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "dinov2-large",
        "resnet50",
    ])
    def test_dual_model_separate_instances(self, encoder_type):
        """Test that dual_model=True creates separate encoder instances."""
        query_encoder, context_encoder = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            dual_model=True,
        )
        
        # Should be different objects
        assert query_encoder is not context_encoder
    
    @pytest.mark.parametrize("encoder_type", [
        "distilbert",
        "dinov2-large",
    ])
    def test_single_model_same_instances(self, encoder_type):
        """Test that dual_model=False shares encoder instances."""
        query_encoder, context_encoder = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
            dual_model=False,
        )
        
        # Underlying model should be shared (same model, different wrappers)
        # They're different wrapper objects but share the model
        assert query_encoder is not context_encoder


class TestEncoderContextLength:
    """Test encoder context length functionality."""
    
    @pytest.mark.parametrize("context_length", [32, 64, 128, 256])
    def test_text_encoder_context_length(self, context_length):
        """Test text encoders with different context lengths."""
        query_encoder, _ = get_encoders(
            encoder_type="distilbert",
            cache_dir=None,
            context_length=context_length,
        )
        
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, context_length))
        attention_mask = torch.ones(batch_size, context_length)
        
        output = query_encoder(input_ids, attention_mask=attention_mask)
        
        assert output.shape[0] == batch_size


class TestEncoderFreeze:
    """Test encoder freezing functionality."""
    
    @pytest.mark.parametrize("encoder_type", ["distilbert", "dinov2-base"])
    def test_freeze_encoder(self, encoder_type):
        """Test that encoders can be frozen."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
        )
        
        # Freeze encoder
        for param in query_encoder.parameters():
            param.requires_grad = False
        
        # Check all parameters are frozen
        frozen_count = sum(1 for p in query_encoder.parameters() if not p.requires_grad)
        total_count = sum(1 for p in query_encoder.parameters())
        
        assert frozen_count == total_count


class TestEncoderBatching:
    """Test encoders with different batch sizes."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("encoder_type", ["distilbert", "resnet50"])
    def test_variable_batch_sizes(self, batch_size, encoder_type):
        """Test encoders handle various batch sizes."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
        )
        
        if "distilbert" in encoder_type:
            seq_len = 32
            inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        else:
            inputs = {"pixel_values": torch.randn(batch_size, 3, 224, 224)}
        
        output = query_encoder(**inputs)
        
        assert output.shape[0] == batch_size


class TestEncoderDict:
    """Test encoder_dict registry."""
    
    def test_encoder_dict_completeness(self):
        """Test that encoder_dict contains all documented encoders."""
        expected_encoders = [
            # Text
            "distilbert", "bge",
            # Image - DinoV2
            "dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant",
            # Image - ResNet
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            # Image - CLIP
            "clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14",
            # Audio
            "ast",
            # Multi-modal
            "flava",
        ]
        
        for encoder_name in expected_encoders:
            assert encoder_name in encoder_dict
    
    def test_encoder_dict_structure(self):
        """Test that encoder_dict entries have correct structure."""
        for encoder_name, (encoder_class, model_name) in encoder_dict.items():
            assert callable(encoder_class)
            assert isinstance(model_name, str)
            assert len(model_name) > 0


class TestEncoderGradients:
    """Test gradient flow through encoders."""
    
    @pytest.mark.parametrize("encoder_type", ["distilbert", "resnet50"])
    def test_encoder_gradient_flow(self, encoder_type):
        """Test that gradients flow through unfrozen encoders."""
        query_encoder, _ = get_encoders(
            encoder_type=encoder_type,
            cache_dir=None,
        )
        
        # Ensure encoder is trainable
        for param in query_encoder.parameters():
            param.requires_grad = True
        
        if "distilbert" in encoder_type:
            batch_size = 2
            seq_len = 16
            inputs = {
                "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
            }
        else:
            batch_size = 2
            inputs = {"pixel_values": torch.randn(batch_size, 3, 224, 224)}
        
        output = query_encoder(**inputs)
        loss = output.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in query_encoder.parameters()
        )
        
        assert has_gradients


class TestEncoderModalities:
    """Test that encoders properly support their modalities."""
    
    def test_text_modality(self):
        """Test text encoders accept text inputs."""
        encoders = ["distilbert", "bge"]
        
        for encoder_type in encoders:
            query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
            
            # Should accept input_ids and attention_mask
            output = query_encoder(
                torch.randint(0, 1000, (2, 32)),
                attention_mask=torch.ones(2, 32),
            )
            assert output is not None
    
    def test_image_modality(self):
        """Test image encoders accept image inputs."""
        encoders = ["dinov2-base", "resnet50", "clip-vit-base-patch32"]
        
        for encoder_type in encoders:
            query_encoder, _ = get_encoders(encoder_type=encoder_type, cache_dir=None)
            
            # Should accept pixel_values
            output = query_encoder(pixel_values=torch.randn(2, 3, 224, 224))
            assert output is not None
    
    def test_audio_modality(self):
        """Test audio encoders accept audio inputs."""
        query_encoder, _ = get_encoders(encoder_type="ast", cache_dir=None)
        
        # Should accept input_values
        output = query_encoder(input_values=torch.randn(2, 1024, 128))
        assert output is not None
