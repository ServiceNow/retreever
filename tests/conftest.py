"""Pytest configuration and fixtures for retreever tests."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


@pytest.fixture
def device():
    """Pytest fixture for device selection."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Pytest fixture for batch size."""
    return 4


@pytest.fixture
def emb_dim():
    """Pytest fixture for embedding dimension."""
    return 128


@pytest.fixture
def cache_dir():
    """Return the configured cache directory."""
    from retreever import config
    return config.HF_CACHE_DIR


# ==============================================================================
# ENCODER MOCKING FIXTURES  
# ==============================================================================

class MockModel(nn.Module):
    """Mock model for encoder.model."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config._name_or_path = "mock/model"
        
    def forward(self, *args, **kwargs):
        batch_size = kwargs.get('input_ids', kwargs.get('pixel_values', torch.zeros(2))).shape[0]
        return (torch.randn(batch_size, 10, self.config.hidden_size),)


class MockEncoder:
    """Mock encoder that mimics the interface of real encoders."""
    
    def __init__(self, output_size=768, token_level=False):
        self.output_size = output_size
        self.token_level = token_level
        self.model = MockModel(output_size)
        self.prefix = ""
        
        # Mock tokenizer
        self.tokenizer = MagicMock()
        def mock_tokenizer_call(*args, **kwargs):
            batch_size = len(args[0]) if args and isinstance(args[0], (list, tuple)) else 2
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, 10)),
                'attention_mask': torch.ones(batch_size, 10)
            }
        self.tokenizer.side_effect = mock_tokenizer_call
        
        # Mock processor (for image/audio encoders)
        self.processor = MagicMock()
        def mock_processor_call(*args, **kwargs):
            if 'images' in kwargs:
                batch_size = len(kwargs['images'])
            elif len(args) > 0 and isinstance(args[0], (list, tuple)):
                batch_size = len(args[0])
            else:
                batch_size = 2
            return {'pixel_values': torch.randn(batch_size, 3, 224, 224)}
        self.processor.side_effect = mock_processor_call
        
    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        # Determine batch size from inputs
        if 'input_ids' in kwargs:
            batch_size = kwargs['input_ids'].shape[0]
        elif 'pixel_values' in kwargs:
            batch_size = kwargs['pixel_values'].shape[0]
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            batch_size = args[0].shape[0]
        else:
            batch_size = 2
            
        if self.token_level:
            return torch.randn(batch_size, 10, self.output_size)
        return torch.randn(batch_size, self.output_size)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        """Return mock parameters."""
        return iter([torch.nn.Parameter(torch.randn(10, 10))])
    
    def named_parameters(self):
        """Return mock named parameters."""
        return iter([("mock_param", torch.nn.Parameter(torch.randn(10, 10)))])
    
    def to(self, *args, **kwargs):
        """Mock to() method."""
        return self
    
    def eval(self):
        """Mock eval() method."""
        return self
    
    def train(self, mode=True):
        """Mock train() method."""
        return self


@pytest.fixture
def mock_text_encoder():
    """Mock text encoder (DistilBERT/BGE)."""
    return MockEncoder(output_size=768, token_level=False)


@pytest.fixture
def mock_image_encoder():  
    """Mock image encoder (DinoV2/ResNet/CLIP)."""
    return MockEncoder(output_size=768, token_level=False)


@pytest.fixture
def mock_get_encoders(monkeypatch):
    """Mock get_encoders function to return mock encoders for testing.
    
    Usage in tests:
        def test_something(mock_get_encoders):
            # get_encoders is automatically mocked
            model = ReTreever(encoder_type="distilbert", ...)
    """
    def _mock_get_encoders(encoder_type, cache_dir=None, **kwargs):
        query_encoder = MockEncoder(
            output_size=768, 
            token_level=kwargs.get('token_level', False)
        )
        context_encoder = MockEncoder(
            output_size=768,
            token_level=kwargs.get('token_level', False)
        )
        return query_encoder, context_encoder
    
    from retreever.models import encoders
    monkeypatch.setattr(encoders, "get_encoders", _mock_get_encoders)
    return _mock_get_encoders


# ==============================================================================
# DATA FIXTURES
# ==============================================================================

@pytest.fixture
def simple_batch():
    """Simple batch of text data for testing."""
    return {
        'query_input_ids': torch.randint(0, 1000, (4, 32)),
        'query_attention_mask': torch.ones(4, 32),
        'context_input_ids': torch.randint(0, 1000, (4, 32)),
        'context_attention_mask': torch.ones(4, 32),
        'labels': torch.arange(4),
    }


@pytest.fixture
def image_batch():
    """Batch of image data for testing."""
    return {
        'query_pixel_values': torch.randn(4, 3, 224, 224),
        'context_pixel_values': torch.randn(16, 3, 224, 224),
        'labels': torch.arange(4),
    }


# ==============================================================================
# TEST CONFIGURATION
# ==============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in all tests."""
    torch.manual_seed(42)
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    yield


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_download: marks tests that download models")

