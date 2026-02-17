"""Pytest configuration and fixtures."""

import pytest
import torch


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


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
