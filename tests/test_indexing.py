"""Tests for supported indexing strategies.

Supported indexing strategies:
- faiss_tree_rep (TreeRepFaissIndexing)
- tree_rep_multi_index_faiss (TreeRepMultiIndexFaissIndexing)
"""

import pytest
import torch
import numpy as np
from retreever.models.indexing_strategies import (
    TreeRepFaissIndexing,
    TreeRepMultiIndexFaissIndexing,
    index_strategy_dict,
)

NUM_DIMENSIONS = 16  # Simulates 2^4 leaves


class TestIndexingStrategyDict:
    """Test the indexing strategy registry."""

    def test_only_faiss_strategies_in_dict(self):
        """Test that only FAISS strategies are registered."""
        assert set(index_strategy_dict.keys()) == {"faiss_tree_rep", "tree_rep_multi_index_faiss"}

    def test_strategy_classes_match(self):
        """Test that strategy names map to correct classes."""
        assert index_strategy_dict["faiss_tree_rep"] is TreeRepFaissIndexing
        assert index_strategy_dict["tree_rep_multi_index_faiss"] is TreeRepMultiIndexFaissIndexing

    def test_removed_strategies_absent(self):
        """Test that removed strategies are NOT in the dict."""
        assert "greedy" not in index_strategy_dict
        assert "tree_rep" not in index_strategy_dict


class TestTreeRepFaissIndexing:
    """Test FAISS-based tree representation indexing."""

    def test_faiss_instantiation(self):
        """Test FAISS indexing instantiation."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        assert strategy is not None

    def test_faiss_is_empty_initially(self):
        """Test that index is empty after creation."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()
        assert strategy.is_empty()

    def test_faiss_index_ctxs(self):
        """Test indexing contexts."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 20
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)

        strategy.index_ctxs(context_assignments, context_names)
        assert not strategy.is_empty()

    def test_faiss_top_contexts(self):
        """Test retrieving top-k contexts."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 50
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)
        strategy.build_index()

        num_queries = 3
        question_assignments = torch.softmax(torch.randn(num_queries, NUM_DIMENSIONS), dim=1)
        topk = strategy.top_contexts(question_assignments, k=5)

        assert len(topk) == num_queries
        for result in topk:
            assert len(result) <= 5

    def test_faiss_reset_index(self):
        """Test resetting the index."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 10
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)
        
        assert not strategy.is_empty()
        strategy.reset_index()
        assert strategy.is_empty()

    def test_faiss_size(self):
        """Test size tracking."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 15
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)

        assert strategy.size() == num_samples

    @pytest.mark.parametrize("k", [1, 5, 10])
    def test_faiss_top_k_values(self, k):
        """Test retrieval with different k values."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 50
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)
        strategy.build_index()

        question_assignments = torch.softmax(torch.randn(2, NUM_DIMENSIONS), dim=1)
        topk = strategy.top_contexts(question_assignments, k=k)

        assert len(topk) == 2
        for result in topk:
            assert len(result) <= k


class TestTreeRepMultiIndexFaissIndexing:
    """Test FAISS multi-index strategy."""

    def test_multi_index_instantiation(self):
        """Test FAISS multi-index instantiation."""
        strategy = TreeRepMultiIndexFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        assert strategy is not None

    def test_multi_index_is_empty_initially(self):
        """Test that multi-index is empty after creation."""
        strategy = TreeRepMultiIndexFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()
        assert strategy.is_empty()

    def test_multi_index_ctxs(self):
        """Test indexing contexts."""
        strategy = TreeRepMultiIndexFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 20
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)

        assert not strategy.is_empty()

    def test_multi_index_top_contexts(self):
        """Test retrieving top-k contexts from multi-index."""
        strategy = TreeRepMultiIndexFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 50
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)
        strategy.build_index()

        num_queries = 3
        question_assignments = torch.softmax(torch.randn(num_queries, NUM_DIMENSIONS), dim=1)
        topk = strategy.top_contexts(question_assignments, k=5)

        assert len(topk) == num_queries

    def test_multi_index_reset(self):
        """Test resetting the multi-index."""
        strategy = TreeRepMultiIndexFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        num_samples = 10
        context_assignments = torch.softmax(torch.randn(num_samples, NUM_DIMENSIONS), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)

        assert not strategy.is_empty()
        strategy.reset_index()
        assert strategy.is_empty()


class TestIndexingIncrementalAdd:
    """Test incremental addition to indices."""

    def test_faiss_incremental_add(self):
        """Test adding data incrementally to FAISS index."""
        strategy = TreeRepFaissIndexing(num_dimensions=NUM_DIMENSIONS)
        strategy.reset_index()

        batch_sizes = [10, 15, 20]
        total = 0
        for batch_size in batch_sizes:
            context_assignments = torch.softmax(torch.randn(batch_size, NUM_DIMENSIONS), dim=1)
            context_names = torch.arange(total, total + batch_size)
            strategy.index_ctxs(context_assignments, context_names)
            total += batch_size

        assert strategy.size() == sum(batch_sizes)


class TestIndexingWithVariedDimensions:
    """Test indexing strategies with different sizes."""

    @pytest.mark.parametrize("num_dimensions", [8, 16, 32, 64])
    def test_faiss_different_dimensions(self, num_dimensions):
        """Test FAISS indexing with different number of dimensions (leaf counts)."""
        strategy = TreeRepFaissIndexing(num_dimensions=num_dimensions)
        strategy.reset_index()

        num_samples = 20
        context_assignments = torch.softmax(torch.randn(num_samples, num_dimensions), dim=1)
        context_names = torch.arange(num_samples)
        strategy.index_ctxs(context_assignments, context_names)
        strategy.build_index()

        question_assignments = torch.softmax(torch.randn(2, num_dimensions), dim=1)
        topk = strategy.top_contexts(question_assignments, k=5)

        assert len(topk) == 2
