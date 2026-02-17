"""Comprehensive tests for indexing strategies."""

import pytest
import torch
import numpy as np
from retreever.models.indexing_strategies import (
    GreedyIndexing,
    TreeRepAnnoyIndexing,
    TreeRepFaissIndexing,
    index_strategy_dict,
)


class TestIndexingStrategyTypes:
    """Test all indexing strategy types."""
    
    @pytest.mark.parametrize("strategy_name", [
        "greedy",
        "tree_rep",
        "faiss_tree_rep",
    ])
    def test_strategy_in_dict(self, strategy_name):
        """Test that all strategies are registered."""
        assert strategy_name in index_strategy_dict
        assert callable(index_strategy_dict[strategy_name])


class TestGreedyIndexing:
    """Test greedy indexing strategy."""
    
    def test_greedy_instantiation(self):
        """Test greedy indexing instantiation."""
        strategy = GreedyIndexing(
            tree_depth=4,
            emb_dim=768,
        )
        
        assert strategy is not None
    
    def test_greedy_add_embeddings(self):
        """Test adding embeddings to greedy index."""
        strategy = GreedyIndexing(
            tree_depth=4,
            emb_dim=768,
        )
        
        num_samples = 10
        routes = torch.randn(num_samples, 2 ** 4)
        routes = torch.softmax(routes, dim=1)  # Make it a valid distribution
        embeddings = torch.randn(num_samples, 768)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Check that data was added
        assert strategy.ntotal() > 0
    
    def test_greedy_search(self):
        """Test greedy search functionality."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add some data
        num_samples = 20
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Search
        query_routes = torch.randn(2, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(2, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=5)
        
        assert distances.shape == (2, 5)
        assert indices.shape == (2, 5)


class TestTreeRepAnnoyIndexing:
    """Test Annoy-based tree representation indexing."""
    
    def test_annoy_instantiation(self):
        """Test Annoy indexing instantiation."""
        strategy = TreeRepAnnoyIndexing(
            tree_depth=4,
            emb_dim=768,
        )
        
        assert strategy is not None
    
    def test_annoy_build_and_search(self):
        """Test Annoy index building and search."""
        tree_depth = 4
        emb_dim = 768
        strategy = TreeRepAnnoyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add data
        num_samples = 50
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Build index
        strategy.build(n_trees=10)
        
        # Search
        query_routes = torch.randn(2, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(2, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=10)
        
        assert distances.shape == (2, 10)
        assert indices.shape == (2, 10)


class TestTreeRepFaissIndexing:
    """Test FAISS-based tree representation indexing."""
    
    def test_faiss_instantiation(self):
        """Test FAISS indexing instantiation."""
        strategy = TreeRepFaissIndexing(
            tree_depth=4,
            emb_dim=768,
        )
        
        assert strategy is not None
    
    def test_faiss_add_and_search(self):
        """Test FAISS index add and search."""
        tree_depth = 4
        emb_dim = 768
        strategy = TreeRepFaissIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add data
        num_samples = 50
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        assert strategy.ntotal() == num_samples
        
        # Search
        query_routes = torch.randn(2, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(2, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=10)
        
        assert distances.shape == (2, 10)
        assert indices.shape == (2, 10)
    
    def test_faiss_different_metrics(self):
        """Test FAISS with different distance metrics."""
        for metric in ["cosine", "l2"]:
            strategy = TreeRepFaissIndexing(
                tree_depth=4,
                emb_dim=768,
                metric=metric,
            )
            
            assert strategy is not None


class TestIndexingWithDifferentDepths:
    """Test indexing strategies with various tree depths."""
    
    @pytest.mark.parametrize("depth", [2, 4, 6, 8])
    def test_greedy_different_depths(self, depth):
        """Test greedy indexing with different tree depths."""
        strategy = GreedyIndexing(
            tree_depth=depth,
            emb_dim=768,
        )
        
        num_samples = 10
        routes = torch.randn(num_samples, 2 ** depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, 768)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        assert strategy.ntotal() > 0


class TestIndexingBatchProcessing:
    """Test indexing strategies with batch processing."""
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_batch_search(self, batch_size):
        """Test search with different batch sizes."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add data
        num_samples = 50
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Search with batch
        query_routes = torch.randn(batch_size, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(batch_size, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=5)
        
        assert distances.shape == (batch_size, 5)
        assert indices.shape == (batch_size, 5)


class TestIndexingTopK:
    """Test indexing with different top-k values."""
    
    @pytest.mark.parametrize("k", [1, 5, 10, 20])
    def test_different_k_values(self, k):
        """Test retrieval with different k values."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add data
        num_samples = 50
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Search
        query_routes = torch.randn(2, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(2, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=k)
        
        assert distances.shape == (2, k)
        assert indices.shape == (2, k)


class TestIndexingScaling:
    """Test indexing scalability."""
    
    def test_large_corpus(self):
        """Test indexing with larger corpus."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add larger dataset
        num_samples = 500
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        assert strategy.ntotal() == num_samples
        
        # Search should still work
        query_routes = torch.randn(1, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(1, emb_dim)
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=10)
        
        assert distances.shape == (1, 10)


class TestIndexingIncremental:
    """Test incremental index building."""
    
    def test_incremental_add(self):
        """Test adding data incrementally."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add in batches
        batch_sizes = [10, 15, 20]
        total_added = 0
        
        for batch_size in batch_sizes:
            routes = torch.randn(batch_size, 2 ** tree_depth)
            routes = torch.softmax(routes, dim=1)
            embeddings = torch.randn(batch_size, emb_dim)
            
            strategy.add(routes, embeddings, list(range(total_added, total_added + batch_size)))
            total_added += batch_size
        
        assert strategy.ntotal() == sum(batch_sizes)


class TestIndexingDict:
    """Test index_strategy_dict registry."""
    
    def test_strategy_dict_completeness(self):
        """Test that index_strategy_dict contains expected strategies."""
        expected_strategies = [
            "greedy",
            "tree_rep",
            "faiss_tree_rep",
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in index_strategy_dict
    
    def test_strategy_classes_match(self):
        """Test that strategy names map to correct classes."""
        assert index_strategy_dict["greedy"] == GreedyIndexing
        assert index_strategy_dict["tree_rep"] == TreeRepAnnoyIndexing
        assert index_strategy_dict["faiss_tree_rep"] == TreeRepFaissIndexing


class TestIndexingEdgeCases:
    """Test edge cases for indexing."""
    
    def test_search_before_add(self):
        """Test searching on empty index."""
        strategy = GreedyIndexing(
            tree_depth=4,
            emb_dim=768,
        )
        
        # Try to search without adding data
        query_routes = torch.randn(1, 2 ** 4)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(1, 768)
        
        # Should handle gracefully or raise appropriate error
        try:
            distances, indices = strategy.search(query_routes, query_embeddings, k=5)
            # If it succeeds, check output shape
            assert distances.shape[0] == 1
        except (RuntimeError, ValueError):
            # Expected for empty index
            pass
    
    def test_k_larger_than_corpus(self):
        """Test when k is larger than corpus size."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add small corpus
        num_samples = 5
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Search with k > corpus size
        query_routes = torch.randn(1, 2 ** tree_depth)
        query_routes = torch.softmax(query_routes, dim=1)
        query_embeddings = torch.randn(1, emb_dim)
        
        # Should handle gracefully
        distances, indices = strategy.search(query_routes, query_embeddings, k=10)
        
        # Might return fewer than k results
        assert distances.shape[0] == 1


class TestIndexingAccuracy:
    """Test indexing accuracy."""
    
    def test_exact_match_retrieval(self):
        """Test that exact matches are retrieved."""
        tree_depth = 4
        emb_dim = 768
        strategy = GreedyIndexing(
            tree_depth=tree_depth,
            emb_dim=emb_dim,
        )
        
        # Add data
        num_samples = 20
        routes = torch.randn(num_samples, 2 ** tree_depth)
        routes = torch.softmax(routes, dim=1)
        embeddings = torch.randn(num_samples, emb_dim)
        
        strategy.add(routes, embeddings, list(range(num_samples)))
        
        # Search with one of the added items
        query_routes = routes[0:1]
        query_embeddings = embeddings[0:1]
        
        distances, indices = strategy.search(query_routes, query_embeddings, k=1)
        
        # Should retrieve itself (or very close)
        assert distances[0, 0] < 1e-3 or indices[0, 0] == 0
