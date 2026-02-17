"""Comprehensive tests for depth schedulers."""

import pytest
import torch
import numpy as np
from retreever.training.depth_schedulers import (
    LinearDepthScheduler,
    ExponentialDepthScheduler,
    RandomDepthScheduler,
    RandomHeavyTailedDepthScheduler,
    RandomUniformDepthScheduler,
    KNOWN_SCHEDULERS,
)


class TestDepthSchedulerTypes:
    """Test all depth scheduler types."""
    
    @pytest.mark.parametrize("scheduler_name", list(KNOWN_SCHEDULERS.keys()))
    def test_scheduler_instantiation(self, scheduler_name):
        """Test that all schedulers can be instantiated."""
        scheduler_class = KNOWN_SCHEDULERS[scheduler_name]
        
        if scheduler_name in ["linear", "exponential"]:
            scheduler = scheduler_class(min_value=0, max_value=10, max_steps=1000)
        else:
            scheduler = scheduler_class(min_value=0, max_value=10)
        
        assert scheduler is not None
    
    @pytest.mark.parametrize("scheduler_name", list(KNOWN_SCHEDULERS.keys()))
    def test_scheduler_get_depth(self, scheduler_name):
        """Test that all schedulers can generate depths."""
        scheduler_class = KNOWN_SCHEDULERS[scheduler_name]
        
        if scheduler_name in ["linear", "exponential"]:
            scheduler = scheduler_class(min_value=0, max_value=10, max_steps=1000)
        else:
            scheduler = scheduler_class(min_value=0, max_value=10)
        
        depth = scheduler.get_depth(step=100)
        
        assert isinstance(depth, int)
        assert -1 <= depth <= 10  # min_value-1 to max_value


class TestLinearDepthScheduler:
    """Test LinearDepthScheduler functionality."""
    
    def test_linear_progression(self):
        """Test linear depth progression."""
        scheduler = LinearDepthScheduler(min_value=0, max_value=8, max_steps=1000)
        
        # Collect depths over time
        depths = []
        for step in range(0, 1500, 100):
            depths.append(scheduler.get_depth(step))
        
        # Should reach max eventually
        assert max(depths) >= 7
    
    def test_linear_starts_at_min_minus_one(self):
        """Test that linear starts at min_value - 1."""
        scheduler = LinearDepthScheduler(min_value=3, max_value=10, max_steps=1000)
        
        # First call should return min-1
        depth = scheduler.get_depth(0)
        assert depth == 2  # min_value - 1


class TestExponentialDepthScheduler:
    """Test ExponentialDepthScheduler functionality."""
    
    def test_exponential_progression(self):
        """Test exponential depth progression."""
        scheduler = ExponentialDepthScheduler(
            min_value=0,
            max_value=8,
            max_steps=10000,
        )
        
        # Collect depths
        depths = []
        for step in range(0, 12000, 500):
            depths.append(scheduler.get_depth(step))
        
        # Should eventually reach high values
        assert max(depths) >= 5


class TestRandomDepthScheduler:
    """Test RandomDepthScheduler (linear weights)."""
    
    def test_random_linear_sampling(self):
        """Test random linear scheduler samples within range."""
        scheduler = RandomDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(100)]
        
        # All depths should be in range
        assert all(0 <= d <= 10 for d in depths)
        
        # Should have variety
        assert len(set(depths)) > 1
    
    def test_random_linear_bias_towards_high(self):
        """Test that linear weights bias towards higher values."""
        scheduler = RandomDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(1000)]
        mean_depth = np.mean(depths)
        
        # Mean should be above midpoint due to linear weights
        assert mean_depth > 5.0


class TestRandomHeavyTailedDepthScheduler:
    """Test RandomHeavyTailedDepthScheduler (quadratic weights)."""
    
    def test_random_heavy_tailed_sampling(self):
        """Test heavy-tailed scheduler samples within range."""
        scheduler = RandomHeavyTailedDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(100)]
        
        # All depths should be in range
        assert all(0 <= d <= 10 for d in depths)
    
    def test_random_heavy_tailed_strongly_biased(self):
        """Test that quadratic weights strongly bias towards max."""
        scheduler = RandomHeavyTailedDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(1000)]
        mean_depth = np.mean(depths)
        
        # Mean should be heavily biased towards max due to d² weights
        assert mean_depth > 6.5
    
    def test_default_scheduler_is_heavy_tailed(self):
        """Test that 'random' maps to RandomHeavyTailedDepthScheduler."""
        assert KNOWN_SCHEDULERS["random"] == RandomHeavyTailedDepthScheduler


class TestRandomUniformDepthScheduler:
    """Test RandomUniformDepthScheduler."""
    
    def test_random_uniform_sampling(self):
        """Test uniform scheduler samples within range."""
        scheduler = RandomUniformDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(100)]
        
        # All depths should be in range
        assert all(0 <= d <= 10 for d in depths)
    
    def test_random_uniform_distribution(self):
        """Test that uniform scheduler has even distribution."""
        scheduler = RandomUniformDepthScheduler(min_value=0, max_value=10)
        
        depths = [scheduler.get_depth(i) for i in range(1000)]
        mean_depth = np.mean(depths)
        
        # Mean should be near midpoint for uniform distribution
        assert 4.0 < mean_depth < 6.0


class TestSchedulerKnownDict:
    """Test KNOWN_SCHEDULERS registry."""
    
    def test_all_schedulers_registered(self):
        """Test that all expected schedulers are registered."""
        expected_schedulers = [
            "linear",
            "exponential",
            "random",
            "random_uniform",
            "random_linear",
        ]
        
        for scheduler_name in expected_schedulers:
            assert scheduler_name in KNOWN_SCHEDULERS
    
    def test_scheduler_names_match_classes(self):
        """Test that scheduler names map to correct classes."""
        assert KNOWN_SCHEDULERS["linear"] == LinearDepthScheduler
        assert KNOWN_SCHEDULERS["exponential"] == ExponentialDepthScheduler
        assert KNOWN_SCHEDULERS["random"] == RandomHeavyTailedDepthScheduler
        assert KNOWN_SCHEDULERS["random_uniform"] == RandomUniformDepthScheduler
        assert KNOWN_SCHEDULERS["random_linear"] == RandomDepthScheduler


class TestSchedulerComparison:
    """Compare behavior of different schedulers."""
    
    def test_random_schedulers_have_different_distributions(self):
        """Test that different random schedulers produce different distributions."""
        n_samples = 1000
        
        uniform_scheduler = RandomUniformDepthScheduler(min_value=0, max_value=10)
        linear_scheduler = RandomDepthScheduler(min_value=0, max_value=10)
        heavy_scheduler = RandomHeavyTailedDepthScheduler(min_value=0, max_value=10)
        
        uniform_depths = [uniform_scheduler.get_depth(i) for i in range(n_samples)]
        linear_depths = [linear_scheduler.get_depth(i) for i in range(n_samples)]
        heavy_depths = [heavy_scheduler.get_depth(i) for i in range(n_samples)]
        
        uniform_mean = np.mean(uniform_depths)
        linear_mean = np.mean(linear_depths)
        heavy_mean = np.mean(heavy_depths)
        
        # Expected ordering: uniform_mean < linear_mean < heavy_mean
        assert uniform_mean < linear_mean
        assert linear_mean < heavy_mean


class TestSchedulerEdgeCases:
    """Test edge cases for schedulers."""
    
    @pytest.mark.parametrize("scheduler_name", ["random", "random_uniform", "random_linear"])
    def test_single_depth_range(self, scheduler_name):
        """Test schedulers when min_value == max_value."""
        scheduler_class = KNOWN_SCHEDULERS[scheduler_name]
        scheduler = scheduler_class(min_value=5, max_value=5)
        
        depths = [scheduler.get_depth(i) for i in range(10)]
        
        # All depths should be 5
        assert all(d == 5 for d in depths)
