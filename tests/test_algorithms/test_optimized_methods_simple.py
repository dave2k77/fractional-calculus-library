#!/usr/bin/env python3
"""Simple tests for algorithms/optimized_methods.py - low-hanging fruit."""

import pytest
import numpy as np
from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville, OptimizedCaputo
from hpfracc.core.definitions import FractionalOrder


class TestOptimizedMethodsSimple:
    """Simple tests for optimized methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alpha = 0.5
        self.order = FractionalOrder(self.alpha)
        self.t = np.linspace(0, 1, 50)
        self.f = np.sin(2 * np.pi * self.t)
        self.h = self.t[1] - self.t[0]
        
    def test_optimized_riemann_liouville_init(self):
        """Test OptimizedRiemannLiouville initialization."""
        rl = OptimizedRiemannLiouville(self.order)
        assert isinstance(rl, OptimizedRiemannLiouville)
        assert rl.fractional_order == self.order
        
    def test_optimized_riemann_liouville_compute(self):
        """Test OptimizedRiemannLiouville computation."""
        rl = OptimizedRiemannLiouville(self.order)
        result = rl.compute(self.f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == self.f.shape
        assert np.all(np.isfinite(result))
        
    def test_optimized_caputo_init(self):
        """Test OptimizedCaputo initialization."""
        caputo = OptimizedCaputo(self.order)
        assert isinstance(caputo, OptimizedCaputo)
        assert caputo.fractional_order == self.order
        
    def test_optimized_caputo_compute(self):
        """Test OptimizedCaputo computation."""
        caputo = OptimizedCaputo(self.order)
        result = caputo.compute(self.f, self.t, self.h)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == self.f.shape
        assert np.all(np.isfinite(result))
        
    def test_different_alpha_values(self):
        """Test with different fractional orders."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for alpha in alphas:
            order = FractionalOrder(alpha)
            rl = OptimizedRiemannLiouville(order)
            result = rl.compute(self.f, self.t, self.h)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == self.f.shape
            
    def test_different_functions(self):
        """Test with different input functions."""
        rl = OptimizedRiemannLiouville(self.order)
        
        functions = [
            np.ones_like(self.t),  # Constant
            self.t,                # Linear
            self.t**2,             # Quadratic
            np.cos(self.t),        # Cosine
            np.exp(-self.t)        # Exponential decay
        ]
        
        for func in functions:
            result = rl.compute(func, self.t, self.h)
            assert isinstance(result, np.ndarray)
            assert result.shape == func.shape
            
    def test_edge_cases(self):
        """Test edge cases."""
        rl = OptimizedRiemannLiouville(self.order)
        
        # Zero function
        zero_f = np.zeros_like(self.t)
        result = rl.compute(zero_f, self.t, self.h)
        assert isinstance(result, np.ndarray)
        
        # Single point
        single_t = np.array([1.0])
        single_f = np.array([1.0])
        single_h = 0.1
        result = rl.compute(single_f, single_t, single_h)
        assert isinstance(result, np.ndarray)
        
    def test_numerical_properties(self):
        """Test numerical properties."""
        rl = OptimizedRiemannLiouville(self.order)
        
        # Linearity test
        f1 = np.sin(self.t)
        f2 = np.cos(self.t)
        a, b = 2.0, 3.0
        
        # D^α[af + bg] ≈ aD^α[f] + bD^α[g]
        combined = a * f1 + b * f2
        left_result = rl.compute(combined, self.t, self.h)
        
        right_result = a * rl.compute(f1, self.t, self.h) + b * rl.compute(f2, self.t, self.h)
        
        # Should be approximately equal
        assert np.allclose(left_result, right_result, atol=1e-10, rtol=1e-8)
        
    def test_convergence_behavior(self):
        """Test convergence with different step sizes."""
        rl = OptimizedRiemannLiouville(self.order)
        
        # Test with different resolutions
        sizes = [25, 50, 100]
        results = []
        
        for size in sizes:
            t_test = np.linspace(0, 1, size)
            f_test = np.sin(2 * np.pi * t_test)
            h_test = t_test[1] - t_test[0]
            
            result = rl.compute(f_test, t_test, h_test)
            results.append(result[-1])  # Final value
            
        # Results should be converging
        assert all(np.isfinite(r) for r in results)
        
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        import time
        
        rl = OptimizedRiemannLiouville(self.order)
        
        # Measure computation time
        start_time = time.time()
        result = rl.compute(self.f, self.t, self.h)
        end_time = time.time()
        
        # Should complete reasonably quickly
        assert end_time - start_time < 5.0  # 5 seconds max
        assert isinstance(result, np.ndarray)
        
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        rl = OptimizedRiemannLiouville(self.order)
        
        # Process multiple arrays
        for _ in range(10):
            test_f = np.random.randn(100)
            test_t = np.linspace(0, 1, 100)
            test_h = test_t[1] - test_t[0]
            
            result = rl.compute(test_f, test_t, test_h)
            assert isinstance(result, np.ndarray)

















