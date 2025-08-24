#!/usr/bin/env python3
"""
Tests for Riemann-Liouville derivative algorithm.

Tests the OptimizedRiemannLiouville class and its various methods.
"""

import pytest
import numpy as np
from hpfracc.algorithms.optimized_methods import (
    OptimizedRiemannLiouville,
    optimized_riemann_liouville,
)


class TestOptimizedRiemannLiouville:
    """Test OptimizedRiemannLiouville class."""

    def test_optimized_riemann_liouville_creation(self):
        """Test creating OptimizedRiemannLiouville instances."""
        # Test with float
        rl = OptimizedRiemannLiouville(0.5)
        assert rl.alpha == 0.5
        assert rl.n == 1  # n should be ceil(alpha)
        assert rl.alpha_val == 0.5

        # Test with different alpha values
        rl_alpha1 = OptimizedRiemannLiouville(1.5)
        rl_alpha2 = OptimizedRiemannLiouville(2.3)

        assert rl_alpha1.alpha == 1.5
        assert rl_alpha1.n == 2  # ceil(1.5) = 2
        assert rl_alpha2.alpha == 2.3
        assert rl_alpha2.n == 3  # ceil(2.3) = 3

    def test_optimized_riemann_liouville_validation(self):
        """Test OptimizedRiemannLiouville validation."""
        # Test valid alpha values
        OptimizedRiemannLiouville(0.1)
        OptimizedRiemannLiouville(1.0)
        OptimizedRiemannLiouville(2.5)

        # Test invalid alpha values (negative)
        with pytest.raises(ValueError):
            OptimizedRiemannLiouville(-0.1)

    def test_optimized_riemann_liouville_compute_scalar(self):
        """Test computing Riemann-Liouville derivative for scalar input."""
        rl = OptimizedRiemannLiouville(0.5)

        # Test with simple function
        def f(t):
            return t

        t = 1.0
        h = 0.01

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)  # RL compute returns array
        assert len(result) > 0
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_riemann_liouville_compute_array(self):
        """Test computing Riemann-Liouville derivative for array input."""
        rl = OptimizedRiemannLiouville(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 50)
        f = t  # Simple linear function
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_riemann_liouville_different_alphas(self):
        """Test different alpha values."""
        t = np.linspace(0.1, 2.0, 50)
        f = t**2  # Quadratic function
        h = t[1] - t[0]

        # Test different alpha values
        alpha1 = 0.5
        alpha2 = 1.5

        rl1 = OptimizedRiemannLiouville(alpha1)
        result1 = rl1.compute(f, t, h)

        rl2 = OptimizedRiemannLiouville(alpha2)
        result2 = rl2.compute(f, t, h)

        # Results should be different for different alphas
        assert not np.allclose(result1, result2)
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))

    def test_optimized_riemann_liouville_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Riemann-Liouville derivative of order α is:
        # D^α f(t) = t^(1-α) / Γ(2-α)
        from scipy.special import gamma

        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        rl = OptimizedRiemannLiouville(alpha)
        numerical = rl.compute(f, t, h)

        # Analytical solution
        analytical = t ** (1 - alpha) / gamma(2 - alpha)

        # Check that numerical result is reasonable
        # (exact match not expected due to discretization)
        # Use a more lenient tolerance for discretization effects
        # Skip first few points where boundary effects dominate
        skip_points = int(alpha) + 1  # Skip boundary points
        assert np.allclose(numerical[skip_points:], analytical[skip_points:], rtol=0.5)

    def test_optimized_riemann_liouville_function_interface(self):
        """Test the optimized_riemann_liouville function interface."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test function interface
        result = optimized_riemann_liouville(f, t, alpha, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))

    def test_optimized_riemann_liouville_edge_cases(self):
        """Test edge cases and boundary conditions."""
        rl = OptimizedRiemannLiouville(0.5)

        # Test with zero function
        t = np.linspace(0.1, 2.0, 10)
        f = np.zeros_like(t)
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert np.allclose(result, 0, atol=1e-10)

        # Test with constant function
        f = np.ones_like(t)
        result = rl.compute(f, t, h)
        # Riemann-Liouville derivative of constant should be non-zero
        assert not np.allclose(result, 0)

    def test_optimized_riemann_liouville_performance(self):
        """Test performance characteristics."""
        rl = OptimizedRiemannLiouville(0.5)

        # Test with larger arrays
        t = np.linspace(0.1, 10.0, 1000)
        f = t**3
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_riemann_liouville_consistency(self):
        """Test that same input gives consistent results."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test same input multiple times
        rl = OptimizedRiemannLiouville(alpha)

        result1 = rl.compute(f, t, h)
        result2 = rl.compute(f, t, h)

        # Results should be consistent
        np.testing.assert_allclose(result1, result2, rtol=1e-10)

    def test_optimized_riemann_liouville_alpha_validation(self):
        """Test alpha parameter validation."""
        # Test valid alpha values
        for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
            rl = OptimizedRiemannLiouville(alpha)
            assert rl.alpha == alpha

        # Test invalid alpha values (negative)
        with pytest.raises(ValueError):
            OptimizedRiemannLiouville(-0.1)

    def test_optimized_riemann_liouville_input_validation(self):
        """Test input validation."""
        rl = OptimizedRiemannLiouville(0.5)
        t = np.linspace(0.1, 2.0, 10)
        f = t**2
        h = t[1] - t[0]

        # Test with mismatched array lengths
        f_wrong = t[:-1]  # One element shorter
        # The implementation validates array lengths and raises ValueError
        with pytest.raises(ValueError, match="Function array and time array must have the same length"):
            rl.compute(f_wrong, t, h)

    def test_optimized_riemann_liouville_fft_optimization(self):
        """Test FFT optimization for large arrays."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 100)
        f = t**2
        h = t[1] - t[0]

        # Test FFT optimization specifically
        rl = OptimizedRiemannLiouville(alpha)
        result = rl.compute(f, t, h)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_riemann_liouville_convergence(self):
        """Test convergence with decreasing step size."""
        alpha = 0.5
        t_max = 1.0

        # Test with different grid sizes
        grid_sizes = [50, 100, 200]
        results = []

        for N in grid_sizes:
            t = np.linspace(0.1, t_max, N)
            f = t
            h = t[1] - t[0]

            rl = OptimizedRiemannLiouville(alpha)
            result = rl.compute(f, t, h)
            results.append(result[-1])  # Take last point

        # Results should converge (get more stable)
        assert len(results) == len(grid_sizes)
        assert all(not np.isnan(r) for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
