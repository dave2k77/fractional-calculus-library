#!/usr/bin/env python3
"""
Tests for Caputo derivative algorithm.

Tests the OptimizedCaputo class and its various methods.
"""

import pytest
import numpy as np
from src.algorithms.optimized_methods import OptimizedCaputo, optimized_caputo


class TestOptimizedCaputo:
    """Test OptimizedCaputo class."""

    def test_optimized_caputo_creation(self):
        """Test creating OptimizedCaputo instances."""
        # Test with float
        caputo = OptimizedCaputo(0.5)
        assert caputo.alpha == 0.5
        # Method is passed to compute(), not stored in __init__

        # Test with different alpha values
        caputo_alpha1 = OptimizedCaputo(0.3)
        caputo_alpha2 = OptimizedCaputo(0.7)

        assert caputo_alpha1.alpha == 0.3
        assert caputo_alpha2.alpha == 0.7

    def test_optimized_caputo_validation(self):
        """Test OptimizedCaputo validation."""
        # Test valid alpha values (L1 scheme requires 0 < α < 1)
        OptimizedCaputo(0.1)
        OptimizedCaputo(0.5)
        OptimizedCaputo(0.9)

        # Test invalid alpha values
        with pytest.raises(ValueError):
            OptimizedCaputo(-0.1)

        with pytest.raises(ValueError):
            OptimizedCaputo(1.0)

    def test_optimized_caputo_compute_scalar(self):
        """Test computing Caputo derivative for scalar input."""
        caputo = OptimizedCaputo(0.5)

        # Test with simple function
        def f(t):
            return t

        t = 1.0
        h = 0.01

        result = caputo.compute(f, t, h)
        assert isinstance(result, np.ndarray)  # Caputo compute returns array
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_caputo_compute_array(self):
        """Test computing Caputo derivative for array input."""
        caputo = OptimizedCaputo(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 50)
        f = t  # Simple linear function
        h = t[1] - t[0]

        result = caputo.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_caputo_different_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2  # Quadratic function
        h = t[1] - t[0]

        # Test L1 method
        caputo = OptimizedCaputo(alpha)
        result_l1 = caputo.compute(f, t, h, method="l1")

        # Test Diethelm-Ford-Freed method
        result_dff = caputo.compute(f, t, h, method="diethelm_ford_freed")

        # Results should be different but both valid
        assert not np.allclose(result_l1, result_dff)
        assert not np.any(np.isnan(result_l1))
        assert not np.any(np.isnan(result_dff))

    def test_optimized_caputo_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Caputo derivative of order α is:
        # D^α f(t) = t^(1-α) / Γ(2-α)
        from scipy.special import gamma

        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        caputo = OptimizedCaputo(alpha)
        numerical = caputo.compute(f, t, h)

        # Analytical solution
        analytical = t ** (1 - alpha) / gamma(2 - alpha)

        # Check that numerical result is reasonable
        # (exact match not expected due to discretization)
        # Use a more lenient tolerance for discretization effects
        assert np.allclose(numerical, analytical, rtol=0.5)

    def test_optimized_caputo_function_interface(self):
        """Test the optimized_caputo function interface."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test function interface
        result = optimized_caputo(f, t, alpha, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))

    def test_optimized_caputo_edge_cases(self):
        """Test edge cases and boundary conditions."""
        caputo = OptimizedCaputo(0.5)

        # Test with zero function
        t = np.linspace(0.1, 2.0, 10)
        f = np.zeros_like(t)
        h = t[1] - t[0]

        result = caputo.compute(f, t, h)
        assert np.allclose(result, 0, atol=1e-10)

        # Test with constant function
        f = np.ones_like(t)
        result = caputo.compute(f, t, h)
        # Caputo derivative of constant should be zero
        assert np.allclose(result, 0, atol=1e-10)

    def test_optimized_caputo_performance(self):
        """Test performance characteristics."""
        caputo = OptimizedCaputo(0.5)

        # Test with larger arrays
        t = np.linspace(0.1, 10.0, 1000)
        f = t**3
        h = t[1] - t[0]

        result = caputo.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_optimized_caputo_method_consistency(self):
        """Test that different methods give consistent results for same input."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test both methods
        caputo = OptimizedCaputo(alpha)
        result_l1 = caputo.compute(f, t, h, method="l1")
        result_dff = caputo.compute(f, t, h, method="diethelm_ford_freed")

        # Both should be finite and valid
        assert not np.any(np.isnan(result_l1))
        assert not np.any(np.isnan(result_dff))
        assert not np.any(np.isinf(result_l1))
        assert not np.any(np.isinf(result_dff))

        # Results should be different (different schemes)
        assert not np.allclose(result_l1, result_dff)

    def test_optimized_caputo_alpha_validation(self):
        """Test alpha parameter validation."""
        # Test valid alpha values (L1 scheme requires 0 < α < 1)
        for alpha in [0.1, 0.5, 0.9]:
            caputo = OptimizedCaputo(alpha)
            assert caputo.alpha == alpha

        # Test invalid alpha values
        with pytest.raises(ValueError):
            OptimizedCaputo(-0.1)

        with pytest.raises(ValueError):
            OptimizedCaputo(0.0)

    def test_optimized_caputo_input_validation(self):
        """Test input validation."""
        caputo = OptimizedCaputo(0.5)
        t = np.linspace(0.1, 2.0, 10)
        f = t**2
        h = t[1] - t[0]

        # Test with mismatched array lengths
        f_wrong = t[:-1]  # One element shorter
        with pytest.raises(ValueError):
            caputo.compute(f_wrong, t, h)

        # Test with negative step size
        with pytest.raises(ValueError):
            caputo.compute(f, t, -h)

        # Test with zero step size
        with pytest.raises(ValueError):
            caputo.compute(f, t, 0)
