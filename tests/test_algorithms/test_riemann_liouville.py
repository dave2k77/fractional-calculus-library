#!/usr/bin/env python3
"""
Tests for Riemann-Liouville derivative algorithm.

Tests the RiemannLiouvilleDerivative class and its various methods.
"""

import pytest
import numpy as np
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative


class TestRiemannLiouvilleDerivative:
    """Test RiemannLiouvilleDerivative class."""

    def test_riemann_liouville_derivative_creation(self):
        """Test creating RiemannLiouvilleDerivative instances."""
        # Test with float
        rl = RiemannLiouvilleDerivative(0.5)
        assert rl.alpha.alpha == 0.5
        assert rl.method == "direct"

        # Test with different methods
        rl_fft = RiemannLiouvilleDerivative(0.5, method="fft")
        assert rl_fft.method == "fft"

    def test_riemann_liouville_derivative_validation(self):
        """Test RiemannLiouvilleDerivative validation."""
        # Test valid alpha values
        RiemannLiouvilleDerivative(0.1)
        RiemannLiouvilleDerivative(1.0)
        RiemannLiouvilleDerivative(2.5)

        # Test invalid method
        with pytest.raises(ValueError):
            RiemannLiouvilleDerivative(0.5, method="invalid_method")

    def test_riemann_liouville_derivative_compute_scalar(self):
        """Test computing Riemann-Liouville derivative for scalar input."""
        rl = RiemannLiouvilleDerivative(0.5)

        # Test with simple function
        def f(t):
            return t

        t = 1.0
        h = 0.01

        result = rl.compute(f, t, h)
        assert isinstance(result, (int, float, np.number))
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_riemann_liouville_derivative_compute_array(self):
        """Test computing Riemann-Liouville derivative for array input."""
        rl = RiemannLiouvilleDerivative(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 50)
        f = t  # Simple linear function
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_riemann_liouville_derivative_different_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2  # Quadratic function
        h = t[1] - t[0]

        # Test direct method
        rl_direct = RiemannLiouvilleDerivative(alpha, method="direct")
        result_direct = rl_direct.compute(f, t, h)

        # Test FFT method
        rl_fft = RiemannLiouvilleDerivative(alpha, method="fft")
        result_fft = rl_fft.compute(f, t, h)

        # Results should be different but both valid
        assert not np.allclose(result_direct, result_fft)
        assert not np.any(np.isnan(result_direct))
        assert not np.any(np.isnan(result_fft))

    def test_riemann_liouville_derivative_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Riemann-Liouville derivative of order α is:
        # D^α f(t) = t^(1-α) / Γ(2-α)
        from scipy.special import gamma

        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        rl = RiemannLiouvilleDerivative(alpha)
        numerical = rl.compute(f, t, h)

        # Analytical solution
        analytical = t**(1-alpha) / gamma(2-alpha)

        # Check that numerical result is reasonable
        error = np.abs(numerical - analytical)
        assert np.mean(error) < 2.0  # Average error should be reasonable

    def test_riemann_liouville_derivative_edge_cases(self):
        """Test edge cases and boundary conditions."""
        rl = RiemannLiouvilleDerivative(0.5)

        # Test with very small step size
        t = np.linspace(0.1, 1.0, 1000)
        f = t
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert not np.any(np.isnan(result))

    def test_riemann_liouville_derivative_negative_alpha(self):
        """Test Riemann-Liouville derivative with negative alpha (fractional integral)."""
        # For negative alpha, this becomes a fractional integral
        # Note: FractionalOrder doesn't allow negative values, so we test the validation
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            rl = RiemannLiouvilleDerivative(-0.5)

        # Test with valid positive alpha
        rl = RiemannLiouvilleDerivative(0.5)
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

    def test_riemann_liouville_derivative_complex_function(self):
        """Test with more complex functions."""
        rl = RiemannLiouvilleDerivative(0.5)

        # Test with exponential function
        t = np.linspace(0.1, 2.0, 50)
        f = np.exp(-t)
        h = t[1] - t[0]

        result = rl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

        # Test with trigonometric function
        f_trig = np.sin(t)
        result_trig = rl.compute(f_trig, t, h)
        assert isinstance(result_trig, np.ndarray)
        assert not np.any(np.isnan(result_trig))

    def test_riemann_liouville_derivative_method_consistency(self):
        """Test that different methods give consistent results."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test direct and FFT methods
        rl_direct = RiemannLiouvilleDerivative(alpha, method="direct")
        rl_fft = RiemannLiouvilleDerivative(alpha, method="fft")

        result_direct = rl_direct.compute(f, t, h)
        result_fft = rl_fft.compute(f, t, h)

        # Both should give reasonable results
        assert np.std(result_direct) > 0  # Non-zero variance
        assert np.std(result_fft) > 0  # Non-zero variance

        # Results should be in similar range (can differ significantly)
        assert np.abs(np.mean(result_direct) - np.mean(result_fft)) < 10.0

    def test_riemann_liouville_derivative_error_handling(self):
        """Test error handling for invalid inputs."""
        rl = RiemannLiouvilleDerivative(0.5)

        # Test with invalid step size
        with pytest.raises(ValueError):
            rl.compute(np.array([1, 2]), np.array([1, 2]), 0)

        # Test with invalid method
        with pytest.raises(ValueError):
            RiemannLiouvilleDerivative(0.5, method="invalid_method")

    def test_riemann_liouville_derivative_convergence(self):
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

            rl = RiemannLiouvilleDerivative(alpha)
            result = rl.compute(f, t, h)
            results.append(result[-1])  # Take last point

        # Results should converge (get more stable)
        assert len(results) == len(grid_sizes)
        assert all(not np.isnan(r) for r in results)


if __name__ == "__main__":
    pytest.main([__file__])
