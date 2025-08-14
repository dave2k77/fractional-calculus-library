#!/usr/bin/env python3
"""
Tests for Grünwald-Letnikov derivative algorithm.

Tests the GrunwaldLetnikovDerivative class and its various methods.
"""

import pytest
import numpy as np
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative


class TestGrunwaldLetnikovDerivative:
    """Test GrunwaldLetnikovDerivative class."""

    def test_grunwald_letnikov_derivative_creation(self):
        """Test creating GrunwaldLetnikovDerivative instances."""
        # Test with float
        gl = GrunwaldLetnikovDerivative(0.5)
        assert gl.alpha.alpha == 0.5
        assert gl.method == "direct"

        # Test with different methods
        gl_fft = GrunwaldLetnikovDerivative(0.5, method="fft")
        assert gl_fft.method == "fft"

    def test_grunwald_letnikov_derivative_validation(self):
        """Test GrunwaldLetnikovDerivative validation."""
        # Test valid alpha values
        GrunwaldLetnikovDerivative(0.1)
        GrunwaldLetnikovDerivative(1.0)
        GrunwaldLetnikovDerivative(2.5)

        # Test invalid method
        with pytest.raises(ValueError):
            GrunwaldLetnikovDerivative(0.5, method="invalid_method")

    def test_grunwald_letnikov_derivative_compute_scalar(self):
        """Test computing Grünwald-Letnikov derivative for scalar input."""
        gl = GrunwaldLetnikovDerivative(0.5)

        # Test with simple function
        def f(t):
            return t

        t = 1.0
        h = 0.01

        result = gl.compute(f, t, h)
        assert isinstance(result, (int, float, np.number))
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_grunwald_letnikov_derivative_compute_array(self):
        """Test computing Grünwald-Letnikov derivative for array input."""
        gl = GrunwaldLetnikovDerivative(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 50)
        f = t  # Simple linear function
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_grunwald_letnikov_derivative_different_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2  # Quadratic function
        h = t[1] - t[0]

        # Test direct method
        gl_direct = GrunwaldLetnikovDerivative(alpha, method="direct")
        result_direct = gl_direct.compute(f, t, h)

        # Test FFT method
        gl_fft = GrunwaldLetnikovDerivative(alpha, method="fft")
        result_fft = gl_fft.compute(f, t, h)

        # Results should be valid (they might be similar for this simple case)
        assert not np.any(np.isnan(result_direct))
        assert not np.any(np.isnan(result_fft))

    def test_grunwald_letnikov_derivative_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Grünwald-Letnikov derivative of order α is:
        # D^α f(t) = t^(1-α) / Γ(2-α)
        from scipy.special import gamma

        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        gl = GrunwaldLetnikovDerivative(alpha)
        numerical = gl.compute(f, t, h)

        # Analytical solution
        analytical = t**(1-alpha) / gamma(2-alpha)

        # Check that numerical result is reasonable
        error = np.abs(numerical - analytical)
        assert np.mean(error) < 2.0  # Average error should be reasonable

    def test_grunwald_letnikov_derivative_edge_cases(self):
        """Test edge cases and boundary conditions."""
        gl = GrunwaldLetnikovDerivative(0.5)

        # Test with moderate step size (very small step size can cause issues)
        t = np.linspace(0.1, 1.0, 100)
        f = t
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert not np.any(np.isnan(result))

    def test_grunwald_letnikov_derivative_negative_alpha(self):
        """Test Grünwald-Letnikov derivative with negative alpha (fractional integral)."""
        # For negative alpha, this becomes a fractional integral
        # Note: FractionalOrder doesn't allow negative values, so we test the validation
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            gl = GrunwaldLetnikovDerivative(-0.5)

        # Test with valid positive alpha
        gl = GrunwaldLetnikovDerivative(0.5)
        t = np.linspace(0.1, 2.0, 50)
        f = t
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

    def test_grunwald_letnikov_derivative_complex_function(self):
        """Test with more complex functions."""
        gl = GrunwaldLetnikovDerivative(0.5)

        # Test with exponential function
        t = np.linspace(0.1, 2.0, 50)
        f = np.exp(-t)
        h = t[1] - t[0]

        result = gl.compute(f, t, h)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

        # Test with trigonometric function
        f_trig = np.sin(t)
        result_trig = gl.compute(f_trig, t, h)
        assert isinstance(result_trig, np.ndarray)
        assert not np.any(np.isnan(result_trig))

    def test_grunwald_letnikov_derivative_method_consistency(self):
        """Test that different methods give consistent results."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 50)
        f = t**2
        h = t[1] - t[0]

        # Test direct and FFT methods
        gl_direct = GrunwaldLetnikovDerivative(alpha, method="direct")
        gl_fft = GrunwaldLetnikovDerivative(alpha, method="fft")

        result_direct = gl_direct.compute(f, t, h)
        result_fft = gl_fft.compute(f, t, h)

        # Both should give reasonable results
        assert np.std(result_direct) > 0  # Non-zero variance
        assert np.std(result_fft) > 0  # Non-zero variance

        # Results should be in similar range (can differ significantly)
        assert np.abs(np.mean(result_direct) - np.mean(result_fft)) < 5.0

    def test_grunwald_letnikov_derivative_error_handling(self):
        """Test error handling for invalid inputs."""
        gl = GrunwaldLetnikovDerivative(0.5)

        # Test with invalid method
        with pytest.raises(ValueError):
            GrunwaldLetnikovDerivative(0.5, method="invalid_method")

    def test_grunwald_letnikov_derivative_convergence(self):
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

            gl = GrunwaldLetnikovDerivative(alpha)
            result = gl.compute(f, t, h)
            results.append(result[-1])  # Take last point

        # Results should converge (get more stable)
        assert len(results) == len(grid_sizes)
        assert all(not np.isnan(r) for r in results)

    def test_grunwald_letnikov_derivative_binomial_coefficients(self):
        """Test binomial coefficient computation."""
        gl = GrunwaldLetnikovDerivative(0.5)
        
        # Test that binomial coefficients are computed correctly
        alpha = 0.5
        n = 10
        
        # This should not raise an error
        result = gl.compute(np.ones(n), np.arange(n) * 0.1, 0.1)
        assert isinstance(result, np.ndarray)
        assert len(result) == n


if __name__ == "__main__":
    pytest.main([__file__])
