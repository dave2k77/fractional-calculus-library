#!/usr/bin/env python3
"""
Tests for novel fractional derivatives.

Tests the novel fractional derivative implementations including
Caputo-Fabrizio and Atangana-Baleanu derivatives.
"""

import pytest
import numpy as np
from hpfracc.algorithms.novel_derivatives import (
    CaputoFabrizioDerivative,
    AtanganaBaleanuDerivative
)


class TestCaputoFabrizioDerivative:
    """Test Caputo-Fabrizio derivative implementation."""

    def test_caputo_fabrizio_creation(self):
        """Test creating CaputoFabrizioDerivative instances."""
        # Test with valid alpha values
        cf = CaputoFabrizioDerivative(0.5)
        assert cf.alpha == 0.5
        assert cf.method == "auto"
        assert cf.optimize_memory is True

        # Test with different methods
        cf_fft = CaputoFabrizioDerivative(0.3, method="fft")
        assert cf_fft.method == "fft"

        cf_direct = CaputoFabrizioDerivative(0.7, method="direct")
        assert cf_direct.method == "direct"

    def test_caputo_fabrizio_validation(self):
        """Test CaputoFabrizioDerivative validation."""
        # Test valid alpha values (0 ≤ α < 1)
        CaputoFabrizioDerivative(0.0)
        CaputoFabrizioDerivative(0.5)
        CaputoFabrizioDerivative(0.99)

        # Test invalid alpha values
        with pytest.raises(ValueError):
            CaputoFabrizioDerivative(-0.1)

        with pytest.raises(ValueError):
            CaputoFabrizioDerivative(1.0)

        # Test invalid methods
        with pytest.raises(ValueError):
            CaputoFabrizioDerivative(0.5, method="invalid")

    def test_caputo_fabrizio_compute_scalar(self):
        """Test computing Caputo-Fabrizio derivative for scalar input."""
        cf = CaputoFabrizioDerivative(0.5)

        # Test with simple function
        def f(t):
            return t

        t = np.linspace(0.1, 2.0, 100)
        result = cf.compute(f, t)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_caputo_fabrizio_compute_array(self):
        """Test computing Caputo-Fabrizio derivative for array input."""
        cf = CaputoFabrizioDerivative(0.5)

        # Test with array function values
        t = np.linspace(0.1, 2.0, 100)
        f_values = t**2  # Quadratic function
        result = cf.compute(f_values, t)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_caputo_fabrizio_different_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x**2

        # Test different methods
        cf_auto = CaputoFabrizioDerivative(alpha, method="auto")
        cf_fft = CaputoFabrizioDerivative(alpha, method="fft")
        cf_direct = CaputoFabrizioDerivative(alpha, method="direct")

        result_auto = cf_auto.compute(f, t)
        result_fft = cf_fft.compute(f, t)
        result_direct = cf_direct.compute(f, t)

        # All results should be valid
        assert not np.any(np.isnan(result_auto))
        assert not np.any(np.isnan(result_fft))
        assert not np.any(np.isnan(result_direct))

        # Results should be similar (within numerical tolerance)
        assert np.allclose(result_auto, result_fft, rtol=1e-2)
        assert np.allclose(result_auto, result_direct, rtol=1e-2)

    def test_caputo_fabrizio_analytical_comparison(self):
        """Test against known analytical results."""
        # For f(t) = t, the Caputo-Fabrizio derivative should be well-behaved
        alpha = 0.5
        cf = CaputoFabrizioDerivative(alpha)

        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x

        result = cf.compute(f, t)

        # The derivative should be positive and increasing
        assert np.all(result >= 0)
        assert np.all(np.diff(result) >= -1e-10)  # Allow small numerical errors

    def test_caputo_fabrizio_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small alpha
        cf_small = CaputoFabrizioDerivative(0.01)
        t = np.linspace(0.1, 1.0, 50)
        f = lambda x: x
        result = cf_small.compute(f, t)
        assert not np.any(np.isnan(result))

        # Test with alpha close to 1
        cf_large = CaputoFabrizioDerivative(0.99)
        result = cf_large.compute(f, t)
        assert not np.any(np.isnan(result))

    def test_caputo_fabrizio_performance(self):
        """Test performance characteristics."""
        cf = CaputoFabrizioDerivative(0.5)
        t = np.linspace(0.1, 2.0, 1000)
        f = lambda x: x**2

        # Should complete in reasonable time
        import time
        start_time = time.time()
        result = cf.compute(f, t)
        end_time = time.time()

        assert end_time - start_time < 10.0  # Should complete within 10 seconds
        assert result.shape == t.shape


class TestAtanganaBaleanuDerivative:
    """Test Atangana-Baleanu derivative implementation."""

    def test_atangana_baleanu_creation(self):
        """Test creating AtanganaBaleanuDerivative instances."""
        # Test with valid alpha values
        ab = AtanganaBaleanuDerivative(0.5)
        assert ab.alpha == 0.5
        assert ab.method == "auto"

        # Test with different methods
        ab_fft = AtanganaBaleanuDerivative(0.3, method="fft")
        assert ab_fft.method == "fft"

    def test_atangana_baleanu_validation(self):
        """Test AtanganaBaleanuDerivative validation."""
        # Test valid alpha values (0 ≤ α < 1)
        AtanganaBaleanuDerivative(0.0)
        AtanganaBaleanuDerivative(0.5)
        AtanganaBaleanuDerivative(0.99)

        # Test invalid alpha values
        with pytest.raises(ValueError):
            AtanganaBaleanuDerivative(-0.1)

        with pytest.raises(ValueError):
            AtanganaBaleanuDerivative(1.0)

    def test_atangana_baleanu_compute(self):
        """Test computing Atangana-Baleanu derivative."""
        ab = AtanganaBaleanuDerivative(0.5)
        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x

        result = ab.compute(f, t)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_atangana_baleanu_properties(self):
        """Test mathematical properties of Atangana-Baleanu derivative."""
        alpha = 0.5
        ab = AtanganaBaleanuDerivative(alpha)
        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x**2

        result = ab.compute(f, t)

        # Should preserve some basic properties
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestNovelDerivativesIntegration:
    """Test integration between different novel derivatives."""

    def test_derivative_consistency(self):
        """Test consistency between different derivative types."""
        alpha = 0.5
        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x

        # Create different derivative types
        cf = CaputoFabrizioDerivative(alpha)
        ab = AtanganaBaleanuDerivative(alpha)

        # Compute derivatives
        result_cf = cf.compute(f, t)
        result_ab = ab.compute(f, t)

        # Both should produce valid results
        assert not np.any(np.isnan(result_cf))
        assert not np.any(np.isnan(result_ab))
        assert result_cf.shape == result_ab.shape

    def test_method_consistency(self):
        """Test consistency of different computation methods."""
        alpha = 0.5
        cf = CaputoFabrizioDerivative(alpha)
        t = np.linspace(0.1, 2.0, 100)
        f = lambda x: x**2

        # Test different methods
        result_auto = cf.compute(f, t, method="auto")
        result_fft = cf.compute(f, t, method="fft")
        result_direct = cf.compute(f, t, method="direct")

        # All should produce similar results
        assert np.allclose(result_auto, result_fft, rtol=1e-2)
        assert np.allclose(result_auto, result_direct, rtol=1e-2)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with invalid function
        cf = CaputoFabrizioDerivative(0.5)
        t = np.linspace(0.1, 2.0, 100)

        with pytest.raises(Exception):
            cf.compute(None, t)

        # Test with invalid time array
        f = lambda x: x
        with pytest.raises(Exception):
            cf.compute(f, None)

    def test_performance_comparison(self):
        """Test performance comparison between methods."""
        alpha = 0.5
        cf = CaputoFabrizioDerivative(alpha)
        t = np.linspace(0.1, 2.0, 1000)
        f = lambda x: x**2

        import time

        # Time different methods
        start_time = time.time()
        result_fft = cf.compute(f, t, method="fft")
        fft_time = time.time() - start_time

        start_time = time.time()
        result_direct = cf.compute(f, t, method="direct")
        direct_time = time.time() - start_time

        # Both should complete successfully
        assert result_fft.shape == t.shape
        assert result_direct.shape == t.shape
        assert not np.any(np.isnan(result_fft))
        assert not np.any(np.isnan(result_direct))


if __name__ == "__main__":
    pytest.main([__file__])
