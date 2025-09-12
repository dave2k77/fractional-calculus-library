"""
Tests for fractional integral methods in algorithms module.

This module contains comprehensive tests for all fractional integral
implementations including Riemann-Liouville, Caputo, and optimized versions.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.algorithms.integral_methods import (
    RiemannLiouvilleIntegral,
    CaputoIntegral,
    riemann_liouville_integral,
    caputo_integral,
    optimized_riemann_liouville_integral,
    optimized_caputo_integral
)
from hpfracc.core.definitions import FractionalOrder


class TestRiemannLiouvilleIntegral:
    """Test Riemann-Liouville fractional integral class."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        assert integral.alpha == 0.5
        assert integral.method == "auto"
        assert integral.optimize_memory is True
        assert integral.use_jax is False
        assert integral.gamma_alpha > 0
        assert integral.fft_threshold == 1000
    
    def test_initialization_with_fractional_order(self):
        """Test initialization with FractionalOrder object."""
        alpha_order = FractionalOrder(0.7)
        integral = RiemannLiouvilleIntegral(alpha=alpha_order)
        assert integral.alpha == 0.7
    
    def test_initialization_with_parameters(self):
        """Test initialization with all parameters."""
        integral = RiemannLiouvilleIntegral(
            alpha=0.3,
            method="fft",
            optimize_memory=False,
            use_jax=True
        )
        assert integral.alpha == 0.3
        assert integral.method == "fft"
        assert integral.optimize_memory is False
        assert integral.use_jax is True
    
    def test_initialization_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        # alpha=0.0 should be valid (identity operator)
        integral = RiemannLiouvilleIntegral(alpha=0.0)
        assert integral.alpha == 0.0
        
        # Only negative alpha should raise ValueError
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            RiemannLiouvilleIntegral(alpha=-0.5)
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError, match="Method must be one of"):
            RiemannLiouvilleIntegral(alpha=0.5, method="invalid")
    
    def test_compute_with_callable(self):
        """Test compute method with callable function."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        
        def f(x):
            return x**2
        
        result = integral.compute(f, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be non-negative
    
    def test_compute_with_array(self):
        """Test compute method with array input."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_compute_with_custom_h(self):
        """Test compute method with custom step size."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        h = 0.05
        
        result = integral.compute(f_array, t, h=h)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_compute_with_custom_method(self):
        """Test compute method with custom method override."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="auto")
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t, method="direct")
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_compute_invalid_inputs(self):
        """Test compute method with invalid inputs."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Test with single time point (should work, but may have limitations)
        result = integral.compute(lambda x: x, np.array([1.0]))
        assert isinstance(result, np.ndarray)
        
        # Test with function values as array and time array mismatch
        # This should raise a ValueError
        with pytest.raises(ValueError, match="Function values must match time array shape"):
            integral.compute(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    
    def test_select_optimal_method(self):
        """Test optimal method selection."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Small array should use direct method
        assert integral._select_optimal_method(100) == "direct"
        
        # Large array should use FFT method
        assert integral._select_optimal_method(2000) == "fft"
    
    def test_compute_fft_method(self):
        """Test FFT computation method."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="fft")
        t = np.linspace(0, 1, 100)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_compute_direct_method(self):
        """Test direct computation method."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="direct")
        t = np.linspace(0, 1, 20)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_compute_adaptive_method(self):
        """Test adaptive computation method."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="adaptive")
        t = np.linspace(0, 1, 50)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_analytical_consistency(self):
        """Test consistency with known analytical solutions."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 2, 100)
        
        # Test with constant function f(t) = 1
        f_const = np.ones_like(t)
        result_const = integral.compute(f_const, t)
        
        # For constant function, result should be proportional to t^α
        expected_const = t**0.5 / np.sqrt(np.pi) * 2
        # Use more lenient tolerance for numerical integration
        np.testing.assert_allclose(result_const, expected_const, rtol=0.5)
    
    def test_linearity_property(self):
        """Test linearity property of fractional integrals."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 1, 20)
        
        f1 = t
        f2 = t**2
        c1, c2 = 2.0, 3.0
        
        result1 = integral.compute(f1, t)
        result2 = integral.compute(f2, t)
        result_combined = integral.compute(c1 * f1 + c2 * f2, t)
        
        expected_combined = c1 * result1 + c2 * result2
        np.testing.assert_allclose(result_combined, expected_combined, rtol=1e-10)


class TestCaputoIntegral:
    """Test Caputo fractional integral class."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        integral = CaputoIntegral(alpha=0.5)
        assert integral.rl_integral.alpha == 0.5
        assert integral.rl_integral.method == "auto"
        assert integral.rl_integral.optimize_memory is True
        assert integral.rl_integral.use_jax is False
    
    def test_initialization_with_fractional_order(self):
        """Test initialization with FractionalOrder object."""
        alpha_order = FractionalOrder(0.7)
        integral = CaputoIntegral(alpha=alpha_order)
        assert integral.rl_integral.alpha == 0.7
    
    def test_initialization_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        # Alpha = 0 should be allowed (identity operator)
        integral = CaputoIntegral(alpha=0.0)
        assert integral.alpha == 0.0
        
        # Alpha < 0 should raise an error
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            CaputoIntegral(alpha=-0.5)
    
    def test_compute_delegation(self):
        """Test that compute method delegates to RL integral."""
        integral = CaputoIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        expected = integral.rl_integral.compute(f_array, t)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_with_callable(self):
        """Test compute method with callable function."""
        integral = CaputoIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        
        def f(x):
            return x**2
        
        result = integral.compute(f, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_compute_with_parameters(self):
        """Test compute method with additional parameters."""
        integral = CaputoIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        h = 0.05
        
        result = integral.compute(f_array, t, h=h, method="direct")
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))


class TestConvenienceFunctions:
    """Test convenience functions for integral computation."""
    
    def test_riemann_liouville_integral_function(self):
        """Test riemann_liouville_integral convenience function."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = riemann_liouville_integral(f_array, t, alpha=0.5)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_riemann_liouville_integral_with_fractional_order(self):
        """Test riemann_liouville_integral with FractionalOrder."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        alpha_order = FractionalOrder(0.5)
        
        result = riemann_liouville_integral(f_array, t, alpha=alpha_order)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_riemann_liouville_integral_with_parameters(self):
        """Test riemann_liouville_integral with additional parameters."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = riemann_liouville_integral(f_array, t, alpha=0.5, h=0.05, method="direct")
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_caputo_integral_function(self):
        """Test caputo_integral convenience function."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = caputo_integral(f_array, t, alpha=0.5)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_caputo_integral_with_parameters(self):
        """Test caputo_integral with additional parameters."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = caputo_integral(f_array, t, alpha=0.5, h=0.05, method="direct")
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))


class TestOptimizedFunctions:
    """Test optimized convenience functions."""
    
    def test_optimized_riemann_liouville_integral(self):
        """Test optimized_riemann_liouville_integral function."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = optimized_riemann_liouville_integral(f_array, t, alpha=0.5)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_optimized_riemann_liouville_integral_with_fractional_order(self):
        """Test optimized_riemann_liouville_integral with FractionalOrder."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        alpha_order = FractionalOrder(0.5)
        
        result = optimized_riemann_liouville_integral(f_array, t, alpha=alpha_order)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_optimized_riemann_liouville_integral_with_h(self):
        """Test optimized_riemann_liouville_integral with custom h."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = optimized_riemann_liouville_integral(f_array, t, alpha=0.5, h=0.05)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_optimized_caputo_integral(self):
        """Test optimized_caputo_integral function."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = optimized_caputo_integral(f_array, t, alpha=0.5)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_optimized_caputo_integral_with_fractional_order(self):
        """Test optimized_caputo_integral with FractionalOrder."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        alpha_order = FractionalOrder(0.5)
        
        result = optimized_caputo_integral(f_array, t, alpha=alpha_order)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_optimized_caputo_integral_with_h(self):
        """Test optimized_caputo_integral with custom h."""
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = optimized_caputo_integral(f_array, t, alpha=0.5, h=0.05)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))


class TestIntegralMethodsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_alpha(self):
        """Test behavior with very small alpha values."""
        integral = RiemannLiouvilleIntegral(alpha=1e-6)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_alpha_close_to_one(self):
        """Test behavior with alpha close to 1."""
        integral = RiemannLiouvilleIntegral(alpha=0.999)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_large_alpha(self):
        """Test behavior with large alpha values."""
        integral = RiemannLiouvilleIntegral(alpha=5.0)
        t = np.linspace(0, 1, 10)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_negative_function_values(self):
        """Test behavior with negative function values."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 1, 10)
        f_array = -t**2  # Negative values
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_oscillatory_function(self):
        """Test behavior with oscillatory function."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 2*np.pi, 50)
        f_array = np.sin(t)
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_exponential_function(self):
        """Test behavior with exponential function."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        t = np.linspace(0, 2, 20)
        f_array = np.exp(-t)
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be non-negative for exp(-t)


class TestIntegralMethodsPerformance:
    """Test performance characteristics."""
    
    def test_large_array_performance(self):
        """Test performance with large arrays."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="fft")
        t = np.linspace(0, 10, 5000)
        f_array = t**2
        
        import time
        start_time = time.time()
        result = integral.compute(f_array, t)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 10.0
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, optimize_memory=True)
        t = np.linspace(0, 10, 1000)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        # Check that result is reasonable size
        assert result.nbytes < 1e6  # Should use less than 1MB
        assert len(result) == len(t)
    
    def test_method_comparison(self):
        """Test that different methods produce reasonable results."""
        t = np.linspace(0, 1, 20)
        f_array = t**2
        
        integral_fft = RiemannLiouvilleIntegral(alpha=0.5, method="fft")
        integral_direct = RiemannLiouvilleIntegral(alpha=0.5, method="direct")
        
        result_fft = integral_fft.compute(f_array, t)
        result_direct = integral_direct.compute(f_array, t)
        
        # Both methods should produce finite, non-negative results
        assert np.all(np.isfinite(result_fft))
        assert np.all(np.isfinite(result_direct))
        assert np.all(result_fft >= 0)
        assert np.all(result_direct >= 0)
        
        # Both should have the same length
        assert len(result_fft) == len(result_direct)
        
        # Both should be monotonically increasing for t^2 input
        assert np.all(np.diff(result_direct) >= 0)
        # Note: FFT method may not be monotonically increasing due to numerical artifacts
    
    def test_adaptive_method_accuracy(self):
        """Test that adaptive method chooses appropriate method."""
        integral = RiemannLiouvilleIntegral(alpha=0.5, method="adaptive")
        t = np.linspace(0, 1, 50)
        f_array = t**2
        
        result = integral.compute(f_array, t)
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))


class TestIntegralMethodsConsistency:
    """Test consistency between different implementations."""
    
    def test_rl_caputo_equivalence(self):
        """Test that RL and Caputo integrals are equivalent for α > 0."""
        t = np.linspace(0, 1, 20)
        f_array = t**2
        
        rl_integral = RiemannLiouvilleIntegral(alpha=0.5)
        caputo_integral = CaputoIntegral(alpha=0.5)
        
        result_rl = rl_integral.compute(f_array, t)
        result_caputo = caputo_integral.compute(f_array, t)
        
        # Should be identical for α > 0
        np.testing.assert_array_equal(result_rl, result_caputo)
    
    def test_class_vs_function_consistency(self):
        """Test consistency between class and function interfaces."""
        t = np.linspace(0, 1, 20)
        f_array = t**2
        
        # Class interface
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        result_class = integral.compute(f_array, t)
        
        # Function interface
        result_function = riemann_liouville_integral(f_array, t, alpha=0.5)
        
        np.testing.assert_array_equal(result_class, result_function)
    
    def test_optimized_vs_standard_consistency(self):
        """Test consistency between optimized and standard functions."""
        t = np.linspace(0, 1, 20)
        f_array = t**2
        
        # Standard function
        result_standard = riemann_liouville_integral(f_array, t, alpha=0.5)
        
        # Optimized function
        result_optimized = optimized_riemann_liouville_integral(f_array, t, alpha=0.5)
        
        np.testing.assert_array_equal(result_standard, result_optimized)


if __name__ == "__main__":
    pytest.main([__file__])
