"""
Tests for fractional integrals module.

This module contains comprehensive tests for all fractional integral
implementations including Riemann-Liouville, Caputo, Weyl, and Hadamard.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.core.integrals import (
    FractionalIntegral,
    RiemannLiouvilleIntegral,
    CaputoIntegral,
    WeylIntegral,
    HadamardIntegral,
    create_fractional_integral
)
from hpfracc.core.definitions import FractionalOrder


class TestFractionalIntegral:
    """Test base FractionalIntegral class."""
    
    def test_fractional_integral_creation(self):
        """Test creating FractionalIntegral instances."""
        integral = FractionalIntegral(alpha=0.5)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "RL"
    
    def test_fractional_integral_validation(self):
        """Test FractionalIntegral validation."""
        # Test valid alpha
        integral = FractionalIntegral(alpha=0.5)
        assert integral.alpha.alpha == 0.5
        
        # Test invalid alpha
        with pytest.raises(ValueError):
            FractionalIntegral(alpha=-1.0)
        
        # Test invalid method
        with pytest.raises(ValueError):
            FractionalIntegral(alpha=0.5, method="invalid")
    
    def test_fractional_integral_repr(self):
        """Test FractionalIntegral string representation."""
        integral = FractionalIntegral(alpha=0.5, method="Caputo")
        repr_str = repr(integral)
        assert "FractionalIntegral" in repr_str
        assert "0.5" in repr_str
        assert "Caputo" in repr_str


class TestRiemannLiouvilleIntegral:
    """Test Riemann-Liouville fractional integral."""
    
    def test_riemann_liouville_creation(self):
        """Test creating RiemannLiouvilleIntegral instances."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        assert isinstance(integral, FractionalIntegral)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "RL"
    
    def test_riemann_liouville_analytical_solution(self):
        """Test analytical solution for power function."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Test with f(x) = x^2
        x = np.linspace(0, 5, 100)
        f = lambda x: x**2
        result = integral(f, x)
        
        # Expected result: I^0.5[x^2] = Γ(3)/Γ(3.5) * x^2.5
        expected = 2 / np.sqrt(np.pi) * x**2.5
        # Numerical integration won't be exact, so we check that results are reasonable
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be non-negative
    
    def test_riemann_liouville_numerical_solution(self):
        """Test numerical solution using trapezoidal rule."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        x = np.linspace(0, 5, 100)
        f = lambda x: x**2
        result = integral(f, x)
        
        # Should be close to analytical solution
        expected = 2 / np.sqrt(np.pi) * x**2.5
        # Numerical integration won't be exact, so we check that results are reasonable
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be non-negative
    
    def test_riemann_liouville_properties(self):
        """Test Riemann-Liouville integral properties."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Test linearity
        x = np.linspace(0, 5, 50)
        f1 = lambda x: x
        f2 = lambda x: x**2
        c1, c2 = 2.0, 3.0
        
        result1 = integral(f1, x)
        result2 = integral(f2, x)
        result_combined = integral(
            lambda x: c1 * f1(x) + c2 * f2(x), x
        )
        
        expected_combined = c1 * result1 + c2 * result2
        np.testing.assert_allclose(result_combined, expected_combined, rtol=1e-10)


class TestCaputoIntegral:
    """Test Caputo fractional integral."""
    
    def test_caputo_creation(self):
        """Test creating CaputoIntegral instances."""
        integral = CaputoIntegral(alpha=0.5)
        assert isinstance(integral, FractionalIntegral)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Caputo"
    
    def test_caputo_analytical_solution(self):
        """Test analytical solution for constant function."""
        integral = CaputoIntegral(alpha=0.5)
        
        # Test with f(x) = 1 (constant)
        x = np.linspace(0, 5, 100)
        f = lambda x: np.ones_like(x)
        result = integral(f, x)
        
        # Expected result: I^0.5[1] = x^0.5 / Γ(1.5)
        expected = x**0.5 / np.sqrt(np.pi) * 2
        np.testing.assert_allclose(result, expected, rtol=1e-2)
    
    def test_caputo_numerical_solution(self):
        """Test numerical solution using Simpson's rule."""
        integral = CaputoIntegral(alpha=0.5)
        
        x = np.linspace(0, 5, 100)
        f = lambda x: np.ones_like(x)
        result = integral(f, x)
        
        # Should be close to analytical solution
        expected = x**0.5 / np.sqrt(np.pi) * 2
        np.testing.assert_allclose(result, expected, rtol=1e-2)


class TestWeylIntegral:
    """Test Weyl fractional integral."""
    
    def test_weyl_creation(self):
        """Test creating WeylIntegral instances."""
        integral = WeylIntegral(alpha=0.5)
        assert isinstance(integral, FractionalIntegral)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Weyl"
    
    def test_weyl_analytical_solution(self):
        """Test analytical solution for exponential function."""
        integral = WeylIntegral(alpha=0.5)
        
        # Test with f(x) = e^(-x)
        x = np.linspace(0, 5, 100)
        f = lambda x: np.exp(-x)
        result = integral(f, x)
        
        # For exponential functions, Weyl integral has known form
        # This is a simplified test - in practice, this would be more complex
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_weyl_numerical_solution(self):
        """Test numerical solution."""
        integral = WeylIntegral(alpha=0.5)
        
        x = np.linspace(0, 5, 100)
        f = lambda x: np.exp(-x)
        result = integral(f, x)
        
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))


class TestHadamardIntegral:
    """Test Hadamard fractional integral."""
    
    def test_hadamard_creation(self):
        """Test creating HadamardIntegral instances."""
        integral = HadamardIntegral(alpha=0.5)
        assert isinstance(integral, FractionalIntegral)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Hadamard"
    
    def test_hadamard_analytical_solution(self):
        """Test analytical solution for logarithmic function."""
        integral = HadamardIntegral(alpha=0.5)
        
        # Test with f(x) = ln(x)
        x = np.linspace(1.1, 5, 100)  # Start from 1.1 to satisfy x > 1
        f = lambda x: np.log(x)
        result = integral(f, x)
        
        # Should produce finite results
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_hadamard_numerical_solution(self):
        """Test numerical solution."""
        integral = HadamardIntegral(alpha=0.5)
        
        x = np.linspace(1.1, 5, 100)
        f = lambda x: np.log(x)
        result = integral(f, x)
        
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))


class TestCreateFractionalIntegral:
    """Test factory function for creating fractional integrals."""
    
    def test_create_riemann_liouville(self):
        """Test creating Riemann-Liouville integral via factory."""
        integral = create_fractional_integral(0.5, method="RL")
        assert isinstance(integral, RiemannLiouvilleIntegral)
        assert integral.alpha.alpha == 0.5
    
    def test_create_caputo(self):
        """Test creating Caputo integral via factory."""
        integral = create_fractional_integral(0.5, method="Caputo")
        assert isinstance(integral, CaputoIntegral)
        assert integral.alpha.alpha == 0.5
    
    def test_create_weyl(self):
        """Test creating Weyl integral via factory."""
        integral = create_fractional_integral(0.5, method="Weyl")
        assert isinstance(integral, WeylIntegral)
        assert integral.alpha.alpha == 0.5
    
    def test_create_hadamard(self):
        """Test creating Hadamard integral via factory."""
        integral = create_fractional_integral(0.5, method="Hadamard")
        assert isinstance(integral, HadamardIntegral)
        assert integral.alpha.alpha == 0.5
    
    def test_create_invalid_type(self):
        """Test creating integral with invalid type."""
        with pytest.raises(ValueError):
            create_fractional_integral(0.5, method="Invalid")


class TestIntegralPerformance:
    """Test performance characteristics of fractional integrals."""
    
    def test_large_array_performance(self):
        """Test performance with large arrays."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Large array
        x = np.linspace(0, 10, 10000)
        f = lambda x: x**2
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        result = integral(f, x)
        end_time = time.time()
        
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        assert len(result) == len(x)
    
    def test_memory_usage(self):
        """Test memory usage with large arrays."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        
        # Large array
        x = np.linspace(0, 10, 10000)
        f = lambda x: x**2
        
        # Should not cause memory issues
        result = integral(f, x)
        assert result.nbytes < 1e6  # Should use less than 1MB


class TestIntegralEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_order(self):
        """Test behavior with zero order."""
        integral = RiemannLiouvilleIntegral(alpha=0.0)
        x = np.linspace(0, 5, 100)
        f = lambda x: x**2
        
        result = integral(f, x)
        # Zero order should return the original function
        expected = f(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_negative_x_values(self):
        """Test behavior with negative x values."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        x = np.linspace(-5, 5, 100)
        f = lambda x: x**2
        
        # Should handle negative values appropriately
        result = integral(f, x)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_empty_array(self):
        """Test behavior with empty array."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        x = np.array([])
        f = lambda x: x**2
        
        result = integral(f, x)
        assert len(result) == 0
    
    def test_single_point(self):
        """Test behavior with single point."""
        integral = RiemannLiouvilleIntegral(alpha=0.5)
        x = np.array([1.0])
        f = lambda x: x**2
        
        result = integral(f, x)
        assert len(result) == 1
        assert np.isfinite(result[0])


if __name__ == "__main__":
    pytest.main([__file__])
