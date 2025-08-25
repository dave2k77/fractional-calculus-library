"""
Tests for utilities module.

This module contains comprehensive tests for all utility functions
including mathematical utilities, validation, performance monitoring,
and error handling.
"""

import pytest
import numpy as np
import time
import psutil
import os
from scipy.special import gamma
from unittest.mock import patch, MagicMock

from hpfracc.core.utilities import (
    # Mathematical utilities
    factorial_fractional,
    binomial_coefficient,
    pochhammer_symbol,
    hypergeometric_series,
    bessel_function_first_kind,
    modified_bessel_function_first_kind,
    
    # Type checking and validation
    validate_fractional_order,
    validate_function,
    validate_tensor_input,
    
    # Performance monitoring
    timing_decorator,
    memory_usage_decorator,
    PerformanceMonitor,
    
    # Error handling
    safe_divide,
    check_numerical_stability,
    
    # Common mathematical operations
    vectorize_function,
    normalize_array,
    smooth_function,
    fractional_power,
    fractional_exponential,
    
    # Configuration utilities
    get_default_precision,
    set_default_precision,
    get_available_methods,
    get_method_properties,
    
    # Logging utilities
    setup_logging,
    get_logger
)


class TestMathematicalUtilities:
    """Test mathematical utility functions."""
    
    def test_factorial_fractional(self):
        """Test fractional factorial function."""
        # Test integer values
        assert factorial_fractional(5) == 120
        assert factorial_fractional(0) == 1
        
        # Test fractional values
        result = factorial_fractional(0.5)
        expected = gamma(1.5)  # gamma(0.5 + 1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # Test negative values
        with pytest.raises(ValueError):
            factorial_fractional(-1)
    
    def test_binomial_coefficient(self):
        """Test binomial coefficient function."""
        # Test standard cases
        assert binomial_coefficient(5, 2) == 10
        assert binomial_coefficient(5, 0) == 1
        assert binomial_coefficient(5, 5) == 1
        
        # Test fractional cases
        result = binomial_coefficient(0.5, 2)
        expected = -0.125
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # Test invalid cases
        with pytest.raises(ValueError):
            binomial_coefficient(5, -1)
    
    def test_pochhammer_symbol(self):
        """Test Pochhammer symbol function."""
        # Test standard cases
        assert pochhammer_symbol(5, 3) == 210  # 5 * 6 * 7
        assert pochhammer_symbol(5, 0) == 1
        
        # Test fractional cases
        result = pochhammer_symbol(0.5, 2)
        expected = 0.75
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_hypergeometric_series(self):
        """Test hypergeometric series function."""
        # Test basic case
        result = interm = hypergeometric_series(1, 1, 1, 0.5, max_terms=10)
        assert np.isfinite(result)
        assert result > 0
        
        # Test convergence
        result = hypergeometric_series(1, 1, 1, 0.5, max_terms=5)
        # result2 = hypergeometric_series(1, 1, 1, 0.5, max_terms=10)
        assert abs(interm - result) < 1e-6
    
    def test_bessel_functions(self):
        """Test Bessel function implementations."""
        x = np.linspace(0.1, 5, 10)
        
        # Test Bessel function first kind
        j_result = bessel_function_first_kind(0, x[0])
        assert np.isfinite(j_result)
        
        # Test modified Bessel function first kind
        i_result = modified_bessel_function_first_kind(0, x[0])
        assert np.isfinite(i_result)


class TestTypeCheckingAndValidation:
    """Test type checking and validation functions."""
    
    def test_validate_fractional_order(self):
        """Test fractional order validation with error raising."""
        # Valid cases
        result = validate_fractional_order(0.5)
        assert result.alpha == 0.5
        result = validate_fractional_order(1.0)
        assert result.alpha == 1.0
        
        # Invalid cases
        with pytest.raises(ValueError):
            validate_fractional_order(-0.5)
    
    def test_validate_function(self):
        """Test function validation."""
        def valid_func(x):
            return x**2
        
        assert validate_function(valid_func)
        assert validate_function(lambda x: x**2)
        assert not validate_function("not a function")
    
    def test_validate_tensor_input(self):
        """Test tensor input validation."""
        # Valid arrays
        valid_array = np.array([1, 2, 3])
        assert validate_tensor_input(valid_array)
        
        # Invalid arrays
        assert not validate_tensor_input("not an array")


class TestPerformanceMonitoring:
    """Test performance monitoring decorators and class."""
    
    def test_timing_decorator(self):
        """Test timing decorator."""
        @timing_decorator
        def test_function():
            time.sleep(0.1)
            return "test"
        
        result = test_function()
        assert result == "test"
    
    def test_memory_usage_decorator(self):
        """Test memory usage decorator."""
        @memory_usage_decorator
        def test_function():
            # Create some memory usage
            large_array = np.zeros(1000000)
            return len(large_array)
        
        result = test_function()
        assert result == 1000000
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor class."""
        monitor = PerformanceMonitor()
        
        # Test context manager
        with monitor.timer("test_operation"):
            time.sleep(0.1)
        
        # Test memory monitoring
        with monitor.memory_tracker("test_memory"):
            large_array = np.zeros(1000000)
        
        # Test getting statistics
        stats = monitor.get_statistics()
        assert "test_operation" in stats
        assert "test_memory" in stats
        
        # Test reset
        monitor.reset()
        stats_after_reset = monitor.get_statistics()
        assert len(stats_after_reset) == 0


class TestErrorHandling:
    """Test error handling functions."""
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0  # Default value
        assert safe_divide(10, 0, default=1.0) == 1.0
    
    def test_check_numerical_stability(self):
        """Test numerical stability check."""
        stable_array = np.array([1.0, 2.0, 3.0])
        assert check_numerical_stability(stable_array)
        
        unstable_array = np.array([1e-20, 1e20, np.nan])
        assert not check_numerical_stability(unstable_array)


class TestCommonMathematicalOperations:
    """Test common mathematical operations."""
    
    def test_vectorize_function(self):
        """Test function vectorization."""
        def scalar_func(x):
            return x**2
        
        vectorized_func = vectorize_function(scalar_func)
        x = np.array([1, 2, 3, 4])
        result = vectorized_func(x)
        expected = x**2
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_normalize_array(self):
        """Test array normalization."""
        x = np.array([1, 2, 3, 4])
        normalized = normalize_array(x)
        assert np.all(np.isfinite(normalized))
        assert len(normalized) == len(x)
    
    def test_smooth_function(self):
        """Test function smoothing."""
        def noisy_func(x):
            return x**2 + 0.1 * np.random.random()
        
        smoothed_func = smooth_function(noisy_func)
        x = np.array([1.0])
        result = smoothed_func(x)
        assert np.isfinite(result)
    
    def test_fractional_power(self):
        """Test fractional power function."""
        x = np.array([1, 2, 3, 4])
        alpha = 0.5
        
        result = fractional_power(x, alpha)
        expected = x**alpha
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # Test with negative base
        x_neg = np.array([-1, -2, -3])
        result_neg = fractional_power(x_neg, alpha)
        assert np.all(np.isnan(result_neg))  # Should be NaN for negative base
    
    def test_fractional_exponential(self):
        """Test fractional exponential function."""
        x = np.array([0, 1, 2, 3])
        alpha = 0.5
        
        result = fractional_exponential(x, alpha)
        expected = np.exp(alpha * x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestConfigurationUtilities:
    """Test configuration utility functions."""
    
    def test_get_default_precision(self):
        """Test getting default precision."""
        precision = get_default_precision()
        assert isinstance(precision, int)
        assert precision > 0
    
    def test_set_default_precision(self):
        """Test setting default precision."""
        original_precision = get_default_precision()
        set_default_precision(64)
        assert get_default_precision() == 64
        set_default_precision(original_precision)  # Restore
    
    def test_get_available_methods(self):
        """Test getting available methods."""
        methods = get_available_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
    
    def test_get_method_properties(self):
        """Test getting method properties."""
        # Test with existing method
        properties = get_method_properties("riemann_liouville")
        assert isinstance(properties, dict)
        
        # Test with non-existing method
        properties = get_method_properties("non_existing_method")
        assert properties is None


class TestLoggingUtilities:
    """Test logging utility functions."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Test basic setup
        logger = setup_logging("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
        
        # Test with custom level
        logger = setup_logging("test_logger_custom", level="DEBUG")
        assert logger is not None
    
    def test_get_logger(self):
        """Test getting logger."""
        logger = get_logger("test_get_logger")
        assert logger is not None
        assert logger.name == "test_get_logger"
        
        # Test getting same logger multiple times
        logger2 = get_logger("test_get_logger")
        assert logger is logger2  # Should be the same object


class TestUtilitiesIntegration:
    """Test integration between utility functions."""
    
    def test_mathematical_utilities_integration(self):
        """Test integration of mathematical utilities."""
        # Test factorial and binomial coefficient
        n = 5
        k = 2
        factorial_n = factorial_fractional(n)
        binomial_nk = binomial_coefficient(n, k)
        
        # Verify relationship: C(n,k) = n! / (k! * (n-k)!)
        expected_binomial = factorial_n / (factorial_fractional(k) * factorial_fractional(n - k))
        np.testing.assert_allclose(binomial_nk, expected_binomial, rtol=1e-10)
    
    def test_validation_and_performance_integration(self):
        """Test integration of validation and performance monitoring."""
        monitor = PerformanceMonitor()
        
        @timing_decorator
        def validated_function(x):
            validate_fractional_order(x)
            return factorial_fractional(x)
        
        with monitor.timer("validated_operation"):
            result = validated_function(0.5)
        
        assert np.isfinite(result)
        stats = monitor.get_statistics()
        assert "validated_operation" in stats


class TestUtilitiesEdgeCases:
    """Test edge cases and error conditions for utilities."""
    
    def test_factorial_edge_cases(self):
        """Test factorial function edge cases."""
        # Test very small positive number
        result = factorial_fractional(1e-10)
        assert np.isfinite(result)
        
        # Test very large number
        with pytest.raises(OverflowError):
            factorial_fractional(1e10)
    
    def test_binomial_edge_cases(self):
        """Test binomial coefficient edge cases."""
        # Test n = k
        assert binomial_coefficient(5, 5) == 1
        
        # Test k = 0
        assert binomial_coefficient(5, 0) == 1
        
        # Test n < k
        with pytest.raises(ValueError):
            binomial_coefficient(2, 5)
    
    def test_bessel_edge_cases(self):
        """Test Bessel function edge cases."""
        # Test x = 0
        assert bessel_function_first_kind(0, 0) == 1
        assert modified_bessel_function_first_kind(0, 0) == 1
        
        # Test negative x
        x_neg = -1.0
        j_neg = bessel_function_first_kind(0, x_neg)
        assert np.isfinite(j_neg)
    
    def test_performance_monitor_edge_cases(self):
        """Test performance monitor edge cases."""
        monitor = PerformanceMonitor()
        
        # Test nested timers
        with monitor.timer("outer"):
            with monitor.timer("inner"):
                time.sleep(0.01)
        
        stats = monitor.get_statistics()
        assert "outer" in stats
        assert "inner" in stats
        
        # Test memory tracking with large arrays
        with monitor.memory_tracker("large_array"):
            large_array = np.zeros(10000000)  # 80MB array
        
        stats = monitor.get_statistics()
        assert "large_array" in stats


if __name__ == "__main__":
    pytest.main([__file__])
