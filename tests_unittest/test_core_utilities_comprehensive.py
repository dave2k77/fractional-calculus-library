"""
Comprehensive tests for hpfracc/core/utilities.py
Testing all mathematical utilities, validation functions, performance monitoring, and error handling
"""

import unittest
import numpy as np
import torch
import logging
import warnings
from unittest.mock import patch, MagicMock
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.utilities import (
    # Mathematical utilities
    factorial_fractional, binomial_coefficient, pochhammer_symbol, 
    _hypergeometric_series_impl, hypergeometric_series,
    bessel_function_first_kind, modified_bessel_function_first_kind,
    
    # Validation utilities
    validate_fractional_order, validate_function, validate_tensor_input,
    
    # Performance monitoring
    timing_decorator, memory_usage_decorator, PerformanceMonitor,
    TimerContext, MemoryTrackerContext,
    
    # Error handling
    FractionalCalculusError, ConvergenceError, ValidationError,
    safe_divide, check_numerical_stability,
    
    # Mathematical operations
    vectorize_function, normalize_array, smooth_function,
    fractional_power, fractional_exponential,
    
    # Configuration utilities
    get_default_precision, set_default_precision, get_available_methods,
    get_method_properties, setup_logging, get_logger
)


class TestMathematicalUtilities(unittest.TestCase):
    """Test mathematical utility functions"""
    
    def test_factorial_fractional(self):
        """Test factorial_fractional function"""
        # Test integer values
        self.assertEqual(factorial_fractional(5), 120.0)
        self.assertEqual(factorial_fractional(0), 1.0)
        
        # Test fractional values
        result = factorial_fractional(2.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        # Test overflow
        with self.assertRaises(OverflowError):
            factorial_fractional(1e7)
        
        # Test invalid values
        with self.assertRaises(ValueError):
            factorial_fractional(-1)
        
        # Test that -0.5 works (gamma is defined for values > -1)
        result = factorial_fractional(-0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_binomial_coefficient(self):
        """Test binomial_coefficient function"""
        # Test basic cases
        self.assertEqual(binomial_coefficient(5, 2), 10.0)
        self.assertEqual(binomial_coefficient(5, 0), 1.0)
        self.assertEqual(binomial_coefficient(5, 5), 1.0)
        
        # Test fractional values
        result = binomial_coefficient(2.5, 1.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        # Test edge cases
        self.assertEqual(binomial_coefficient(5, 0), 1.0)
        
        # Test error cases
        with self.assertRaises(ValueError):
            binomial_coefficient(5, -1)
        
        with self.assertRaises(ValueError):
            binomial_coefficient(2, 5)  # n < k for integers
    
    def test_pochhammer_symbol(self):
        """Test pochhammer_symbol function"""
        # Test basic cases
        self.assertEqual(pochhammer_symbol(2, 0), 1.0)
        self.assertEqual(pochhammer_symbol(2, 1), 2.0)
        self.assertEqual(pochhammer_symbol(2, 3), 24.0)  # 2*3*4
        
        # Test fractional values
        result = pochhammer_symbol(2.5, 3)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_hypergeometric_series_impl(self):
        """Test _hypergeometric_series_impl function"""
        # Test basic hypergeometric series
        result = _hypergeometric_series_impl([1], [1], 0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        # Test with single values
        result = _hypergeometric_series_impl(1, 1, 0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        # Test convergence
        result = _hypergeometric_series_impl([1], [1], 0.1, max_terms=50)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_hypergeometric_series(self):
        """Test hypergeometric_series function"""
        # Test basic functionality
        result = hypergeometric_series([1], [1], 0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        # Test with single values
        result = hypergeometric_series(1, 1, 0.5)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_bessel_function_first_kind(self):
        """Test bessel_function_first_kind function"""
        # Test at x=0
        self.assertEqual(bessel_function_first_kind(0, 0), 1.0)
        self.assertEqual(bessel_function_first_kind(1, 0), 0.0)
        
        # Test at x>0
        result = bessel_function_first_kind(0, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        result = bessel_function_first_kind(1, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
    
    def test_modified_bessel_function_first_kind(self):
        """Test modified_bessel_function_first_kind function"""
        # Test at x=0
        self.assertEqual(modified_bessel_function_first_kind(0, 0), 1.0)
        self.assertEqual(modified_bessel_function_first_kind(1, 0), 0.0)
        
        # Test at x>0
        result = modified_bessel_function_first_kind(0, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        
        result = modified_bessel_function_first_kind(1, 1.0)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))


class TestValidationUtilities(unittest.TestCase):
    """Test validation utility functions"""
    
    def test_validate_fractional_order(self):
        """Test validate_fractional_order function"""
        # Test with float
        result = validate_fractional_order(0.5)
        self.assertIsInstance(result, FractionalOrder)
        self.assertEqual(result.alpha, 0.5)
        
        # Test with FractionalOrder
        alpha = FractionalOrder(0.3)
        result = validate_fractional_order(alpha)
        self.assertIsInstance(result, FractionalOrder)
        self.assertEqual(result.alpha, 0.3)
        
        # Test with custom range
        result = validate_fractional_order(1.5, min_val=1.0, max_val=2.0)
        self.assertIsInstance(result, FractionalOrder)
        self.assertEqual(result.alpha, 1.5)
        
        # Test validation errors
        with self.assertRaises(ValueError):
            validate_fractional_order(2.5, min_val=0.0, max_val=2.0)
        
        with self.assertRaises(ValueError):
            validate_fractional_order(-0.5, min_val=0.0, max_val=2.0)
    
    def test_validate_function(self):
        """Test validate_function function"""
        # Test valid function
        def valid_func(x):
            return x**2
        
        self.assertTrue(validate_function(valid_func))
        
        # Test function with domain validation
        self.assertTrue(validate_function(valid_func, domain=(0, 1), n_points=10))
        
        # Test invalid function (non-callable)
        self.assertFalse(validate_function("not a function"))
        
        # Test function that returns non-finite values
        def invalid_func(x):
            return np.inf * np.ones_like(x)
        
        self.assertFalse(validate_function(invalid_func))
        
        # Test function that raises exception
        def error_func(x):
            raise ValueError("Test error")
        
        self.assertFalse(validate_function(error_func))
    
    def test_validate_tensor_input(self):
        """Test validate_tensor_input function"""
        # Test valid numpy array
        valid_array = np.array([1.0, 2.0, 3.0])
        self.assertTrue(validate_tensor_input(valid_array))
        
        # Test valid torch tensor
        valid_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(validate_tensor_input(valid_tensor))
        
        # Test with expected shape
        self.assertTrue(validate_tensor_input(valid_array, expected_shape=(3,)))
        self.assertFalse(validate_tensor_input(valid_array, expected_shape=(2,)))
        
        # Test invalid inputs
        invalid_array = np.array([1.0, np.inf, 3.0])
        self.assertFalse(validate_tensor_input(invalid_array))
        
        invalid_tensor = torch.tensor([1.0, float('inf'), 3.0])
        self.assertFalse(validate_tensor_input(invalid_tensor))
        
        # Test non-tensor input
        self.assertFalse(validate_tensor_input([1, 2, 3]))


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring utilities"""
    
    def test_timing_decorator(self):
        """Test timing_decorator function"""
        @timing_decorator
        def test_func(x):
            return x * 2
        
        # Test that decorator works and returns correct result
        result = test_func(5)
        self.assertEqual(result, 10)
        
        # Test that decorator preserves function name
        self.assertEqual(test_func.__name__, "test_func")
    
    def test_memory_usage_decorator(self):
        """Test memory_usage_decorator function"""
        @memory_usage_decorator
        def test_func(x):
            return x * 2
        
        # Test that decorator works and returns correct result
        result = test_func(5)
        self.assertEqual(result, 10)
        
        # Test that decorator preserves function name
        self.assertEqual(test_func.__name__, "test_func")
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor class"""
        monitor = PerformanceMonitor()
        
        # Test timer functionality
        monitor.start_timer("test_operation")
        import time
        time.sleep(0.001)  # Small delay
        duration = monitor.end_timer("test_operation")
        
        self.assertIsInstance(duration, float)
        self.assertGreater(duration, 0)
        
        # Test statistics
        stats = monitor.get_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("test_operation", stats)
        
        # Test reset
        monitor.reset()
        stats_after_reset = monitor.get_statistics()
        self.assertEqual(len(stats_after_reset), 0)
    
    def test_timer_context(self):
        """Test TimerContext class"""
        monitor = PerformanceMonitor()
        
        with monitor.timer("context_test") as timer:
            import time
            time.sleep(0.001)
        
        # Check that timing was recorded
        stats = monitor.get_statistics()
        self.assertIn("context_test", stats)
    
    def test_memory_tracker_context(self):
        """Test MemoryTrackerContext class"""
        monitor = PerformanceMonitor()
        
        with monitor.memory_tracker("memory_test") as tracker:
            # Create some memory usage
            data = np.random.random(1000)
        
        # Check that memory tracking was recorded
        stats = monitor.get_statistics()
        self.assertIn("memory_test", stats)


class TestErrorHandling(unittest.TestCase):
    """Test error handling utilities"""
    
    def test_custom_exceptions(self):
        """Test custom exception classes"""
        # Test FractionalCalculusError
        with self.assertRaises(FractionalCalculusError):
            raise FractionalCalculusError("Test error")
        
        # Test ConvergenceError
        with self.assertRaises(ConvergenceError):
            raise ConvergenceError("Convergence failed")
        
        # Test ValidationError
        with self.assertRaises(ValidationError):
            raise ValidationError("Validation failed")
    
    def test_safe_divide(self):
        """Test safe_divide function"""
        # Test normal division
        result = safe_divide(10, 2)
        self.assertEqual(result, 5.0)
        
        # Test division by zero with default
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_divide(10, 0, default=999)
            self.assertEqual(result, 999)
        
        # Test division by very small number
        result = safe_divide(10, 1e-15, default=0)
        self.assertEqual(result, 0)
    
    def test_check_numerical_stability(self):
        """Test check_numerical_stability function"""
        # Test stable numpy array
        stable_array = np.array([1.0, 2.0, 3.0])
        self.assertTrue(check_numerical_stability(stable_array))
        
        # Test stable torch tensor
        stable_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(check_numerical_stability(stable_tensor))
        
        # Test unstable arrays
        unstable_array = np.array([1e20, 2.0, 3.0])
        self.assertFalse(check_numerical_stability(unstable_array))
        
        unstable_tensor = torch.tensor([float('inf'), 2.0, 3.0])
        self.assertFalse(check_numerical_stability(unstable_tensor))
        
        # Test non-tensor input
        self.assertFalse(check_numerical_stability([1, 2, 3]))


class TestMathematicalOperations(unittest.TestCase):
    """Test mathematical operation utilities"""
    
    def test_vectorize_function(self):
        """Test vectorize_function utility"""
        def square(x):
            return x ** 2
        
        # Test with numpy vectorize
        vectorized = vectorize_function(square, vectorize=True)
        result = vectorized([1, 2, 3, 4])
        expected = np.array([1, 4, 9, 16])
        np.testing.assert_array_equal(result, expected)
        
        # Test without numpy vectorize
        vectorized = vectorize_function(square, vectorize=False)
        result = vectorized([1, 2, 3, 4])
        expected = np.array([1, 4, 9, 16])
        np.testing.assert_array_equal(result, expected)
        
        # Test with numpy array
        result = vectorized(np.array([1, 2, 3, 4]))
        expected = np.array([1, 4, 9, 16])
        np.testing.assert_array_equal(result, expected)
        
        # Test with torch tensor
        result = vectorized(torch.tensor([1, 2, 3, 4]))
        expected = torch.tensor([1, 4, 9, 16], dtype=torch.float32)
        torch.testing.assert_close(result, expected)
    
    def test_normalize_array(self):
        """Test normalize_array function"""
        # Test numpy array normalization
        arr = np.array([1, 2, 3, 4])
        
        # Test L2 normalization
        normalized = normalize_array(arr, "l2")
        expected_norm = np.sqrt(np.sum(normalized ** 2))
        self.assertAlmostEqual(expected_norm, 1.0, places=10)
        
        # Test L1 normalization
        normalized = normalize_array(arr, "l1")
        expected_norm = np.sum(np.abs(normalized))
        self.assertAlmostEqual(expected_norm, 1.0, places=10)
        
        # Test max normalization
        normalized = normalize_array(arr, "max")
        expected_max = np.max(np.abs(normalized))
        self.assertAlmostEqual(expected_max, 1.0, places=10)
        
        # Test minmax normalization
        normalized = normalize_array(arr, "minmax")
        self.assertAlmostEqual(np.min(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=10)
        
        # Test torch tensor normalization
        tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        normalized = normalize_array(tensor, "l2")
        expected_norm = torch.sqrt(torch.sum(normalized ** 2))
        self.assertAlmostEqual(expected_norm.item(), 1.0, places=5)
        
        # Test error cases
        with self.assertRaises(ValueError):
            normalize_array(arr, "invalid")
        
        with self.assertRaises(TypeError):
            normalize_array([1, 2, 3], "l2")
    
    def test_smooth_function(self):
        """Test smooth_function utility"""
        def noisy_func(x):
            if isinstance(x, (int, float)):
                return x + np.random.random() * 0.1
            else:
                return x + np.random.random(len(x)) * 0.1
        
        smoothed = smooth_function(noisy_func, smoothing_factor=0.1)
        
        # Test with scalar input
        result = smoothed(5.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array input
        x = np.linspace(0, 10, 100)
        result = smoothed(x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(x))
    
    def test_fractional_power(self):
        """Test fractional_power function"""
        # Test with scalar values
        result = fractional_power(4, 0.5)
        self.assertEqual(result, 2.0)
        
        # Test negative values with non-integer alpha
        result = fractional_power(-4, 0.5)
        self.assertTrue(np.isnan(result))
        
        # Test negative values with integer alpha
        result = fractional_power(-4, 2)
        self.assertEqual(result, 16.0)
        
        # Test numpy array
        x = np.array([1, 4, 9, 16])
        result = fractional_power(x, 0.5)
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)
        
        # Test torch tensor
        x = torch.tensor([1, 4, 9, 16], dtype=torch.float32)
        result = fractional_power(x, 0.5)
        expected = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        torch.testing.assert_close(result, expected)
        
        # Test error case
        with self.assertRaises(TypeError):
            fractional_power("invalid", 0.5)
    
    def test_fractional_exponential(self):
        """Test fractional_exponential function"""
        # Test with scalar values
        result = fractional_exponential(1.0, 2.0)
        expected = np.exp(2.0)
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test numpy array
        x = np.array([1, 2, 3])
        result = fractional_exponential(x, 2.0)
        expected = np.exp(2.0 * x)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test torch tensor
        x = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = fractional_exponential(x, 2.0)
        expected = torch.exp(2.0 * x)
        torch.testing.assert_close(result, expected)
        
        # Test error case
        with self.assertRaises(TypeError):
            fractional_exponential("invalid", 2.0)


class TestConfigurationUtilities(unittest.TestCase):
    """Test configuration utility functions"""
    
    def test_precision_functions(self):
        """Test precision-related functions"""
        # Test get default precision
        precision = get_default_precision()
        self.assertEqual(precision, 64)
        
        # Test set precision with valid values
        set_default_precision(32)
        set_default_precision(64)
        set_default_precision(128)
        
        # Test set precision with invalid value
        with self.assertRaises(ValueError):
            set_default_precision(16)
    
    def test_method_functions(self):
        """Test method-related functions"""
        # Test get available methods
        methods = get_available_methods()
        self.assertIsInstance(methods, list)
        self.assertIn("RL", methods)
        self.assertIn("Caputo", methods)
        
        # Test get method properties
        properties = get_method_properties("RL")
        self.assertIsInstance(properties, dict)
        self.assertIn("full_name", properties)
        self.assertEqual(properties["full_name"], "Riemann-Liouville")
        
        # Test with invalid method
        properties = get_method_properties("invalid")
        self.assertIsNone(properties)
    
    def test_logging_functions(self):
        """Test logging utility functions"""
        # Test setup_logging with valid level
        logger = setup_logging("test_logger", level="INFO")
        self.assertIsInstance(logger, logging.Logger)
        
        # Test setup_logging with invalid level (should default to INFO)
        logger = setup_logging("test_logger2", level="INVALID")
        self.assertIsInstance(logger, logging.Logger)
        
        # Test get_logger
        logger = get_logger("test_logger3")
        self.assertIsInstance(logger, logging.Logger)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""
    
    def test_mathematical_functions_integration(self):
        """Test integration of mathematical functions"""
        # Test factorial and binomial coefficient relationship
        n = 5
        k = 2
        fact_n = factorial_fractional(n)
        fact_k = factorial_fractional(k)
        fact_nk = factorial_fractional(n - k)
        
        binomial = binomial_coefficient(n, k)
        expected = fact_n / (fact_k * fact_nk)
        self.assertAlmostEqual(binomial, expected, places=10)
    
    def test_validation_and_performance_integration(self):
        """Test integration of validation and performance monitoring"""
        monitor = PerformanceMonitor()
        
        def test_function(x):
            return x ** 2
        
        # Validate function first
        self.assertTrue(validate_function(test_function))
        
        # Then time it
        with monitor.timer("validation_test"):
            result = test_function(5)
        
        self.assertEqual(result, 25)
        
        # Check timing was recorded
        stats = monitor.get_statistics()
        self.assertIn("validation_test", stats)
    
    def test_error_handling_integration(self):
        """Test integration of error handling utilities"""
        # Test safe division in mathematical operations
        def safe_power(x, alpha):
            # Use safe division to avoid division by zero
            if alpha == 0:
                return 1.0
            else:
                return safe_divide(x ** alpha, 1.0, default=0.0)
        
        result = safe_power(2, 0)
        self.assertEqual(result, 1.0)
        
        # Test numerical stability check
        values = np.array([1.0, 2.0, 3.0])
        self.assertTrue(check_numerical_stability(values))
    
    def test_vectorization_and_normalization_integration(self):
        """Test integration of vectorization and normalization"""
        def complex_function(x):
            return x ** 2 + np.sin(x)
        
        # Vectorize the function
        vectorized = vectorize_function(complex_function, vectorize=False)
        
        # Test with array input
        x = np.linspace(0, np.pi, 10)
        result = vectorized(x)
        
        # Normalize the result
        normalized = normalize_array(result, "l2")
        
        # Check that result is normalized
        norm = np.sqrt(np.sum(normalized ** 2))
        self.assertAlmostEqual(norm, 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
