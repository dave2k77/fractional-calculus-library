"""
Comprehensive test suite for analytical solutions to achieve 85%+ coverage.

This module tests all functionality in the hpfracc.validation.analytical_solutions module
including edge cases, error handling, and all code paths.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import validation module components
from hpfracc.validation.analytical_solutions import (
    AnalyticalSolutions,
    PowerFunctionSolutions,
    ExponentialSolutions,
    TrigonometricSolutions,
    get_analytical_solution,
    validate_against_analytical,
)


class TestAnalyticalSolutionsComprehensive:
    """Comprehensive test for AnalyticalSolutions functionality."""

    def test_power_function_derivative_edge_cases(self):
        """Test power function derivative edge cases."""
        solutions = AnalyticalSolutions()
        
        # Test alpha = 0 case
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.power_function_derivative(x, 0.0, 1.0)
        expected = np.zeros_like(x)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test alpha = 0 with order != 1
        result = solutions.power_function_derivative(x, 0.0, 0.5)
        expected = np.zeros_like(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_function_derivative_negative_alpha(self):
        """Test power function derivative with negative alpha."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        # Test with alpha = -0.5 (should work)
        result = solutions.power_function_derivative(x, -0.5, 0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        
        # Test with alpha = -2.0 (should raise ValueError)
        with pytest.raises(ValueError):
            solutions.power_function_derivative(x, -2.0, 0.5)

    def test_power_function_derivative_negative_order(self):
        """Test power function derivative with negative order."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            solutions.power_function_derivative(x, 2.0, -0.5)

    def test_exponential_derivative_comprehensive(self):
        """Test exponential derivative comprehensive cases."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        
        # Test with different coefficients
        for a in [0.5, 1.0, 2.0]:
            for order in [0.25, 0.5, 0.75]:
                result = solutions.exponential_derivative(x, a, order)
                assert isinstance(result, np.ndarray)
                assert len(result) == len(x)
                assert not np.any(np.isnan(result))

    def test_exponential_derivative_negative_order(self):
        """Test exponential derivative with negative order."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        
        with pytest.raises(ValueError):
            solutions.exponential_derivative(x, 1.0, -0.5)

    def test_trigonometric_derivative_comprehensive(self):
        """Test trigonometric derivative comprehensive cases."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        
        # Test different function types
        for func_type in ['sin', 'cos']:
            for omega in [0.5, 1.0, 2.0]:
                for order in [0.25, 0.5, 0.75]:
                    result = solutions.trigonometric_derivative(x, order, omega, func_type)
                    assert isinstance(result, np.ndarray)
                    assert len(result) == len(x)
                    assert not np.any(np.isnan(result))

    def test_trigonometric_derivative_negative_order(self):
        """Test trigonometric derivative with negative order."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        
        with pytest.raises(ValueError):
            solutions.trigonometric_derivative(x, -0.5, 1.0, 'sin')

    def test_constant_function_derivative_comprehensive(self):
        """Test constant function derivative comprehensive cases."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        # Test with different constants
        for c in [0.0, 1.0, 2.5, -1.0]:
            for order in [0.25, 0.5, 0.75, 1.0]:
                result = solutions.constant_function_derivative(x, order, c)
                assert isinstance(result, np.ndarray)
                assert len(result) == len(x)
                # For Caputo fractional derivatives, derivative of constant is zero for order > 0
                # Note: Riemann-Liouville derivatives of constants are non-zero: D^α(c) = c * x^(-α) / Γ(1-α)
                if order > 0:
                    np.testing.assert_array_almost_equal(result, np.zeros_like(x))

    def test_constant_function_derivative_negative_order(self):
        """Test constant function derivative with negative order."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            solutions.constant_function_derivative(x, -0.5, 1.0)


class TestPowerFunctionSolutionsComprehensive:
    """Comprehensive test for PowerFunctionSolutions functionality."""

    def test_get_solution_comprehensive(self):
        """Test get_solution comprehensive cases."""
        solutions = PowerFunctionSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        # Test various alpha and order combinations
        test_cases = [
            (0.5, 0.25), (1.0, 0.5), (1.5, 0.75), (2.0, 1.0),
            (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)
        ]
        
        for alpha, order in test_cases:
            result = solutions.get_solution(x, alpha, order)
            assert isinstance(result, np.ndarray)
            assert len(result) == len(x)
            assert not np.any(np.isnan(result))

    def test_get_test_cases_comprehensive(self):
        """Test get_test_cases comprehensive."""
        solutions = PowerFunctionSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        
        # Verify test case structure
        for test_case in test_cases:
            assert isinstance(test_case, dict)
            assert 'alpha' in test_case
            assert 'order' in test_case
            assert 'x' in test_case
            assert 'expected' in test_case


class TestExponentialSolutionsComprehensive:
    """Comprehensive test for ExponentialSolutions functionality."""

    def test_get_solution_comprehensive(self):
        """Test get_solution comprehensive cases."""
        solutions = ExponentialSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        
        # Test various coefficient and order combinations
        test_cases = [
            (0.5, 0.25), (1.0, 0.5), (1.5, 0.75), (2.0, 1.0),
            (0.0, 0.0), (1.0, 0.0)
        ]
        
        for a, order in test_cases:
            result = solutions.get_solution(x, a, order)
            assert isinstance(result, np.ndarray)
            assert len(result) == len(x)
            assert not np.any(np.isnan(result))

    def test_get_test_cases_comprehensive(self):
        """Test get_test_cases comprehensive."""
        solutions = ExponentialSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        
        # Verify test case structure
        for test_case in test_cases:
            assert isinstance(test_case, dict)
            assert 'a' in test_case
            assert 'order' in test_case
            assert 'x' in test_case
            assert 'expected' in test_case


class TestTrigonometricSolutionsComprehensive:
    """Comprehensive test for TrigonometricSolutions functionality."""

    def test_get_solution_comprehensive(self):
        """Test get_solution comprehensive cases."""
        solutions = TrigonometricSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        
        # Test various function types, omega, and order combinations
        test_cases = [
            ('sin', 0.5, 0.25), ('cos', 1.0, 0.5), ('tan', 1.5, 0.75),
            ('sin', 1.0, 0.0), ('cos', 2.0, 1.0)
        ]
        
        for func_type, omega, order in test_cases:
            result = solutions.get_solution(x, func_type, omega, order)
            assert isinstance(result, np.ndarray)
            assert len(result) == len(x)
            assert not np.any(np.isnan(result))

    def test_get_test_cases_comprehensive(self):
        """Test get_test_cases comprehensive."""
        solutions = TrigonometricSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0
        
        # Verify test case structure
        for test_case in test_cases:
            assert isinstance(test_case, dict)
            assert 'func_type' in test_case
            assert 'omega' in test_case
            assert 'order' in test_case
            assert 'x' in test_case
            assert 'expected' in test_case


class TestUtilityFunctionsComprehensive:
    """Comprehensive test for utility functions."""

    def test_get_analytical_solution_comprehensive(self):
        """Test get_analytical_solution comprehensive cases."""
        x = np.array([1.0, 2.0, 3.0])
        
        # Test different solution types
        solution_types = ['power', 'exponential', 'trigonometric', 'constant']
        
        for sol_type in solution_types:
            if sol_type == 'power':
                result = get_analytical_solution(sol_type, x, alpha=2.0, order=0.5)
            elif sol_type == 'exponential':
                result = get_analytical_solution(sol_type, x, a=1.0, order=0.5)
            elif sol_type == 'trigonometric':
                result = get_analytical_solution(sol_type, x, func_type='sin', omega=1.0, order=0.5)
            elif sol_type == 'constant':
                result = get_analytical_solution(sol_type, x, c=1.0, order=0.5)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(x)

    def test_get_analytical_solution_invalid_type(self):
        """Test get_analytical_solution with invalid type."""
        x = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            get_analytical_solution('invalid_type', x)

    def test_validate_against_analytical_comprehensive(self):
        """Test validate_against_analytical comprehensive cases."""
        def mock_method(x):
            return x**2 + 0.01
        
        def mock_analytical(x):
            return x**2
        
        # Test with different input sizes
        for n_points in [10, 50, 100]:
            x = np.linspace(0, 1, n_points)
            result = validate_against_analytical(mock_method, mock_analytical, x)
            assert isinstance(result, dict)
            assert 'n_points' in result
            assert 'results' in result
            assert result['n_points'] == n_points

    def test_validate_against_analytical_error_cases(self):
        """Test validate_against_analytical error cases."""
        def mock_method(x):
            return x**2 + 0.01
        
        def mock_analytical(x):
            return x**2
        
        # Test with single point
        x = np.array([1.0])
        result = validate_against_analytical(mock_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result

    def test_validate_against_analytical_failure_cases(self):
        """Test validate_against_analytical with failing methods."""
        def failing_method(x):
            raise RuntimeError("Method failed")
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = validate_against_analytical(failing_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result
        
        # Check that failures are recorded
        for res in result['results']:
            assert 'success' in res
            if not res['success']:
                assert 'error' in res


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_numpy_array_edge_cases(self):
        """Test with various numpy array edge cases."""
        solutions = AnalyticalSolutions()
        
        # Test with single element array
        x_single = np.array([1.0])
        result = solutions.power_function_derivative(x_single, 2.0, 0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        
        # Test with empty array
        x_empty = np.array([])
        result = solutions.power_function_derivative(x_empty, 2.0, 0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        
        # Test with 2D array (should be flattened)
        x_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = solutions.power_function_derivative(x_2d, 2.0, 0.5)
        assert isinstance(result, np.ndarray)

    def test_extreme_values(self):
        """Test with extreme values."""
        solutions = AnalyticalSolutions()
        
        # Test with very small values
        x_small = np.array([1e-10, 1e-8, 1e-6])
        result = solutions.power_function_derivative(x_small, 2.0, 0.5)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        
        # Test with very large values
        x_large = np.array([1e6, 1e8, 1e10])
        result = solutions.power_function_derivative(x_large, 2.0, 0.5)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))

    def test_special_mathematical_cases(self):
        """Test special mathematical cases."""
        solutions = AnalyticalSolutions()
        
        # Test with order = 0 (should return original function)
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.power_function_derivative(x, 2.0, 0.0)
        expected = x**2.0
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with order = 1 (should return first derivative)
        result = solutions.power_function_derivative(x, 2.0, 1.0)
        expected = 2.0 * x
        np.testing.assert_array_almost_equal(result, expected)

    def test_gamma_function_edge_cases(self):
        """Test gamma function edge cases."""
        solutions = AnalyticalSolutions()
        
        # Test with values that might cause gamma function issues
        x = np.array([1.0, 2.0, 3.0])
        
        # Test with alpha close to negative integers
        result = solutions.power_function_derivative(x, -0.1, 0.5)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))


if __name__ == '__main__':
    pytest.main([__file__])
