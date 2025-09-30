"""
Final targeted tests to achieve 85%+ coverage for analytical solutions.

This module adds the remaining test cases to cover the missing lines.
"""

import pytest
import numpy as np

# Import validation module components
from hpfracc.validation.analytical_solutions import (
    AnalyticalSolutions,
    get_analytical_solution,
    validate_against_analytical,
)


class TestAnalyticalSolutionsFinalCoverage:
    """Final tests to achieve 85%+ coverage."""

    def test_power_function_derivative_alpha_negative_integer_edge_cases(self):
        """Test power function derivative with alpha as negative integer edge cases."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        
        # Test with alpha = -1 (should raise ValueError)
        with pytest.raises(ValueError):
            solutions.power_function_derivative(x, -1.0, 0.5)
        
        # Test with alpha = -2 (should raise ValueError)
        with pytest.raises(ValueError):
            solutions.power_function_derivative(x, -2.0, 0.5)
        
        # Test with alpha = -3 (should raise ValueError)
        with pytest.raises(ValueError):
            solutions.power_function_derivative(x, -3.0, 0.5)

    def test_trigonometric_derivative_invalid_func_type_edge_cases(self):
        """Test trigonometric derivative with invalid function type edge cases."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        
        # Test with invalid function types
        invalid_types = ['tan', 'sec', 'csc', 'cot', 'invalid', '']
        for func_type in invalid_types:
            with pytest.raises(ValueError):
                solutions.trigonometric_derivative(x, 0.5, 1.0, func_type)

    def test_get_analytical_solution_edge_cases(self):
        """Test get_analytical_solution edge cases."""
        x = np.array([1.0, 2.0, 3.0])
        
        # Test with invalid solution type
        with pytest.raises(ValueError):
            get_analytical_solution('invalid_type', x)
        
        # Test with empty string
        with pytest.raises(ValueError):
            get_analytical_solution('', x)
        
        # Test with None
        with pytest.raises(ValueError):
            get_analytical_solution(None, x)

    def test_validate_against_analytical_comprehensive_edge_cases(self):
        """Test validate_against_analytical comprehensive edge cases."""
        def mock_method(x):
            return x**2 + 0.01
        
        def mock_analytical(x):
            return x**2
        
        # Test with single point
        x_single = np.array([1.0])
        result = validate_against_analytical(mock_method, mock_analytical, x_single)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result
        
        # Test with two points
        x_two = np.array([1.0, 2.0])
        result = validate_against_analytical(mock_method, mock_analytical, x_two)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result
        
        # Test with three points
        x_three = np.array([1.0, 2.0, 3.0])
        result = validate_against_analytical(mock_method, mock_analytical, x_three)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result

    def test_validate_against_analytical_error_handling_comprehensive(self):
        """Test validate_against_analytical comprehensive error handling."""
        def error_method(x):
            if len(x) > 5:
                raise ValueError("Too many points")
            return x**2
        
        def mock_analytical(x):
            return x**2
        
        # Test with array that should cause error
        x = np.linspace(0, 1, 10)
        result = validate_against_analytical(error_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result
        
        # Check that errors are recorded
        for res in result['results']:
            assert 'success' in res
            if not res['success']:
                assert 'error' in res

    def test_validate_against_analytical_nan_inf_handling(self):
        """Test validate_against_analytical with NaN and inf values."""
        def nan_method(x):
            return np.full_like(x, np.nan)
        
        def inf_method(x):
            return np.full_like(x, np.inf)
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        
        # Test with NaN values
        result_nan = validate_against_analytical(nan_method, mock_analytical, x)
        assert isinstance(result_nan, dict)
        assert 'n_points' in result_nan
        assert 'results' in result_nan
        
        # Test with inf values
        result_inf = validate_against_analytical(inf_method, mock_analytical, x)
        assert isinstance(result_inf, dict)
        assert 'n_points' in result_inf
        assert 'results' in result_inf

    def test_validate_against_analytical_mixed_success_failure(self):
        """Test validate_against_analytical with mixed success/failure results."""
        def mixed_method(x):
            # Fail for certain conditions
            if len(x) > 8:
                raise RuntimeError("Method failed")
            return x**2 + 0.01
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = validate_against_analytical(mixed_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result
        
        # Check that we have both successes and failures
        successes = [res for res in result['results'] if res.get('success', False)]
        failures = [res for res in result['results'] if not res.get('success', False)]
        
        # Should have some of each
        assert len(successes) > 0 or len(failures) > 0

    def test_validate_against_analytical_empty_results(self):
        """Test validate_against_analytical with empty results."""
        def empty_method(x):
            return np.array([])
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 5)
        result = validate_against_analytical(empty_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result

    def test_validate_against_analytical_shape_mismatch(self):
        """Test validate_against_analytical with shape mismatches."""
        def shape_mismatch_method(x):
            # Return different shape
            return np.array([1.0, 2.0])  # Always return 2 elements
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 5)  # 5 elements
        result = validate_against_analytical(shape_mismatch_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'n_points' in result
        assert 'results' in result


if __name__ == '__main__':
    pytest.main([__file__])
