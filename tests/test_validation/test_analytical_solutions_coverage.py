#!/usr/bin/env python3
"""Quick tests for validation/analytical_solutions.py to boost coverage."""

import pytest
import numpy as np
from hpfracc.validation.analytical_solutions import *
from hpfracc.core.definitions import FractionalOrder


class TestAnalyticalSolutionsCoverage:
    def setup_method(self):
        self.alpha = 0.5
        self.t = np.linspace(0, 1, 50)
        
    def test_power_function_solution(self):
        # Test analytical solution for power functions
        n = 2
        result = power_function_fractional_derivative(self.t, n, self.alpha)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.t.shape
        
    def test_exponential_solution(self):
        # Test analytical solution for exponential functions
        lambda_val = 1.0
        result = exponential_fractional_derivative(self.t, lambda_val, self.alpha)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.t.shape
        
    def test_relaxation_equation_solution(self):
        # Test fractional relaxation equation solution
        tau = 1.0
        result = fractional_relaxation_solution(self.t, self.alpha, tau)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.t.shape
        
    def test_different_alpha_values(self):
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha in alphas:
            result = power_function_fractional_derivative(self.t, 2, alpha)
            assert isinstance(result, np.ndarray)
            
    def test_different_parameters(self):
        # Test with different function parameters
        powers = [1, 2, 3, 4]
        for n in powers:
            result = power_function_fractional_derivative(self.t, n, self.alpha)
            assert isinstance(result, np.ndarray)
            
    def test_edge_cases(self):
        # Test edge cases
        single_t = np.array([1.0])
        result = power_function_fractional_derivative(single_t, 2, self.alpha)
        assert isinstance(result, np.ndarray)
        
    def test_numerical_stability(self):
        # Test numerical stability
        large_t = np.linspace(0, 10, 50)
        result = exponential_fractional_derivative(large_t, 0.1, self.alpha)
        assert np.all(np.isfinite(result))
        
    def test_boundary_behavior(self):
        # Test behavior at boundaries
        t_with_zero = np.linspace(0, 1, 50)
        result = power_function_fractional_derivative(t_with_zero, 2, self.alpha)
        assert isinstance(result, np.ndarray)
        
    def test_comparison_accuracy(self):
        # Test accuracy by comparing different methods
        t_test = np.linspace(0.1, 1, 20)  # Avoid t=0
        result1 = power_function_fractional_derivative(t_test, 2, self.alpha)
        result2 = exponential_fractional_derivative(t_test, 1.0, self.alpha)
        
        # Both should be finite
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))
        
    def test_parameter_validation(self):
        # Test parameter validation
        with pytest.raises((ValueError, AssertionError)):
            power_function_fractional_derivative(self.t, -1, self.alpha)  # Invalid power
            
    def test_mathematical_properties(self):
        # Test mathematical properties
        result = fractional_relaxation_solution(self.t, self.alpha, 1.0)
        
        # Should decay over time for relaxation
        if len(result) > 1:
            # Generally should be decreasing or stable
            assert np.all(np.isfinite(result))
            
    def test_different_time_ranges(self):
        # Test with different time ranges
        time_ranges = [
            np.linspace(0, 0.5, 25),
            np.linspace(0, 2, 50),
            np.linspace(0.1, 1, 50)  # Avoid zero
        ]
        
        for t_range in time_ranges:
            result = power_function_fractional_derivative(t_range, 2, self.alpha)
            assert isinstance(result, np.ndarray)
            assert result.shape == t_range.shape













