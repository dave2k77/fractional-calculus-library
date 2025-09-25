
"""
Additional validation tests to improve coverage.
"""

import pytest
import numpy as np
from hpfracc.validation.analytical_solutions import AnalyticalSolutions
from hpfracc.validation.convergence_tests import ConvergenceTester
from hpfracc.validation.benchmarks import BenchmarkSuite

class TestAdditionalValidation:
    """Additional tests for validation modules."""
    
    def test_analytical_solutions_edge_cases(self):
        """Test analytical solutions with edge cases."""
        solutions = AnalyticalSolutions()
        
        # Test with very small arrays
        x = np.array([0.1])
        result = solutions.power_function_derivative(x, alpha=1.0, order=0.5)
        assert len(result) == 1
        assert np.isfinite(result[0])
        
        # Test with large alpha values
        x = np.linspace(0.1, 1, 10)
        result = solutions.power_function_derivative(x, alpha=5.0, order=0.5)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_convergence_tester_additional(self):
        """Test convergence tester with additional scenarios."""
        tester = ConvergenceTester()
        
        # Test with different grid sizes
        def simple_func(x, **kwargs):
            return x**2
        
        grid_sizes = [10, 20, 40]
        errors = [0.1, 0.05, 0.025]
        
        rate = tester.estimate_convergence_rate(grid_sizes, errors)
        assert rate > 0
    
    def test_benchmark_suite_comprehensive(self):
        """Test benchmark suite with comprehensive scenarios."""
        suite = BenchmarkSuite()
        
        def mock_method(x, **kwargs):
            return x**1.5
        
        def mock_analytical(x, **kwargs):
            return x**1.5
        
        methods = {"test": mock_method}
        test_cases = [{"x": np.linspace(0, 1, 50)}]
        
        results = suite.run_comprehensive_benchmark(
            methods, mock_analytical, test_cases, n_runs=1
        )
        
        assert "accuracy_results" in results
        assert "performance_results" in results
