"""
Comprehensive test suite for the validation module.

This module tests all functionality in the hpfracc.validation module including:
- Analytical solutions for fractional calculus
- Convergence tests and analysis
- Benchmarking tools for performance and accuracy
"""

import pytest
import numpy as np
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import validation module components
from hpfracc.validation import (
    # Analytical solutions
    AnalyticalSolutions,
    PowerFunctionSolutions,
    ExponentialSolutions,
    TrigonometricSolutions,
    get_analytical_solution,
    validate_against_analytical,
    # Convergence tests
    ConvergenceTester,
    ConvergenceAnalyzer,
    OrderOfAccuracy,
    run_convergence_study,
    run_method_convergence_test,
    estimate_convergence_rate,
    # Benchmarks
    BenchmarkSuite,
    PerformanceBenchmark,
    AccuracyBenchmark,
    run_benchmarks,
    compare_methods,
    generate_benchmark_report,
)


class TestAnalyticalSolutions:
    """Test AnalyticalSolutions functionality."""

    def test_analytical_solutions_creation(self):
        """Test AnalyticalSolutions creation."""
        solutions = AnalyticalSolutions()
        assert hasattr(solutions, 'power_function_derivative')

    def test_power_function_derivative(self):
        """Test power function derivative computation."""
        solutions = AnalyticalSolutions()
        
        # Test with simple case
        x = np.array([1.0, 2.0, 3.0])
        alpha = 2.0
        order = 0.5
        
        result = solutions.power_function_derivative(x, alpha, order)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_power_function_derivative_special_cases(self):
        """Test power function derivative special cases."""
        solutions = AnalyticalSolutions()
        
        # Test order = 0 (should return original function)
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.power_function_derivative(x, 2.0, 0.0)
        expected = x**2.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_function_derivative_validation(self):
        """Test power function derivative validation."""
        solutions = AnalyticalSolutions()
        
        # Test invalid alpha
        with pytest.raises(ValueError):
            solutions.power_function_derivative(np.array([1.0]), -2.0, 0.5)
        
        # Test invalid order
        with pytest.raises(ValueError):
            solutions.power_function_derivative(np.array([1.0]), 2.0, -0.5)

    def test_exponential_derivative(self):
        """Test exponential function derivative."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        result = solutions.exponential_derivative(x, 0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_trigonometric_derivative(self):
        """Test trigonometric function derivative."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        result = solutions.trigonometric_derivative(x, 0.5, 1.0, 'sin')
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestPowerFunctionSolutions:
    """Test PowerFunctionSolutions functionality."""

    def test_power_function_solutions_creation(self):
        """Test PowerFunctionSolutions creation."""
        solutions = PowerFunctionSolutions()
        assert hasattr(solutions, 'get_derivative')

    def test_get_derivative(self):
        """Test derivative computation."""
        solutions = PowerFunctionSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        alpha = 2.0
        order = 0.5
        
        result = solutions.get_derivative(x, alpha, order)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestExponentialSolutions:
    """Test ExponentialSolutions functionality."""

    def test_exponential_solutions_creation(self):
        """Test ExponentialSolutions creation."""
        solutions = ExponentialSolutions()
        assert hasattr(solutions, 'get_derivative')

    def test_get_derivative(self):
        """Test exponential derivative computation."""
        solutions = ExponentialSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        order = 0.5
        
        result = solutions.get_derivative(x, order)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestTrigonometricSolutions:
    """Test TrigonometricSolutions functionality."""

    def test_trigonometric_solutions_creation(self):
        """Test TrigonometricSolutions creation."""
        solutions = TrigonometricSolutions()
        assert hasattr(solutions, 'get_derivative')

    def test_get_derivative(self):
        """Test trigonometric derivative computation."""
        solutions = TrigonometricSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        order = 0.5
        func_type = 'sin'
        
        result = solutions.get_derivative(x, order, func_type)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestConvergenceTester:
    """Test ConvergenceTester functionality."""

    def test_convergence_tester_creation(self):
        """Test ConvergenceTester creation."""
        tester = ConvergenceTester()
        assert hasattr(tester, 'tolerance')
        assert tester.tolerance == 1e-10

    def test_convergence_tester_custom_tolerance(self):
        """Test ConvergenceTester with custom tolerance."""
        tester = ConvergenceTester(tolerance=1e-6)
        assert tester.tolerance == 1e-6

    def test_test_convergence(self):
        """Test convergence testing."""
        tester = ConvergenceTester()
        
        # Create mock functions
        def mock_method(x, h):
            return x**2 + 0.1 * h  # Simulate numerical method with error
        
        def mock_analytical(x):
            return x**2  # Analytical solution
        
        grid_sizes = [10, 20, 40]
        test_params = {'x': np.linspace(0, 1, 10)}
        
        result = tester.test_convergence(
            mock_method, mock_analytical, grid_sizes, test_params
        )
        
        assert isinstance(result, dict)
        assert 'errors' in result
        assert 'convergence_rate' in result

    def test_estimate_convergence_rate(self):
        """Test convergence rate estimation."""
        tester = ConvergenceTester()
        
        # Test with known convergence data
        errors = [1.0, 0.5, 0.25, 0.125]
        grid_sizes = [10, 20, 40, 80]
        
        rate = tester.estimate_convergence_rate(errors, grid_sizes)
        assert isinstance(rate, float)
        assert rate > 0  # Should be positive for decreasing errors


class TestConvergenceAnalyzer:
    """Test ConvergenceAnalyzer functionality."""

    def test_convergence_analyzer_creation(self):
        """Test ConvergenceAnalyzer creation."""
        analyzer = ConvergenceAnalyzer()
        assert hasattr(analyzer, 'analyze')

    def test_analyze(self):
        """Test convergence analysis."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with convergence data
        methods = ['method1', 'method2']
        grid_sizes = [10, 20, 40]
        errors = {
            'method1': [1.0, 0.5, 0.25],
            'method2': [1.0, 0.25, 0.0625]
        }
        
        result = analyzer.analyze(methods, grid_sizes, errors)
        assert isinstance(result, dict)
        assert 'convergence_rates' in result
        assert 'best_method' in result


class TestOrderOfAccuracy:
    """Test OrderOfAccuracy enum."""

    def test_order_of_accuracy_values(self):
        """Test OrderOfAccuracy enum values."""
        assert OrderOfAccuracy.FIRST_ORDER.value == 1.0
        assert OrderOfAccuracy.SECOND_ORDER.value == 2.0
        assert OrderOfAccuracy.THIRD_ORDER.value == 3.0
        assert OrderOfAccuracy.FOURTH_ORDER.value == 4.0


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark functionality."""

    def test_performance_benchmark_creation(self):
        """Test PerformanceBenchmark creation."""
        benchmark = PerformanceBenchmark()
        assert hasattr(benchmark, 'warmup_runs')
        assert benchmark.warmup_runs == 3

    def test_performance_benchmark_custom_warmup(self):
        """Test PerformanceBenchmark with custom warmup."""
        benchmark = PerformanceBenchmark(warmup_runs=5)
        assert benchmark.warmup_runs == 5

    def test_benchmark_function(self):
        """Test function benchmarking."""
        benchmark = PerformanceBenchmark()
        
        def test_func():
            return np.random.rand(1000)
        
        result = benchmark.benchmark_function(test_func, 'test_function')
        assert isinstance(result, dict)
        assert 'execution_time' in result
        assert 'memory_usage' in result
        assert 'success' in result


class TestAccuracyBenchmark:
    """Test AccuracyBenchmark functionality."""

    def test_accuracy_benchmark_creation(self):
        """Test AccuracyBenchmark creation."""
        benchmark = AccuracyBenchmark()
        assert hasattr(benchmark, 'benchmark_accuracy')

    def test_benchmark_accuracy(self):
        """Test accuracy benchmarking."""
        benchmark = AccuracyBenchmark()
        
        def mock_method(x):
            return x**2 + 0.01  # Simulate numerical method with small error
        
        def mock_analytical(x):
            return x**2  # Analytical solution
        
        x = np.linspace(0, 1, 100)
        
        result = benchmark.benchmark_accuracy(
            mock_method, mock_analytical, x, 'test_method'
        )
        
        assert isinstance(result, dict)
        assert 'accuracy_metrics' in result
        assert 'success' in result


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite creation."""
        suite = BenchmarkSuite()
        assert hasattr(suite, 'add_benchmark')
        assert hasattr(suite, 'run_benchmarks')

    def test_add_benchmark(self):
        """Test adding benchmarks to suite."""
        suite = BenchmarkSuite()
        
        def test_func():
            return np.random.rand(100)
        
        suite.add_benchmark('test_benchmark', test_func)
        assert 'test_benchmark' in suite.benchmarks

    def test_run_benchmarks(self):
        """Test running benchmark suite."""
        suite = BenchmarkSuite()
        
        def test_func():
            return np.random.rand(100)
        
        suite.add_benchmark('test_benchmark', test_func)
        results = suite.run_benchmarks()
        
        assert isinstance(results, dict)
        assert 'test_benchmark' in results


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_analytical_solution(self):
        """Test get_analytical_solution function."""
        x = np.array([1.0, 2.0, 3.0])
        result = get_analytical_solution('power', x, alpha=2.0, order=0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_validate_against_analytical(self):
        """Test validate_against_analytical function."""
        def mock_method(x):
            return x**2 + 0.01
        
        def mock_analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = validate_against_analytical(mock_method, mock_analytical, x)
        assert isinstance(result, dict)
        assert 'error_metrics' in result

    def test_run_convergence_study(self):
        """Test run_convergence_study function."""
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        grid_sizes = [10, 20, 40]
        result = run_convergence_study(mock_method, mock_analytical, grid_sizes)
        assert isinstance(result, dict)
        assert 'convergence_rate' in result

    def test_run_method_convergence_test(self):
        """Test run_method_convergence_test function."""
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        grid_sizes = [10, 20, 40]
        result = run_method_convergence_test(mock_method, mock_analytical, grid_sizes)
        assert isinstance(result, dict)
        assert 'convergence_rate' in result

    def test_estimate_convergence_rate(self):
        """Test estimate_convergence_rate function."""
        errors = [1.0, 0.5, 0.25, 0.125]
        grid_sizes = [10, 20, 40, 80]
        
        rate = estimate_convergence_rate(errors, grid_sizes)
        assert isinstance(rate, float)
        assert rate > 0

    def test_run_benchmarks(self):
        """Test run_benchmarks function."""
        def test_func():
            return np.random.rand(100)
        
        benchmarks = {'test_benchmark': test_func}
        results = run_benchmarks(benchmarks)
        assert isinstance(results, dict)
        assert 'test_benchmark' in results

    def test_compare_methods(self):
        """Test compare_methods function."""
        def method1(x):
            return x**2 + 0.01
        
        def method2(x):
            return x**2 + 0.02
        
        def analytical(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        methods = {'method1': method1, 'method2': method2}
        
        result = compare_methods(methods, analytical, x)
        assert isinstance(result, dict)
        assert 'method1' in result
        assert 'method2' in result

    def test_generate_benchmark_report(self):
        """Test generate_benchmark_report function."""
        results = {
            'method1': {'execution_time': 0.1, 'accuracy': 0.99},
            'method2': {'execution_time': 0.2, 'accuracy': 0.98}
        }
        
        report = generate_benchmark_report(results)
        assert isinstance(report, str)
        assert len(report) > 0


class TestIntegration:
    """Test integration between validation components."""

    def test_analytical_solutions_with_convergence(self):
        """Test analytical solutions with convergence testing."""
        solutions = AnalyticalSolutions()
        tester = ConvergenceTester()
        
        # Test power function derivative
        x = np.array([1.0, 2.0, 3.0])
        analytical = solutions.power_function_derivative(x, 2.0, 0.5)
        
        # Test convergence
        def mock_method(x, h):
            return analytical + 0.01 * h
        
        def mock_analytical_func(x):
            return analytical
        
        grid_sizes = [10, 20, 40]
        test_params = {'x': x}
        
        result = tester.test_convergence(
            mock_method, mock_analytical_func, grid_sizes, test_params
        )
        
        assert isinstance(result, dict)
        assert 'convergence_rate' in result

    def test_benchmarking_with_analytical_solutions(self):
        """Test benchmarking with analytical solutions."""
        solutions = AnalyticalSolutions()
        benchmark = PerformanceBenchmark()
        
        # Create function that uses analytical solutions
        def test_function():
            x = np.linspace(0, 1, 1000)
            return solutions.power_function_derivative(x, 2.0, 0.5)
        
        result = benchmark.benchmark_function(test_function, 'analytical_test')
        assert isinstance(result, dict)
        assert 'execution_time' in result
        assert 'success' in result

    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow."""
        # Create analytical solutions
        solutions = AnalyticalSolutions()
        
        # Create convergence tester
        tester = ConvergenceTester()
        
        # Create benchmark suite
        suite = BenchmarkSuite()
        
        # Test data
        x = np.linspace(0, 1, 100)
        analytical = solutions.power_function_derivative(x, 2.0, 0.5)
        
        # Add benchmark
        def test_method(x, h):
            return analytical + 0.01 * h
        
        suite.add_benchmark('test_method', lambda: test_method(x, 0.01))
        
        # Run benchmarks
        benchmark_results = suite.run_benchmarks()
        
        # Test convergence
        def analytical_func(x):
            return analytical
        
        grid_sizes = [10, 20, 40]
        test_params = {'x': x}
        
        convergence_result = tester.test_convergence(
            test_method, analytical_func, grid_sizes, test_params
        )
        
        # Verify results
        assert isinstance(benchmark_results, dict)
        assert isinstance(convergence_result, dict)
        assert 'test_method' in benchmark_results
        assert 'convergence_rate' in convergence_result


if __name__ == '__main__':
    pytest.main([__file__])
