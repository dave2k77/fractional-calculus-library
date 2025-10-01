"""
Final comprehensive test suite for the validation module.

This module tests all functionality in the hpfracc.validation module using the actual API.
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
        result = solutions.exponential_derivative(x, 1.0, 0.5)  # a=1.0, order=0.5
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_trigonometric_derivative(self):
        """Test trigonometric function derivative."""
        solutions = AnalyticalSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        result = solutions.trigonometric_derivative(x, 0.5, 1.0, 'sin')
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_constant_function_derivative(self):
        """Test constant function derivative."""
        solutions = AnalyticalSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.constant_function_derivative(x, 0.5, 1.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestPowerFunctionSolutions:
    """Test PowerFunctionSolutions functionality."""

    def test_power_function_solutions_creation(self):
        """Test PowerFunctionSolutions creation."""
        solutions = PowerFunctionSolutions()
        assert hasattr(solutions, 'get_solution')

    def test_get_solution(self):
        """Test solution computation."""
        solutions = PowerFunctionSolutions()
        
        x = np.array([1.0, 2.0, 3.0])
        alpha = 2.0
        order = 0.5
        
        result = solutions.get_solution(x, alpha, order)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_get_test_cases(self):
        """Test test cases generation."""
        solutions = PowerFunctionSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0


class TestExponentialSolutions:
    """Test ExponentialSolutions functionality."""

    def test_exponential_solutions_creation(self):
        """Test ExponentialSolutions creation."""
        solutions = ExponentialSolutions()
        assert hasattr(solutions, 'get_solution')

    def test_get_solution(self):
        """Test exponential solution computation."""
        solutions = ExponentialSolutions()
        
        x = np.array([0.0, 1.0, 2.0])
        a = 1.0
        order = 0.5
        
        result = solutions.get_solution(x, a, order)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_get_test_cases(self):
        """Test test cases generation."""
        solutions = ExponentialSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0


class TestTrigonometricSolutions:
    """Test TrigonometricSolutions functionality."""

    def test_trigonometric_solutions_creation(self):
        """Test TrigonometricSolutions creation."""
        solutions = TrigonometricSolutions()
        assert hasattr(solutions, 'get_solution')

    def test_get_solution(self):
        """Test trigonometric solution computation."""
        solutions = TrigonometricSolutions()
        
        x = np.array([0.0, np.pi/4, np.pi/2])
        func_type = 'sin'
        omega = 1.0
        order = 0.5
        
        # Correct parameter order: x, order, func_type, omega
        result = solutions.get_solution(x, order, func_type, omega)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

    def test_get_test_cases(self):
        """Test test cases generation."""
        solutions = TrigonometricSolutions()
        
        test_cases = solutions.get_test_cases()
        assert isinstance(test_cases, list)
        assert len(test_cases) > 0


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
        
        # Create mock functions that return proper results
        def mock_method(x, h):
            return x**2 + 0.1 * h  # Simulate numerical method with error
        
        def mock_analytical(x):
            return x**2  # Analytical solution
        
        grid_sizes = [10, 20, 40, 80]  # Need at least 4 points for convergence
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
        grid_sizes = [10, 20, 40, 80]
        errors = [1.0, 0.5, 0.25, 0.125]
        
        rate = tester.estimate_convergence_rate(grid_sizes, errors)
        assert isinstance(rate, float)
        assert rate > 0  # Should be positive for decreasing errors

    def test_test_multiple_norms(self):
        """Test multiple norm testing."""
        tester = ConvergenceTester()
        
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        grid_sizes = [10, 20, 40, 80]
        test_params = {'x': np.linspace(0, 1, 10)}
        
        result = tester.test_multiple_norms(
            mock_method, mock_analytical, grid_sizes, test_params
        )
        
        assert isinstance(result, dict)
        assert 'l1' in result
        assert 'l2' in result
        assert 'linf' in result


class TestConvergenceAnalyzer:
    """Test ConvergenceAnalyzer functionality."""

    def test_convergence_analyzer_creation(self):
        """Test ConvergenceAnalyzer creation."""
        analyzer = ConvergenceAnalyzer()
        assert hasattr(analyzer, 'analyze_method_convergence')

    def test_analyze_method_convergence(self):
        """Test method convergence analysis."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with convergence data
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        test_cases = [{'x': np.linspace(0, 1, 10)}]
        grid_sizes = [10, 20, 40, 80]
        
        result = analyzer.analyze_method_convergence(
            mock_method, mock_analytical, test_cases, grid_sizes
        )
        assert isinstance(result, dict)
        assert 'grid_sizes' in result
        assert 'summary' in result

    def test_estimate_optimal_grid_size(self):
        """Test optimal grid size estimation."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with convergence data
        errors = [1.0, 0.5, 0.25, 0.125]  # Errors decreasing with grid refinement
        grid_sizes = [10, 20, 40, 80]
        target_accuracy = 0.01
        
        optimal_size = analyzer.estimate_optimal_grid_size(
            errors, grid_sizes, target_accuracy
        )
        assert isinstance(optimal_size, int)
        assert optimal_size > 0

    def test_validate_convergence_order(self):
        """Test convergence order validation."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with known convergence data
        # Errors that follow a first-order convergence: error ~ C * h
        errors = [1.0, 0.5, 0.25, 0.125]  # error proportional to 1/n
        grid_sizes = [10, 20, 40, 80]
        expected_order = OrderOfAccuracy.FIRST_ORDER.value  # 1.0
        
        result = analyzer.validate_convergence_order(errors, grid_sizes, expected_order)
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'actual_order' in result


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

    def test_benchmark_method(self):
        """Test method benchmarking."""
        benchmark = PerformanceBenchmark()
        
        def test_func(**kwargs):
            return np.random.rand(1000)
        
        test_params = {'x': np.linspace(0, 1, 100)}
        result = benchmark.benchmark_method(test_func, test_params)
        
        # Result is a dict, not an object with attributes
        assert isinstance(result, dict)
        assert 'method_name' in result
        assert 'execution_time' in result
        assert 'success' in result

    def test_benchmark_multiple_methods(self):
        """Test multiple methods benchmarking."""
        benchmark = PerformanceBenchmark()
        
        def test_func1(**kwargs):
            return np.random.rand(1000)
        
        def test_func2(**kwargs):
            return np.random.rand(2000)
        
        methods = {'method1': test_func1, 'method2': test_func2}
        n_runs = 5
        
        results = benchmark.benchmark_multiple_methods(methods, n_runs)
        
        # Results should be a dict with method names as keys
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'method1' in results
        assert 'method2' in results
        assert all(isinstance(r, dict) for r in results.values())


class TestAccuracyBenchmark:
    """Test AccuracyBenchmark functionality."""

    def test_accuracy_benchmark_creation(self):
        """Test AccuracyBenchmark creation."""
        benchmark = AccuracyBenchmark()
        assert hasattr(benchmark, 'benchmark_method')

    def test_benchmark_method(self):
        """Test accuracy benchmarking."""
        benchmark = AccuracyBenchmark()
        
        # Modified to match actual API: benchmark_method(method_func, analytical_func, x, method_name)
        x = np.linspace(0, 1, 100)
        
        def mock_method(x_input):
            return x_input**2 + 0.01  # Simulate numerical method with small error
        
        def mock_analytical(x_input):
            return x_input**2  # Analytical solution
        
        result = benchmark.benchmark_method(
            mock_method, mock_analytical, x, 'test_method'
        )
        
        # Result is a dict
        assert isinstance(result, dict)
        assert 'method_name' in result
        assert result['method_name'] == 'test_method'
        # Check for any error-related keys
        assert any(k in result for k in ['accuracy_metrics', 'absolute_error', 'relative_error', 'max_error'])

    def test_benchmark_multiple_methods(self):
        """Test multiple methods accuracy benchmarking."""
        benchmark = AccuracyBenchmark()
        
        def mock_method1(x):
            return x**2 + 0.01
        
        def mock_method2(x):
            return x**2 + 0.02
        
        def mock_analytical(x):
            return x**2
        
        methods = {'method1': mock_method1, 'method2': mock_method2}
        x = np.linspace(0, 1, 100)
        
        results = benchmark.benchmark_multiple_methods(methods, mock_analytical, x)
        
        # Results should be a dict with method names as keys
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'method1' in results
        assert 'method2' in results
        assert all(isinstance(r, dict) for r in results.values())


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_benchmark_suite_creation(self):
        """Test BenchmarkSuite creation."""
        suite = BenchmarkSuite()
        assert hasattr(suite, 'run_comprehensive_benchmark')

    def test_run_comprehensive_benchmark(self):
        """Test comprehensive benchmark suite."""
        suite = BenchmarkSuite()
        
        def test_func(**kwargs):
            return np.random.rand(100)
        
        def analytical_func(x):
            return x**2
        
        test_cases = [{'x': np.linspace(0, 1, 10)}]
        
        results = suite.run_comprehensive_benchmark(
            {'test_method': test_func}, analytical_func, test_cases
        )
        
        assert isinstance(results, dict)
        assert 'accuracy_results' in results
        assert 'performance_results' in results
        assert 'summary' in results


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
        assert 'n_points' in result
        assert 'results' in result

    def test_run_convergence_study(self):
        """Test run_convergence_study function."""
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        grid_sizes = [10, 20, 40, 80]
        result = run_convergence_study(mock_method, mock_analytical, grid_sizes)
        assert isinstance(result, dict)
        assert 'grid_sizes' in result
        assert 'summary' in result

    def test_run_method_convergence_test(self):
        """Test run_method_convergence_test function."""
        def mock_method(x, h):
            return x**2 + 0.1 * h
        
        def mock_analytical(x):
            return x**2
        
        grid_sizes = [10, 20, 40, 80]
        test_params = {'x': np.linspace(0, 1, 10)}
        
        result = run_method_convergence_test(mock_method, mock_analytical, grid_sizes, test_params)
        assert isinstance(result, dict)
        assert 'l1' in result
        assert 'l2' in result
        assert 'linf' in result

    def test_estimate_convergence_rate(self):
        """Test estimate_convergence_rate function."""
        errors = [1.0, 0.5, 0.25, 0.125]
        grid_sizes = [10, 20, 40, 80]
        
        rate = estimate_convergence_rate(errors, grid_sizes)
        assert isinstance(rate, float)
        assert rate > 0

    def test_run_benchmarks(self):
        """Test run_benchmarks function."""
        def test_func(**kwargs):
            return np.random.rand(100)
        
        def analytical_func(x):
            return x**2
        
        test_cases = [{'x': np.linspace(0, 1, 10)}]
        benchmarks = {'test_benchmark': test_func}
        
        results = run_benchmarks(benchmarks, analytical_func, test_cases)
        assert isinstance(results, dict)
        assert 'accuracy_results' in results
        assert 'performance_results' in results

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
        # Check for actual keys returned by compare_methods
        assert 'methods' in result or 'accuracy_results' in result
        assert 'performance_results' in result or 'summary' in result

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
        
        grid_sizes = [10, 20, 40, 80]  # Need at least 4 points
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
        def test_function(**kwargs):
            x = np.linspace(0, 1, 1000)
            return solutions.power_function_derivative(x, 2.0, 0.5)
        
        method_name = 'test_analytical_method'
        result = benchmark.benchmark_method(test_function, method_name, n_runs=5)
        
        assert isinstance(result, dict)
        assert 'method_name' in result
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
        
        # Test convergence
        def test_method(x, h):
            return analytical + 0.01 * h
        
        def analytical_func(x):
            return analytical
        
        grid_sizes = [10, 20, 40, 80]
        test_params = {'x': x}
        
        convergence_result = tester.test_convergence(
            test_method, analytical_func, grid_sizes, test_params
        )
        
        # Test benchmarking
        def benchmark_func(**kwargs):
            return test_method(x, 0.01)
        
        def analytical_benchmark(x):
            return analytical
        
        test_cases = [{'x': x}]
        
        benchmark_results = suite.run_comprehensive_benchmark(
            {'test_method': benchmark_func}, analytical_benchmark, test_cases
        )
        
        # Verify results
        assert isinstance(benchmark_results, dict)
        assert isinstance(convergence_result, dict)
        assert 'accuracy_results' in benchmark_results
        assert 'convergence_rate' in convergence_result


if __name__ == '__main__':
    pytest.main([__file__])
