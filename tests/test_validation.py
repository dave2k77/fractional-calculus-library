"""
Tests for validation modules.
"""

import numpy as np
import pytest
from hpfracc.validation.analytical_solutions import (
    AnalyticalSolutions,
    PowerFunctionSolutions,
    ExponentialSolutions,
    TrigonometricSolutions,
    get_analytical_solution,
    validate_against_analytical,
)
from hpfracc.validation.convergence_tests import (
    ConvergenceTester,
    ConvergenceAnalyzer,
    OrderOfAccuracy,
    run_convergence_study,
    run_method_convergence_test,
    estimate_convergence_rate,
)
from hpfracc.validation.benchmarks import (
    BenchmarkSuite,
    PerformanceBenchmark,
    AccuracyBenchmark,
    run_benchmarks,
    compare_methods,
    generate_benchmark_report,
)


class TestAnalyticalSolutions:
    """Test analytical solutions."""

    def test_analytical_solutions_basic(self):
        """Test basic analytical solutions functionality."""
        solutions = AnalyticalSolutions()

        # Test power function
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.power_function_derivative(x, alpha=2.0, order=0.5)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

        # Test exponential function
        result = solutions.exponential_derivative(x, a=1.0, order=0.5)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

        # Test trigonometric function
        result = solutions.trigonometric_derivative(x, 0.5, 1.0, "sin")
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_power_function_solutions(self):
        """Test power function solutions."""
        solutions = PowerFunctionSolutions()

        # Test solution computation
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.get_solution(x, alpha=1.0, order=0.5)
        assert result.shape == x.shape

        # Test test cases
        test_cases = solutions.get_test_cases()
        assert len(test_cases) > 0
        assert "alpha" in test_cases[0]
        assert "order" in test_cases[0]

    def test_exponential_solutions(self):
        """Test exponential solutions."""
        solutions = ExponentialSolutions()

        # Test solution computation
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.get_solution(x, a=1.0, order=0.5)
        assert result.shape == x.shape

        # Test test cases
        test_cases = solutions.get_test_cases()
        assert len(test_cases) > 0
        assert "a" in test_cases[0]
        assert "order" in test_cases[0]

    def test_trigonometric_solutions(self):
        """Test trigonometric solutions."""
        solutions = TrigonometricSolutions()

        # Test solution computation
        x = np.array([1.0, 2.0, 3.0])
        result = solutions.get_solution(x, 0.5, "sin", 1.0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)

        # Test test cases
        test_cases = solutions.get_test_cases()
        assert len(test_cases) > 0
        assert "func_type" in test_cases[0]
        assert "omega" in test_cases[0]

    def test_get_analytical_solution(self):
        """Test convenience function."""
        x = np.array([1.0, 2.0, 3.0])

        # Test power function
        result = get_analytical_solution("power", x, alpha=1.0, order=0.5)
        assert result.shape == x.shape

        # Test exponential function
        result = get_analytical_solution("exponential", x, a=1.0, order=0.5)
        assert result.shape == x.shape

        # Test invalid function type
        with pytest.raises(ValueError):
            get_analytical_solution("invalid", x)

    def test_validate_against_analytical(self):
        """Test validation against analytical solution."""

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1  # Slight difference

        test_params = [{"order": 0.5}]

        results = validate_against_analytical(
            mock_numerical, mock_analytical, test_params
        )

        assert "results" in results
        assert "summary" in results
        assert len(results["results"]) == 1
        assert results["results"][0]["success"] is True


class TestConvergenceTests:
    """Test convergence tests."""

    def test_convergence_tester_basic(self):
        """Test basic convergence tester functionality."""
        tester = ConvergenceTester()

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1 / len(x)  # Error decreases with grid size

        grid_sizes = [10, 20, 40]
        test_params = {"order": 0.5}

        result = tester.test_convergence(
            mock_numerical, mock_analytical, grid_sizes, test_params
        )

        assert "convergence_rate" in result
        assert "errors" in result
        assert "grid_sizes" in result
        assert result["convergence_rate"] > 0

    def test_convergence_tester_multiple_norms(self):
        """Test convergence tester with multiple norms."""
        tester = ConvergenceTester()

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1 / len(x)

        grid_sizes = [10, 20, 40]
        test_params = {"order": 0.5}

        results = tester.test_multiple_norms(
            mock_numerical, mock_analytical, grid_sizes, test_params
        )

        assert "l1" in results
        assert "l2" in results
        assert "linf" in results
        assert all(result is not None for result in results.values())

    def test_convergence_analyzer(self):
        """Test convergence analyzer."""
        analyzer = ConvergenceAnalyzer()

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1 / len(x)

        test_cases = [{"order": 0.5}, {"order": 1.0}]
        grid_sizes = [10, 20, 40]

        results = analyzer.analyze_method_convergence(
            mock_numerical, mock_analytical, test_cases, grid_sizes
        )

        assert "test_cases" in results
        assert "summary" in results
        assert len(results["test_cases"]) == 2
        assert results["summary"]["total_test_cases"] == 2

    def test_order_of_accuracy(self):
        """Test order of accuracy enum."""
        assert OrderOfAccuracy.FIRST_ORDER.value == 1.0
        assert OrderOfAccuracy.SECOND_ORDER.value == 2.0
        assert OrderOfAccuracy.THIRD_ORDER.value == 3.0
        assert OrderOfAccuracy.FOURTH_ORDER.value == 4.0

    def test_convenience_functions(self):
        """Test convenience functions."""

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1 / len(x)

        test_cases = [{"order": 0.5}]
        grid_sizes = [10, 20, 40]

        # Test run_convergence_study
        results = run_convergence_study(
            mock_numerical, mock_analytical, test_cases, grid_sizes
        )
        assert "summary" in results

    def test_error_reporting(self):
        """Test error reporting."""

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1 / len(x)

        test_cases = [{"order": 0.5}]
        grid_sizes = [10, 20, 40]

        # Test run_convergence_study
        results = run_convergence_study(
            mock_numerical, mock_analytical, test_cases, grid_sizes
        )
        assert "summary" in results

        # Test run_method_convergence_test
        test_params = {"order": 0.5}
        results = run_method_convergence_test(
            mock_numerical, mock_analytical, grid_sizes, test_params
        )
        assert "l1" in results

        # Test estimate_convergence_rate
        grid_sizes = [10, 20, 40]
        errors = [1.0, 0.5, 0.25]
        rate = estimate_convergence_rate(grid_sizes, errors)
        assert rate > 0


class TestBenchmarks:
    """Test benchmarks."""

    def test_performance_benchmark(self):
        """Test performance benchmark."""
        benchmark = PerformanceBenchmark()

        # Mock function
        def mock_function(x, **kwargs):
            return x**2

        test_params = {"x": np.array([1.0, 2.0, 3.0])}

        result = benchmark.benchmark_method(mock_function, test_params, n_runs=3)

        assert result['success'] is True
        assert result['execution_time'] > 0
        assert result['method_name'] == "mock_function"

    def test_accuracy_benchmark(self):
        """Test accuracy benchmark."""
        benchmark = AccuracyBenchmark()

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1

        test_params = {"x": np.array([1.0, 2.0, 3.0])}

        result = benchmark.benchmark_method(
            'mock_numerical', mock_numerical, mock_analytical, test_params
        )

        assert result['success'] is True
        assert "l2" in result['accuracy_metrics']
        assert "linf" in result['accuracy_metrics']

    def test_benchmark_suite(self):
        """Test benchmark suite."""
        suite = BenchmarkSuite()

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1

        methods = {"test_method": mock_numerical}
        test_cases = [{"x": np.array([1.0, 2.0, 3.0])}]

        results = suite.run_comprehensive_benchmark(
            methods, mock_analytical, test_cases, n_runs=2
        )

        assert "accuracy_results" in results
        assert "performance_results" in results
        assert "summary" in results
        assert len(results["accuracy_results"]) > 0

    def test_convenience_functions(self):
        """Test convenience functions."""

        # Mock functions
        def mock_numerical(x, **kwargs):
            return x**1.5

        def mock_analytical(x, **kwargs):
            return x**1.5 + 0.1

        methods = {"test_method": mock_numerical}
        test_cases = [{"x": np.array([1.0, 2.0, 3.0])}]

        # Test run_benchmarks
        results = run_benchmarks(methods, mock_analytical, test_cases, n_runs=2)
        assert "accuracy_results" in results

        # Test compare_methods
        test_params = {"x": np.array([1.0, 2.0, 3.0])}
        comparison = compare_methods(methods, mock_analytical, test_params)
        assert "methods" in comparison
        assert "summary" in comparison

        # Test generate_benchmark_report
        report = generate_benchmark_report(results)
        assert "FRACTIONAL CALCULUS BENCHMARK REPORT" in report


def test_integration():
    """Test integration between validation modules."""
    # Create a simple test scenario
    x = np.linspace(0, 1, 100)

    # Mock numerical method
    def mock_numerical_method(x, order=0.5, alpha=1.0):
        # Simple approximation
        return x ** (alpha - order) * 0.8

    # Get analytical solution
    analytical = get_analytical_solution("power", x, alpha=1.0, order=0.5)

    # Test validation
    test_params = [{"order": 0.5, "alpha": 1.0}]
    validation_results = validate_against_analytical(
        mock_numerical_method,
        lambda x, **kwargs: get_analytical_solution("power", x, **kwargs),
        test_params,
    )

    assert validation_results["summary"]["success_rate"] > 0

    # Test convergence
    grid_sizes = [50, 100, 200]
    convergence_results = run_method_convergence_test(
        mock_numerical_method,
        lambda x, **kwargs: get_analytical_solution("power", x, **kwargs),
        grid_sizes,
        {"order": 0.5, "alpha": 1.0},
    )

    assert "l2" in convergence_results
    assert convergence_results["l2"] is not None

    # Test benchmarks
    methods = {"mock_method": mock_numerical_method}
    test_cases = [{"x": x, "order": 0.5, "alpha": 1.0}]

    benchmark_results = run_benchmarks(
        methods,
        lambda x, **kwargs: get_analytical_solution("power", x, **kwargs),
        test_cases,
        n_runs=2,
    )

    assert "accuracy_results" in benchmark_results
    assert len(benchmark_results["accuracy_results"]) > 0
