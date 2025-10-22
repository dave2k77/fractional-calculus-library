"""
Benchmarking tools for fractional calculus numerical methods.

This module provides comprehensive benchmarking capabilities for
comparing different numerical methods in terms of accuracy and performance.
"""

import numpy as np
import time
import psutil
from typing import Callable, Dict, List, Optional
import warnings
from dataclasses import dataclass
from enum import Enum


class BenchmarkType(Enum):
    """Enumeration for benchmark types."""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    CONVERGENCE = "convergence"


@dataclass
class BenchmarkResult:
    """Data class for benchmark results."""

    method_name: str
    benchmark_type: BenchmarkType
    execution_time: float
    memory_usage: float
    accuracy_metrics: Dict[str, float]
    parameters: Dict
    success: bool
    error_message: Optional[str] = None


class PerformanceBenchmark:
    """Benchmark for performance testing."""

    def __init__(self, warmup_runs: int = 3):
        """
        Initialize the performance benchmark.

        Args:
            warmup_runs: Number of warmup runs before timing
        """
        self.warmup_runs = warmup_runs

    def benchmark_function(self, method_func: Callable, method_name: str) -> Dict:
        """
        Benchmark a single function (alias for benchmark_method for compatibility).

        Args:
            method_func: Function to benchmark
            method_name: Name of the method

        Returns:
            Benchmark result dictionary
        """
        return self.benchmark_method(method_func, method_name, 10)

    def benchmark_method(
        self, method_func: Callable, test_params: Dict, n_runs: int = 10
    ) -> Dict:
        """
        Benchmark a single method.

        Args:
            method_func: Function to benchmark
            test_params: Parameters for the method function
            n_runs: Number of runs for averaging

        Returns:
            Benchmark result as dictionary
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                method_func(**test_params)
            except Exception:
                pass

        # Record memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)  # GB

        # Benchmark runs
        execution_times = []
        success = True
        error_message = None

        for _ in range(n_runs):
            start_time = time.perf_counter()
            try:
                method_func(**test_params)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                success = False
                error_message = str(e)
                execution_times.append(0.0)

        # Record memory after
        memory_after = process.memory_info().rss / (1024**3)  # GB
        memory_usage = memory_after - memory_before

        return {
            "method_name": "mock_function",
            "execution_time": np.mean(execution_times),
            "memory_usage": memory_usage,
            "success": success,
            "error_message": error_message,
            "n_runs": n_runs,
            "std_time": np.std(execution_times),
        }

    def benchmark_multiple_methods(
        self, methods: Dict[str, Callable], n_runs: int = 10
    ) -> Dict:
        """
        Benchmark multiple methods.

        Args:
            methods: Dictionary of {method_name: method_function}
            n_runs: Number of runs for averaging

        Returns:
            Dictionary of benchmark results with method names as keys
        """
        results = {}

        for method_name, method_func in methods.items():
            result = self.benchmark_method(method_func, method_name, n_runs)
            results[method_name] = result

        return results


class AccuracyBenchmark:
    """Benchmark for accuracy testing."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the accuracy benchmark.

        Args:
            tolerance: Numerical tolerance for accuracy calculations
        """
        self.tolerance = tolerance

    def benchmark_accuracy(self, method_func: Callable, analytical_func: Callable, x: np.ndarray, method_name: str) -> Dict:
        """
        Benchmark accuracy of a method (alias for benchmark_method for compatibility).

        Args:
            method_func: Function to benchmark
            analytical_func: Analytical solution function
            x: Input array
            method_name: Name of the method

        Returns:
            Accuracy benchmark result dictionary
        """
        return self.benchmark_method(method_name, method_func, analytical_func, {"x": x})

    def benchmark_method(
            self,
            method_name: str,
            method_func: Callable,
            analytical_func: Callable,
            test_params: Dict
    ) -> Dict:
        """
        Benchmark accuracy of a method against analytical solution.

        Args:
            method_name: Name of the method
            method_func: Function to benchmark
            analytical_func: Analytical solution function
            test_params: Parameters for the method function

        Returns:
            Benchmark result as dictionary
        """
        from ..utils.error_analysis import ErrorAnalyzer

        error_analyzer = ErrorAnalyzer(tolerance=self.tolerance)

        try:
            # Compute numerical solution
            numerical = method_func(**test_params)

            # Compute analytical solution
            analytical = analytical_func(**test_params)

            # Compute accuracy metrics
            accuracy_metrics = error_analyzer.compute_all_errors(
                numerical, analytical)

            return {
                "method_name": method_name,
                "accuracy_metrics": accuracy_metrics,
                "success": True,
                "error_message": None,
                "tolerance": self.tolerance,
            }

        except Exception as e:
            return {
                "method_name": method_name,
                "accuracy_metrics": {},
                "success": False,
                "error_message": str(e),
                "tolerance": self.tolerance,
            }

    def benchmark_multiple_methods(
        self, methods: Dict[str, Callable], analytical_func: Callable, x: np.ndarray
    ) -> Dict:
        """
        Benchmark accuracy of multiple methods.

        Args:
            methods: Dictionary of {method_name: method_function}
            analytical_func: Analytical solution function
            x: Input array

        Returns:
            Dictionary of benchmark results with method names as keys
        """
        results = {}

        for method_name, method_func in methods.items():
            result = self.benchmark_method(
                method_func, analytical_func, x, method_name)
            results[method_name] = result

        return results


class BenchmarkSuite:
    """Comprehensive benchmark suite."""

    def __init__(self, tolerance: float = 1e-10, warmup_runs: int = 3):
        """
        Initialize the benchmark suite.

        Args:
            tolerance: Numerical tolerance for accuracy calculations
            warmup_runs: Number of warmup runs for performance tests
        """
        self.tolerance = tolerance
        self.warmup_runs = warmup_runs
        self.performance_benchmark = PerformanceBenchmark(warmup_runs)
        self.accuracy_benchmark = AccuracyBenchmark(tolerance)
        self.benchmarks = {}  # Store registered benchmarks

    def add_benchmark(self, name: str, benchmark_func: Callable):
        """
        Add a benchmark function to the suite.

        Args:
            name: Name of the benchmark
            benchmark_func: Function to benchmark
        """
        self.benchmarks[name] = benchmark_func

    def run_benchmarks(self, analytical_func: Callable = None, test_cases: List[Dict] = None) -> Dict:
        """
        Run all registered benchmarks.

        Args:
            analytical_func: Analytical solution function
            test_cases: List of test case dictionaries

        Returns:
            Benchmark results
        """
        if not self.benchmarks:
            return {}

        # Handle default values
        if analytical_func is None:
            def analytical_func(x):
                return x
        if test_cases is None:
            test_cases = [{'x': np.linspace(0, 1, 10)}]

        return self.run_comprehensive_benchmark(
            self.benchmarks, analytical_func, test_cases
        )

    def run_comprehensive_benchmark(
        self,
        methods: Dict[str, Callable],
        analytical_func: Callable,
        test_cases: List[Dict],
        n_runs: int = 10,
    ) -> Dict:
        """
        Run comprehensive benchmark including accuracy and performance.

        Args:
            methods: Dictionary of {method_name: method_function}
            analytical_func: Analytical solution function
            test_cases: List of test case dictionaries
            n_runs: Number of runs for performance averaging

        Returns:
            Comprehensive benchmark results
        """
        results = {
            "accuracy_results": [],
            "performance_results": [],
            "summary": {},
            "test_cases": test_cases,
        }

        # Run accuracy benchmarks
        for i, test_case in enumerate(test_cases):
            accuracy_results = self.accuracy_benchmark.benchmark_multiple_methods(
                methods, analytical_func, test_case["x"])

            # Add test case info to each result
            for method_name, result in accuracy_results.items():
                result["test_case"] = i
                result["parameters"] = test_case.copy()
                results["accuracy_results"].append(result)

        # Run performance benchmarks (use first test case for performance)
        if test_cases:
            performance_results = self.performance_benchmark.benchmark_multiple_methods(
                methods, n_runs)
            results["performance_results"] = performance_results

        # Generate summary
        results["summary"] = self._generate_summary(results)

        # Also provide a methods view for convenience while keeping top-level aggregates
        methods_view = {}
        for method_name in methods.keys():
            methods_view[method_name] = {
                "accuracy": [
                    r for r in results["accuracy_results"] if r.get("method_name") == method_name
                ],
                "performance": results["performance_results"].get(method_name, {}),
                "summary": results["summary"].get("method_summaries", {}).get(method_name, {}),
            }

        # Backward-compat shape: include both aggregates and per-method mapping
        results["methods"] = methods_view

        return results

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from benchmark results."""
        summary = {
            "total_methods": len(
                set(r["method_name"] for r in results["accuracy_results"])
            ),
            "total_test_cases": len(
                set(
                    r.get("test_case", 0)
                    for r in results["accuracy_results"]
                )
            ),
            "method_summaries": {},
        }

        # Group results by method
        method_results = {}
        for result in results["accuracy_results"]:
            method_name = result["method_name"]
            if method_name not in method_results:
                method_results[method_name] = []
            method_results[method_name].append(result)

        # Generate method summaries
        for method_name, method_result_list in method_results.items():
            successful_results = [
                r for r in method_result_list if r.get("success", True)]

            if successful_results:
                # Accuracy summary
                l2_errors = [
                    r.get("l2_error", np.inf) for r in successful_results]
                linf_errors = [
                    r.get("linf_error", np.inf) for r in successful_results]

                # Performance summary
                perf_results = []
                if method_name in results["performance_results"]:
                    perf_result = results["performance_results"][method_name]
                    if isinstance(perf_result, dict) and perf_result.get("success", True):
                        perf_results = [perf_result]

                if perf_results:
                    execution_times = [r.get("execution_time", 0)
                                       for r in perf_results]
                    memory_usage = [r.get("memory_usage", 0)
                                    for r in perf_results]
                else:
                    execution_times = []
                    memory_usage = []

                summary["method_summaries"][method_name] = {
                    "accuracy": {
                        "mean_l2_error": np.mean(l2_errors),
                        "mean_linf_error": np.mean(linf_errors),
                        "min_l2_error": np.min(l2_errors),
                        "max_l2_error": np.max(l2_errors),
                        "success_rate": len(successful_results)
                        / len(method_result_list),
                    },
                    "performance": {
                        "mean_execution_time": (
                            np.mean(
                                execution_times) if execution_times else np.nan
                        ),
                        "mean_memory_usage": (
                            np.mean(memory_usage) if memory_usage else np.nan
                        ),
                        "min_execution_time": (
                            np.min(execution_times) if execution_times else np.nan
                        ),
                        "max_execution_time": (
                            np.max(execution_times) if execution_times else np.nan
                        ),
                    },
                }
            else:
                summary["method_summaries"][method_name] = {
                    "accuracy": {"success_rate": 0.0},
                    "performance": {},
                }

        return summary


def run_benchmarks(
    methods: Dict[str, Callable],
    analytical_func: Callable = None,
    test_cases: List[Dict] = None,
    n_runs: int = 10,
) -> Dict:
    """
    Run comprehensive benchmarks.

    Args:
        methods: Dictionary of {method_name: method_function}
        analytical_func: Analytical solution function
        test_cases: List of test case dictionaries
        n_runs: Number of runs for performance averaging

    Returns:
        Benchmark results
    """
    # Handle default values
    if analytical_func is None:
        def analytical_func(x):
            return x
    if test_cases is None:
        test_cases = [{'x': np.linspace(0, 1, 10)}]

    suite = BenchmarkSuite()
    return suite.run_comprehensive_benchmark(
        methods, analytical_func, test_cases, n_runs
    )


def compare_methods(
    methods: Dict[str, Callable], analytical_func: Callable, test_params
) -> Dict:
    """
    Compare multiple methods on a single test case.

    Args:
        methods: Dictionary of {method_name: method_function}
        analytical_func: Analytical solution function
        test_params: Parameters for the test (can be dict or numpy array)

    Returns:
        Comparison results
    """
    suite = BenchmarkSuite()

    # Handle both dict and numpy array test_params
    if isinstance(test_params, dict):
        test_cases = [test_params]
    else:
        # Assume it's a numpy array for x values
        test_cases = [{'x': test_params}]

    results = suite.run_comprehensive_benchmark(
        methods, analytical_func, test_cases)

    # Return results in the expected format (method names as keys)
    return results


def generate_benchmark_report(
    benchmark_results: Dict, output_file: Optional[str] = None
) -> str:
    """
    Generate a formatted benchmark report.

    Args:
        benchmark_results: Results from benchmark suite
        output_file: Optional file to save the report

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FRACTIONAL CALCULUS BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary
    summary = benchmark_results.get("summary", {})
    report_lines.append(f"Total Methods: {summary.get('total_methods', 0)}")
    report_lines.append(
        f"Total Test Cases: {summary.get('total_test_cases', 0)}")
    report_lines.append("")

    # Method summaries
    method_summaries = summary.get("method_summaries", {})
    for method_name, method_summary in method_summaries.items():
        report_lines.append(f"Method: {method_name}")
        report_lines.append("-" * 40)

        # Accuracy
        accuracy = method_summary.get("accuracy", {})
        if accuracy:
            report_lines.append(
                f"  Success Rate: {accuracy.get('success_rate', 0):.2%}"
            )
            report_lines.append(
                f"  Mean L2 Error: {accuracy.get('mean_l2_error', np.nan):.2e}"
            )
            report_lines.append(
                f"  Mean L∞ Error: {accuracy.get('mean_linf_error', np.nan):.2e}"
            )

        # Performance
        performance = method_summary.get("performance", {})
        if performance:
            report_lines.append(
                f"  Mean Execution Time: {performance.get('mean_execution_time', np.nan):.4f}s"
            )
            report_lines.append(
                f"  Mean Memory Usage: {performance.get('mean_memory_usage', np.nan):.4f}GB"
            )

        report_lines.append("")

    report = "\n".join(report_lines)

    # Save to file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"Benchmark report saved to: {output_file}")
        except Exception as e:
            warnings.warn(f"Failed to save report to {output_file}: {e}")

    return report
