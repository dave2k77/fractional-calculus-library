"""
Benchmarking tools for fractional calculus numerical methods.

This module provides comprehensive benchmarking capabilities for
comparing different numerical methods in terms of accuracy and performance.
"""

import numpy as np
import time
import psutil
from typing import Callable, Dict, List, Optional, Tuple, Union
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
    
    def benchmark_method(self, method_func: Callable, test_params: Dict,
                        n_runs: int = 10) -> BenchmarkResult:
        """
        Benchmark a single method.
        
        Args:
            method_func: Function to benchmark
            test_params: Parameters for the test
            n_runs: Number of runs for averaging
            
        Returns:
            Benchmark result
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
        
        for _ in range(n_runs):
            start_time = time.perf_counter()
            try:
                result = method_func(**test_params)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            except Exception as e:
                return BenchmarkResult(
                    method_name=method_func.__name__,
                    benchmark_type=BenchmarkType.PERFORMANCE,
                    execution_time=0.0,
                    memory_usage=0.0,
                    accuracy_metrics={},
                    parameters=test_params,
                    success=False,
                    error_message=str(e)
                )
        
        # Record memory after
        memory_after = process.memory_info().rss / (1024**3)  # GB
        memory_usage = memory_after - memory_before
        
        return BenchmarkResult(
            method_name=method_func.__name__,
            benchmark_type=BenchmarkType.PERFORMANCE,
            execution_time=np.mean(execution_times),
            memory_usage=memory_usage,
            accuracy_metrics={'std_time': np.std(execution_times)},
            parameters=test_params,
            success=True
        )
    
    def benchmark_multiple_methods(self, methods: Dict[str, Callable],
                                  test_params: Dict, n_runs: int = 10) -> List[BenchmarkResult]:
        """
        Benchmark multiple methods.
        
        Args:
            methods: Dictionary of {method_name: method_function}
            test_params: Parameters for the test
            n_runs: Number of runs for averaging
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for method_name, method_func in methods.items():
            result = self.benchmark_method(method_func, test_params, n_runs)
            result.method_name = method_name
            results.append(result)
        
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
    
    def benchmark_method(self, method_func: Callable, analytical_func: Callable,
                        test_params: Dict) -> BenchmarkResult:
        """
        Benchmark accuracy of a method against analytical solution.
        
        Args:
            method_func: Function to benchmark
            analytical_func: Analytical solution function
            test_params: Parameters for the test
            
        Returns:
            Benchmark result
        """
        from ..utils.error_analysis import ErrorAnalyzer
        
        error_analyzer = ErrorAnalyzer(tolerance=self.tolerance)
        
        try:
            # Compute numerical solution
            numerical = method_func(**test_params)
            
            # Compute analytical solution
            analytical = analytical_func(**test_params)
            
            # Compute accuracy metrics
            accuracy_metrics = error_analyzer.compute_all_errors(numerical, analytical)
            
            return BenchmarkResult(
                method_name=method_func.__name__,
                benchmark_type=BenchmarkType.ACCURACY,
                execution_time=0.0,
                memory_usage=0.0,
                accuracy_metrics=accuracy_metrics,
                parameters=test_params,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                method_name=method_func.__name__,
                benchmark_type=BenchmarkType.ACCURACY,
                execution_time=0.0,
                memory_usage=0.0,
                accuracy_metrics={},
                parameters=test_params,
                success=False,
                error_message=str(e)
            )
    
    def benchmark_multiple_methods(self, methods: Dict[str, Callable],
                                  analytical_func: Callable,
                                  test_params: Dict) -> List[BenchmarkResult]:
        """
        Benchmark accuracy of multiple methods.
        
        Args:
            methods: Dictionary of {method_name: method_function}
            analytical_func: Analytical solution function
            test_params: Parameters for the test
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for method_name, method_func in methods.items():
            result = self.benchmark_method(method_func, analytical_func, test_params)
            result.method_name = method_name
            results.append(result)
        
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
    
    def run_comprehensive_benchmark(self, methods: Dict[str, Callable],
                                   analytical_func: Callable,
                                   test_cases: List[Dict],
                                   n_runs: int = 10) -> Dict:
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
            'accuracy_results': [],
            'performance_results': [],
            'summary': {},
            'test_cases': test_cases,
        }
        
        # Run accuracy benchmarks
        for i, test_case in enumerate(test_cases):
            accuracy_results = self.accuracy_benchmark.benchmark_multiple_methods(
                methods, analytical_func, test_case
            )
            
            for result in accuracy_results:
                result.parameters['test_case_index'] = i
            
            results['accuracy_results'].extend(accuracy_results)
        
        # Run performance benchmarks (use first test case for performance)
        if test_cases:
            performance_results = self.performance_benchmark.benchmark_multiple_methods(
                methods, test_cases[0], n_runs
            )
            results['performance_results'] = performance_results
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from benchmark results."""
        summary = {
            'total_methods': len(set(r.method_name for r in results['accuracy_results'])),
            'total_test_cases': len(set(r.parameters.get('test_case_index', 0) 
                                      for r in results['accuracy_results'])),
            'method_summaries': {},
        }
        
        # Group results by method
        method_results = {}
        for result in results['accuracy_results']:
            if result.method_name not in method_results:
                method_results[result.method_name] = []
            method_results[result.method_name].append(result)
        
        # Generate method summaries
        for method_name, method_result_list in method_results.items():
            successful_results = [r for r in method_result_list if r.success]
            
            if successful_results:
                # Accuracy summary
                l2_errors = [r.accuracy_metrics.get('l2', np.inf) for r in successful_results]
                linf_errors = [r.accuracy_metrics.get('linf', np.inf) for r in successful_results]
                
                # Performance summary
                perf_results = [r for r in results['performance_results'] 
                              if r.method_name == method_name and r.success]
                
                if perf_results:
                    execution_times = [r.execution_time for r in perf_results]
                    memory_usage = [r.memory_usage for r in perf_results]
                else:
                    execution_times = []
                    memory_usage = []
                
                summary['method_summaries'][method_name] = {
                    'accuracy': {
                        'mean_l2_error': np.mean(l2_errors),
                        'mean_linf_error': np.mean(linf_errors),
                        'min_l2_error': np.min(l2_errors),
                        'max_l2_error': np.max(l2_errors),
                        'success_rate': len(successful_results) / len(method_result_list),
                    },
                    'performance': {
                        'mean_execution_time': np.mean(execution_times) if execution_times else np.nan,
                        'mean_memory_usage': np.mean(memory_usage) if memory_usage else np.nan,
                        'min_execution_time': np.min(execution_times) if execution_times else np.nan,
                        'max_execution_time': np.max(execution_times) if execution_times else np.nan,
                    }
                }
            else:
                summary['method_summaries'][method_name] = {
                    'accuracy': {'success_rate': 0.0},
                    'performance': {}
                }
        
        return summary


def run_benchmarks(methods: Dict[str, Callable], analytical_func: Callable,
                   test_cases: List[Dict], n_runs: int = 10) -> Dict:
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
    suite = BenchmarkSuite()
    return suite.run_comprehensive_benchmark(methods, analytical_func, test_cases, n_runs)


def compare_methods(methods: Dict[str, Callable], analytical_func: Callable,
                   test_params: Dict) -> Dict:
    """
    Compare multiple methods on a single test case.
    
    Args:
        methods: Dictionary of {method_name: method_function}
        analytical_func: Analytical solution function
        test_params: Parameters for the test
        
    Returns:
        Comparison results
    """
    suite = BenchmarkSuite()
    test_cases = [test_params]
    results = suite.run_comprehensive_benchmark(methods, analytical_func, test_cases)
    
    # Extract comparison data
    comparison = {
        'methods': list(methods.keys()),
        'accuracy_comparison': {},
        'performance_comparison': {},
    }
    
    # Accuracy comparison
    for result in results['accuracy_results']:
        if result.success:
            comparison['accuracy_comparison'][result.method_name] = {
                'l2_error': result.accuracy_metrics.get('l2', np.inf),
                'linf_error': result.accuracy_metrics.get('linf', np.inf),
            }
    
    # Performance comparison
    for result in results['performance_results']:
        if result.success:
            comparison['performance_comparison'][result.method_name] = {
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
            }
    
    return comparison


def generate_benchmark_report(benchmark_results: Dict, output_file: Optional[str] = None) -> str:
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
    summary = benchmark_results.get('summary', {})
    report_lines.append(f"Total Methods: {summary.get('total_methods', 0)}")
    report_lines.append(f"Total Test Cases: {summary.get('total_test_cases', 0)}")
    report_lines.append("")
    
    # Method summaries
    method_summaries = summary.get('method_summaries', {})
    for method_name, method_summary in method_summaries.items():
        report_lines.append(f"Method: {method_name}")
        report_lines.append("-" * 40)
        
        # Accuracy
        accuracy = method_summary.get('accuracy', {})
        if accuracy:
            report_lines.append(f"  Success Rate: {accuracy.get('success_rate', 0):.2%}")
            report_lines.append(f"  Mean L2 Error: {accuracy.get('mean_l2_error', np.nan):.2e}")
            report_lines.append(f"  Mean L∞ Error: {accuracy.get('mean_linf_error', np.nan):.2e}")
        
        # Performance
        performance = method_summary.get('performance', {})
        if performance:
            report_lines.append(f"  Mean Execution Time: {performance.get('mean_execution_time', np.nan):.4f}s")
            report_lines.append(f"  Mean Memory Usage: {performance.get('mean_memory_usage', np.nan):.4f}GB")
        
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Benchmark report saved to: {output_file}")
        except Exception as e:
            warnings.warn(f"Failed to save report to {output_file}: {e}")
    
    return report
