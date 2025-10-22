"""
Dedicated benchmarking module for hpfracc library.

This module provides a systematic way to run different types of benchmarks:
- Performance benchmarks (speedup comparisons)
- Accuracy benchmarks (error analysis)
- Scaling benchmarks (performance vs array size)
- Memory usage benchmarks
- GPU vs CPU comparisons
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    array_size: int
    execution_time: float
    memory_usage: float
    accuracy: Optional[float] = None
    speedup: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    array_sizes: List[int] = None
    methods: List[str] = None
    fractional_orders: List[float] = None
    test_functions: List[str] = None
    iterations: int = 3
    warmup_runs: int = 2
    use_gpu: bool = False
    parallel: bool = False
    save_results: bool = True
    output_dir: str = "benchmark_results"

    def __post_init__(self):
        if self.array_sizes is None:
            self.array_sizes = [100, 500, 1000, 2000, 5000]
        if self.methods is None:
            self.methods = ["RL", "GL", "Caputo",
                            "Weyl", "Marchaud", "Hadamard"]
        if self.fractional_orders is None:
            self.fractional_orders = [0.25, 0.5, 0.75]
        if self.test_functions is None:
            self.test_functions = ["polynomial",
                                   "exponential", "trigonometric"]


class BenchmarkRunner:
    """Main benchmarking class for running various types of benchmarks."""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.test_functions = self._setup_test_functions()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _setup_test_functions(self) -> Dict[str, Callable]:
        """Setup test functions for benchmarking."""
        return {
            "polynomial": lambda t: t**2 + 2 * t + 1,
            "exponential": lambda t: np.exp(-t) * np.sin(t),
            "trigonometric": lambda t: np.sin(2 * np.pi * t) + np.cos(np.pi * t),
            "rational": lambda t: 1 / (1 + t**2),
            "logarithmic": lambda t: np.log(1 + t),
            "power": lambda t: t**0.5 + t**1.5
        }

    def run_performance_benchmark(self) -> List[BenchmarkResult]:
        """Run performance benchmarks comparing different methods."""
        logger.info("Starting performance benchmark...")

        results = []

        for array_size in self.config.array_sizes:
            logger.info(f"Testing array size: {array_size}")

            # Generate test data
            t = np.linspace(0, 10, array_size)
            h = t[1] - t[0]

            for alpha in self.config.fractional_orders:
                for func_name, test_func in self.test_functions.items():
                    if func_name == "polynomial":  # Use polynomial for main comparison
                        for method in self.config.methods:
                            result = self._benchmark_method(
                                method, test_func, t, alpha, h, func_name
                            )
                            results.append(result)

        self.results.extend(results)
        return results

    def run_accuracy_benchmark(self) -> List[BenchmarkResult]:
        """Run accuracy benchmarks comparing against analytical solutions."""
        logger.info("Starting accuracy benchmark...")

        results = []

        # Use smaller arrays for accuracy tests
        array_sizes = [50, 100, 200]

        for array_size in array_sizes:
            t = np.linspace(0.1, 2.0, array_size)
            h = t[1] - t[0]

            for alpha in [0.5]:  # Focus on alpha = 0.5 for accuracy
                # Test with known analytical solutions
                from scipy.special import gamma
                test_cases = [
                    ("power", lambda t: t**2, lambda t,
                     a: 2 * t**(2 - a) / gamma(3 - a)),
                    ("exponential", lambda t: np.exp(t), lambda t,
                     a: np.exp(t) * t**(-a) / gamma(1 - a))
                ]

                for func_name, test_func, analytical in test_cases:
                    for method in self.config.methods:
                        result = self._benchmark_accuracy(
                            method, test_func, analytical, t, alpha, h, func_name)
                        results.append(result)

        self.results.extend(results)
        return results

    def run_scaling_benchmark(self) -> List[BenchmarkResult]:
        """Run scaling benchmarks to analyze performance vs array size."""
        logger.info("Starting scaling benchmark...")

        results = []
        array_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]

        for array_size in array_sizes:
            t = np.linspace(0, 10, array_size)
            h = t[1] - t[0]
            alpha = 0.5

            for method in self.config.methods:
                result = self._benchmark_scaling(
                    method, self.test_functions["polynomial"], t, alpha, h
                )
                results.append(result)

        self.results.extend(results)
        return results

    def run_memory_benchmark(self) -> List[BenchmarkResult]:
        """Run memory usage benchmarks."""
        logger.info("Starting memory benchmark...")

        results = []
        array_sizes = [1000, 5000, 10000, 20000]

        for array_size in array_sizes:
            t = np.linspace(0, 10, array_size)
            h = t[1] - t[0]
            alpha = 0.5

            for method in self.config.methods:
                result = self._benchmark_memory(
                    method, self.test_functions["polynomial"], t, alpha, h
                )
                results.append(result)

        self.results.extend(results)
        return results

    def _benchmark_method(self, method: str, test_func: Callable,
                          t: np.ndarray, alpha: float, h: float,
                          func_name: str) -> BenchmarkResult:
        """Benchmark a specific method."""
        try:
            # Import method dynamically
            method_func = self._get_method_function(method)

            # Warmup runs
            for _ in range(self.config.warmup_runs):
                _ = method_func(test_func, t, alpha, h)

            # Actual benchmark runs
            times = []
            memory_usage = []

            for _ in range(self.config.iterations):
                gc.collect()  # Force garbage collection
                process = psutil.Process()
                mem_before = process.memory_info().rss / 1024 / 1024  # MB

                start_time = time.time()
                method_func(test_func, t, alpha, h)
                end_time = time.time()

                mem_after = process.memory_info().rss / 1024 / 1024  # MB

                times.append(end_time - start_time)
                memory_usage.append(mem_after - mem_before)

            # Calculate statistics
            avg_time = np.mean(times)
            avg_memory = np.mean(memory_usage)

            return BenchmarkResult(
                method_name=method,
                array_size=len(t),
                execution_time=avg_time,
                memory_usage=avg_memory,
                parameters={"alpha": alpha, "function": func_name},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            logger.error(f"Error benchmarking {method}: {e}")
            return BenchmarkResult(
                method_name=method,
                array_size=len(t),
                execution_time=float('inf'),
                memory_usage=0.0,
                parameters={"alpha": alpha,
                            "function": func_name, "error": str(e)}
            )

    def _benchmark_accuracy(
            self,
            method: str,
            test_func: Callable,
            analytical: Callable,
            t: np.ndarray,
            alpha: float,
            h: float,
            func_name: str) -> BenchmarkResult:
        """Benchmark accuracy against analytical solutions."""
        try:
            method_func = self._get_method_function(method)

            # Compute numerical result
            numerical_result = method_func(test_func, t, alpha, h)

            # Compute analytical result
            analytical_result = analytical(t, alpha)

            # Calculate relative error
            relative_error = np.mean(
                np.abs(
                    (numerical_result -
                     analytical_result) /
                    analytical_result))

            # Measure execution time
            start_time = time.time()
            _ = method_func(test_func, t, alpha, h)
            execution_time = time.time() - start_time

            return BenchmarkResult(
                method_name=method,
                array_size=len(t),
                execution_time=execution_time,
                memory_usage=0.0,  # Not measured for accuracy
                accuracy=relative_error,
                parameters={"alpha": alpha, "function": func_name,
                            "test_type": "accuracy"}
            )

        except Exception as e:
            logger.error(f"Error in accuracy benchmark for {method}: {e}")
            return BenchmarkResult(
                method_name=method,
                array_size=len(t),
                execution_time=float('inf'),
                memory_usage=0.0,
                accuracy=float('inf'),
                parameters={"alpha": alpha,
                            "function": func_name, "error": str(e)}
            )

    def _benchmark_scaling(
            self,
            method: str,
            test_func: Callable,
            t: np.ndarray,
            alpha: float,
            h: float) -> BenchmarkResult:
        """Benchmark scaling performance."""
        return self._benchmark_method(
            method, test_func, t, alpha, h, "scaling")

    def _benchmark_memory(
            self,
            method: str,
            test_func: Callable,
            t: np.ndarray,
            alpha: float,
            h: float) -> BenchmarkResult:
        """Benchmark memory usage."""
        return self._benchmark_method(method, test_func, t, alpha, h, "memory")

    def _get_method_function(self, method: str) -> Callable:
        """Get the actual method function from the hpfracc library."""
        try:
            if method == "RL":
                from hpfracc.algorithms.optimized_methods import optimized_riemann_liouville
                return optimized_riemann_liouville
            elif method == "GL":
                from hpfracc.algorithms.optimized_methods import optimized_grunwald_letnikov
                return optimized_grunwald_letnikov
            elif method == "Caputo":
                from hpfracc.algorithms.optimized_methods import optimized_caputo
                return optimized_caputo
            elif method == "Weyl":
                # Note: optimized_weyl_derivative function may need to be implemented
                from hpfracc.algorithms.optimized_methods import OptimizedRiemannLiouville
                return lambda f, t, alpha, h: OptimizedRiemannLiouville(alpha).compute(f, t, h)
            elif method == "Marchaud":
                # Note: optimized_marchaud_derivative function may need to be implemented
                from hpfracc.algorithms.optimized_methods import OptimizedCaputo
                return lambda f, t, alpha, h: OptimizedCaputo(alpha).compute(f, t, h)
            elif method == "Hadamard":
                # Note: optimized_hadamard_derivative function may need to be implemented
                from hpfracc.algorithms.optimized_methods import OptimizedCaputo
                return lambda f, t, alpha, h: OptimizedCaputo(alpha).compute(f, t, h)
            else:
                raise ValueError(f"Unknown method: {method}")
        except ImportError as e:
            logger.error(f"Could not import method {method}: {e}")
            raise

    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmark types."""
        logger.info("Running all benchmarks...")

        results = {}

        # Performance benchmark
        results["performance"] = self.run_performance_benchmark()

        # Accuracy benchmark
        results["accuracy"] = self.run_accuracy_benchmark()

        # Scaling benchmark
        results["scaling"] = self.run_scaling_benchmark()

        # Memory benchmark
        results["memory"] = self.run_memory_benchmark()

        if self.config.save_results:
            self.save_results(results)

        return results

    def save_results(self, results: Dict[str, List[BenchmarkResult]]):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self._results_to_dict(results), f, indent=2)

        # Save as CSV
        csv_file = self.output_dir / f"benchmark_results_{timestamp}.csv"
        self._save_results_csv(results, csv_file)

        logger.info(f"Results saved to {self.output_dir}")

    def _results_to_dict(
            self, results: Dict[str, List[BenchmarkResult]]) -> Dict:
        """Convert results to dictionary format for JSON serialization."""
        output = {}
        for benchmark_type, result_list in results.items():
            output[benchmark_type] = []
            for result in result_list:
                output[benchmark_type].append({
                    "method_name": result.method_name,
                    "array_size": result.array_size,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "accuracy": result.accuracy,
                    "speedup": result.speedup,
                    "parameters": result.parameters,
                    "timestamp": result.timestamp
                })
        return output

    def _save_results_csv(self,
                          results: Dict[str,
                                        List[BenchmarkResult]],
                          csv_file: Path):
        """Save results as CSV file."""
        import pandas as pd

        all_results = []
        for benchmark_type, result_list in results.items():
            for result in result_list:
                row = {
                    "benchmark_type": benchmark_type,
                    "method_name": result.method_name,
                    "array_size": result.array_size,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "accuracy": result.accuracy,
                    "speedup": result.speedup,
                    "alpha": result.parameters.get("alpha", None),
                    "function": result.parameters.get("function", None),
                    "timestamp": result.timestamp
                }
                all_results.append(row)

        df = pd.DataFrame(all_results)
        df.to_csv(csv_file, index=False)

    def generate_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate a comprehensive benchmark report."""
        logger.info("Generating benchmark report...")

        # Create plots
        self._plot_performance_results(results.get("performance", []))
        self._plot_scaling_results(results.get("scaling", []))
        self._plot_accuracy_results(results.get("accuracy", []))
        self._plot_memory_results(results.get("memory", []))

        # Generate summary statistics
        self._generate_summary_stats(results)

    def _plot_performance_results(self, results: List[BenchmarkResult]):
        """Plot performance benchmark results."""
        if not results:
            return

        # Group by array size and method
        data = {}
        for result in results:
            if result.execution_time != float('inf'):
                size = result.array_size
                method = result.method_name
                if size not in data:
                    data[size] = {}
                if method not in data[size]:
                    data[size][method] = []
                data[size][method].append(result.execution_time)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        methods = list(set(r.method_name for r in results))
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        for i, method in enumerate(methods):
            sizes = []
            times = []
            for size in sorted(data.keys()):
                if method in data[size]:
                    sizes.append(size)
                    times.append(np.mean(data[size][method]))

            if sizes:
                ax.plot(sizes, times, 'o-', label=method,
                        color=colors[i], linewidth=2, markersize=8)

        ax.set_xlabel('Array Size')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Performance Benchmark Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_benchmark.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_scaling_results(self, results: List[BenchmarkResult]):
        """Plot scaling benchmark results."""
        if not results:
            return

        # Similar to performance but focus on scaling
        self._plot_performance_results(results)

    def _plot_accuracy_results(self, results: List[BenchmarkResult]):
        """Plot accuracy benchmark results."""
        if not results:
            return

        # Group by method
        data = {}
        for result in results:
            if result.accuracy is not None and result.accuracy != float('inf'):
                method = result.method_name
                if method not in data:
                    data[method] = []
                data[method].append(result.accuracy)

        if not data:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(data.keys())
        accuracies = [np.mean(data[method]) for method in methods]

        bars = ax.bar(methods, accuracies, color=plt.cm.viridis(
            np.linspace(0, 1, len(methods))))

        ax.set_xlabel('Method')
        ax.set_ylabel('Relative Error')
        ax.set_title('Accuracy Benchmark Results')
        ax.set_yscale('log')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{acc:.2e}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_benchmark.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_memory_results(self, results: List[BenchmarkResult]):
        """Plot memory usage benchmark results."""
        if not results:
            return

        # Group by array size and method
        data = {}
        for result in results:
            if result.memory_usage > 0:
                size = result.array_size
                method = result.method_name
                if size not in data:
                    data[size] = {}
                if method not in data[size]:
                    data[size][method] = []
                data[size][method].append(result.memory_usage)

        fig, ax = plt.subplots(figsize=(12, 8))

        methods = list(set(r.method_name for r in results))
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        for i, method in enumerate(methods):
            sizes = []
            memory = []
            for size in sorted(data.keys()):
                if method in data[size]:
                    sizes.append(size)
                    memory.append(np.mean(data[size][method]))

            if sizes:
                ax.plot(sizes, memory, 'o-', label=method,
                        color=colors[i], linewidth=2, markersize=8)

        ax.set_xlabel('Array Size')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory Usage Benchmark Results')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_benchmark.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_stats(
            self, results: Dict[str, List[BenchmarkResult]]):
        """Generate summary statistics."""
        summary_file = self.output_dir / 'benchmark_summary.txt'

        with open(summary_file, 'w') as f:
            f.write("HPFRACC Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")

            for benchmark_type, result_list in results.items():
                f.write(f"{benchmark_type.upper()} BENCHMARK\n")
                f.write("-" * 30 + "\n")

                if not result_list:
                    f.write("No results available\n\n")
                    continue

                # Group by method
                method_stats = {}
                for result in result_list:
                    method = result.method_name
                    if method not in method_stats:
                        method_stats[method] = []
                    method_stats[method].append(result)

                for method, method_results in method_stats.items():
                    f.write(f"\nMethod: {method}\n")

                    # Performance stats
                    times = [
                        r.execution_time for r in method_results if r.execution_time != float('inf')]
                    if times:
                        f.write(
                            f"  Execution Time: {np.mean(times):.6f}s ± {np.std(times):.6f}s\n")

                    # Memory stats
                    memory = [
                        r.memory_usage for r in method_results if r.memory_usage > 0]
                    if memory:
                        f.write(
                            f"  Memory Usage: {np.mean(memory):.2f}MB ± {np.std(memory):.2f}MB\n")

                    # Accuracy stats
                    accuracy = [
                        r.accuracy for r in method_results if r.accuracy is not None and r.accuracy != float('inf')]
                    if accuracy:
                        f.write(
                            f"  Relative Error: {np.mean(accuracy):.2e} ± {np.std(accuracy):.2e}\n")

                f.write("\n" + "=" * 50 + "\n\n")


def run_quick_benchmark():
    """Quick benchmark for development testing."""
    config = BenchmarkConfig(
        array_sizes=[100, 500, 1000],
        methods=["RL", "GL", "Caputo"],
        fractional_orders=[0.5],
        test_functions=["polynomial"],
        iterations=2,
        warmup_runs=1
    )

    runner = BenchmarkRunner(config)
    results = runner.run_performance_benchmark()

    print("Quick Benchmark Results:")
    print("=" * 50)
    for result in results:
        print(f"{result.method_name:8s} | Size: {result.array_size:4d} | "
              f"Time: {result.execution_time:.6f}s | "
              f"Memory: {result.memory_usage:.2f}MB")

    return results


if __name__ == "__main__":
    # Run quick benchmark if script is executed directly
    run_quick_benchmark()
