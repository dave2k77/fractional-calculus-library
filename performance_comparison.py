#!/usr/bin/env python3
"""
Performance Comparison: Optimized vs Current Methods

This script compares the performance of optimized fractional calculus methods
against the current implementations to measure efficiency gains.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import warnings

warnings.filterwarnings("ignore")

# Import current methods
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative

# Import optimized methods
from src.algorithms.optimized_methods import (
    OptimizedRiemannLiouville,
    OptimizedCaputo,
    OptimizedGrunwaldLetnikov,
    OptimizedFractionalMethods,
)


def test_function(t: float) -> float:
    """Test function: f(t) = t^2 + sin(t)"""
    return t**2 + np.sin(t)


def analytical_caputo_derivative(t: float, alpha: float) -> float:
    """Analytical Caputo derivative of f(t) = t^2 + sin(t)"""
    from scipy.special import gamma

    # For f(t) = t^2 + sin(t), the Caputo derivative is:
    # D^α t^2 = 2*t^(2-α) / Γ(3-α) for α < 2
    # D^α sin(t) = t^(1-α) * E_{2-α,2-α}(-t^2) where E is Mittag-Leffler
    if alpha < 2:
        term1 = 2 * t ** (2 - alpha) / gamma(3 - alpha)
        # Simplified approximation for sin term
        term2 = t ** (1 - alpha) * np.sin(t) / gamma(2 - alpha)
        return term1 + term2
    else:
        return 0.0


class PerformanceBenchmark:
    """Performance benchmarking for fractional calculus methods."""

    def __init__(self):
        self.results = {}

    def benchmark_method(
        self,
        method_name: str,
        method_func: Callable,
        test_params: Dict,
        n_runs: int = 5,
    ) -> Dict:
        """Benchmark a single method."""
        print(f"Benchmarking {method_name}...")

        # Warmup runs
        for _ in range(3):
            try:
                method_func(**test_params)
            except Exception:
                pass

        # Actual benchmark
        execution_times = []
        memory_usage = []

        for run in range(n_runs):
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()

            try:
                result = method_func(**test_params)
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()

                execution_times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)

            except Exception as e:
                print(f"Error in {method_name}: {e}")
                execution_times.append(float("inf"))
                memory_usage.append(0.0)

        return {
            "method": method_name,
            "avg_time": np.mean(execution_times),
            "std_time": np.std(execution_times),
            "avg_memory": np.mean(memory_usage),
            "success_rate": sum(1 for t in execution_times if t != float("inf"))
            / len(execution_times),
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            return 0.0

    def run_comparison(self, alpha: float = 0.5, sizes: List[int] = None) -> Dict:
        """Run comprehensive performance comparison."""
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]

        print(f"Running performance comparison for α = {alpha}")
        print("=" * 60)

        all_results = {}

        for size in sizes:
            print(f"\nTesting with array size: {size}")
            print("-" * 40)

            # Generate test data
            t_max = 10.0
            h = t_max / size
            t_array = np.linspace(0, t_max, size)
            f_array = np.array([test_function(t) for t in t_array])

            # Test parameters
            test_params = {"f": f_array, "t": t_array, "h": h}

            # Current methods
            current_methods = {
                "Caputo (Direct)": lambda **params: CaputoDerivative(
                    alpha, method="direct"
                ).compute(**params),
                "Caputo (L1)": lambda **params: CaputoDerivative(
                    alpha, method="l1"
                ).compute(**params),
                "RL (Direct)": lambda **params: RiemannLiouvilleDerivative(
                    alpha, method="direct"
                ).compute(**params),
                "RL (FFT)": lambda **params: RiemannLiouvilleDerivative(
                    alpha, method="fft"
                ).compute(**params),
                "GL (Direct)": lambda **params: GrunwaldLetnikovDerivative(
                    alpha, method="direct"
                ).compute(**params),
            }

            # Optimized methods
            optimized_methods = {
                "Caputo (L1 Optimized)": lambda **params: OptimizedCaputo(
                    alpha
                ).compute(**params, method="l1"),
                "Caputo (Diethelm-Ford-Freed)": lambda **params: OptimizedCaputo(
                    alpha
                ).compute(**params, method="diethelm_ford_freed"),
                "RL (FFT Optimized)": lambda **params: OptimizedRiemannLiouville(
                    alpha
                ).compute(**params),
                "GL (JAX Optimized)": lambda **params: OptimizedGrunwaldLetnikov(
                    alpha
                ).compute(**params),
            }

            # Benchmark current methods
            current_results = {}
            for name, method in current_methods.items():
                result = self.benchmark_method(name, method, test_params)
                current_results[name] = result

            # Benchmark optimized methods
            optimized_results = {}
            for name, method in optimized_methods.items():
                result = self.benchmark_method(name, method, test_params)
                optimized_results[name] = result

            all_results[size] = {
                "current": current_results,
                "optimized": optimized_results,
            }

        return all_results

    def plot_results(self, results: Dict):
        """Plot performance comparison results."""
        sizes = list(results.keys())

        # Extract data for plotting
        methods = []
        avg_times = []
        std_times = []

        for size in sizes:
            for category in ["current", "optimized"]:
                for method_name, result in results[size][category].items():
                    if method_name not in methods:
                        methods.append(method_name)

                    # Find the corresponding result for this method and size
                    if method_name in results[size][category]:
                        avg_times.append(
                            results[size][category][method_name]["avg_time"]
                        )
                        std_times.append(
                            results[size][category][method_name]["std_time"]
                        )
                    else:
                        avg_times.append(float("inf"))
                        std_times.append(0.0)

        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time comparison
        current_methods = [m for m in methods if "Optimized" not in m]
        optimized_methods = [m for m in methods if "Optimized" in m]

        # Plot current methods
        for method in current_methods:
            times = []
            for size in sizes:
                if method in results[size]["current"]:
                    times.append(results[size]["current"][method]["avg_time"])
                else:
                    times.append(float("inf"))
            ax1.plot(sizes, times, "o-", label=method, alpha=0.7)

        # Plot optimized methods
        for method in optimized_methods:
            times = []
            for size in sizes:
                if method in results[size]["optimized"]:
                    times.append(results[size]["optimized"][method]["avg_time"])
                else:
                    times.append(float("inf"))
            ax1.plot(sizes, times, "s-", label=method, linewidth=2)

        ax1.set_xlabel("Array Size")
        ax1.set_ylabel("Execution Time (seconds)")
        ax1.set_title("Performance Comparison: Execution Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Speedup comparison
        speedups = {}
        for size in sizes:
            for method in current_methods:
                if method in results[size]["current"]:
                    current_time = results[size]["current"][method]["avg_time"]

                    # Find corresponding optimized method
                    optimized_name = (
                        method.replace(" (Direct)", " (L1 Optimized)")
                        .replace(" (FFT)", " (FFT Optimized)")
                        .replace(" (L1)", " (L1 Optimized)")
                    )

                    if optimized_name in results[size]["optimized"]:
                        optimized_time = results[size]["optimized"][optimized_name][
                            "avg_time"
                        ]
                        speedup = (
                            current_time / optimized_time if optimized_time > 0 else 1.0
                        )

                        if method not in speedups:
                            speedups[method] = []
                        speedups[method].append(speedup)

        # Plot speedups
        for method, speedup_list in speedups.items():
            if len(speedup_list) == len(sizes):
                ax2.plot(sizes, speedup_list, "o-", label=method, linewidth=2)

        ax2.set_xlabel("Array Size")
        ax2.set_ylabel("Speedup Factor")
        ax2.set_title("Performance Improvement: Speedup Factor")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def print_summary(self, results: Dict):
        """Print summary of performance comparison."""
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 80)

        for size in results.keys():
            print(f"\nArray Size: {size}")
            print("-" * 40)

            # Current methods
            print("Current Methods:")
            for method_name, result in results[size]["current"].items():
                print(
                    f"  {method_name:30s}: {result['avg_time']:8.4f}s ± {result['std_time']:6.4f}s"
                )

            print("\nOptimized Methods:")
            for method_name, result in results[size]["optimized"].items():
                print(
                    f"  {method_name:30s}: {result['avg_time']:8.4f}s ± {result['std_time']:6.4f}s"
                )

            # Calculate speedups
            print("\nSpeedup Factors:")
            for method in results[size]["current"].keys():
                current_time = results[size]["current"][method]["avg_time"]

                # Find corresponding optimized method
                optimized_name = (
                    method.replace(" (Direct)", " (L1 Optimized)")
                    .replace(" (FFT)", " (FFT Optimized)")
                    .replace(" (L1)", " (L1 Optimized)")
                )

                if optimized_name in results[size]["optimized"]:
                    optimized_time = results[size]["optimized"][optimized_name][
                        "avg_time"
                    ]
                    speedup = (
                        current_time / optimized_time if optimized_time > 0 else 1.0
                    )
                    print(f"  {method:30s}: {speedup:6.2f}x faster")


def main():
    """Main function to run performance comparison."""
    print("Fractional Calculus Performance Comparison")
    print("Testing optimized methods vs current implementations")
    print("=" * 60)

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Test different fractional orders
    alphas = [0.3, 0.5, 0.7]

    for alpha in alphas:
        print(f"\nTesting with α = {alpha}")

        # Run comparison
        results = benchmark.run_comparison(alpha=alpha, sizes=[100, 500, 1000, 2000])

        # Print summary
        benchmark.print_summary(results)

        # Plot results
        benchmark.plot_results(results)

        print(f"\nResults saved as 'performance_comparison_alpha_{alpha}.png'")


if __name__ == "__main__":
    main()
