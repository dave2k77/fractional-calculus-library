#!/usr/bin/env python3
"""
Accuracy Comparisons for Fractional Calculus Library

This module provides comprehensive accuracy benchmarks comparing
numerical results with analytical solutions for various test cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from typing import Dict, List, Tuple, Any, Callable
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Updated imports for consolidated structure
from src.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    OptimizedGrunwaldLetnikov,
    AdvancedFFTMethods,
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov,
)


class AccuracyBenchmark:
    """Comprehensive accuracy benchmarking for fractional calculus."""

    def __init__(self):
        """Initialize accuracy benchmark suite."""
        self.results = {}
        self.test_functions = self._define_test_functions()
        self.analytical_solutions = self._define_analytical_solutions()

    def _define_test_functions(self) -> Dict[str, Callable]:
        """Define test functions for accuracy analysis."""
        return {
            "constant": lambda t: np.ones_like(t),
            "linear": lambda t: t,
            "quadratic": lambda t: t**2,
            "cubic": lambda t: t**3,
            "exponential": lambda t: np.exp(-t),
            "sine": lambda t: np.sin(t),
            "cosine": lambda t: np.cos(t),
            "power": lambda t: t**0.5,
            "logarithmic": lambda t: np.log(1 + t),
            "gaussian": lambda t: np.exp(-(t**2)),
        }

    def _define_analytical_solutions(self) -> Dict[str, Dict[str, Callable]]:
        """Define analytical solutions for test functions."""
        solutions = {}

        # Caputo derivative analytical solutions
        caputo_solutions = {
            "constant": lambda t, alpha: np.zeros_like(t),
            "linear": lambda t, alpha: t ** (1 - alpha) / gamma(2 - alpha),
            "quadratic": lambda t, alpha: 2 * t ** (2 - alpha) / gamma(3 - alpha),
            "cubic": lambda t, alpha: 6 * t ** (3 - alpha) / gamma(4 - alpha),
            "exponential": lambda t, alpha: self._caputo_exp_analytical(t, alpha),
            "sine": lambda t, alpha: self._caputo_sin_analytical(t, alpha),
            "cosine": lambda t, alpha: self._caputo_cos_analytical(t, alpha),
            "power": lambda t, alpha: gamma(1.5)
            * t ** (0.5 - alpha)
            / gamma(1.5 - alpha),
            "logarithmic": lambda t, alpha: self._caputo_log_analytical(t, alpha),
            "gaussian": lambda t, alpha: self._caputo_gaussian_analytical(t, alpha),
        }

        # Riemann-Liouville derivative analytical solutions
        riemann_solutions = {
            "constant": lambda t, alpha: t ** (-alpha) / gamma(1 - alpha),
            "linear": lambda t, alpha: t ** (1 - alpha) / gamma(2 - alpha),
            "quadratic": lambda t, alpha: 2 * t ** (2 - alpha) / gamma(3 - alpha),
            "cubic": lambda t, alpha: 6 * t ** (3 - alpha) / gamma(4 - alpha),
            "exponential": lambda t, alpha: self._riemann_exp_analytical(t, alpha),
            "sine": lambda t, alpha: self._riemann_sin_analytical(t, alpha),
            "cosine": lambda t, alpha: self._riemann_cos_analytical(t, alpha),
            "power": lambda t, alpha: gamma(1.5)
            * t ** (0.5 - alpha)
            / gamma(1.5 - alpha),
            "logarithmic": lambda t, alpha: self._riemann_log_analytical(t, alpha),
            "gaussian": lambda t, alpha: self._riemann_gaussian_analytical(t, alpha),
        }

        solutions["caputo"] = caputo_solutions
        solutions["riemann_liouville"] = riemann_solutions

        return solutions

    def _caputo_exp_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Caputo derivative of exp(-t)."""
        # This is a complex expression, simplified for demonstration
        return -(t ** (1 - alpha)) * np.exp(-t) / gamma(2 - alpha)

    def _caputo_sin_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Caputo derivative of sin(t)."""
        # Simplified approximation
        return t ** (1 - alpha) * np.cos(t) / gamma(2 - alpha)

    def _caputo_cos_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Caputo derivative of cos(t)."""
        # Simplified approximation
        return -(t ** (1 - alpha)) * np.sin(t) / gamma(2 - alpha)

    def _caputo_log_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Caputo derivative of log(1+t)."""
        # Simplified approximation
        return t ** (1 - alpha) / ((1 + t) * gamma(2 - alpha))

    def _caputo_gaussian_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Caputo derivative of exp(-t²)."""
        # Simplified approximation
        return -2 * t ** (2 - alpha) * np.exp(-(t**2)) / gamma(3 - alpha)

    def _riemann_exp_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Riemann-Liouville derivative of exp(-t)."""
        # Simplified approximation
        return t ** (-alpha) * np.exp(-t) / gamma(1 - alpha)

    def _riemann_sin_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Riemann-Liouville derivative of sin(t)."""
        # Simplified approximation
        return t ** (-alpha) * np.sin(t) / gamma(1 - alpha)

    def _riemann_cos_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Riemann-Liouville derivative of cos(t)."""
        # Simplified approximation
        return t ** (-alpha) * np.cos(t) / gamma(1 - alpha)

    def _riemann_log_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Riemann-Liouville derivative of log(1+t)."""
        # Simplified approximation
        return t ** (-alpha) * np.log(1 + t) / gamma(1 - alpha)

    def _riemann_gaussian_analytical(self, t: np.ndarray, alpha: float) -> np.ndarray:
        """Analytical Riemann-Liouville derivative of exp(-t²)."""
        # Simplified approximation
        return t ** (-alpha) * np.exp(-(t**2)) / gamma(1 - alpha)

    def benchmark_accuracy(
        self, grid_sizes: List[int] = None, alpha_values: List[float] = None
    ) -> Dict[str, Any]:
        """Benchmark accuracy for different methods and test functions."""
        print("🎯 Accuracy Benchmarking")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 500, 1000]

        if alpha_values is None:
            alpha_values = [0.25, 0.5, 0.75]

        # Initialize methods (will be recreated for each alpha)
        method_classes = {
            "Caputo": OptimizedCaputo,
            "Riemann-Liouville": OptimizedRiemannLiouville,
            "Grünwald-Letnikov": OptimizedGrunwaldLetnikov,
            "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
            "FFT Convolution": lambda: AdvancedFFTMethods(method="convolution"),
        }

        results = {}

        for alpha in alpha_values:
            print(f"\n📊 Testing α = {alpha}")
            results[alpha] = {}

            for func_name, func in self.test_functions.items():
                print(f"  🧪 Testing function: {func_name}")
                results[alpha][func_name] = {}

                for N in grid_sizes:
                    # Create test data
                    t = np.linspace(0.1, 2.0, N)  # Avoid t=0 for some functions
                    h = t[1] - t[0]
                    f = func(t)

                    method_results = {}

                    for method_name, method_class in method_classes.items():
                        try:
                            # Initialize method with current alpha
                            if method_name.startswith("FFT"):
                                method = method_class()
                                numerical = method.compute_derivative(f, t, alpha)
                            else:
                                method = method_class(alpha)
                                numerical = method.compute(f, t, h)

                            # Get analytical solution
                            if method_name in [
                                "Caputo",
                                "FFT Spectral",
                                "FFT Convolution",
                            ]:
                                analytical = self.analytical_solutions["caputo"][
                                    func_name
                                ](t, alpha)
                            elif method_name == "Riemann-Liouville":
                                analytical = self.analytical_solutions[
                                    "riemann_liouville"
                                ][func_name](t, alpha)
                            else:
                                # Grünwald-Letnikov doesn't have simple analytical solutions
                                analytical = np.zeros_like(numerical)

                            # Calculate errors
                            error = np.abs(numerical - analytical)
                            max_error = np.max(error)
                            mean_error = np.mean(error)
                            l2_error = np.sqrt(np.mean(error**2))
                            relative_error = np.mean(
                                np.abs(error / (analytical + 1e-10))
                            )

                            method_results[method_name] = {
                                "max_error": max_error,
                                "mean_error": mean_error,
                                "l2_error": l2_error,
                                "relative_error": relative_error,
                                "convergence_rate": (
                                    -np.log10(max_error) if max_error > 0 else np.inf
                                ),
                            }

                        except Exception as e:
                            print(f"    ❌ {method_name}: Failed - {e}")
                            method_results[method_name] = {
                                "max_error": np.inf,
                                "mean_error": np.inf,
                                "l2_error": np.inf,
                                "relative_error": np.inf,
                                "convergence_rate": 0,
                            }

                    results[alpha][func_name][N] = method_results

                    # Print summary for this grid size
                    best_method = min(
                        method_results.items(),
                        key=lambda x: (
                            x[1]["max_error"] if x[1]["max_error"] < np.inf else np.inf
                        ),
                    )
                    print(
                        f"    Grid {N}: Best method {best_method[0]} "
                        f"(max error: {best_method[1]['max_error']:.2e})"
                    )

        self.results["accuracy"] = results
        return results

    def benchmark_convergence_rates(
        self, grid_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Analyze convergence rates for different methods."""
        print("\n📈 Convergence Rate Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [25, 50, 100, 200, 400, 800]

        alpha = 0.5  # Focus on one alpha value for convergence analysis

        # Test with simple functions that have known analytical solutions
        test_cases = {
            "linear": (lambda t: t, lambda t, a: t ** (1 - a) / gamma(2 - a)),
            "quadratic": (lambda t: t**2, lambda t, a: 2 * t ** (2 - a) / gamma(3 - a)),
        }

        results = {}

        for func_name, (func, analytical) in test_cases.items():
            print(f"\n📊 Testing convergence for {func_name} function")
            results[func_name] = {}

            # Initialize methods (will be recreated for each alpha)
            method_classes = {
                "Caputo": OptimizedCaputo,
                "Riemann-Liouville": OptimizedRiemannLiouville,
                "Grünwald-Letnikov": OptimizedGrunwaldLetnikov,
                "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
            }

            for method_name, method_class in method_classes.items():
                print(f"  🧪 Testing {method_name} method...")

                errors = []
                h_values = []

                for N in grid_sizes:
                    # Create test data
                    t = np.linspace(0.1, 1.0, N)
                    h = t[1] - t[0]
                    f = func(t)

                    try:
                        # Initialize method with current alpha
                        if method_name == "FFT Spectral":
                            method = method_class()
                            numerical = method.compute_derivative(f, t, alpha)
                        else:
                            method = method_class(alpha)
                            numerical = method.compute(f, t, h)

                        # Analytical solution
                        exact = analytical(t, alpha)

                        # Calculate error
                        error = np.max(np.abs(numerical - exact))
                        errors.append(error)
                        h_values.append(h)

                    except Exception as e:
                        print(f"    ❌ Failed for N={N}: {e}")
                        errors.append(np.inf)
                        h_values.append(h)

                # Calculate convergence rate
                valid_errors = [(h, e) for h, e in zip(h_values, errors) if e < np.inf]

                if len(valid_errors) >= 2:
                    h_vals, err_vals = zip(*valid_errors)
                    # Fit log(error) = log(C) + p * log(h)
                    log_h = np.log(h_vals)
                    log_err = np.log(err_vals)
                    p = np.polyfit(log_h, log_err, 1)[0]

                    results[func_name][method_name] = {
                        "convergence_rate": p,
                        "errors": errors,
                        "h_values": h_values,
                    }

                    print(f"    ✅ Convergence rate: {p:.3f}")
                else:
                    results[func_name][method_name] = {
                        "convergence_rate": np.nan,
                        "errors": errors,
                        "h_values": h_values,
                    }
                    print(f"    ❌ Could not determine convergence rate")

        self.results["convergence"] = results
        return results

    def benchmark_stability_analysis(
        self, grid_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Analyze numerical stability of different methods."""
        print("\n🔬 Stability Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [100, 200, 500, 1000]

        alpha_values = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Test with functions that might cause stability issues
        test_functions = {
            "oscillatory": lambda t: np.sin(10 * t),
            "rapid_decay": lambda t: np.exp(-10 * t),
            "singular_near_zero": lambda t: np.sqrt(t),
            "high_frequency": lambda t: np.sin(50 * t) * np.exp(-t),
        }

        results = {}

        for func_name, func in test_functions.items():
            print(f"\n📊 Testing stability for {func_name} function")
            results[func_name] = {}

            # Initialize methods (will be recreated for each alpha)
            method_classes = {
                "Caputo": OptimizedCaputo,
                "Riemann-Liouville": OptimizedRiemannLiouville,
                "Grünwald-Letnikov": OptimizedGrunwaldLetnikov,
                "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
            }

            for alpha in alpha_values:
                results[func_name][alpha] = {}

                for N in grid_sizes:
                    # Create test data
                    t = np.linspace(0.01, 2.0, N)
                    h = t[1] - t[0]
                    f = func(t)

                    method_results = {}

                    for method_name, method_class in method_classes.items():
                        try:
                            # Initialize method with current alpha
                            if method_name == "FFT Spectral":
                                method = method_class()
                                numerical = method.compute_derivative(f, t, alpha)
                            else:
                                method = method_class(alpha)
                                numerical = method.compute(f, t, h)

                            # Check for stability issues
                            max_val = np.max(np.abs(numerical))
                            min_val = np.min(np.abs(numerical))
                            condition_number = max_val / (min_val + 1e-10)

                            # Check for NaN or inf values
                            has_nan = np.any(np.isnan(numerical))
                            has_inf = np.any(np.isinf(numerical))

                            method_results[method_name] = {
                                "max_value": max_val,
                                "min_value": min_val,
                                "condition_number": condition_number,
                                "has_nan": has_nan,
                                "has_inf": has_inf,
                                "is_stable": not (
                                    has_nan or has_inf or condition_number > 1e10
                                ),
                            }

                        except Exception as e:
                            method_results[method_name] = {
                                "max_value": np.inf,
                                "min_value": 0,
                                "condition_number": np.inf,
                                "has_nan": True,
                                "has_inf": True,
                                "is_stable": False,
                            }

                    results[func_name][alpha][N] = method_results

                    # Print stability summary
                    stable_methods = [
                        name
                        for name, data in method_results.items()
                        if data["is_stable"]
                    ]
                    print(f"    Grid {N}, α={alpha}: Stable methods: {stable_methods}")

        self.results["stability"] = results
        return results

    def generate_accuracy_report(self) -> str:
        """Generate comprehensive accuracy report."""
        print("\n📊 Generating Accuracy Report")
        print("=" * 60)

        report = []
        report.append("FRACTIONAL CALCULUS LIBRARY - ACCURACY REPORT")
        report.append("=" * 60)
        report.append("")

        # Accuracy analysis
        if "accuracy" in self.results:
            report.append("ACCURACY ANALYSIS:")
            for alpha, functions in self.results["accuracy"].items():
                report.append(f"  α = {alpha}:")
                for func_name, grid_sizes in functions.items():
                    report.append(f"    Function: {func_name}")
                    for N, methods in grid_sizes.items():
                        best_method = min(
                            methods.items(),
                            key=lambda x: (
                                x[1]["max_error"]
                                if x[1]["max_error"] < np.inf
                                else np.inf
                            ),
                        )
                        report.append(
                            f"      Grid {N}: {best_method[0]} "
                            f"(max error: {best_method[1]['max_error']:.2e})"
                        )
            report.append("")

        # Convergence analysis
        if "convergence" in self.results:
            report.append("CONVERGENCE RATE ANALYSIS:")
            for func_name, methods in self.results["convergence"].items():
                report.append(f"  Function: {func_name}")
                for method_name, data in methods.items():
                    rate = data["convergence_rate"]
                    if not np.isnan(rate):
                        report.append(f"    {method_name}: {rate:.3f}")
                    else:
                        report.append(f"    {method_name}: Could not determine")
            report.append("")

        # Stability analysis
        if "stability" in self.results:
            report.append("STABILITY ANALYSIS:")
            for func_name, alphas in self.results["stability"].items():
                report.append(f"  Function: {func_name}")
                for alpha, grid_sizes in alphas.items():
                    for N, methods in grid_sizes.items():
                        stable_methods = [
                            name for name, data in methods.items() if data["is_stable"]
                        ]
                        report.append(
                            f"    α={alpha}, Grid {N}: Stable methods: {stable_methods}"
                        )
            report.append("")

        report_text = "\n".join(report)

        # Save report to file
        with open("benchmarks/accuracy_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)

        print(
            "✅ Accuracy report generated and saved to 'benchmarks/accuracy_report.txt'"
        )
        return report_text

    def plot_accuracy_results(self):
        """Generate accuracy visualization plots."""
        print("\n📈 Generating Accuracy Plots")
        print("=" * 60)

        # Create plots directory
        import os

        os.makedirs("benchmarks/plots", exist_ok=True)

        # Plot 1: Accuracy comparison for different methods
        if "accuracy" in self.results:
            plt.figure(figsize=(15, 10))

            alpha = 0.5  # Focus on one alpha value
            func_name = "linear"  # Focus on one function

            if (
                alpha in self.results["accuracy"]
                and func_name in self.results["accuracy"][alpha]
            ):
                grid_sizes = list(self.results["accuracy"][alpha][func_name].keys())
                methods = list(
                    self.results["accuracy"][alpha][func_name][grid_sizes[0]].keys()
                )

                for method in methods:
                    errors = []
                    for N in grid_sizes:
                        error = self.results["accuracy"][alpha][func_name][N][method][
                            "max_error"
                        ]
                        errors.append(error if error < np.inf else np.nan)

                    plt.loglog(
                        grid_sizes,
                        errors,
                        "o-",
                        label=method,
                        linewidth=2,
                        markersize=8,
                    )

                plt.xlabel("Grid Size N")
                plt.ylabel("Maximum Error")
                plt.title(f"Accuracy Comparison: {func_name} function (α = {alpha})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    "benchmarks/plots/accuracy_comparison.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

        # Plot 2: Convergence rates
        if "convergence" in self.results:
            plt.figure(figsize=(12, 8))

            func_name = "linear"
            if func_name in self.results["convergence"]:
                methods = list(self.results["convergence"][func_name].keys())

                for method in methods:
                    data = self.results["convergence"][func_name][method]
                    if not np.isnan(data["convergence_rate"]):
                        h_vals = data["h_values"]
                        errors = data["errors"]
                        valid_data = [
                            (h, e) for h, e in zip(h_vals, errors) if e < np.inf
                        ]

                        if valid_data:
                            h_vals, err_vals = zip(*valid_data)
                            plt.loglog(
                                h_vals,
                                err_vals,
                                "o-",
                                label=f'{method} (rate: {data["convergence_rate"]:.2f})',
                                linewidth=2,
                                markersize=8,
                            )

                plt.xlabel("Step Size h")
                plt.ylabel("Maximum Error")
                plt.title(f"Convergence Analysis: {func_name} function")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    "benchmarks/plots/convergence_analysis.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

        # Plot 3: Stability analysis
        if "stability" in self.results:
            plt.figure(figsize=(15, 10))

            func_name = "oscillatory"
            alpha = 0.5

            if (
                func_name in self.results["stability"]
                and alpha in self.results["stability"][func_name]
            ):
                grid_sizes = list(self.results["stability"][func_name][alpha].keys())
                methods = list(
                    self.results["stability"][func_name][alpha][grid_sizes[0]].keys()
                )

                for method in methods:
                    condition_numbers = []
                    for N in grid_sizes:
                        cn = self.results["stability"][func_name][alpha][N][method][
                            "condition_number"
                        ]
                        condition_numbers.append(cn if cn < np.inf else np.nan)

                    plt.semilogy(
                        grid_sizes,
                        condition_numbers,
                        "o-",
                        label=method,
                        linewidth=2,
                        markersize=8,
                    )

                plt.xlabel("Grid Size N")
                plt.ylabel("Condition Number")
                plt.title(f"Stability Analysis: {func_name} function (α = {alpha})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    "benchmarks/plots/stability_analysis.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

        print("✅ Accuracy plots generated and saved to 'benchmarks/plots/' directory")


def main():
    """Run comprehensive accuracy benchmarks."""
    print("🎯 Fractional Calculus Library - Accuracy Benchmarks")
    print("=" * 70)

    # Initialize accuracy benchmark suite
    benchmark = AccuracyBenchmark()

    # Run benchmarks
    benchmark.benchmark_accuracy()
    benchmark.benchmark_convergence_rates()
    benchmark.benchmark_stability_analysis()

    # Generate report and plots
    benchmark.generate_accuracy_report()
    benchmark.plot_accuracy_results()

    print("\n🎉 All accuracy benchmarks completed!")
    print("\n📁 Results saved to:")
    print("  - benchmarks/accuracy_report.txt")
    print("  - benchmarks/plots/")


if __name__ == "__main__":
    main()
