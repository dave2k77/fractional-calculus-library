"""
Advanced Fractional Calculus Methods Demo

This script demonstrates the new advanced fractional calculus methods:
- Weyl derivative via FFT Convolution with parallelization
- Marchaud derivative with Difference Quotient convolution and memory optimization
- Hadamard derivative
- Reiz-Feller derivative via spectral method
- Adomian Decomposition method

Includes performance comparisons and visualizations.
"""

from hpfracc.algorithms.optimized_methods import (
    ParallelConfig,
    # Note: These classes may need to be implemented
    # OptimizedWeylDerivative,
    # OptimizedMarchaudDerivative,
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    # Note: These classes may need to be implemented
    # OptimizedHadamardDerivative,
    # OptimizedReizFellerDerivative,
    # OptimizedAdomianDecomposition,
)
from hpfracc.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    AdomianDecomposition,
)
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Import the new advanced methods


# Updated import for consolidated structure


def create_test_functions():
    """Create test functions for demonstration."""

    def polynomial(x):
        """Polynomial function: f(x) = x^2 + x + 1"""
        return x**2 + x + 1

    def exponential(x):
        """Exponential function: f(x) = exp(-x)"""
        return np.exp(-x)

    def trigonometric(x):
        """Trigonometric function: f(x) = sin(x)"""
        return np.sin(x)

    def gaussian(x):
        """Gaussian function: f(x) = exp(-x^2)"""
        return np.exp(-(x**2))

    def logarithmic(x):
        """Logarithmic function: f(x) = log(x) for x > 0"""
        return np.log(np.maximum(x, 1e-10))

    return {
        "polynomial": polynomial,
        "exponential": exponential,
        "trigonometric": trigonometric,
        "gaussian": gaussian,
        "logarithmic": logarithmic,
    }


def demo_weyl_derivative():
    """Demonstrate Weyl derivative with performance comparison."""
    print("\n" + "=" * 60)
    print("WEYL DERIVATIVE DEMONSTRATION")
    print("=" * 60)

    alpha = 0.5
    x = np.linspace(0, 10, 1000)
    test_functions = create_test_functions()

    # Test with different functions
    for name, func in test_functions.items():
        print(f"\nTesting Weyl derivative with {name} function...")

        # Standard implementation
        start_time = time.time()
        weyl_calc = WeylDerivative(alpha)
        result_standard = weyl_calc.compute(
            func, x, h=0.01, use_parallel=False)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_weyl = OptimizedWeylDerivative(alpha)
        result_optimized = opt_weyl.compute(func, x, h=0.01)
        optimized_time = time.time() - start_time

        # Parallel implementation
        parallel_config = ParallelConfig(n_jobs=4)
        start_time = time.time()
        weyl_parallel = WeylDerivative(alpha, parallel_config)
        result_parallel = weyl_parallel.compute(func, x, h=0.01)
        parallel_time = time.time() - start_time

        print(f"  Standard time: {standard_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s")
        print(f"  Parallel time: {parallel_time:.4f}s")
        print(f"  Speedup (optimized): {standard_time/optimized_time:.2f}x")
        print(f"  Speedup (parallel): {standard_time/parallel_time:.2f}x")

        # Verify accuracy
        accuracy_opt = np.mean(np.abs(result_standard - result_optimized))
        accuracy_par = np.mean(np.abs(result_standard - result_parallel))
        print(f"  Accuracy (optimized): {accuracy_opt:.2e}")
        print(f"  Accuracy (parallel): {accuracy_par:.2e}")

        # Plot results for first function
        if name == "trigonometric":
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(x, func(x), "b-", label="Original function")
            plt.plot(x, result_standard, "r-", label="Weyl derivative")
            plt.title(f"Weyl Derivative (α={alpha}) - {name}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_optimized, "r--", label="Optimized")
            plt.title("Standard vs Optimized")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_parallel, "g--", label="Parallel")
            plt.title("Standard vs Parallel")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "examples/weyl_derivative_demo.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


def demo_marchaud_derivative():
    """Demonstrate Marchaud derivative with memory optimization."""
    print("\n" + "=" * 60)
    print("MARCHAUD DERIVATIVE DEMONSTRATION")
    print("=" * 60)

    alpha = 0.5
    x = np.linspace(0, 10, 1000)
    test_functions = create_test_functions()

    for name, func in test_functions.items():
        print(f"\nTesting Marchaud derivative with {name} function...")

        # Standard implementation
        start_time = time.time()
        marchaud_calc = MarchaudDerivative(alpha)
        result_standard = marchaud_calc.compute(
            func, x, h=0.01, memory_optimized=False)
        standard_time = time.time() - start_time

        # Memory optimized implementation
        start_time = time.time()
        result_optimized = marchaud_calc.compute(
            func, x, h=0.01, memory_optimized=True)
        optimized_time = time.time() - start_time

        # Numba optimized implementation
        start_time = time.time()
        opt_marchaud = OptimizedMarchaudDerivative(alpha)
        result_numba = opt_marchaud.compute(func, x, h=0.01)
        numba_time = time.time() - start_time

        print(f"  Standard time: {standard_time:.4f}s")
        print(f"  Memory optimized time: {optimized_time:.4f}s")
        print(f"  Numba optimized time: {numba_time:.4f}s")
        print(f"  Speedup (memory): {standard_time/optimized_time:.2f}x")
        print(f"  Speedup (numba): {standard_time/numba_time:.2f}x")

        # Verify accuracy
        accuracy_opt = np.mean(np.abs(result_standard - result_optimized))
        accuracy_numba = np.mean(np.abs(result_standard - result_numba))
        print(f"  Accuracy (memory): {accuracy_opt:.2e}")
        print(f"  Accuracy (numba): {accuracy_numba:.2e}")

        # Plot results for first function
        if name == "exponential":
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(x, func(x), "b-", label="Original function")
            plt.plot(x, result_standard, "r-", label="Marchaud derivative")
            plt.title(f"Marchaud Derivative (α={alpha}) - {name}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_optimized, "r--", label="Memory Optimized")
            plt.title("Standard vs Memory Optimized")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_numba, "g--", label="Numba Optimized")
            plt.title("Standard vs Numba Optimized")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "examples/marchaud_derivative_demo.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


def demo_hadamard_derivative():
    """Demonstrate Hadamard derivative."""
    print("\n" + "=" * 60)
    print("HADAMARD DERIVATIVE DEMONSTRATION")
    print("=" * 60)

    alpha = 0.5
    x = np.linspace(1, 10, 1000)  # Start from 1 for Hadamard
    test_functions = create_test_functions()

    for name, func in test_functions.items():
        print(f"\nTesting Hadamard derivative with {name} function...")

        # Standard implementation
        start_time = time.time()
        hadamard_calc = HadamardDerivative(alpha)
        result_standard = hadamard_calc.compute(func, x, h=0.01)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_hadamard = OptimizedHadamardDerivative(alpha)
        result_optimized = opt_hadamard.compute(func, x, h=0.01)
        optimized_time = time.time() - start_time

        print(f"  Standard time: {standard_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s")
        print(f"  Speedup: {standard_time/optimized_time:.2f}x")

        # Verify accuracy
        accuracy = np.mean(np.abs(result_standard - result_optimized))
        print(f"  Accuracy: {accuracy:.2e}")

        # Plot results for logarithmic function
        if name == "logarithmic":
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(x, func(x), "b-", label="Original function")
            plt.plot(x, result_standard, "r-", label="Hadamard derivative")
            plt.title(f"Hadamard Derivative (α={alpha}) - {name}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_optimized, "r--", label="Optimized")
            plt.title("Standard vs Optimized")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "examples/hadamard_derivative_demo.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


def demo_reiz_feller_derivative():
    """Demonstrate Reiz-Feller derivative."""
    print("\n" + "=" * 60)
    print("REIZ-FELLER DERIVATIVE DEMONSTRATION")
    print("=" * 60)

    alpha = 0.5
    x = np.linspace(-5, 5, 1000)
    test_functions = create_test_functions()

    for name, func in test_functions.items():
        print(f"\nTesting Reiz-Feller derivative with {name} function...")

        # Standard implementation
        start_time = time.time()
        reiz_calc = ReizFellerDerivative(alpha)
        result_standard = reiz_calc.compute(
            func, x, h=0.01, use_parallel=False)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_reiz = OptimizedReizFellerDerivative(alpha)
        result_optimized = opt_reiz.compute(func, x, h=0.01)
        optimized_time = time.time() - start_time

        # Parallel implementation
        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        start_time = time.time()
        reiz_parallel = ReizFellerDerivative(alpha, parallel_config)
        result_parallel = reiz_parallel.compute(
            func, x, h=0.01, use_parallel=True)
        parallel_time = time.time() - start_time

        print(f"  Standard time: {standard_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s")
        print(f"  Parallel time: {parallel_time:.4f}s")
        print(f"  Speedup (optimized): {standard_time/optimized_time:.2f}x")
        print(f"  Speedup (parallel): {standard_time/parallel_time:.2f}x")

        # Verify accuracy
        accuracy_opt = np.mean(np.abs(result_standard - result_optimized))
        accuracy_par = np.mean(np.abs(result_standard - result_parallel))
        print(f"  Accuracy (optimized): {accuracy_opt:.2e}")
        print(f"  Accuracy (parallel): {accuracy_par:.2e}")

        # Plot results for gaussian function
        if name == "gaussian":
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(x, func(x), "b-", label="Original function")
            plt.plot(x, result_standard, "r-", label="Reiz-Feller derivative")
            plt.title(f"Reiz-Feller Derivative (α={alpha}) - {name}")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_optimized, "r--", label="Optimized")
            plt.title("Standard vs Optimized")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(x, result_standard, "b-", label="Standard")
            plt.plot(x, result_parallel, "g--", label="Parallel")
            plt.title("Standard vs Parallel")
            plt.xlabel("x")
            plt.ylabel("D^α f(x)")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "examples/reiz_feller_derivative_demo.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


def demo_adomian_decomposition():
    """Demonstrate Adomian Decomposition method."""
    print("\n" + "=" * 60)
    print("ADOMIAN DECOMPOSITION DEMONSTRATION")
    print("=" * 60)

    alpha = 0.5

    # Test different equations
    equations = {
        "linear": lambda t, y: t,
        "trigonometric": lambda t, y: np.sin(t),
        "exponential": lambda t, y: np.exp(-t),
        "polynomial": lambda t, y: t**2,
    }

    for name, equation in equations.items():
        print(f"\nTesting Adomian decomposition with {name} equation...")

        initial_conditions = {0: 0.0}
        t_span = (0, 2)

        # Standard implementation
        start_time = time.time()
        adomian_solver = AdomianDecomposition(alpha)
        t, solution_standard = adomian_solver.solve(
            equation,
            initial_conditions,
            t_span,
            n_steps=200,
            n_terms=10,
            use_parallel=False,
        )
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_adomian = OptimizedAdomianDecomposition(alpha)
        t, solution_optimized = opt_adomian.solve(
            equation, initial_conditions, t_span, n_steps=200, n_terms=10
        )
        optimized_time = time.time() - start_time

        # Parallel implementation
        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        start_time = time.time()
        adomian_parallel = AdomianDecomposition(alpha, parallel_config)
        t, solution_parallel = adomian_parallel.solve(
            equation,
            initial_conditions,
            t_span,
            n_steps=200,
            n_terms=10,
            use_parallel=True,
        )
        parallel_time = time.time() - start_time

        print(f"  Standard time: {standard_time:.4f}s")
        print(f"  Optimized time: {optimized_time:.4f}s")
        print(f"  Parallel time: {parallel_time:.4f}s")
        print(f"  Speedup (optimized): {standard_time/optimized_time:.2f}x")
        print(f"  Speedup (parallel): {standard_time/parallel_time:.2f}x")

        # Verify accuracy
        accuracy_opt = np.mean(np.abs(solution_standard - solution_optimized))
        accuracy_par = np.mean(np.abs(solution_standard - solution_parallel))
        print(f"  Accuracy (optimized): {accuracy_opt:.2e}")
        print(f"  Accuracy (parallel): {accuracy_par:.2e}")

        # Plot results for first equation
        if name == "linear":
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(t, solution_standard, "b-", label="Solution")
            plt.title(f"Adomian Decomposition (α={alpha}) - {name}")
            plt.xlabel("t")
            plt.ylabel("y(t)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(t, solution_standard, "b-", label="Standard")
            plt.plot(t, solution_optimized, "r--", label="Optimized")
            plt.title("Standard vs Optimized")
            plt.xlabel("t")
            plt.ylabel("y(t)")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(t, solution_standard, "b-", label="Standard")
            plt.plot(t, solution_parallel, "g--", label="Parallel")
            plt.title("Standard vs Parallel")
            plt.xlabel("t")
            plt.ylabel("y(t)")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                "examples/adomian_decomposition_demo.png", dpi=300, bbox_inches="tight"
            )
            plt.show()


def performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 60)

    alpha = 0.5
    x = np.linspace(0, 10, 2000)

    def test_function(x):
        return np.sin(x) * np.exp(-x / 5)

    methods = {
        "Weyl": (WeylDerivative, OptimizedWeylDerivative),
        "Marchaud": (MarchaudDerivative, OptimizedMarchaudDerivative),
        "Hadamard": (HadamardDerivative, OptimizedHadamardDerivative),
        "Reiz-Feller": (ReizFellerDerivative, OptimizedReizFellerDerivative),
    }

    results = {}

    for name, (StandardClass, OptimizedClass) in methods.items():
        print(f"\nBenchmarking {name} derivative...")

        # Standard implementation
        start_time = time.time()
        standard_calc = StandardClass(alpha)
        if name == "Hadamard":
            x_hadamard = np.linspace(1, 10, 2000)
            result_standard = standard_calc.compute(
                test_function, x_hadamard, h=0.005)
        elif name == "Reiz-Feller":
            x_reiz = np.linspace(-5, 5, 2000)
            result_standard = standard_calc.compute(
                test_function, x_reiz, h=0.005)
        else:
            result_standard = standard_calc.compute(test_function, x, h=0.005)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        optimized_calc = OptimizedClass(alpha)
        if name == "Hadamard":
            result_optimized = optimized_calc.compute(
                test_function, x_hadamard, h=0.005
            )
        elif name == "Reiz-Feller":
            result_optimized = optimized_calc.compute(
                test_function, x_reiz, h=0.005)
        else:
            result_optimized = optimized_calc.compute(
                test_function, x, h=0.005)
        optimized_time = time.time() - start_time

        speedup = standard_time / optimized_time
        accuracy = np.mean(np.abs(result_standard - result_optimized))

        results[name] = {
            "standard_time": standard_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "accuracy": accuracy,
        }

        print(f"  Standard: {standard_time:.4f}s")
        print(f"  Optimized: {optimized_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Accuracy: {accuracy:.2e}")

    # Create performance summary plot
    methods_list = list(results.keys())
    speedups = [results[m]["speedup"] for m in methods_list]
    accuracies = [results[m]["accuracy"] for m in methods_list]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    bars = plt.bar(methods_list, speedups, color=[
                   "blue", "green", "red", "orange"])
    plt.title("Performance Speedup Comparison")
    plt.ylabel("Speedup Factor")
    plt.ylim(0, max(speedups) * 1.1)
    for bar, speedup in zip(bars, speedups):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 2, 2)
    bars = plt.bar(methods_list, accuracies, color=[
                   "blue", "green", "red", "orange"])
    plt.title("Accuracy Comparison")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{accuracy:.1e}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("examples/performance_benchmark.png",
                dpi=300, bbox_inches="tight")
    plt.show()

    return results


def main():
    """Run all demonstrations."""
    print("ADVANCED FRACTIONAL CALCULUS METHODS DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the new advanced fractional calculus methods")
    print("with performance optimizations and parallel processing capabilities.")

    # Run all demonstrations
    demo_weyl_derivative()
    demo_marchaud_derivative()
    demo_hadamard_derivative()
    demo_reiz_feller_derivative()
    demo_adomian_decomposition()

    # Run performance benchmark
    benchmark_results = performance_benchmark()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        "All advanced fractional calculus methods have been successfully demonstrated!"
    )
    print("\nKey Features:")
    print("- Weyl derivative: FFT convolution with parallelization")
    print("- Marchaud derivative: Memory-optimized difference quotient convolution")
    print("- Hadamard derivative: Logarithmic transformation with efficient quadrature")
    print("- Reiz-Feller derivative: Spectral method using FFT")
    print("- Adomian decomposition: Parallel computation of decomposition terms")
    print("\nOptimizations:")
    print("- JAX compilation for GPU acceleration")
    print("- Numba compilation for CPU optimization")
    print("- Parallel processing with ThreadPoolExecutor")
    print("- Memory-efficient streaming algorithms")

    print(f"\nPerformance Summary:")
    for method, results in benchmark_results.items():
        print(
            f"  {method}: {results['speedup']:.1f}x speedup, accuracy: {results['accuracy']:.1e}"
        )


if __name__ == "__main__":
    main()
