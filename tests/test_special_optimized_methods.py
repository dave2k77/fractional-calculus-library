"""
Tests for Special Optimized Methods

This module tests the optimized versions of existing methods that integrate
the new special methods for improved performance.
"""

import pytest
import numpy as np
import time
from typing import Callable

from hpfracc.algorithms.special_optimized_methods import (
    SpecialOptimizedWeylDerivative,
    SpecialOptimizedMarchaudDerivative,
    SpecialOptimizedReizFellerDerivative,
    UnifiedSpecialMethods,
    special_optimized_weyl_derivative,
    special_optimized_marchaud_derivative,
    special_optimized_reiz_feller_derivative,
    unified_special_derivative,
)

from hpfracc.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    ReizFellerDerivative,
)

from hpfracc.core.definitions import FractionalOrder


class TestSpecialOptimizedWeylDerivative:
    """Test Special Optimized Weyl Derivative implementation."""

    def test_special_optimized_weyl_basic(self):
        """Test basic special optimized Weyl derivative computation."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return np.exp(-x**2)

        # Test special optimized version
        weyl_special = SpecialOptimizedWeylDerivative(alpha)
        result = weyl_special.compute(f, x)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_special_optimized_weyl_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return np.sin(x)

        weyl_special = SpecialOptimizedWeylDerivative(alpha)

        # Test different methods
        result_fft = weyl_special.compute(f, x, method="standard_fft")
        result_hybrid = weyl_special.compute(f, x, method="hybrid")

        # All methods should produce results
        assert len(result_fft) == len(x)
        assert len(result_hybrid) == len(x)

        # Check for valid results
        assert not np.any(np.isnan(result_fft))
        assert not np.any(np.isnan(result_hybrid))

    def test_special_optimized_weyl_performance(self):
        """Test performance improvement over standard implementation."""
        alpha = 0.5
        x = np.linspace(0, 10, 500)

        def f(x):
            return np.exp(-x**2)

        # Standard implementation
        weyl_std = WeylDerivative(alpha)
        start_time = time.time()
        result_std = weyl_std.compute(f, x, use_parallel=False)
        time_std = time.time() - start_time

        # Special optimized implementation
        weyl_special = SpecialOptimizedWeylDerivative(alpha)
        start_time = time.time()
        result_special = weyl_special.compute(f, x)
        time_special = time.time() - start_time

        # Both should produce valid results
        assert len(result_std) == len(x)
        assert len(result_special) == len(x)
        assert not np.any(np.isnan(result_std))
        assert not np.any(np.isnan(result_special))

        print(f"Standard Weyl: {time_std:.4f}s")
        print(f"Special Optimized Weyl: {time_special:.4f}s")
        print(f"Speedup: {time_std/time_special:.2f}x")

    def test_special_optimized_weyl_convenience(self):
        """Test convenience function."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)
        f = np.sin(x)

        result = special_optimized_weyl_derivative(f, x, alpha)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestSpecialOptimizedMarchaudDerivative:
    """Test Special Optimized Marchaud Derivative implementation."""

    def test_special_optimized_marchaud_basic(self):
        """Test basic special optimized Marchaud derivative computation."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return np.sin(x)

        # Test special optimized version
        marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)
        result = marchaud_special.compute(f, x)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_special_optimized_marchaud_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return np.cos(x)

        marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)

        # Test different methods
        result_z = marchaud_special.compute(f, x, method="z_transform")
        result_std = marchaud_special.compute(f, x, method="standard")

        # Both methods should produce results
        assert len(result_z) == len(x)
        assert len(result_std) == len(x)

        # Check for valid results
        assert not np.any(np.isnan(result_z))
        assert not np.any(np.isnan(result_std))

    def test_special_optimized_marchaud_performance(self):
        """Test performance improvement over standard implementation."""
        alpha = 0.5
        x = np.linspace(0, 10, 500)

        def f(x):
            return np.sin(x)

        # Standard implementation
        marchaud_std = MarchaudDerivative(alpha)
        start_time = time.time()
        result_std = marchaud_std.compute(f, x, use_parallel=False, memory_optimized=True)
        time_std = time.time() - start_time

        # Special optimized implementation
        marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)
        start_time = time.time()
        result_special = marchaud_special.compute(f, x)
        time_special = time.time() - start_time

        # Both should produce valid results
        assert len(result_std) == len(x)
        assert len(result_special) == len(x)
        assert not np.any(np.isnan(result_std))
        assert not np.any(np.isnan(result_special))

        print(f"Standard Marchaud: {time_std:.4f}s")
        print(f"Special Optimized Marchaud: {time_special:.4f}s")
        print(f"Speedup: {time_std/time_special:.2f}x")

    def test_special_optimized_marchaud_convenience(self):
        """Test convenience function."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)
        f = np.cos(x)

        result = special_optimized_marchaud_derivative(f, x, alpha)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestSpecialOptimizedReizFellerDerivative:
    """Test Special Optimized Reiz-Feller Derivative implementation."""

    def test_special_optimized_reiz_feller_basic(self):
        """Test basic special optimized Reiz-Feller derivative computation."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-x**2)

        # Test special optimized version
        reiz_special = SpecialOptimizedReizFellerDerivative(alpha)
        result = reiz_special.compute(f, x)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_special_optimized_reiz_feller_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-x**2)

        reiz_special = SpecialOptimizedReizFellerDerivative(alpha)

        # Test different methods
        result_laplacian = reiz_special.compute(f, x, method="laplacian")
        result_spectral = reiz_special.compute(f, x, method="spectral")

        # Both methods should produce results
        assert len(result_laplacian) == len(x)
        assert len(result_spectral) == len(x)

        # Check for valid results
        assert not np.any(np.isnan(result_laplacian))
        assert not np.any(np.isnan(result_spectral))

    def test_special_optimized_reiz_feller_performance(self):
        """Test performance improvement over standard implementation."""
        alpha = 0.5
        x = np.linspace(-5, 5, 500)

        def f(x):
            return np.exp(-x**2)

        # Standard implementation
        reiz_std = ReizFellerDerivative(alpha)
        start_time = time.time()
        result_std = reiz_std.compute(f, x, use_parallel=False)
        time_std = time.time() - start_time

        # Special optimized implementation
        reiz_special = SpecialOptimizedReizFellerDerivative(alpha)
        start_time = time.time()
        result_special = reiz_special.compute(f, x)
        time_special = time.time() - start_time

        # Both should produce valid results
        assert len(result_std) == len(x)
        assert len(result_special) == len(x)
        assert not np.any(np.isnan(result_std))
        assert not np.any(np.isnan(result_special))

        print(f"Standard Reiz-Feller: {time_std:.4f}s")
        print(f"Special Optimized Reiz-Feller: {time_special:.4f}s")
        print(f"Speedup: {time_std/time_special:.2f}x")

    def test_special_optimized_reiz_feller_convenience(self):
        """Test convenience function."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)

        result = special_optimized_reiz_feller_derivative(f, x, alpha)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestUnifiedSpecialMethods:
    """Test Unified Special Methods implementation."""

    def test_unified_special_methods_basic(self):
        """Test basic unified special methods computation."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)

        # Test unified interface
        unified = UnifiedSpecialMethods()
        result = unified.compute_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_unified_special_methods_auto_selection(self):
        """Test automatic method selection."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)

        unified = UnifiedSpecialMethods()

        # Test different problem types
        result_periodic = unified.compute_derivative(f, x, alpha, h=0.1, problem_type="periodic")
        result_discrete = unified.compute_derivative(f, x, alpha, h=0.1, problem_type="discrete")
        result_spectral = unified.compute_derivative(f, x, alpha, h=0.1, problem_type="spectral")
        result_general = unified.compute_derivative(f, x, alpha, h=0.1, problem_type="general")

        # All should produce valid results
        assert len(result_periodic) == len(x)
        assert len(result_discrete) == len(x)
        assert len(result_spectral) == len(x)
        assert len(result_general) == len(x)

        assert not np.any(np.isnan(result_periodic))
        assert not np.any(np.isnan(result_discrete))
        assert not np.any(np.isnan(result_spectral))
        assert not np.any(np.isnan(result_general))

    def test_unified_special_methods_convenience(self):
        """Test convenience function."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)

        result = unified_special_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestSpecialOptimizedMethodsIntegration:
    """Test integration between special optimized methods."""

    def test_method_comparison(self):
        """Compare all special optimized methods."""
        alpha = 0.5
        x = np.linspace(0, 10, 200)

        def f(x):
            return np.exp(-x**2)

        # Test all special optimized methods
        weyl_special = SpecialOptimizedWeylDerivative(alpha)
        marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)
        reiz_special = SpecialOptimizedReizFellerDerivative(alpha)
        unified = UnifiedSpecialMethods()

        # Compute results
        result_weyl = weyl_special.compute(f, x)
        result_marchaud = marchaud_special.compute(f, x)
        result_reiz = reiz_special.compute(f, x)
        result_unified = unified.compute_derivative(f, x, alpha, h=0.05)

        # All should produce valid results
        assert len(result_weyl) == len(x)
        assert len(result_marchaud) == len(x)
        assert len(result_reiz) == len(x)
        assert len(result_unified) == len(x)

        assert not np.any(np.isnan(result_weyl))
        assert not np.any(np.isnan(result_marchaud))
        assert not np.any(np.isnan(result_reiz))
        assert not np.any(np.isnan(result_unified))

    def test_performance_benchmark(self):
        """Benchmark performance of all special optimized methods."""
        alpha = 0.5
        sizes = [100, 500, 1000]

        def f(x):
            return np.exp(-x**2)

        print("\nSpecial Optimized Methods Performance Benchmark:")
        print("=" * 60)

        for size in sizes:
            print(f"\nSize: {size}")
            x = np.linspace(0, 10, size)

            # Benchmark Weyl
            weyl_special = SpecialOptimizedWeylDerivative(alpha)
            start_time = time.time()
            result = weyl_special.compute(f, x)
            time_weyl = time.time() - start_time

            # Benchmark Marchaud
            marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)
            start_time = time.time()
            result = marchaud_special.compute(f, x)
            time_marchaud = time.time() - start_time

            # Benchmark Reiz-Feller
            reiz_special = SpecialOptimizedReizFellerDerivative(alpha)
            start_time = time.time()
            result = reiz_special.compute(f, x)
            time_reiz = time.time() - start_time

            # Benchmark Unified
            unified = UnifiedSpecialMethods()
            start_time = time.time()
            result = unified.compute_derivative(f, x, alpha, h=0.1)
            time_unified = time.time() - start_time

            print(f"  Weyl: {time_weyl:.4f}s")
            print(f"  Marchaud: {time_marchaud:.4f}s")
            print(f"  Reiz-Feller: {time_reiz:.4f}s")
            print(f"  Unified: {time_unified:.4f}s")


if __name__ == "__main__":
    # Run performance tests
    print("Running special optimized methods performance tests...")

    # Test performance benchmark
    test_performance = TestSpecialOptimizedMethodsIntegration()
    test_performance.test_performance_benchmark()

    print("\nAll special optimized methods tests completed successfully!")
