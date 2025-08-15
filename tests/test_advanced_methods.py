"""
Tests for Advanced Fractional Calculus Methods

This module tests the new advanced fractional calculus methods:
- Weyl derivative via FFT Convolution with parallelization
- Marchaud derivative with Difference Quotient convolution and memory optimization
- Hadamard derivative
- Reiz-Feller derivative via spectral method
- Adomian Decomposition method
"""

import pytest
import numpy as np
import time
from typing import Callable

from src.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    AdomianDecomposition,
    weyl_derivative,
    marchaud_derivative,
    hadamard_derivative,
    reiz_feller_derivative,
)

from src.algorithms.advanced_optimized_methods import (
    OptimizedWeylDerivative,
    OptimizedMarchaudDerivative,
    OptimizedHadamardDerivative,
    OptimizedReizFellerDerivative,
    OptimizedAdomianDecomposition,
    optimized_weyl_derivative,
    optimized_marchaud_derivative,
    optimized_hadamard_derivative,
    optimized_reiz_feller_derivative,
    optimized_adomian_decomposition,
)

from src.algorithms.parallel_optimized_methods import ParallelConfig


class TestWeylDerivative:
    """Test Weyl derivative implementation."""

    def test_weyl_derivative_basic(self):
        """Test basic Weyl derivative computation."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        # Test function: f(x) = x^2
        def f(x):
            return x**2

        # Standard implementation
        weyl_calc = WeylDerivative(alpha)
        result = weyl_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_weyl_derivative_parallel(self):
        """Test Weyl derivative with parallel processing."""
        alpha = 0.5
        x = np.linspace(0, 10, 1000)

        def f(x):
            return np.sin(x)

        # Test with parallel processing
        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        weyl_calc = WeylDerivative(alpha, parallel_config)
        result_parallel = weyl_calc.compute(f, x, h=0.01, use_parallel=True)

        # Test without parallel processing
        result_serial = weyl_calc.compute(f, x, h=0.01, use_parallel=False)

        # Results should be similar (allowing for numerical differences)
        np.testing.assert_allclose(result_parallel, result_serial, rtol=1e-10)

    def test_weyl_derivative_convenience(self):
        """Test convenience function for Weyl derivative."""
        alpha = 0.5
        x = np.linspace(0, 5, 50)
        f = np.sin(x)

        result = weyl_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestMarchaudDerivative:
    """Test Marchaud derivative implementation."""

    def test_marchaud_derivative_basic(self):
        """Test basic Marchaud derivative computation."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return x**2

        marchaud_calc = MarchaudDerivative(alpha)
        result = marchaud_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_marchaud_derivative_memory_optimized(self):
        """Test Marchaud derivative with memory optimization."""
        alpha = 0.5
        x = np.linspace(0, 10, 1000)

        def f(x):
            return np.exp(-x)

        marchaud_calc = MarchaudDerivative(alpha)

        # Test with memory optimization
        result_optimized = marchaud_calc.compute(f, x, h=0.01, memory_optimized=True)

        # Test without memory optimization
        result_standard = marchaud_calc.compute(f, x, h=0.01, memory_optimized=False)

        # Results should be similar
        np.testing.assert_allclose(result_optimized, result_standard, rtol=1e-10)

    def test_marchaud_derivative_parallel(self):
        """Test Marchaud derivative with parallel processing."""
        alpha = 0.5
        x = np.linspace(0, 10, 1000)

        def f(x):
            return np.cos(x)

        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        marchaud_calc = MarchaudDerivative(alpha, parallel_config)

        result_parallel = marchaud_calc.compute(f, x, h=0.01, use_parallel=True)
        result_serial = marchaud_calc.compute(f, x, h=0.01, use_parallel=False)

        np.testing.assert_allclose(result_parallel, result_serial, rtol=1e-10)

    def test_marchaud_derivative_convenience(self):
        """Test convenience function for Marchaud derivative."""
        alpha = 0.5
        x = np.linspace(0, 5, 50)
        f = np.exp(-x)

        result = marchaud_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestHadamardDerivative:
    """Test Hadamard derivative implementation."""

    def test_hadamard_derivative_basic(self):
        """Test basic Hadamard derivative computation."""
        alpha = 0.5
        x = np.linspace(1, 10, 100)  # Start from 1 for Hadamard

        def f(x):
            return np.log(x)

        hadamard_calc = HadamardDerivative(alpha)
        result = hadamard_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_hadamard_derivative_convenience(self):
        """Test convenience function for Hadamard derivative."""
        alpha = 0.5
        x = np.linspace(1, 5, 50)
        f = x**2

        result = hadamard_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestReizFellerDerivative:
    """Test Reiz-Feller derivative implementation."""

    def test_reiz_feller_derivative_basic(self):
        """Test basic Reiz-Feller derivative computation."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-(x**2))

        reiz_calc = ReizFellerDerivative(alpha)
        result = reiz_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_reiz_feller_derivative_parallel(self):
        """Test Reiz-Feller derivative with parallel processing."""
        alpha = 0.5
        x = np.linspace(-5, 5, 1000)

        def f(x):
            return np.sin(x)

        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        reiz_calc = ReizFellerDerivative(alpha, parallel_config)

        result_parallel = reiz_calc.compute(f, x, h=0.01, use_parallel=True)
        result_serial = reiz_calc.compute(f, x, h=0.01, use_parallel=False)

        np.testing.assert_allclose(result_parallel, result_serial, rtol=1e-10)

    def test_reiz_feller_derivative_convenience(self):
        """Test convenience function for Reiz-Feller derivative."""
        alpha = 0.5
        x = np.linspace(-3, 3, 50)
        f = np.cos(x)

        result = reiz_feller_derivative(f, x, alpha, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestAdomianDecomposition:
    """Test Adomian Decomposition method."""

    def test_adomian_decomposition_basic(self):
        """Test basic Adomian decomposition solution."""
        alpha = 0.5

        # Simple fractional differential equation: D^α y(t) = t
        def equation(t, y):
            return t

        initial_conditions = {0: 0.0}
        t_span = (0, 1)

        adomian_solver = AdomianDecomposition(alpha)
        t, solution = adomian_solver.solve(
            equation, initial_conditions, t_span, n_steps=100, n_terms=5
        )

        assert len(t) == 100
        assert len(solution) == 100
        assert not np.any(np.isnan(solution))
        assert not np.any(np.isinf(solution))

    def test_adomian_decomposition_parallel(self):
        """Test Adomian decomposition with parallel processing."""
        alpha = 0.5

        def equation(t, y):
            return np.sin(t)

        initial_conditions = {0: 1.0}
        t_span = (0, 2)

        parallel_config = ParallelConfig(n_jobs=4, enabled=True)
        adomian_solver = AdomianDecomposition(alpha, parallel_config)

        t, solution_parallel = adomian_solver.solve(
            equation,
            initial_conditions,
            t_span,
            n_steps=200,
            n_terms=10,
            use_parallel=True,
        )
        t, solution_serial = adomian_solver.solve(
            equation,
            initial_conditions,
            t_span,
            n_steps=200,
            n_terms=10,
            use_parallel=False,
        )

        np.testing.assert_allclose(solution_parallel, solution_serial, rtol=1e-10)


class TestOptimizedMethods:
    """Test optimized versions of advanced methods."""

    def test_optimized_weyl_derivative(self):
        """Test optimized Weyl derivative."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return x**2

        # Test JAX version
        opt_weyl = OptimizedWeylDerivative(alpha)
        result_jax = opt_weyl.compute(f, x, h=0.1, use_jax=True)

        # Test Numba version
        result_numba = opt_weyl.compute(f, x, h=0.1, use_jax=False)

        assert len(result_jax) == len(x)
        assert len(result_numba) == len(x)
        assert not np.any(np.isnan(result_jax))
        assert not np.any(np.isnan(result_numba))

    def test_optimized_marchaud_derivative(self):
        """Test optimized Marchaud derivative."""
        alpha = 0.5
        x = np.linspace(0, 10, 100)

        def f(x):
            return np.sin(x)

        opt_marchaud = OptimizedMarchaudDerivative(alpha)

        result_optimized = opt_marchaud.compute(f, x, h=0.1, memory_optimized=True)
        result_standard = opt_marchaud.compute(f, x, h=0.1, memory_optimized=False)

        assert len(result_optimized) == len(x)
        assert len(result_standard) == len(x)
        assert not np.any(np.isnan(result_optimized))
        assert not np.any(np.isnan(result_standard))

    def test_optimized_hadamard_derivative(self):
        """Test optimized Hadamard derivative."""
        alpha = 0.5
        x = np.linspace(1, 10, 100)

        def f(x):
            return np.log(x)

        opt_hadamard = OptimizedHadamardDerivative(alpha)
        result = opt_hadamard.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_optimized_reiz_feller_derivative(self):
        """Test optimized Reiz-Feller derivative."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-(x**2))

        opt_reiz = OptimizedReizFellerDerivative(alpha)
        result = opt_reiz.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_optimized_adomian_decomposition(self):
        """Test optimized Adomian decomposition."""
        alpha = 0.5

        def equation(t, y):
            return t

        initial_conditions = {0: 0.0}
        t_span = (0, 1)

        opt_adomian = OptimizedAdomianDecomposition(alpha)
        t, solution = opt_adomian.solve(
            equation, initial_conditions, t_span, n_steps=100, n_terms=5
        )

        assert len(t) == 100
        assert len(solution) == 100
        assert not np.any(np.isnan(solution))

    def test_optimized_convenience_functions(self):
        """Test optimized convenience functions."""
        alpha = 0.5
        x = np.linspace(0, 5, 50)
        f = np.sin(x)

        # Test all optimized convenience functions
        result_weyl = optimized_weyl_derivative(f, x, alpha, h=0.1)
        result_marchaud = optimized_marchaud_derivative(f, x, alpha, h=0.1)
        result_hadamard = optimized_hadamard_derivative(f, x, alpha, h=0.1)
        result_reiz = optimized_reiz_feller_derivative(f, x, alpha, h=0.1)

        assert len(result_weyl) == len(x)
        assert len(result_marchaud) == len(x)
        assert len(result_hadamard) == len(x)
        assert len(result_reiz) == len(x)

        # Test Adomian decomposition convenience function
        def equation(t, y):
            return t

        initial_conditions = {0: 0.0}
        t_span = (0, 1)

        t, solution = optimized_adomian_solve(
            equation, initial_conditions, t_span, alpha, n_steps=50, n_terms=3
        )

        assert len(t) == 50
        assert len(solution) == 50


class TestPerformanceComparison:
    """Test performance comparison between standard and optimized methods."""

    def test_weyl_performance_comparison(self):
        """Compare performance of standard vs optimized Weyl derivative."""
        alpha = 0.5
        x = np.linspace(0, 10, 1000)

        def f(x):
            return np.sin(x)

        # Standard implementation
        start_time = time.time()
        weyl_calc = WeylDerivative(alpha)
        result_standard = weyl_calc.compute(f, x, h=0.01, use_parallel=False)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_weyl = OptimizedWeylDerivative(alpha)
        result_optimized = opt_weyl.compute(f, x, h=0.01, use_jax=True)
        optimized_time = time.time() - start_time

        # Results should be similar
        np.testing.assert_allclose(result_standard, result_optimized, rtol=1e-5)

        # Optimized should be faster (or at least not significantly slower)
        assert optimized_time <= standard_time * 2  # Allow some tolerance

    def test_marchaud_performance_comparison(self):
        """Compare performance of standard vs optimized Marchaud derivative."""
        alpha = 0.5
        x = np.linspace(0, 10, 1000)

        def f(x):
            return np.exp(-x)

        # Standard implementation
        start_time = time.time()
        marchaud_calc = MarchaudDerivative(alpha)
        result_standard = marchaud_calc.compute(f, x, h=0.01, memory_optimized=False)
        standard_time = time.time() - start_time

        # Optimized implementation
        start_time = time.time()
        opt_marchaud = OptimizedMarchaudDerivative(alpha)
        result_optimized = opt_marchaud.compute(f, x, h=0.01, memory_optimized=True)
        optimized_time = time.time() - start_time

        # Results should be similar
        np.testing.assert_allclose(result_standard, result_optimized, rtol=1e-5)

        # Optimized should be faster
        assert optimized_time <= standard_time * 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_alpha(self):
        """Test behavior with alpha = 0."""
        alpha = 0.0
        x = np.linspace(0, 5, 50)
        f = np.sin(x)

        # Should handle alpha = 0 gracefully
        weyl_calc = WeylDerivative(alpha)
        result = weyl_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_large_alpha(self):
        """Test behavior with large alpha."""
        alpha = 2.5
        x = np.linspace(0, 5, 50)
        f = np.sin(x)

        weyl_calc = WeylDerivative(alpha)
        result = weyl_calc.compute(f, x, h=0.1)

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_empty_array(self):
        """Test behavior with empty input array."""
        alpha = 0.5
        x = np.array([])
        f = np.array([])

        weyl_calc = WeylDerivative(alpha)
        result = weyl_calc.compute(f, x, h=0.1)

        assert len(result) == 0

    def test_single_point(self):
        """Test behavior with single point."""
        alpha = 0.5
        x = np.array([1.0])
        f = np.array([1.0])

        weyl_calc = WeylDerivative(alpha)
        result = weyl_calc.compute(f, x, h=0.1)

        assert len(result) == 1
        assert not np.isnan(result[0])


if __name__ == "__main__":
    # Run performance tests
    print("Running performance tests...")

    # Test Weyl derivative performance
    alpha = 0.5
    x = np.linspace(0, 10, 2000)

    def f(x):
        return np.sin(x)

    # Standard implementation
    start_time = time.time()
    weyl_calc = WeylDerivative(alpha)
    result_standard = weyl_calc.compute(f, x, h=0.005, use_parallel=False)
    standard_time = time.time() - start_time

    # Optimized implementation
    start_time = time.time()
    opt_weyl = OptimizedWeylDerivative(alpha)
    result_optimized = opt_weyl.compute(f, x, h=0.005, use_jax=True)
    optimized_time = time.time() - start_time

    print(f"Weyl Derivative Performance:")
    print(f"  Standard: {standard_time:.4f}s")
    print(f"  Optimized: {optimized_time:.4f}s")
    print(f"  Speedup: {standard_time/optimized_time:.2f}x")

    # Test Marchaud derivative performance
    start_time = time.time()
    marchaud_calc = MarchaudDerivative(alpha)
    result_standard = marchaud_calc.compute(f, x, h=0.005, memory_optimized=False)
    standard_time = time.time() - start_time

    start_time = time.time()
    opt_marchaud = OptimizedMarchaudDerivative(alpha)
    result_optimized = opt_marchaud.compute(f, x, h=0.005, memory_optimized=True)
    optimized_time = time.time() - start_time

    print(f"\nMarchaud Derivative Performance:")
    print(f"  Standard: {standard_time:.4f}s")
    print(f"  Optimized: {optimized_time:.4f}s")
    print(f"  Speedup: {standard_time/optimized_time:.2f}x")

    print("\nAll tests completed successfully!")
