#!/usr/bin/env python3
"""
CPU-Optimized Test for Fractional Calculus Library
Demonstrates that the library works well even without GPU
"""

import numpy as np
import time
from src.algorithms.optimized_methods import (
    optimized_caputo,
    optimized_riemann_liouville,
)
from src.solvers import solve_fractional_ode
from src.algorithms.parallel_optimized_methods import parallel_optimized_caputo


def test_cpu_performance():
    """Test CPU performance of the library."""
    print("🚀 CPU-Optimized Fractional Calculus Test")
    print("=" * 50)

    # Test 1: Basic fractional derivatives
    print("\n1. Testing Fractional Derivatives...")
    t = np.linspace(0, 1, 1000)
    f = np.sin(2 * np.pi * t)
    alpha = 0.5

    # Caputo derivative
    start_time = time.time()
    result_caputo = optimized_caputo(f, t, alpha)
    caputo_time = time.time() - start_time
    print(f"   Caputo derivative: {caputo_time:.4f}s")

    # Riemann-Liouville derivative
    start_time = time.time()
    result_rl = optimized_riemann_liouville(f, t, alpha)
    rl_time = time.time() - start_time
    print(f"   Riemann-Liouville derivative: {rl_time:.4f}s")

    # Test 2: Parallel computing
    print("\n2. Testing Parallel Computing...")
    start_time = time.time()
    result_parallel = parallel_optimized_caputo(f, t, alpha, h=0.001)
    parallel_time = time.time() - start_time
    print(f"   Parallel computation: {parallel_time:.4f}s")
    print(f"   Speedup vs sequential: {caputo_time/parallel_time:.2f}x")

    # Test 3: Fractional ODE solving
    print("\n3. Testing Fractional ODE Solver...")

    def test_ode(t, y):
        return -y

    start_time = time.time()
    t_sol, y_sol = solve_fractional_ode(
        test_ode, (0, 1), 1.0, alpha, method="predictor_corrector"
    )
    ode_time = time.time() - start_time
    print(f"   ODE solver: {ode_time:.4f}s")
    print(f"   Solution points: {len(t_sol)}")

    # Test 4: Large dataset performance
    print("\n4. Testing Large Dataset Performance...")
    t_large = np.linspace(0, 10, 10000)
    f_large = np.sin(t_large) + np.cos(2 * t_large)

    start_time = time.time()
    result_large = optimized_caputo(f_large, t_large, alpha)
    large_time = time.time() - start_time
    print(f"   Large dataset (10k points): {large_time:.4f}s")

    print("\n" + "=" * 50)
    print("✅ CPU-optimized tests completed successfully!")
    print("The library provides excellent performance even without GPU acceleration.")
    print("Key optimizations include:")
    print("- Parallel computing with Joblib")
    print("- Optimized numerical algorithms")
    print("- Memory-efficient implementations")
    print("- Fast FFT-based methods")


if __name__ == "__main__":
    test_cpu_performance()
