#!/usr/bin/env python3
"""
Test script to verify that optimized methods are properly integrated into main algorithms.
"""

import numpy as np
import time
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative


def test_function(t):
    """Test function: f(t) = t^2"""
    return t**2


def test_integration():
    """Test that optimized methods are properly integrated."""
    print("Testing integration of optimized methods into main algorithms...")
    print("=" * 60)

    # Test data
    t = np.linspace(0, 10, 1000)
    f = test_function(t)
    alpha = 0.5

    # Test Riemann-Liouville
    print("\n1. Testing Riemann-Liouville Derivative:")
    print("-" * 40)

    # Standard FFT method
    start_time = time.time()
    rl_standard = RiemannLiouvilleDerivative(alpha, method="fft")
    result_standard = rl_standard.compute(f, t, h=0.01)
    time_standard = time.time() - start_time

    # Optimized FFT method
    start_time = time.time()
    rl_optimized = RiemannLiouvilleDerivative(alpha, method="optimized_fft")
    result_optimized = rl_optimized.compute(f, t, h=0.01)
    time_optimized = time.time() - start_time

    print(f"Standard FFT: {time_standard:.4f}s")
    print(f"Optimized FFT: {time_optimized:.4f}s")
    print(f"Speedup: {time_standard/time_optimized:.2f}x")
    print(
        f"Results match: {np.allclose(result_standard, result_optimized, rtol=1e-10)}"
    )

    # Test Caputo
    print("\n2. Testing Caputo Derivative:")
    print("-" * 40)

    # Standard L1 method
    start_time = time.time()
    caputo_standard = CaputoDerivative(alpha, method="l1")
    result_standard = caputo_standard.compute(f, t, h=0.01)
    time_standard = time.time() - start_time

    # Optimized L1 method
    start_time = time.time()
    caputo_optimized = CaputoDerivative(alpha, method="optimized_l1")
    result_optimized = caputo_optimized.compute(f, t, h=0.01)
    time_optimized = time.time() - start_time

    print(f"Standard L1: {time_standard:.4f}s")
    print(f"Optimized L1: {time_optimized:.4f}s")
    print(f"Speedup: {time_standard/time_optimized:.2f}x")
    print(
        f"Results match: {np.allclose(result_standard, result_optimized, rtol=1e-10)}"
    )

    # Test Grünwald-Letnikov
    print("\n3. Testing Grünwald-Letnikov Derivative:")
    print("-" * 40)

    # Standard direct method
    start_time = time.time()
    gl_standard = GrunwaldLetnikovDerivative(alpha, method="direct")
    result_standard = gl_standard.compute(f, t, h=0.01)
    time_standard = time.time() - start_time

    # Optimized direct method
    start_time = time.time()
    gl_optimized = GrunwaldLetnikovDerivative(alpha, method="optimized_direct")
    result_optimized = gl_optimized.compute(f, t, h=0.01)
    time_optimized = time.time() - start_time

    print(f"Standard Direct: {time_standard:.4f}s")
    print(f"Optimized Direct: {time_optimized:.4f}s")
    print(f"Speedup: {time_standard/time_optimized:.2f}x")
    print(
        f"Results match: {np.allclose(result_standard, result_optimized, rtol=1e-10)}"
    )

    print("\n" + "=" * 60)
    print("Integration test completed!")


if __name__ == "__main__":
    test_integration()
