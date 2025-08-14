"""
Advanced NUMBA Kernels for Fractional Calculus

This module provides advanced NUMBA optimizations including CPU parallelization,
memory optimization, and specialized kernels for fractional calculus operations.
"""

import numpy as np
from numba import jit, prange, cuda, njit, vectorize, guvectorize
from numba.core.types import float64, float32, int64, int32
from numba.cuda import jit as cuda_jit
from typing import Union, Optional, Tuple, Callable, Dict, Any
import time
import psutil
import threading

from src.core.definitions import FractionalOrder


class NumbaOptimizer:
    """
    Advanced NUMBA optimizer for fractional calculus operations.

    Provides CPU parallelization, memory optimization, and specialized
    kernels for high-performance fractional calculus computations.
    """

    def __init__(
        self, parallel: bool = True, fastmath: bool = True, cache: bool = True
    ):
        """
        Initialize NUMBA optimizer.

        Args:
            parallel: Enable parallel execution
            fastmath: Enable fast math optimizations
            cache: Enable function caching
        """
        self.parallel = parallel
        self.fastmath = fastmath
        self.cache = cache

    def optimize_kernel(
        self, kernel_func: Callable, signature: Optional[str] = None, **kwargs
    ) -> Callable:
        """
        Optimize a kernel function with NUMBA.

        Args:
            kernel_func: Kernel function to optimize
            signature: Function signature for compilation
            **kwargs: Optimization parameters

        Returns:
            Optimized kernel function
        """
        # Apply JIT compilation with optimizations
        optimized_kernel = jit(
            kernel_func,
            nopython=True,
            parallel=self.parallel,
            fastmath=self.fastmath,
            cache=self.cache,
            **kwargs,
        )

        return optimized_kernel

    def create_parallel_kernel(self, kernel_func: Callable, **kwargs) -> Callable:
        """
        Create a parallel kernel for CPU execution.

        Args:
            kernel_func: Kernel function to parallelize
            **kwargs: Parallelization parameters

        Returns:
            Parallelized kernel function
        """
        # Force parallel execution
        parallel_kernel = jit(
            kernel_func,
            nopython=True,
            parallel=True,
            fastmath=self.fastmath,
            cache=self.cache,
            **kwargs,
        )

        return parallel_kernel


class NumbaFractionalKernels:
    """
    Specialized NUMBA kernels for fractional calculus operations.

    Provides optimized CPU kernels for various fractional derivative
    and integral computations.
    """

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def caputo_l1_kernel(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
        """
        Optimized L1 scheme kernel for Caputo derivative.

        Args:
            f: Function values array
            alpha: Fractional order (0 < α < 1)
            h: Step size

        Returns:
            Caputo derivative values
        """
        N = len(f)
        result = np.zeros(N)

        # Coefficients for L1 scheme
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - j**alpha

        # Compute derivative with parallel loop
        for n in prange(1, N):
            sum_val = 0.0
            for j in range(n + 1):
                sum_val += coeffs[j] * (f[n] - f[n - 1])
            result[n] = (h ** (-alpha) / gamma_approx(2 - alpha)) * sum_val

        return result

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def caputo_l2_kernel(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
        """
        Optimized L2 scheme kernel for Caputo derivative.

        Args:
            f: Function values array
            alpha: Fractional order (0 < α < 1)
            h: Step size

        Returns:
            Caputo derivative values
        """
        N = len(f)
        result = np.zeros(N)

        # Coefficients for L2 scheme
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - 2 * j**alpha + (j - 1) ** alpha

        # Compute derivative with parallel loop
        for n in prange(2, N):
            sum_val = 0.0
            for j in range(n + 1):
                sum_val += coeffs[j] * f[n - j]
            result[n] = (h ** (-alpha) / gamma_approx(3 - alpha)) * sum_val

        return result

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def grunwald_letnikov_kernel(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
        """
        Optimized Grünwald-Letnikov derivative kernel.

        Args:
            f: Function values array
            alpha: Fractional order
            h: Step size

        Returns:
            Grünwald-Letnikov derivative values
        """
        N = len(f)
        result = np.zeros(N)

        # Coefficients for Grünwald-Letnikov
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = coeffs[j - 1] * (1 - (alpha + 1) / j)

        # Compute derivative with parallel loop
        for n in prange(1, N):
            sum_val = 0.0
            for j in range(n + 1):
                sum_val += coeffs[j] * f[n - j]
            result[n] = (h ** (-alpha)) * sum_val

        return result

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def riemann_liouville_kernel(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
        """
        Optimized Riemann-Liouville derivative kernel.

        Args:
            f: Function values array
            alpha: Fractional order
            h: Step size

        Returns:
            Riemann-Liouville derivative values
        """
        N = len(f)
        result = np.zeros(N)
        n = int(np.ceil(alpha))

        # Compute derivative with parallel loop
        for i in prange(n, N):
            # Compute integral part
            integral = 0.0
            for j in range(i):
                tau = j * h
                weight = (i * h - tau) ** (n - alpha - 1)
                integral += weight * f[j] * h

            # Apply nth derivative using finite differences
            if n == 1:
                if i < N - 1:
                    result[i] = (integral - result[i - 1]) / (2 * h)
                else:
                    result[i] = (integral - result[i - 1]) / h
            else:
                # Simplified higher derivative
                result[i] = integral / gamma_approx(n - alpha)

        return result

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def fft_convolution_kernel(
        f: np.ndarray, kernel: np.ndarray, h: float
    ) -> np.ndarray:
        """
        Optimized FFT convolution kernel.

        Args:
            f: Function values array
            kernel: Convolution kernel
            h: Step size

        Returns:
            Convolution result
        """
        N = len(f)

        # Pad arrays for circular convolution
        f_padded = np.zeros(2 * N)
        kernel_padded = np.zeros(2 * N)

        f_padded[:N] = f
        kernel_padded[:N] = kernel

        # FFT convolution (using numpy FFT for NUMBA compatibility)
        # Note: NUMBA doesn't support np.fft directly, so we'll use a simplified approach
        # For now, we'll use a direct convolution implementation
        conv = np.zeros_like(f_padded)
        for i in range(len(f_padded)):
            for j in range(len(kernel_padded)):
                if i - j >= 0:
                    conv[i] += f_padded[j] * kernel_padded[i - j]

        return conv[:N] * h


class NumbaMemoryOptimizer:
    """
    Memory optimization utilities for NUMBA kernels.

    Provides memory-efficient implementations and memory management
    for large-scale fractional calculus computations.
    """

    @staticmethod
    @jit(nopython=True, parallel=True)
    def memory_efficient_caputo(
        f: np.ndarray, alpha: float, h: float, memory_limit: int = 1000
    ) -> np.ndarray:
        """
        Memory-efficient Caputo derivative computation.

        Args:
            f: Function values array
            alpha: Fractional order
            h: Step size
            memory_limit: Maximum memory usage in MB

        Returns:
            Caputo derivative values
        """
        N = len(f)
        result = np.zeros(N)

        # Use short memory principle for large arrays
        if N > memory_limit:
            L = min(memory_limit, N // 10)  # Memory length

            # Coefficients
            coeffs = np.zeros(L + 1)
            coeffs[0] = 1.0
            for j in range(1, L + 1):
                coeffs[j] = (j + 1) ** alpha - j**alpha

            # Compute derivative with limited memory
            for n in prange(1, N):
                j_max = min(n, L)
                sum_val = 0.0
                for j in range(j_max + 1):
                    sum_val += coeffs[j] * (f[n] - f[n - 1])
                result[n] = (h ** (-alpha) / gamma_approx(2 - alpha)) * sum_val
        else:
            # Use full memory for small arrays - inline the computation
            N = len(f)
            result = np.zeros(N)

            # Coefficients for L1 scheme
            coeffs = np.zeros(N)
            coeffs[0] = 1.0
            for j in range(1, N):
                coeffs[j] = (j + 1) ** alpha - j**alpha

            # Compute derivative
            for n in range(1, N):
                sum_val = 0.0
                for j in range(n + 1):
                    sum_val += coeffs[j] * (f[n] - f[n - 1])
                result[n] = (h ** (-alpha) / gamma_approx(2 - alpha)) * sum_val

        return result

    @staticmethod
    @jit(nopython=True, parallel=True)
    def block_processing_kernel(
        f: np.ndarray, alpha: float, h: float, block_size: int = 1000
    ) -> np.ndarray:
        """
        Block processing kernel for large arrays.

        Args:
            f: Function values array
            alpha: Fractional order
            h: Step size
            block_size: Size of processing blocks

        Returns:
            Processed result
        """
        N = len(f)
        result = np.zeros(N)

        # Process in blocks
        num_blocks = (N + block_size - 1) // block_size

        for block in prange(num_blocks):
            start_idx = block * block_size
            end_idx = min((block + 1) * block_size, N)

            # Process block
            for i in range(start_idx, end_idx):
                if i == 0:
                    result[i] = 0.0
                else:
                    # Simplified computation for block processing
                    result[i] = (f[i] - f[i - 1]) / h

        return result


class NumbaSpecializedKernels:
    """
    Specialized NUMBA kernels for specific fractional calculus applications.

    Provides optimized kernels for common use cases and special functions.
    """

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def power_law_kernel(t: np.ndarray, alpha: float, n: int) -> np.ndarray:
        """
        Optimized power-law kernel computation.

        Args:
            t: Time points
            alpha: Fractional order
            n: Integer part of alpha

        Returns:
            Power-law kernel values
        """
        N = len(t)
        kernel = np.zeros(N)

        for i in prange(N):
            if t[i] > 0:
                kernel[i] = (t[i] ** (n - alpha - 1)) / gamma_approx(n - alpha)
            else:
                kernel[i] = 0.0

        return kernel

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def binomial_coefficients_kernel(alpha: float, max_k: int) -> np.ndarray:
        """
        Optimized binomial coefficients computation.

        Args:
            alpha: Fractional order
            max_k: Maximum coefficient index

        Returns:
            Binomial coefficients array
        """
        coeffs = np.zeros(max_k + 1)
        coeffs[0] = 1.0

        for k in range(1, max_k + 1):
            coeffs[k] = coeffs[k - 1] * (1 - (alpha + 1) / k)

        return coeffs

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def mittag_leffler_kernel(
        t: np.ndarray, alpha: float, beta: float, max_terms: int = 100
    ) -> np.ndarray:
        """
        Optimized Mittag-Leffler function kernel.

        Args:
            t: Time points
            alpha: First parameter
            beta: Second parameter
            max_terms: Maximum number of series terms

        Returns:
            Mittag-Leffler function values
        """
        N = len(t)
        result = np.zeros(N)

        for i in prange(N):
            if t[i] == 0:
                result[i] = 1.0 / gamma_approx(beta)
            else:
                # Series expansion
                sum_val = 0.0
                for k in range(max_terms):
                    term = (t[i] ** k) / gamma_approx(alpha * k + beta)
                    sum_val += term
                    if abs(term) < 1e-10:
                        break
                result[i] = sum_val

        return result


class NumbaParallelManager:
    """
    Parallel execution manager for NUMBA kernels.

    Provides utilities for managing parallel execution, load balancing,
    and thread management for fractional calculus computations.
    """

    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize parallel manager.

        Args:
            num_threads: Number of threads to use (None for auto)
        """
        if num_threads is None:
            self.num_threads = psutil.cpu_count(logical=True)
        else:
            self.num_threads = num_threads

        # Set NUMBA threading configuration
        import numba

        numba.set_num_threads(self.num_threads)

    def get_optimal_chunk_size(self, array_size: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.

        Args:
            array_size: Size of array to process

        Returns:
            Optimal chunk size
        """
        # Simple heuristic for chunk size
        chunk_size = max(1, array_size // (self.num_threads * 4))
        return min(chunk_size, 1000)  # Cap at 1000

    def parallel_map(
        self, func: Callable, arrays: Tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """
        Apply function in parallel to arrays.

        Args:
            func: Function to apply
            arrays: Arrays to process

        Returns:
            Result array
        """
        # Check if we're dealing with JAX arrays (which are not compatible with Numba)
        try:
            # This is a simplified implementation
            # In practice, you would use more sophisticated parallel mapping
            result = func(*arrays)
            return result
        except Exception as e:
            # If we get a Numba error (likely due to JAX arrays), fall back to non-parallel
            if "Cannot determine Numba type" in str(e) or "pyobject" in str(e):
                # Convert to numpy arrays if possible and use simple mapping
                numpy_arrays = []
                for arr in arrays:
                    if hasattr(arr, "__array__"):
                        numpy_arrays.append(np.array(arr))
                    else:
                        numpy_arrays.append(arr)
                return func(*numpy_arrays)
            else:
                raise e


# Utility functions
@jit(nopython=True, fastmath=True)
def _gamma_numba_scalar(z: float) -> float:
    """
    Fast approximation of gamma function for NUMBA (scalar version).

    Args:
        z: Input value

    Returns:
        Approximate gamma function value
    """
    # Lanczos approximation for positive real numbers
    if z <= 0:
        return np.inf

    # Coefficients for Lanczos approximation
    g = 7.0
    p = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )

    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * _gamma_numba_scalar(1 - z))

    z -= 1
    a = p[0]
    for i in range(1, len(p)):
        a += p[i] / (z + i)

    t = z + g + 0.5
    return np.sqrt(2 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * a


@jit(nopython=True, fastmath=True)
def gamma_approx(x: float) -> float:
    """
    Fast approximation of gamma function for NUMBA.

    Args:
        x: Input value

    Returns:
        Approximate gamma function value
    """
    # Lanczos approximation for positive real numbers
    if x <= 0:
        return np.inf

    # Coefficients for Lanczos approximation
    g = 7.0
    p = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )

    if x < 0.5:
        return np.pi / (np.sin(np.pi * x) * gamma_approx(1 - x))

    x -= 1
    a = p[0]
    for i in range(1, len(p)):
        a += p[i] / (x + i)

    t = x + g + 0.5
    return np.sqrt(2 * np.pi) * (t ** (x + 0.5)) * np.exp(-t) * a


# Convenience functions
def optimize_fractional_kernel_numba(
    kernel_func: Callable, parallel: bool = True, fastmath: bool = True, **kwargs
) -> Callable:
    """
    Optimize a fractional calculus kernel with NUMBA.

    Args:
        kernel_func: Kernel function to optimize
        parallel: Enable parallel execution
        fastmath: Enable fast math optimizations
        **kwargs: Additional optimization parameters

    Returns:
        Optimized kernel function
    """
    optimizer = NumbaOptimizer(parallel, fastmath)
    return optimizer.optimize_kernel(kernel_func, **kwargs)


def compute_caputo_derivative_numba_optimized(
    f: np.ndarray, alpha: float, h: float, method: str = "l1"
) -> np.ndarray:
    """
    Compute Caputo derivative using optimized NUMBA kernels.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size
        method: Computation method

    Returns:
        Caputo derivative values
    """
    if method == "l1":
        return NumbaFractionalKernels.caputo_l1_kernel(f, alpha, h)
    elif method == "l2":
        return NumbaFractionalKernels.caputo_l2_kernel(f, alpha, h)
    elif method == "memory_efficient":
        return NumbaMemoryOptimizer.memory_efficient_caputo(f, alpha, h)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_grunwald_letnikov_numba_optimized(
    f: np.ndarray, alpha: float, h: float
) -> np.ndarray:
    """
    Compute Grünwald-Letnikov derivative using optimized NUMBA kernels.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size

    Returns:
        Grünwald-Letnikov derivative values
    """
    return NumbaFractionalKernels.grunwald_letnikov_kernel(f, alpha, h)


def compute_riemann_liouville_numba_optimized(
    f: np.ndarray, alpha: float, h: float
) -> np.ndarray:
    """
    Compute Riemann-Liouville derivative using optimized NUMBA kernels.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size

    Returns:
        Riemann-Liouville derivative values
    """
    return NumbaFractionalKernels.riemann_liouville_kernel(f, alpha, h)
