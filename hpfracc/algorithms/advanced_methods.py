"""
Advanced Fractional Calculus Methods

This module implements advanced fractional calculus methods with optimizations:
- Weyl derivative via FFT Convolution with parallelization
- Marchaud derivative with Difference Quotient convolution and memory optimization
- Hadamard derivative
- Reiz/Reiz-Feller derivative via spectral method
- Adomian Decomposition method

All methods include parallel processing and memory optimizations.
"""

import numpy as np
from typing import Union, Optional, Tuple, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from ..core.definitions import FractionalOrder
from ..special import gamma
from .parallel_optimized_methods import ParallelConfig


class WeylDerivative:
    """
    Weyl fractional derivative via FFT Convolution with parallelization.

    The Weyl derivative is defined as:
    D^α f(x) = (1/Γ(n-α)) (d/dx)^n ∫_x^∞ (τ-x)^(n-α-1) f(τ) dτ

    This implementation uses FFT convolution for efficiency and parallel processing.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize Weyl derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.n = int(np.ceil(self.alpha.alpha))
        self.alpha_val = self.alpha.alpha
        self.parallel_config = parallel_config or ParallelConfig()

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        use_parallel: bool = True,
    ) -> Union[float, np.ndarray]:
        """Compute Weyl derivative using optimized FFT convolution."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(0, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if use_parallel and self.parallel_config.enabled:
            return self._compute_parallel(f_array, x_array, h or 1.0)
        else:
            return self._compute_serial(f_array, x_array, h or 1.0)

    def _compute_serial(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Serial computation using optimized FFT convolution."""
        N = len(f)
        n = self.n
        alpha = self.alpha_val

        # Create Weyl kernel: (τ-x)^(n-α-1) / Γ(n-α)
        kernel = np.zeros(N)
        for i in range(N):
            if x[i] > 0:
                kernel[i] = (x[i] ** (n - alpha - 1)) / gamma(n - alpha)

        # Pad arrays for circular convolution
        f_padded = np.pad(f, (0, N), mode="constant")
        kernel_padded = np.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = np.fft.fft(f_padded)
        kernel_fft = np.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = np.real(np.fft.ifft(conv_fft))

        # Apply nth derivative using finite differences
        result = np.zeros(N)
        result[:n] = 0.0

        for i in range(n, N):
            if n == 1:
                if i < N - 1:
                    result[i] = (conv[i + 1] - conv[i - 1]) / (2 * h)
                else:
                    result[i] = (conv[i] - conv[i - 1]) / h
            else:
                if i < N - 1:
                    result[i] = (conv[i + 1] - 2 * conv[i] +
                                 conv[i - 1]) / (h**2)
                else:
                    result[i] = (conv[i] - conv[i - 1]) / h

        return result * h

    def _compute_parallel(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Parallel computation using chunked processing."""
        N = len(f)

        # Handle empty arrays
        if N == 0:
            return np.array([])

        chunk_size = max(1, N // self.parallel_config.n_jobs)
        chunks = [
            (f[i: i + chunk_size], x[i: i + chunk_size], h)
            for i in range(0, N, chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=self.parallel_config.n_jobs) as executor:
            results = list(executor.map(self._process_chunk, chunks))

        return np.concatenate(results)

    def _process_chunk(
        self, chunk_data: Tuple[np.ndarray, np.ndarray, float]
    ) -> np.ndarray:
        """Process a chunk of data for parallel computation."""
        f_chunk, x_chunk, h = chunk_data
        return self._compute_serial(f_chunk, x_chunk, h)


class MarchaudDerivative:
    """
    Marchaud fractional derivative with Difference Quotient convolution
    and memory optimization.

    The Marchaud derivative is defined as:
    D^α f(x) = α/Γ(1-α) ∫_0^∞ [f(x) - f(x-τ)] / τ^(α+1) dτ

    This implementation uses difference quotient convolution with memory optimization.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize Marchaud derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.parallel_config = parallel_config or ParallelConfig()

        # Precompute constants
        self.coeff = self.alpha_val / gamma(1 - self.alpha_val)

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        use_parallel: bool = True,
        memory_optimized: bool = True,
    ) -> Union[float, np.ndarray]:
        """Compute Marchaud derivative with memory optimization."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(0, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if memory_optimized:
            return self._compute_memory_optimized(
                f_array, x_array, h or 1.0, use_parallel
            )
        else:
            return self._compute_standard(
                f_array, x_array, h or 1.0, use_parallel)

    def _compute_memory_optimized(
        self, f: np.ndarray, x: np.ndarray, h: float, use_parallel: bool
    ) -> np.ndarray:
        """Memory-optimized computation using streaming approach."""
        N = len(f)

        # Handle empty arrays
        if N == 0:
            return np.array([])

        result = np.zeros(N)

        # Use smaller chunks to reduce memory usage
        chunk_size = max(1, min(1000, N // 4))

        if use_parallel and self.parallel_config.enabled:
            chunks = [(f, x, h, i, min(i + chunk_size, N))
                      for i in range(0, N, chunk_size)]

            with ThreadPoolExecutor(
                max_workers=self.parallel_config.n_jobs
            ) as executor:
                chunk_results = list(executor.map(
                    self._process_marchaud_chunk, chunks))

            for i, chunk_result in enumerate(chunk_results):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, N)
                result[start_idx:end_idx] = chunk_result
        else:
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                result[i:end_idx] = self._compute_marchaud_segment(
                    f, x, h, i, end_idx)

        return result

    def _process_marchaud_chunk(
        self, chunk_data: Tuple[np.ndarray, np.ndarray, float, int, int]
    ) -> np.ndarray:
        """Process a chunk for Marchaud derivative."""
        f, x, h, start_idx, end_idx = chunk_data
        return self._compute_marchaud_segment(f, x, h, start_idx, end_idx)

    def _compute_marchaud_segment(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float,
            start_idx: int,
            end_idx: int) -> np.ndarray:
        """Compute Marchaud derivative for a segment."""
        result = np.zeros(end_idx - start_idx)

        for i in range(start_idx, end_idx):
            if i == 0:
                result[i - start_idx] = 0.0
                continue

            # Compute difference quotient integral
            integral = 0.0
            max_tau = min(i, 1000)  # Limit integration range for efficiency

            for j in range(1, max_tau + 1):
                tau = j * h
                if i - j >= 0:
                    diff = f[i] - f[i - j]
                    integral += diff / (tau ** (self.alpha_val + 1))

            result[i - start_idx] = self.coeff * integral * h

        return result

    def _compute_standard(
        self, f: np.ndarray, x: np.ndarray, h: float, use_parallel: bool
    ) -> np.ndarray:
        """Standard computation without memory optimization."""
        N = len(f)
        result = np.zeros(N)

        for i in range(N):
            if i == 0:
                result[i] = 0.0
                continue

            integral = 0.0
            for j in range(1, i + 1):
                tau = j * h
                diff = f[i] - f[i - j]
                integral += diff / (tau ** (self.alpha_val + 1))

            result[i] = self.coeff * integral * h

        return result


class HadamardDerivative:
    """
    Hadamard fractional derivative.

    The Hadamard derivative is defined as:
    D^α f(x) = (1/Γ(n-α)) (x d/dx)^n ∫_1^x (log(x/t))^(n-α-1) f(t) dt/t

    This implementation uses logarithmic transformation and efficient quadrature.
    """

    def __init__(self, alpha: Union[float, FractionalOrder]):
        """Initialize Hadamard derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.n = int(np.ceil(self.alpha.alpha))
        self.alpha_val = self.alpha.alpha

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
    ) -> Union[float, np.ndarray]:
        """Compute Hadamard derivative using logarithmic transformation."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                # Start from 1 for Hadamard
                x_array = np.arange(1, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(1, len(f) + 1) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        return self._compute_hadamard(f_array, x_array, h or 1.0)

    def _compute_hadamard(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Compute Hadamard derivative using logarithmic transformation."""
        N = len(f)
        result = np.zeros(N)

        for i in range(N):
            if i < self.n:
                result[i] = 0.0
                continue

            # Transform to logarithmic coordinates
            log_x = np.log(x[i])

            # Compute integral part
            integral = 0.0
            for j in range(i):
                log_t = np.log(x[j])
                log_kernel = (log_x - log_t) ** (self.n - self.alpha_val - 1)
                integral += f[j] * log_kernel / x[j]

            # Apply differential operator (x d/dx)^n
            if self.n == 1:
                result[i] = x[i] * integral / gamma(self.n - self.alpha_val)
            else:
                # Higher derivatives using recursive application
                temp = integral / gamma(self.n - self.alpha_val)
                for k in range(self.n):
                    temp = x[i] * np.gradient(temp, x[i])
                result[i] = temp

        return result * h


class ReizFellerDerivative:
    """
    Reiz-Feller fractional derivative via spectral method.

    The Reiz-Feller derivative is defined as:
    D^α f(x) = (1/2π) ∫_{-∞}^∞ |ξ|^α F[f](ξ) e^(iξx) dξ

    This implementation uses FFT for spectral computation.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize Reiz-Feller derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.parallel_config = parallel_config or ParallelConfig()

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        use_parallel: bool = True,
    ) -> Union[float, np.ndarray]:
        """Compute Reiz-Feller derivative using spectral method."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(-x_max, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        return self._compute_spectral(f_array, x_array, h or 1.0, use_parallel)

    def _compute_spectral(
        self, f: np.ndarray, x: np.ndarray, h: float, use_parallel: bool
    ) -> np.ndarray:
        """Compute using spectral method with FFT."""
        N = len(f)

        # Ensure N is even for FFT
        if N % 2 == 1:
            N += 1
            f = np.pad(f, (0, 1), mode="edge")
            x = np.pad(x, (0, 1), mode="edge")

        # Compute FFT
        f_fft = np.fft.fft(f)

        # Create frequency array
        freq = np.fft.fftfreq(N, h)

        # Apply spectral filter |ξ|^α
        spectral_filter = np.abs(freq) ** self.alpha_val
        spectral_filter[0] = 0  # Handle zero frequency

        # Apply filter in frequency domain
        filtered_fft = f_fft * spectral_filter

        # Inverse FFT
        result = np.real(np.fft.ifft(filtered_fft))

        return result


class AdomianDecomposition:
    """
    Adomian Decomposition Method for solving fractional differential equations.

    This method decomposes the solution into a series and computes each term
    using the Adomian polynomials.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        parallel_config: Optional[ParallelConfig] = None,
    ):
        """Initialize Adomian Decomposition solver."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.parallel_config = parallel_config or ParallelConfig()

    def solve(
        self,
        equation: Callable,
        initial_conditions: Dict,
        t_span: Tuple[float, float],
        n_steps: int = 1000,
        n_terms: int = 10,
        use_parallel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve fractional differential equation using Adomian decomposition.

        Args:
            equation: Function representing the FDE
            initial_conditions: Dictionary of initial conditions
            t_span: Time span (t0, tf)
            n_steps: Number of time steps
            n_terms: Number of terms in the decomposition
            use_parallel: Whether to use parallel processing

        Returns:
            Tuple of (time_points, solution)
        """
        t0, tf = t_span
        t = np.linspace(t0, tf, n_steps)
        h = (tf - t0) / (n_steps - 1)

        # Initialize solution series
        solution = np.zeros(n_steps)

        # Add initial condition
        if 0 in initial_conditions:
            solution += initial_conditions[0]

        # Compute decomposition terms
        if use_parallel and self.parallel_config.enabled:
            terms = self._compute_terms_parallel(equation, t, h, n_terms)
        else:
            terms = self._compute_terms_serial(equation, t, h, n_terms)

        # Sum all terms
        for term in terms:
            solution += term

        return t, solution

    def _compute_terms_serial(
        self, equation: Callable, t: np.ndarray, h: float, n_terms: int
    ) -> List[np.ndarray]:
        """Compute decomposition terms serially."""
        terms = []

        for n in range(1, n_terms + 1):
            # Compute Adomian polynomial using the same time array
            adomian = self._compute_adomian_polynomial(equation, terms, n, t)

            # Compute integral term
            integral_term = self._compute_integral_term(adomian, t, h)

            terms.append(integral_term)

        return terms

    def _compute_terms_parallel(
        self, equation: Callable, t: np.ndarray, h: float, n_terms: int
    ) -> List[np.ndarray]:
        """Compute decomposition terms in parallel."""
        term_indices = list(range(1, n_terms + 1))

        with ThreadPoolExecutor(max_workers=self.parallel_config.n_jobs) as executor:
            futures = []
            for n in term_indices:
                future = executor.submit(
                    self._compute_single_term, equation, t, h, n)
                futures.append(future)

            terms = [future.result() for future in futures]

        return terms

    def _compute_adomian_polynomial(self,
                                    equation: Callable,
                                    previous_terms: List[np.ndarray],
                                    n: int,
                                    t: np.ndarray) -> np.ndarray:
        """Compute the nth Adomian polynomial."""
        N = len(t)
        adomian = np.zeros(N)

        for i in range(N):
            # Simplified polynomial computation using the provided time array
            adomian[i] = equation(t[i], 0) * (t[i] ** n) / gamma(n + 1)

        return adomian

    def _compute_integral_term(
        self, adomian: np.ndarray, t: np.ndarray, h: float
    ) -> np.ndarray:
        """Compute integral term using fractional integration."""
        N = len(adomian)
        result = np.zeros(N)

        for i in range(N):
            integral = 0.0
            for j in range(i + 1):
                # Avoid division by zero and negative powers
                if t[i] > t[j] and self.alpha_val > 0:
                    kernel = ((t[i] - t[j]) ** (self.alpha_val - 1)
                              ) / gamma(self.alpha_val)
                    integral += adomian[j] * kernel

            result[i] = integral * h

        return result

    def _compute_single_term(
        self, equation: Callable, t: np.ndarray, h: float, n: int
    ) -> np.ndarray:
        """Compute a single decomposition term."""
        # This is a simplified implementation
        # In practice, you would need the previous terms to compute the Adomian
        # polynomial
        adomian = np.zeros_like(t)

        # For demonstration, use a simple approximation
        for i in range(len(t)):
            adomian[i] = equation(t[i], 0) * (t[i] ** n) / gamma(n + 1)

        return self._compute_integral_term(adomian, t, h)


# Convenience functions for easy access
def weyl_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: float,
    h: Optional[float] = None,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Convenience function for Weyl derivative."""
    calculator = WeylDerivative(alpha, **kwargs)
    return calculator.compute(f, x, h)


def marchaud_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: float,
    h: Optional[float] = None,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Convenience function for Marchaud derivative."""
    calculator = MarchaudDerivative(alpha, **kwargs)
    return calculator.compute(f, x, h)


def hadamard_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: float,
    h: Optional[float] = None,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Convenience function for Hadamard derivative."""
    calculator = HadamardDerivative(alpha, **kwargs)
    return calculator.compute(f, x, h)


def reiz_feller_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: float,
    h: Optional[float] = None,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Convenience function for Reiz-Feller derivative."""
    calculator = ReizFellerDerivative(alpha, **kwargs)
    return calculator.compute(f, x, h)
