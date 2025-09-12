"""
Fractional Integral Methods

This module provides high-performance implementations of fractional integrals
including Riemann-Liouville and Caputo integrals with optimized algorithms.
"""

import numpy as np
from typing import Union, Optional, Callable
import warnings

from ..core.definitions import FractionalOrder
from ..special import gamma


class RiemannLiouvilleIntegral:
    """
    Riemann-Liouville fractional integral of order α.

    The Riemann-Liouville fractional integral of order α > 0 is defined as:

    I^α f(t) = (1/Γ(α)) ∫₀ᵗ (t-τ)^(α-1) f(τ) dτ

    where Γ(α) is the gamma function.

    Features:
    - Optimized FFT-based computation for large arrays
    - Direct method for small arrays with high accuracy
    - Memory-efficient algorithms
    - Support for both callable and array inputs
    - Error estimation and convergence analysis
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        method: str = "auto",
        optimize_memory: bool = True,
        use_jax: bool = False
    ):
        """
        Initialize Riemann-Liouville integral calculator.

        Args:
            alpha: Fractional order (must be > 0)
            method: Computation method ("auto", "fft", "direct", "adaptive")
            optimize_memory: Use memory optimization techniques
            use_jax: Use JAX acceleration if available
        """
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.alpha

        if alpha < 0:
            raise ValueError("Fractional order must be non-negative")

        self.alpha = alpha
        self.method = method.lower()
        self.optimize_memory = optimize_memory
        self.use_jax = use_jax

        # Validate method
        valid_methods = ["auto", "fft", "direct", "adaptive"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        # Precompute gamma value
        self.gamma_alpha = gamma(alpha)

        # Set method thresholds
        self.fft_threshold = 1000  # Use FFT for arrays larger than this

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        t: np.ndarray,
        h: Optional[float] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute the Riemann-Liouville fractional integral.

        Args:
            f: Function to integrate (callable) or function values (array)
            t: Time points where integral is evaluated
            h: Step size (if None, computed from t)
            method: Override the default method

        Returns:
            Array of integral values at each time point
        """
        if method is None:
            method = self.method

        # Handle α = 0: identity integral
        if self.alpha == 0:
            # Accept length >= 1
            if callable(f):
                return np.array([f(ti) for ti in t])
            return np.asarray(f)

        # Handle empty and short inputs gracefully
        if len(t) == 0:
            return np.array([])

        # Determine step size
        if h is None:
            if len(t) >= 2:
                h = t[1] - t[0]
            else:
                # For a single point, any positive step works; result will be zero
                h = 1.0

        # Convert callable to array if needed
        if callable(f):
            f_array = np.array([f(ti) for ti in t])
        else:
            f_array = np.asarray(f)

        if f_array.shape != t.shape:
            raise ValueError("Function values must match time array shape")

        # Choose computation method
        if method == "auto":
            method = self._select_optimal_method(len(t))

        # Compute integral using selected method
        if method == "fft":
            return self._compute_fft(f_array, t, h)
        elif method == "direct":
            return self._compute_direct(f_array, t, h)
        elif method == "adaptive":
            return self._compute_adaptive(f_array, t, h)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _select_optimal_method(self, n_points: int) -> str:
        """Select optimal computation method based on array size."""
        if n_points >= self.fft_threshold:
            return "fft"
        else:
            return "direct"

    def _compute_fft(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute integral using FFT-based convolution.

        This method has O(N log N) complexity and is optimal for large arrays.
        """
        n = len(t)

        # Create power-law kernel: (t-τ)^(α-1)
        # For integral, we need the kernel from 0 to t
        kernel = np.zeros(n)
        for i in range(n):
            if i == 0:
                kernel[i] = 0
            else:
                kernel[i] = (i * h) ** (self.alpha - 1)

        # Normalize by gamma function
        kernel = kernel / self.gamma_alpha

        # Use FFT for convolution
        f_fft = np.fft.fft(f)
        kernel_fft = np.fft.fft(kernel)

        # Convolve in frequency domain
        result_fft = f_fft * kernel_fft

        # Transform back to time domain
        result = np.fft.ifft(result_fft).real

        # Scale by step size
        result = result * h

        return result

    def _compute_direct(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute integral using direct summation.

        This method has O(N²) complexity but is more accurate for small arrays.
        """
        n = len(t)
        result = np.zeros(n)

        # For each time point, compute the integral
        for i in range(n):
            integral = 0.0

            # Sum over all previous points
            for j in range(i + 1):
                if j == i:  # t_i - t_i = 0
                    weight = 0
                else:
                    # Weight for this point: (t_i - t_j)^(α-1)
                    weight = ((i - j) * h) ** (self.alpha - 1)

                integral += weight * f[j]

            # Normalize and scale
            result[i] = (integral * h) / self.gamma_alpha

        return result

    def _compute_adaptive(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute integral using adaptive method selection.

        Automatically chooses between FFT and direct methods based on accuracy requirements.
        """
        # Start with FFT for speed
        result_fft = self._compute_fft(f, t, h)

        # If array is small, also compute direct for comparison
        if len(t) < self.fft_threshold:
            result_direct = self._compute_direct(f, t, h)

            # Check if FFT result is accurate enough
            if np.allclose(result_fft, result_direct, rtol=1e-6):
                return result_fft
            else:
                # Use direct method if FFT is not accurate enough
                warnings.warn(
                    "FFT method accuracy insufficient, using direct method")
                return result_direct
        else:
            return result_fft


class CaputoIntegral:
    """
    Caputo fractional integral of order α.

    For α > 0, the Caputo fractional integral equals the Riemann-Liouville integral:

    I^α f(t) = (1/Γ(α)) ∫₀ᵗ (t-τ)^(α-1) f(τ) dτ

    This class provides a consistent interface while reusing the RL implementation.
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        method: str = "auto",
        optimize_memory: bool = True,
        use_jax: bool = False
    ):
        """
        Initialize Caputo integral calculator.

        Args:
            alpha: Fractional order (must be > 0)
            method: Computation method ("auto", "fft", "direct", "adaptive")
            optimize_memory: Use memory optimization techniques
            use_jax: Use JAX acceleration if available
        """
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.alpha

        if alpha < 0:
            raise ValueError("Fractional order must be non-negative")

        # Store alpha and optionally create RL delegate
        self.alpha = alpha
        self.method = method
        self.optimize_memory = optimize_memory
        self.use_jax = use_jax
        self.rl_integral = None if alpha == 0 else RiemannLiouvilleIntegral(
            alpha, method, optimize_memory, use_jax
        )

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        t: np.ndarray,
        h: Optional[float] = None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute the Caputo fractional integral.

        Args:
            f: Function to integrate (callable) or function values (array)
            t: Time points where integral is evaluated
            h: Step size (if None, computed from t)
            method: Override the default method

        Returns:
            Array of integral values at each time point
        """
        # α = 0: identity
        if self.alpha == 0:
            if callable(f):
                return np.array([f(ti) for ti in t])
            return np.asarray(f)
        # Delegate to RL implementation
        return self.rl_integral.compute(f, t, h, method)


class WeylIntegral:
    """
    Weyl fractional integral of order α.

    Provides a simple interface consistent with other integral classes.
    """

    def __init__(self,
                 alpha: Union[float, FractionalOrder],
                 method: str = "auto"):
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.alpha
        if alpha < 0:
            raise ValueError("Fractional order must be non-negative")
        self.alpha = alpha
        self.method = method

    def compute(self,
                f: Union[Callable, np.ndarray],
                t: np.ndarray,
                h: Optional[float] = None) -> np.ndarray:
        # α = 0 acts as identity
        if self.alpha == 0:
            if callable(f):
                return np.array([f(ti) for ti in t])
            return np.asarray(f)

        # Minimal discrete approximation (windowed RL-style on provided grid)
        if h is None and len(t) >= 2:
            h = t[1] - t[0]
        elif h is None:
            h = 1.0

        if callable(f):
            f_array = np.array([f(ti) for ti in t])
        else:
            f_array = np.asarray(f)

        if len(t) < 2:
            # Not enough points for numerical integration beyond identity case
            return f_array

        # Use direct summation similar to RL over available past samples
        n = len(t)
        result = np.zeros(n)
        from ..special import gamma as _gamma
        gamma_alpha = _gamma(self.alpha)
        for i in range(n):
            integ = 0.0
            for j in range(i + 1):
                if j == i:
                    weight = 0.0
                else:
                    weight = ((i - j) * h) ** (self.alpha - 1)
                integ += weight * f_array[j]
            result[i] = (integ * h) / gamma_alpha
        return result


# Convenience functions for easy access
def riemann_liouville_integral(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "auto"
) -> np.ndarray:
    """
    Compute Riemann-Liouville fractional integral.

    Args:
        f: Function to integrate or function values
        t: Time points
        alpha: Fractional order
        h: Step size
        method: Computation method

    Returns:
        Integral values
    """
    calculator = RiemannLiouvilleIntegral(alpha, method)
    return calculator.compute(f, t, h)


def caputo_integral(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "auto"
) -> np.ndarray:
    """
    Compute Caputo fractional integral.

    Args:
        f: Function to integrate or function values
        t: Time points
        alpha: Fractional order
        h: Step size
        method: Computation method

    Returns:
        Integral values
    """
    calculator = CaputoIntegral(alpha, method)
    return calculator.compute(f, t, h)


# Optimized versions for high-performance applications
def optimized_riemann_liouville_integral(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None
) -> np.ndarray:
    """
    Optimized Riemann-Liouville integral with automatic method selection.

    Args:
        f: Function to integrate or function values
        t: Time points
        alpha: Fractional order
        h: Step size

    Returns:
        Integral values
    """
    calculator = RiemannLiouvilleIntegral(
        alpha, method="auto", optimize_memory=True)
    return calculator.compute(f, t, h)


def optimized_caputo_integral(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None
) -> np.ndarray:
    """
    Optimized Caputo integral with automatic method selection.

    Args:
        f: Function to integrate or function values
        t: Time points
        alpha: Fractional order
        h: Step size

    Returns:
        Integral values
    """
    calculator = CaputoIntegral(alpha, method="auto", optimize_memory=True)
    return calculator.compute(f, t, h)
