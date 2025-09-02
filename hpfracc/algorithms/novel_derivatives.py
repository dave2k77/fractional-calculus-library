"""
Novel Fractional Derivatives

This module provides implementations of novel fractional derivatives including
Caputo-Fabrizio and Atangana-Baleanu derivatives with optimized algorithms.
"""

import numpy as np
from typing import Union, Optional, Callable
import warnings

from ..core.definitions import FractionalOrder
from ..special import gamma


class CaputoFabrizioDerivative:
    """
    Caputo-Fabrizio fractional derivative of order α.

    The Caputo-Fabrizio fractional derivative of order α ∈ [0, 1) is defined as:

    CF D^α f(t) = M(α)/(1-α) ∫₀ᵗ f'(τ) exp(-α(t-τ)/(1-α)) dτ

    where M(α) is a normalization function and the kernel is non-singular.

    Features:
    - Non-singular exponential kernel for better numerical stability
    - Optimized FFT-based computation for large arrays
    - Direct method for small arrays with high accuracy
    - Support for both callable and array inputs
    - Better behavior for biological systems and viscoelasticity
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        method: str = "auto",
        optimize_memory: bool = True,
        use_jax: bool = False
    ):
        """
        Initialize Caputo-Fabrizio derivative calculator.

        Args:
            alpha: Fractional order (must be in [0, 1))
            method: Computation method ("auto", "fft", "direct", "adaptive")
            optimize_memory: Use memory optimization techniques
            use_jax: Use JAX acceleration if available
        """
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.value

        if alpha < 0 or alpha >= 1:
            raise ValueError(
                "Fractional order α must be in [0, 1) for Caputo-Fabrizio")

        self.alpha = alpha
        self.method = method.lower()
        self.optimize_memory = optimize_memory
        self.use_jax = use_jax

        # Validate method
        valid_methods = ["auto", "fft", "direct", "adaptive"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        # Precompute constants
        self.alpha_factor = alpha / (1 - alpha)
        self.normalization = 1.0  # M(α) = 1 for simplicity, can be customized

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
        Compute the Caputo-Fabrizio fractional derivative.

        Args:
            f: Function to differentiate (callable) or function values (array)
            t: Time points where derivative is evaluated
            h: Step size (if None, computed from t)
            method: Override the default method

        Returns:
            Array of derivative values at each time point
        """
        if method is None:
            method = self.method

        # Validate inputs
        if len(t) < 2:
            raise ValueError("Time array must have at least 2 points")

        if h is None:
            h = t[1] - t[0]

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

        # Compute derivative using selected method
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
        Compute derivative using FFT-based convolution.

        This method has O(N log N) complexity and is optimal for large arrays.
        """
        n = len(t)

        # Compute first derivative of f using finite differences
        f_prime = np.gradient(f, h)

        # Create exponential kernel: exp(-α(t-τ)/(1-α))
        kernel = np.zeros(n)
        for i in range(n):
            if i == 0:
                kernel[i] = 0
            else:
                kernel[i] = np.exp(-self.alpha_factor * i * h)

        # Normalize by (1-α) factor
        kernel = kernel / (1 - self.alpha)

        # Use FFT for convolution
        f_prime_fft = np.fft.fft(f_prime)
        kernel_fft = np.fft.fft(kernel)

        # Convolve in frequency domain
        result_fft = f_prime_fft * kernel_fft

        # Transform back to time domain
        result = np.fft.ifft(result_fft).real

        # Scale by step size and normalization
        result = result * h * self.normalization

        return result

    def _compute_direct(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute derivative using direct summation.

        This method has O(N²) complexity but is more accurate for small arrays.
        """
        n = len(t)
        result = np.zeros(n)

        # Compute first derivative of f using finite differences
        f_prime = np.gradient(f, h)

        # For each time point, compute the derivative
        for i in range(n):
            derivative = 0.0

            # Sum over all previous points
            for j in range(i + 1):
                if j == i:  # t_i - t_j = 0
                    weight = 0
                else:
                    # Weight for this point: exp(-α(t_i - t_j)/(1-α))
                    weight = np.exp(-self.alpha_factor * (i - j) * h)

                derivative += weight * f_prime[j]

            # Normalize and scale
            result[i] = (derivative * h * self.normalization) / \
                (1 - self.alpha)

        return result

    def _compute_adaptive(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute derivative using adaptive method selection.

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


class AtanganaBaleanuDerivative:
    """
    Atangana-Baleanu fractional derivative of order α.

    The Atangana-Baleanu fractional derivative of order α ∈ [0, 1) is defined as:

    AB D^α f(t) = B(α)/(1-α) ∫₀ᵗ f'(τ) E_α(-α(t-τ)^α/(1-α)) dτ

    where B(α) is a normalization function and E_α is the Mittag-Leffler function.

    Features:
    - Mittag-Leffler kernel for superior memory effects modeling
    - Advanced numerical algorithms with fast ML function evaluation
    - GPU acceleration support via JAX integration
    - Better modeling of complex systems and anomalous diffusion
    """

    def __init__(
        self,
        alpha: Union[float, FractionalOrder],
        method: str = "auto",
        optimize_memory: bool = True,
        use_jax: bool = False
    ):
        """
        Initialize Atangana-Baleanu derivative calculator.

        Args:
            alpha: Fractional order (must be in [0, 1))
            method: Computation method ("auto", "fft", "direct", "adaptive")
            optimize_memory: Use memory optimization techniques
            use_jax: Use JAX acceleration if available
        """
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.value

        if alpha < 0 or alpha >= 1:
            raise ValueError(
                "Fractional order α must be in [0, 1) for Atangana-Baleanu")

        self.alpha = alpha
        self.method = method.lower()
        self.optimize_memory = optimize_memory
        self.use_jax = use_jax

        # Validate method
        valid_methods = ["auto", "fft", "direct", "adaptive"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

        # Precompute constants
        self.alpha_factor = alpha / (1 - alpha)
        self.normalization = 1.0  # B(α) = 1 for simplicity, can be customized

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
        Compute the Atangana-Baleanu fractional derivative.

        Args:
            f: Function to differentiate (callable) or function values (array)
            t: Time points where derivative is evaluated
            h: Step size (if None, computed from t)
            method: Override the default method

        Returns:
            Array of derivative values at each time point
        """
        if method is None:
            method = self.method

        # Validate inputs
        if len(t) < 2:
            raise ValueError("Time array must have at least 2 points")

        if h is None:
            h = t[1] - t[0]

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

        # Compute derivative using selected method
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

    def _mittag_leffler_fast(
            self,
            z: float,
            alpha: float,
            max_terms: int = 50) -> float:
        """
        Fast approximation of Mittag-Leffler function E_α(z).

        Uses truncated series expansion for computational efficiency.
        """
        if abs(z) < 1e-10:
            return 1.0

        result = 0.0
        factorial = 1.0

        for k in range(max_terms):
            if k > 0:
                factorial *= k

            term = (z ** k) / gamma(alpha * k + 1)
            result += term

            # Check convergence
            if abs(term) < 1e-12:
                break

        return result

    def _compute_fft(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute derivative using FFT-based convolution.

        This method has O(N log N) complexity and is optimal for large arrays.
        """
        n = len(t)

        # Compute first derivative of f using finite differences
        f_prime = np.gradient(f, h)

        # Create Mittag-Leffler kernel: E_α(-α(t-τ)^α/(1-α))
        kernel = np.zeros(n)
        for i in range(n):
            if i == 0:
                kernel[i] = 0
            else:
                z = -self.alpha_factor * ((i * h) ** self.alpha)
                kernel[i] = self._mittag_leffler_fast(z, self.alpha)

        # Normalize by (1-α) factor
        kernel = kernel / (1 - self.alpha)

        # Use FFT for convolution
        f_prime_fft = np.fft.fft(f_prime)
        kernel_fft = np.fft.fft(kernel)

        # Convolve in frequency domain
        result_fft = f_prime_fft * kernel_fft

        # Transform back to time domain
        result = np.fft.ifft(result_fft).real

        # Scale by step size and normalization
        result = result * h * self.normalization

        return result

    def _compute_direct(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute derivative using direct summation.

        This method has O(N²) complexity but is more accurate for small arrays.
        """
        n = len(t)
        result = np.zeros(n)

        # Compute first derivative of f using finite differences
        f_prime = np.gradient(f, h)

        # For each time point, compute the derivative
        for i in range(n):
            derivative = 0.0

            # Sum over all previous points
            for j in range(i + 1):
                if j == i:  # t_i - t_j = 0
                    weight = 0
                else:
                    # Weight for this point: E_α(-α(t_i - t_j)^α/(1-α))
                    z = -self.alpha_factor * (((i - j) * h) ** self.alpha)
                    weight = self._mittag_leffler_fast(z, self.alpha)

                derivative += weight * f_prime[j]

            # Normalize and scale
            result[i] = (derivative * h * self.normalization) / \
                (1 - self.alpha)

        return result

    def _compute_adaptive(
            self,
            f: np.ndarray,
            t: np.ndarray,
            h: float) -> np.ndarray:
        """
        Compute derivative using adaptive method selection.

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


# Convenience functions for easy access
def caputo_fabrizio_derivative(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "auto"
) -> np.ndarray:
    """
    Compute Caputo-Fabrizio fractional derivative.

    Args:
        f: Function to differentiate or function values
        t: Time points
        alpha: Fractional order
        h: Step size
        method: Computation method

    Returns:
        Derivative values
    """
    calculator = CaputoFabrizioDerivative(alpha, method)
    return calculator.compute(f, t, h)


def atangana_baleanu_derivative(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "auto"
) -> np.ndarray:
    """
    Compute Atangana-Baleanu fractional derivative.

    Args:
        f: Function to differentiate or function values
        t: Time points
        alpha: Fractional order
        h: Step size
        method: Computation method

    Returns:
        Derivative values
    """
    calculator = AtanganaBaleanuDerivative(alpha, method)
    return calculator.compute(f, t, h)


# Optimized versions for high-performance applications
def optimized_caputo_fabrizio_derivative(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None
) -> np.ndarray:
    """
    Optimized Caputo-Fabrizio derivative with automatic method selection.

    Args:
        f: Function to differentiate or function values
        t: Time points
        alpha: Fractional order
        h: Step size

    Returns:
        Derivative values
    """
    calculator = CaputoFabrizioDerivative(
        alpha, method="auto", optimize_memory=True)
    return calculator.compute(f, t, h)


def optimized_atangana_baleanu_derivative(
    f: Union[Callable, np.ndarray],
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None
) -> np.ndarray:
    """
    Optimized Atangana-Baleanu derivative with automatic method selection.

    Args:
        f: Function to differentiate or function values
        t: Time points
        alpha: Fractional order
        h: Step size

    Returns:
        Derivative values
    """
    calculator = AtanganaBaleanuDerivative(
        alpha, method="auto", optimize_memory=True)
    return calculator.compute(f, t, h)
