"""
Caputo Fractional Derivative Algorithms

This module implements various numerical algorithms for computing the Caputo
fractional derivative, including direct methods, FFT-based approaches, and
optimized implementations using JAX and NUMBA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit, prange
from typing import Union, Optional, Tuple, Callable
from scipy import interpolate

from src.core.definitions import FractionalOrder, CaputoDefinition
from src.special import gamma, mittag_leffler
from src.optimisation.numba_kernels import gamma_approx


class CaputoDerivative:
    """
    Numerical implementation of the Caputo fractional derivative.

    The Caputo derivative is defined as:
    D^α f(t) = (1/Γ(n-α)) ∫₀ᵗ (t-τ)^(n-α-1) f^(n)(τ) dτ

    where n = ⌈α⌉ and f^(n) is the nth derivative of f.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], method: str = "direct"):
        """
        Initialize Caputo derivative calculator.

        Args:
            alpha: Fractional order (0 < α < n)
            method: Numerical method ("direct", "fft", "l1", "l2")
        """
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.definition = CaputoDefinition(self.alpha)
        self.method = method.lower()

        # Validate method
        valid_methods = ["direct", "fft", "l1", "l2", "predictor_corrector"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: Optional[float] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Compute Caputo derivative of function f at point(s) t.

        Args:
            f: Function or array of function values
            t: Point(s) where to evaluate the derivative
            h: Step size (for discrete methods)
            **kwargs: Additional method-specific parameters

        Returns:
            Caputo derivative value(s)
        """
        if self.method == "direct":
            return self._compute_direct(f, t, **kwargs)
        elif self.method == "fft":
            return self._compute_fft(f, t, h, **kwargs)
        elif self.method == "l1":
            return self._compute_l1(f, t, h, **kwargs)
        elif self.method == "l2":
            return self._compute_l2(f, t, h, **kwargs)
        elif self.method == "predictor_corrector":
            return self._compute_predictor_corrector(f, t, h, **kwargs)

    def _compute_direct(
        self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Direct computation using quadrature.

        For scalar t, uses adaptive quadrature.
        For array t, uses vectorized computation.
        """
        if np.isscalar(t):
            return self._compute_direct_scalar(f, t, **kwargs)
        else:
            return self._compute_direct_vectorized(f, t, **kwargs)

    def _compute_direct_scalar(
        self, f: Union[Callable, np.ndarray], t: float, **kwargs
    ) -> float:
        """Direct computation for scalar t."""
        from scipy import integrate

        n = self.definition.n
        alpha = self.alpha.alpha

        if callable(f):
            # Function case
            def integrand(tau):
                # Handle edge case where t - tau is very small
                if abs(t - tau) < 1e-12:
                    return 0.0
                return (t - tau) ** (n - alpha - 1) * self._nth_derivative(f, tau, n)

            # Add error handling for integration
            try:
                result, _ = integrate.quad(integrand, 0, t, **kwargs)
                return result / gamma_approx(n - alpha)
            except (ValueError, RuntimeWarning) as e:
                # Handle integration errors gracefully
                if "divide by zero" in str(e) or "invalid value" in str(e):
                    # Use a small epsilon to avoid singularity
                    eps = 1e-10
                    result, _ = integrate.quad(integrand, eps, t, **kwargs)
                    return result / gamma_approx(n - alpha)
                else:
                    raise e
        else:
            # Array case - interpolate and integrate
            t_array = kwargs.get("t_array", np.linspace(0, t, len(f)))
            f_interp = interpolate.interp1d(
                t_array, f, kind="cubic", fill_value="extrapolate"
            )

            def integrand(tau):
                # Handle edge case where t - tau is very small
                if abs(t - tau) < 1e-12:
                    return 0.0
                return (t - tau) ** (n - alpha - 1) * self._nth_derivative(
                    f_interp, tau, n
                )

            # Add error handling for integration
            try:
                result, _ = integrate.quad(integrand, 0, t, **kwargs)
                return result / gamma_approx(n - alpha)
            except (ValueError, RuntimeWarning) as e:
                # Handle integration errors gracefully
                if "divide by zero" in str(e) or "invalid value" in str(e):
                    # Use a small epsilon to avoid singularity
                    eps = 1e-10
                    result, _ = integrate.quad(integrand, eps, t, **kwargs)
                    return result / gamma_approx(n - alpha)
                else:
                    raise e

    def _compute_direct_vectorized(
        self, f: Union[Callable, np.ndarray], t: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Direct computation for array t."""
        return np.array([self._compute_direct_scalar(f, ti, **kwargs) for ti in t])

    def _compute_fft(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        FFT-based computation using convolution.

        Uses the fact that Caputo derivative can be written as a convolution
        with a power-law kernel.
        """
        if callable(f):
            # Sample the function
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._fft_convolution(f_array, t_array, h)

    def _fft_convolution(self, f: np.ndarray, t: np.ndarray, h: float) -> np.ndarray:
        """FFT-based convolution for Caputo derivative."""
        n = self.definition.n
        alpha = self.alpha.alpha

        # Create power-law kernel with error handling
        kernel = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti > 0:
                try:
                    kernel[i] = (ti ** (n - alpha - 1)) / gamma_approx(n - alpha)
                except (ValueError, RuntimeWarning):
                    # Handle edge cases where power operation fails
                    kernel[i] = 0.0
            else:
                kernel[i] = 0.0

        # Pad arrays for circular convolution
        N = len(f)
        f_padded = np.pad(f, (0, N), mode="constant")
        kernel_padded = np.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = np.fft.fft(f_padded)
        kernel_fft = np.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = np.real(np.fft.ifft(conv_fft))

        # Return valid part
        return conv[:N] * h

    def _compute_l1(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        L1 scheme for Caputo derivative.

        First-order accurate scheme suitable for 0 < α < 1.
        """
        if self.alpha.alpha >= 1:
            raise ValueError("L1 scheme requires 0 < α < 1")

        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._l1_scheme(f_array, t_array, h)

    def _l1_scheme(self, f: np.ndarray, t: np.ndarray, h: float) -> np.ndarray:
        """L1 scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Coefficients for L1 scheme
        coeffs = np.zeros(N)
        coeffs[0] = 1
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - j**alpha

        # Compute derivative
        for n in range(1, N):
            result[n] = (h ** (-alpha) / gamma_approx(2 - alpha)) * np.sum(
                coeffs[: n + 1] * (f[n] - f[n - 1])
            )

        return result

    def _compute_l2(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        L2 scheme for Caputo derivative.

        Second-order accurate scheme suitable for 0 < α < 1.
        """
        if self.alpha.alpha >= 1:
            raise ValueError("L2 scheme requires 0 < α < 1")

        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._l2_scheme(f_array, t_array, h)

    def _l2_scheme(self, f: np.ndarray, t: np.ndarray, h: float) -> np.ndarray:
        """L2 scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Coefficients for L2 scheme
        coeffs = np.zeros(N)
        coeffs[0] = 1
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - 2 * j**alpha + (j - 1) ** alpha

        # Compute derivative
        for n in range(2, N):
            result[n] = (h ** (-alpha) / gamma_approx(3 - alpha)) * np.sum(
                coeffs[: n + 1] * f[n - j]
            )

        return result

    def _compute_predictor_corrector(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Predictor-corrector method for Caputo derivative.

        Uses Adams-Bashforth-Moulton type scheme.
        """
        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._predictor_corrector_scheme(f_array, t_array, h)

    def _predictor_corrector_scheme(
        self, f: np.ndarray, t: np.ndarray, h: float
    ) -> np.ndarray:
        """Predictor-corrector scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Initial values (use L1 for first few points)
        for i in range(1, min(4, N)):
            result[i] = self._l1_scheme(f[: i + 1], t[: i + 1], h)[i]

        # Predictor-corrector for remaining points
        for n in range(4, N):
            # Predictor (Adams-Bashforth)
            pred = self._predictor_step(f, result, n, h)

            # Corrector (Adams-Moulton)
            result[n] = self._corrector_step(f, result, pred, n, h)

        return result

    def _predictor_step(
        self, f: np.ndarray, result: np.ndarray, n: int, h: float
    ) -> float:
        """Predictor step of Adams-Bashforth type."""
        alpha = self.alpha.alpha
        # Simplified predictor - can be enhanced with proper Adams-Bashforth weights
        return result[n - 1] + h * (f[n] - f[n - 1])

    def _corrector_step(
        self, f: np.ndarray, result: np.ndarray, pred: float, n: int, h: float
    ) -> float:
        """Corrector step of Adams-Moulton type."""
        alpha = self.alpha.alpha
        # Simplified corrector - can be enhanced with proper Adams-Moulton weights
        return 0.5 * (result[n - 1] + pred + h * (f[n] - f[n - 1]))

    def _nth_derivative(self, f: Callable, t: float, n: int) -> float:
        """Compute nth derivative of function f at point t."""
        if n == 0:
            return f(t)
        elif n == 1:
            h = 1e-6
            return (f(t + h) - f(t - h)) / (2 * h)
        else:
            # Recursive computation for higher derivatives
            h = 1e-6
            return (
                self._nth_derivative(f, t + h, n - 1)
                - self._nth_derivative(f, t - h, n - 1)
            ) / (2 * h)


# JAX-optimized implementations
class JAXCaputoDerivative:
    """JAX-optimized Caputo derivative implementation."""

    def __init__(self, alpha: Union[float, FractionalOrder]):
        self.alpha = (
            FractionalOrder(alpha) if isinstance(alpha, (int, float)) else alpha
        )
        self.definition = CaputoDefinition(self.alpha)

    @staticmethod
    @jax.jit
    def compute_jax(
        f_values: jnp.ndarray, t_values: jnp.ndarray, alpha: float, h: float
    ) -> jnp.ndarray:
        """
        JAX-compiled Caputo derivative computation.

        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size

        Returns:
            Caputo derivative values
        """
        n = int(np.ceil(alpha))

        # Create convolution kernel
        kernel = (t_values ** (n - alpha - 1)) / gamma(n - alpha)

        # Pad for convolution
        N = len(f_values)
        f_padded = jnp.pad(f_values, (0, N), mode="constant")
        kernel_padded = jnp.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = jnp.fft.fft(f_padded)
        kernel_fft = jnp.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(jnp.fft.ifft(conv_fft))

        return conv[:N] * h


# NUMBA-optimized implementations
@jit(nopython=True, parallel=True)
def caputo_l1_numba(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    NUMBA-optimized L1 scheme for Caputo derivative.

    Args:
        f: Function values array
        alpha: Fractional order (0 < α < 1)
        h: Step size

    Returns:
        Caputo derivative values
    """
    N = len(f)
    result = np.zeros(N)

    # Coefficients
    coeffs = np.zeros(N)
    coeffs[0] = 1.0
    for j in range(1, N):
        coeffs[j] = (j + 1) ** alpha - j**alpha

    # Compute derivative
    for n in prange(1, N):
        sum_val = 0.0
        for j in range(n + 1):
            sum_val += coeffs[j] * (f[n] - f[n - 1])
        result[n] = (h ** (-alpha) / gamma_approx(2 - alpha)) * sum_val

    return result


@jit(nopython=True)
def caputo_direct_numba(f: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    NUMBA-optimized direct computation of Caputo derivative.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order

    Returns:
        Caputo derivative values
    """
    N = len(f)
    result = np.zeros(N)
    n = int(np.ceil(alpha))

    for i in range(1, N):
        # Simple trapezoidal rule for integration
        integral = 0.0
        for j in range(i):
            tau = t[j]
            weight = (t[i] - tau) ** (n - alpha - 1)
            # Approximate nth derivative using finite differences
            if j == 0:
                deriv = (f[j + 1] - f[j]) / (t[j + 1] - t[j])
            elif j == i - 1:
                deriv = (f[j] - f[j - 1]) / (t[j] - t[j - 1])
            else:
                deriv = (f[j + 1] - f[j - 1]) / (t[j + 1] - t[j - 1])
            integral += weight * deriv * (t[j + 1] - t[j])

        result[i] = integral / gamma_approx(n - alpha)

    return result


# Convenience functions
def caputo_derivative(
    f: Union[Callable, np.ndarray],
    t: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "direct",
    **kwargs,
) -> Union[float, np.ndarray]:
    """
    Convenience function for computing Caputo derivative.

    Args:
        f: Function or function values
        t: Evaluation point(s)
        alpha: Fractional order
        method: Numerical method
        **kwargs: Additional parameters

    Returns:
        Caputo derivative value(s)
    """
    calculator = CaputoDerivative(alpha, method)
    return calculator.compute(f, t, **kwargs)


def caputo_derivative_jax(
    f_values: np.ndarray,
    t_values: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: float,
) -> np.ndarray:
    """
    JAX-optimized Caputo derivative computation.

    Args:
        f_values: Function values array
        t_values: Time points array
        alpha: Fractional order
        h: Step size

    Returns:
        Caputo derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    return JAXCaputoDerivative.compute_jax(
        jnp.array(f_values), jnp.array(t_values), alpha_val, h
    )


def caputo_derivative_numba(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "l1",
) -> np.ndarray:
    """
    NUMBA-optimized Caputo derivative computation.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        method: Method ("l1" or "direct")

    Returns:
        Caputo derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    if method == "l1":
        h = t[1] - t[0] if len(t) > 1 else 1.0
        return caputo_l1_numba(f, alpha_val, h)
    elif method == "direct":
        return caputo_direct_numba(f, t, alpha_val)
    else:
        raise ValueError("Method must be 'l1' or 'direct' for NUMBA implementation")
