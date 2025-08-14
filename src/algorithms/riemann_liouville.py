"""
Riemann-Liouville Fractional Derivative Algorithms

This module implements various numerical algorithms for computing the Riemann-Liouville
fractional derivative, including direct methods, FFT-based approaches, and
optimized implementations using JAX and NUMBA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit, prange
from typing import Union, Optional, Tuple, Callable
from scipy import interpolate, integrate

from src.core.definitions import FractionalOrder, RiemannLiouvilleDefinition
from src.special import gamma, mittag_leffler
from src.optimisation.numba_kernels import gamma_approx


class RiemannLiouvilleDerivative:
    """
    Numerical implementation of the Riemann-Liouville fractional derivative.

    The Riemann-Liouville derivative is defined as:
    D^α f(t) = (1/Γ(n-α)) (d/dt)^n ∫₀ᵗ (t-τ)^(n-α-1) f(τ) dτ

    where n = ⌈α⌉.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], method: str = "direct"):
        """
        Initialize Riemann-Liouville derivative calculator.

        Args:
            alpha: Fractional order (0 < α < n)
            method: Numerical method ("direct", "fft", "grunwald_letnikov")
        """
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.definition = RiemannLiouvilleDefinition(self.alpha)
        self.method = method.lower()

        # Validate method
        valid_methods = ["direct", "fft", "grunwald_letnikov", "predictor_corrector"]
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
        Compute Riemann-Liouville derivative of function f at point(s) t.

        Args:
            f: Function or array of function values
            t: Point(s) where to evaluate the derivative
            h: Step size (for discrete methods)
            **kwargs: Additional method-specific parameters

        Returns:
            Riemann-Liouville derivative value(s)
        """
        if self.method == "direct":
            return self._compute_direct(f, t, **kwargs)
        elif self.method == "fft":
            return self._compute_fft(f, t, h, **kwargs)
        elif self.method == "grunwald_letnikov":
            return self._compute_grunwald_letnikov(f, t, h, **kwargs)
        elif self.method == "predictor_corrector":
            return self._compute_predictor_corrector(f, t, h, **kwargs)

    def _compute_direct(
        self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Direct computation using quadrature and differentiation.

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
                return (t - tau) ** (n - alpha - 1) * f(tau)

            # Compute the integral with error handling
            try:
                integral, _ = integrate.quad(integrand, 0, t, **kwargs)
            except (ValueError, RuntimeWarning) as e:
                # Handle integration errors gracefully
                if "divide by zero" in str(e) or "invalid value" in str(e):
                    # Use a small epsilon to avoid singularity
                    eps = 1e-10
                    integral, _ = integrate.quad(integrand, eps, t, **kwargs)
                else:
                    raise e

            # Apply the nth derivative
            return self._nth_derivative_of_integral(f, t, n, alpha, **kwargs)
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
                return (t - tau) ** (n - alpha - 1) * f_interp(tau)

            # Compute the integral with error handling
            try:
                integral, _ = integrate.quad(integrand, 0, t, **kwargs)
            except (ValueError, RuntimeWarning) as e:
                # Handle integration errors gracefully
                if "divide by zero" in str(e) or "invalid value" in str(e):
                    # Use a small epsilon to avoid singularity
                    eps = 1e-10
                    integral, _ = integrate.quad(integrand, eps, t, **kwargs)
                else:
                    raise e

            return self._nth_derivative_of_integral(f_interp, t, n, alpha, **kwargs)

    def _compute_direct_vectorized(
        self, f: Union[Callable, np.ndarray], t: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Direct computation for array t."""
        return np.array([self._compute_direct_scalar(f, ti, **kwargs) for ti in t])

    def _nth_derivative_of_integral(
        self, f: Callable, t: float, n: int, alpha: float, **kwargs
    ) -> float:
        """Compute the nth derivative of the fractional integral."""
        if n == 0:
            # No derivative needed
            def integrand(tau):
                return (t - tau) ** (-alpha - 1) * f(tau)

            integral, _ = integrate.quad(integrand, 0, t, **kwargs)
            return integral / gamma_approx(-alpha)
        else:
            # Use finite differences to approximate the nth derivative
            h = kwargs.get("h", 1e-6)
            if n == 1:
                # First derivative
                t_plus = t + h
                t_minus = t - h

                def integrand_plus(tau):
                    return (t_plus - tau) ** (n - alpha - 1) * f(tau)

                def integrand_minus(tau):
                    return (t_minus - tau) ** (n - alpha - 1) * f(tau)

                integral_plus, _ = integrate.quad(integrand_plus, 0, t_plus, **kwargs)
                integral_minus, _ = integrate.quad(
                    integrand_minus, 0, t_minus, **kwargs
                )

                return (integral_plus - integral_minus) / (
                    2 * h * gamma_approx(n - alpha)
                )
            else:
                # Higher derivatives using recursive finite differences
                return (
                    self._nth_derivative_of_integral(f, t + h, n - 1, alpha, **kwargs)
                    - self._nth_derivative_of_integral(f, t - h, n - 1, alpha, **kwargs)
                ) / (2 * h)

    def _compute_fft(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        FFT-based computation using convolution.

        Uses the fact that Riemann-Liouville derivative can be written as a convolution
        with a power-law kernel followed by differentiation.
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
        """FFT-based convolution for Riemann-Liouville derivative."""
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

        # Apply nth derivative using finite differences
        result = np.zeros(N)
        result[:n] = 0  # First n points are zero

        for i in range(n, N):
            if n == 1:
                # First derivative
                if i < N - 1:
                    result[i] = (conv[i + 1] - conv[i - 1]) / (2 * h)
                else:
                    result[i] = (conv[i] - conv[i - 1]) / h
            else:
                # Higher derivatives using recursive finite differences
                if i < N - n:
                    result[i] = self._finite_difference_nth(conv, i, n, h)
                else:
                    # Use backward differences for boundary points
                    result[i] = self._backward_difference_nth(conv, i, n, h)

        return result * h

    def _finite_difference_nth(
        self, arr: np.ndarray, i: int, n: int, h: float
    ) -> float:
        """Compute nth derivative using central finite differences."""
        if n == 1:
            return (arr[i + 1] - arr[i - 1]) / (2 * h)
        else:
            return (
                self._finite_difference_nth(arr, i + 1, n - 1, h)
                - self._finite_difference_nth(arr, i - 1, n - 1, h)
            ) / (2 * h)

    def _backward_difference_nth(
        self, arr: np.ndarray, i: int, n: int, h: float
    ) -> float:
        """Compute nth derivative using backward finite differences."""
        if n == 1:
            return (arr[i] - arr[i - 1]) / h
        else:
            return (
                self._backward_difference_nth(arr, i, n - 1, h)
                - self._backward_difference_nth(arr, i - 1, n - 1, h)
            ) / h

    def _compute_grunwald_letnikov(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Grünwald-Letnikov approximation for Riemann-Liouville derivative.

        This is equivalent to Riemann-Liouville for sufficiently smooth functions.
        """
        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._grunwald_letnikov_scheme(f_array, t_array, h)

    def _grunwald_letnikov_scheme(
        self, f: np.ndarray, t: np.ndarray, h: float
    ) -> np.ndarray:
        """Grünwald-Letnikov scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Import Grünwald-Letnikov coefficients
        from src.special import grunwald_letnikov_coefficients

        # Get coefficients
        coeffs = grunwald_letnikov_coefficients(alpha, N)

        # Compute derivative
        for n in range(1, N):
            result[n] = (h ** (-alpha)) * np.sum(coeffs[: n + 1] * f[n:0:-1])

        return result

    def _compute_predictor_corrector(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Predictor-corrector method for Riemann-Liouville derivative.

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

        # Initial values (use Grünwald-Letnikov for first few points)
        for i in range(1, min(4, N)):
            result[i] = self._grunwald_letnikov_scheme(f[: i + 1], t[: i + 1], h)[i]

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


# JAX-optimized implementations
class JAXRiemannLiouvilleDerivative:
    """JAX-optimized Riemann-Liouville derivative implementation."""

    def __init__(self, alpha: Union[float, FractionalOrder]):
        self.alpha = (
            FractionalOrder(alpha) if isinstance(alpha, (int, float)) else alpha
        )
        self.definition = RiemannLiouvilleDefinition(self.alpha)

    @staticmethod
    @jax.jit
    def compute_jax(
        f_values: jnp.ndarray, t_values: jnp.ndarray, alpha: float, h: float
    ) -> jnp.ndarray:
        """
        JAX-compiled Riemann-Liouville derivative computation.

        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size

        Returns:
            Riemann-Liouville derivative values
        """
        n = int(np.ceil(alpha))

        # Create convolution kernel
        kernel = (t_values ** (n - alpha - 1)) / gamma_approx(n - alpha)

        # Pad for convolution
        N = len(f_values)
        f_padded = jnp.pad(f_values, (0, N), mode="constant")
        kernel_padded = jnp.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = jnp.fft.fft(f_padded)
        kernel_fft = jnp.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(jnp.fft.ifft(conv_fft))

        # Apply nth derivative using finite differences
        result = jnp.zeros(N)

        # First n points are zero
        result = result.at[:n].set(0.0)

        # Apply finite differences for remaining points
        for i in range(n, N):
            if n == 1:
                if i < N - 1:
                    result = result.at[i].set((conv[i + 1] - conv[i - 1]) / (2 * h))
                else:
                    result = result.at[i].set((conv[i] - conv[i - 1]) / h)
            else:
                # Simplified higher derivative computation
                if i < N - n:
                    result = result.at[i].set(
                        (conv[i + 1] - 2 * conv[i] + conv[i - 1]) / (h**2)
                    )
                else:
                    result = result.at[i].set((conv[i] - conv[i - 1]) / h)

        return result * h


# NUMBA-optimized implementations
@jit(nopython=True, parallel=True)
def riemann_liouville_grunwald_numba(
    f: np.ndarray, alpha: float, h: float
) -> np.ndarray:
    """
    NUMBA-optimized Grünwald-Letnikov scheme for Riemann-Liouville derivative.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size

    Returns:
        Riemann-Liouville derivative values
    """
    N = len(f)
    result = np.zeros(N)

    # Coefficients for Grünwald-Letnikov
    coeffs = np.zeros(N)
    coeffs[0] = 1.0
    for j in range(1, N):
        coeffs[j] = coeffs[j - 1] * (1 - (alpha + 1) / j)

    # Compute derivative
    for n in prange(1, N):
        sum_val = 0.0
        for j in range(n + 1):
            sum_val += coeffs[j] * f[n - j]
        result[n] = (h ** (-alpha)) * sum_val

    return result


@jit(nopython=True)
def riemann_liouville_direct_numba(
    f: np.ndarray, t: np.ndarray, alpha: float
) -> np.ndarray:
    """
    NUMBA-optimized direct computation of Riemann-Liouville derivative.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order

    Returns:
        Riemann-Liouville derivative values
    """
    N = len(f)
    result = np.zeros(N)
    n = int(np.ceil(alpha))

    for i in range(n, N):
        # Compute integral part
        integral = 0.0
        for j in range(i):
            tau = t[j]
            weight = (t[i] - tau) ** (n - alpha - 1)
            integral += weight * f[j] * (t[j + 1] - t[j])

        # Apply nth derivative using finite differences
        if n == 1:
            if i < N - 1:
                result[i] = (integral - result[i - 1]) / (2 * (t[i] - t[i - 1]))
            else:
                result[i] = (integral - result[i - 1]) / (t[i] - t[i - 1])
        else:
            # Simplified higher derivative
            result[i] = integral / gamma_approx(n - alpha)

    return result


# Convenience functions
def riemann_liouville_derivative(
    f: Union[Callable, np.ndarray],
    t: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "direct",
    **kwargs,
) -> Union[float, np.ndarray]:
    """
    Convenience function for computing Riemann-Liouville derivative.

    Args:
        f: Function or function values
        t: Evaluation point(s)
        alpha: Fractional order
        method: Numerical method
        **kwargs: Additional parameters

    Returns:
        Riemann-Liouville derivative value(s)
    """
    calculator = RiemannLiouvilleDerivative(alpha, method)
    return calculator.compute(f, t, **kwargs)


def riemann_liouville_derivative_jax(
    f_values: np.ndarray,
    t_values: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: float,
) -> np.ndarray:
    """
    JAX-optimized Riemann-Liouville derivative computation.

    Args:
        f_values: Function values array
        t_values: Time points array
        alpha: Fractional order
        h: Step size

    Returns:
        Riemann-Liouville derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    return JAXRiemannLiouvilleDerivative.compute_jax(
        jnp.array(f_values), jnp.array(t_values), alpha_val, h
    )


def riemann_liouville_derivative_numba(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "grunwald",
) -> np.ndarray:
    """
    NUMBA-optimized Riemann-Liouville derivative computation.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        method: Method ("grunwald" or "direct")

    Returns:
        Riemann-Liouville derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    if method == "grunwald":
        h = t[1] - t[0] if len(t) > 1 else 1.0
        return riemann_liouville_grunwald_numba(f, alpha_val, h)
    elif method == "direct":
        return riemann_liouville_direct_numba(f, t, alpha_val)
    else:
        raise ValueError(
            "Method must be 'grunwald' or 'direct' for NUMBA implementation"
        )
