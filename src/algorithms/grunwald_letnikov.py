"""
Grünwald-Letnikov Fractional Derivative Algorithms

This module implements various numerical algorithms for computing the Grünwald-Letnikov
fractional derivative, including direct methods, FFT-based approaches, and
optimized implementations using JAX and NUMBA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit, prange
from typing import Union, Optional, Tuple, Callable
from scipy import interpolate

from src.core.definitions import FractionalOrder, GrunwaldLetnikovDefinition
from src.special import gamma, grunwald_letnikov_coefficients
from src.optimisation.numba_kernels import gamma_approx


class GrunwaldLetnikovDerivative:
    """
    Numerical implementation of the Grünwald-Letnikov fractional derivative.

    The Grünwald-Letnikov derivative is defined as:
    D^α f(t) = lim_{h→0} h^(-α) Σ_{j=0}^∞ (-1)^j (α choose j) f(t - jh)

    This is the most general definition and is equivalent to Riemann-Liouville
    for sufficiently smooth functions.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], method: str = "direct"):
        """
        Initialize Grünwald-Letnikov derivative calculator.

        Args:
            alpha: Fractional order (can be any real number)
            method: Numerical method ("direct", "fft", "short_memory", "variable_step")
        """
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.definition = GrunwaldLetnikovDefinition(self.alpha)
        self.method = method.lower()

        # Validate method
        valid_methods = [
            "direct",
            "fft",
            "short_memory",
            "variable_step",
            "predictor_corrector",
            "optimized_direct",
        ]
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
        Compute Grünwald-Letnikov derivative of function f at point(s) t.

        Args:
            f: Function or array of function values
            t: Point(s) where to evaluate the derivative
            h: Step size (for discrete methods)
            **kwargs: Additional method-specific parameters

        Returns:
            Grünwald-Letnikov derivative value(s)
        """
        if self.method == "direct":
            return self._compute_direct(f, t, h, **kwargs)
        elif self.method == "fft":
            return self._compute_fft(f, t, h, **kwargs)
        elif self.method == "short_memory":
            return self._compute_short_memory(f, t, h, **kwargs)
        elif self.method == "variable_step":
            return self._compute_variable_step(f, t, h, **kwargs)
        elif self.method == "predictor_corrector":
            return self._compute_predictor_corrector(f, t, h, **kwargs)
        elif self.method == "optimized_direct":
            return self._compute_optimized_direct(f, t, h, **kwargs)

    def _compute_direct(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Direct computation using the Grünwald-Letnikov formula.

        For scalar t, computes at single point.
        For array t, computes at all points.
        """
        if np.isscalar(t):
            return self._compute_direct_scalar(f, t, h, **kwargs)
        else:
            return self._compute_direct_vectorized(f, t, h, **kwargs)

    def _compute_direct_scalar(
        self, f: Union[Callable, np.ndarray], t: float, h: float, **kwargs
    ) -> float:
        """Direct computation for scalar t."""
        alpha = self.alpha.alpha

        if callable(f):
            # Function case - need to sample at discrete points
            max_steps = kwargs.get("max_steps", 1000)
            j_max = min(int(t / h), max_steps)

            result = 0.0
            for j in range(j_max + 1):
                coeff = self._grunwald_coefficient(alpha, j)
                t_j = t - j * h
                if t_j >= 0:
                    result += coeff * f(t_j)

            return result * (h ** (-alpha))
        else:
            # Array case - f is function values array
            # Find the index corresponding to time t
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

            # Find the closest index to time t
            if len(t_array) > 0:
                idx = np.argmin(np.abs(t_array - t))
                t_target = t_array[idx]
            else:
                raise ValueError("Empty time array")

            # Ensure we don't exceed array bounds
            if idx >= len(f):
                raise ValueError(f"Time t={t} exceeds available data range")

            result = 0.0
            for j in range(idx + 1):
                coeff = self._grunwald_coefficient(alpha, j)
                if idx - j >= 0:
                    result += coeff * f[idx - j]

            return result * (h ** (-alpha))

    def _compute_direct_vectorized(
        self, f: Union[Callable, np.ndarray], t: np.ndarray, h: float, **kwargs
    ) -> np.ndarray:
        """Direct computation for array t."""
        if not callable(f):
            # If f is an array, pass the time array information
            kwargs["t_array"] = t
        return np.array([self._compute_direct_scalar(f, ti, h, **kwargs) for ti in t])

    def _grunwald_coefficient(self, alpha: float, j: int) -> float:
        """Compute Grünwald-Letnikov coefficient (-1)^j (α choose j)."""
        if j == 0:
            return 1.0
        else:
            # Use gamma function for fractional binomial coefficient
            return (
                (-1) ** j
                * gamma_approx(alpha + 1)
                / (gamma_approx(j + 1) * gamma_approx(alpha - j + 1))
            )

    def _compute_fft(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        FFT-based computation using convolution.

        Uses the fact that Grünwald-Letnikov derivative can be written as a convolution
        with the Grünwald-Letnikov coefficients.
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
        """FFT-based convolution for Grünwald-Letnikov derivative."""
        alpha = self.alpha.alpha
        N = len(f)

        # Create Grünwald-Letnikov coefficient kernel
        kernel = np.zeros(N)
        for j in range(N):
            kernel[j] = self._grunwald_coefficient(alpha, j)

        # Pad arrays for circular convolution
        f_padded = np.pad(f, (0, N), mode="constant")
        kernel_padded = np.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = np.fft.fft(f_padded)
        kernel_fft = np.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = np.real(np.fft.ifft(conv_fft))

        # Return valid part
        return conv[:N] * (h ** (-alpha))

    def _compute_short_memory(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Short memory principle for Grünwald-Letnikov derivative.

        Uses only recent history to reduce computational cost.
        """
        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._short_memory_scheme(f_array, t_array, h, **kwargs)

    def _short_memory_scheme(
        self, f: np.ndarray, t: np.ndarray, h: float, **kwargs
    ) -> np.ndarray:
        """Short memory scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Memory length parameter
        L = kwargs.get("memory_length", min(100, N))

        # Compute derivative
        for n in range(1, N):
            # Use only recent history
            j_max = min(n, L)
            sum_val = 0.0

            for j in range(j_max + 1):
                coeff = self._grunwald_coefficient(alpha, j)
                sum_val += coeff * f[n - j]

            result[n] = sum_val * (h ** (-alpha))

        return result

    def _compute_variable_step(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Variable step size method for Grünwald-Letnikov derivative.

        Adapts step size based on local behavior of the function.
        """
        if callable(f):
            t_max = np.max(t) if hasattr(t, "__len__") else t
            t_array = np.arange(0, t_max + h, h)
            f_array = np.array([f(ti) for ti in t_array])
        else:
            f_array = f
            t_array = kwargs.get("t_array", np.arange(len(f)) * h)

        return self._variable_step_scheme(f_array, t_array, h, **kwargs)

    def _variable_step_scheme(
        self, f: np.ndarray, t: np.ndarray, h: float, **kwargs
    ) -> np.ndarray:
        """Variable step size scheme implementation."""
        alpha = self.alpha.alpha
        N = len(f)
        result = np.zeros(N)

        # Adaptive step size parameters
        tol = kwargs.get("tolerance", 1e-6)
        max_iter = kwargs.get("max_iterations", 10)

        for n in range(1, N):
            # Start with base step size
            current_h = h
            prev_result = 0.0

            for iter_count in range(max_iter):
                # Compute with current step size
                j_max = int(t[n] / current_h)
                sum_val = 0.0

                for j in range(j_max + 1):
                    coeff = self._grunwald_coefficient(alpha, j)
                    t_j = t[n] - j * current_h
                    # Interpolate function value at t_j
                    if t_j <= t[0]:
                        f_val = f[0]
                    elif t_j >= t[-1]:
                        f_val = f[-1]
                    else:
                        # Linear interpolation
                        idx = int(t_j / h)
                        if idx < len(f) - 1:
                            f_val = f[idx] + (f[idx + 1] - f[idx]) * (t_j - t[idx]) / h
                        else:
                            f_val = f[-1]

                    sum_val += coeff * f_val

                current_result = sum_val * (current_h ** (-alpha))

                # Check convergence
                if abs(current_result - prev_result) < tol:
                    break

                prev_result = current_result
                current_h *= 0.5  # Reduce step size

            result[n] = current_result

        return result

    def _compute_predictor_corrector(
        self,
        f: Union[Callable, np.ndarray],
        t: Union[float, np.ndarray],
        h: float,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Predictor-corrector method for Grünwald-Letnikov derivative.

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

        # Initial values (use direct method for first few points)
        for i in range(1, min(4, N)):
            result[i] = self._compute_direct_scalar(f, t[i], h)

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


    def _compute_optimized_direct(
        self, f: Union[Callable, np.ndarray], t: Union[float, np.ndarray], h: Optional[float] = None, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Optimized direct implementation for Grünwald-Letnikov derivative.
        
        This method uses the optimized direct computation for maximum efficiency.
        """
        from src.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
        
        # Create optimized calculator
        optimized_calc = OptimizedGrunwaldLetnikov(self.alpha)
        
        # Compute using optimized direct method
        return optimized_calc._grunwald_letnikov_numpy(f, h)


# JAX-optimized implementations
class JAXGrunwaldLetnikovDerivative:
    """JAX-optimized Grünwald-Letnikov derivative implementation."""

    def __init__(self, alpha: Union[float, FractionalOrder]):
        self.alpha = (
            FractionalOrder(alpha) if isinstance(alpha, (int, float)) else alpha
        )
        self.definition = GrunwaldLetnikovDefinition(self.alpha)

    @staticmethod
    @jax.jit
    def compute_jax(
        f_values: jnp.ndarray, t_values: jnp.ndarray, alpha: float, h: float
    ) -> jnp.ndarray:
        """
        JAX-compiled Grünwald-Letnikov derivative computation.

        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size

        Returns:
            Grünwald-Letnikov derivative values
        """
        N = len(f_values)

        # Create Grünwald-Letnikov coefficient kernel
        kernel = jnp.zeros(N)

        # Compute coefficients
        def compute_coeff(j):
            return jax.lax.cond(
                j == 0,
                lambda _: 1.0,
                lambda _: (-1) ** j
                * gamma_approx(alpha + 1)
                / (gamma_approx(j + 1) * gamma_approx(alpha - j + 1)),
                operand=None,
            )

        kernel = jax.vmap(compute_coeff)(jnp.arange(N))

        # Pad for convolution
        f_padded = jnp.pad(f_values, (0, N), mode="constant")
        kernel_padded = jnp.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = jnp.fft.fft(f_padded)
        kernel_fft = jnp.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(jnp.fft.ifft(conv_fft))

        return conv[:N] * (h ** (-alpha))


# NUMBA-optimized implementations
@jit(nopython=True, parallel=True)
def grunwald_letnikov_direct_numba(f: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """
    NUMBA-optimized direct computation of Grünwald-Letnikov derivative.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size

    Returns:
        Grünwald-Letnikov derivative values
    """
    N = len(f)
    result = np.zeros(N)

    # Compute coefficients
    coeffs = np.zeros(N)
    coeffs[0] = 1.0
    for j in range(1, N):
        coeffs[j] = coeffs[j - 1] * (1 - (alpha + 1) / j)

    # Compute derivative
    for n in prange(1, N):
        sum_val = 0.0
        for j in range(n + 1):
            sum_val += coeffs[j] * f[n - j]
        result[n] = sum_val * (h ** (-alpha))

    return result


@jit(nopython=True, parallel=True)
def grunwald_letnikov_short_memory_numba(
    f: np.ndarray, alpha: float, h: float, memory_length: int
) -> np.ndarray:
    """
    NUMBA-optimized short memory principle for Grünwald-Letnikov derivative.

    Args:
        f: Function values array
        alpha: Fractional order
        h: Step size
        memory_length: Number of history points to use

    Returns:
        Grünwald-Letnikov derivative values
    """
    N = len(f)
    result = np.zeros(N)

    # Compute coefficients
    coeffs = np.zeros(memory_length + 1)
    coeffs[0] = 1.0
    for j in range(1, memory_length + 1):
        coeffs[j] = coeffs[j - 1] * (1 - (alpha + 1) / j)

    # Compute derivative
    for n in prange(1, N):
        j_max = min(n, memory_length)
        sum_val = 0.0
        for j in range(j_max + 1):
            sum_val += coeffs[j] * f[n - j]
        result[n] = sum_val * (h ** (-alpha))

    return result


@jit(nopython=True)
def grunwald_coefficient_numba(alpha: float, j: int) -> float:
    """
    NUMBA-optimized computation of Grünwald-Letnikov coefficient.

    Args:
        alpha: Fractional order
        j: Index

    Returns:
        Coefficient value
    """
    if j == 0:
        return 1.0
    else:
        # Use recursive formula for efficiency
        result = 1.0
        for k in range(1, j + 1):
            result *= 1 - (alpha + 1) / k
        return result


# Convenience functions
def grunwald_letnikov_derivative(
    f: Union[Callable, np.ndarray],
    t: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "direct",
    **kwargs,
) -> Union[float, np.ndarray]:
    """
    Convenience function for computing Grünwald-Letnikov derivative.

    Args:
        f: Function or function values
        t: Evaluation point(s)
        alpha: Fractional order
        method: Numerical method
        **kwargs: Additional parameters

    Returns:
        Grünwald-Letnikov derivative value(s)
    """
    calculator = GrunwaldLetnikovDerivative(alpha, method)
    return calculator.compute(f, t, **kwargs)


def grunwald_letnikov_derivative_jax(
    f_values: np.ndarray,
    t_values: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: float,
) -> np.ndarray:
    """
    JAX-optimized Grünwald-Letnikov derivative computation.

    Args:
        f_values: Function values array
        t_values: Time points array
        alpha: Fractional order
        h: Step size

    Returns:
        Grünwald-Letnikov derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    return JAXGrunwaldLetnikovDerivative.compute_jax(
        jnp.array(f_values), jnp.array(t_values), alpha_val, h
    )


def grunwald_letnikov_derivative_numba(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "direct",
    **kwargs,
) -> np.ndarray:
    """
    NUMBA-optimized Grünwald-Letnikov derivative computation.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        method: Method ("direct" or "short_memory")
        **kwargs: Additional parameters

    Returns:
        Grünwald-Letnikov derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    h = t[1] - t[0] if len(t) > 1 else 1.0

    if method == "direct":
        return grunwald_letnikov_direct_numba(f, alpha_val, h)
    elif method == "short_memory":
        memory_length = kwargs.get("memory_length", 100)
        return grunwald_letnikov_short_memory_numba(f, alpha_val, h, memory_length)
    else:
        raise ValueError(
            "Method must be 'direct' or 'short_memory' for NUMBA implementation"
        )
