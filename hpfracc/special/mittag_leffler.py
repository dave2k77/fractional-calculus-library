"""
Mittag-Leffler function for fractional calculus.

This module provides optimized implementations of the Mittag-Leffler function,
which is a fundamental special function in fractional calculus and appears
naturally in solutions of fractional differential equations.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit
from typing import Union
from .gamma_beta import gamma


class MittagLefflerFunction:
    """
    Mittag-Leffler function implementation with multiple optimization strategies.

    The Mittag-Leffler function is defined as:
    E_α,β(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)

    Special cases:
    - E_1,1(z) = e^z (exponential function)
    - E_2,1(-z^2) = cos(z) (cosine function)
    - E_2,2(-z^2) = sin(z)/z (sinc function)
    """

    def __init__(
            self,
            use_jax: bool = False,
            use_numba: bool = True,
            max_terms: int = 100):
        """
        Initialize Mittag-Leffler function calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            max_terms: Maximum number of terms in the series expansion
        """
        self.use_jax = use_jax
        self.use_numba = use_numba
        self.max_terms = max_terms

        if use_jax:
            self._ml_jax = jax.jit(self._ml_jax_impl)

    def compute(
        self,
        z: Union[float, np.ndarray, jnp.ndarray],
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Union[float, np.ndarray, jnp.ndarray]:
        """
        Compute the Mittag-Leffler function E_α,β(z).

        Args:
            z: Input value(s)
            alpha: First parameter (default: 1.0)
            beta: Second parameter (default: 1.0)

        Returns:
            Mittag-Leffler function value(s)
        """
        # Handle array inputs with fallback mechanism
        if isinstance(z, np.ndarray):
            return self._ml_numpy_array(z, alpha, beta)
        elif self.use_jax and isinstance(z, (jnp.ndarray, float)):
            try:
                return self._ml_jax(z, alpha, beta)
            except:
                # Fallback to NumPy if JAX fails
                return self._ml_numpy_array(np.array([z]), alpha, beta)[0] if np.isscalar(z) else self._ml_numpy_array(z, alpha, beta)
        elif self.use_numba and isinstance(z, (float, int)):
            try:
                return self._ml_numba_scalar(z, alpha, beta)
            except:
                # Fallback to NumPy if Numba fails
                return self._ml_numpy_array(np.array([z]), alpha, beta)[0]
        else:
            return self._ml_scipy(z, alpha, beta)

    @staticmethod
    def _ml_scipy(
        z: Union[float, np.ndarray], alpha: float, beta: float
    ) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        # SciPy doesn't have mittag_leffler in all versions, so we use our own
        # implementation
        if alpha == 1.0 and beta == 1.0:
            return np.exp(z)
        elif alpha == 2.0 and beta == 1.0:
            return np.cos(np.sqrt(-z))
        elif alpha == 2.0 and beta == 2.0:
            if z == 0:
                return 1.0
            else:
                return np.sin(np.sqrt(z)) / np.sqrt(z)
        else:
            # Fallback to our own implementation
            return MittagLefflerFunction._ml_numba_scalar(z, alpha, beta)

    @staticmethod
    @jit(nopython=True)
    def _ml_numba_scalar(z: float, alpha: float, beta: float) -> float:
        """
        NUMBA-optimized Mittag-Leffler function for scalar inputs.

        Uses series expansion with early termination for convergence.
        """
        # Handle special cases
        if abs(z) < 1e-15:  # Near zero case
            return 1.0
        elif alpha == 1.0 and beta == 1.0:
            return np.exp(z)
        elif alpha == 2.0 and beta == 1.0:
            return np.cos(np.sqrt(-z))
        elif alpha == 2.0 and beta == 2.0:
            if z == 0:
                return 1.0
            else:
                return np.sin(np.sqrt(z)) / np.sqrt(z)

        # Series expansion
        result = 0.0
        term = 1.0
        k = 0

        while k < 100 and abs(term) > 1e-15:
            result += term
            k += 1
            if k > 0:
                # Compute next term: z^k / Γ(αk + β)
                denominator = alpha * k + beta - alpha
                if abs(denominator) < 1e-15:  # Avoid division by zero
                    break
                term = term * z / denominator

        return result

    @staticmethod
    def _ml_numpy_array(z: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        NumPy implementation for array inputs (fallback method).
        
        Handles array inputs by applying the Mittag-Leffler function element-wise.
        """
        result = np.zeros_like(z)
        
        for i, zi in enumerate(z.flat):
            # Handle special cases for each element
            if abs(zi) < 1e-15:  # Near zero case
                result.flat[i] = 1.0
            elif alpha == 1.0 and beta == 1.0:
                result.flat[i] = np.exp(zi)
            elif alpha == 2.0 and beta == 1.0:
                result.flat[i] = np.cos(np.sqrt(zi))
            elif alpha == 2.0 and beta == 2.0:
                if abs(zi) < 1e-15:
                    result.flat[i] = 1.0
                else:
                    result.flat[i] = np.sin(np.sqrt(zi)) / np.sqrt(zi)
            else:
                # Series expansion for general case
                term = 1.0
                k = 0
                series_sum = 0.0
                
                while k < 100 and abs(term) > 1e-15:
                    series_sum += term
                    k += 1
                    if k > 0:
                        denominator = alpha * k + beta - alpha
                        if abs(denominator) < 1e-15:  # Avoid division by zero
                            break
                        term = term * zi / denominator
                
                result.flat[i] = series_sum
        
        return result

    @staticmethod
    def _ml_jax_impl(z: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
        """
        JAX implementation of Mittag-Leffler function.

        Uses series expansion with vectorized operations.
        """

        # Handle special cases
        def ml_general(z):
            # Series expansion for general case
            k = jnp.arange(100)
            terms = z[..., None] ** k / \
                jax.scipy.special.gamma(alpha * k + beta)
            return jnp.sum(terms, axis=-1)

        # Use jax.lax.cond for conditional computation
        return jax.lax.cond(
            jnp.logical_and(alpha == 1.0, beta == 1.0),
            lambda z: jnp.exp(z),
            lambda z: jax.lax.cond(
                jnp.logical_and(alpha == 2.0, beta == 1.0),
                lambda z: jnp.cos(jnp.sqrt(-z)),
                lambda z: jax.lax.cond(
                    jnp.logical_and(alpha == 2.0, beta == 2.0),
                    lambda z: jnp.where(
                        z == 0, 1.0, jnp.sin(jnp.sqrt(z)) / jnp.sqrt(z)
                    ),
                    lambda z: ml_general(z),
                    z,
                ),
                z,
            ),
            z,
        )

    def compute_derivative(
        self,
        z: Union[float, np.ndarray, jnp.ndarray],
        alpha: float = 1.0,
        beta: float = 1.0,
        order: int = 1,
    ) -> Union[float, np.ndarray, jnp.ndarray]:
        """
        Compute the derivative of the Mittag-Leffler function.

        The derivative is given by:
        d/dz E_α,β(z) = E_α,α+β(z) / α

        Args:
            z: Input value(s)
            alpha: First parameter
            beta: Second parameter
            order: Order of derivative (default: 1)

        Returns:
            Derivative value(s)
        """
        if order == 0:
            return self.compute(z, alpha, beta)
        elif order == 1:
            return self.compute(z, alpha, alpha + beta) / alpha
        else:
            # Higher order derivatives can be computed recursively
            result = z
            for _ in range(order):
                result = self.compute(result, alpha, alpha + beta) / alpha
            return result


class MittagLefflerMatrix:
    """
    Matrix version of the Mittag-Leffler function.

    Computes E_α,β(A) where A is a matrix.
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True):
        """
        Initialize matrix Mittag-Leffler function calculator.

        Args:
            use_jax: Whether to use JAX implementation
            use_numba: Whether to use NUMBA implementation
        """
        self.use_jax = use_jax
        self.use_numba = use_numba
        self.ml = MittagLefflerFunction(use_jax=use_jax, use_numba=use_numba)

    def compute(
        self,
        A: Union[np.ndarray, jnp.ndarray],
        alpha: float = 1.0,
        beta: float = 1.0,
        max_terms: int = 50,
    ) -> Union[np.ndarray, jnp.ndarray]:
        """
        Compute the matrix Mittag-Leffler function E_α,β(A).

        Args:
            A: Input matrix
            alpha: First parameter
            beta: Second parameter
            max_terms: Maximum number of terms in the series

        Returns:
            Matrix Mittag-Leffler function value
        """
        if self.use_jax and isinstance(A, jnp.ndarray):
            return self._ml_matrix_jax(A, alpha, beta, max_terms)
        else:
            return self._ml_matrix_numpy(A, alpha, beta, max_terms)

    def _ml_matrix_numpy(
        self, A: np.ndarray, alpha: float, beta: float, max_terms: int
    ) -> np.ndarray:
        """NumPy implementation of matrix Mittag-Leffler function."""
        A.shape[0]
        result = np.zeros_like(A)

        for k in range(max_terms):
            term = A**k / gamma(alpha * k + beta, use_numba=self.use_numba)
            result += term

            # Check for convergence
            if np.linalg.norm(term) < 1e-15:
                break

        return result

    def _ml_matrix_jax(
        self, A: jnp.ndarray, alpha: float, beta: float, max_terms: int
    ) -> jnp.ndarray:
        """JAX implementation of matrix Mittag-Leffler function."""

        def body_fun(k, result):
            term = jnp.linalg.matrix_power(A, k) / jax.scipy.special.gamma(
                alpha * k + beta
            )
            return result + term

        result = jnp.zeros_like(A)
        result = jax.lax.fori_loop(0, max_terms, body_fun, result)
        return result


# Note: NUMBA vectorization removed for compatibility
# Use the class methods for optimized computations instead


# Convenience functions
def mittag_leffler(
    z: Union[float, np.ndarray, jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, jnp.ndarray]:
    """
    Convenience function to compute Mittag-Leffler function.

    Args:
        z: Input value(s)
        alpha: First parameter
        beta: Second parameter
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Mittag-Leffler function value(s)
    """
    ml_func = MittagLefflerFunction(use_jax=use_jax, use_numba=use_numba)
    return ml_func.compute(z, alpha, beta)


def mittag_leffler_derivative(
    z: Union[float, np.ndarray, jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    order: int = 1,
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, jnp.ndarray]:
    """
    Convenience function to compute derivative of Mittag-Leffler function.

    Args:
        z: Input value(s)
        alpha: First parameter
        beta: Second parameter
        order: Order of derivative
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Derivative value(s)
    """
    ml_func = MittagLefflerFunction(use_jax=use_jax, use_numba=use_numba)
    return ml_func.compute_derivative(z, alpha, beta, order)


def mittag_leffler_matrix(
    A: Union[np.ndarray, jnp.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Convenience function to compute matrix Mittag-Leffler function.

    Args:
        A: Input matrix
        alpha: First parameter
        beta: Second parameter
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Matrix Mittag-Leffler function value
    """
    ml_matrix = MittagLefflerMatrix(use_jax=use_jax, use_numba=use_numba)
    return ml_matrix.compute(A, alpha, beta)


# Special cases for common values
def exponential(
    z: Union[float, np.ndarray, jnp.ndarray],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, jnp.ndarray]:
    """E_1,1(z) = e^z"""
    return mittag_leffler(
        z,
        alpha=1.0,
        beta=1.0,
        use_jax=use_jax,
        use_numba=use_numba)


def cosine_fractional(
    z: Union[float, np.ndarray, jnp.ndarray],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, jnp.ndarray]:
    """E_2,1(-z^2) = cos(z)"""
    return mittag_leffler(
        -(z**2), alpha=2.0, beta=1.0, use_jax=use_jax, use_numba=use_numba
    )


def sinc_fractional(
    z: Union[float, np.ndarray, jnp.ndarray],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, jnp.ndarray]:
    """E_2,2(-z^2) = sin(z)/z"""
    return mittag_leffler(
        -(z**2), alpha=2.0, beta=2.0, use_jax=use_jax, use_numba=use_numba
    )
