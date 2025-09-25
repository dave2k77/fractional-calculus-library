"""
Gamma and Beta functions for fractional calculus.

This module provides optimized implementations of the Gamma and Beta functions,
which are fundamental special functions used throughout fractional calculus.
"""

import numpy as np
from typing import Union
import scipy.special as scipy_special
# Simple module-level convenience wrappers expected by tests
def gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.gamma(x)


def beta_function(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.beta(a, b)


def log_gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.gammaln(x)


def digamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return scipy_special.digamma(x)

# Optional numba import
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Optional JAX import
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


# Module-level gamma function for Numba compatibility
@jit(nopython=True)
def _gamma_numba_scalar(z: float) -> float:
    """
    NUMBA-optimized Gamma function for scalar inputs.

    Uses Lanczos approximation for accuracy and performance.
    """
    # Lanczos approximation coefficients
    g = 7.0
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * _gamma_numba_scalar(1 - z))
    
    z -= 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    
    t = z + g + 0.5
    return np.sqrt(2 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x


class GammaFunction:
    """
    Gamma function implementation with multiple optimization strategies.

    The Gamma function is defined as:
    Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt

    For positive integers n: Γ(n) = (n-1)!
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True, cache_size: int = 1000):
        """
        Initialize Gamma function calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            cache_size: Size of the cache for frequently used values
        """
        self.use_jax = use_jax
        self.use_numba = use_numba
        self.cache_size = cache_size
        self._cache = {}

        if use_jax:
            self._gamma_jax = jax.jit(self._gamma_jax_impl)

    def compute(
        self, z: Union[float, np.ndarray, "jnp.ndarray"]
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the Gamma function.

        Args:
            z: Input value(s), can be scalar or array

        Returns:
            Gamma function value(s)
        """
        if self.use_jax and JAX_AVAILABLE and isinstance(z, (jnp.ndarray, float, int)):
            return self._gamma_jax(z)
        elif self.use_numba and isinstance(z, (float, int)):
            # Temporarily disable Numba due to compilation issues
            return self._gamma_scipy(z)
        else:
            return self._gamma_scipy(z)

    @staticmethod
    def _gamma_scipy(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        return scipy_special.gamma(z)


    @staticmethod
    def _gamma_jax_impl(z: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of Gamma function.

        Uses JAX's built-in gamma function for vectorized operations.
        """
        return jax.scipy.special.gamma(z)

    def log_gamma(
        self, z: Union[float, np.ndarray, "jnp.ndarray"]
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the natural logarithm of the Gamma function.

        Args:
            z: Input value(s)

        Returns:
            Log Gamma function value(s)
        """
        if self.use_jax and JAX_AVAILABLE and isinstance(z, (jnp.ndarray, float)):
            return jax.scipy.special.gammaln(z)
        else:
            return scipy_special.gammaln(z)


class BetaFunction:
    """
    Beta function implementation with multiple optimization strategies.

    The Beta function is defined as:
    B(x, y) = ∫₀¹ t^(x-1) (1-t)^(y-1) dt = Γ(x)Γ(y)/Γ(x+y)
    """

    def __init__(self, use_jax: bool = False, use_numba: bool = True, cache_size: int = 1000):
        """
        Initialize Beta function calculator.

        Args:
            use_jax: Whether to use JAX implementation for vectorized operations
            use_numba: Whether to use NUMBA JIT compilation for scalar operations
            cache_size: Size of the cache for frequently used values
        """
        self.use_jax = use_jax
        self.use_numba = use_numba
        self.cache_size = cache_size
        self._cache = {}
        self.gamma = GammaFunction(use_jax=use_jax, use_numba=use_numba, cache_size=cache_size)

        if use_jax:
            self._beta_jax = jax.jit(self._beta_jax_impl)

    def compute(
        self,
        x: Union[float, np.ndarray, "jnp.ndarray"],
        y: Union[float, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the Beta function.

        Args:
            x: First parameter
            y: Second parameter

        Returns:
            Beta function value(s)
        """
        if (
            self.use_jax
            and JAX_AVAILABLE
            and isinstance(x, (jnp.ndarray, float, int))
            and isinstance(y, (jnp.ndarray, float, int))
        ):
            return self._beta_jax(x, y)
        elif (
            self.use_numba
            and isinstance(x, (float, int))
            and isinstance(y, (float, int))
        ):
            # For very large inputs, use SciPy to avoid numerical underflow
            if x > 50 or y > 50 or (x + y) > 100:
                return self._beta_scipy(x, y)
            return self._beta_numba_scalar(x, y)
        else:
            return self._beta_scipy(x, y)

    @staticmethod
    def _beta_scipy(
        x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """SciPy implementation for reference and fallback."""
        return scipy_special.beta(x, y)

    @staticmethod
    @jit(nopython=True)
    def _beta_numba_scalar(x: float, y: float) -> float:
        """
        NUMBA-optimized Beta function for scalar inputs.

        Uses the relationship B(x,y) = Γ(x)Γ(y)/Γ(x+y)
        """
        gamma_x = _gamma_numba_scalar(x)
        gamma_y = _gamma_numba_scalar(y)
        gamma_sum = _gamma_numba_scalar(x + y)
        return gamma_x * gamma_y / gamma_sum

    @staticmethod
    def _beta_jax_impl(x: "jnp.ndarray", y: "jnp.ndarray") -> "jnp.ndarray":
        """
        JAX implementation of Beta function.

        Uses JAX's built-in beta function for vectorized operations.
        """
        return jax.scipy.special.beta(x, y)

    def log_beta(
        self,
        x: Union[float, np.ndarray, "jnp.ndarray"],
        y: Union[float, np.ndarray, "jnp.ndarray"],
    ) -> Union[float, np.ndarray, "jnp.ndarray"]:
        """
        Compute the natural logarithm of the Beta function.

        Args:
            x: First parameter
            y: Second parameter

        Returns:
            Log Beta function value(s)
        """
        if (
            self.use_jax
            and JAX_AVAILABLE
            and isinstance(x, (jnp.ndarray, float))
            and isinstance(y, (jnp.ndarray, float))
        ):
            return jax.scipy.special.betaln(x, y)
        else:
            return scipy_special.betaln(x, y)


# Note: NUMBA vectorization removed for compatibility
# Use the class methods for optimized computations instead


# Convenience functions
def gamma(
    z: Union[float, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute Gamma function.

    Args:
        z: Input value(s)
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Gamma function value(s)
    """
    gamma_func = GammaFunction(use_jax=use_jax, use_numba=use_numba)
    return gamma_func.compute(z)


def beta(
    x: Union[float, np.ndarray, "jnp.ndarray"],
    y: Union[float, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
    use_numba: bool = True,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute Beta function.

    Args:
        x: First parameter
        y: Second parameter
        use_jax: Whether to use JAX implementation
        use_numba: Whether to use NUMBA implementation

    Returns:
        Beta function value(s)
    """
    beta_func = BetaFunction(use_jax=use_jax, use_numba=use_numba)
    return beta_func.compute(x, y)


def log_gamma(
    z: Union[float, np.ndarray, "jnp.ndarray"], use_jax: bool = False
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute log Gamma function.

    Args:
        z: Input value(s)
        use_jax: Whether to use JAX implementation

    Returns:
        Log Gamma function value(s)
    """
    gamma_func = GammaFunction(use_jax=use_jax, use_numba=False)
    return gamma_func.log_gamma(z)


def log_beta(
    x: Union[float, np.ndarray, "jnp.ndarray"],
    y: Union[float, np.ndarray, "jnp.ndarray"],
    use_jax: bool = False,
) -> Union[float, np.ndarray, "jnp.ndarray"]:
    """
    Convenience function to compute log Beta function.

    Args:
        x: First parameter
        y: Second parameter
        use_jax: Whether to use JAX implementation

    Returns:
        Log Beta function value(s)
    """
    beta_func = BetaFunction(use_jax=use_jax, use_numba=False)
    return beta_func.log_beta(x, y)
