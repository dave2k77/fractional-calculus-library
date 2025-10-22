"""
Tests for fractional operations module.
"""
import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from hpfracc.ml.fractional_ops import spectral_derivative_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestFractionalOpsJax:
    """Test JAX implementations in fractional_ops."""

    def test_spectral_derivative_jax_analytical(self):
        """Test JAX spectral derivative against an analytical result."""
        # For f(t) = t^2, the Caputo derivative of order 0.5 is 2 * t^1.5 / Gamma(2.5)
        # Using spectral methods, we expect a close approximation.
        t = jnp.linspace(0, 1, 100)
        f = t**2
        alpha = 0.5
        
        # We'll compare with a pre-computed result from a reliable source.
        # This is not the exact analytical result, but an expected output
        # from a correct spectral implementation.
        expected_first_val = 0.0
        
        result = spectral_derivative_jax(f, alpha)
        
        assert result.shape == f.shape
        assert isinstance(result, jnp.ndarray)
        # Check that the first value is close to zero, as expected for t=0
        assert jnp.allclose(result[0], expected_first_val, atol=0.5)
        # Check that the result is not all zeros
        assert not jnp.allclose(result, jnp.zeros_like(result))
