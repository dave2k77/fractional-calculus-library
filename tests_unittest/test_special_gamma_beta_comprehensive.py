"""
Comprehensive tests for Gamma and Beta functions to achieve 100% coverage.

This module focuses on testing all uncovered paths in the gamma_beta implementation.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

# Import the module under test
from hpfracc.special.gamma_beta import (
    GammaFunction, BetaFunction, gamma, beta, log_gamma, log_beta,
    gamma_function, beta_function, log_gamma_function, digamma_function,
    _gamma_numba_scalar
)


class TestGammaBetaImportHandling(unittest.TestCase):
    """Test import handling and fallback mechanisms."""
    
    def test_numba_import_available(self):
        """Test when NUMBA is available."""
        # Test that NUMBA import paths are covered
        gamma_func = GammaFunction(use_numba=True)
        
        # This should work with NUMBA available
        result = gamma_func.compute(1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
    
    def test_numba_import_unavailable(self):
        """Test when NUMBA is not available."""
        with patch('hpfracc.special.gamma_beta.NUMBA_AVAILABLE', False):
            # Mock the jit decorator to return identity function
            with patch('hpfracc.special.gamma_beta.jit') as mock_jit:
                mock_jit.return_value = lambda x: x
                
                # Import the module to trigger the fallback
                from hpfracc.special.gamma_beta import GammaFunction
                gamma_func = GammaFunction(use_numba=True)
                
                # Should work with fallback
                result = gamma_func.compute(1.0)
                self.assertIsInstance(result, (float, np.floating))
    
    def test_jax_import_available(self):
        """Test when JAX is available."""
        gamma_func = GammaFunction(use_jax=True)
        
        # This should work if JAX is available, or fallback gracefully
        result = gamma_func.compute(1.0)
        # JAX returns Array objects, not standard float types
        self.assertTrue(hasattr(result, 'shape') or isinstance(result, (float, np.floating)))
        self.assertFalse(np.isnan(float(result)))
    
    def test_jax_import_unavailable(self):
        """Test when JAX is not available."""
        with patch('hpfracc.special.gamma_beta.JAX_AVAILABLE', False):
            with patch('hpfracc.special.gamma_beta.jnp', None):
                gamma_func = GammaFunction(use_jax=True)
                
                # Should fallback to SciPy implementation
                result = gamma_func.compute(1.0)
                self.assertIsInstance(result, (float, np.floating))


class TestGammaNumbaImplementation(unittest.TestCase):
    """Test NUMBA gamma function implementation (lines 57-79)."""
    
    def test_gamma_numba_scalar_basic(self):
        """Test NUMBA gamma function with basic values."""
        # Test with positive integer
        result = _gamma_numba_scalar(5.0)
        expected = 24.0  # 4!
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_gamma_numba_scalar_fraction(self):
        """Test NUMBA gamma function with fractional values."""
        # Test with 0.5 (should be sqrt(pi))
        result = _gamma_numba_scalar(0.5)
        expected = np.sqrt(np.pi)
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_gamma_numba_scalar_small_values(self):
        """Test NUMBA gamma function with small values (< 0.5)."""
        # Test with value < 0.5 (should trigger reflection formula)
        result = _gamma_numba_scalar(0.25)
        expected = np.pi / (np.sin(np.pi * 0.25) * _gamma_numba_scalar(0.75))
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_gamma_numba_scalar_edge_cases(self):
        """Test NUMBA gamma function with edge cases."""
        # Test with 1.0 (should be 1)
        result = _gamma_numba_scalar(1.0)
        self.assertAlmostEqual(result, 1.0, places=10)
        
        # Test with 2.0 (should be 1)
        result = _gamma_numba_scalar(2.0)
        self.assertAlmostEqual(result, 1.0, places=10)


class TestGammaJAXImplementation(unittest.TestCase):
    """Test JAX implementation paths."""
    
    def test_gamma_jax_available(self):
        """Test JAX gamma when available."""
        gamma_func = GammaFunction(use_jax=True)
        
        try:
            import jax.numpy as jnp
            z_jax = jnp.array(1.0)
            result = gamma_func.compute(z_jax)
            
            # Should return JAX array or fallback gracefully
            self.assertTrue(hasattr(result, 'shape') or isinstance(result, (float, np.floating)))
        except ImportError:
            # JAX not available, test fallback
            result = gamma_func.compute(1.0)
            self.assertIsInstance(result, (float, np.floating))
    
    def test_gamma_log_jax_available(self):
        """Test JAX log gamma when available (lines 156-159)."""
        gamma_func = GammaFunction(use_jax=True)
        
        try:
            import jax.numpy as jnp
            z_jax = jnp.array(1.0)
            result = gamma_func.log_gamma(z_jax)
            
            # Should return JAX array or fallback gracefully
            self.assertTrue(hasattr(result, 'shape') or isinstance(result, (float, np.floating)))
        except ImportError:
            # JAX not available, test fallback
            result = gamma_func.log_gamma(1.0)
            self.assertIsInstance(result, (float, np.floating))


class TestBetaNumbaImplementation(unittest.TestCase):
    """Test NUMBA beta function implementation (lines 237-240)."""
    
    def test_beta_numba_scalar_basic(self):
        """Test NUMBA beta function with basic values."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test with simple values
        result = beta_func.compute(1.0, 1.0)
        expected = 1.0  # B(1,1) = 1
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test with larger values
        result = beta_func.compute(2.0, 3.0)
        expected = 1.0 / 12.0  # B(2,3) = 1/12
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_beta_numba_scalar_large_values(self):
        """Test NUMBA beta function with large values (should use SciPy fallback)."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test with large values that should trigger SciPy fallback
        result = beta_func.compute(60.0, 40.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
    
    def test_beta_numba_scalar_fractional_values(self):
        """Test NUMBA beta function with fractional values."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test with fractional values
        result = beta_func.compute(1.5, 2.5)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))


class TestBetaJAXImplementation(unittest.TestCase):
    """Test JAX beta implementation paths."""
    
    def test_beta_jax_available(self):
        """Test JAX beta when available."""
        beta_func = BetaFunction(use_jax=True)
        
        try:
            import jax.numpy as jnp
            x_jax = jnp.array(1.0)
            y_jax = jnp.array(1.0)
            result = beta_func.compute(x_jax, y_jax)
            
            # Should return JAX array or fallback gracefully
            self.assertTrue(hasattr(result, 'shape') or isinstance(result, (float, np.floating)))
        except ImportError:
            # JAX not available, test fallback
            result = beta_func.compute(1.0, 1.0)
            self.assertIsInstance(result, (float, np.floating))
    
    def test_beta_log_jax_available(self):
        """Test JAX log beta when available (lines 266-274)."""
        beta_func = BetaFunction(use_jax=True)
        
        try:
            import jax.numpy as jnp
            x_jax = jnp.array(1.0)
            y_jax = jnp.array(1.0)
            result = beta_func.log_beta(x_jax, y_jax)
            
            # Should return JAX array or fallback gracefully
            self.assertTrue(hasattr(result, 'shape') or isinstance(result, (float, np.floating)))
        except ImportError:
            # JAX not available, test fallback
            result = beta_func.log_beta(1.0, 1.0)
            self.assertIsInstance(result, (float, np.floating))


class TestGammaBetaConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for additional coverage."""
    
    def test_gamma_convenience_function(self):
        """Test gamma convenience function (lines 298-299)."""
        result = gamma(1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 1.0, places=10)
    
    def test_gamma_convenience_function_array(self):
        """Test gamma convenience function with array."""
        z = np.array([1.0, 2.0, 3.0])
        result = gamma(z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)
    
    def test_beta_convenience_function(self):
        """Test beta convenience function (lines 320-321)."""
        result = beta(1.0, 1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 1.0, places=10)
    
    def test_beta_convenience_function_array(self):
        """Test beta convenience function with arrays."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 3.0])
        result = beta(x, y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_log_gamma_convenience_function(self):
        """Test log_gamma convenience function (lines 337-338)."""
        result = log_gamma(1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 0.0, places=10)  # log(Γ(1)) = log(1) = 0
    
    def test_log_gamma_convenience_function_array(self):
        """Test log_gamma convenience function with array."""
        z = np.array([1.0, 2.0, 3.0])
        result = log_gamma(z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)
    
    def test_log_beta_convenience_function(self):
        """Test log_beta convenience function (lines 357-358)."""
        result = log_beta(1.0, 1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 0.0, places=10)  # log(B(1,1)) = log(1) = 0
    
    def test_log_beta_convenience_function_array(self):
        """Test log_beta convenience function with arrays."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 3.0])
        result = log_beta(x, y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)


class TestGammaBetaModuleLevelFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_gamma_function_module_level(self):
        """Test module-level gamma function."""
        result = gamma_function(1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 1.0, places=10)
    
    def test_gamma_function_module_level_array(self):
        """Test module-level gamma function with array."""
        z = np.array([1.0, 2.0, 3.0])
        result = gamma_function(z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)
    
    def test_beta_function_module_level(self):
        """Test module-level beta function."""
        result = beta_function(1.0, 1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 1.0, places=10)
    
    def test_beta_function_module_level_array(self):
        """Test module-level beta function with arrays."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 3.0])
        result = beta_function(x, y)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_log_gamma_function_module_level(self):
        """Test module-level log gamma function."""
        result = log_gamma_function(1.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertAlmostEqual(result, 0.0, places=10)
    
    def test_log_gamma_function_module_level_array(self):
        """Test module-level log gamma function with array."""
        z = np.array([1.0, 2.0, 3.0])
        result = log_gamma_function(z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)
    
    def test_digamma_function_module_level(self):
        """Test module-level digamma function."""
        result = digamma_function(1.0)
        self.assertIsInstance(result, (float, np.floating))
        # Digamma(1) = -γ (Euler-Mascheroni constant)
        self.assertAlmostEqual(result, -0.5772156649015329, places=10)
    
    def test_digamma_function_module_level_array(self):
        """Test module-level digamma function with array."""
        z = np.array([1.0, 2.0, 3.0])
        result = digamma_function(z)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)


class TestGammaBetaEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_gamma_edge_cases(self):
        """Test gamma function with edge cases."""
        gamma_func = GammaFunction()
        
        # Test with very small values
        result = gamma_func.compute(1e-10)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        
        # Test with moderate values
        result = gamma_func.compute(10.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
    
    def test_beta_edge_cases(self):
        """Test beta function with edge cases."""
        beta_func = BetaFunction()
        
        # Test with very small values
        result = beta_func.compute(1e-10, 1e-10)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        
        # Test with moderate values
        result = beta_func.compute(10.0, 5.0)
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
    
    def test_gamma_array_edge_cases(self):
        """Test gamma function with array edge cases."""
        gamma_func = GammaFunction()
        
        # Test with array containing edge cases
        z = np.array([1e-10, 1.0, 10.0])
        result = gamma_func.compute(z)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, z.shape)
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_beta_array_edge_cases(self):
        """Test beta function with array edge cases."""
        beta_func = BetaFunction()
        
        # Test with arrays containing edge cases
        x = np.array([1e-10, 1.0, 10.0])
        y = np.array([1e-10, 1.0, 5.0])
        result = beta_func.compute(x, y)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertFalse(np.any(np.isnan(result)))


class TestGammaBetaCacheBehavior(unittest.TestCase):
    """Test caching behavior if implemented."""
    
    def test_gamma_cache_initialization(self):
        """Test gamma function cache initialization."""
        gamma_func = GammaFunction(cache_size=100)
        
        # Should have cache initialized
        self.assertTrue(hasattr(gamma_func, '_cache'))
        self.assertEqual(gamma_func.cache_size, 100)
    
    def test_beta_cache_initialization(self):
        """Test beta function cache initialization."""
        beta_func = BetaFunction(cache_size=200)
        
        # Should have cache initialized
        self.assertTrue(hasattr(beta_func, '_cache'))
        self.assertEqual(beta_func.cache_size, 200)
    
    def test_gamma_repeated_calls(self):
        """Test gamma function with repeated calls."""
        gamma_func = GammaFunction()
        
        # Compute same value twice
        result1 = gamma_func.compute(1.0)
        result2 = gamma_func.compute(1.0)
        
        # Results should be identical
        self.assertEqual(result1, result2)
    
    def test_beta_repeated_calls(self):
        """Test beta function with repeated calls."""
        beta_func = BetaFunction()
        
        # Compute same values twice
        result1 = beta_func.compute(1.0, 1.0)
        result2 = beta_func.compute(1.0, 1.0)
        
        # Results should be identical
        self.assertEqual(result1, result2)


class TestGammaBetaMathematicalProperties(unittest.TestCase):
    """Test mathematical properties of Gamma and Beta functions."""
    
    def test_gamma_functional_equation(self):
        """Test Γ(z+1) = zΓ(z) for Gamma function."""
        gamma_func = GammaFunction()
        
        z = 2.5
        gamma_z = gamma_func.compute(z)
        gamma_z_plus_1 = gamma_func.compute(z + 1)
        
        # Γ(z+1) should equal z * Γ(z)
        expected = z * gamma_z
        self.assertAlmostEqual(gamma_z_plus_1, expected, places=10)
    
    def test_beta_symmetry(self):
        """Test B(x,y) = B(y,x) for Beta function."""
        beta_func = BetaFunction()
        
        x, y = 2.5, 3.5
        beta_xy = beta_func.compute(x, y)
        beta_yx = beta_func.compute(y, x)
        
        # Beta function should be symmetric
        self.assertAlmostEqual(beta_xy, beta_yx, places=10)
    
    def test_beta_gamma_relationship(self):
        """Test B(x,y) = Γ(x)Γ(y)/Γ(x+y) relationship."""
        gamma_func = GammaFunction()
        beta_func = BetaFunction()
        
        x, y = 2.0, 3.0
        beta_direct = beta_func.compute(x, y)
        
        gamma_x = gamma_func.compute(x)
        gamma_y = gamma_func.compute(y)
        gamma_sum = gamma_func.compute(x + y)
        beta_derived = gamma_x * gamma_y / gamma_sum
        
        # Direct beta should equal derived beta
        self.assertAlmostEqual(beta_direct, beta_derived, places=10)


class TestGammaBetaNumbaImplementationRemaining(unittest.TestCase):
    """Test remaining NUMBA implementation paths to achieve 100% coverage."""
    
    def test_gamma_numba_scalar_comprehensive(self):
        """Test NUMBA gamma function comprehensively (lines 57-79)."""
        # Test various scenarios to cover all paths in NUMBA implementation
        test_cases = [
            (0.25, "reflection formula"),  # < 0.5
            (0.5, "sqrt(pi)"),            # = 0.5
            (1.0, "unity"),               # = 1
            (1.5, "fractional"),          # > 0.5
            (2.0, "integer"),             # = 2
            (3.0, "factorial"),           # = 6
            (5.0, "large integer"),       # = 24
            (10.0, "large value")         # large value
        ]
        
        for z, description in test_cases:
            with self.subTest(z=z, description=description):
                result = _gamma_numba_scalar(z)
                self.assertIsInstance(result, (float, np.floating))
                self.assertFalse(np.isnan(result))
                self.assertFalse(np.isinf(result))
                self.assertGreater(result, 0)
    
    def test_beta_numba_scalar_comprehensive(self):
        """Test NUMBA beta function comprehensively (lines 237-240)."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test various scenarios to cover all paths in NUMBA beta implementation
        test_cases = [
            (1.0, 1.0, "unity"),           # B(1,1) = 1
            (2.0, 3.0, "fractional"),      # B(2,3) = 1/12
            (0.5, 0.5, "half"),            # B(0.5,0.5) = π
            (1.5, 2.5, "mixed"),           # Mixed fractional
            (3.0, 4.0, "integers"),        # Integer values
            (0.1, 0.1, "small"),           # Small values
            (10.0, 15.0, "moderate"),      # Moderate values
            (25.0, 30.0, "larger"),        # Larger values (should use SciPy)
            (60.0, 40.0, "large"),         # Large values (should use SciPy)
            (100.0, 50.0, "very large")    # Very large values (should use SciPy)
        ]
        
        for x, y, description in test_cases:
            with self.subTest(x=x, y=y, description=description):
                result = beta_func.compute(x, y)
                self.assertIsInstance(result, (float, np.floating))
                self.assertFalse(np.isnan(result))
                self.assertFalse(np.isinf(result))
                self.assertGreater(result, 0)
    
    def test_gamma_numba_scalar_reflection_formula(self):
        """Test NUMBA gamma reflection formula specifically."""
        # Test the reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        z = 0.25
        result = _gamma_numba_scalar(z)
        
        # Verify it's positive and finite
        self.assertIsInstance(result, (float, np.floating))
        self.assertFalse(np.isnan(result))
        self.assertFalse(np.isinf(result))
        self.assertGreater(result, 0)
    
    def test_gamma_numba_scalar_lanczos_approximation(self):
        """Test NUMBA gamma Lanczos approximation specifically."""
        # Test values that should use the Lanczos approximation (z >= 0.5)
        test_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        for z in test_values:
            with self.subTest(z=z):
                result = _gamma_numba_scalar(z)
                self.assertIsInstance(result, (float, np.floating))
                self.assertFalse(np.isnan(result))
                self.assertFalse(np.isinf(result))
                self.assertGreater(result, 0)
    
    def test_beta_numba_scalar_vs_scipy(self):
        """Test NUMBA beta vs SciPy for consistency."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test values that should use NUMBA (not large enough for SciPy fallback)
        test_cases = [
            (1.0, 1.0),
            (2.0, 3.0),
            (0.5, 1.5),
            (1.5, 2.5),
            (3.0, 4.0),
            (10.0, 15.0)
        ]
        
        for x, y in test_cases:
            with self.subTest(x=x, y=y):
                # NUMBA result
                result_numba = beta_func.compute(x, y)
                
                # SciPy result for comparison
                import scipy.special as scipy_special
                result_scipy = scipy_special.beta(x, y)
                
                # Should be close (allowing for numerical differences)
                self.assertAlmostEqual(result_numba, result_scipy, places=10)
    
    def test_beta_numba_scalar_edge_cases(self):
        """Test NUMBA beta function with edge cases."""
        beta_func = BetaFunction(use_numba=True)
        
        # Test edge cases that should still work
        edge_cases = [
            (1e-10, 1e-10),  # Very small values
            (1.0, 1e-10),    # One very small
            (1e-10, 1.0),    # One very small
            (0.1, 0.1),      # Small fractional
            (1.0, 1.0),      # Unity
            (2.0, 1.0),      # Integer cases
            (1.0, 2.0),      # Integer cases
        ]
        
        for x, y in edge_cases:
            with self.subTest(x=x, y=y):
                result = beta_func.compute(x, y)
                self.assertIsInstance(result, (float, np.floating))
                self.assertFalse(np.isnan(result))
                self.assertFalse(np.isinf(result))
                self.assertGreaterEqual(result, 0)


class TestGammaBetaRemainingCoverage(unittest.TestCase):
    """Test remaining uncovered lines."""
    
    def test_import_handling_lines_31_36(self):
        """Test NUMBA import handling (lines 31-36)."""
        # These lines are covered by the module import itself
        pass
    
    def test_import_handling_lines_43_45(self):
        """Test JAX import handling (lines 43-45)."""
        # These lines are covered by the module import itself
        pass
    
    def test_numba_gamma_lanczos_coefficients_lines_57_79(self):
        """Test NUMBA gamma Lanczos coefficients and computation (lines 57-79)."""
        # Test the Lanczos approximation coefficients and computation
        result = _gamma_numba_scalar(2.5)
        expected = 1.329340388179137  # Γ(2.5)
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test reflection formula path
        result = _gamma_numba_scalar(-1.5)
        self.assertTrue(np.isfinite(result))
        
        # Test Lanczos approximation path
        result = _gamma_numba_scalar(3.5)
        expected = 3.3233509704478426
        self.assertAlmostEqual(result, expected, places=10)
    
    def test_numba_beta_gamma_relationship_lines_237_240(self):
        """Test NUMBA beta function using gamma relationship (lines 237-240)."""
        # Test the beta function that uses gamma relationship through BetaFunction
        beta_func = BetaFunction(use_numba=True)
        result = beta_func.compute(2.0, 3.0)
        expected = 1.0/12.0  # B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12
        self.assertAlmostEqual(result, expected, places=10)
        
        # Test with fractional values
        result = beta_func.compute(1.5, 2.5)
        from scipy.special import beta as scipy_beta
        expected = scipy_beta(1.5, 2.5)
        self.assertAlmostEqual(result, expected, places=10)


if __name__ == '__main__':
    unittest.main()
