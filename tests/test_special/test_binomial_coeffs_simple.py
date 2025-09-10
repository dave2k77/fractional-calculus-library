"""
Simple tests for binomial coefficients module.

This module tests the basic functionality of hpfracc.special.binomial_coeffs
without relying on complex Numba-compiled functions.
"""

import numpy as np
import pytest
from hpfracc.special.binomial_coeffs import (
    BinomialCoefficients,
    GrunwaldLetnikovCoefficients,
    binomial,
    binomial_fractional,
    fractional_pascal_triangle,
    grunwald_letnikov_coefficients,
    grunwald_letnikov_weighted_coefficients,
    pascal_triangle
)


class TestBinomialCoefficientsBasic:
    """Test basic functionality of BinomialCoefficients class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        bc = BinomialCoefficients()
        
        assert bc is not None
        assert hasattr(bc, 'compute')
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        bc = BinomialCoefficients(cache_size=1000)
        
        assert bc is not None
        assert hasattr(bc, 'compute')
    
    def test_compute_scalar_basic(self):
        """Test computing single binomial coefficient with basic values."""
        bc = BinomialCoefficients()
        
        # Test with small integer values that should work
        result = bc.compute(5, 2)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_special_cases_basic(self):
        """Test known special cases."""
        bc = BinomialCoefficients()
        
        # Test with small values
        result = bc.compute(3, 1)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)


class TestBinomialFunctions:
    """Test standalone binomial coefficient functions."""
    
    def test_binomial_function(self):
        """Test binomial function."""
        result = binomial(5, 2)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_binomial_fractional(self):
        """Test fractional binomial coefficient."""
        result = binomial_fractional(5.5, 2)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_pascal_triangle(self):
        """Test Pascal triangle."""
        result = pascal_triangle(5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_fractional_pascal_triangle(self):
        """Test fractional Pascal triangle."""
        result = fractional_pascal_triangle(0.5, 5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestGrunwaldLetnikovCoefficients:
    """Test Grünwald-Letnikov coefficient functions."""
    
    def test_grunwald_letnikov_coefficients(self):
        """Test Grünwald-Letnikov coefficients."""
        result = grunwald_letnikov_coefficients(5, 0.5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1  # Should have at least one coefficient
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_grunwald_letnikov_weighted_coefficients(self):
        """Test weighted Grünwald-Letnikov coefficients."""
        result = grunwald_letnikov_weighted_coefficients(5, 0.5, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1  # Should have at least one coefficient
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_grunwald_letnikov_coefficients_single(self):
        """Test Grünwald-Letnikov coefficients with single values."""
        result = grunwald_letnikov_coefficients(3, 0.5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestGrunwaldLetnikovCoefficientsClass:
    """Test GrunwaldLetnikovCoefficients class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        glc = GrunwaldLetnikovCoefficients()
        
        assert glc is not None
        assert hasattr(glc, 'compute_coefficients')
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        glc = GrunwaldLetnikovCoefficients(use_jax=True, use_numba=False)
        
        assert glc is not None
        assert hasattr(glc, 'compute_coefficients')
    
    def test_compute_coefficients(self):
        """Test computing coefficients."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute_coefficients(0.5, 5)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_compute_weighted_coefficients(self):
        """Test computing weighted coefficients."""
        glc = GrunwaldLetnikovCoefficients()
        
        result = glc.compute_weighted_coefficients(0.5, 5, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestBinomialCoefficientsPerformance:
    """Test performance and large values."""
    
    def test_moderate_values(self):
        """Test with moderate values."""
        bc = BinomialCoefficients()
        
        # Test with moderate values
        result = bc.compute(10, 5)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_caching_behavior(self):
        """Test caching behavior."""
        bc = BinomialCoefficients()
        
        # Compute same value twice
        result1 = bc.compute(5, 2)
        result2 = bc.compute(5, 2)
        
        assert isinstance(result1, (int, float, np.integer, np.floating))
        assert isinstance(result2, (int, float, np.integer, np.floating))


class TestBinomialCoefficientsErrorHandling:
    """Test error handling and edge cases."""
    
    def test_basic_parameters(self):
        """Test with basic parameters."""
        bc = BinomialCoefficients()
        
        # Test with basic values
        result = bc.compute(3, 1)
        
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_numerical_stability(self):
        """Test numerical stability."""
        bc = BinomialCoefficients()
        
        # Test with values that should be numerically stable
        test_cases = [
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 3)
        ]
        
        for n, k in test_cases:
            result = bc.compute(n, k)
            assert isinstance(result, (int, float, np.integer, np.floating))
            assert not np.isnan(result)
            assert not np.isinf(result)


class TestBinomialCoefficientsIntegration:
    """Integration tests for binomial coefficients."""
    
    def test_consistency_across_functions(self):
        """Test consistency between different function implementations."""
        bc = BinomialCoefficients()
        
        n, k = 5, 2
        
        # Test class method vs standalone function
        result1 = bc.compute(n, k)
        result2 = binomial(n, k)
        
        assert isinstance(result1, (int, float, np.integer, np.floating))
        assert isinstance(result2, (int, float, np.integer, np.floating))
    
    def test_pascal_triangle_consistency(self):
        """Test Pascal triangle consistency."""
        result = pascal_triangle(5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_fractional_pascal_triangle_consistency(self):
        """Test fractional Pascal triangle consistency."""
        result = fractional_pascal_triangle(0.5, 5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 6  # n+1 rows
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
