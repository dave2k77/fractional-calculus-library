"""
Tests for Fractional Mellin Transform

This module tests the FractionalMellinTransform class and its methods.
"""

import pytest
import numpy as np
from src.algorithms.special_methods import FractionalMellinTransform, fractional_mellin_transform
from src.core.definitions import FractionalOrder


class TestFractionalMellinTransform:
    """Test suite for FractionalMellinTransform."""
    
    def test_initialization(self):
        """Test FractionalMellinTransform initialization."""
        # Test with float
        fmt = FractionalMellinTransform(0.5)
        assert fmt.alpha_val == 0.5
        
        # Test with FractionalOrder
        fmt = FractionalMellinTransform(FractionalOrder(1.0))
        assert fmt.alpha_val == 1.0
        
        # Test with negative alpha (should warn but not fail)
        with pytest.warns(UserWarning):
            fmt = FractionalMellinTransform(-0.1)
    
    def test_numerical_method(self):
        """Test numerical integration method."""
        fmt = FractionalMellinTransform(0.5)
        
        # Test with exponential function
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = 1.0
        
        result = fmt.transform(f, x, s, method="numerical")
        assert isinstance(result, complex)
        assert not np.isnan(result)
        assert not np.isinf(result)
        
        # Test with array of s values
        s_array = np.array([0.5, 1.0, 1.5])
        result_array = fmt.transform(f, x, s_array, method="numerical")
        assert len(result_array) == 3
        assert all(isinstance(r, complex) for r in result_array)
    
    def test_analytical_method(self):
        """Test analytical method for special functions."""
        fmt = FractionalMellinTransform(0.5)
        
        # Test with exponential function
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = 1.0
        
        result = fmt.transform(f, x, s, method="analytical")
        assert isinstance(result, complex)
        assert not np.isnan(result)
        
        # Test with power function
        f_power = x ** 2
        result_power = fmt.transform(f_power, x, s, method="analytical")
        assert isinstance(result_power, complex)
    
    def test_fft_method(self):
        """Test FFT-based method."""
        fmt = FractionalMellinTransform(0.5)
        
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = 1.0
        
        result = fmt.transform(f, x, s, method="fft")
        assert isinstance(result, complex)
        assert not np.isnan(result)
    
    def test_inverse_transform(self):
        """Test inverse transform methods."""
        fmt = FractionalMellinTransform(0.5)
        
        # Test numerical inverse
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = np.array([0.5, 1.0, 1.5])
        
        # Forward transform
        F = fmt.transform(f, x, s, method="numerical")
        
        # Inverse transform
        f_reconstructed = fmt.inverse_transform(F, s, x, method="numerical")
        assert len(f_reconstructed) == len(x)
        assert not np.any(np.isnan(f_reconstructed))
        
        # Test FFT inverse
        f_reconstructed_fft = fmt.inverse_transform(F, s, x, method="fft")
        assert len(f_reconstructed_fft) == len(x)
        assert not np.any(np.isnan(f_reconstructed_fft))
    
    def test_callable_input(self):
        """Test with callable function input."""
        fmt = FractionalMellinTransform(0.5)
        
        def test_function(x):
            return np.exp(-x)
        
        x_max = 10.0
        s = 1.0
        
        result = fmt.transform(test_function, x_max, s, method="numerical")
        assert isinstance(result, complex)
        assert not np.isnan(result)
    
    def test_positive_domain_validation(self):
        """Test that negative domain points are rejected."""
        fmt = FractionalMellinTransform(0.5)
        
        x = np.array([-1, 0, 1, 2])  # Contains negative and zero values
        f = np.ones_like(x)
        s = 1.0
        
        with pytest.raises(ValueError, match="Domain points must be positive"):
            fmt.transform(f, x, s)
    
    def test_convenience_function(self):
        """Test the convenience function."""
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = 1.0
        alpha = 0.5
        
        result = fractional_mellin_transform(f, x, s, alpha, method="numerical")
        assert isinstance(result, complex)
        assert not np.isnan(result)
    
    def test_method_validation(self):
        """Test that invalid methods are rejected."""
        fmt = FractionalMellinTransform(0.5)
        
        x = np.logspace(-2, 2, 100)
        f = np.exp(-x)
        s = 1.0
        
        with pytest.raises(ValueError, match="Unknown method"):
            fmt.transform(f, x, s, method="invalid_method")
        
        with pytest.raises(ValueError, match="Unknown method"):
            fmt.inverse_transform(f, s, x, method="invalid_method")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        fmt = FractionalMellinTransform(0.0)  # No fractional component
        
        # Single point
        x = np.array([1.0])
        f = np.array([1.0])
        s = 1.0
        
        result = fmt.transform(f, x, s, method="numerical")
        assert isinstance(result, complex)
        
        # Very small x values
        x = np.logspace(-10, -5, 100)
        f = np.exp(-x)
        s = 1.0
        
        result = fmt.transform(f, x, s, method="numerical")
        assert isinstance(result, complex)
        assert not np.isnan(result)


if __name__ == "__main__":
    pytest.main([__file__])
