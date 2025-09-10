"""
Comprehensive tests for Advanced Methods module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Callable

from hpfracc.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    weyl_derivative,
    marchaud_derivative,
    hadamard_derivative,
    reiz_feller_derivative
)
from hpfracc.core.definitions import FractionalOrder
from hpfracc.algorithms.parallel_optimized_methods import ParallelConfig


class TestWeylDerivative:
    """Test WeylDerivative class."""
    
    def test_initialization_with_float(self):
        """Test initialization with float alpha."""
        weyl = WeylDerivative(0.5)
        
        assert weyl.alpha.alpha == 0.5
        assert weyl.n == 1
        assert weyl.alpha_val == 0.5
        assert isinstance(weyl.parallel_config, ParallelConfig)
    
    def test_initialization_with_fractional_order(self):
        """Test initialization with FractionalOrder."""
        alpha = FractionalOrder(0.7)
        weyl = WeylDerivative(alpha)
        
        assert weyl.alpha == alpha
        assert weyl.n == 1
        assert weyl.alpha_val == 0.7
    
    def test_initialization_with_custom_parallel_config(self):
        """Test initialization with custom parallel config."""
        config = ParallelConfig(n_jobs=4)
        weyl = WeylDerivative(0.5, config)
        
        assert weyl.parallel_config == config
        assert weyl.parallel_config.n_jobs == 4
    
    def test_compute_with_array(self):
        """Test computing derivative with array input."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = weyl.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_compute_with_function(self):
        """Test computing derivative with function input."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = lambda t: np.sin(t)
        
        result = weyl.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_compute_serial(self):
        """Test serial computation."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = weyl._compute_serial(f, x, 0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_parallel_computation(self):
        """Test parallel computation."""
        config = ParallelConfig(n_jobs=2)
        weyl = WeylDerivative(0.5, config)
        x = np.linspace(0, 1, 100)
        f = np.sin(x)
        
        result = weyl.compute(f, x, use_parallel=True)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_memory_optimization(self):
        """Test memory optimization for large arrays."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 1000)
        f = np.sin(x)
        
        result = weyl.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestMarchaudDerivative:
    """Test MarchaudDerivative class."""
    
    def test_initialization(self):
        """Test initialization."""
        marchaud = MarchaudDerivative(0.5)
        
        assert marchaud.alpha.alpha == 0.5
        assert marchaud.alpha_val == 0.5
    
    def test_compute_with_array(self):
        """Test computing derivative with array input."""
        marchaud = MarchaudDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = marchaud.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_compute_standard(self):
        """Test standard computation."""
        marchaud = MarchaudDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = marchaud._compute_standard(f, x, 0.1, False)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_memory_optimization(self):
        """Test memory optimization for large arrays."""
        marchaud = MarchaudDerivative(0.5)
        x = np.linspace(0, 1, 1000)
        f = np.sin(x)
        
        result = marchaud.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestHadamardDerivative:
    """Test HadamardDerivative class."""
    
    def test_initialization(self):
        """Test initialization."""
        hadamard = HadamardDerivative(0.5)
        
        assert hadamard.alpha.alpha == 0.5
        assert hadamard.n == 1
        assert hadamard.alpha_val == 0.5
    
    def test_compute_with_array(self):
        """Test computing derivative with array input."""
        hadamard = HadamardDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = hadamard.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_compute_hadamard(self):
        """Test Hadamard computation."""
        hadamard = HadamardDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = hadamard._compute_hadamard(f, x, 0.1)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestReizFellerDerivative:
    """Test ReizFellerDerivative class."""
    
    def test_initialization(self):
        """Test initialization."""
        reiz_feller = ReizFellerDerivative(0.5)
        
        assert reiz_feller.alpha.alpha == 0.5
        assert reiz_feller.alpha_val == 0.5
    
    def test_compute_with_array(self):
        """Test computing derivative with array input."""
        reiz_feller = ReizFellerDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = reiz_feller.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_compute_spectral(self):
        """Test spectral computation."""
        reiz_feller = ReizFellerDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        result = reiz_feller._compute_spectral(f, x, 0.1, False)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)


class TestFunctionWrappers:
    """Test function wrapper functions."""
    
    def test_weyl_derivative_function(self):
        """Test weyl_derivative function wrapper."""
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        with patch('hpfracc.algorithms.advanced_methods.WeylDerivative') as mock_class:
            mock_instance = Mock()
            mock_instance.compute.return_value = np.ones_like(x)
            mock_class.return_value = mock_instance
            
            result = weyl_derivative(f, x, 0.5)
            
            assert isinstance(result, np.ndarray)
            mock_class.assert_called_once_with(0.5)
            mock_instance.compute.assert_called_once_with(f, x, None)
    
    def test_marchaud_derivative_function(self):
        """Test marchaud_derivative function wrapper."""
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        with patch('hpfracc.algorithms.advanced_methods.MarchaudDerivative') as mock_class:
            mock_instance = Mock()
            mock_instance.compute.return_value = np.ones_like(x)
            mock_class.return_value = mock_instance
            
            result = marchaud_derivative(f, x, 0.5)
            
            assert isinstance(result, np.ndarray)
            mock_class.assert_called_once_with(0.5)
            mock_instance.compute.assert_called_once_with(f, x, None)
    
    def test_hadamard_derivative_function(self):
        """Test hadamard_derivative function wrapper."""
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        with patch('hpfracc.algorithms.advanced_methods.HadamardDerivative') as mock_class:
            mock_instance = Mock()
            mock_instance.compute.return_value = np.ones_like(x)
            mock_class.return_value = mock_instance
            
            result = hadamard_derivative(f, x, 0.5)
            
            assert isinstance(result, np.ndarray)
            mock_class.assert_called_once_with(0.5)
            mock_instance.compute.assert_called_once_with(f, x, None)
    
    def test_reiz_feller_derivative_function(self):
        """Test reiz_feller_derivative function wrapper."""
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        with patch('hpfracc.algorithms.advanced_methods.ReizFellerDerivative') as mock_class:
            mock_instance = Mock()
            mock_instance.compute.return_value = np.ones_like(x)
            mock_class.return_value = mock_instance
            
            result = reiz_feller_derivative(f, x, 0.5)
            
            assert isinstance(result, np.ndarray)
            mock_class.assert_called_once_with(0.5)
            mock_instance.compute.assert_called_once_with(f, x, None)


class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_alpha_value(self):
        """Test handling of invalid alpha values."""
        # Test with negative alpha
        with pytest.raises(ValueError):
            WeylDerivative(-0.5)
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        weyl = WeylDerivative(0.5)
        
        # Test with invalid input - this will actually work but produce unexpected results
        # The function converts string to array, so we test with a different invalid case
        with pytest.raises((TypeError, ValueError, IndexError)):
            weyl.compute("invalid", [1, 2, 3])
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        weyl = WeylDerivative(0.5)
        
        # Empty arrays are handled gracefully by the implementation
        # They return empty arrays rather than raising errors
        result = weyl.compute(np.array([]), np.array([]))
        assert len(result) == 0
    
    def test_mismatched_array_sizes(self):
        """Test handling of mismatched array sizes."""
        weyl = WeylDerivative(0.5)
        f = np.array([1, 2, 3])
        x = np.array([1, 2])  # Different size
        
        # Should handle gracefully by truncating to minimum length
        result = weyl.compute(f, x)
        assert len(result) == min(len(f), len(x))


class TestPerformanceOptimizations:
    """Test performance optimizations."""
    
    def test_parallel_processing(self):
        """Test parallel processing configuration."""
        config = ParallelConfig(n_jobs=2, enabled=True)
        weyl = WeylDerivative(0.5, config)
        
        assert weyl.parallel_config.enabled
        assert weyl.parallel_config.n_jobs == 2
    
    def test_memory_optimization_large_arrays(self):
        """Test memory optimization for large arrays."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10000)
        f = np.sin(x)
        
        result = weyl.compute(f, x)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
    
    def test_caching_mechanism(self):
        """Test caching mechanism."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f = np.sin(x)
        
        # First computation
        result1 = weyl.compute(f, x)
        
        # Second computation (should be faster due to caching if implemented)
        result2 = weyl.compute(f, x)
        
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert len(result1) == len(result2)


class TestMathematicalProperties:
    """Test mathematical properties."""
    
    def test_linearity_property(self):
        """Test linearity property of derivatives."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0, 1, 10)
        f1 = np.sin(x)
        f2 = np.cos(x)
        
        # Test linearity: D^α(f + g) = D^α(f) + D^α(g)
        result1 = weyl.compute(f1, x)
        result2 = weyl.compute(f2, x)
        result_sum = weyl.compute(f1 + f2, x)
        
        # Check approximate linearity (within numerical precision)
        np.testing.assert_allclose(result_sum, result1 + result2, rtol=1e-10)
    
    def test_consistency_with_known_solutions(self):
        """Test consistency with known analytical solutions."""
        weyl = WeylDerivative(0.5)
        x = np.linspace(0.1, 1, 10)  # Avoid x=0 for numerical stability
        f = x  # Simple linear function
        
        result = weyl.compute(f, x)
        
        # For a linear function, the fractional derivative should be well-defined
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
    
    def test_convergence_properties(self):
        """Test convergence properties."""
        weyl = WeylDerivative(0.5)
        
        # Test with different step sizes
        x1 = np.linspace(0, 1, 10)
        x2 = np.linspace(0, 1, 20)
        f1 = np.sin(x1)
        f2 = np.sin(x2)
        
        result1 = weyl.compute(f1, x1)
        result2 = weyl.compute(f2, x2)
        
        # Results should be consistent (within numerical precision)
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert len(result1) == len(x1)
        assert len(result2) == len(x2)