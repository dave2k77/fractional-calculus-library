#!/usr/bin/env python3
"""
Comprehensive tests for hpfracc.core module functionality.

This test suite focuses on actual functionality rather than coverage metrics,
ensuring that the core mathematical operations work correctly across different
use cases and backend configurations.
"""

import pytest
import numpy as np
from typing import Callable, Union
import tempfile
import os

# Test the core module imports work without heavy dependencies
def test_core_imports_work():
    """Test that core module can be imported without heavy dependencies."""
    # This should work even if torch/jax are not available
    from hpfracc.core import definitions, derivatives, integrals, utilities
    from hpfracc.core.definitions import FractionalOrder, DefinitionType
    from hpfracc.core.derivatives import BaseFractionalDerivative, create_fractional_derivative
    from hpfracc.core.integrals import RiemannLiouvilleIntegral, create_fractional_integral
    
    # Basic functionality should be available
    assert FractionalOrder is not None
    assert DefinitionType is not None
    assert BaseFractionalDerivative is not None


class TestFractionalOrder:
    """Test FractionalOrder class functionality."""
    
    def test_fractional_order_creation(self):
        """Test creating FractionalOrder objects."""
        from hpfracc.core.definitions import FractionalOrder
        
        # Test with float
        alpha = FractionalOrder(0.5)
        assert alpha.alpha == 0.5
        assert alpha.value == 0.5
        assert not alpha.is_integer
        assert alpha.is_fractional
        
        # Test with integer
        alpha = FractionalOrder(1.0)
        assert alpha.alpha == 1.0
        assert alpha.is_integer
        assert not alpha.is_fractional
        
        # Test properties
        alpha = FractionalOrder(1.7)
        assert alpha.integer_part == 1
        assert abs(alpha.fractional_part - 0.7) < 1e-10
    
    def test_fractional_order_validation(self):
        """Test FractionalOrder validation."""
        from hpfracc.core.definitions import FractionalOrder
        
        # Valid orders
        FractionalOrder(0.0)
        FractionalOrder(0.5)
        FractionalOrder(1.0)
        FractionalOrder(2.0)
        
        # Invalid orders should raise ValueError
        with pytest.raises(ValueError):
            FractionalOrder(-0.1)
        
        with pytest.raises(ValueError):
            FractionalOrder(float('inf'))
        
        with pytest.raises(ValueError):
            FractionalOrder(float('nan'))
    
    def test_fractional_order_equality(self):
        """Test FractionalOrder equality and hashing."""
        from hpfracc.core.definitions import FractionalOrder
        
        alpha1 = FractionalOrder(0.5)
        alpha2 = FractionalOrder(0.5)
        alpha3 = FractionalOrder(0.6)
        
        assert alpha1 == alpha2
        assert alpha1 != alpha3
        assert hash(alpha1) == hash(alpha2)
        assert hash(alpha1) != hash(alpha3)


class TestFractionalDefinitions:
    """Test fractional derivative definitions."""
    
    def test_definition_types(self):
        """Test DefinitionType enum."""
        from hpfracc.core.definitions import DefinitionType
        
        assert DefinitionType.CAPUTO.value == "caputo"
        assert DefinitionType.RIEMANN_LIOUVILLE.value == "riemann_liouville"
        assert DefinitionType.GRUNWALD_LETNIKOV.value == "grunwald_letnikov"
    
    def test_caputo_definition(self):
        """Test Caputo definition."""
        from hpfracc.core.definitions import CaputoDefinition, FractionalOrder
        
        alpha = FractionalOrder(0.5)
        caputo = CaputoDefinition(alpha)
        
        assert caputo.order == alpha
        assert caputo.definition_type.value == "caputo"
        assert caputo.n == 1  # ceil(0.5) = 1
        
        # Test advantages and limitations
        advantages = caputo.get_advantages()
        limitations = caputo.get_limitations()
        
        assert isinstance(advantages, list)
        assert isinstance(limitations, list)
        assert len(advantages) > 0
        assert len(limitations) > 0
    
    def test_riemann_liouville_definition(self):
        """Test Riemann-Liouville definition."""
        from hpfracc.core.definitions import RiemannLiouvilleDefinition, FractionalOrder
        
        alpha = FractionalOrder(0.7)
        rl = RiemannLiouvilleDefinition(alpha)
        
        assert rl.order == alpha
        assert rl.definition_type.value == "riemann_liouville"
        assert rl.n == 1  # ceil(0.7) = 1
        
        # Test properties
        properties = rl.get_properties()
        assert "linearity" in properties
        assert "semigroup_property" in properties


class TestFractionalDerivatives:
    """Test fractional derivative implementations."""
    
    def test_derivative_factory_creation(self):
        """Test creating derivatives via factory."""
        from hpfracc.core.derivatives import create_fractional_derivative, DefinitionType
        
        # Test creating different types of derivatives
        try:
            rl_deriv = create_fractional_derivative(DefinitionType.RIEMANN_LIOUVILLE, 0.5)
            assert rl_deriv is not None
            assert rl_deriv.alpha.alpha == 0.5
        except ImportError:
            # If algorithms module is not available, this is expected
            pytest.skip("Algorithms module not available")
    
    def test_derivative_operator_creation(self):
        """Test creating derivative operators."""
        from hpfracc.core.derivatives import create_derivative_operator, DefinitionType
        
        try:
            op = create_derivative_operator(DefinitionType.CAPUTO, 0.5)
            assert op is not None
            assert op.alpha.alpha == 0.5
            assert op.definition_type == DefinitionType.CAPUTO
        except ImportError:
            pytest.skip("Algorithms module not available")
    
    def test_derivative_chain(self):
        """Test derivative chaining."""
        from hpfracc.core.derivatives import FractionalDerivativeChain
        from hpfracc.core.derivatives import create_fractional_derivative, DefinitionType
        
        try:
            # Create a simple function
            def f(x):
                return x**2
            
            # Create derivatives
            d1 = create_fractional_derivative(DefinitionType.RIEMANN_LIOUVILLE, 0.3)
            d2 = create_fractional_derivative(DefinitionType.RIEMANN_LIOUVILLE, 0.2)
            
            # Create chain
            chain = FractionalDerivativeChain([d1, d2])
            
            # Test total order
            total_order = chain.get_total_order()
            assert abs(total_order - 0.5) < 1e-10
            
            # Test chain info
            info = chain.get_chain_info()
            assert len(info) == 2
            
        except ImportError:
            pytest.skip("Algorithms module not available")


class TestFractionalIntegrals:
    """Test fractional integral implementations."""
    
    def test_riemann_liouville_integral(self):
        """Test Riemann-Liouville fractional integral."""
        from hpfracc.core.integrals import RiemannLiouvilleIntegral
        
        # Create integral
        integral = RiemannLiouvilleIntegral(0.5)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "RL"
        
        # Test with simple function
        def f(x):
            return 1.0  # Constant function
        
        # Test at a point
        result = integral(f, 1.0)
        assert isinstance(result, float)
        assert result > 0  # Should be positive for positive function
    
    def test_caputo_integral(self):
        """Test Caputo fractional integral."""
        from hpfracc.core.integrals import CaputoIntegral
        
        # Create integral
        integral = CaputoIntegral(0.5)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Caputo"
        
        # Test with simple function
        def f(x):
            return 1.0
        
        # For 0 < alpha < 1, Caputo equals Riemann-Liouville
        result = integral(f, 1.0)
        assert isinstance(result, float)
        assert result > 0
    
    def test_weyl_integral(self):
        """Test Weyl fractional integral."""
        from hpfracc.core.integrals import WeylIntegral
        
        # Create integral
        integral = WeylIntegral(0.5)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Weyl"
        
        # Test with simple function
        def f(x):
            return 1.0
        
        result = integral(f, 1.0)
        assert isinstance(result, float)
    
    def test_hadamard_integral(self):
        """Test Hadamard fractional integral."""
        from hpfracc.core.integrals import HadamardIntegral
        
        # Create integral
        integral = HadamardIntegral(0.5)
        assert integral.alpha.alpha == 0.5
        assert integral.method == "Hadamard"
        
        # Test with simple function
        def f(x):
            return 1.0
        
        # Hadamard requires x > 1
        result = integral(f, 2.0)
        assert isinstance(result, float)
        assert result > 0
        
        # Test validation
        with pytest.raises(ValueError):
            integral(f, 0.5)  # Should fail for x <= 1
    
    def test_integral_factory(self):
        """Test integral factory function."""
        from hpfracc.core.integrals import create_fractional_integral
        
        # Test creating different types
        rl = create_fractional_integral(0.5, "RL")
        assert rl.method == "RL"
        
        caputo = create_fractional_integral(0.5, "Caputo")
        assert caputo.method == "Caputo"
        
        weyl = create_fractional_integral(0.5, "Weyl")
        assert weyl.method == "Weyl"
        
        hadamard = create_fractional_integral(0.5, "Hadamard")
        assert hadamard.method == "Hadamard"
        
        # Test invalid method
        with pytest.raises(ValueError):
            create_fractional_integral(0.5, "Invalid")


class TestMathematicalUtilities:
    """Test mathematical utility functions."""
    
    def test_factorial_fractional(self):
        """Test fractional factorial function."""
        from hpfracc.core.utilities import factorial_fractional
        
        # Test integer factorial
        assert factorial_fractional(5) == 120.0
        assert factorial_fractional(0) == 1.0
        
        # Test fractional factorial (gamma function)
        result = factorial_fractional(0.5)
        assert isinstance(result, float)
        assert result > 0
    
    def test_binomial_coefficient(self):
        """Test binomial coefficient function."""
        from hpfracc.core.utilities import binomial_coefficient
        
        # Test integer cases
        assert binomial_coefficient(5, 2) == 10.0
        assert binomial_coefficient(5, 0) == 1.0
        assert binomial_coefficient(5, 5) == 1.0
        
        # Test fractional cases
        result = binomial_coefficient(0.5, 0.3)
        assert isinstance(result, float)
        assert result > 0
    
    def test_pochhammer_symbol(self):
        """Test Pochhammer symbol function."""
        from hpfracc.core.utilities import pochhammer_symbol
        
        # Test basic cases
        assert pochhammer_symbol(2.0, 0) == 1.0
        assert pochhammer_symbol(2.0, 1) == 2.0
        assert pochhammer_symbol(2.0, 2) == 6.0  # 2 * 3
    
    def test_safe_divide(self):
        """Test safe division function."""
        from hpfracc.core.utilities import safe_divide
        
        # Test normal division
        assert safe_divide(10, 2) == 5.0
        
        # Test division by zero
        result = safe_divide(10, 0)
        assert result == 0.0  # Should return 0 for division by zero
        
        # Test with very small numbers
        result = safe_divide(1e-10, 1e-20)
        assert isinstance(result, float)


class TestPerformanceMonitoring:
    """Test performance monitoring utilities."""
    
    def test_timing_decorator(self):
        """Test timing decorator."""
        from hpfracc.core.utilities import timing_decorator
        import time
        
        @timing_decorator
        def slow_function():
            time.sleep(0.01)  # 10ms
            return "done"
        
        result = slow_function()
        assert result == "done"
    
    def test_memory_usage_decorator(self):
        """Test memory usage decorator."""
        from hpfracc.core.utilities import memory_usage_decorator
        
        @memory_usage_decorator
        def memory_function():
            # Create some data
            data = [i for i in range(1000)]
            return len(data)
        
        result = memory_function()
        assert result == 1000
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor class."""
        from hpfracc.core.utilities import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Test timing
        with monitor.timer("test_operation"):
            import time
            time.sleep(0.001)
        
        # Test memory tracking using context manager
        with monitor.memory_tracker("test_data"):
            # Create some data to track
            data = [i for i in range(1000)]
            _ = len(data)
        
        # Test getting stats
        stats = monitor.get_statistics()
        assert "test_operation" in stats
        assert "test_data" in stats


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_validate_function(self):
        """Test function validation."""
        from hpfracc.core.utilities import validate_function
        
        # Test valid function
        def f(x):
            return x**2
        
        assert validate_function(f) is True
        
        # Test invalid function
        assert validate_function(None) is False
        assert validate_function("not_a_function") is False
    
    def test_validate_tensor_input(self):
        """Test tensor input validation."""
        from hpfracc.core.utilities import validate_tensor_input
        
        # Test valid inputs
        assert validate_tensor_input(np.array([1, 2, 3])) is True
        
        # Test invalid inputs (scalars are not valid tensor inputs)
        assert validate_tensor_input(1.0) is False
        assert validate_tensor_input(None) is False
        assert validate_tensor_input("not_a_tensor") is False
    
    def test_check_numerical_stability(self):
        """Test numerical stability checking."""
        from hpfracc.core.utilities import check_numerical_stability
        
        # Test stable array
        stable_array = np.array([1.0, 2.0, 3.0])
        assert check_numerical_stability(stable_array) == True
        
        # Test unstable array
        unstable_array = np.array([1e-20, 1e20, np.inf])
        assert check_numerical_stability(unstable_array) == False


class TestIntegrationWithAdapters:
    """Test integration with the adapter system."""
    
    def test_core_works_without_heavy_dependencies(self):
        """Test that core module works without torch/jax."""
        # This test should pass even if torch/jax are not installed
        from hpfracc.core.definitions import FractionalOrder
        from hpfracc.core.integrals import RiemannLiouvilleIntegral
        
        # Create basic objects
        alpha = FractionalOrder(0.5)
        integral = RiemannLiouvilleIntegral(alpha)
        
        # Test basic functionality
        def f(x):
            return 1.0
        
        result = integral(f, 1.0)
        assert isinstance(result, float)
        assert result > 0
    
    def test_graceful_handling_of_missing_dependencies(self):
        """Test graceful handling when heavy dependencies are missing."""
        # Test that the module doesn't crash when dependencies are missing
        from hpfracc.core import definitions, derivatives, integrals, utilities
        
        # These should all work even without torch/jax
        assert definitions is not None
        assert derivatives is not None
        assert integrals is not None
        assert utilities is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
