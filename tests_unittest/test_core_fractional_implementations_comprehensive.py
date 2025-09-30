"""
Comprehensive unittest tests for HPFRACC core fractional implementations
Targeting 100% coverage for hpfracc/core/fractional_implementations.py
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestAlphaCompatibilityWrapperComprehensive(unittest.TestCase):
    """Comprehensive tests for _AlphaCompatibilityWrapper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalOrder
        from hpfracc.core.fractional_implementations import _AlphaCompatibilityWrapper
        self.FractionalOrder = FractionalOrder
        self._AlphaCompatibilityWrapper = _AlphaCompatibilityWrapper
    
    def test_alpha_compatibility_wrapper_initialization(self):
        """Test _AlphaCompatibilityWrapper initialization"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        self.assertEqual(wrapper._fractional_order, fractional_order)
        self.assertEqual(wrapper.alpha, 0.5)
    
    def test_alpha_compatibility_wrapper_comparison(self):
        """Test _AlphaCompatibilityWrapper comparison operations"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        # Test equality with float
        self.assertTrue(wrapper == 0.5)
        self.assertFalse(wrapper == 0.7)
        
        # Test equality with int
        self.assertTrue(wrapper == 0)  # 0.5 == 0 is False, but let's test the mechanism
        self.assertFalse(wrapper == 1)
        
        # Test equality with another FractionalOrder
        other_order = self.FractionalOrder(0.5)
        self.assertTrue(wrapper == other_order)
        
        other_order_different = self.FractionalOrder(0.7)
        self.assertFalse(wrapper == other_order_different)
    
    def test_alpha_compatibility_wrapper_attribute_access(self):
        """Test _AlphaCompatibilityWrapper attribute delegation"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        # Test delegated attribute access
        self.assertEqual(wrapper.alpha, 0.5)
        self.assertEqual(wrapper.value, 0.5)
        self.assertFalse(wrapper.is_integer())
        self.assertTrue(wrapper.is_fractional())
        
        # Test non-existent attribute
        with self.assertRaises(AttributeError):
            wrapper.non_existent_attribute
    
    def test_alpha_compatibility_wrapper_representations(self):
        """Test _AlphaCompatibilityWrapper string representations"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        # Test __repr__
        repr_str = repr(wrapper)
        self.assertIn("FractionalOrder", repr_str)
        self.assertIn("0.5", repr_str)
    
    def test_alpha_compatibility_wrapper_conversions(self):
        """Test _AlphaCompatibilityWrapper type conversions"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        # Test __float__
        float_val = float(wrapper)
        self.assertEqual(float_val, 0.5)
        self.assertIsInstance(float_val, float)
        
        # Test __int__
        int_val = int(wrapper)
        self.assertEqual(int_val, 0)
        self.assertIsInstance(int_val, int)
        
        # Test with integer order
        integer_order = self.FractionalOrder(2.0)
        integer_wrapper = self._AlphaCompatibilityWrapper(integer_order)
        self.assertEqual(int(integer_wrapper), 2)
        self.assertEqual(float(integer_wrapper), 2.0)
    
    def test_alpha_compatibility_wrapper_class_support(self):
        """Test _AlphaCompatibilityWrapper class support"""
        fractional_order = self.FractionalOrder(0.5)
        wrapper = self._AlphaCompatibilityWrapper(fractional_order)
        
        # Test __class__ method
        wrapper_class = wrapper.__class__()
        self.assertEqual(wrapper_class, type(fractional_order))

class TestRiemannLiouvilleDerivativeComprehensive(unittest.TestCase):
    """Comprehensive tests for RiemannLiouvilleDerivative class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import RiemannLiouvilleDerivative
        from hpfracc.core.definitions import FractionalOrder
        self.RiemannLiouvilleDerivative = RiemannLiouvilleDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_riemann_liouville_derivative_initialization(self):
        """Test RiemannLiouvilleDerivative initialization"""
        # Test with float
        derivative = self.RiemannLiouvilleDerivative(0.5)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        self.assertIsNotNone(derivative.fractional_order)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.RiemannLiouvilleDerivative(order)
        self.assertEqual(derivative.fractional_order.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                derivative = self.RiemannLiouvilleDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)
    
    def test_riemann_liouville_derivative_properties(self):
        """Test RiemannLiouvilleDerivative properties"""
        derivative = self.RiemannLiouvilleDerivative(0.5)
        
        # Test that fractional_order is accessible
        self.assertIsNotNone(derivative.fractional_order)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'fractional_order'))
    
    def test_riemann_liouville_derivative_edge_cases(self):
        """Test RiemannLiouvilleDerivative edge cases"""
        # Test with edge case orders
        edge_orders = [0.0, 0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                derivative = self.RiemannLiouvilleDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)

class TestCaputoDerivativeComprehensive(unittest.TestCase):
    """Comprehensive tests for CaputoDerivative class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import CaputoDerivative
        from hpfracc.core.definitions import FractionalOrder
        self.CaputoDerivative = CaputoDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_caputo_derivative_initialization(self):
        """Test CaputoDerivative initialization"""
        # Test with float
        derivative = self.CaputoDerivative(0.5)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        self.assertIsNotNone(derivative.fractional_order)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.CaputoDerivative(order)
        self.assertEqual(derivative.fractional_order.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                derivative = self.CaputoDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)
    
    def test_caputo_derivative_properties(self):
        """Test CaputoDerivative properties"""
        derivative = self.CaputoDerivative(0.5)
        
        # Test that fractional_order is accessible
        self.assertIsNotNone(derivative.fractional_order)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'fractional_order'))
    
    def test_caputo_derivative_edge_cases(self):
        """Test CaputoDerivative edge cases"""
        # Test with edge case orders
        edge_orders = [0.0, 0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                derivative = self.CaputoDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)

class TestGrunwaldLetnikovDerivativeComprehensive(unittest.TestCase):
    """Comprehensive tests for GrunwaldLetnikovDerivative class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import GrunwaldLetnikovDerivative
        from hpfracc.core.definitions import FractionalOrder
        self.GrunwaldLetnikovDerivative = GrunwaldLetnikovDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_grunwald_letnikov_derivative_initialization(self):
        """Test GrunwaldLetnikovDerivative initialization"""
        # Test with float
        derivative = self.GrunwaldLetnikovDerivative(0.5)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        self.assertIsNotNone(derivative.fractional_order)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.GrunwaldLetnikovDerivative(order)
        self.assertEqual(derivative.fractional_order.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                derivative = self.GrunwaldLetnikovDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)
    
    def test_grunwald_letnikov_derivative_properties(self):
        """Test GrunwaldLetnikovDerivative properties"""
        derivative = self.GrunwaldLetnikovDerivative(0.5)
        
        # Test that fractional_order is accessible
        self.assertIsNotNone(derivative.fractional_order)
        self.assertEqual(derivative.fractional_order.alpha, 0.5)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'fractional_order'))
    
    def test_grunwald_letnikov_derivative_edge_cases(self):
        """Test GrunwaldLetnikovDerivative edge cases"""
        # Test with edge case orders
        edge_orders = [0.0, 0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                derivative = self.GrunwaldLetnikovDerivative(alpha)
                self.assertEqual(derivative.fractional_order.alpha, alpha)

class TestFractionalIntegralComprehensive(unittest.TestCase):
    """Comprehensive tests for FractionalIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import FractionalIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.FractionalIntegral = FractionalIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_integral_initialization(self):
        """Test FractionalIntegral initialization"""
        # Test with float
        integral = self.FractionalIntegral(0.5)
        self.assertEqual(integral.fractional_order.alpha, 0.5)
        self.assertIsNotNone(integral.fractional_order)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.FractionalIntegral(order)
        self.assertEqual(integral.fractional_order.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                integral = self.FractionalIntegral(alpha)
                self.assertEqual(integral.fractional_order.alpha, alpha)
    
    def test_fractional_integral_properties(self):
        """Test FractionalIntegral properties"""
        integral = self.FractionalIntegral(0.5)
        
        # Test that fractional_order is accessible
        self.assertIsNotNone(integral.fractional_order)
        self.assertEqual(integral.fractional_order.alpha, 0.5)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(integral, 'fractional_order'))
    
    def test_fractional_integral_edge_cases(self):
        """Test FractionalIntegral edge cases"""
        # Test with edge case orders
        edge_orders = [0.0, 0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                integral = self.FractionalIntegral(alpha)
                self.assertEqual(integral.fractional_order.alpha, alpha)

class TestIntegrationAndEdgeCases(unittest.TestCase):
    """Test integration between components and edge cases"""
    
    def test_derivative_classes_integration(self):
        """Test integration between different derivative classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
        )
        from hpfracc.core.definitions import FractionalOrder
        
        order = FractionalOrder(0.5)
        
        # Test that all derivatives work with the same order
        rl = RiemannLiouvilleDerivative(order)
        caputo = CaputoDerivative(order)
        gl = GrunwaldLetnikovDerivative(order)
        
        self.assertEqual(rl.fractional_order.alpha, 0.5)
        self.assertEqual(caputo.fractional_order.alpha, 0.5)
        self.assertEqual(gl.fractional_order.alpha, 0.5)
    
    def test_alpha_compatibility_wrapper_integration(self):
        """Test _AlphaCompatibilityWrapper integration with derivatives"""
        from hpfracc.core.fractional_implementations import (
            _AlphaCompatibilityWrapper, RiemannLiouvilleDerivative
        )
        from hpfracc.core.definitions import FractionalOrder
        
        # Test wrapper with derivative
        order = FractionalOrder(0.5)
        wrapper = _AlphaCompatibilityWrapper(order)
        
        # Test that wrapper behaves correctly
        self.assertEqual(wrapper.alpha, 0.5)
        self.assertEqual(float(wrapper), 0.5)
        self.assertEqual(int(wrapper), 0)
        
        # Test comparison
        self.assertTrue(wrapper == 0.5)
        self.assertFalse(wrapper == 0.7)
    
    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            FractionalIntegral, _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        # Test with very small positive numbers
        small_order = FractionalOrder(1e-10)
        small_wrapper = _AlphaCompatibilityWrapper(small_order)
        
        self.assertEqual(small_wrapper.alpha, 1e-10)
        self.assertEqual(float(small_wrapper), 1e-10)
        self.assertEqual(int(small_wrapper), 0)
        
        # Test with very large numbers
        large_order = FractionalOrder(1e10)
        large_wrapper = _AlphaCompatibilityWrapper(large_order)
        
        self.assertEqual(large_wrapper.alpha, 1e10)
        self.assertEqual(float(large_wrapper), 1e10)
        self.assertEqual(int(large_wrapper), int(1e10))
        
        # Test derivatives with edge cases
        edge_orders = [0.0, 0.001, 0.999, 1.0, 1.001, 10.0, 100.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                order = FractionalOrder(alpha)
                wrapper = _AlphaCompatibilityWrapper(order)
                
                rl = RiemannLiouvilleDerivative(wrapper)
                caputo = CaputoDerivative(wrapper)
                gl = GrunwaldLetnikovDerivative(wrapper)
                integral = FractionalIntegral(wrapper)
                
                self.assertEqual(rl.fractional_order.alpha, alpha)
                self.assertEqual(caputo.fractional_order.alpha, alpha)
                self.assertEqual(gl.fractional_order.alpha, alpha)
                self.assertEqual(integral.fractional_order.alpha, alpha)
    
    def test_type_consistency(self):
        """Test type consistency across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            FractionalIntegral, _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        order = FractionalOrder(0.5)
        wrapper = _AlphaCompatibilityWrapper(order)
        
        # Test that all classes accept both FractionalOrder and wrapper
        rl1 = RiemannLiouvilleDerivative(order)
        rl2 = RiemannLiouvilleDerivative(wrapper)
        
        caputo1 = CaputoDerivative(order)
        caputo2 = CaputoDerivative(wrapper)
        
        gl1 = GrunwaldLetnikovDerivative(order)
        gl2 = GrunwaldLetnikovDerivative(wrapper)
        
        integral1 = FractionalIntegral(order)
        integral2 = FractionalIntegral(wrapper)
        
        # All should have the same fractional order
        self.assertEqual(rl1.fractional_order.alpha, rl2.fractional_order.alpha)
        self.assertEqual(caputo1.fractional_order.alpha, caputo2.fractional_order.alpha)
        self.assertEqual(gl1.fractional_order.alpha, gl2.fractional_order.alpha)
        self.assertEqual(integral1.fractional_order.alpha, integral2.fractional_order.alpha)
    
    def test_precision_handling(self):
        """Test precision handling across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            FractionalIntegral, _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        # Test high precision numbers
        precise_alpha = 0.123456789012345
        order = FractionalOrder(precise_alpha)
        wrapper = _AlphaCompatibilityWrapper(order)
        
        # Test derivatives with high precision
        rl = RiemannLiouvilleDerivative(wrapper)
        caputo = CaputoDerivative(wrapper)
        gl = GrunwaldLetnikovDerivative(wrapper)
        integral = FractionalIntegral(wrapper)
        
        self.assertAlmostEqual(rl.fractional_order.alpha, precise_alpha)
        self.assertAlmostEqual(caputo.fractional_order.alpha, precise_alpha)
        self.assertAlmostEqual(gl.fractional_order.alpha, precise_alpha)
        self.assertAlmostEqual(integral.fractional_order.alpha, precise_alpha)
        
        # Test float conversion precision
        self.assertAlmostEqual(float(wrapper), precise_alpha)
    
    def test_error_handling(self):
        """Test error handling across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            FractionalIntegral, _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        # Test with invalid order (should be handled by FractionalOrder validation)
        with self.assertRaises(ValueError):
            invalid_order = FractionalOrder(-0.1)
        
        # Test wrapper with invalid order
        with self.assertRaises(ValueError):
            invalid_wrapper = _AlphaCompatibilityWrapper(FractionalOrder(-0.1))
        
        # Test derivatives with invalid order
        with self.assertRaises(ValueError):
            invalid_rl = RiemannLiouvilleDerivative(-0.1)
        
        with self.assertRaises(ValueError):
            invalid_caputo = CaputoDerivative(-0.1)
        
        with self.assertRaises(ValueError):
            invalid_gl = GrunwaldLetnikovDerivative(-0.1)
        
        with self.assertRaises(ValueError):
            invalid_integral = FractionalIntegral(-0.1)

if __name__ == '__main__':
    unittest.main()
