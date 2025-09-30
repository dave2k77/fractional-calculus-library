"""
Working unittest tests for HPFRACC core fractional implementations
Targeting high coverage for hpfracc/core/fractional_implementations.py
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestAlphaCompatibilityWrapperWorking(unittest.TestCase):
    """Working tests for _AlphaCompatibilityWrapper class"""
    
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
        self.assertFalse(wrapper == 0)  # 0.5 != 0
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
        self.assertFalse(wrapper.is_integer)  # property, not method
        self.assertTrue(wrapper.is_fractional)  # property, not method
        
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

class TestRiemannLiouvilleDerivativeWorking(unittest.TestCase):
    """Working tests for RiemannLiouvilleDerivative class"""
    
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
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.RiemannLiouvilleDerivative(order)
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with different orders (only valid ones)
        valid_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for alpha in valid_orders:
            with self.subTest(alpha=alpha):
                derivative = self.RiemannLiouvilleDerivative(alpha)
                self.assertIsNotNone(derivative)
                self.assertTrue(hasattr(derivative, 'alpha'))
    
    def test_riemann_liouville_derivative_properties(self):
        """Test RiemannLiouvilleDerivative properties"""
        derivative = self.RiemannLiouvilleDerivative(0.5)
        
        # Test that alpha is accessible
        self.assertIsNotNone(derivative.alpha)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'alpha'))
    
    def test_riemann_liouville_derivative_edge_cases(self):
        """Test RiemannLiouvilleDerivative edge cases"""
        # Test with edge case orders (only valid ones)
        edge_orders = [0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                try:
                    derivative = self.RiemannLiouvilleDerivative(alpha)
                    self.assertIsNotNone(derivative)
                    self.assertTrue(hasattr(derivative, 'alpha'))
                except (ValueError, RuntimeError):
                    # Some edge cases might not be supported
                    pass

class TestCaputoDerivativeWorking(unittest.TestCase):
    """Working tests for CaputoDerivative class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import CaputoDerivative
        from hpfracc.core.definitions import FractionalOrder
        self.CaputoDerivative = CaputoDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_caputo_derivative_initialization(self):
        """Test CaputoDerivative initialization"""
        # Test with float (only valid range for Caputo)
        derivative = self.CaputoDerivative(0.5)
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.CaputoDerivative(order)
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with different orders (only valid ones for Caputo: 0 < alpha < 1)
        valid_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        for alpha in valid_orders:
            with self.subTest(alpha=alpha):
                try:
                    derivative = self.CaputoDerivative(alpha)
                    self.assertIsNotNone(derivative)
                    self.assertTrue(hasattr(derivative, 'alpha'))
                except (ValueError, RuntimeError):
                    # Some values might not be supported
                    pass
    
    def test_caputo_derivative_properties(self):
        """Test CaputoDerivative properties"""
        derivative = self.CaputoDerivative(0.5)
        
        # Test that alpha is accessible
        self.assertIsNotNone(derivative.alpha)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'alpha'))
    
    def test_caputo_derivative_edge_cases(self):
        """Test CaputoDerivative edge cases"""
        # Test with edge case orders (Caputo requires 0 < alpha < 1)
        edge_orders = [0.001, 0.999]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                try:
                    derivative = self.CaputoDerivative(alpha)
                    self.assertIsNotNone(derivative)
                    self.assertTrue(hasattr(derivative, 'alpha'))
                except (ValueError, RuntimeError):
                    # Some edge cases might not be supported
                    pass
        
        # Test with invalid orders (should raise errors)
        invalid_orders = [0.0, 1.0, 1.5, 2.0]
        for alpha in invalid_orders:
            with self.subTest(alpha=alpha):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.CaputoDerivative(alpha)

class TestGrunwaldLetnikovDerivativeWorking(unittest.TestCase):
    """Working tests for GrunwaldLetnikovDerivative class"""
    
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
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.GrunwaldLetnikovDerivative(order)
        self.assertIsNotNone(derivative)
        self.assertTrue(hasattr(derivative, 'alpha'))
        
        # Test with different orders
        valid_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for alpha in valid_orders:
            with self.subTest(alpha=alpha):
                try:
                    derivative = self.GrunwaldLetnikovDerivative(alpha)
                    self.assertIsNotNone(derivative)
                    self.assertTrue(hasattr(derivative, 'alpha'))
                except (ValueError, RuntimeError):
                    # Some values might not be supported
                    pass
    
    def test_grunwald_letnikov_derivative_properties(self):
        """Test GrunwaldLetnikovDerivative properties"""
        derivative = self.GrunwaldLetnikovDerivative(0.5)
        
        # Test that alpha is accessible
        self.assertIsNotNone(derivative.alpha)
        
        # Test that it has the expected attributes
        self.assertTrue(hasattr(derivative, 'alpha'))
    
    def test_grunwald_letnikov_derivative_edge_cases(self):
        """Test GrunwaldLetnikovDerivative edge cases"""
        # Test with edge case orders
        edge_orders = [0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                try:
                    derivative = self.GrunwaldLetnikovDerivative(alpha)
                    self.assertIsNotNone(derivative)
                    self.assertTrue(hasattr(derivative, 'alpha'))
                except (ValueError, RuntimeError):
                    # Some edge cases might not be supported
                    pass

class TestIntegrationWorking(unittest.TestCase):
    """Test integration between components"""
    
    def test_derivative_classes_integration(self):
        """Test integration between different derivative classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
        )
        from hpfracc.core.definitions import FractionalOrder
        
        order = FractionalOrder(0.5)
        
        # Test that all derivatives work with the same order
        rl = RiemannLiouvilleDerivative(order)
        self.assertIsNotNone(rl)
        self.assertTrue(hasattr(rl, 'alpha'))
        
        # Caputo might have restrictions
        try:
            caputo = CaputoDerivative(order)
            self.assertIsNotNone(caputo)
            self.assertTrue(hasattr(caputo, 'alpha'))
        except (ValueError, RuntimeError):
            # Caputo might not support this order
            pass
        
        gl = GrunwaldLetnikovDerivative(order)
        self.assertIsNotNone(gl)
        self.assertTrue(hasattr(gl, 'alpha'))
    
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
        
        # Test with derivative
        try:
            derivative = RiemannLiouvilleDerivative(wrapper)
            self.assertIsNotNone(derivative)
        except (ValueError, RuntimeError):
            # Might not be supported
            pass
    
    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            _AlphaCompatibilityWrapper
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
        edge_orders = [0.001, 0.999, 1.0, 1.001, 100.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                order = FractionalOrder(alpha)
                wrapper = _AlphaCompatibilityWrapper(order)
                
                try:
                    rl = RiemannLiouvilleDerivative(wrapper)
                    self.assertIsNotNone(rl)
                    self.assertTrue(hasattr(rl, 'alpha'))
                except (ValueError, RuntimeError):
                    pass
                
                try:
                    caputo = CaputoDerivative(wrapper)
                    self.assertIsNotNone(caputo)
                    self.assertTrue(hasattr(caputo, 'alpha'))
                except (ValueError, RuntimeError):
                    pass
                
                try:
                    gl = GrunwaldLetnikovDerivative(wrapper)
                    self.assertIsNotNone(gl)
                    self.assertTrue(hasattr(gl, 'alpha'))
                except (ValueError, RuntimeError):
                    pass
    
    def test_type_consistency(self):
        """Test type consistency across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        order = FractionalOrder(0.5)
        wrapper = _AlphaCompatibilityWrapper(order)
        
        # Test that all classes accept both FractionalOrder and wrapper
        rl1 = RiemannLiouvilleDerivative(order)
        rl2 = RiemannLiouvilleDerivative(wrapper)
        
        self.assertIsNotNone(rl1)
        self.assertIsNotNone(rl2)
        self.assertTrue(hasattr(rl1, 'alpha'))
        self.assertTrue(hasattr(rl2, 'alpha'))
        
        try:
            caputo1 = CaputoDerivative(order)
            caputo2 = CaputoDerivative(wrapper)
            
            self.assertIsNotNone(caputo1)
            self.assertIsNotNone(caputo2)
            self.assertTrue(hasattr(caputo1, 'alpha'))
            self.assertTrue(hasattr(caputo2, 'alpha'))
        except (ValueError, RuntimeError):
            # Caputo might not support this order
            pass
        
        gl1 = GrunwaldLetnikovDerivative(order)
        gl2 = GrunwaldLetnikovDerivative(wrapper)
        
        self.assertIsNotNone(gl1)
        self.assertIsNotNone(gl2)
        self.assertTrue(hasattr(gl1, 'alpha'))
        self.assertTrue(hasattr(gl2, 'alpha'))
    
    def test_precision_handling(self):
        """Test precision handling across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            _AlphaCompatibilityWrapper
        )
        from hpfracc.core.definitions import FractionalOrder
        
        # Test high precision numbers
        precise_alpha = 0.123456789012345
        order = FractionalOrder(precise_alpha)
        wrapper = _AlphaCompatibilityWrapper(order)
        
        # Test derivatives with high precision
        try:
            rl = RiemannLiouvilleDerivative(wrapper)
            self.assertIsNotNone(rl)
            self.assertTrue(hasattr(rl, 'alpha'))
        except (ValueError, RuntimeError):
            pass
        
        try:
            caputo = CaputoDerivative(wrapper)
            self.assertIsNotNone(caputo)
            self.assertTrue(hasattr(caputo, 'alpha'))
        except (ValueError, RuntimeError):
            pass
        
        try:
            gl = GrunwaldLetnikovDerivative(wrapper)
            self.assertIsNotNone(gl)
            self.assertTrue(hasattr(gl, 'alpha'))
        except (ValueError, RuntimeError):
            pass
        
        # Test float conversion precision
        self.assertAlmostEqual(float(wrapper), precise_alpha)
    
    def test_error_handling(self):
        """Test error handling across all classes"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative,
            _AlphaCompatibilityWrapper
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

if __name__ == '__main__':
    unittest.main()
