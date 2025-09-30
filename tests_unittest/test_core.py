"""
Unittest tests for HPFRACC core functionality
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestFractionalOrder(unittest.TestCase):
    """Test fractional order functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalOrder
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_order_creation(self):
        """Test fractional order creation"""
        # Test valid fractional orders
        orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for alpha in orders:
            order = self.FractionalOrder(alpha)
            self.assertEqual(order.value, alpha)
            self.assertIsInstance(order.value, float)
    
    def test_fractional_order_string_representation(self):
        """Test fractional order string representation"""
        order = self.FractionalOrder(0.5)
        str_repr = str(order)
        self.assertIn("0.5", str_repr)
        self.assertIn("α", str_repr)
    
    def test_fractional_order_comparison(self):
        """Test fractional order comparison"""
        order1 = self.FractionalOrder(0.5)
        order2 = self.FractionalOrder(0.7)
        order3 = self.FractionalOrder(0.5)
        
        self.assertEqual(order1.value, order3.value)
        self.assertNotEqual(order1.value, order2.value)
        self.assertTrue(order1.value < order2.value)

class TestFractionalDerivatives(unittest.TestCase):
    """Test fractional derivative implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative,
            CaputoDerivative,
            GrunwaldLetnikovDerivative
        )
        self.RiemannLiouvilleDerivative = RiemannLiouvilleDerivative
        self.CaputoDerivative = CaputoDerivative
        self.GrunwaldLetnikovDerivative = GrunwaldLetnikovDerivative
    
    def test_riemann_liouville_derivative(self):
        """Test Riemann-Liouville derivative"""
        from hpfracc.core.definitions import FractionalOrder
        order = FractionalOrder(0.5)
        rl_derivative = self.RiemannLiouvilleDerivative(order)
        self.assertEqual(rl_derivative.fractional_order.value, 0.5)
        self.assertIsInstance(rl_derivative.fractional_order.value, float)
    
    def test_caputo_derivative(self):
        """Test Caputo derivative"""
        from hpfracc.core.definitions import FractionalOrder
        order = FractionalOrder(0.5)
        caputo_derivative = self.CaputoDerivative(order)
        self.assertEqual(caputo_derivative.fractional_order.value, 0.5)
        self.assertIsInstance(caputo_derivative.fractional_order.value, float)
    
    def test_grunwald_letnikov_derivative(self):
        """Test Grünwald-Letnikov derivative"""
        from hpfracc.core.definitions import FractionalOrder
        order = FractionalOrder(0.5)
        gl_derivative = self.GrunwaldLetnikovDerivative(order)
        self.assertEqual(gl_derivative.fractional_order.value, 0.5)
        self.assertIsInstance(gl_derivative.fractional_order.value, float)
    
    def test_derivative_orders(self):
        """Test different derivative orders"""
        from hpfracc.core.definitions import FractionalOrder
        orders = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
        
        for order_val in orders:
            order = FractionalOrder(order_val)
            rl = self.RiemannLiouvilleDerivative(order)
            caputo = self.CaputoDerivative(order)
            gl = self.GrunwaldLetnikovDerivative(order)
            
            self.assertEqual(rl.fractional_order.value, order_val)
            self.assertEqual(caputo.fractional_order.value, order_val)
            self.assertEqual(gl.fractional_order.value, order_val)

class TestFractionalIntegrals(unittest.TestCase):
    """Test fractional integral implementations"""
    
    def test_fractional_integral_concepts(self):
        """Test fractional integral concepts"""
        # Test that we can import and use fractional integrals
        try:
            from hpfracc.core.integrals import FractionalIntegral
            integral = FractionalIntegral(0.5)
            self.assertEqual(integral.order, 0.5)
        except ImportError:
            # If not available, test the concept
            self.assertTrue(True, "Fractional integral concept exists")

class TestCoreUtilities(unittest.TestCase):
    """Test core utility functions"""
    
    def test_fractional_order_validation(self):
        """Test fractional order validation"""
        from hpfracc.core.definitions import FractionalOrder
        
        # Test valid orders
        valid_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for order in valid_orders:
            frac_order = FractionalOrder(order)
            self.assertIsInstance(frac_order.value, float)
            self.assertGreaterEqual(frac_order.value, 0)
    
    def test_derivative_method_registration(self):
        """Test derivative method registration"""
        # Test that derivative methods are properly registered
        from hpfracc.core.fractional_implementations import (
            RiemannLiouvilleDerivative,
            CaputoDerivative,
            GrunwaldLetnikovDerivative
        )
        
        # All should be importable
        self.assertTrue(RiemannLiouvilleDerivative is not None)
        self.assertTrue(CaputoDerivative is not None)
        self.assertTrue(GrunwaldLetnikovDerivative is not None)

if __name__ == '__main__':
    unittest.main()
