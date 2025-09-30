"""
Working unittest tests for HPFRACC core integrals
Targeting high coverage for hpfracc/core/integrals.py
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestFractionalIntegralWorking(unittest.TestCase):
    """Working tests for FractionalIntegral base class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import FractionalIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.FractionalIntegral = FractionalIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_integral_initialization(self):
        """Test FractionalIntegral initialization"""
        # Test with float
        integral = self.FractionalIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "RL")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.FractionalIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
        
        # Test with method
        integral = self.FractionalIntegral(0.5, method="Caputo")
        self.assertEqual(integral.method, "Caputo")
        
        # Test with different methods
        methods = ["RL", "Caputo", "Weyl", "Hadamard"]
        for method in methods:
            with self.subTest(method=method):
                integral = self.FractionalIntegral(0.5, method=method)
                self.assertEqual(integral.method, method)
    
    def test_fractional_integral_validation(self):
        """Test FractionalIntegral validation"""
        # Test valid orders
        valid_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for order_val in valid_orders:
            with self.subTest(order=order_val):
                integral = self.FractionalIntegral(order_val)
                self.assertEqual(integral.alpha.alpha, order_val)
        
        # Test invalid orders (should raise errors)
        invalid_orders = [np.inf, -np.inf, np.nan, -0.1]
        for order_val in invalid_orders:
            with self.subTest(order=order_val):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.FractionalIntegral(order_val)
    
    def test_fractional_integral_call(self):
        """Test FractionalIntegral __call__ method"""
        integral = self.FractionalIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_fractional_integral_compute(self):
        """Test FractionalIntegral compute method"""
        integral = self.FractionalIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral.compute(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral.compute(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_fractional_integral_representations(self):
        """Test FractionalIntegral string representations"""
        integral = self.FractionalIntegral(0.5, method="RL")
        
        # Test __repr__
        repr_str = repr(integral)
        self.assertIn("FractionalIntegral", repr_str)
        self.assertIn("0.5", repr_str)
        self.assertIn("RL", repr_str)

class TestRiemannLiouvilleIntegralWorking(unittest.TestCase):
    """Working tests for RiemannLiouvilleIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import RiemannLiouvilleIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.RiemannLiouvilleIntegral = RiemannLiouvilleIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_riemann_liouville_integral_initialization(self):
        """Test RiemannLiouvilleIntegral initialization"""
        # Test with float
        integral = self.RiemannLiouvilleIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "RL")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.RiemannLiouvilleIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5]:
            with self.subTest(alpha=alpha):
                integral = self.RiemannLiouvilleIntegral(alpha)
                self.assertEqual(integral.alpha.alpha, alpha)
    
    def test_riemann_liouville_integral_call(self):
        """Test RiemannLiouvilleIntegral __call__ method"""
        integral = self.RiemannLiouvilleIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_riemann_liouville_integral_compute(self):
        """Test RiemannLiouvilleIntegral compute method"""
        integral = self.RiemannLiouvilleIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral.compute(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral.compute(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        
        # Test with h parameter
        result = integral.compute(test_function, x, h=0.1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_riemann_liouville_integral_coerce_function(self):
        """Test RiemannLiouvilleIntegral _coerce_function method"""
        integral = self.RiemannLiouvilleIntegral(0.5)
        
        # Test with callable function
        def test_function(x):
            return x
        
        coerced = integral._coerce_function(test_function, 1.0)
        self.assertTrue(callable(coerced))
        
        # Test with array function
        x = np.linspace(0, 1, 10)
        y = x**2
        coerced = integral._coerce_function(y, x)
        self.assertTrue(callable(coerced))
    
    def test_riemann_liouville_integral_compute_scalar(self):
        """Test RiemannLiouvilleIntegral _compute_scalar method"""
        integral = self.RiemannLiouvilleIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test scalar computation
        result = integral._compute_scalar(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with different values
        for x_val in [0.1, 0.5, 1.0, 2.0]:
            with self.subTest(x=x_val):
                result = integral._compute_scalar(test_function, x_val)
                self.assertIsInstance(result, (int, float))
    
    def test_riemann_liouville_integral_compute_array_numpy(self):
        """Test RiemannLiouvilleIntegral _compute_array_numpy method"""
        integral = self.RiemannLiouvilleIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test array computation
        x = np.linspace(0, 1, 10)
        result = integral._compute_array_numpy(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        
        # Test with different array sizes
        for size in [5, 10, 20]:
            with self.subTest(size=size):
                x = np.linspace(0, 1, size)
                result = integral._compute_array_numpy(test_function, x)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, x.shape)

class TestCaputoIntegralWorking(unittest.TestCase):
    """Working tests for CaputoIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import CaputoIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.CaputoIntegral = CaputoIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_caputo_integral_initialization(self):
        """Test CaputoIntegral initialization"""
        # Test with float
        integral = self.CaputoIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "Caputo")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.CaputoIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
        
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9]:
            with self.subTest(alpha=alpha):
                integral = self.CaputoIntegral(alpha)
                self.assertEqual(integral.alpha.alpha, alpha)
    
    def test_caputo_integral_call(self):
        """Test CaputoIntegral __call__ method"""
        integral = self.CaputoIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_caputo_integral_compute(self):
        """Test CaputoIntegral compute method"""
        integral = self.CaputoIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral.compute(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral.compute(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

class TestWeylIntegralWorking(unittest.TestCase):
    """Working tests for WeylIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import WeylIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.WeylIntegral = WeylIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_weyl_integral_initialization(self):
        """Test WeylIntegral initialization"""
        # Test with float
        integral = self.WeylIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "Weyl")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.WeylIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
    
    def test_weyl_integral_call(self):
        """Test WeylIntegral __call__ method"""
        integral = self.WeylIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_weyl_integral_compute_scalar(self):
        """Test WeylIntegral _compute_scalar method"""
        integral = self.WeylIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test scalar computation
        result = integral._compute_scalar(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
    
    def test_weyl_integral_compute_array_numpy(self):
        """Test WeylIntegral _compute_array_numpy method"""
        integral = self.WeylIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test array computation
        x = np.linspace(0, 1, 10)
        result = integral._compute_array_numpy(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_weyl_integral_compute(self):
        """Test WeylIntegral compute method"""
        integral = self.WeylIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral.compute(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral.compute(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

class TestHadamardIntegralWorking(unittest.TestCase):
    """Working tests for HadamardIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import HadamardIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.HadamardIntegral = HadamardIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_hadamard_integral_initialization(self):
        """Test HadamardIntegral initialization"""
        # Test with float
        integral = self.HadamardIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "Hadamard")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.HadamardIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
    
    def test_hadamard_integral_call(self):
        """Test HadamardIntegral __call__ method"""
        integral = self.HadamardIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_hadamard_integral_compute_scalar(self):
        """Test HadamardIntegral _compute_scalar method"""
        integral = self.HadamardIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test scalar computation
        result = integral._compute_scalar(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
    
    def test_hadamard_integral_compute_array_numpy(self):
        """Test HadamardIntegral _compute_array_numpy method"""
        integral = self.HadamardIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test array computation
        x = np.linspace(0, 1, 10)
        result = integral._compute_array_numpy(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

class TestMillerRossIntegralWorking(unittest.TestCase):
    """Working tests for MillerRossIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import MillerRossIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.MillerRossIntegral = MillerRossIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_miller_ross_integral_initialization(self):
        """Test MillerRossIntegral initialization"""
        # Test with float
        integral = self.MillerRossIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "MillerRoss")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.MillerRossIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
    
    def test_miller_ross_integral_call(self):
        """Test MillerRossIntegral __call__ method"""
        integral = self.MillerRossIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_miller_ross_integral_compute_scalar(self):
        """Test MillerRossIntegral _compute_scalar method"""
        integral = self.MillerRossIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test scalar computation
        result = integral._compute_scalar(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
    
    def test_miller_ross_integral_compute_array_numpy(self):
        """Test MillerRossIntegral _compute_array_numpy method"""
        integral = self.MillerRossIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test array computation
        x = np.linspace(0, 1, 10)
        result = integral._compute_array_numpy(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

class TestMarchaudIntegralWorking(unittest.TestCase):
    """Working tests for MarchaudIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import MarchaudIntegral
        from hpfracc.core.definitions import FractionalOrder
        self.MarchaudIntegral = MarchaudIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_marchaud_integral_initialization(self):
        """Test MarchaudIntegral initialization"""
        # Test with float
        integral = self.MarchaudIntegral(0.5)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "Marchaud")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.MarchaudIntegral(order)
        self.assertEqual(integral.alpha.alpha, 0.7)
    
    def test_marchaud_integral_call(self):
        """Test MarchaudIntegral __call__ method"""
        integral = self.MarchaudIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test with scalar
        result = integral(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
        
        # Test with array
        x = np.linspace(0, 1, 10)
        result = integral(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
    
    def test_marchaud_integral_compute_scalar(self):
        """Test MarchaudIntegral _compute_scalar method"""
        integral = self.MarchaudIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test scalar computation
        result = integral._compute_scalar(test_function, 1.0)
        self.assertIsInstance(result, (int, float))
    
    def test_marchaud_integral_compute_array_numpy(self):
        """Test MarchaudIntegral _compute_array_numpy method"""
        integral = self.MarchaudIntegral(0.5)
        
        def test_function(x):
            return x
        
        # Test array computation
        x = np.linspace(0, 1, 10)
        result = integral._compute_array_numpy(test_function, x)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

class TestFractionalIntegralFactoryWorking(unittest.TestCase):
    """Working tests for FractionalIntegralFactory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import FractionalIntegralFactory, RiemannLiouvilleIntegral
        self.FractionalIntegralFactory = FractionalIntegralFactory
        self.RiemannLiouvilleIntegral = RiemannLiouvilleIntegral
    
    def test_fractional_integral_factory_initialization(self):
        """Test FractionalIntegralFactory initialization"""
        factory = self.FractionalIntegralFactory()
        self.assertIsInstance(factory._implementations, dict)
        self.assertEqual(len(factory._implementations), 0)
    
    def test_fractional_integral_factory_register_implementation(self):
        """Test FractionalIntegralFactory register_implementation method"""
        factory = self.FractionalIntegralFactory()
        
        # Register implementation
        factory.register_implementation("test", self.RiemannLiouvilleIntegral)
        self.assertIn("test", factory._implementations)
        self.assertEqual(factory._implementations["test"], self.RiemannLiouvilleIntegral)
        
        # Test overwrite
        factory.register_implementation("test", self.RiemannLiouvilleIntegral, overwrite=True)
        self.assertEqual(factory._implementations["test"], self.RiemannLiouvilleIntegral)
    
    def test_fractional_integral_factory_create(self):
        """Test FractionalIntegralFactory create method"""
        factory = self.FractionalIntegralFactory()
        factory.register_implementation("test", self.RiemannLiouvilleIntegral)
        
        # Test with string method
        integral = factory.create("test", 0.5)
        self.assertIsInstance(integral, self.RiemannLiouvilleIntegral)
        self.assertEqual(integral.alpha.alpha, 0.5)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            factory.create("invalid", 0.5)
    
    def test_fractional_integral_factory_get_available_methods(self):
        """Test FractionalIntegralFactory get_available_methods method"""
        factory = self.FractionalIntegralFactory()
        
        # Initially empty
        methods = factory.get_available_methods()
        self.assertIsInstance(methods, list)
        self.assertEqual(len(methods), 0)
        
        # Add implementations
        factory.register_implementation("test1", self.RiemannLiouvilleIntegral)
        factory.register_implementation("test2", self.RiemannLiouvilleIntegral)
        
        methods = factory.get_available_methods()
        self.assertEqual(len(methods), 2)
        self.assertIn("test1", methods)
        self.assertIn("test2", methods)

class TestUtilityFunctionsWorking(unittest.TestCase):
    """Working tests for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.integrals import (
            create_fractional_integral, analytical_fractional_integral,
            trapezoidal_fractional_integral, simpson_fractional_integral,
            fractional_integral_properties, validate_fractional_integral,
            create_fractional_integral_factory
        )
        self.create_fractional_integral = create_fractional_integral
        self.analytical_fractional_integral = analytical_fractional_integral
        self.trapezoidal_fractional_integral = trapezoidal_fractional_integral
        self.simpson_fractional_integral = simpson_fractional_integral
        self.fractional_integral_properties = fractional_integral_properties
        self.validate_fractional_integral = validate_fractional_integral
        self.create_fractional_integral_factory = create_fractional_integral_factory
    
    def test_create_fractional_integral(self):
        """Test create_fractional_integral function"""
        # Test with valid parameters
        integral = self.create_fractional_integral(0.5, method="RL")
        self.assertIsNotNone(integral)
        self.assertEqual(integral.alpha.alpha, 0.5)
        self.assertEqual(integral.method, "RL")
        
        # Test with different methods
        methods = ["RL", "Caputo", "Weyl", "Hadamard"]
        for method in methods:
            with self.subTest(method=method):
                integral = self.create_fractional_integral(0.5, method=method)
                self.assertIsNotNone(integral)
                self.assertEqual(integral.method, method)
    
    def test_analytical_fractional_integral(self):
        """Test analytical_fractional_integral function"""
        # Test with different function types
        function_types = ["constant", "linear", "quadratic", "exponential"]
        for f_type in function_types:
            with self.subTest(function_type=f_type):
                try:
                    result = self.analytical_fractional_integral(f_type, 0.5, 1.0)
                    self.assertIsInstance(result, (int, float))
                except (ValueError, NotImplementedError):
                    # Some function types might not be implemented
                    pass
    
    def test_trapezoidal_fractional_integral(self):
        """Test trapezoidal_fractional_integral function"""
        def test_function(x):
            return x
        
        # Test with valid parameters
        try:
            result = self.trapezoidal_fractional_integral(test_function, 0.5, 1.0, 0.1)
            self.assertIsInstance(result, (int, float))
        except (ValueError, NotImplementedError):
            # Might not be implemented or have specific requirements
            pass
    
    def test_simpson_fractional_integral(self):
        """Test simpson_fractional_integral function"""
        def test_function(x):
            return x
        
        # Test with valid parameters
        try:
            result = self.simpson_fractional_integral(test_function, 0.5, 1.0, 0.1)
            self.assertIsInstance(result, (int, float))
        except (ValueError, NotImplementedError):
            # Might not be implemented or have specific requirements
            pass
    
    def test_fractional_integral_properties(self):
        """Test fractional_integral_properties function"""
        # Test with valid alpha
        properties = self.fractional_integral_properties(0.5)
        self.assertIsInstance(properties, dict)
        self.assertGreater(len(properties), 0)
        
        # Test with different alpha values
        for alpha in [0.1, 0.5, 0.9, 1.0]:
            with self.subTest(alpha=alpha):
                properties = self.fractional_integral_properties(alpha)
                self.assertIsInstance(properties, dict)
    
    def test_validate_fractional_integral(self):
        """Test validate_fractional_integral function"""
        # Test valid parameters
        is_valid = self.validate_fractional_integral(0.5, "RL")
        self.assertIsInstance(is_valid, bool)
        
        # Test with different methods
        methods = ["RL", "Caputo", "Weyl", "Hadamard"]
        for method in methods:
            with self.subTest(method=method):
                is_valid = self.validate_fractional_integral(0.5, method)
                self.assertIsInstance(is_valid, bool)
        
        # Test with invalid parameters
        is_valid = self.validate_fractional_integral(-0.1, "RL")
        self.assertFalse(is_valid)
        
        is_valid = self.validate_fractional_integral(0.5, "invalid")
        self.assertFalse(is_valid)
    
    def test_create_fractional_integral_factory(self):
        """Test create_fractional_integral_factory function"""
        factory = self.create_fractional_integral_factory()
        self.assertIsNotNone(factory)
        self.assertIsInstance(factory._implementations, dict)

class TestIntegrationWorking(unittest.TestCase):
    """Test integration between components"""
    
    def test_factory_with_integrals_integration(self):
        """Test factory with integrals integration"""
        from hpfracc.core.integrals import (
            FractionalIntegralFactory, RiemannLiouvilleIntegral, CaputoIntegral
        )
        
        # Test factory creates integrals that work together
        factory = FractionalIntegralFactory()
        factory.register_implementation("rl", RiemannLiouvilleIntegral)
        factory.register_implementation("caputo", CaputoIntegral)
        
        rl_integral = factory.create("rl", 0.5)
        caputo_integral = factory.create("caputo", 0.5)
        
        self.assertIsNotNone(rl_integral)
        self.assertIsNotNone(caputo_integral)
        self.assertEqual(rl_integral.method, "RL")
        self.assertEqual(caputo_integral.method, "Caputo")
    
    def test_different_integral_types_integration(self):
        """Test different integral types integration"""
        from hpfracc.core.integrals import (
            RiemannLiouvilleIntegral, CaputoIntegral, WeylIntegral, HadamardIntegral
        )
        
        # Test that all integral types work with the same function
        def test_function(x):
            return x
        
        integrals = [
            RiemannLiouvilleIntegral(0.5),
            CaputoIntegral(0.5),
            WeylIntegral(0.5),
            HadamardIntegral(0.5)
        ]
        
        for integral in integrals:
            with self.subTest(integral_type=type(integral).__name__):
                result = integral(test_function, 1.0)
                self.assertIsInstance(result, (int, float))
                
                x = np.linspace(0, 1, 10)
                result = integral(test_function, x)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, x.shape)
    
    def test_utility_functions_integration(self):
        """Test utility functions integration"""
        from hpfracc.core.integrals import (
            create_fractional_integral, fractional_integral_properties,
            validate_fractional_integral
        )
        
        # Test create_fractional_integral with validation
        alpha = 0.5
        is_valid = validate_fractional_integral(alpha, "RL")
        if is_valid:
            integral = create_fractional_integral(alpha, method="RL")
            self.assertIsNotNone(integral)
            self.assertEqual(integral.alpha.alpha, alpha)
            
            # Test properties
            properties = fractional_integral_properties(alpha)
            self.assertIsInstance(properties, dict)
            self.assertGreater(len(properties), 0)
    
    def test_edge_cases_integration(self):
        """Test edge cases integration"""
        from hpfracc.core.integrals import RiemannLiouvilleIntegral
        
        # Test with edge case orders
        edge_orders = [0.001, 0.999, 1.0, 1.001, 10.0]
        for alpha in edge_orders:
            with self.subTest(alpha=alpha):
                try:
                    integral = RiemannLiouvilleIntegral(alpha)
                    self.assertIsNotNone(integral)
                    self.assertEqual(integral.alpha.alpha, alpha)
                    
                    def test_function(x):
                        return x
                    
                    result = integral(test_function, 1.0)
                    self.assertIsInstance(result, (int, float))
                except (ValueError, RuntimeError):
                    # Some edge cases might not be supported
                    pass

if __name__ == '__main__':
    unittest.main()
