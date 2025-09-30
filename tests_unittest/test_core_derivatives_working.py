"""
Working unittest tests for HPFRACC core derivatives
Targeting high coverage for hpfracc/core/derivatives.py
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class MockFractionalDerivative(unittest.TestCase):
    """Mock implementation for testing BaseFractionalDerivative"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import BaseFractionalDerivative
        from hpfracc.core.definitions import FractionalOrder, DefinitionType, FractionalDefinition
        
        class MockDerivative(BaseFractionalDerivative):
            """Mock implementation for testing"""
            
            def compute(self, function, x, **kwargs):
                """Mock compute method"""
                return np.zeros_like(x)
            
            def compute_numerical(self, function, x, **kwargs):
                """Mock numerical compute method"""
                return np.zeros_like(x)
        
        self.MockDerivative = MockDerivative
        self.FractionalOrder = FractionalOrder
        self.DefinitionType = DefinitionType
        self.FractionalDefinition = FractionalDefinition

class TestBaseFractionalDerivativeWorking(unittest.TestCase):
    """Working tests for BaseFractionalDerivative class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import BaseFractionalDerivative
        from hpfracc.core.definitions import FractionalOrder, DefinitionType, FractionalDefinition
        
        class MockDerivative(BaseFractionalDerivative):
            """Mock implementation for testing"""
            
            def compute(self, function, x, **kwargs):
                """Mock compute method"""
                return np.zeros_like(x)
            
            def compute_numerical(self, function, x, **kwargs):
                """Mock numerical compute method"""
                return np.zeros_like(x)
        
        self.MockDerivative = MockDerivative
        self.FractionalOrder = FractionalOrder
        self.DefinitionType = DefinitionType
        self.FractionalDefinition = FractionalDefinition
    
    def test_base_fractional_derivative_initialization(self):
        """Test BaseFractionalDerivative initialization"""
        # Test with float
        derivative = self.MockDerivative(0.5)
        self.assertEqual(derivative.alpha.alpha, 0.5)
        self.assertEqual(derivative._alpha_value, 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.MockDerivative(order)
        self.assertEqual(derivative.alpha.alpha, 0.7)
        self.assertEqual(derivative._alpha_value, 0.7)
        
        # Test with definition
        definition = self.FractionalDefinition(DefinitionType.CAPUTO, 0.5)
        derivative = self.MockDerivative(0.5, definition=definition)
        self.assertEqual(derivative.definition, definition)
        
        # Test with JAX
        derivative = self.MockDerivative(0.5, use_jax=True)
        self.assertTrue(derivative.use_jax)
        
        # Test with NUMBA
        derivative = self.MockDerivative(0.5, use_numba=False)
        self.assertFalse(derivative.use_numba)
    
    def test_base_fractional_derivative_compute(self):
        """Test BaseFractionalDerivative compute method"""
        derivative = self.MockDerivative(0.5)
        
        def test_function(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = derivative.compute(test_function, x)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.allclose(result, 0))  # Mock returns zeros
    
    def test_base_fractional_derivative_compute_numerical(self):
        """Test BaseFractionalDerivative compute_numerical method"""
        derivative = self.MockDerivative(0.5)
        
        def test_function(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = derivative.compute_numerical(test_function, x)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.allclose(result, 0))  # Mock returns zeros
    
    def test_base_fractional_derivative_get_definition_info(self):
        """Test BaseFractionalDerivative get_definition_info method"""
        derivative = self.MockDerivative(0.5)
        info = derivative.get_definition_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("alpha", info)
        self.assertIn("use_jax", info)
        self.assertIn("use_numba", info)
        self.assertEqual(info["alpha"], 0.5)
    
    def test_base_fractional_derivative_representations(self):
        """Test BaseFractionalDerivative string representations"""
        derivative = self.MockDerivative(0.5)
        
        # Test __repr__
        repr_str = repr(derivative)
        self.assertIn("MockDerivative", repr_str)
        self.assertIn("0.5", repr_str)
        
        # Test __str__
        str_repr = str(derivative)
        self.assertIn("MockDerivative", str_repr)
        self.assertIn("0.5", str_repr)
    
    def test_base_fractional_derivative_validation(self):
        """Test BaseFractionalDerivative validation"""
        # Test valid orders
        valid_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        for order_val in valid_orders:
            with self.subTest(order=order_val):
                derivative = self.MockDerivative(order_val)
                self.assertEqual(derivative.alpha.alpha, order_val)
        
        # Test invalid orders
        invalid_orders = [np.inf, -np.inf, np.nan, -0.1]
        for order_val in invalid_orders:
            with self.subTest(order=order_val):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.MockDerivative(order_val)

class TestFractionalDerivativeOperatorWorking(unittest.TestCase):
    """Working tests for FractionalDerivativeOperator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import FractionalDerivativeOperator, BaseFractionalDerivative
        from hpfracc.core.definitions import FractionalOrder
        
        class MockDerivative(BaseFractionalDerivative):
            """Mock implementation for testing"""
            
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        self.FractionalDerivativeOperator = FractionalDerivativeOperator
        self.MockDerivative = MockDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_derivative_operator_initialization(self):
        """Test FractionalDerivativeOperator initialization"""
        # Test with float
        operator = self.FractionalDerivativeOperator(0.5)
        self.assertEqual(operator.alpha.alpha, 0.5)
        self.assertIsNone(operator.implementation)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        operator = self.FractionalDerivativeOperator(order)
        self.assertEqual(operator.alpha.alpha, 0.7)
        
        # Test with implementation
        mock_impl = self.MockDerivative(0.5)
        operator = self.FractionalDerivativeOperator(0.5, implementation=mock_impl)
        self.assertEqual(operator.implementation, mock_impl)
    
    def test_fractional_derivative_operator_call(self):
        """Test FractionalDerivativeOperator __call__ method"""
        mock_impl = self.MockDerivative(0.5)
        operator = self.FractionalDerivativeOperator(0.5, implementation=mock_impl)
        
        def test_function(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = operator(test_function, x)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.allclose(result, 0))  # Mock returns zeros
    
    def test_fractional_derivative_operator_compute_numerical(self):
        """Test FractionalDerivativeOperator compute_numerical method"""
        mock_impl = self.MockDerivative(0.5)
        operator = self.FractionalDerivativeOperator(0.5, implementation=mock_impl)
        
        def test_function(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = operator.compute_numerical(test_function, x)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.allclose(result, 0))  # Mock returns zeros
    
    def test_fractional_derivative_operator_set_implementation(self):
        """Test FractionalDerivativeOperator set_implementation method"""
        operator = self.FractionalDerivativeOperator(0.5)
        mock_impl = self.MockDerivative(0.5)
        
        operator.set_implementation(mock_impl)
        self.assertEqual(operator.implementation, mock_impl)
    
    def test_fractional_derivative_operator_get_info(self):
        """Test FractionalDerivativeOperator get_info method"""
        operator = self.FractionalDerivativeOperator(0.5)
        info = operator.get_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("alpha", info)
        self.assertIn("implementation", info)
        self.assertEqual(info["alpha"], 0.5)
        self.assertIsNone(info["implementation"])

class TestFractionalDerivativeFactoryWorking(unittest.TestCase):
    """Working tests for FractionalDerivativeFactory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import FractionalDerivativeFactory, BaseFractionalDerivative
        from hpfracc.core.definitions import FractionalOrder, DefinitionType
        
        class MockDerivative(BaseFractionalDerivative):
            """Mock implementation for testing"""
            
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        self.FractionalDerivativeFactory = FractionalDerivativeFactory
        self.MockDerivative = MockDerivative
        self.FractionalOrder = FractionalOrder
        self.DefinitionType = DefinitionType
    
    def test_fractional_derivative_factory_initialization(self):
        """Test FractionalDerivativeFactory initialization"""
        factory = self.FractionalDerivativeFactory()
        self.assertIsInstance(factory._implementations, dict)
        self.assertEqual(len(factory._implementations), 0)
    
    def test_fractional_derivative_factory_register_implementation(self):
        """Test FractionalDerivativeFactory register_implementation method"""
        factory = self.FractionalDerivativeFactory()
        
        # Register implementation
        factory.register_implementation("test", self.MockDerivative)
        self.assertIn("test", factory._implementations)
        self.assertEqual(factory._implementations["test"], self.MockDerivative)
        
        # Test overwrite
        factory.register_implementation("test", self.MockDerivative, overwrite=True)
        self.assertEqual(factory._implementations["test"], self.MockDerivative)
        
        # Test no overwrite (should raise error)
        with self.assertRaises(ValueError):
            factory.register_implementation("test", self.MockDerivative, overwrite=False)
    
    def test_fractional_derivative_factory_create(self):
        """Test FractionalDerivativeFactory create method"""
        factory = self.FractionalDerivativeFactory()
        factory.register_implementation("test", self.MockDerivative)
        
        # Test with string definition type
        derivative = factory.create("test", 0.5)
        self.assertIsInstance(derivative, self.MockDerivative)
        self.assertEqual(derivative.alpha.alpha, 0.5)
        
        # Test with DefinitionType enum
        derivative = factory.create(self.DefinitionType.CAPUTO, 0.7)
        self.assertIsInstance(derivative, self.MockDerivative)
        self.assertEqual(derivative.alpha.alpha, 0.7)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.3)
        derivative = factory.create("test", order)
        self.assertIsInstance(derivative, self.MockDerivative)
        self.assertEqual(derivative.alpha.alpha, 0.3)
        
        # Test with invalid definition type
        with self.assertRaises(ValueError):
            factory.create("invalid", 0.5)
    
    def test_fractional_derivative_factory_get_available_implementations(self):
        """Test FractionalDerivativeFactory get_available_implementations method"""
        factory = self.FractionalDerivativeFactory()
        
        # Initially empty
        implementations = factory.get_available_implementations()
        self.assertIsInstance(implementations, list)
        self.assertEqual(len(implementations), 0)
        
        # Add implementations
        factory.register_implementation("test1", self.MockDerivative)
        factory.register_implementation("test2", self.MockDerivative)
        
        implementations = factory.get_available_implementations()
        self.assertEqual(len(implementations), 2)
        self.assertIn("test1", implementations)
        self.assertIn("test2", implementations)

class TestFractionalDerivativeChainWorking(unittest.TestCase):
    """Working tests for FractionalDerivativeChain class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import FractionalDerivativeChain, BaseFractionalDerivative
        from hpfracc.core.definitions import FractionalOrder
        
        class MockDerivative(BaseFractionalDerivative):
            """Mock implementation for testing"""
            
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        self.FractionalDerivativeChain = FractionalDerivativeChain
        self.MockDerivative = MockDerivative
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_derivative_chain_initialization(self):
        """Test FractionalDerivativeChain initialization"""
        # Create mock derivatives
        derivative1 = self.MockDerivative(0.5)
        derivative2 = self.MockDerivative(0.3)
        
        # Test with list of derivatives
        chain = self.FractionalDerivativeChain([derivative1, derivative2])
        self.assertEqual(len(chain.derivatives), 2)
        self.assertEqual(chain.derivatives[0], derivative1)
        self.assertEqual(chain.derivatives[1], derivative2)
        
        # Test with single derivative
        chain = self.FractionalDerivativeChain([derivative1])
        self.assertEqual(len(chain.derivatives), 1)
    
    def test_fractional_derivative_chain_compute(self):
        """Test FractionalDerivativeChain compute method"""
        derivative1 = self.MockDerivative(0.5)
        derivative2 = self.MockDerivative(0.3)
        chain = self.FractionalDerivativeChain([derivative1, derivative2])
        
        def test_function(x):
            return x**2
        
        x = np.linspace(0, 1, 10)
        result = chain.compute(test_function, x)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.allclose(result, 0))  # Mock returns zeros
    
    def test_fractional_derivative_chain_get_total_order(self):
        """Test FractionalDerivativeChain get_total_order method"""
        derivative1 = self.MockDerivative(0.5)
        derivative2 = self.MockDerivative(0.3)
        chain = self.FractionalDerivativeChain([derivative1, derivative2])
        
        total_order = chain.get_total_order()
        self.assertEqual(total_order, 0.8)  # 0.5 + 0.3
    
    def test_fractional_derivative_chain_get_chain_info(self):
        """Test FractionalDerivativeChain get_chain_info method"""
        derivative1 = self.MockDerivative(0.5)
        derivative2 = self.MockDerivative(0.3)
        chain = self.FractionalDerivativeChain([derivative1, derivative2])
        
        chain_info = chain.get_chain_info()
        self.assertIsInstance(chain_info, list)
        self.assertEqual(len(chain_info), 2)
        
        for info in chain_info:
            self.assertIsInstance(info, dict)
            self.assertIn("alpha", info)
    
    def test_fractional_derivative_chain_validation(self):
        """Test FractionalDerivativeChain validation"""
        # Test with empty list
        with self.assertRaises(ValueError):
            self.FractionalDerivativeChain([])
        
        # Test with valid derivatives
        derivative1 = self.MockDerivative(0.5)
        derivative2 = self.MockDerivative(0.3)
        chain = self.FractionalDerivativeChain([derivative1, derivative2])
        self.assertIsNotNone(chain)

class TestFractionalDerivativePropertiesWorking(unittest.TestCase):
    """Working tests for FractionalDerivativeProperties class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import FractionalDerivativeProperties
        self.FractionalDerivativeProperties = FractionalDerivativeProperties
    
    def test_fractional_derivative_properties_check_linearity(self):
        """Test FractionalDerivativeProperties check_linearity method"""
        properties = self.FractionalDerivativeProperties()
        
        def linear_function(x):
            return x
        
        def nonlinear_function(x):
            return x**2
        
        # Test linear function
        is_linear = properties.check_linearity(linear_function, alpha=0.5, tolerance=1e-6)
        self.assertIsInstance(is_linear, bool)
        
        # Test nonlinear function
        is_linear = properties.check_linearity(nonlinear_function, alpha=0.5, tolerance=1e-6)
        self.assertIsInstance(is_linear, bool)
    
    def test_fractional_derivative_properties_check_semigroup_property(self):
        """Test FractionalDerivativeProperties check_semigroup_property method"""
        properties = self.FractionalDerivativeProperties()
        
        def test_function(x):
            return x
        
        # Test semigroup property
        satisfies_semigroup = properties.check_semigroup_property(
            test_function, alpha=0.5, beta=0.3, tolerance=1e-6
        )
        self.assertIsInstance(satisfies_semigroup, bool)
    
    def test_fractional_derivative_properties_get_analytical_solutions(self):
        """Test FractionalDerivativeProperties get_analytical_solutions method"""
        properties = self.FractionalDerivativeProperties()
        solutions = properties.get_analytical_solutions()
        
        self.assertIsInstance(solutions, dict)
        self.assertGreater(len(solutions), 0)
        
        # Check that all values are callable
        for name, solution_func in solutions.items():
            self.assertIsInstance(solution_func, type(lambda x: x))
            self.assertTrue(callable(solution_func))

class TestUtilityFunctionsWorking(unittest.TestCase):
    """Working tests for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.derivatives import create_fractional_derivative, create_derivative_operator
        from hpfracc.core.definitions import FractionalOrder
        
        self.create_fractional_derivative = create_fractional_derivative
        self.create_derivative_operator = create_derivative_operator
        self.FractionalOrder = FractionalOrder
    
    def test_create_fractional_derivative(self):
        """Test create_fractional_derivative function"""
        # Test with valid parameters
        derivative = self.create_fractional_derivative(
            alpha=0.5,
            definition_type="caputo",
            use_jax=False,
            use_numba=True
        )
        self.assertIsNotNone(derivative)
        self.assertEqual(derivative.alpha.alpha, 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        derivative = self.create_fractional_derivative(
            alpha=order,
            definition_type="riemann_liouville",
            use_jax=True,
            use_numba=False
        )
        self.assertIsNotNone(derivative)
        self.assertEqual(derivative.alpha.alpha, 0.7)
    
    def test_create_derivative_operator(self):
        """Test create_derivative_operator function"""
        # Test with valid parameters
        operator = self.create_derivative_operator(
            alpha=0.5,
            definition_type="grunwald_letnikov",
            use_jax=False,
            use_numba=True
        )
        self.assertIsNotNone(operator)
        self.assertEqual(operator.alpha.alpha, 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.3)
        operator = self.create_derivative_operator(
            alpha=order,
            definition_type="caputo",
            use_jax=True,
            use_numba=False
        )
        self.assertIsNotNone(operator)
        self.assertEqual(operator.alpha.alpha, 0.3)

class TestIntegrationWorking(unittest.TestCase):
    """Test integration between components"""
    
    def test_factory_with_operator_integration(self):
        """Test factory with operator integration"""
        from hpfracc.core.derivatives import FractionalDerivativeFactory, FractionalDerivativeOperator, BaseFractionalDerivative
        
        class MockDerivative(BaseFractionalDerivative):
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        # Test factory creates derivatives that work with operators
        factory = FractionalDerivativeFactory()
        factory.register_implementation("test", MockDerivative)
        
        derivative = factory.create("test", 0.5)
        operator = FractionalDerivativeOperator(0.5, implementation=derivative)
        
        self.assertIsNotNone(operator)
        self.assertEqual(operator.implementation, derivative)
    
    def test_chain_with_factory_integration(self):
        """Test chain with factory integration"""
        from hpfracc.core.derivatives import FractionalDerivativeFactory, FractionalDerivativeChain, BaseFractionalDerivative
        
        class MockDerivative(BaseFractionalDerivative):
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        # Test factory creates derivatives that work in chains
        factory = FractionalDerivativeFactory()
        factory.register_implementation("test", MockDerivative)
        
        derivative1 = factory.create("test", 0.5)
        derivative2 = factory.create("test", 0.3)
        
        chain = FractionalDerivativeChain([derivative1, derivative2])
        
        self.assertIsNotNone(chain)
        self.assertEqual(len(chain.derivatives), 2)
        self.assertEqual(chain.get_total_order(), 0.8)
    
    def test_properties_with_derivatives_integration(self):
        """Test properties with derivatives integration"""
        from hpfracc.core.derivatives import FractionalDerivativeProperties, BaseFractionalDerivative
        
        class MockDerivative(BaseFractionalDerivative):
            def compute(self, function, x, **kwargs):
                return np.zeros_like(x)
            def compute_numerical(self, function, x, **kwargs):
                return np.zeros_like(x)
        
        # Test properties work with derivatives
        derivative = MockDerivative(0.5)
        properties = FractionalDerivativeProperties()
        
        def test_function(x):
            return x
        
        # Test linearity check
        is_linear = properties.check_linearity(test_function, alpha=0.5)
        self.assertIsInstance(is_linear, bool)
        
        # Test semigroup property
        satisfies_semigroup = properties.check_semigroup_property(test_function, alpha=0.5, beta=0.3)
        self.assertIsInstance(satisfies_semigroup, bool)
        
        # Test analytical solutions
        solutions = properties.get_analytical_solutions()
        self.assertIsInstance(solutions, dict)
        self.assertGreater(len(solutions), 0)

if __name__ == '__main__':
    unittest.main()
