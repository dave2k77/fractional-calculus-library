"""
Comprehensive unittest tests for HPFRACC core definitions
Targeting 100% coverage for hpfracc/core/definitions.py
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestFractionalOrderComprehensive(unittest.TestCase):
    """Comprehensive tests for FractionalOrder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalOrder
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_order_initialization_comprehensive(self):
        """Test comprehensive FractionalOrder initialization"""
        # Test with float
        order1 = self.FractionalOrder(0.5)
        self.assertEqual(order1.alpha, 0.5)
        self.assertIsNone(order1.method)
        
        # Test with int
        order2 = self.FractionalOrder(1)
        self.assertEqual(order2.alpha, 1.0)
        
        # Test with method
        order3 = self.FractionalOrder(0.7, method="caputo")
        self.assertEqual(order3.alpha, 0.7)
        self.assertEqual(order3.method, "caputo")
        
        # Test without validation
        order4 = self.FractionalOrder(-1.0, validate=False)
        self.assertEqual(order4.alpha, -1.0)
        
        # Test copy from another FractionalOrder
        original = self.FractionalOrder(0.8, method="riemann")
        copy = self.FractionalOrder(original)
        self.assertEqual(copy.alpha, 0.8)
        self.assertEqual(copy.method, "riemann")
        
        # Test copy with new method
        copy_new_method = self.FractionalOrder(original, method="grunwald")
        self.assertEqual(copy_new_method.alpha, 0.8)
        self.assertEqual(copy_new_method.method, "grunwald")
    
    def test_fractional_order_validation_comprehensive(self):
        """Test comprehensive validation scenarios"""
        # Test valid orders
        valid_orders = [0.0, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0, 10.0]
        for order_val in valid_orders:
            with self.subTest(order=order_val):
                order = self.FractionalOrder(order_val)
                self.assertEqual(order.alpha, order_val)
        
        # Test invalid orders
        invalid_orders = [np.inf, -np.inf, np.nan, -0.1, -1.0]
        for order_val in invalid_orders:
            with self.subTest(order=order_val):
                with self.assertRaises(ValueError):
                    self.FractionalOrder(order_val)
        
        # Test edge cases
        order_edge = self.FractionalOrder(0.0)
        self.assertEqual(order_edge.alpha, 0.0)
        
        order_large = self.FractionalOrder(100.0)
        self.assertEqual(order_large.alpha, 100.0)
    
    def test_fractional_order_properties_comprehensive(self):
        """Test comprehensive property access"""
        # Test value property
        order = self.FractionalOrder(0.7)
        self.assertEqual(order.value, 0.7)
        
        # Test is_integer
        integer_order = self.FractionalOrder(2.0)
        self.assertTrue(integer_order.is_integer())
        
        fractional_order = self.FractionalOrder(0.7)
        self.assertFalse(fractional_order.is_integer())
        
        # Test is_fractional
        self.assertFalse(integer_order.is_fractional())
        self.assertTrue(fractional_order.is_fractional())
        
        # Test integer_part
        self.assertEqual(integer_order.integer_part(), 2)
        self.assertEqual(fractional_order.integer_part(), 0)
        
        order_1_5 = self.FractionalOrder(1.5)
        self.assertEqual(order_1_5.integer_part(), 1)
        
        # Test fractional_part
        self.assertEqual(fractional_order.fractional_part(), 0.7)
        self.assertEqual(order_1_5.fractional_part(), 0.5)
        self.assertEqual(integer_order.fractional_part(), 0.0)
    
    def test_fractional_order_string_representations(self):
        """Test string representations"""
        order = self.FractionalOrder(0.5)
        
        # Test __repr__
        repr_str = repr(order)
        self.assertIn("FractionalOrder", repr_str)
        self.assertIn("0.5", repr_str)
        
        # Test __str__
        str_repr = str(order)
        self.assertIn("α", str_repr)
        self.assertIn("0.5", str_repr)
        
        # Test with method
        order_with_method = self.FractionalOrder(0.7, method="caputo")
        str_with_method = str(order_with_method)
        self.assertIn("caputo", str_with_method)
    
    def test_fractional_order_comparison_comprehensive(self):
        """Test comprehensive comparison operations"""
        order1 = self.FractionalOrder(0.5)
        order2 = self.FractionalOrder(0.5)
        order3 = self.FractionalOrder(0.7)
        
        # Test equality
        self.assertEqual(order1, order2)
        self.assertNotEqual(order1, order3)
        
        # Test with floats
        self.assertEqual(order1.alpha, 0.5)
        self.assertNotEqual(order1.alpha, 0.7)
        
        # Test hash
        self.assertEqual(hash(order1), hash(order2))
        self.assertNotEqual(hash(order1), hash(order3))
        
        # Test in sets
        order_set = {order1, order2, order3}
        self.assertEqual(len(order_set), 2)  # order1 and order2 are equal
    
    def test_fractional_order_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test very small positive number
        small_order = self.FractionalOrder(1e-10)
        self.assertEqual(small_order.alpha, 1e-10)
        
        # Test very large number
        large_order = self.FractionalOrder(1e10)
        self.assertEqual(large_order.alpha, 1e10)
        
        # Test precision
        precise_order = self.FractionalOrder(0.123456789)
        self.assertAlmostEqual(precise_order.alpha, 0.123456789)
        
        # Test copy with no method
        original_no_method = self.FractionalOrder(0.5)
        copy_no_method = self.FractionalOrder(original_no_method)
        self.assertIsNone(copy_no_method.method)

class TestDefinitionTypeComprehensive(unittest.TestCase):
    """Comprehensive tests for DefinitionType enum"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import DefinitionType
        self.DefinitionType = DefinitionType
    
    def test_definition_type_values(self):
        """Test DefinitionType enum values"""
        self.assertEqual(self.DefinitionType.RIEMANN_LIOUVILLE.value, "riemann_liouville")
        self.assertEqual(self.DefinitionType.CAPUTO.value, "caputo")
        self.assertEqual(self.DefinitionType.GRUNWALD_LETNIKOV.value, "grunwald_letnikov")
        self.assertEqual(self.DefinitionType.HADAMARD.value, "hadamard")
        self.assertEqual(self.DefinitionType.CAPUTO_FABRIZIO.value, "caputo_fabrizio")
    
    def test_definition_type_iteration(self):
        """Test iterating over DefinitionType"""
        types = list(self.DefinitionType)
        self.assertGreater(len(types), 0)
        self.assertIn(self.DefinitionType.RIEMANN_LIOUVILLE, types)
        self.assertIn(self.DefinitionType.CAPUTO, types)

class TestFractionalDefinitionComprehensive(unittest.TestCase):
    """Comprehensive tests for FractionalDefinition base class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalDefinition, FractionalOrder
        self.FractionalDefinition = FractionalDefinition
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_definition_initialization(self):
        """Test FractionalDefinition initialization"""
        # Test with float
        definition = self.FractionalDefinition(0.5)
        self.assertEqual(definition.order.alpha, 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        definition = self.FractionalDefinition(order)
        self.assertEqual(definition.order.alpha, 0.7)
        
        # Test with method
        definition = self.FractionalDefinition(0.5, method="test_method")
        self.assertEqual(definition.order.alpha, 0.5)
        self.assertEqual(definition.method, "test_method")
    
    def test_fractional_definition_methods(self):
        """Test FractionalDefinition methods"""
        definition = self.FractionalDefinition(0.5)
        
        # Test get_definition_formula
        formula = definition.get_definition_formula()
        self.assertIsInstance(formula, str)
        self.assertGreater(len(formula), 0)
        
        # Test get_properties
        properties = definition.get_properties()
        self.assertIsInstance(properties, dict)
        self.assertIn("order", properties)
        self.assertIn("method", properties)
        
        # Test __repr__
        repr_str = repr(definition)
        self.assertIn("FractionalDefinition", repr_str)
        self.assertIn("0.5", repr_str)

class TestCaputoDefinitionComprehensive(unittest.TestCase):
    """Comprehensive tests for CaputoDefinition class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import CaputoDefinition, FractionalOrder
        self.CaputoDefinition = CaputoDefinition
        self.FractionalOrder = FractionalOrder
    
    def test_caputo_definition_initialization(self):
        """Test CaputoDefinition initialization"""
        # Test with float
        caputo = self.CaputoDefinition(0.5)
        self.assertEqual(caputo.alpha, 0.5)
        self.assertEqual(caputo.definition_type.value, "caputo")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        caputo = self.CaputoDefinition(order)
        self.assertEqual(caputo.alpha, 0.7)
    
    def test_caputo_definition_properties(self):
        """Test CaputoDefinition properties"""
        caputo = self.CaputoDefinition(0.5)
        
        # Test n property
        n = caputo.n
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 1)
        
        # Test get_advantages
        advantages = caputo.get_advantages()
        self.assertIsInstance(advantages, list)
        self.assertGreater(len(advantages), 0)
        
        # Test get_limitations
        limitations = caputo.get_limitations()
        self.assertIsInstance(limitations, list)
        self.assertGreater(len(limitations), 0)
    
    def test_caputo_definition_edge_cases(self):
        """Test CaputoDefinition edge cases"""
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                caputo = self.CaputoDefinition(alpha)
                self.assertEqual(caputo.alpha, alpha)
                self.assertGreaterEqual(caputo.n, 1)

class TestRiemannLiouvilleDefinitionComprehensive(unittest.TestCase):
    """Comprehensive tests for RiemannLiouvilleDefinition class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import RiemannLiouvilleDefinition, FractionalOrder
        self.RiemannLiouvilleDefinition = RiemannLiouvilleDefinition
        self.FractionalOrder = FractionalOrder
    
    def test_riemann_liouville_definition_initialization(self):
        """Test RiemannLiouvilleDefinition initialization"""
        # Test with float
        rl = self.RiemannLiouvilleDefinition(0.5)
        self.assertEqual(rl.alpha, 0.5)
        self.assertEqual(rl.definition_type.value, "riemann_liouville")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        rl = self.RiemannLiouvilleDefinition(order)
        self.assertEqual(rl.alpha, 0.7)
    
    def test_riemann_liouville_definition_properties(self):
        """Test RiemannLiouvilleDefinition properties"""
        rl = self.RiemannLiouvilleDefinition(0.5)
        
        # Test n property
        n = rl.n
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 1)
        
        # Test get_advantages
        advantages = rl.get_advantages()
        self.assertIsInstance(advantages, list)
        self.assertGreater(len(advantages), 0)
        
        # Test get_limitations
        limitations = rl.get_limitations()
        self.assertIsInstance(limitations, list)
        self.assertGreater(len(limitations), 0)
    
    def test_riemann_liouville_definition_edge_cases(self):
        """Test RiemannLiouvilleDefinition edge cases"""
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                rl = self.RiemannLiouvilleDefinition(alpha)
                self.assertEqual(rl.alpha, alpha)
                self.assertGreaterEqual(rl.n, 1)

class TestGrunwaldLetnikovDefinitionComprehensive(unittest.TestCase):
    """Comprehensive tests for GrunwaldLetnikovDefinition class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import GrunwaldLetnikovDefinition, FractionalOrder
        self.GrunwaldLetnikovDefinition = GrunwaldLetnikovDefinition
        self.FractionalOrder = FractionalOrder
    
    def test_grunwald_letnikov_definition_initialization(self):
        """Test GrunwaldLetnikovDefinition initialization"""
        # Test with float
        gl = self.GrunwaldLetnikovDefinition(0.5)
        self.assertEqual(gl.alpha, 0.5)
        self.assertEqual(gl.definition_type.value, "grunwald_letnikov")
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        gl = self.GrunwaldLetnikovDefinition(order)
        self.assertEqual(gl.alpha, 0.7)
    
    def test_grunwald_letnikov_definition_properties(self):
        """Test GrunwaldLetnikovDefinition properties"""
        gl = self.GrunwaldLetnikovDefinition(0.5)
        
        # Test get_advantages
        advantages = gl.get_advantages()
        self.assertIsInstance(advantages, list)
        self.assertGreater(len(advantages), 0)
        
        # Test get_limitations
        limitations = gl.get_limitations()
        self.assertIsInstance(limitations, list)
        self.assertGreater(len(limitations), 0)
    
    def test_grunwald_letnikov_definition_edge_cases(self):
        """Test GrunwaldLetnikovDefinition edge cases"""
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                gl = self.GrunwaldLetnikovDefinition(alpha)
                self.assertEqual(gl.alpha, alpha)

class TestFractionalIntegralComprehensive(unittest.TestCase):
    """Comprehensive tests for FractionalIntegral class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalIntegral, FractionalOrder
        self.FractionalIntegral = FractionalIntegral
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_integral_initialization(self):
        """Test FractionalIntegral initialization"""
        # Test with float
        integral = self.FractionalIntegral(0.5)
        self.assertEqual(integral.alpha, 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        integral = self.FractionalIntegral(order)
        self.assertEqual(integral.alpha, 0.7)
    
    def test_fractional_integral_methods(self):
        """Test FractionalIntegral methods"""
        integral = self.FractionalIntegral(0.5)
        
        # Test get_formula
        formula = integral.get_formula()
        self.assertIsInstance(formula, str)
        self.assertGreater(len(formula), 0)
        
        # Test get_properties
        properties = integral.get_properties()
        self.assertIsInstance(properties, dict)
        self.assertIn("order", properties)
        self.assertIn("type", properties)
        
        # Test __str__
        str_repr = str(integral)
        self.assertIn("Fractional Integral", str_repr)
    
    def test_fractional_integral_edge_cases(self):
        """Test FractionalIntegral edge cases"""
        # Test with different orders
        for alpha in [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
            with self.subTest(alpha=alpha):
                integral = self.FractionalIntegral(alpha)
                self.assertEqual(integral.alpha, alpha)

class TestFractionalCalculusPropertiesComprehensive(unittest.TestCase):
    """Comprehensive tests for FractionalCalculusProperties class"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import FractionalCalculusProperties
        self.FractionalCalculusProperties = FractionalCalculusProperties
    
    def test_static_properties(self):
        """Test static property methods"""
        # Test linearity property
        linearity = self.FractionalCalculusProperties.linearity_property()
        self.assertIsInstance(linearity, str)
        self.assertGreater(len(linearity), 0)
        
        # Test semigroup property
        semigroup = self.FractionalCalculusProperties.semigroup_property()
        self.assertIsInstance(semigroup, str)
        self.assertGreater(len(semigroup), 0)
        
        # Test Leibniz rule
        leibniz = self.FractionalCalculusProperties.leibniz_rule()
        self.assertIsInstance(leibniz, str)
        self.assertGreater(len(leibniz), 0)
        
        # Test chain rule
        chain = self.FractionalCalculusProperties.chain_rule()
        self.assertIsInstance(chain, str)
        self.assertGreater(len(chain), 0)
    
    def test_relationship_methods(self):
        """Test relationship methods"""
        # Test relationship between definitions
        relationships = self.FractionalCalculusProperties.relationship_between_definitions()
        self.assertIsInstance(relationships, dict)
        self.assertGreater(len(relationships), 0)
        
        # Test analytical solutions
        solutions = self.FractionalCalculusProperties.get_analytical_solutions()
        self.assertIsInstance(solutions, dict)
        self.assertGreater(len(solutions), 0)
    
    def test_property_methods(self):
        """Test property methods"""
        props = self.FractionalCalculusProperties()
        
        # Test get_properties
        properties = props.get_properties()
        self.assertIsInstance(properties, dict)
        self.assertGreater(len(properties), 0)
        
        # Test get_definition_properties
        def_props = props.get_definition_properties()
        self.assertIsInstance(def_props, dict)
        self.assertGreater(len(def_props), 0)
        
        # Test get_integral_properties
        int_props = props.get_integral_properties()
        self.assertIsInstance(int_props, dict)
        self.assertGreater(len(int_props), 0)

class TestUtilityFunctionsComprehensive(unittest.TestCase):
    """Comprehensive tests for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.core.definitions import (
            create_definition, get_available_definitions, validate_fractional_order,
            FractionalOrder
        )
        self.create_definition = create_definition
        self.get_available_definitions = get_available_definitions
        self.validate_fractional_order = validate_fractional_order
        self.FractionalOrder = FractionalOrder
    
    def test_create_definition_comprehensive(self):
        """Test comprehensive create_definition function"""
        # Test with valid definition types
        valid_types = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        for def_type in valid_types:
            with self.subTest(definition_type=def_type):
                definition = self.create_definition(def_type, 0.5)
                self.assertIsNotNone(definition)
                self.assertEqual(definition.alpha, 0.5)
        
        # Test with invalid definition type
        with self.assertRaises(ValueError):
            self.create_definition("invalid_type", 0.5)
        
        # Test with FractionalOrder
        order = self.FractionalOrder(0.7)
        definition = self.create_definition("caputo", order)
        self.assertEqual(definition.alpha, 0.7)
    
    def test_get_available_definitions(self):
        """Test get_available_definitions function"""
        definitions = self.get_available_definitions()
        self.assertIsInstance(definitions, list)
        self.assertGreater(len(definitions), 0)
        
        # Check that common definitions are available
        expected_definitions = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        for expected in expected_definitions:
            self.assertIn(expected, definitions)
    
    def test_validate_fractional_order_comprehensive(self):
        """Test comprehensive validate_fractional_order function"""
        # Test valid orders
        valid_orders = [0.0, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0]
        for order_val in valid_orders:
            with self.subTest(order=order_val):
                self.assertTrue(self.validate_fractional_order(order_val))
        
        # Test invalid orders
        invalid_orders = [np.inf, -np.inf, np.nan, -0.1, -1.0]
        for order_val in invalid_orders:
            with self.subTest(order=order_val):
                self.assertFalse(self.validate_fractional_order(order_val))
        
        # Test with custom range
        self.assertTrue(self.validate_fractional_order(0.5, min_val=0.0, max_val=1.0))
        self.assertFalse(self.validate_fractional_order(1.5, min_val=0.0, max_val=1.0))
        self.assertFalse(self.validate_fractional_order(-0.1, min_val=0.0, max_val=1.0))

class TestIntegrationAndEdgeCases(unittest.TestCase):
    """Test integration between components and edge cases"""
    
    def test_fractional_order_with_definitions(self):
        """Test FractionalOrder integration with definitions"""
        from hpfracc.core.definitions import (
            FractionalOrder, CaputoDefinition, RiemannLiouvilleDefinition, 
            GrunwaldLetnikovDefinition
        )
        
        order = FractionalOrder(0.5)
        
        # Test with different definitions
        caputo = CaputoDefinition(order)
        rl = RiemannLiouvilleDefinition(order)
        gl = GrunwaldLetnikovDefinition(order)
        
        self.assertEqual(caputo.alpha, 0.5)
        self.assertEqual(rl.alpha, 0.5)
        self.assertEqual(gl.alpha, 0.5)
    
    def test_definition_copying_and_methods(self):
        """Test definition copying and method preservation"""
        from hpfracc.core.definitions import FractionalOrder, CaputoDefinition
        
        original_order = FractionalOrder(0.5, method="test_method")
        copy_order = FractionalOrder(original_order)
        
        self.assertEqual(copy_order.alpha, 0.5)
        self.assertEqual(copy_order.method, "test_method")
        
        # Test with new method
        new_method_order = FractionalOrder(original_order, method="new_method")
        self.assertEqual(new_method_order.alpha, 0.5)
        self.assertEqual(new_method_order.method, "new_method")
    
    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases"""
        from hpfracc.core.definitions import FractionalOrder, FractionalIntegral
        
        # Test very small positive numbers
        small_order = FractionalOrder(1e-15)
        self.assertEqual(small_order.alpha, 1e-15)
        
        # Test very large numbers
        large_order = FractionalOrder(1e15)
        self.assertEqual(large_order.alpha, 1e15)
        
        # Test precision handling
        precise_order = FractionalOrder(0.123456789012345)
        self.assertAlmostEqual(precise_order.alpha, 0.123456789012345)
        
        # Test integral with edge cases
        for alpha in [0.0, 1e-10, 1e10]:
            with self.subTest(alpha=alpha):
                integral = FractionalIntegral(alpha)
                self.assertEqual(integral.alpha, alpha)

if __name__ == '__main__':
    unittest.main()
