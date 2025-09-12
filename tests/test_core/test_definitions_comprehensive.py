"""
Comprehensive tests for core definitions module.

This module tests all definition functionality including fractional orders,
definition types, and mathematical properties to ensure high coverage.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.core.definitions import (
    FractionalOrder, DefinitionType, FractionalDefinition,
    CaputoDefinition, RiemannLiouvilleDefinition, GrunwaldLetnikovDefinition,
    FractionalIntegral, FractionalCalculusProperties, get_available_definitions
)


class TestFractionalOrder:
    """Test FractionalOrder class."""
    
    def test_fractional_order_initialization(self):
        """Test FractionalOrder initialization."""
        order = FractionalOrder(0.5)
        assert order.alpha == 0.5
        assert order.is_integer is False
        assert order.is_fractional is True
    
    def test_fractional_order_from_another_order(self):
        """Test FractionalOrder initialization from another FractionalOrder."""
        original = FractionalOrder(0.3)
        order = FractionalOrder(original)
        assert order.alpha == 0.3
    
    def test_fractional_order_validation(self):
        """Test FractionalOrder validation."""
        # Valid orders
        for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
            order = FractionalOrder(alpha)
            assert order.alpha == alpha
        
        # Invalid orders
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            FractionalOrder(float('inf'))
        
        with pytest.raises(ValueError, match="Fractional order must be finite"):
            FractionalOrder(float('nan'))
        
        with pytest.raises(ValueError, match="Fractional order must be non-negative"):
            FractionalOrder(-0.1)
    
    def test_fractional_order_no_validation(self):
        """Test FractionalOrder without validation."""
        # Should not raise error even for invalid values
        order = FractionalOrder(-0.1, validate=False)
        assert order.alpha == -0.1
    
    def test_fractional_order_properties(self):
        """Test FractionalOrder properties."""
        # Integer order
        order = FractionalOrder(1.0)
        assert order.is_integer is True
        assert order.is_fractional is False
        
        # Fractional order
        order = FractionalOrder(0.5)
        assert order.is_integer is False
        assert order.is_fractional is True
        
        # Zero order
        order = FractionalOrder(0.0)
        assert order.is_integer is True
        assert order.is_fractional is False
    
    def test_fractional_order_string_representation(self):
        """Test FractionalOrder string representation."""
        order = FractionalOrder(0.5)
        assert "0.5" in str(order)
        assert "FractionalOrder" in repr(order)
    
    def test_fractional_order_equality(self):
        """Test FractionalOrder equality."""
        order1 = FractionalOrder(0.5)
        order2 = FractionalOrder(0.5)
        order3 = FractionalOrder(0.3)
        
        assert order1 == order2
        assert order1 != order3
        assert order1 != 0.5  # Different types
    
    def test_fractional_order_hash(self):
        """Test FractionalOrder hash."""
        order1 = FractionalOrder(0.5)
        order2 = FractionalOrder(0.5)
        
        assert hash(order1) == hash(order2)
        
        # Should be usable as dictionary key
        d = {order1: "test"}
        assert d[order2] == "test"


class TestDefinitionType:
    """Test DefinitionType enum."""
    
    def test_definition_type_values(self):
        """Test DefinitionType enum values."""
        assert DefinitionType.CAPUTO.value == "caputo"
        assert DefinitionType.RIEMANN_LIOUVILLE.value == "riemann_liouville"
        assert DefinitionType.GRUNWALD_LETNIKOV.value == "grunwald_letnikov"
    
    def test_definition_type_enumeration(self):
        """Test DefinitionType enumeration."""
        types = list(DefinitionType)
        assert len(types) == 6
        assert DefinitionType.CAPUTO in types
        assert DefinitionType.RIEMANN_LIOUVILLE in types
        assert DefinitionType.GRUNWALD_LETNIKOV in types
        assert DefinitionType.MILLER_ROSS in types
        assert DefinitionType.WEYL in types
        assert DefinitionType.MARCHAUD in types


class TestFractionalDefinition:
    """Test FractionalDefinition base class."""
    
    def test_fractional_definition_initialization(self):
        """Test FractionalDefinition initialization."""
        definition = FractionalDefinition(
            FractionalOrder(0.5),
            DefinitionType.CAPUTO
        )
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.CAPUTO
    
    def test_fractional_definition_properties(self):
        """Test FractionalDefinition properties."""
        definition = FractionalDefinition(
            FractionalOrder(0.5),
            DefinitionType.CAPUTO
        )
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.CAPUTO
    
    def test_fractional_definition_string_representation(self):
        """Test FractionalDefinition string representation."""
        definition = FractionalDefinition(
            FractionalOrder(0.5),
            DefinitionType.CAPUTO
        )
        assert "0.5" in str(definition)
        assert "caputo" in str(definition).lower()


class TestCaputoDefinition:
    """Test CaputoDefinition class."""
    
    def test_caputo_definition_initialization(self):
        """Test CaputoDefinition initialization."""
        definition = CaputoDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.CAPUTO
    
    def test_caputo_definition_properties(self):
        """Test CaputoDefinition properties."""
        definition = CaputoDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.CAPUTO
    
    def test_caputo_definition_different_orders(self):
        """Test CaputoDefinition with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            definition = CaputoDefinition(FractionalOrder(alpha))
            assert definition.order.alpha == alpha


class TestRiemannLiouvilleDefinition:
    """Test RiemannLiouvilleDefinition class."""
    
    def test_riemann_liouville_definition_initialization(self):
        """Test RiemannLiouvilleDefinition initialization."""
        definition = RiemannLiouvilleDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.RIEMANN_LIOUVILLE
    
    def test_riemann_liouville_definition_properties(self):
        """Test RiemannLiouvilleDefinition properties."""
        definition = RiemannLiouvilleDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.RIEMANN_LIOUVILLE
    
    def test_riemann_liouville_definition_different_orders(self):
        """Test RiemannLiouvilleDefinition with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            definition = RiemannLiouvilleDefinition(FractionalOrder(alpha))
            assert definition.order.alpha == alpha


class TestGrunwaldLetnikovDefinition:
    """Test GrunwaldLetnikovDefinition class."""
    
    def test_grunwald_letnikov_definition_initialization(self):
        """Test GrunwaldLetnikovDefinition initialization."""
        definition = GrunwaldLetnikovDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.GRUNWALD_LETNIKOV
    
    def test_grunwald_letnikov_definition_properties(self):
        """Test GrunwaldLetnikovDefinition properties."""
        definition = GrunwaldLetnikovDefinition(FractionalOrder(0.5))
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.GRUNWALD_LETNIKOV
    
    def test_grunwald_letnikov_definition_different_orders(self):
        """Test GrunwaldLetnikovDefinition with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            definition = GrunwaldLetnikovDefinition(FractionalOrder(alpha))
            assert definition.order.alpha == alpha


class TestFractionalIntegral:
    """Test FractionalIntegral class."""
    
    def test_fractional_integral_initialization(self):
        """Test FractionalIntegral initialization."""
        integral = FractionalIntegral(FractionalOrder(0.5))
        assert integral.alpha.alpha == 0.5
    
    def test_fractional_integral_properties(self):
        """Test FractionalIntegral properties."""
        integral = FractionalIntegral(FractionalOrder(0.5))
        assert integral.alpha.alpha == 0.5
    
    def test_fractional_integral_different_orders(self):
        """Test FractionalIntegral with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]:
            integral = FractionalIntegral(FractionalOrder(alpha))
            assert integral.alpha.alpha == alpha
    
    def test_fractional_integral_string_representation(self):
        """Test FractionalIntegral string representation."""
        integral = FractionalIntegral(FractionalOrder(0.5))
        assert "0.5" in str(integral)
        assert "FractionalIntegral" in repr(integral)


class TestFractionalCalculusProperties:
    """Test FractionalCalculusProperties class."""
    
    def test_fractional_calculus_properties_initialization(self):
        """Test FractionalCalculusProperties initialization."""
        properties = FractionalCalculusProperties()
        assert properties is not None
    
    def test_fractional_calculus_properties_methods(self):
        """Test FractionalCalculusProperties methods."""
        properties = FractionalCalculusProperties()
        
        # Test that methods exist and can be called
        assert hasattr(properties, 'get_properties')
        assert hasattr(properties, 'get_definition_properties')
        assert hasattr(properties, 'get_integral_properties')
    
    def test_fractional_calculus_properties_get_properties(self):
        """Test get_properties method."""
        properties = FractionalCalculusProperties()
        props = properties.get_properties()
        
        assert isinstance(props, dict)
        assert 'definitions' in props
        assert 'integrals' in props
    
    def test_fractional_calculus_properties_get_definition_properties(self):
        """Test get_definition_properties method."""
        properties = FractionalCalculusProperties()
        props = properties.get_definition_properties()
        
        assert isinstance(props, dict)
        assert 'caputo' in props
        assert 'riemann_liouville' in props
        assert 'grunwald_letnikov' in props
    
    def test_fractional_calculus_properties_get_integral_properties(self):
        """Test get_integral_properties method."""
        properties = FractionalCalculusProperties()
        props = properties.get_integral_properties()
        
        assert isinstance(props, dict)
        assert 'riemann_liouville' in props
        assert 'caputo' in props


class TestGetAvailableDefinitions:
    """Test get_available_definitions function."""
    
    def test_get_available_definitions(self):
        """Test get_available_definitions function."""
        definitions = get_available_definitions()
        
        assert isinstance(definitions, list)
        assert len(definitions) > 0
        
        # Check that all expected definitions are present
        expected = ['caputo', 'riemann_liouville', 'grunwald_letnikov']
        for expected_def in expected:
            assert expected_def in definitions
    
    def test_get_available_definitions_content(self):
        """Test get_available_definitions content."""
        definitions = get_available_definitions()
        
        # Should contain all definition types
        assert 'caputo' in definitions
        assert 'riemann_liouville' in definitions
        assert 'grunwald_letnikov' in definitions
        
        # Should not contain duplicates
        assert len(definitions) == len(set(definitions))


class TestDefinitionsIntegration:
    """Test definitions integration scenarios."""
    
    def test_definition_creation_workflow(self):
        """Test complete definition creation workflow."""
        # Create fractional order
        order = FractionalOrder(0.5)
        assert order.alpha == 0.5
        
        # Create definition
        definition = CaputoDefinition(order)
        assert definition.order.alpha == 0.5
        assert definition.definition_type == DefinitionType.CAPUTO
        
        # Create integral
        integral = FractionalIntegral(order)
        assert integral.alpha.alpha == 0.5
    
    def test_definition_properties_workflow(self):
        """Test definition properties workflow."""
        properties = FractionalCalculusProperties()
        
        # Get all properties
        all_props = properties.get_properties()
        assert isinstance(all_props, dict)
        
        # Get definition properties
        def_props = properties.get_definition_properties()
        assert isinstance(def_props, dict)
        
        # Get integral properties
        int_props = properties.get_integral_properties()
        assert isinstance(int_props, dict)
    
    def test_definition_enumeration_workflow(self):
        """Test definition enumeration workflow."""
        # Get available definitions
        available = get_available_definitions()
        assert isinstance(available, list)
        
        # Check against enum values
        enum_values = [dt.value for dt in DefinitionType]
        for value in enum_values:
            assert value in available


class TestDefinitionsEdgeCases:
    """Test definitions edge cases and error handling."""
    
    def test_fractional_order_edge_cases(self):
        """Test FractionalOrder edge cases."""
        # Very small positive number
        order = FractionalOrder(1e-10)
        assert order.alpha == 1e-10
        
        # Very large number
        order = FractionalOrder(1e10)
        assert order.alpha == 1e10
        
        # Exactly zero
        order = FractionalOrder(0.0)
        assert order.alpha == 0.0
        assert order.is_integer is True
    
    def test_definition_edge_cases(self):
        """Test definition edge cases."""
        # Very small order
        order = FractionalOrder(1e-10)
        definition = CaputoDefinition(order)
        assert definition.order.alpha == 1e-10
        
        # Very large order
        order = FractionalOrder(1e10)
        definition = CaputoDefinition(order)
        assert definition.order.alpha == 1e10
    
    def test_integral_edge_cases(self):
        """Test integral edge cases."""
        # Very small order
        order = FractionalOrder(1e-10)
        integral = FractionalIntegral(order)
        assert integral.alpha.alpha == 1e-10
        
        # Very large order
        order = FractionalOrder(1e10)
        integral = FractionalIntegral(order)
        assert integral.alpha.alpha == 1e10
    
    def test_properties_edge_cases(self):
        """Test properties edge cases."""
        properties = FractionalCalculusProperties()
        
        # Should handle empty or missing data gracefully
        props = properties.get_properties()
        assert isinstance(props, dict)
        
        # Should return consistent structure
        def_props = properties.get_definition_properties()
        int_props = properties.get_integral_properties()
        
        assert isinstance(def_props, dict)
        assert isinstance(int_props, dict)


class TestDefinitionsPerformance:
    """Test definitions performance scenarios."""
    
    def test_fractional_order_creation_performance(self):
        """Test FractionalOrder creation performance."""
        # Create many orders
        orders = [FractionalOrder(i * 0.1) for i in range(100)]
        assert len(orders) == 100
        
        # All should be valid
        for order in orders:
            assert order.alpha >= 0
    
    def test_definition_creation_performance(self):
        """Test definition creation performance."""
        # Create many definitions
        definitions = [CaputoDefinition(FractionalOrder(i * 0.1)) for i in range(100)]
        assert len(definitions) == 100
        
        # All should be valid
        for definition in definitions:
            assert definition.order.alpha >= 0
            assert definition.definition_type == DefinitionType.CAPUTO
    
    def test_properties_access_performance(self):
        """Test properties access performance."""
        properties = FractionalCalculusProperties()
        
        # Access properties many times
        for _ in range(100):
            props = properties.get_properties()
            def_props = properties.get_definition_properties()
            int_props = properties.get_integral_properties()
            
            assert isinstance(props, dict)
            assert isinstance(def_props, dict)
            assert isinstance(int_props, dict)


if __name__ == "__main__":
    pytest.main([__file__])
