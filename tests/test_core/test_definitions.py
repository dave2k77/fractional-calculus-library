#!/usr/bin/env python3
"""
Tests for core definitions module.

Tests the FractionalOrder, DefinitionType, and various derivative definitions.
"""

import pytest
import numpy as np
from src.core.definitions import (
    FractionalOrder,
    DefinitionType,
    CaputoDefinition,
    RiemannLiouvilleDefinition,
    GrunwaldLetnikovDefinition,
    FractionalDefinition,
    FractionalIntegral,
    FractionalCalculusProperties,
)


class TestFractionalOrder:
    """Test FractionalOrder class."""

    def test_fractional_order_creation(self):
        """Test creating FractionalOrder instances."""
        # Test integer order
        alpha_int = FractionalOrder(2)
        assert alpha_int.alpha == 2.0
        assert alpha_int.is_integer
        assert alpha_int.integer_part == 2
        assert alpha_int.fractional_part == 0.0

        # Test fractional order
        alpha_frac = FractionalOrder(0.5)
        assert alpha_frac.alpha == 0.5
        assert not alpha_frac.is_integer
        assert alpha_frac.integer_part == 0
        assert alpha_frac.fractional_part == 0.5

        # Test zero order
        alpha_zero = FractionalOrder(0.0)
        assert alpha_zero.alpha == 0.0
        assert alpha_zero.is_integer
        assert alpha_zero.integer_part == 0
        assert alpha_zero.fractional_part == 0.0

    def test_fractional_order_validation(self):
        """Test FractionalOrder validation."""
        # Test with valid inputs
        FractionalOrder(0.0)
        FractionalOrder(1.0)
        FractionalOrder(2.5)

        # Test with numpy types
        FractionalOrder(np.float64(0.5))
        FractionalOrder(np.int32(2))

        # Test that negative values raise ValueError
        with pytest.raises(ValueError):
            FractionalOrder(-1.0)

        with pytest.raises(ValueError):
            FractionalOrder(-0.5)

    def test_fractional_order_repr(self):
        """Test FractionalOrder string representation."""
        alpha = FractionalOrder(0.5)
        assert "FractionalOrder" in repr(alpha)
        assert "0.5" in repr(alpha)

    def test_fractional_order_equality(self):
        """Test FractionalOrder equality."""
        alpha1 = FractionalOrder(0.5)
        alpha2 = FractionalOrder(0.5)
        alpha3 = FractionalOrder(1.0)

        assert alpha1 == alpha2
        assert alpha1 != alpha3
        assert hash(alpha1) == hash(alpha2)


class TestCaputoDefinition:
    """Test CaputoDefinition class."""

    def test_caputo_definition_creation(self):
        """Test creating CaputoDefinition instances."""
        alpha = FractionalOrder(0.5)
        caputo = CaputoDefinition(alpha)

        assert caputo.alpha == alpha
        assert caputo.definition_type == DefinitionType.CAPUTO

        # Test with different orders
        caputo2 = CaputoDefinition(1.5)
        assert caputo2.alpha.alpha == 1.5

    def test_caputo_definition_formula(self):
        """Test Caputo definition formula."""
        caputo = CaputoDefinition(0.5)
        # The formula is returned by the base class method
        formula = caputo.get_definition_formula()

        # Check that it contains the mathematical formula
        assert "D^α" in formula
        assert "∫" in formula
        assert "f" in formula

    def test_caputo_advantages(self):
        """Test Caputo definition advantages."""
        caputo = CaputoDefinition(0.5)
        advantages = caputo.get_advantages()

        assert isinstance(advantages, list)
        assert len(advantages) > 0
        assert all(isinstance(adv, str) for adv in advantages)

    def test_caputo_limitations(self):
        """Test Caputo definition limitations."""
        caputo = CaputoDefinition(0.5)
        limitations = caputo.get_limitations()

        assert isinstance(limitations, list)
        assert len(limitations) > 0
        assert all(isinstance(lim, str) for lim in limitations)


class TestRiemannLiouvilleDefinition:
    """Test RiemannLiouvilleDefinition class."""

    def test_riemann_liouville_definition_creation(self):
        """Test creating RiemannLiouvilleDefinition instances."""
        alpha = FractionalOrder(0.5)
        rl = RiemannLiouvilleDefinition(alpha)

        assert rl.alpha == alpha
        assert rl.definition_type == DefinitionType.RIEMANN_LIOUVILLE

    def test_riemann_liouville_formula(self):
        """Test Riemann-Liouville definition formula."""
        rl = RiemannLiouvilleDefinition(0.5)
        formula = rl.get_definition_formula()

        # Check that it contains the mathematical formula
        assert "D^α" in formula
        assert "∫" in formula
        assert "f" in formula

    def test_riemann_liouville_advantages(self):
        """Test Riemann-Liouville definition advantages."""
        rl = RiemannLiouvilleDefinition(0.5)
        advantages = rl.get_advantages()

        assert isinstance(advantages, list)
        assert len(advantages) > 0

    def test_riemann_liouville_limitations(self):
        """Test Riemann-Liouville definition limitations."""
        rl = RiemannLiouvilleDefinition(0.5)
        limitations = rl.get_limitations()

        assert isinstance(limitations, list)
        assert len(limitations) > 0


class TestGrunwaldLetnikovDefinition:
    """Test GrunwaldLetnikovDefinition class."""

    def test_grunwald_letnikov_definition_creation(self):
        """Test creating GrunwaldLetnikovDefinition instances."""
        alpha = FractionalOrder(0.5)
        gl = GrunwaldLetnikovDefinition(alpha)

        assert gl.alpha == alpha
        assert gl.definition_type == DefinitionType.GRUNWALD_LETNIKOV

    def test_grunwald_letnikov_formula(self):
        """Test Grünwald-Letnikov definition formula."""
        gl = GrunwaldLetnikovDefinition(0.5)
        formula = gl.get_definition_formula()

        # Check that it contains the mathematical formula
        assert "D^α" in formula
        assert "lim" in formula
        assert "Σ" in formula

    def test_grunwald_letnikov_advantages(self):
        """Test Grünwald-Letnikov definition advantages."""
        gl = GrunwaldLetnikovDefinition(0.5)
        advantages = gl.get_advantages()

        assert isinstance(advantages, list)
        assert len(advantages) > 0

    def test_grunwald_letnikov_limitations(self):
        """Test Grünwald-Letnikov definition limitations."""
        gl = GrunwaldLetnikovDefinition(0.5)
        limitations = gl.get_limitations()

        assert isinstance(limitations, list)
        assert len(limitations) > 0


class TestFractionalDefinition:
    """Test FractionalDefinition base class."""

    def test_fractional_definition_creation(self):
        """Test creating FractionalDefinition instances."""
        alpha = FractionalOrder(0.5)
        caputo = CaputoDefinition(alpha)

        assert isinstance(caputo, FractionalDefinition)
        assert caputo.alpha == alpha

    def test_fractional_definition_abstract_methods(self):
        """Test that base class has required methods."""
        alpha = FractionalOrder(0.5)
        caputo = CaputoDefinition(alpha)

        # Should have required methods
        assert hasattr(caputo, "get_definition_formula")
        assert hasattr(caputo, "get_advantages")
        assert hasattr(caputo, "get_limitations")
        assert callable(caputo.get_definition_formula)
        assert callable(caputo.get_advantages)
        assert callable(caputo.get_limitations)


class TestFractionalIntegral:
    """Test FractionalIntegral class."""

    def test_fractional_integral_creation(self):
        """Test creating FractionalIntegral instances."""
        alpha = FractionalOrder(0.5)
        integral = FractionalIntegral(alpha)

        assert integral.alpha == alpha

    def test_fractional_integral_properties(self):
        """Test FractionalIntegral properties."""
        alpha = FractionalOrder(0.5)
        integral = FractionalIntegral(alpha)

        # Test basic properties
        assert integral.alpha == alpha

        # Test formula method
        formula = integral.get_formula()
        assert "I^α" in formula
        assert "∫" in formula

        # Test properties method
        properties = integral.get_properties()
        assert isinstance(properties, dict)
        assert "linearity" in properties
        assert "semigroup_property" in properties


class TestFractionalCalculusProperties:
    """Test FractionalCalculusProperties class."""

    def test_properties_creation(self):
        """Test creating FractionalCalculusProperties instances."""
        properties = FractionalCalculusProperties()

        assert isinstance(properties, FractionalCalculusProperties)

    def test_properties_methods(self):
        """Test FractionalCalculusProperties methods."""
        properties = FractionalCalculusProperties()

        # Test static methods
        linearity = properties.linearity_property()
        assert isinstance(linearity, str)
        assert "D^α" in linearity

        semigroup = properties.semigroup_property()
        assert isinstance(semigroup, str)
        assert "D^α" in semigroup

        leibniz = properties.leibniz_rule()
        assert isinstance(leibniz, str)
        assert "D^α" in leibniz

        chain = properties.chain_rule()
        assert isinstance(chain, str)
        assert "D^α" in chain

        # Test relationship method
        relationships = properties.relationship_between_definitions()
        assert isinstance(relationships, dict)
        assert len(relationships) > 0

        # Test analytical solutions method
        solutions = properties.get_analytical_solutions()
        assert isinstance(solutions, dict)
        assert len(solutions) > 0


if __name__ == "__main__":
    pytest.main([__file__])
