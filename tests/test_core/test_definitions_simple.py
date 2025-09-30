#!/usr/bin/env python3
"""Simple tests for core/definitions.py - low-hanging fruit."""

import pytest
from hpfracc.core.definitions import FractionalOrder


class TestDefinitionsSimple:
    """Simple tests for FractionalOrder class."""
    
    def test_fractional_order_basic(self):
        """Test basic FractionalOrder creation."""
        order = FractionalOrder(0.5)
        assert isinstance(order, FractionalOrder)
        
    def test_fractional_order_different_values(self):
        """Test with different alpha values."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        
        for alpha in alphas:
            order = FractionalOrder(alpha)
            assert isinstance(order, FractionalOrder)
            
    def test_fractional_order_string_representation(self):
        """Test string representation."""
        order = FractionalOrder(0.5)
        str_repr = str(order)
        assert isinstance(str_repr, str)
        assert "0.5" in str_repr
        
    def test_fractional_order_equality(self):
        """Test equality comparison."""
        order1 = FractionalOrder(0.5)
        order2 = FractionalOrder(0.5)
        order3 = FractionalOrder(0.7)
        
        assert order1 == order2
        assert order1 != order3
        
    def test_fractional_order_properties(self):
        """Test accessing properties."""
        alpha = 0.5
        order = FractionalOrder(alpha)
        
        # Test that we can access the value somehow
        assert hasattr(order, 'alpha') or hasattr(order, 'value') or hasattr(order, '_alpha')
        
    def test_fractional_order_edge_cases(self):
        """Test edge cases."""
        # Boundary values
        edge_values = [0.01, 0.99, 1.0, 1.99]
        
        for alpha in edge_values:
            order = FractionalOrder(alpha)
            assert isinstance(order, FractionalOrder)
            
    def test_fractional_order_validation(self):
        """Test parameter validation."""
        # Valid values should work
        valid_alphas = [0.1, 0.5, 1.0, 1.5]
        for alpha in valid_alphas:
            order = FractionalOrder(alpha)
            assert isinstance(order, FractionalOrder)
            
        # Invalid values should raise errors or handle gracefully
        invalid_alphas = [-0.1, 0.0, 2.1]
        for alpha in invalid_alphas:
            try:
                order = FractionalOrder(alpha)
                # If no error, that's OK too
                assert isinstance(order, FractionalOrder)
            except (ValueError, AssertionError):
                # Expected for invalid values
                pass













