#!/usr/bin/env python3
"""Coverage tests for core/definitions.py - currently 94% coverage."""

import pytest
from hpfracc.core.definitions import FractionalOrder


class TestDefinitionsCoverage:
    """Tests to push definitions.py to 100% coverage."""
    
    def test_fractional_order_edge_cases(self):
        """Test edge cases for FractionalOrder."""
        # Valid edge cases
        orders = [0.01, 0.99, 1.0, 1.99]
        for alpha in orders:
            order = FractionalOrder(alpha)
            assert order.value == alpha
            
    def test_fractional_order_methods(self):
        """Test different methods."""
        methods = ["RL", "Caputo", "GL", "CF", "AB"]
        for method in methods:
            order = FractionalOrder(0.5, method=method)
            assert order.method == method
            
    def test_fractional_order_properties(self):
        """Test properties and methods."""
        order = FractionalOrder(0.5)
        
        # Test string representation
        str_repr = str(order)
        assert "0.5" in str_repr
        
        # Test equality
        order2 = FractionalOrder(0.5)
        assert order == order2
        
        order3 = FractionalOrder(0.7)
        assert order != order3
        
    def test_fractional_order_validation(self):
        """Test parameter validation."""
        # Valid ranges
        valid_alphas = [0.1, 0.5, 1.0, 1.5, 1.9]
        for alpha in valid_alphas:
            order = FractionalOrder(alpha)
            assert 0 < order.value < 2
            
    def test_fractional_order_copy(self):
        """Test copying fractional orders."""
        original = FractionalOrder(0.5, method="Caputo")
        
        # Test that we can create similar orders
        copy_order = FractionalOrder(original.value, method=original.method)
        assert copy_order.value == original.value
        assert copy_order.method == original.method














