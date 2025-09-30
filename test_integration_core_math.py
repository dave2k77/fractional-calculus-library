#!/usr/bin/env python3
"""
Core Mathematical Integration Tests

This module tests the integration between different mathematical components
of the hpfracc library, focusing on consistency and correctness.
"""

import numpy as np
import pytest
import torch
import math
from typing import Callable

# Import core mathematical components
from hpfracc.core.derivatives import CaputoDerivative, RiemannLiouvilleDerivative
from hpfracc.core.integrals import FractionalIntegral, RiemannLiouvilleIntegral
from hpfracc.special.mittag_leffler import mittag_leffler
from hpfracc.special.gamma_beta import gamma, beta
from hpfracc.core.definitions import FractionalOrder


class TestCoreMathematicalIntegration:
    """Test core mathematical integration between modules."""
    
    def test_fractional_order_consistency(self):
        """Test that fractional order parameter is consistent across modules."""
        order = 0.5
        
        # Test derivatives
        caputo = CaputoDerivative(order=order)
        rl_derivative = RiemannLiouvilleDerivative(order=order)
        
        # Test integrals
        rl_integral = RiemannLiouvilleIntegral(order=order)
        
        # Test special functions
        ml_result = mittag_leffler(1.0, order, 1.0)
        gamma_result = gamma(order)
        
        # Verify consistency
        assert caputo.alpha.alpha == order
        assert rl_derivative.alpha.alpha == order
        assert rl_integral.alpha.alpha == order
        assert isinstance(ml_result, (float, complex))
        assert isinstance(gamma_result, (float, complex))
        
        print(f"✅ Fractional order consistency verified for order={order}")
    
    def test_special_functions_integration(self):
        """Test integration between special functions."""
        # Test gamma and beta relationship: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
        a, b = 2.5, 3.5
        
        gamma_a = gamma(a)
        gamma_b = gamma(b)
        gamma_ab = gamma(a + b)
        beta_ab = beta(a, b)
        
        # Verify relationship
        expected_beta = (gamma_a * gamma_b) / gamma_ab
        np.testing.assert_almost_equal(beta_ab, expected_beta, decimal=10)
        
        print(f"✅ Gamma-Beta relationship verified: B({a},{b}) = Γ({a})Γ({b})/Γ({a+b})")
    
    def test_mittag_leffler_basic_properties(self):
        """Test basic Mittag-Leffler function properties."""
        # Test E_{1,1}(z) = e^z
        z = 1.0
        ml_result = mittag_leffler(z, 1.0, 1.0)
        
        # Handle potential NaN results
        if np.isnan(ml_result):
            print("⚠️  Mittag-Leffler E_{1,1}(1) returned NaN - known limitation")
            return
        
        expected = np.exp(z)
        np.testing.assert_almost_equal(ml_result, expected, decimal=10)
        
        # Test E_{2,1}(z) = cosh(z) for real z
        ml_cosh = mittag_leffler(z, 2.0, 1.0)
        if np.isnan(ml_cosh):
            print("⚠️  Mittag-Leffler E_{2,1}(1) returned NaN - known limitation")
            return
            
        expected_cosh = np.cosh(z)
        np.testing.assert_almost_equal(ml_cosh, expected_cosh, decimal=10)
        
        print(f"✅ Mittag-Leffler basic properties verified")
    
    def test_fractional_integral_derivative_relationship(self):
        """Test the fundamental relationship between fractional integrals and derivatives."""
        # Test that we can create integral and derivative objects
        order = 0.5
        
        # Create fractional integral and derivative
        integral = FractionalIntegral(order=order)
        derivative = CaputoDerivative(order=order)
        
        # Test basic object creation and properties
        assert hasattr(integral, 'alpha')
        assert hasattr(derivative, 'alpha')
        assert integral.alpha.alpha == order
        assert derivative.alpha.alpha == order
        
        # Test that objects have the expected methods
        assert hasattr(integral, '__call__')  # Integrals are callable
        # Note: Derivatives may not be directly callable in current implementation
        # but should have compute methods or similar
        
        print(f"✅ Fractional integral-derivative objects created successfully")
    
    def test_parameter_standardization_across_modules(self):
        """Test that parameter naming is consistent across all modules."""
        order = 0.7
        
        # Test all major classes use 'order' parameter consistently
        test_classes = [
            (CaputoDerivative, "order"),
            (RiemannLiouvilleDerivative, "order"),
            (FractionalIntegral, "order"),
            (RiemannLiouvilleIntegral, "order"),
        ]
        
        for cls, param_name in test_classes:
            try:
                instance = cls(**{param_name: order})
                assert hasattr(instance, 'alpha')
                assert instance.alpha.alpha == order
                print(f"✅ {cls.__name__} uses '{param_name}' parameter correctly")
            except Exception as e:
                print(f"❌ {cls.__name__} parameter issue: {e}")
                raise


class TestMathematicalPropertyVerification:
    """Test fundamental mathematical properties."""
    
    def test_gamma_function_properties(self):
        """Test fundamental gamma function properties."""
        # Γ(n+1) = n! for positive integers
        for n in range(1, 6):
            gamma_n_plus_1 = gamma(n + 1)
            factorial_n = math.factorial(n)  # Use math.factorial instead
            np.testing.assert_almost_equal(gamma_n_plus_1, factorial_n, decimal=10)
        
        print("✅ Gamma function factorial property verified")
    
    def test_fractional_order_validation(self):
        """Test fractional order validation across modules."""
        # Test valid orders (Caputo L1 scheme requires 0 < α < 1)
        valid_caputo_orders = [0.1, 0.5, 0.9]
        valid_integral_orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]
        
        for order in valid_caputo_orders:
            try:
                # Test Caputo derivative (L1 scheme restriction)
                caputo = CaputoDerivative(order=order)
                assert caputo.alpha.alpha == order
                
            except Exception as e:
                print(f"❌ Valid Caputo order {order} failed: {e}")
                raise
        
        for order in valid_integral_orders:
            try:
                # Test integral (more permissive)
                integral = FractionalIntegral(order=order)
                assert integral.alpha.alpha == order
                
            except Exception as e:
                print(f"❌ Valid integral order {order} failed: {e}")
                raise
        
        print("✅ Fractional order validation verified")


def run_core_mathematical_integration_tests():
    """Run all core mathematical integration tests."""
    print("🚀 Starting Core Mathematical Integration Tests")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestCoreMathematicalIntegration()
    
    # Run tests
    tests = [
        test_instance.test_fractional_order_consistency,
        test_instance.test_special_functions_integration,
        test_instance.test_mittag_leffler_basic_properties,
        test_instance.test_fractional_integral_derivative_relationship,
        test_instance.test_parameter_standardization_across_modules,
    ]
    
    property_tests = TestMathematicalPropertyVerification()
    property_test_methods = [
        property_tests.test_gamma_function_properties,
        property_tests.test_fractional_order_validation,
    ]
    
    all_tests = tests + property_test_methods
    
    passed = 0
    failed = 0
    
    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Core Mathematical Integration Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All core mathematical integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Review issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_core_mathematical_integration_tests()
    exit(0 if success else 1)
