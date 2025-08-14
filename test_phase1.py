#!/usr/bin/env python3
"""
Test script for Phase 1: Core Mathematical Foundation

This script tests the special functions and core definitions
implemented in Phase 1 of the fractional calculus library.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_special_functions():
    """Test the special functions implementations."""
    print("Testing Special Functions...")
    
    try:
        from special import gamma, beta, mittag_leffler, binomial_fractional
        
        # Test Gamma function
        print("  Testing Gamma function...")
        x = 2.5
        gamma_val = gamma(x)
        expected_gamma = 1.329340388179137
        assert abs(gamma_val - expected_gamma) < 1e-10
        print(f"    ✓ Gamma({x}) = {gamma_val}")
        
        # Test Beta function
        print("  Testing Beta function...")
        a, b = 2.5, 1.5
        beta_val = beta(a, b)
        expected_beta = 0.196349540849362
        assert abs(beta_val - expected_beta) < 1e-10
        print(f"    ✓ Beta({a}, {b}) = {beta_val}")
        
        # Test Mittag-Leffler function
        print("  Testing Mittag-Leffler function...")
        z, alpha, beta_param = 1.0, 1.0, 1.0
        ml_val = mittag_leffler(z, alpha, beta_param)
        expected_ml = np.exp(z)  # E_1,1(z) = e^z
        assert abs(ml_val - expected_ml) < 1e-10
        print(f"    ✓ E_{alpha},{beta_param}({z}) = {ml_val}")
        
        # Test fractional binomial coefficients
        print("  Testing fractional binomial coefficients...")
        alpha, k = 0.5, 2
        binom_val = binomial_fractional(alpha, k)
        expected_binom = -0.125  # C(0.5, 2) = -1/8
        assert abs(binom_val - expected_binom) < 1e-10
        print(f"    ✓ C({alpha}, {k}) = {binom_val}")
        
        print("  ✓ All special functions tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Special functions test failed: {e}")
        return False


def test_core_definitions():
    """Test the core definitions and base classes."""
    print("Testing Core Definitions...")
    
    try:
        from core import (
            FractionalOrder, DefinitionType, CaputoDefinition,
            RiemannLiouvilleDefinition, GrunwaldLetnikovDefinition,
            BaseFractionalDerivative, FractionalDerivativeOperator
        )
        
        # Test FractionalOrder
        print("  Testing FractionalOrder...")
        alpha = FractionalOrder(0.5)
        assert alpha.alpha == 0.5
        assert not alpha.is_integer
        assert alpha.integer_part == 0
        assert alpha.fractional_part == 0.5
        print(f"    ✓ FractionalOrder({alpha.alpha}) created successfully")
        
        # Test Caputo definition
        print("  Testing Caputo definition...")
        caputo = CaputoDefinition(0.5)
        assert caputo.alpha.alpha == 0.5
        assert caputo.n == 1
        formula = caputo.get_definition_formula()
        assert "Caputo" in str(caputo)
        print(f"    ✓ Caputo definition created: {formula[:50]}...")
        
        # Test Riemann-Liouville definition
        print("  Testing Riemann-Liouville definition...")
        rl = RiemannLiouvilleDefinition(0.5)
        assert rl.alpha.alpha == 0.5
        assert rl.n == 1
        advantages = rl.get_advantages()
        assert len(advantages) > 0
        print(f"    ✓ Riemann-Liouville definition created with {len(advantages)} advantages")
        
        # Test Grünwald-Letnikov definition
        print("  Testing Grünwald-Letnikov definition...")
        gl = GrunwaldLetnikovDefinition(0.5)
        assert gl.alpha.alpha == 0.5
        limitations = gl.get_limitations()
        assert len(limitations) > 0
        print(f"    ✓ Grünwald-Letnikov definition created with {len(limitations)} limitations")
        
        # Test FractionalDerivativeOperator
        print("  Testing FractionalDerivativeOperator...")
        operator = FractionalDerivativeOperator(0.5, DefinitionType.CAPUTO)
        info = operator.get_info()
        assert info['alpha'] == 0.5
        assert info['definition_type'] == 'caputo'
        print(f"    ✓ FractionalDerivativeOperator created: {info}")
        
        print("  ✓ All core definitions tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Core definitions test failed: {e}")
        return False


def test_jax_integration():
    """Test JAX integration."""
    print("Testing JAX Integration...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        from special import gamma, mittag_leffler
        
        # Test JAX Gamma function
        print("  Testing JAX Gamma function...")
        x = jnp.array([1.0, 2.0, 3.0])
        gamma_vals = gamma(x, use_jax=True)
        assert gamma_vals.shape == (3,)
        print(f"    ✓ JAX Gamma function: {gamma_vals}")
        
        # Test JAX Mittag-Leffler function
        print("  Testing JAX Mittag-Leffler function...")
        z = jnp.array([0.5, 1.0, 1.5])
        ml_vals = mittag_leffler(z, alpha=1.0, beta=1.0, use_jax=True)
        assert ml_vals.shape == (3,)
        print(f"    ✓ JAX Mittag-Leffler function: {ml_vals}")
        
        print("  ✓ All JAX integration tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ JAX integration test failed: {e}")
        return False


def test_numba_integration():
    """Test NUMBA integration."""
    print("Testing NUMBA Integration...")
    
    try:
        from special import gamma_vectorized, binomial_fractional_vectorized
        
        # Test NUMBA vectorized Gamma function
        print("  Testing NUMBA vectorized Gamma function...")
        x = np.array([1.0, 2.0, 3.0])
        gamma_vals = gamma_vectorized(x)
        assert gamma_vals.shape == (3,)
        print(f"    ✓ NUMBA vectorized Gamma function: {gamma_vals}")
        
        # Test NUMBA vectorized binomial coefficients
        print("  Testing NUMBA vectorized binomial coefficients...")
        alpha = np.array([0.5, 1.0, 1.5])
        k = np.array([1, 2, 1])
        binom_vals = binomial_fractional_vectorized(alpha, k)
        assert binom_vals.shape == (3,)
        print(f"    ✓ NUMBA vectorized binomial coefficients: {binom_vals}")
        
        print("  ✓ All NUMBA integration tests passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ NUMBA integration test failed: {e}")
        return False


def main():
    """Run all Phase 1 tests."""
    print("=" * 60)
    print("Phase 1: Core Mathematical Foundation - Test Suite")
    print("=" * 60)
    
    tests = [
        test_special_functions,
        test_core_definitions,
        test_jax_integration,
        test_numba_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 1 tests passed! Core foundation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
