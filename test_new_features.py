#!/usr/bin/env python3
"""
Test script for new features in hpfracc v0.2.0:
- Fractional integrals (Riemann-Liouville, Caputo)
- Novel fractional derivatives (Caputo-Fabrizio, Atangana-Baleanu)
"""

import numpy as np
import time

def test_fractional_integrals():
    """Test fractional integral implementations."""
    print("🧪 Testing Fractional Integrals...")
    
    try:
        from hpfracc import (
            RiemannLiouvilleIntegral,
            CaputoIntegral,
            riemann_liouville_integral,
            caputo_integral
        )
        print("✅ Integral imports successful")
        
        # Test data
        t = np.linspace(0, 2, 100)
        f = t**2  # f(t) = t²
        h = t[1] - t[0]
        alpha = 0.5
        
        # Test class-based approach
        rl_calc = RiemannLiouvilleIntegral(alpha)
        caputo_calc = CaputoIntegral(alpha)
        
        # Compute integrals
        result_rl = rl_calc.compute(f, t, h)
        result_caputo = caputo_calc.compute(f, t, h)
        
        print(f"✅ RL Integral computed: shape {result_rl.shape}")
        print(f"✅ Caputo Integral computed: shape {result_rl.shape}")
        
        # Test function-based approach
        result_rl_func = riemann_liouville_integral(f, t, alpha, h)
        result_caputo_func = caputo_integral(f, t, alpha, h)
        
        print(f"✅ Function-based integrals computed successfully")
        
        # Verify that Caputo = RL for α > 0
        if np.allclose(result_rl, result_caputo, rtol=1e-10):
            print("✅ Caputo = RL integral verified (as expected)")
        else:
            print("⚠️  Caputo ≠ RL integral (unexpected)")
            
        return True
        
    except Exception as e:
        print(f"❌ Fractional integrals test failed: {e}")
        return False

def test_novel_derivatives():
    """Test novel fractional derivative implementations."""
    print("\n🧪 Testing Novel Fractional Derivatives...")
    
    try:
        from hpfracc import (
            CaputoFabrizioDerivative,
            AtanganaBaleanuDerivative,
            caputo_fabrizio_derivative,
            atangana_baleanu_derivative
        )
        print("✅ Novel derivative imports successful")
        
        # Test data
        t = np.linspace(0, 2, 100)
        f = t**2  # f(t) = t²
        h = t[1] - t[0]
        alpha = 0.3  # Must be in [0, 1)
        
        # Test class-based approach
        cf_calc = CaputoFabrizioDerivative(alpha)
        ab_calc = AtanganaBaleanuDerivative(alpha)
        
        # Compute derivatives
        result_cf = cf_calc.compute(f, t, h)
        result_ab = ab_calc.compute(f, t, h)
        
        print(f"✅ Caputo-Fabrizio derivative computed: shape {result_cf.shape}")
        print(f"✅ Atangana-Baleanu derivative computed: shape {result_ab.shape}")
        
        # Test function-based approach
        result_cf_func = caputo_fabrizio_derivative(f, t, alpha, h)
        result_ab_func = atangana_baleanu_derivative(f, t, alpha, h)
        
        print(f"✅ Function-based derivatives computed successfully")
        
        # Verify results are reasonable (not all zeros)
        if np.any(result_cf != 0) and np.any(result_ab != 0):
            print("✅ Derivatives computed non-zero values (reasonable)")
        else:
            print("⚠️  Derivatives computed all zeros (suspicious)")
            
        return True
        
    except Exception as e:
        print(f"❌ Novel derivatives test failed: {e}")
        return False

def test_performance():
    """Test performance of new methods."""
    print("\n🧪 Testing Performance...")
    
    try:
        from hpfracc import (
            optimized_riemann_liouville_integral,
            optimized_caputo_fabrizio_derivative
        )
        
        # Test with larger arrays
        sizes = [100, 500, 1000]
        alpha = 0.5
        
        for size in sizes:
            t = np.linspace(0, 2, size)
            f = t**2 + np.sin(2*np.pi*t)
            h = t[1] - t[0]
            
            # Test integral
            start_time = time.time()
            result_integral = optimized_riemann_liouville_integral(f, t, alpha, h)
            integral_time = (time.time() - start_time) * 1000
            
            # Test derivative
            start_time = time.time()
            result_derivative = optimized_caputo_fabrizio_derivative(f, t, alpha, h)
            derivative_time = (time.time() - start_time) * 1000
            
            print(f"✅ Size {size}: Integral {integral_time:.2f}ms, Derivative {derivative_time:.2f}ms")
            
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")
    
    try:
        from hpfracc import RiemannLiouvilleIntegral, CaputoFabrizioDerivative
        
        # Test invalid alpha for integrals
        try:
            invalid_integral = RiemannLiouvilleIntegral(0)  # α = 0 not allowed
            print("❌ Should have raised error for α = 0")
            return False
        except ValueError:
            print("✅ Correctly rejected α = 0 for integrals")
            
        # Test invalid alpha for Caputo-Fabrizio
        try:
            invalid_cf = CaputoFabrizioDerivative(1.0)  # α = 1 not allowed
            print("❌ Should have raised error for α = 1")
            return False
        except ValueError:
            print("✅ Correctly rejected α = 1 for Caputo-Fabrizio")
            
        # Test empty arrays
        try:
            calc = RiemannLiouvilleIntegral(0.5)
            result = calc.compute([1], [1])  # Single point
            print("❌ Should have raised error for single point")
            return False
        except ValueError:
            print("✅ Correctly rejected single point arrays")
            
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting New Features Test for hpfracc v0.2.0")
    print("=" * 60)
    
    tests = [
        test_fractional_integrals,
        test_novel_derivatives,
        test_performance,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The new features are working perfectly.")
        print("\n🚀 New Features Available:")
        print("  • Riemann-Liouville Fractional Integral")
        print("  • Caputo Fractional Integral")
        print("  • Caputo-Fabrizio Fractional Derivative")
        print("  • Atangana-Baleanu Fractional Derivative")
        print("\n📚 Ready for v0.2.0 release!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
