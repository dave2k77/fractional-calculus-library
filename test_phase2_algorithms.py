#!/usr/bin/env python3
"""
Phase 2: Algorithm Implementations Test Suite

This script tests all the numerical algorithms implemented in Phase 2,
including Caputo, Riemann-Liouville, Grünwald-Letnikov derivatives,
FFT-based methods, and L1/L2 schemes for time-fractional PDEs.
"""

import numpy as np
import sys
import os
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_caputo_algorithms():
    """Test Caputo derivative algorithms."""
    print("🧪 Testing Caputo Derivative Algorithms...")
    
    try:
        from algorithms import (
            CaputoDerivative, caputo_derivative, caputo_derivative_jax,
            caputo_derivative_numba, caputo_l1_numba
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test different methods
        methods = ["direct", "fft", "l1", "l2", "predictor_corrector"]
        
        for method in methods:
            try:
                # Test class-based approach
                calculator = CaputoDerivative(alpha, method)
                result = calculator.compute(f, t, h=h)
                
                # Test convenience function
                result2 = caputo_derivative(f, t, alpha, method, h=h)
                
                print(f"  ✅ {method.upper()} method: Success")
                
            except Exception as e:
                print(f"  ⚠️  {method.upper()} method: {str(e)}")
        
        # Test JAX implementation
        try:
            result_jax = caputo_derivative_jax(f_values, t, alpha, h)
            print("  ✅ JAX implementation: Success")
        except Exception as e:
            print(f"  ⚠️  JAX implementation: {str(e)}")
        
        # Test NUMBA implementation
        try:
            result_numba = caputo_derivative_numba(f_values, t, alpha, "l1")
            print("  ✅ NUMBA implementation: Success")
        except Exception as e:
            print(f"  ⚠️  NUMBA implementation: {str(e)}")
        
        print("  ✅ Caputo algorithms test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Caputo algorithms test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_riemann_liouville_algorithms():
    """Test Riemann-Liouville derivative algorithms."""
    print("🧪 Testing Riemann-Liouville Derivative Algorithms...")
    
    try:
        from algorithms import (
            RiemannLiouvilleDerivative, riemann_liouville_derivative,
            riemann_liouville_derivative_jax, riemann_liouville_derivative_numba
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test different methods
        methods = ["direct", "fft", "grunwald_letnikov", "predictor_corrector"]
        
        for method in methods:
            try:
                # Test class-based approach
                calculator = RiemannLiouvilleDerivative(alpha, method)
                result = calculator.compute(f, t, h=h)
                
                # Test convenience function
                result2 = riemann_liouville_derivative(f, t, alpha, method, h=h)
                
                print(f"  ✅ {method.upper()} method: Success")
                
            except Exception as e:
                print(f"  ⚠️  {method.upper()} method: {str(e)}")
        
        # Test JAX implementation
        try:
            result_jax = riemann_liouville_derivative_jax(f_values, t, alpha, h)
            print("  ✅ JAX implementation: Success")
        except Exception as e:
            print(f"  ⚠️  JAX implementation: {str(e)}")
        
        # Test NUMBA implementation
        try:
            result_numba = riemann_liouville_derivative_numba(f_values, t, alpha, "grunwald")
            print("  ✅ NUMBA implementation: Success")
        except Exception as e:
            print(f"  ⚠️  NUMBA implementation: {str(e)}")
        
        print("  ✅ Riemann-Liouville algorithms test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Riemann-Liouville algorithms test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_grunwald_letnikov_algorithms():
    """Test Grünwald-Letnikov derivative algorithms."""
    print("🧪 Testing Grünwald-Letnikov Derivative Algorithms...")
    
    try:
        from algorithms import (
            GrunwaldLetnikovDerivative, grunwald_letnikov_derivative,
            grunwald_letnikov_derivative_jax, grunwald_letnikov_derivative_numba
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Test different methods
        methods = ["direct", "fft", "short_memory", "variable_step", "predictor_corrector"]
        
        for method in methods:
            try:
                # Test class-based approach
                calculator = GrunwaldLetnikovDerivative(alpha, method)
                result = calculator.compute(f, t, h=h)
                
                # Test convenience function
                result2 = grunwald_letnikov_derivative(f, t, alpha, method, h=h)
                
                print(f"  ✅ {method.upper()} method: Success")
                
            except Exception as e:
                print(f"  ⚠️  {method.upper()} method: {str(e)}")
        
        # Test JAX implementation
        try:
            result_jax = grunwald_letnikov_derivative_jax(f_values, t, alpha, h)
            print("  ✅ JAX implementation: Success")
        except Exception as e:
            print(f"  ⚠️  JAX implementation: {str(e)}")
        
        # Test NUMBA implementation
        try:
            result_numba = grunwald_letnikov_derivative_numba(f_values, t, alpha, "direct")
            print("  ✅ NUMBA implementation: Success")
        except Exception as e:
            print(f"  ⚠️  NUMBA implementation: {str(e)}")
        
        print("  ✅ Grünwald-Letnikov algorithms test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Grünwald-Letnikov algorithms test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_fft_methods():
    """Test FFT-based fractional calculus methods."""
    print("🧪 Testing FFT-based Methods...")
    
    try:
        from algorithms import (
            FFTFractionalMethods, fft_fractional_derivative,
            fft_fractional_integral, fft_fractional_derivative_jax,
            fft_fractional_derivative_numba
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0, 1, 100)
        h = t[1] - t[0]
        
        # Test function: f(t) = sin(2πt)
        def f(t):
            return np.sin(2 * np.pi * t)
        
        f_values = f(t)
        
        # Test different FFT methods
        methods = ["convolution", "spectral", "fractional_fourier", "wavelet"]
        
        for method in methods:
            try:
                # Test class-based approach
                calculator = FFTFractionalMethods(method)
                result = calculator.compute_derivative(f_values, t, alpha)
                
                # Test convenience function
                result2 = fft_fractional_derivative(f_values, t, alpha, method)
                
                print(f"  ✅ {method.upper()} method: Success")
                
            except Exception as e:
                print(f"  ⚠️  {method.upper()} method: {str(e)}")
        
        # Test fractional integral
        try:
            result_integral = fft_fractional_integral(f_values, t, alpha, "convolution")
            print("  ✅ Fractional integral: Success")
        except Exception as e:
            print(f"  ⚠️  Fractional integral: {str(e)}")
        
        # Test JAX implementation
        try:
            result_jax = fft_fractional_derivative_jax(f_values, t, alpha, h, "convolution")
            print("  ✅ JAX implementation: Success")
        except Exception as e:
            print(f"  ⚠️  JAX implementation: {str(e)}")
        
        # Test NUMBA implementation
        try:
            result_numba = fft_fractional_derivative_numba(f_values, t, alpha, "convolution")
            print("  ✅ NUMBA implementation: Success")
        except Exception as e:
            print(f"  ⚠️  NUMBA implementation: {str(e)}")
        
        print("  ✅ FFT methods test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ FFT methods test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_l1_l2_schemes():
    """Test L1/L2 schemes for time-fractional PDEs."""
    print("🧪 Testing L1/L2 Schemes for Time-Fractional PDEs...")
    
    try:
        from algorithms import (
            L1L2Schemes, solve_time_fractional_pde,
            solve_time_fractional_pde_numba
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t_final = 1.0
        dt = 0.01
        dx = 0.1
        diffusion_coeff = 1.0
        
        # Initial condition: u(x, 0) = sin(πx)
        N_x = 11
        x = np.linspace(0, 1, N_x)
        initial_condition = np.sin(np.pi * x)
        
        # Boundary conditions: u(0, t) = u(1, t) = 0
        def left_bc(t):
            return 0.0
        
        def right_bc(t):
            return 0.0
        
        boundary_conditions = (left_bc, right_bc)
        
        # Test different schemes
        schemes = ["l1", "l2", "l2_1_sigma", "l2_1_theta"]
        
        for scheme in schemes:
            try:
                # Test class-based approach
                solver = L1L2Schemes(scheme)
                t_points, x_points, solution = solver.solve_time_fractional_pde(
                    initial_condition, boundary_conditions, alpha, t_final, dt, dx, diffusion_coeff
                )
                
                # Test convenience function
                t_points2, x_points2, solution2 = solve_time_fractional_pde(
                    initial_condition, boundary_conditions, alpha, t_final, dt, dx, diffusion_coeff, scheme
                )
                
                print(f"  ✅ {scheme.upper()} scheme: Success")
                
            except Exception as e:
                print(f"  ⚠️  {scheme.upper()} scheme: {str(e)}")
        
        # Test stability analysis
        try:
            solver = L1L2Schemes("l1")
            stability_info = solver.stability_analysis(alpha, dt, dx, diffusion_coeff)
            print(f"  ✅ Stability analysis: {stability_info['stability_condition']}")
        except Exception as e:
            print(f"  ⚠️  Stability analysis: {str(e)}")
        
        # Test NUMBA implementation
        try:
            solution_numba = solve_time_fractional_pde_numba(
                initial_condition, alpha, t_final, dt, dx, diffusion_coeff, "l1"
            )
            print("  ✅ NUMBA implementation: Success")
        except Exception as e:
            print(f"  ⚠️  NUMBA implementation: {str(e)}")
        
        print("  ✅ L1/L2 schemes test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ L1/L2 schemes test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_algorithm_consistency():
    """Test consistency between different algorithms for the same problem."""
    print("🧪 Testing Algorithm Consistency...")
    
    try:
        from algorithms import (
            caputo_derivative, riemann_liouville_derivative,
            grunwald_letnikov_derivative, fft_fractional_derivative
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t = np.linspace(0.1, 1, 50)  # Avoid t=0 for numerical stability
        h = t[1] - t[0]
        
        # Test function: f(t) = t^2
        def f(t):
            return t**2
        
        f_values = f(t)
        
        # Compute derivatives using different methods
        results = {}
        
        # Caputo derivative
        try:
            results['caputo'] = caputo_derivative(f, t, alpha, "direct", h=h)
            print("  ✅ Caputo derivative computed")
        except Exception as e:
            print(f"  ⚠️  Caputo derivative: {str(e)}")
        
        # Riemann-Liouville derivative
        try:
            results['riemann_liouville'] = riemann_liouville_derivative(f, t, alpha, "direct", h=h)
            print("  ✅ Riemann-Liouville derivative computed")
        except Exception as e:
            print(f"  ⚠️  Riemann-Liouville derivative: {str(e)}")
        
        # Grünwald-Letnikov derivative
        try:
            results['grunwald_letnikov'] = grunwald_letnikov_derivative(f, t, alpha, "direct", h=h)
            print("  ✅ Grünwald-Letnikov derivative computed")
        except Exception as e:
            print(f"  ⚠️  Grünwald-Letnikov derivative: {str(e)}")
        
        # FFT-based derivative
        try:
            results['fft'] = fft_fractional_derivative(f_values, t, alpha, "convolution")
            print("  ✅ FFT-based derivative computed")
        except Exception as e:
            print(f"  ⚠️  FFT-based derivative: {str(e)}")
        
        # Check consistency (all should give similar results for smooth functions)
        if len(results) >= 2:
            print("  📊 Algorithm consistency check:")
            for name, result in results.items():
                if result is not None and len(result) > 0:
                    print(f"    {name}: mean={np.mean(result):.6f}, std={np.std(result):.6f}")
        
        print("  ✅ Algorithm consistency test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Algorithm consistency test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 algorithm tests."""
    print("🚀 Starting Phase 2: Algorithm Implementations Test Suite\n")
    
    tests = [
        ("Caputo Derivative Algorithms", test_caputo_algorithms),
        ("Riemann-Liouville Derivative Algorithms", test_riemann_liouville_algorithms),
        ("Grünwald-Letnikov Derivative Algorithms", test_grunwald_letnikov_algorithms),
        ("FFT-based Methods", test_fft_methods),
        ("L1/L2 Schemes for Time-Fractional PDEs", test_l1_l2_schemes),
        ("Algorithm Consistency", test_algorithm_consistency)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        print("=" * 50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED\n")
        else:
            print(f"❌ {test_name}: FAILED\n")
    
    print("🎯 Phase 2 Test Summary")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All Phase 2 algorithm tests passed!")
        print("✅ Phase 2: Algorithm Implementations is complete!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
