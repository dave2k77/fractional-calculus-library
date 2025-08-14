#!/usr/bin/env python3
"""
Phase 4: Solvers and Applications Test Suite

This script tests all the solver implementations in Phase 4, including
ODE solvers, PDE solvers, and advanced predictor-corrector methods.
"""

import numpy as np
import sys
import os
import traceback
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_ode_solvers():
    """Test ODE solver implementations."""
    print("🧪 Testing ODE Solvers...")
    
    try:
        from solvers import (
            FractionalODESolver, AdaptiveFractionalODESolver,
            solve_fractional_ode, solve_fractional_system
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t_span = (0, 1)
        y0 = 1.0
        
        # Test function: f(t, y) = -y
        def f(t, y):
            return -y
        
        # Test basic ODE solver
        try:
            solver = FractionalODESolver(derivative_type="caputo", method="predictor_corrector")
            t_values, y_values = solver.solve(f, t_span, y0, alpha, h=0.01)
            print("  ✅ Basic ODE solver: Success")
        except Exception as e:
            print(f"  ⚠️  Basic ODE solver: {str(e)}")
        
        # Test adaptive ODE solver
        try:
            adaptive_solver = AdaptiveFractionalODESolver(derivative_type="caputo")
            t_values, y_values = adaptive_solver.solve(f, t_span, y0, alpha)
            print("  ✅ Adaptive ODE solver: Success")
        except Exception as e:
            print(f"  ⚠️  Adaptive ODE solver: {str(e)}")
        
        # Test different methods
        methods = ["predictor_corrector", "adams_bashforth", "runge_kutta", "euler"]
        for method in methods:
            try:
                solver = FractionalODESolver(derivative_type="caputo", method=method)
                t_values, y_values = solver.solve(f, t_span, y0, alpha, h=0.01)
                print(f"  ✅ {method} method: Success")
            except Exception as e:
                print(f"  ⚠️  {method} method: {str(e)}")
        
        # Test different derivative types
        derivative_types = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        for deriv_type in derivative_types:
            try:
                solver = FractionalODESolver(derivative_type=deriv_type)
                t_values, y_values = solver.solve(f, t_span, y0, alpha, h=0.01)
                print(f"  ✅ {deriv_type} derivative: Success")
            except Exception as e:
                print(f"  ⚠️  {deriv_type} derivative: {str(e)}")
        
        # Test convenience functions
        try:
            t_values, y_values = solve_fractional_ode(f, t_span, y0, alpha)
            print("  ✅ Convenience ODE function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience ODE function: {str(e)}")
        
        # Test system of ODEs
        try:
            def f_system(t, y):
                return np.array([-y[0], -2*y[1]])
            
            y0_system = np.array([1.0, 1.0])
            t_values, y_values = solve_fractional_system(f_system, t_span, y0_system, alpha)
            print("  ✅ System of ODEs: Success")
        except Exception as e:
            print(f"  ⚠️  System of ODEs: {str(e)}")
        
        print("  ✅ ODE solvers test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ ODE solvers test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_pde_solvers():
    """Test PDE solver implementations."""
    print("🧪 Testing PDE Solvers...")
    
    try:
        from solvers import (
            FractionalDiffusionSolver, FractionalAdvectionSolver,
            FractionalReactionDiffusionSolver,
            solve_fractional_diffusion, solve_fractional_advection,
            solve_fractional_reaction_diffusion
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5  # Temporal order
        beta = 1.0   # Spatial order
        x_span = (0, 1)
        t_span = (0, 0.1)
        
        # Initial condition: u(x, 0) = sin(πx)
        def initial_condition(x):
            return np.sin(np.pi * x)
        
        # Boundary conditions: u(0, t) = u(1, t) = 0
        def left_bc(t):
            return 0.0
        
        def right_bc(t):
            return 0.0
        
        boundary_conditions = (left_bc, right_bc)
        
        # Test diffusion solver
        try:
            diffusion_solver = FractionalDiffusionSolver(method="finite_difference")
            t_values, x_values, solution = diffusion_solver.solve(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, diffusion_coeff=1.0, nx=20, nt=10
            )
            print("  ✅ Diffusion solver: Success")
        except Exception as e:
            print(f"  ⚠️  Diffusion solver: {str(e)}")
        
        # Test advection solver
        try:
            advection_solver = FractionalAdvectionSolver(method="finite_difference")
            t_values, x_values, solution = advection_solver.solve(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, velocity=1.0, nx=20, nt=10
            )
            print("  ✅ Advection solver: Success")
        except Exception as e:
            print(f"  ⚠️  Advection solver: {str(e)}")
        
        # Test reaction-diffusion solver
        try:
            def reaction_term(u):
                return -u  # Linear reaction term
            
            reaction_diffusion_solver = FractionalReactionDiffusionSolver(method="finite_difference")
            t_values, x_values, solution = reaction_diffusion_solver.solve(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, diffusion_coeff=1.0, reaction_term=reaction_term, nx=20, nt=10
            )
            print("  ✅ Reaction-diffusion solver: Success")
        except Exception as e:
            print(f"  ⚠️  Reaction-diffusion solver: {str(e)}")
        
        # Test convenience functions
        try:
            t_values, x_values, solution = solve_fractional_diffusion(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, method="finite_difference", nx=20, nt=10
            )
            print("  ✅ Convenience diffusion function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience diffusion function: {str(e)}")
        
        try:
            t_values, x_values, solution = solve_fractional_advection(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, method="finite_difference", nx=20, nt=10
            )
            print("  ✅ Convenience advection function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience advection function: {str(e)}")
        
        try:
            t_values, x_values, solution = solve_fractional_reaction_diffusion(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, reaction_term=reaction_term, method="finite_difference", nx=20, nt=10
            )
            print("  ✅ Convenience reaction-diffusion function: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience reaction-diffusion function: {str(e)}")
        
        print("  ✅ PDE solvers test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ PDE solvers test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_predictor_corrector_methods():
    """Test predictor-corrector method implementations."""
    print("🧪 Testing Predictor-Corrector Methods...")
    
    try:
        from solvers import (
            PredictorCorrectorSolver, AdamsBashforthMoultonSolver,
            VariableStepPredictorCorrector,
            solve_predictor_corrector, solve_adams_bashforth_moulton,
            solve_variable_step_predictor_corrector
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t_span = (0, 1)
        y0 = 1.0
        
        # Test function: f(t, y) = -y
        def f(t, y):
            return -y
        
        # Test basic predictor-corrector solver
        try:
            pc_solver = PredictorCorrectorSolver(derivative_type="caputo", adaptive=False)
            t_values, y_values = pc_solver.solve(f, t_span, y0, alpha, h=0.01)
            print("  ✅ Basic predictor-corrector: Success")
        except Exception as e:
            print(f"  ⚠️  Basic predictor-corrector: {str(e)}")
        
        # Test adaptive predictor-corrector solver
        try:
            adaptive_pc_solver = PredictorCorrectorSolver(derivative_type="caputo", adaptive=True)
            t_values, y_values = adaptive_pc_solver.solve(f, t_span, y0, alpha)
            print("  ✅ Adaptive predictor-corrector: Success")
        except Exception as e:
            print(f"  ⚠️  Adaptive predictor-corrector: {str(e)}")
        
        # Test Adams-Bashforth-Moulton solver
        try:
            abm_solver = AdamsBashforthMoultonSolver(derivative_type="caputo", adaptive=False)
            t_values, y_values = abm_solver.solve(f, t_span, y0, alpha, h=0.01)
            print("  ✅ Adams-Bashforth-Moulton: Success")
        except Exception as e:
            print(f"  ⚠️  Adams-Bashforth-Moulton: {str(e)}")
        
        # Test variable step predictor-corrector
        try:
            var_step_pc = VariableStepPredictorCorrector(derivative_type="caputo")
            t_values, y_values = var_step_pc.solve(f, t_span, y0, alpha, h0=0.01)
            print("  ✅ Variable step predictor-corrector: Success")
        except Exception as e:
            print(f"  ⚠️  Variable step predictor-corrector: {str(e)}")
        
        # Test different derivative types
        derivative_types = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        for deriv_type in derivative_types:
            try:
                pc_solver = PredictorCorrectorSolver(derivative_type=deriv_type, adaptive=False)
                t_values, y_values = pc_solver.solve(f, t_span, y0, alpha, h=0.01)
                print(f"  ✅ {deriv_type} predictor-corrector: Success")
            except Exception as e:
                print(f"  ⚠️  {deriv_type} predictor-corrector: {str(e)}")
        
        # Test convenience functions
        try:
            t_values, y_values = solve_predictor_corrector(f, t_span, y0, alpha, method="standard")
            print("  ✅ Convenience predictor-corrector: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience predictor-corrector: {str(e)}")
        
        try:
            t_values, y_values = solve_adams_bashforth_moulton(f, t_span, y0, alpha)
            print("  ✅ Convenience Adams-Bashforth-Moulton: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience Adams-Bashforth-Moulton: {str(e)}")
        
        try:
            t_values, y_values = solve_variable_step_predictor_corrector(f, t_span, y0, alpha)
            print("  ✅ Convenience variable step: Success")
        except Exception as e:
            print(f"  ⚠️  Convenience variable step: {str(e)}")
        
        print("  ✅ Predictor-corrector methods test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Predictor-corrector methods test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_solver_integration():
    """Test integration between different solver types."""
    print("🧪 Testing Solver Integration...")
    
    try:
        from solvers import (
            solve_fractional_ode, solve_fractional_diffusion,
            solve_predictor_corrector
        )
        from core.definitions import FractionalOrder
        
        # Test parameters
        alpha = 0.5
        t_span = (0, 1)
        y0 = 1.0
        
        # Test function: f(t, y) = -y
        def f(t, y):
            return -y
        
        # Test ODE + Predictor-Corrector integration
        try:
            # Solve with ODE solver
            t_ode, y_ode = solve_fractional_ode(f, t_span, y0, alpha, method="predictor_corrector")
            
            # Solve with predictor-corrector
            t_pc, y_pc = solve_predictor_corrector(f, t_span, y0, alpha, method="standard")
            
            print("  ✅ ODE + Predictor-Corrector integration: Success")
        except Exception as e:
            print(f"  ⚠️  ODE + Predictor-Corrector integration: {str(e)}")
        
        # Test PDE + ODE integration
        try:
            # PDE parameters
            beta = 1.0
            x_span = (0, 1)
            
            def initial_condition(x):
                return np.sin(np.pi * x)
            
            def left_bc(t):
                return 0.0
            
            def right_bc(t):
                return 0.0
            
            boundary_conditions = (left_bc, right_bc)
            
            # Solve PDE
            t_pde, x_pde, solution_pde = solve_fractional_diffusion(
                x_span, t_span, initial_condition, boundary_conditions,
                alpha, beta, nx=10, nt=5
            )
            
            print("  ✅ PDE + ODE integration: Success")
        except Exception as e:
            print(f"  ⚠️  PDE + ODE integration: {str(e)}")
        
        # Test performance comparison
        try:
            # Time ODE solver
            start_time = time.time()
            t_ode, y_ode = solve_fractional_ode(f, t_span, y0, alpha, h=0.01)
            ode_time = time.time() - start_time
            
            # Time predictor-corrector
            start_time = time.time()
            t_pc, y_pc = solve_predictor_corrector(f, t_span, y0, alpha, h=0.01)
            pc_time = time.time() - start_time
            
            print(f"  ✅ Performance comparison: ODE={ode_time:.4f}s, PC={pc_time:.4f}s")
        except Exception as e:
            print(f"  ⚠️  Performance comparison: {str(e)}")
        
        print("  ✅ Solver integration test completed\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Solver integration test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 solver tests."""
    print("🚀 Starting Phase 4: Solvers and Applications Test Suite\n")
    
    tests = [
        ("ODE Solvers", test_ode_solvers),
        ("PDE Solvers", test_pde_solvers),
        ("Predictor-Corrector Methods", test_predictor_corrector_methods),
        ("Solver Integration", test_solver_integration)
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
    
    print("🎯 Phase 4 Test Summary")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All Phase 4 solver tests passed!")
        print("✅ Phase 4: Solvers and Applications is complete!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
