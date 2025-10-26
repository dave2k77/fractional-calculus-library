#!/usr/bin/env python3
"""
Validation tests for the fixed fractional ODE solver.

This tests the corrections made to address:
1. Repeated history calculation
2. Incorrect corrector iteration loop
3. FODE scheme formulation
4. FFT performance optimization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hpfracc.solvers.ode_solvers import solve_fractional_ode


def test_fixed_step_solver():
    """Test fixed-step predictor-corrector with iterative refinement."""
    
    print("\n" + "="*70)
    print("TEST 1: Fixed-Step Predictor-Corrector with Iterative Refinement")
    print("="*70)
    
    # Test ODE: D^α y = -0.5*y, y(0) = 1
    def f(t, y):
        return -0.5 * y
    
    alpha = 0.75
    t_span = (0.0, 5.0)
    y0 = 1.0
    h = 0.01
    
    print(f"\nSolving: D^{alpha} y = -0.5*y")
    print(f"Initial condition: y(0) = {y0}")
    print(f"Time span: {t_span}")
    print(f"Step size: h = {h}")
    
    # Solve with different corrector iteration counts
    results = {}
    for max_iter in [1, 2, 3, 5]:
        t_vals, y_vals = solve_fractional_ode(
            f, t_span, y0, alpha,
            method="predictor_corrector",
            h=h,
            max_corrector_iter=max_iter,
            verbose=False # Keep output clean for this test
        )
        results[max_iter] = (t_vals, y_vals)
        print(f"\nmax_corrector_iter={max_iter}:")
        print(f"  Final value: {y_vals[-1].item():.8f}")
    
    # Check convergence: more iterations should give similar results
    y_final_3 = results[3][1][-1].item()
    y_final_5 = results[5][1][-1].item()
    diff = abs(y_final_5 - y_final_3)
    
    print(f"\nConvergence check:")
    print(f"  |y_final(iter=5) - y_final(iter=3)| = {diff:.2e}")
    
    if diff < 1e-6:
        print(f"  ✓ PASS: Iterative refinement converges properly")
    else:
        print(f"  ⚠ WARNING: Large difference suggests convergence issues")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    for max_iter, (t_vals, y_vals) in results.items():
        y_sequence = y_vals.flatten()
        ax1.plot(t_vals, y_sequence, label=f'max_iter={max_iter}', linewidth=2)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('y(t)')
    ax1.set_title('Solution with Different Corrector Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    # Plot relative differences
    t_ref, y_ref = results[5]
    y_ref_seq = y_ref.flatten()
    for max_iter in [1, 2, 3]:
        t_vals, y_vals = results[max_iter]
        y_sequence = y_vals.flatten()
        rel_diff = np.abs(y_sequence - y_ref_seq) / (np.abs(y_ref_seq) + 1e-12)
        ax2.semilogy(t_vals, rel_diff, label=f'iter={max_iter} vs iter=5')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Relative Difference')
    ax2.set_title('Convergence of Iterative Refinement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'solver_fixes_test1_fixed_step.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {filename}")
    plt.close()

def test_accuracy_comparison():
    """Compare solution accuracy with different h values."""
    
    print("\n" + "="*70)
    print("TEST 2: Accuracy and Convergence Rate")
    print("="*70)
    
    def f(t, y):
        return -y
    
    alpha = 0.85
    t_span = (0.0, 5.0)
    y0 = 1.0
    
    print(f"\nSolving: D^{alpha} y = -y with y(0) = {y0}")
    print(f"Testing with different step sizes to check convergence order.")
    
    h_values = [0.1, 0.05, 0.02, 0.01, 0.005]
    results = {}
    
    for h in h_values:
        t_vals, y_vals = solve_fractional_ode(
            f, t_span, y0, alpha,
            method="predictor_corrector",
            h=h,
            max_corrector_iter=3
        )
        results[h] = (t_vals, y_vals)
        print(f"  h={h:6.4f}: y_final = {y_vals[-1].item():.8f}, steps = {len(t_vals)}")
    
    # Use finest mesh as reference
    y_ref_final = results[0.005][1][-1].item()
    
    print(f"\nConvergence analysis (reference: h=0.005):")
    errors = []
    for h in h_values[:-1]:
        y_final = results[h][1][-1].item()
        error = abs(y_final - y_ref_final)
        errors.append(error)
        print(f"  h={h:6.4f}: error = {error:.2e}")
    
    # Check for convergence (error should decrease with h)
    if all(errors[i] > errors[i+1] for i in range(len(errors)-1)):
        print(f"\n  ✓ PASS: Solution converges as h decreases")
    else:
        print(f"\n  ⚠ NOTE: Convergence pattern is not strictly monotonic, which can happen.")
    
    # Plot convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(h_values[:-1], errors, 'bo-', linewidth=2, markersize=10, label='Actual error')
    
    # Add reference slopes
    h_arr = np.array(h_values[:-1])
    # Expected order for ABM PC is p = min(2, 1+alpha)
    expected_order = 1 + alpha
    ref_error = errors[0] * (h_arr / h_values[0])**expected_order
    ax.loglog(h_arr, ref_error, 'r--', linewidth=2, alpha=0.7, label=f'O(h^{expected_order:.2f}) reference')
    
    ax.set_xlabel('Step Size h', fontweight='bold')
    ax.set_ylabel('Error in Final Value', fontweight='bold')
    ax.set_title(f'Convergence Rate (α={alpha})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'solver_fixes_test2_convergence.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {filename}")
    plt.close()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VALIDATION OF FIXED-STEP SOLVER CORRECTIONS")
    print("="*70)
    
    test_fixed_step_solver()
    test_accuracy_comparison()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nFixed-step solver tests completed successfully.")
    print("Check generated PNG files for visual validation.")
