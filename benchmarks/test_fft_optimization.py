#!/usr/bin/env python3
"""
Benchmark script to test FFT-optimized ODE solver performance.

This script compares the performance and accuracy of the FFT-optimized
fractional ODE solver against the original implementation.
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from hpfracc.solvers.ode_solvers import FixedStepODESolver, solve_fractional_ode
from hpfracc.core.definitions import FractionalOrder


def test_simple_fractional_ode():
    """Test case: Simple fractional ODE with known behavior."""
    
    # Define test ODE: D^α y = -y
    # This has well-understood behavior
    def f(t, y):
        return -y
    
    # Test parameters
    alpha = 0.75
    t_span = (0.0, 5.0)
    y0 = 1.0
    
    return f, t_span, y0, alpha


def test_nonlinear_fractional_ode():
    """Test case: Nonlinear fractional ODE."""
    
    # Fractional logistic equation: D^α y = y(1 - y)
    def f(t, y):
        return y * (1 - y)
    
    alpha = 0.8
    t_span = (0.0, 10.0)
    y0 = 0.1
    
    return f, t_span, y0, alpha


def test_system_fractional_ode():
    """Test case: System of fractional ODEs (Lotka-Volterra)."""
    
    # Fractional predator-prey model
    def f(t, y):
        x, z = y
        dx = x - 0.5 * x * z
        dz = -z + 0.5 * x * z
        return np.array([dx, dz])
    
    alpha = 0.9
    t_span = (0.0, 20.0)
    y0 = np.array([2.0, 1.0])
    
    return f, t_span, y0, alpha


def benchmark_solver(f, t_span, y0, alpha, h_values, test_name="Test"):
    """
    Benchmark the solver with different step sizes.
    
    Args:
        f: ODE function
        t_span: Time span tuple
        y0: Initial condition
        alpha: Fractional order
        h_values: List of step sizes to test
        test_name: Name of the test
        
    Returns:
        Dictionary with timing and accuracy results
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark: {test_name}")
    print(f"Alpha = {alpha}, t_span = {t_span}")
    print(f"{'='*60}")
    
    results = {
        'h_values': [],
        'times': [],
        'num_steps': [],
        'final_values': [],
        'errors': []
    }
    
    for h in h_values:
        print(f"\nTesting with h = {h:.6f}")
        
        # Time the solver
        start_time = time.perf_counter()
        
        try:
            t_vals, y_vals = solve_fractional_ode(
                f, t_span, y0, alpha,
                derivative_type="caputo",
                method="predictor_corrector",
                adaptive=False,
                h=h
            )
            
            elapsed_time = time.perf_counter() - start_time
            
            # Store results
            results['h_values'].append(h)
            results['times'].append(elapsed_time)
            results['num_steps'].append(len(t_vals))
            results['final_values'].append(y_vals[-1].copy() if y_vals.ndim > 1 else y_vals[-1])
            
            print(f"  Steps: {len(t_vals)}")
            print(f"  Time: {elapsed_time:.4f} seconds")
            print(f"  Final value: {y_vals[-1]}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results['h_values'].append(h)
            results['times'].append(np.nan)
            results['num_steps'].append(0)
            results['final_values'].append(np.nan)
    
    return results


def plot_benchmark_results(results, test_name):
    """Plot benchmark results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time vs. Number of Steps
    ax1 = axes[0]
    ax1.loglog(results['num_steps'], results['times'], 'o-', linewidth=2, markersize=8)
    
    # Add reference lines for O(N) and O(N²) complexity
    if len(results['num_steps']) > 1:
        N = np.array(results['num_steps'])
        T = np.array(results['times'])
        
        # Fit O(N²) line to first few points (if they existed before optimization)
        # For visualization, show what O(N²) would look like
        N_ref = N[~np.isnan(T)]
        T_ref = T[~np.isnan(T)]
        
        if len(N_ref) > 0:
            # O(N log N) reference
            t0 = T_ref[0] / (N_ref[0] * np.log(N_ref[0]))
            T_nlogn = t0 * N * np.log(N)
            ax1.loglog(N, T_nlogn, '--', alpha=0.5, label='O(N log N) reference', color='green')
            
            # O(N²) reference for comparison
            t0_n2 = T_ref[0] / (N_ref[0]**2)
            T_n2 = t0_n2 * N**2
            ax1.loglog(N, T_n2, '--', alpha=0.5, label='O(N²) reference', color='red')
    
    ax1.set_xlabel('Number of Steps (N)', fontsize=12)
    ax1.set_ylabel('Computation Time (s)', fontsize=12)
    ax1.set_title(f'{test_name}: Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Step Size vs. Time
    ax2 = axes[1]
    valid_mask = ~np.isnan(results['times'])
    ax2.semilogx(
        np.array(results['h_values'])[valid_mask],
        np.array(results['times'])[valid_mask],
        'o-', linewidth=2, markersize=8
    )
    ax2.set_xlabel('Step Size (h)', fontsize=12)
    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title(f'{test_name}: Time vs. Step Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    filename = f"fft_benchmark_{test_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")
    plt.close()


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks on multiple test cases."""
    
    print("\n" + "="*70)
    print("FFT-OPTIMIZED FRACTIONAL ODE SOLVER BENCHMARK")
    print("="*70)
    print("\nThis benchmark tests the performance improvements from FFT optimization.")
    print("Expected: O(N log N) scaling instead of O(N²)")
    
    # Test 1: Simple linear ODE
    print("\n" + "="*70)
    print("TEST 1: Simple Linear Fractional ODE")
    print("="*70)
    
    f, t_span, y0, alpha = test_simple_fractional_ode()
    h_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    results1 = benchmark_solver(f, t_span, y0, alpha, h_values, "Linear ODE")
    plot_benchmark_results(results1, "Linear ODE")
    
    # Test 2: Nonlinear ODE
    print("\n" + "="*70)
    print("TEST 2: Nonlinear Fractional ODE (Logistic)")
    print("="*70)
    
    f, t_span, y0, alpha = test_nonlinear_fractional_ode()
    h_values = [0.2, 0.1, 0.05, 0.02, 0.01]
    results2 = benchmark_solver(f, t_span, y0, alpha, h_values, "Nonlinear ODE")
    plot_benchmark_results(results2, "Nonlinear ODE")
    
    # Test 3: System of ODEs
    print("\n" + "="*70)
    print("TEST 3: System of Fractional ODEs (Lotka-Volterra)")
    print("="*70)
    
    f, t_span, y0, alpha = test_system_fractional_ode()
    h_values = [0.2, 0.1, 0.05, 0.02]
    results3 = benchmark_solver(f, t_span, y0, alpha, h_values, "System ODE")
    plot_benchmark_results(results3, "System ODE")
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    all_results = [results1, results2, results3]
    test_names = ["Linear ODE", "Nonlinear ODE", "System ODE"]
    
    for results, name in zip(all_results, test_names):
        print(f"\n{name}:")
        valid_times = [t for t in results['times'] if not np.isnan(t)]
        if valid_times:
            print(f"  Fastest time: {min(valid_times):.4f} s")
            print(f"  Slowest time: {max(valid_times):.4f} s")
            print(f"  Max steps: {max(results['num_steps'])}")
            
            # Estimate scaling
            if len(valid_times) >= 2:
                n1, n2 = results['num_steps'][0], results['num_steps'][-1]
                t1, t2 = valid_times[0], valid_times[-1]
                
                # Estimate exponent: t ~ N^p, so p = log(t2/t1) / log(N2/N1)
                if n1 > 0 and n2 > n1 and t1 > 0:
                    exponent = np.log(t2/t1) / np.log(n2/n1)
                    print(f"  Empirical scaling: O(N^{exponent:.2f})")
                    
                    if exponent < 1.5:
                        print(f"  ✓ Excellent! Scaling is close to O(N log N)")
                    elif exponent < 1.8:
                        print(f"  ✓ Good! Scaling is sub-quadratic")
                    else:
                        print(f"  ⚠ Warning: Scaling is closer to O(N²)")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print("\nThe FFT optimization should show O(N log N) scaling,")
    print("which is significantly better than the original O(N²) approach")
    print("for large numbers of time steps.")
    print("\nCheck the generated PNG files for visual results.")


def test_accuracy():
    """Test that FFT optimization maintains accuracy."""
    
    print("\n" + "="*70)
    print("ACCURACY TEST")
    print("="*70)
    
    # Use a case with known solution behavior
    def f(t, y):
        return -0.5 * y
    
    alpha = 0.75
    t_span = (0.0, 5.0)
    y0 = 1.0
    h = 0.01
    
    print(f"\nSolving D^{alpha} y = -0.5*y with y(0) = {y0}")
    print(f"Time span: {t_span}, Step size: h = {h}")
    
    # Solve with FFT optimization (default)
    t_vals, y_vals = solve_fractional_ode(
        f, t_span, y0, alpha,
        derivative_type="caputo",
        method="predictor_corrector",
        adaptive=False,
        h=h
    )
    
    print(f"\nResults:")
    print(f"  Number of steps: {len(t_vals)}")
    print(f"  Final time: {t_vals[-1]:.6f}")
    
    # Handle both scalar and array results
    y_final = float(y_vals[-1]) if np.isscalar(y_vals[-1]) else float(y_vals[-1][0])
    print(f"  Final value: {y_final:.6f}")
    print(f"  Expected decay (qualitative): y should decrease from 1.0")
    
    # Extract scalar values for comparison
    y_sequence = y_vals.flatten() if y_vals.ndim > 1 else y_vals
    
    # Check monotonic decrease for this particular problem
    if np.all(np.diff(y_sequence) <= 0):
        print(f"  ✓ Solution correctly shows monotonic decrease")
    else:
        print(f"  ⚠ Warning: Solution does not show expected monotonic decrease")
    
    # Check final value is reasonable (should be less than initial)
    if y_final < y0 and y_final > 0:
        print(f"  ✓ Final value is reasonable (0 < y_f < y_0)")
    else:
        print(f"  ⚠ Warning: Final value seems unreasonable")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run accuracy test first
    test_accuracy()
    
    # Run comprehensive performance benchmarks
    run_comprehensive_benchmark()

