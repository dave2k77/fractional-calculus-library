#!/usr/bin/env python3
"""
Stress test for FFT-optimized ODE solver with large step counts.

This demonstrates the performance difference between O(N²) and O(N log N)
scaling for very large problems.
"""

import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hpfracc.solvers.ode_solvers import solve_fractional_ode


def stress_test_large_problem():
    """
    Stress test with progressively larger problems to demonstrate O(N log N) scaling.
    """
    
    print("\n" + "="*70)
    print("FFT OPTIMIZATION STRESS TEST - LARGE PROBLEMS")
    print("="*70)
    print("\nTesting with very large step counts to demonstrate FFT advantage.")
    print("Expected: Sub-quadratic scaling (ideally O(N log N))")
    print()
    
    # Simple test ODE
    def f(t, y):
        return -0.5 * y
    
    alpha = 0.75
    t0, tf = 0.0, 10.0
    y0 = 1.0
    
    # Test with increasingly larger step counts
    step_counts = [100, 200, 500, 1000, 2000, 5000]
    
    results = {
        'N': [],
        'time': [],
        'time_per_step': []
    }
    
    print(f"{'N Steps':<10} {'Time (s)':<12} {'Time/Step (ms)':<18} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_time_per_step = None
    
    for N in step_counts:
        h = (tf - t0) / N
        
        start = time.perf_counter()
        t_vals, y_vals = solve_fractional_ode(
            f, (t0, tf), y0, alpha,
            derivative_type="caputo",
            method="predictor_corrector",
            adaptive=False,
            h=h
        )
        elapsed = time.perf_counter() - start
        
        time_per_step = elapsed / N * 1000  # ms per step
        
        if baseline_time_per_step is None:
            baseline_time_per_step = time_per_step
            speedup_str = "baseline"
        else:
            # In O(N²), time per step grows linearly with N
            # In O(N log N), time per step grows as log N
            expected_o_n2 = baseline_time_per_step * (N / step_counts[0])
            actual = time_per_step
            speedup = expected_o_n2 / actual
            speedup_str = f"{speedup:.2f}x"
        
        results['N'].append(N)
        results['time'].append(elapsed)
        results['time_per_step'].append(time_per_step)
        
        print(f"{N:<10} {elapsed:<12.4f} {time_per_step:<18.6f} {speedup_str:<10}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    N_arr = np.array(results['N'])
    time_arr = np.array(results['time'])
    
    # Plot 1: Total time vs N (log-log)
    ax1 = axes[0]
    ax1.loglog(N_arr, time_arr, 'bo-', linewidth=2, markersize=8, label='Actual')
    
    # Reference lines
    t0_nlogn = time_arr[0] / (N_arr[0] * np.log(N_arr[0]))
    t0_n2 = time_arr[0] / (N_arr[0]**2)
    
    T_nlogn = t0_nlogn * N_arr * np.log(N_arr)
    T_n2 = t0_n2 * N_arr**2
    
    ax1.loglog(N_arr, T_nlogn, 'g--', alpha=0.7, linewidth=2, label='O(N log N) reference')
    ax1.loglog(N_arr, T_n2, 'r--', alpha=0.7, linewidth=2, label='O(N²) reference')
    
    ax1.set_xlabel('Number of Steps (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Time (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time per step vs N
    ax2 = axes[1]
    time_per_step_arr = np.array(results['time_per_step'])
    ax2.semilogx(N_arr, time_per_step_arr, 'mo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Steps (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time per Step (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Amortized Cost per Step', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speedup factor vs N
    ax3 = axes[2]
    # Compute speedup relative to O(N²) expectation
    baseline_ratio = time_arr[0] / (N_arr[0]**2)
    expected_o_n2_times = baseline_ratio * N_arr**2
    speedup_factors = expected_o_n2_times / time_arr
    
    ax3.semilogx(N_arr, speedup_factors, 'co-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('Number of Steps (N)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Speedup Factor vs O(N²)', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Gain from FFT', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'fft_stress_test_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {filename}")
    plt.close()
    
    # Estimate empirical scaling exponent
    log_N = np.log(N_arr)
    log_T = np.log(time_arr)
    
    # Fit: log(T) = p * log(N) + c  =>  T ~ N^p
    p = np.polyfit(log_N, log_T, 1)[0]
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"\nEmpirical complexity: O(N^{p:.2f})")
    print(f"\nLargest problem solved: N = {max(results['N'])} steps")
    print(f"Time for largest problem: {max(results['time']):.4f} seconds")
    
    if p < 1.3:
        print(f"\n✓ EXCELLENT! Scaling is very close to O(N log N) ≈ O(N^1.1)")
    elif p < 1.6:
        print(f"\n✓ VERY GOOD! Scaling is well below O(N^1.6)")
    elif p < 1.8:
        print(f"\n✓ GOOD! Scaling is sub-quadratic")
    elif p < 2.1:
        print(f"\n⚠ Approaching O(N²) - FFT optimization may need tuning")
    else:
        print(f"\n⚠ WARNING: Scaling appears super-quadratic")
    
    # Compare final speedup
    final_speedup = speedup_factors[-1]
    print(f"\nSpeedup at largest N: {final_speedup:.2f}x faster than O(N²) would be")
    print(f"This means FFT optimization avoids ~{(1 - 1/final_speedup)*100:.1f}% of O(N²) overhead")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    stress_test_large_problem()

