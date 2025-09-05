"""
Quick benchmark script for stochastic memory sampling.

A lightweight version for rapid testing of K vs error/variance trade-offs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List

from hpfracc.ml.stochastic_memory_sampling import (
    ImportanceSampler, StratifiedSampler, ControlVariateSampler
)


def quick_benchmark():
    """Run a quick benchmark of stochastic sampling methods."""
    print("Running quick stochastic sampling benchmark...")
    
    # Test configuration
    device = "cpu"
    alpha = 0.5
    length = 128
    k_values = [8, 16, 32, 64, 128]
    methods = ["importance", "stratified", "control_variate"]
    n_trials = 5
    
    # Generate test sequence
    torch.manual_seed(42)
    x = torch.linspace(0, 4 * np.pi, length, device=device)
    signal = torch.sin(x) + 0.1 * torch.randn_like(x)
    
    print(f"Test sequence length: {length}")
    print(f"Alpha: {alpha}")
    print(f"K values: {k_values}")
    print(f"Methods: {methods}")
    print(f"Trials per configuration: {n_trials}")
    
    results = {}
    
    for method in methods:
        print(f"\nTesting {method} sampling...")
        results[method] = {'k': [], 'error': [], 'variance': [], 'time': []}
        
        for k in k_values:
            print(f"  K = {k}")
            
            estimates = []
            times = []
            
            for trial in range(n_trials):
                start_time = time.time()
                
                # Create sampler
                if method == "importance":
                    sampler = ImportanceSampler(alpha=alpha, k=k)
                elif method == "stratified":
                    sampler = StratifiedSampler(alpha=alpha, k=k)
                elif method == "control_variate":
                    sampler = ControlVariateSampler(alpha=alpha, k=k)
                
                # Sample and estimate
                indices = sampler.sample_indices(length, k)
                weights = sampler.compute_weights(indices, length)
                estimate = sampler.estimate_derivative(signal, indices, weights)
                
                end_time = time.time()
                
                estimates.append(estimate.item())
                times.append(end_time - start_time)
            
            # Compute statistics
            mean_estimate = np.mean(estimates)
            variance = np.var(estimates)
            mean_time = np.mean(times)
            
            # Simple error estimate (compare with deterministic approximation)
            # For this quick benchmark, use the mean of estimates as reference
            error = np.std(estimates)  # Use standard deviation as error proxy
            
            results[method]['k'].append(k)
            results[method]['error'].append(error)
            results[method]['variance'].append(variance)
            results[method]['time'].append(mean_time)
            
            print(f"    Error: {error:.6f}, Variance: {variance:.6f}, Time: {mean_time:.6f}s")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Quick Stochastic Sampling Benchmark (α={alpha}, length={length})', fontsize=14)
    
    # Error vs K
    ax1 = axes[0]
    for method in methods:
        ax1.plot(results[method]['k'], results[method]['error'], 'o-', label=method)
    ax1.set_xlabel('K (Number of Samples)')
    ax1.set_ylabel('Error (std of estimates)')
    ax1.set_title('Error vs K')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variance vs K
    ax2 = axes[1]
    for method in methods:
        ax2.plot(results[method]['k'], results[method]['variance'], 'o-', label=method)
    ax2.set_xlabel('K (Number of Samples)')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance vs K')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Time vs K
    ax3 = axes[2]
    for method in methods:
        ax3.plot(results[method]['k'], results[method]['time'], 'o-', label=method)
    ax3.set_xlabel('K (Number of Samples)')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Computation Time vs K')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_stochastic_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("QUICK BENCHMARK SUMMARY")
    print("="*50)
    
    for method in methods:
        print(f"\n{method.upper()} SAMPLING:")
        print("-" * 30)
        
        # Find best K (lowest error)
        best_idx = np.argmin(results[method]['error'])
        best_k = results[method]['k'][best_idx]
        best_error = results[method]['error'][best_idx]
        best_variance = results[method]['variance'][best_idx]
        best_time = results[method]['time'][best_idx]
        
        print(f"Best K: {best_k}")
        print(f"Error: {best_error:.6f}")
        print(f"Variance: {best_variance:.6f}")
        print(f"Time: {best_time:.6f}s")
        
        # Show trend
        if len(results[method]['k']) > 1:
            error_improvement = (results[method]['error'][0] - results[method]['error'][-1]) / results[method]['error'][0] * 100
            print(f"Error improvement (K={k_values[0]} to K={k_values[-1]}): {error_improvement:.1f}%")
    
    print(f"\nPlot saved to: quick_stochastic_benchmark.png")
    
    return results


if __name__ == "__main__":
    results = quick_benchmark()
