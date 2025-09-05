"""
Test script for comprehensive stochastic benchmark with reduced configuration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple

from hpfracc.ml.stochastic_memory_sampling import (
    ImportanceSampler, StratifiedSampler, ControlVariateSampler
)


def test_comprehensive_benchmark():
    """Test comprehensive benchmark with reduced configuration."""
    print("Testing comprehensive stochastic sampling benchmark...")
    
    # Reduced test configuration
    device = "cpu"
    alpha_values = [0.5]  # Just one alpha
    k_values = [8, 16, 32, 64]  # Fewer K values
    methods = ["importance", "stratified", "control_variate"]
    sequence_lengths = [64, 128]  # Fewer lengths
    n_trials = 3  # Fewer trials
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = {}
    
    for alpha in alpha_values:
        print(f"\nTesting alpha = {alpha}")
        results[alpha] = {}
        
        for length in sequence_lengths:
            print(f"  Sequence length = {length}")
            results[alpha][length] = {}
            
            # Generate test sequence
            x = torch.linspace(0, 4 * np.pi, length, device=device)
            signal = torch.sin(x) + 0.1 * torch.randn_like(x)
            
            for method in methods:
                print(f"    Method = {method}")
                results[alpha][length][method] = {}
                
                for k in k_values:
                    print(f"      K = {k}")
                    
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
                    std_estimate = np.std(estimates)
                    variance = np.var(estimates)
                    mean_time = np.mean(times)
                    
                    results[alpha][length][method][k] = {
                        'mean_estimate': mean_estimate,
                        'std_estimate': std_estimate,
                        'variance': variance,
                        'mean_time': mean_time,
                        'estimates': estimates,
                        'times': times
                    }
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Test Comprehensive Stochastic Sampling Benchmark', fontsize=14)
    
    alpha = 0.5
    length = 128
    
    # Error vs K
    ax1 = axes[0]
    for method in methods:
        k_vals = []
        errors = []
        for k in k_values:
            if k in results[alpha][length][method]:
                std_est = results[alpha][length][method][k]['std_estimate']
                errors.append(std_est)
                k_vals.append(k)
        
        ax1.plot(k_vals, errors, 'o-', label=method)
    
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
        k_vals = []
        variances = []
        for k in k_values:
            if k in results[alpha][length][method]:
                variance = results[alpha][length][method][k]['variance']
                variances.append(variance)
                k_vals.append(k)
        
        ax2.plot(k_vals, variances, 'o-', label=method)
    
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
        k_vals = []
        times = []
        for k in k_values:
            if k in results[alpha][length][method]:
                time_val = results[alpha][length][method][k]['mean_time']
                times.append(time_val)
                k_vals.append(k)
        
        ax3.plot(k_vals, times, 'o-', label=method)
    
    ax3.set_xlabel('K (Number of Samples)')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('Computation Time vs K')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_comprehensive_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("TEST COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*50)
    
    for method in methods:
        print(f"\n{method.upper()} SAMPLING:")
        print("-" * 30)
        
        # Find best K (lowest error)
        best_error = float('inf')
        best_k = None
        
        for k in k_values:
            if k in results[alpha][length][method]:
                error = results[alpha][length][method][k]['std_estimate']
                if error < best_error:
                    best_error = error
                    best_k = k
        
        if best_k is not None:
            result = results[alpha][length][method][best_k]
            print(f"Best K: {best_k}")
            print(f"Error: {result['std_estimate']:.6f}")
            print(f"Variance: {result['variance']:.6f}")
            print(f"Time: {result['mean_time']:.6f}s")
    
    print(f"\nPlot saved to: test_comprehensive_benchmark.png")
    
    return results


if __name__ == "__main__":
    results = test_comprehensive_benchmark()
