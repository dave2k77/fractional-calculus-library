"""
Benchmark script for stochastic memory sampling: K vs error/variance analysis.

This script evaluates the trade-offs between computational cost (K samples)
and accuracy/variance in stochastic fractional derivative estimation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import argparse
import json
from pathlib import Path

from hpfracc.ml.stochastic_memory_sampling import (
    ImportanceSampler, StratifiedSampler, ControlVariateSampler,
    StochasticFractionalDerivative
)
from hpfracc.ml.probabilistic_fractional_orders import (
    ProbabilisticFractionalOrder, create_normal_alpha_layer
)
from torch.distributions import Normal


class StochasticSamplingBenchmark:
    """Benchmark stochastic memory sampling methods across different K values."""
    
    def __init__(self, device: str = "cpu", seed: int = 42):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Test configurations
        self.alpha_values = [0.3, 0.5, 0.7]
        self.k_values = [8, 16, 32, 64, 128, 256]
        self.methods = ["importance", "stratified", "control_variate"]
        self.sequence_lengths = [64, 128, 256]
        
        # Results storage
        self.results = {}
    
    def generate_test_sequence(self, length: int, alpha: float) -> torch.Tensor:
        """Generate a test sequence with known fractional derivative."""
        # Generate a smooth test function: sin(x) + noise
        x = torch.linspace(0, 4 * np.pi, length, device=self.device)
        signal = torch.sin(x) + 0.1 * torch.randn_like(x)
        return signal
    
    def compute_reference_derivative(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute reference fractional derivative using full history (deterministic)."""
        # For benchmarking, use a simple Grünwald-Letnikov approximation
        n = len(x)
        result = torch.zeros_like(x)
        
        for i in range(n):
            for j in range(i + 1):
                if j == 0:
                    coeff = 1.0
                else:
                    coeff = coeff * (j - 1 - alpha) / j
                result[i] += coeff * x[i - j]
        
        return result
    
    def benchmark_method(self, x: torch.Tensor, alpha: float, method: str, k: int, 
                        n_trials: int = 10) -> Dict:
        """Benchmark a single method configuration."""
        results = {
            'method': method,
            'k': k,
            'alpha': alpha,
            'times': [],
            'estimates': [],
            'errors': [],
            'variances': []
        }
        
        # Compute reference
        reference = self.compute_reference_derivative(x, alpha)
        
        for trial in range(n_trials):
            start_time = time.time()
            
            # Create sampler
            if method == "importance":
                sampler = ImportanceSampler(alpha=alpha, k=k)
            elif method == "stratified":
                sampler = StratifiedSampler(alpha=alpha, k=k)
            elif method == "control_variate":
                sampler = ControlVariateSampler(alpha=alpha, k=k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Sample indices and weights
            indices = sampler.sample_indices(len(x), k)
            weights = sampler.compute_weights(indices, len(x))
            
            # Estimate derivative
            estimate = sampler.estimate_derivative(x, indices, weights)
            
            end_time = time.time()
            
            # Store results
            results['times'].append(end_time - start_time)
            results['estimates'].append(estimate.item() if estimate.dim() == 0 else estimate[-1].item())
            
            # Compute error (for scalar estimate, compare with last point of reference)
            error = abs(estimate.item() - reference[-1].item())
            results['errors'].append(error)
        
        # Compute statistics
        results['mean_time'] = np.mean(results['times'])
        results['std_time'] = np.std(results['times'])
        results['mean_error'] = np.mean(results['errors'])
        results['std_error'] = np.std(results['errors'])
        results['variance'] = np.var(results['estimates'])
        
        return results
    
    def run_benchmark(self, n_trials: int = 10) -> Dict:
        """Run complete benchmark across all configurations."""
        print("Starting stochastic sampling benchmark...")
        
        for alpha in self.alpha_values:
            print(f"  Testing alpha = {alpha}")
            self.results[alpha] = {}
            
            for length in self.sequence_lengths:
                print(f"    Sequence length = {length}")
                self.results[alpha][length] = {}
                
                # Generate test sequence
                x = self.generate_test_sequence(length, alpha)
                
                for method in self.methods:
                    print(f"      Method = {method}")
                    self.results[alpha][length][method] = {}
                    
                    for k in self.k_values:
                        print(f"        K = {k}")
                        result = self.benchmark_method(x, alpha, method, k, n_trials)
                        self.results[alpha][length][method][k] = result
        
        return self.results
    
    def plot_results(self, save_path: str = "stochastic_sampling_benchmark.png"):
        """Plot benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stochastic Memory Sampling Benchmark: K vs Performance', fontsize=16)
        
        # Plot 1: Error vs K
        ax1 = axes[0, 0]
        for alpha in self.alpha_values:
            for method in self.methods:
                k_vals = []
                errors = []
                for k in self.k_values:
                    # Use middle sequence length for plotting
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    if k in self.results[alpha][length][method]:
                        k_vals.append(k)
                        errors.append(self.results[alpha][length][method][k]['mean_error'])
                
                ax1.plot(k_vals, errors, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax1.set_xlabel('K (Number of Samples)')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.set_title('Error vs K')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance vs K
        ax2 = axes[0, 1]
        for alpha in self.alpha_values:
            for method in self.methods:
                k_vals = []
                variances = []
                for k in self.k_values:
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    if k in self.results[alpha][length][method]:
                        k_vals.append(k)
                        variances.append(self.results[alpha][length][method][k]['variance'])
                
                ax2.plot(k_vals, variances, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax2.set_xlabel('K (Number of Samples)')
        ax2.set_ylabel('Variance of Estimates')
        ax2.set_title('Variance vs K')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs K
        ax3 = axes[1, 0]
        for alpha in self.alpha_values:
            for method in self.methods:
                k_vals = []
                times = []
                for k in self.k_values:
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    if k in self.results[alpha][length][method]:
                        k_vals.append(k)
                        times.append(self.results[alpha][length][method][k]['mean_time'])
                
                ax3.plot(k_vals, times, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax3.set_xlabel('K (Number of Samples)')
        ax3.set_ylabel('Mean Computation Time (s)')
        ax3.set_title('Computation Time vs K')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error vs Time (efficiency)
        ax4 = axes[1, 1]
        for alpha in self.alpha_values:
            for method in self.methods:
                times = []
                errors = []
                for k in self.k_values:
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    if k in self.results[alpha][length][method]:
                        times.append(self.results[alpha][length][method][k]['mean_time'])
                        errors.append(self.results[alpha][length][method][k]['mean_error'])
                
                ax4.plot(times, errors, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax4.set_xlabel('Computation Time (s)')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Error vs Time (Efficiency)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plotted and saved to {save_path}")
    
    def save_results(self, save_path: str = "stochastic_sampling_results.json"):
        """Save benchmark results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Deep convert the results
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        converted_results = deep_convert(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("STOCHASTIC SAMPLING BENCHMARK SUMMARY")
        print("="*60)
        
        for alpha in self.alpha_values:
            print(f"\nAlpha = {alpha}")
            print("-" * 40)
            
            for method in self.methods:
                print(f"\nMethod: {method}")
                
                # Find best K for this method (lowest error)
                best_k = None
                best_error = float('inf')
                
                for k in self.k_values:
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    if k in self.results[alpha][length][method]:
                        error = self.results[alpha][length][method][k]['mean_error']
                        if error < best_error:
                            best_error = error
                            best_k = k
                
                if best_k is not None:
                    length = self.sequence_lengths[len(self.sequence_lengths)//2]
                    result = self.results[alpha][length][method][best_k]
                    print(f"  Best K: {best_k}")
                    print(f"  Error: {result['mean_error']:.6f} ± {result['std_error']:.6f}")
                    print(f"  Variance: {result['variance']:.6f}")
                    print(f"  Time: {result['mean_time']:.6f} ± {result['std_time']:.6f} s")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description='Stochastic Memory Sampling Benchmark')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials per configuration')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-results', type=str, default='stochastic_sampling_results.json', 
                       help='Path to save results JSON')
    parser.add_argument('--save-plot', type=str, default='stochastic_sampling_benchmark.png',
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = StochasticSamplingBenchmark(device=args.device, seed=args.seed)
    
    # Run benchmark
    results = benchmark.run_benchmark(n_trials=args.trials)
    
    # Save and display results
    benchmark.save_results(args.save_results)
    benchmark.print_summary()
    benchmark.plot_results(args.save_plot)
    
    print(f"\nBenchmark completed! Results saved to {args.save_results}")
    print(f"Plot saved to {args.save_plot}")


if __name__ == "__main__":
    main()
