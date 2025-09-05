"""
Comprehensive benchmark script for stochastic memory sampling.

Tests multiple scenarios: different alpha values, sequence lengths, and methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path

from hpfracc.ml.stochastic_memory_sampling import (
    ImportanceSampler, StratifiedSampler, ControlVariateSampler
)


class ComprehensiveStochasticBenchmark:
    """Comprehensive benchmark for stochastic memory sampling."""
    
    def __init__(self, device: str = "cpu", seed: int = 42):
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Test configurations
        self.alpha_values = [0.3, 0.5, 0.7]
        self.k_values = [8, 16, 32, 64, 128]
        self.methods = ["importance", "stratified", "control_variate"]
        self.sequence_lengths = [64, 128, 256]
        self.n_trials = 10
        
        self.results = {}
    
    def generate_test_sequence(self, length: int, signal_type: str = "smooth") -> torch.Tensor:
        """Generate different types of test sequences."""
        x = torch.linspace(0, 4 * np.pi, length, device=self.device)
        
        if signal_type == "smooth":
            # Smooth signal: sin(x) + small noise
            signal = torch.sin(x) + 0.1 * torch.randn_like(x)
        elif signal_type == "noisy":
            # Noisy signal: sin(x) + large noise
            signal = torch.sin(x) + 0.5 * torch.randn_like(x)
        elif signal_type == "step":
            # Step function with noise
            signal = torch.where(x < 2 * np.pi, torch.ones_like(x), -torch.ones_like(x))
            signal += 0.1 * torch.randn_like(x)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        return signal
    
    def compute_deterministic_reference(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute deterministic reference using full history."""
        n = len(x)
        result = torch.zeros_like(x)
        
        # Simple Grünwald-Letnikov approximation
        for i in range(n):
            for j in range(i + 1):
                if j == 0:
                    coeff = 1.0
                else:
                    coeff = coeff * (j - 1 - alpha) / j
                result[i] += coeff * x[i - j]
        
        return result
    
    def benchmark_configuration(self, x: torch.Tensor, alpha: float, method: str, k: int) -> Dict:
        """Benchmark a single configuration."""
        estimates = []
        times = []
        
        for trial in range(self.n_trials):
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
            
            # Sample and estimate
            indices = sampler.sample_indices(len(x), k)
            weights = sampler.compute_weights(indices, len(x))
            estimate = sampler.estimate_derivative(x, indices, weights)
            
            end_time = time.time()
            
            estimates.append(estimate.item())
            times.append(end_time - start_time)
        
        # Compute reference for error calculation
        reference = self.compute_deterministic_reference(x, alpha)
        reference_value = reference[-1].item()
        
        # Compute statistics
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates)
        mean_error = np.mean([abs(est - reference_value) for est in estimates])
        mean_time = np.mean(times)
        
        return {
            'mean_estimate': mean_estimate,
            'std_estimate': std_estimate,
            'variance': np.var(estimates),
            'mean_error': mean_error,
            'mean_time': mean_time,
            'reference': reference_value,
            'estimates': estimates,
            'times': times
        }
    
    def run_benchmark(self, signal_type: str = "smooth") -> Dict:
        """Run comprehensive benchmark."""
        print(f"Running comprehensive stochastic sampling benchmark ({signal_type} signal)...")
        print(f"Alpha values: {self.alpha_values}")
        print(f"K values: {self.k_values}")
        print(f"Methods: {self.methods}")
        print(f"Sequence lengths: {self.sequence_lengths}")
        print(f"Trials per configuration: {self.n_trials}")
        
        for alpha in self.alpha_values:
            print(f"\nTesting alpha = {alpha}")
            self.results[alpha] = {}
            
            for length in self.sequence_lengths:
                print(f"  Sequence length = {length}")
                self.results[alpha][length] = {}
                
                # Generate test sequence
                x = self.generate_test_sequence(length, signal_type)
                
                for method in self.methods:
                    print(f"    Method = {method}")
                    self.results[alpha][length][method] = {}
                    
                    for k in self.k_values:
                        print(f"      K = {k}")
                        result = self.benchmark_configuration(x, alpha, method, k)
                        self.results[alpha][length][method][k] = result
        
        return self.results
    
    def plot_results(self, signal_type: str = "smooth", save_path: str = None):
        """Plot comprehensive results."""
        if save_path is None:
            save_path = f"comprehensive_stochastic_benchmark_{signal_type}.png"
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Comprehensive Stochastic Sampling Benchmark ({signal_type} signal)', fontsize=16)
        
        # Plot 1: Error vs K (different alphas)
        ax1 = axes[0, 0]
        for alpha in self.alpha_values:
            for method in self.methods:
                k_vals = []
                errors = []
                for k in self.k_values:
                    length = 128  # Use middle length
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
                    length = 128
                    if k in self.results[alpha][length][method]:
                        k_vals.append(k)
                        variances.append(self.results[alpha][length][method][k]['variance'])
                
                ax2.plot(k_vals, variances, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax2.set_xlabel('K (Number of Samples)')
        ax2.set_ylabel('Variance')
        ax2.set_title('Variance vs K')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs K
        ax3 = axes[0, 2]
        for alpha in self.alpha_values:
            for method in self.methods:
                k_vals = []
                times = []
                for k in self.k_values:
                    length = 128
                    if k in self.results[alpha][length][method]:
                        k_vals.append(k)
                        times.append(self.results[alpha][length][method][k]['mean_time'])
                
                ax3.plot(k_vals, times, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax3.set_xlabel('K (Number of Samples)')
        ax3.set_ylabel('Mean Time (s)')
        ax3.set_title('Computation Time vs K')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error vs Sequence Length
        ax4 = axes[1, 0]
        for alpha in self.alpha_values:
            for method in self.methods:
                lengths = []
                errors = []
                for length in self.sequence_lengths:
                    k = 32  # Use middle K value
                    if k in self.results[alpha][length][method]:
                        lengths.append(length)
                        errors.append(self.results[alpha][length][method][k]['mean_error'])
                
                ax4.plot(lengths, errors, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Error vs Sequence Length')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Method Comparison (Error)
        ax5 = axes[1, 1]
        alpha = 0.5  # Use middle alpha
        length = 128
        for method in self.methods:
            k_vals = []
            errors = []
            for k in self.k_values:
                if k in self.results[alpha][length][method]:
                    k_vals.append(k)
                    errors.append(self.results[alpha][length][method][k]['mean_error'])
            
            ax5.plot(k_vals, errors, 'o-', label=method, linewidth=2, markersize=6)
        
        ax5.set_xlabel('K (Number of Samples)')
        ax5.set_ylabel('Mean Absolute Error')
        ax5.set_title(f'Method Comparison (α={alpha}, length={length})')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Efficiency (Error vs Time)
        ax6 = axes[1, 2]
        for alpha in self.alpha_values:
            for method in self.methods:
                times = []
                errors = []
                for k in self.k_values:
                    length = 128
                    if k in self.results[alpha][length][method]:
                        times.append(self.results[alpha][length][method][k]['mean_time'])
                        errors.append(self.results[alpha][length][method][k]['mean_error'])
                
                ax6.plot(times, errors, 'o-', label=f'{method} (α={alpha})', alpha=0.7)
        
        ax6.set_xlabel('Mean Time (s)')
        ax6.set_ylabel('Mean Absolute Error')
        ax6.set_title('Efficiency: Error vs Time')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plotted and saved to {save_path}")
    
    def save_results(self, signal_type: str = "smooth", save_path: str = None):
        """Save results to JSON file."""
        if save_path is None:
            save_path = f"comprehensive_stochastic_results_{signal_type}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
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
    
    def print_summary(self, signal_type: str = "smooth"):
        """Print comprehensive summary."""
        print("\n" + "="*70)
        print(f"COMPREHENSIVE STOCHASTIC SAMPLING BENCHMARK SUMMARY ({signal_type.upper()})")
        print("="*70)
        
        for alpha in self.alpha_values:
            print(f"\nAlpha = {alpha}")
            print("-" * 50)
            
            for method in self.methods:
                print(f"\nMethod: {method.upper()}")
                
                # Find best configuration for this method
                best_error = float('inf')
                best_config = None
                
                for length in self.sequence_lengths:
                    for k in self.k_values:
                        if k in self.results[alpha][length][method]:
                            error = self.results[alpha][length][method][k]['mean_error']
                            if error < best_error:
                                best_error = error
                                best_config = (length, k)
                
                if best_config:
                    length, k = best_config
                    result = self.results[alpha][length][method][k]
                    print(f"  Best config: length={length}, K={k}")
                    print(f"  Error: {result['mean_error']:.6f}")
                    print(f"  Variance: {result['variance']:.6f}")
                    print(f"  Time: {result['mean_time']:.6f}s")
                    print(f"  Reference: {result['reference']:.6f}")
                    print(f"  Estimate: {result['mean_estimate']:.6f} ± {result['std_estimate']:.6f}")


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Stochastic Memory Sampling Benchmark')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--signal-type', type=str, default='smooth', 
                       choices=['smooth', 'noisy', 'step'],
                       help='Type of test signal')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Path to save results JSON')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save plot')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = ComprehensiveStochasticBenchmark(device=args.device, seed=args.seed)
    
    # Run benchmark
    results = benchmark.run_benchmark(signal_type=args.signal_type)
    
    # Save and display results
    benchmark.save_results(args.signal_type, args.save_results)
    benchmark.print_summary(args.signal_type)
    benchmark.plot_results(args.signal_type, args.save_plot)
    
    print(f"\nComprehensive benchmark completed!")


if __name__ == "__main__":
    main()
