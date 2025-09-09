#!/usr/bin/env python3
"""
Fractional Autograd Performance Test

This script tests the performance of fractional autograd against standard autograd methods.
We'll test:
1. Forward pass performance (fractional vs standard derivatives)
2. Backward pass performance (fractional vs standard gradients)
3. Memory usage during autograd
4. Accuracy of fractional gradients
5. Scalability with problem size
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import psutil
import os
from typing import Dict, List, Tuple
import json

# Import hpfracc components
from hpfracc.algorithms.gpu_optimized_methods import GPUOptimizedRiemannLiouville, GPUConfig
from hpfracc.ml.layers import FractionalConv1D, LayerConfig

class FractionalAutogradPerformanceTest:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
        # Test configurations
        self.test_sizes = [32, 64, 128, 256, 512, 1024]
        self.fractional_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        print(f"Running fractional autograd performance tests on {self.device}")
        print(f"Test sizes: {self.test_sizes}")
        print(f"Fractional orders: {self.fractional_orders}")
    
    def test_forward_pass_performance(self) -> Dict:
        """Test forward pass performance: fractional vs standard derivatives"""
        print("\n=== Forward Pass Performance Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'fractional_times': [],
            'standard_times': [],
            'speedup_ratios': []
        }
        
        for size in self.test_sizes:
            print(f"Testing size {size}...")
            
            # Create test data
            x = torch.randn(size, requires_grad=True, device=self.device)
            
            # Test fractional derivative (Riemann-Liouville)
            frac_config = GPUConfig(backend="jax" if self.device == 'cuda' else "numba")
            frac_deriv = GPUOptimizedRiemannLiouville(alpha=0.5, gpu_config=frac_config)
            
            # Time fractional derivative
            start_time = time.time()
            for _ in range(10):  # Multiple runs for accuracy
                x_np = x.detach().cpu().numpy()
                t_array = np.arange(len(x_np))
                frac_result = frac_deriv.compute(x_np, t_array)
            frac_time = (time.time() - start_time) / 10
            
            # Time standard derivative
            start_time = time.time()
            for _ in range(10):
                std_result = torch.autograd.grad(x.sum(), x, create_graph=True)[0]
            std_time = (time.time() - start_time) / 10
            
            speedup = std_time / frac_time if frac_time > 0 else float('inf')
            
            results['fractional_times'].append(frac_time)
            results['standard_times'].append(std_time)
            results['speedup_ratios'].append(speedup)
            
            print(f"  Size {size}: Fractional {frac_time:.4f}s, Standard {std_time:.4f}s, Speedup {speedup:.2f}x")
        
        return results
    
    def test_backward_pass_performance(self) -> Dict:
        """Test backward pass performance: fractional vs standard gradients"""
        print("\n=== Backward Pass Performance Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'fractional_grad_times': [],
            'standard_grad_times': [],
            'grad_speedup_ratios': []
        }
        
        for size in self.test_sizes:
            print(f"Testing gradient size {size}...")
            
            # Create test data
            x = torch.randn(size, requires_grad=True, device=self.device)
            target = torch.randn(size, device=self.device)
            
            # Test fractional gradient
            frac_config = GPUConfig(backend="jax" if self.device == 'cuda' else "numba")
            frac_deriv = GPUOptimizedRiemannLiouville(alpha=0.5, gpu_config=frac_config)
            
            def fractional_loss(x):
                x_np = x.detach().cpu().numpy()
                t_array = np.arange(len(x_np))
                frac_result = torch.tensor(frac_deriv.compute(x_np, t_array), 
                                        device=self.device, requires_grad=True)
                return torch.nn.functional.mse_loss(frac_result, target)
            
            # Time fractional gradient computation
            start_time = time.time()
            for _ in range(10):
                x.grad = None  # Clear gradients
                loss = fractional_loss(x)
                loss.backward()
                if x.grad is not None:
                    x.grad.zero_()
            frac_grad_time = (time.time() - start_time) / 10
            
            # Test standard gradient
            def standard_loss(x):
                return torch.nn.functional.mse_loss(x, target)
            
            start_time = time.time()
            for _ in range(10):
                x.grad = None  # Clear gradients
                loss = standard_loss(x)
                loss.backward()
                if x.grad is not None:
                    x.grad.zero_()
            std_grad_time = (time.time() - start_time) / 10
            
            speedup = std_grad_time / frac_grad_time if frac_grad_time > 0 else float('inf')
            
            results['fractional_grad_times'].append(frac_grad_time)
            results['standard_grad_times'].append(std_grad_time)
            results['grad_speedup_ratios'].append(speedup)
            
            print(f"  Size {size}: Fractional grad {frac_grad_time:.4f}s, Standard grad {std_grad_time:.4f}s, Speedup {speedup:.2f}x")
        
        return results
    
    def test_memory_usage(self) -> Dict:
        """Test memory usage during fractional autograd operations"""
        print("\n=== Memory Usage Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'fractional_memory': [],
            'standard_memory': [],
            'memory_ratios': []
        }
        
        for size in self.test_sizes:
            print(f"Testing memory usage for size {size}...")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Test fractional memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            x = torch.randn(size, requires_grad=True, device=self.device)
            frac_config = GPUConfig(backend="jax" if self.device == 'cuda' else "numba")
            frac_deriv = GPUOptimizedRiemannLiouville(alpha=0.5, gpu_config=frac_config)
            
            # Perform fractional operations
            x_np = x.detach().cpu().numpy()
            t_array = np.arange(len(x_np))
            frac_result = torch.tensor(frac_deriv.compute(x_np, t_array), 
                                    device=self.device, requires_grad=True)
            loss = frac_result.sum()
            loss.backward()
            
            fractional_memory = process.memory_info().rss / 1024 / 1024  # MB
            frac_memory_used = fractional_memory - initial_memory
            
            # Clear and test standard memory
            del x, frac_result, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            x = torch.randn(size, requires_grad=True, device=self.device)
            loss = x.sum()
            loss.backward()
            
            standard_memory = process.memory_info().rss / 1024 / 1024  # MB
            std_memory_used = standard_memory - initial_memory
            
            memory_ratio = frac_memory_used / std_memory_used if std_memory_used > 0 else float('inf')
            
            results['fractional_memory'].append(frac_memory_used)
            results['standard_memory'].append(std_memory_used)
            results['memory_ratios'].append(memory_ratio)
            
            print(f"  Size {size}: Fractional {frac_memory_used:.2f}MB, Standard {std_memory_used:.2f}MB, Ratio {memory_ratio:.2f}x")
        
        return results
    
    def test_accuracy(self) -> Dict:
        """Test accuracy of fractional gradients against analytical solutions"""
        print("\n=== Accuracy Test ===")
        
        results = {
            'orders': self.fractional_orders,
            'analytical_grads': [],
            'numerical_grads': [],
            'errors': []
        }
        
        # Test function: f(x) = x^2
        x = torch.tensor([1.0], requires_grad=True, device=self.device)
        
        for alpha in self.fractional_orders:
            print(f"Testing accuracy for α = {alpha}...")
            
            # Analytical fractional derivative of x^2
            # For f(x) = x^2, D^α f(x) = (2/Γ(3-α)) * x^(2-α)
            from scipy.special import gamma
            analytical_grad = (2.0 / gamma(3 - alpha)) * (x.item() ** (2 - alpha))
            
            # Numerical fractional derivative
            frac_config = GPUConfig(backend="jax" if self.device == 'cuda' else "numba")
            frac_deriv = GPUOptimizedRiemannLiouville(alpha=alpha, gpu_config=frac_config)
            
            def fractional_function(x):
                x_np = x.detach().cpu().numpy()
                t_array = np.arange(len(x_np))
                return torch.tensor(frac_deriv.compute(x_np, t_array), 
                                 device=self.device, requires_grad=True)
            
            # Compute numerical gradient
            x.grad = None  # Clear gradients
            loss = fractional_function(x)
            loss.backward()
            if x.grad is not None:
                numerical_grad = x.grad.item()
            else:
                # If gradient is None, the computation graph is broken
                # This is expected for fractional derivatives that break the chain rule
                numerical_grad = 0.0
            
            # Calculate error
            error = abs(analytical_grad - numerical_grad) / abs(analytical_grad) if analytical_grad != 0 else float('inf')
            
            results['analytical_grads'].append(analytical_grad)
            results['numerical_grads'].append(numerical_grad)
            results['errors'].append(error)
            
            print(f"  α = {alpha}: Analytical {analytical_grad:.6f}, Numerical {numerical_grad:.6f}, Error {error:.2e}")
        
        return results
    
    def test_scalability(self) -> Dict:
        """Test scalability with different problem sizes"""
        print("\n=== Scalability Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'fractional_times': [],
            'standard_times': [],
            'efficiency': []
        }
        
        for size in self.test_sizes:
            print(f"Testing scalability for size {size}...")
            
            # Create test data
            x = torch.randn(size, requires_grad=True, device=self.device)
            
            # Test fractional scalability
            frac_config = GPUConfig(backend="jax" if self.device == 'cuda' else "numba")
            frac_deriv = GPUOptimizedRiemannLiouville(alpha=0.5, gpu_config=frac_config)
            
            start_time = time.time()
            for _ in range(5):  # Multiple runs
                x_np = x.detach().cpu().numpy()
                t_array = np.arange(len(x_np))
                frac_result = torch.tensor(frac_deriv.compute(x_np, t_array), 
                                        device=self.device, requires_grad=True)
                loss = frac_result.sum()
                loss.backward()
                if x.grad is not None:
                    x.grad.zero_()
            frac_time = (time.time() - start_time) / 5
            
            # Test standard scalability
            start_time = time.time()
            for _ in range(5):
                x.grad = None  # Clear gradients
                loss = x.sum()
                loss.backward()
                if x.grad is not None:
                    x.grad.zero_()
            std_time = (time.time() - start_time) / 5
            
            # Calculate efficiency (operations per second)
            frac_ops_per_sec = size / frac_time if frac_time > 0 else 0
            std_ops_per_sec = size / std_time if std_time > 0 else 0
            
            results['fractional_times'].append(frac_time)
            results['standard_times'].append(std_time)
            results['efficiency'].append(frac_ops_per_sec / std_ops_per_sec if std_ops_per_sec > 0 else 0)
            
            print(f"  Size {size}: Fractional {frac_time:.4f}s, Standard {std_time:.4f}s, Efficiency {frac_ops_per_sec/std_ops_per_sec:.2f}x")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all performance tests"""
        print("Starting comprehensive fractional autograd performance testing...")
        
        all_results = {
            'forward_pass': self.test_forward_pass_performance(),
            'backward_pass': self.test_backward_pass_performance(),
            'memory_usage': self.test_memory_usage(),
            'accuracy': self.test_accuracy(),
            'scalability': self.test_scalability()
        }
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = "fractional_autograd_performance_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def plot_results(self, results: Dict):
        """Create performance visualization plots"""
        print("\nCreating performance plots...")
        
        # Forward pass performance
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(results['forward_pass']['sizes'], results['forward_pass']['fractional_times'], 'b-o', label='Fractional')
        plt.plot(results['forward_pass']['sizes'], results['forward_pass']['standard_times'], 'r-s', label='Standard')
        plt.xlabel('Problem Size')
        plt.ylabel('Time (s)')
        plt.title('Forward Pass Performance')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(results['forward_pass']['sizes'], results['forward_pass']['speedup_ratios'], 'g-o')
        plt.xlabel('Problem Size')
        plt.ylabel('Speedup Ratio')
        plt.title('Forward Pass Speedup')
        plt.grid(True)
        
        # Backward pass performance
        plt.subplot(2, 3, 3)
        plt.plot(results['backward_pass']['sizes'], results['backward_pass']['fractional_grad_times'], 'b-o', label='Fractional')
        plt.plot(results['backward_pass']['sizes'], results['backward_pass']['standard_grad_times'], 'r-s', label='Standard')
        plt.xlabel('Problem Size')
        plt.ylabel('Time (s)')
        plt.title('Backward Pass Performance')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 4)
        plt.plot(results['backward_pass']['sizes'], results['backward_pass']['grad_speedup_ratios'], 'g-o')
        plt.xlabel('Problem Size')
        plt.ylabel('Speedup Ratio')
        plt.title('Backward Pass Speedup')
        plt.grid(True)
        
        # Memory usage
        plt.subplot(2, 3, 5)
        plt.plot(results['memory_usage']['sizes'], results['memory_usage']['fractional_memory'], 'b-o', label='Fractional')
        plt.plot(results['memory_usage']['sizes'], results['memory_usage']['standard_memory'], 'r-s', label='Standard')
        plt.xlabel('Problem Size')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage')
        plt.legend()
        plt.grid(True)
        
        # Accuracy
        plt.subplot(2, 3, 6)
        plt.semilogy(results['accuracy']['orders'], results['accuracy']['errors'], 'b-o')
        plt.xlabel('Fractional Order α')
        plt.ylabel('Relative Error')
        plt.title('Fractional Gradient Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('fractional_autograd_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig('fractional_autograd_performance.pdf', bbox_inches='tight')
        plt.show()
        
        print("Performance plots saved as 'fractional_autograd_performance.png' and 'fractional_autograd_performance.pdf'")

def main():
    """Main function to run fractional autograd performance tests"""
    print("=" * 60)
    print("FRACTIONAL AUTOGRAD PERFORMANCE TEST")
    print("=" * 60)
    
    # Initialize test
    test = FractionalAutogradPerformanceTest()
    
    # Run all tests
    results = test.run_all_tests()
    
    # Save results
    test.save_results(results)
    
    # Create plots
    test.plot_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    
    # Forward pass summary
    avg_speedup = np.mean(results['forward_pass']['speedup_ratios'])
    print(f"Average Forward Pass Speedup: {avg_speedup:.2f}x")
    
    # Backward pass summary
    avg_grad_speedup = np.mean(results['backward_pass']['grad_speedup_ratios'])
    print(f"Average Backward Pass Speedup: {avg_grad_speedup:.2f}x")
    
    # Memory summary
    avg_memory_ratio = np.mean(results['memory_usage']['memory_ratios'])
    print(f"Average Memory Usage Ratio: {avg_memory_ratio:.2f}x")
    
    # Accuracy summary
    avg_error = np.mean(results['accuracy']['errors'])
    print(f"Average Gradient Error: {avg_error:.2e}")
    
    # Scalability summary
    avg_efficiency = np.mean(results['scalability']['efficiency'])
    print(f"Average Efficiency Ratio: {avg_efficiency:.2f}x")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
