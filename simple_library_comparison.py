#!/usr/bin/env python3
"""
Simple Library Comparison for Fractional Calculus
Compare hpfracc against other fractional calculus libraries

Libraries to compare:
1. hpfracc (our library)
2. differint (Python)
3. Custom implementations

Test problems:
1. Fractional derivatives of simple functions
2. Performance benchmarks
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add hpfracc to path
sys.path.append('/home/davianc/fractional-calculus-library')

class SimpleLibraryComparison:
    """Compare different fractional calculus libraries"""
    
    def __init__(self):
        self.results = {}
        self.x = np.linspace(0, 1, 100)
        self.dx = self.x[1] - self.x[0]
        
    def test_function(self, x):
        """Test function: f(x) = x^2"""
        return x**2
    
    def test_function_fractional_derivative(self, x, alpha):
        """Analytical fractional derivative of x^2"""
        from scipy.special import gamma
        return 2 * gamma(3) / gamma(3 - alpha) * x**(2 - alpha)
    
    def hpfracc_fractional_derivative(self, f, alpha):
        """Fractional derivative using hpfracc"""
        try:
            from hpfracc.algorithms import grunwald_letnikov
            return grunwald_letnikov(f, alpha, self.dx)
        except ImportError:
            print("hpfracc not available")
            return None
    
    def differint_fractional_derivative(self, f, alpha):
        """Fractional derivative using differint"""
        try:
            import differint.differint as df
            # Create a function for differint
            def func(x):
                return x**2
            return df.GL(func, 0, 1, len(f), alpha)
        except Exception as e:
            print(f"differint error: {e}")
            return None
    
    def custom_fractional_derivative(self, f, alpha):
        """Custom fractional derivative implementation"""
        # Grünwald-Letnikov approximation
        n = len(f)
        result = np.zeros_like(f)
        
        for i in range(n):
            for k in range(i + 1):
                if k == 0:
                    coeff = 1
                else:
                    coeff = coeff * (k - 1 - alpha) / k
                result[i] += coeff * f[i - k]
        
        return result / (self.dx ** alpha)
    
    def benchmark_fractional_derivatives(self):
        """Benchmark fractional derivative calculations"""
        print("="*60)
        print("FRACTIONAL DERIVATIVE BENCHMARK")
        print("="*60)
        
        # Test function
        f = self.test_function(self.x)
        alphas = [0.5, 0.7, 0.9, 1.0, 1.3, 1.5]
        
        results = {}
        
        for alpha in alphas:
            print(f"\nTesting α = {alpha}")
            results[alpha] = {}
            
            # Analytical solution
            f_analytical = self.test_function_fractional_derivative(self.x, alpha)
            
            # hpfracc
            start_time = time.time()
            f_hpfracc = self.hpfracc_fractional_derivative(f, alpha)
            hpfracc_time = time.time() - start_time
            
            if f_hpfracc is not None:
                hpfracc_error = np.sqrt(np.mean((f_analytical - f_hpfracc)**2))
                results[alpha]['hpfracc'] = {'error': hpfracc_error, 'time': hpfracc_time}
                print(f"  hpfracc: L2 error = {hpfracc_error:.6f}, Time = {hpfracc_time:.4f}s")
            else:
                results[alpha]['hpfracc'] = {'error': np.inf, 'time': np.inf}
                print(f"  hpfracc: Not available")
            
            # differint
            start_time = time.time()
            f_differint = self.differint_fractional_derivative(f, alpha)
            differint_time = time.time() - start_time
            
            if f_differint is not None:
                differint_error = np.sqrt(np.mean((f_analytical - f_differint)**2))
                results[alpha]['differint'] = {'error': differint_error, 'time': differint_time}
                print(f"  differint: L2 error = {differint_error:.6f}, Time = {differint_time:.4f}s")
            else:
                results[alpha]['differint'] = {'error': np.inf, 'time': np.inf}
                print(f"  differint: Not available")
            
            # Custom implementation
            start_time = time.time()
            f_custom = self.custom_fractional_derivative(f, alpha)
            custom_time = time.time() - start_time
            
            custom_error = np.sqrt(np.mean((f_analytical - f_custom)**2))
            results[alpha]['custom'] = {'error': custom_error, 'time': custom_time}
            print(f"  custom: L2 error = {custom_error:.6f}, Time = {custom_time:.4f}s")
        
        return results
    
    def benchmark_performance(self):
        """Benchmark computational performance"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Test with different problem sizes
        sizes = [50, 100, 200, 500]
        alpha = 0.5
        
        performance_results = {}
        
        for size in sizes:
            print(f"\nTesting with {size} points")
            x = np.linspace(0, 1, size)
            f = x**2
            
            performance_results[size] = {}
            
            # hpfracc
            start_time = time.time()
            f_hpfracc = self.hpfracc_fractional_derivative(f, alpha)
            hpfracc_time = time.time() - start_time
            
            if f_hpfracc is not None:
                performance_results[size]['hpfracc'] = hpfracc_time
                print(f"  hpfracc: {hpfracc_time:.4f}s")
            else:
                performance_results[size]['hpfracc'] = np.inf
                print(f"  hpfracc: Not available")
            
            # differint
            start_time = time.time()
            f_differint = self.differint_fractional_derivative(f, alpha)
            differint_time = time.time() - start_time
            
            if f_differint is not None:
                performance_results[size]['differint'] = differint_time
                print(f"  differint: {differint_time:.4f}s")
            else:
                performance_results[size]['differint'] = np.inf
                print(f"  differint: Not available")
            
            # Custom implementation
            start_time = time.time()
            f_custom = self.custom_fractional_derivative(f, alpha)
            custom_time = time.time() - start_time
            
            performance_results[size]['custom'] = custom_time
            print(f"  custom: {custom_time:.4f}s")
        
        return performance_results
    
    def save_results(self, results, performance_results):
        """Save results to file"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Create results directory
        os.makedirs('library_comparison_results', exist_ok=True)
        
        with open('library_comparison_results/simple_library_comparison_results.txt', 'w') as f:
            f.write("Simple Fractional Calculus Library Comparison Results\n")
            f.write("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)\n")
            f.write("="*60 + "\n\n")
            
            f.write("FRACTIONAL DERIVATIVE ACCURACY:\n")
            f.write("-" * 40 + "\n")
            for alpha, result in results.items():
                f.write(f"α = {alpha}:\n")
                for lib, data in result.items():
                    f.write(f"  {lib}: L2 error = {data['error']:.6f}, Time = {data['time']:.4f}s\n")
                f.write("\n")
            
            f.write("PERFORMANCE SCALING:\n")
            f.write("-" * 40 + "\n")
            for size, result in performance_results.items():
                f.write(f"Size = {size}:\n")
                for lib, time_val in result.items():
                    f.write(f"  {lib}: {time_val:.4f}s\n")
                f.write("\n")
        
        print("Results saved to library_comparison_results/simple_library_comparison_results.txt")

def main():
    """Run simple library comparison"""
    print("SIMPLE FRACTIONAL CALCULUS LIBRARY COMPARISON")
    print("="*60)
    print("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)")
    print("Goal: Compare hpfracc against other fractional calculus libraries")
    print("="*60)
    
    # Initialize comparison
    comparison = SimpleLibraryComparison()
    
    # Run benchmarks
    results = comparison.benchmark_fractional_derivatives()
    performance_results = comparison.benchmark_performance()
    
    # Save results
    comparison.save_results(results, performance_results)
    
    print("\n" + "="*60)
    print("SIMPLE LIBRARY COMPARISON COMPLETED!")
    print("="*60)
    print("Results saved to library_comparison_results/")
    print("Ready to integrate into manuscript.")

if __name__ == "__main__":
    import os
    main()
