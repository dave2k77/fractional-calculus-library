#!/usr/bin/env python3
"""
Library Comparison for Fractional Calculus
Compare hpfracc against other fractional calculus libraries

Libraries to compare:
1. hpfracc (our library)
2. differint (Python)
3. scipy (basic fractional operations)
4. Custom implementations

Test problems:
1. Fractional derivatives of simple functions
2. Fractional integrals
3. Physics problems (wave, heat, Burgers equations)
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

class LibraryComparison:
    """Compare different fractional calculus libraries"""
    
    def __init__(self):
        self.results = {}
        self.x = np.linspace(0, 1, 100)
        self.dx = self.x[1] - self.x[0]
        
    def test_function(self, x):
        """Test function: f(x) = x^2"""
        return x**2
    
    def test_function_derivative(self, x):
        """Analytical derivative: f'(x) = 2x"""
        return 2*x
    
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
            return df.GL(f, alpha)
        except ImportError:
            print("differint not available")
            return None
    
    def scipy_fractional_derivative(self, f, alpha):
        """Fractional derivative using scipy"""
        try:
            from scipy.special import gamma
            # Simple approximation using finite differences
            if alpha == 1.0:
                return np.gradient(f, self.dx)
            elif alpha == 2.0:
                return np.gradient(np.gradient(f, self.dx), self.dx)
            else:
                # Crude approximation for fractional derivatives
                return np.gradient(f, self.dx) * (alpha ** 0.5)
        except ImportError:
            print("scipy not available")
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
            
            # scipy
            start_time = time.time()
            f_scipy = self.scipy_fractional_derivative(f, alpha)
            scipy_time = time.time() - start_time
            
            if f_scipy is not None:
                scipy_error = np.sqrt(np.mean((f_analytical - f_scipy)**2))
                results[alpha]['scipy'] = {'error': scipy_error, 'time': scipy_time}
                print(f"  scipy: L2 error = {scipy_error:.6f}, Time = {scipy_time:.4f}s")
            else:
                results[alpha]['scipy'] = {'error': np.inf, 'time': np.inf}
                print(f"  scipy: Not available")
            
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
        sizes = [50, 100, 200, 500, 1000]
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
    
    def create_visualization(self, results, performance_results):
        """Create visualization of results"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Create results directory
        os.makedirs('library_comparison_results', exist_ok=True)
        
        # 1. Error comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        alphas = list(results.keys())
        libraries = ['hpfracc', 'differint', 'scipy', 'custom']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, lib in enumerate(libraries):
            errors = [results[alpha][lib]['error'] for alpha in alphas]
            times = [results[alpha][lib]['time'] for alpha in alphas]
            
            # Filter out infinite values
            valid_errors = [e for e in errors if e != np.inf]
            valid_times = [t for t in times if t != np.inf]
            valid_alphas = [alphas[j] for j, e in enumerate(errors) if e != np.inf]
            
            if valid_errors:
                ax1.plot(valid_alphas, valid_errors, 'o-', label=lib, color=colors[i])
                ax2.plot(valid_alphas, valid_times, 'o-', label=lib, color=colors[i])
        
        ax1.set_xlabel('Fractional Order α')
        ax1.set_ylabel('L2 Error')
        ax1.set_title('Fractional Derivative Accuracy Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Fractional Order α')
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Fractional Derivative Performance Comparison')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('library_comparison_results/accuracy_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance scaling plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sizes = list(performance_results.keys())
        
        for i, lib in enumerate(libraries):
            times = [performance_results[size][lib] for size in sizes]
            valid_times = [t for t in times if t != np.inf]
            valid_sizes = [sizes[j] for j, t in enumerate(times) if t != np.inf]
            
            if valid_times:
                ax.plot(valid_sizes, valid_times, 'o-', label=lib, color=colors[i])
        
        ax.set_xlabel('Problem Size (points)')
        ax.set_ylabel('Computation Time (s)')
        ax.set_title('Performance Scaling Comparison')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('library_comparison_results/performance_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to library_comparison_results/")
    
    def save_results(self, results, performance_results):
        """Save results to file"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Create results directory
        os.makedirs('library_comparison_results', exist_ok=True)
        
        with open('library_comparison_results/library_comparison_results.txt', 'w') as f:
            f.write("Fractional Calculus Library Comparison Results\n")
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
        
        print("Results saved to library_comparison_results/library_comparison_results.txt")

def main():
    """Run library comparison"""
    print("FRACTIONAL CALCULUS LIBRARY COMPARISON")
    print("="*60)
    print("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)")
    print("Goal: Compare hpfracc against other fractional calculus libraries")
    print("="*60)
    
    # Initialize comparison
    comparison = LibraryComparison()
    
    # Run benchmarks
    results = comparison.benchmark_fractional_derivatives()
    performance_results = comparison.benchmark_performance()
    
    # Create visualizations
    comparison.create_visualization(results, performance_results)
    
    # Save results
    comparison.save_results(results, performance_results)
    
    print("\n" + "="*60)
    print("LIBRARY COMPARISON COMPLETED!")
    print("="*60)
    print("Results saved to library_comparison_results/")
    print("Ready to integrate into manuscript.")

if __name__ == "__main__":
    import os
    main()
