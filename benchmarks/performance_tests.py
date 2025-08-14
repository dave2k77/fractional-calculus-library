#!/usr/bin/env python3
"""
Performance Tests for Fractional Calculus Library

This module provides comprehensive performance benchmarks for all
major components of the fractional calculus library.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative
from src.algorithms.fft_methods import FFTFractionalMethods
from src.optimisation.jax_implementations import JAXFractionalDerivatives
from src.optimisation.numba_kernels import NumbaFractionalKernels
from src.optimisation.parallel_computing import ParallelFractionalComputing


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for fractional calculus."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'platform': os.name,
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    def benchmark_derivative_methods(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark different derivative computation methods."""
        print("🚀 Benchmarking Derivative Computation Methods")
        print("=" * 60)
        
        if grid_sizes is None:
            grid_sizes = [100, 500, 1000, 2000, 5000]
        
        # Test parameters
        alpha = 0.5
        t_max = 2.0
        
        # Initialize methods (will be recreated for each alpha)
        method_classes = {
            'Caputo': CaputoDerivative,
            'Riemann-Liouville': RiemannLiouvilleDerivative,
            'Grünwald-Letnikov': GrunwaldLetnikovDerivative,
            'FFT Spectral': lambda: FFTFractionalMethods(method='spectral'),
            'FFT Convolution': lambda: FFTFractionalMethods(method='convolution')
        }
        
        results = {}
        
        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")
            
            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t/2)
            
            method_results = {}
            
            for method_name, method_class in method_classes.items():
                try:
                    # Initialize method with current alpha
                    if method_name.startswith('FFT'):
                        method = method_class()
                        # Warm-up run
                        _ = method.compute_derivative(f, t, alpha)
                        
                        # Time the computation
                        start_time = time.time()
                        result = method.compute_derivative(f, t, alpha)
                        end_time = time.time()
                    else:
                        method = method_class(alpha)
                        # Warm-up run
                        _ = method.compute(f, t, h)
                        
                        # Time the computation
                        start_time = time.time()
                        result = method.compute(f, t, h)
                        end_time = time.time()
                    
                    method_results[method_name] = {
                        'time': end_time - start_time,
                        'memory': result.nbytes / 1024 / 1024,  # MB
                        'result_shape': result.shape
                    }
                    
                    print(f"  ✅ {method_name}: {method_results[method_name]['time']:.4f}s")
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {e}")
                    method_results[method_name] = {'time': np.inf, 'memory': 0, 'result_shape': (0,)}
            
            results[N] = method_results
        
        self.results['derivative_methods'] = results
        return results
    
    def benchmark_optimization_backends(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark optimization backends (JAX, Numba, Parallel)."""
        print("\n⚡ Benchmarking Optimization Backends")
        print("=" * 60)
        
        if grid_sizes is None:
            grid_sizes = [100, 500, 1000, 2000]
        
        # Test parameters
        alpha = 0.5
        t_max = 2.0
        
        results = {}
        
        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")
            
            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t/2)
            
            backend_results = {}
            
            # Test JAX backend
            try:
                print("  🧪 Testing JAX backend...")
                start_time = time.time()
                result_jax = JAXFractionalDerivatives.caputo_derivative_gpu(f, t, alpha, h)
                end_time = time.time()
                
                backend_results['JAX GPU'] = {
                    'time': end_time - start_time,
                    'memory': result_jax.nbytes / 1024 / 1024,
                    'result_shape': result_jax.shape
                }
                print(f"    ✅ JAX GPU: {backend_results['JAX GPU']['time']:.4f}s")
                
            except Exception as e:
                print(f"    ❌ JAX GPU: Failed - {e}")
                backend_results['JAX GPU'] = {'time': np.inf, 'memory': 0, 'result_shape': (0,)}
            
            # Test Numba backend
            try:
                print("  🧪 Testing Numba backend...")
                start_time = time.time()
                result_numba = NumbaFractionalKernels.caputo_l1_kernel(f, alpha, h)
                end_time = time.time()
                
                backend_results['Numba'] = {
                    'time': end_time - start_time,
                    'memory': result_numba.nbytes / 1024 / 1024,
                    'result_shape': result_numba.shape
                }
                print(f"    ✅ Numba: {backend_results['Numba']['time']:.4f}s")
                
            except Exception as e:
                print(f"    ❌ Numba: Failed - {e}")
                backend_results['Numba'] = {'time': np.inf, 'memory': 0, 'result_shape': (0,)}
            
            # Test Parallel backend
            try:
                print("  🧪 Testing Parallel backend...")
                parallel_computing = ParallelFractionalComputing(backend="joblib")
                
                # Create multiple datasets for parallel processing
                datasets = []
                for i in range(10):
                    f_shifted = f * (1 + 0.1 * i)
                    datasets.append(f_shifted)
                
                start_time = time.time()
                results_parallel = parallel_computing.parallel_fractional_derivative(
                    CaputoDerivative().compute,
                    [datasets] * len(datasets),
                    [t] * len(datasets),
                    alpha,
                    h
                )
                end_time = time.time()
                
                backend_results['Parallel (Joblib)'] = {
                    'time': end_time - start_time,
                    'memory': sum(r.nbytes for r in results_parallel) / 1024 / 1024,
                    'result_shape': (len(results_parallel), len(results_parallel[0]))
                }
                print(f"    ✅ Parallel (Joblib): {backend_results['Parallel (Joblib)']['time']:.4f}s")
                
            except Exception as e:
                print(f"    ❌ Parallel (Joblib): Failed - {e}")
                backend_results['Parallel (Joblib)'] = {'time': np.inf, 'memory': 0, 'result_shape': (0,)}
            
            results[N] = backend_results
        
        self.results['optimization_backends'] = results
        return results
    
    def benchmark_memory_usage(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        print("\n💾 Benchmarking Memory Usage")
        print("=" * 60)
        
        if grid_sizes is None:
            grid_sizes = [100, 500, 1000, 2000, 5000, 10000]
        
        # Test parameters
        alpha = 0.5
        t_max = 2.0
        
        results = {}
        
        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")
            
            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t/2)
            
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Test different methods
            method_results = {}
            
            methods = {
                'Caputo': CaputoDerivative(),
                'Riemann-Liouville': RiemannLiouvilleDerivative(),
                'FFT Spectral': FFTFractionalMethods(method='spectral')
            }
            
            for method_name, method in methods.items():
                try:
                    # Clear memory
                    import gc
                    gc.collect()
                    
                    # Measure memory before
                    memory_before = process.memory_info().rss / 1024 / 1024
                    
                    # Run computation
                    if method_name.startswith('FFT'):
                        result = method.compute_derivative(f, t, alpha)
                    else:
                        result = method.compute(f, t, h)
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024
                    
                    method_results[method_name] = {
                        'memory_used': memory_after - memory_before,
                        'result_memory': result.nbytes / 1024 / 1024,
                        'peak_memory': memory_after - initial_memory
                    }
                    
                    print(f"  ✅ {method_name}: {method_results[method_name]['memory_used']:.2f} MB")
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {e}")
                    method_results[method_name] = {'memory_used': 0, 'result_memory': 0, 'peak_memory': 0}
            
            results[N] = method_results
        
        self.results['memory_usage'] = results
        return results
    
    def benchmark_accuracy_vs_speed(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark accuracy vs speed trade-offs."""
        print("\n🎯 Benchmarking Accuracy vs Speed Trade-offs")
        print("=" * 60)
        
        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 500, 1000]
        
        # Test parameters
        alpha = 0.5
        t_max = 1.0
        
        # Analytical solution for comparison
        from scipy.special import gamma
        def analytical_solution(t, alpha):
            return t**(1-alpha) / gamma(2-alpha)
        
        results = {}
        
        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")
            
            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = t  # Simple linear function
            
            # Analytical solution
            analytical = analytical_solution(t, alpha)
            
            method_results = {}
            
            # Initialize methods (will be recreated for each alpha)
            method_classes = {
                'Caputo': CaputoDerivative,
                'Riemann-Liouville': RiemannLiouvilleDerivative,
                'Grünwald-Letnikov': GrunwaldLetnikovDerivative,
                'FFT Spectral': lambda: FFTFractionalMethods(method='spectral')
            }
            
            for method_name, method_class in method_classes.items():
                try:
                    # Initialize method with current alpha
                    if method_name.startswith('FFT'):
                        method = method_class()
                        # Time the computation
                        start_time = time.time()
                        result = method.compute_derivative(f, t, alpha)
                        end_time = time.time()
                    else:
                        method = method_class(alpha)
                        # Time the computation
                        start_time = time.time()
                        result = method.compute(f, t, h)
                        end_time = time.time()
                    
                    # Calculate error
                    error = np.abs(result - analytical)
                    max_error = np.max(error)
                    mean_error = np.mean(error)
                    l2_error = np.sqrt(np.mean(error**2))
                    
                    method_results[method_name] = {
                        'time': end_time - start_time,
                        'max_error': max_error,
                        'mean_error': mean_error,
                        'l2_error': l2_error,
                        'convergence_rate': -np.log10(max_error) if max_error > 0 else np.inf
                    }
                    
                    print(f"  ✅ {method_name}: {method_results[method_name]['time']:.4f}s, "
                          f"max error: {max_error:.2e}")
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {e}")
                    method_results[method_name] = {
                        'time': np.inf, 'max_error': np.inf, 'mean_error': np.inf,
                        'l2_error': np.inf, 'convergence_rate': 0
                    }
            
            results[N] = method_results
        
        self.results['accuracy_vs_speed'] = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        print("\n📊 Generating Performance Report")
        print("=" * 60)
        
        report = []
        report.append("FRACTIONAL CALCULUS LIBRARY - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # System information
        report.append("SYSTEM INFORMATION:")
        report.append(f"  CPU Count: {self.system_info['cpu_count']}")
        report.append(f"  Total Memory: {self.system_info['memory_total'] / 1024**3:.2f} GB")
        report.append(f"  Platform: {self.system_info['platform']}")
        report.append(f"  Python Version: {self.system_info['python_version']}")
        report.append("")
        
        # Derivative methods performance
        if 'derivative_methods' in self.results:
            report.append("DERIVATIVE METHODS PERFORMANCE:")
            for N, methods in self.results['derivative_methods'].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    if data['time'] < np.inf:
                        report.append(f"    {method}: {data['time']:.4f}s")
            report.append("")
        
        # Optimization backends performance
        if 'optimization_backends' in self.results:
            report.append("OPTIMIZATION BACKENDS PERFORMANCE:")
            for N, backends in self.results['optimization_backends'].items():
                report.append(f"  Grid Size {N}:")
                for backend, data in backends.items():
                    if data['time'] < np.inf:
                        report.append(f"    {backend}: {data['time']:.4f}s")
            report.append("")
        
        # Memory usage
        if 'memory_usage' in self.results:
            report.append("MEMORY USAGE ANALYSIS:")
            for N, methods in self.results['memory_usage'].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    report.append(f"    {method}: {data['memory_used']:.2f} MB")
            report.append("")
        
        # Accuracy vs speed
        if 'accuracy_vs_speed' in self.results:
            report.append("ACCURACY VS SPEED ANALYSIS:")
            for N, methods in self.results['accuracy_vs_speed'].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    if data['time'] < np.inf:
                        report.append(f"    {method}: {data['time']:.4f}s, "
                                   f"max error: {data['max_error']:.2e}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open('benchmarks/performance_report.txt', 'w') as f:
            f.write(report_text)
        
        print("✅ Performance report generated and saved to 'benchmarks/performance_report.txt'")
        return report_text
    
    def plot_performance_results(self):
        """Generate performance visualization plots."""
        print("\n📈 Generating Performance Plots")
        print("=" * 60)
        
        # Create plots directory
        os.makedirs('benchmarks/plots', exist_ok=True)
        
        # Plot 1: Derivative methods performance
        if 'derivative_methods' in self.results:
            plt.figure(figsize=(12, 8))
            
            grid_sizes = list(self.results['derivative_methods'].keys())
            methods = list(self.results['derivative_methods'][grid_sizes[0]].keys())
            
            for method in methods:
                times = []
                for N in grid_sizes:
                    time_val = self.results['derivative_methods'][N][method]['time']
                    times.append(time_val if time_val < np.inf else np.nan)
                
                plt.loglog(grid_sizes, times, 'o-', label=method, linewidth=2, markersize=8)
            
            plt.xlabel('Grid Size N')
            plt.ylabel('Execution Time (s)')
            plt.title('Derivative Methods Performance Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('benchmarks/plots/derivative_methods_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 2: Optimization backends performance
        if 'optimization_backends' in self.results:
            plt.figure(figsize=(12, 8))
            
            grid_sizes = list(self.results['optimization_backends'].keys())
            backends = list(self.results['optimization_backends'][grid_sizes[0]].keys())
            
            for backend in backends:
                times = []
                for N in grid_sizes:
                    time_val = self.results['optimization_backends'][N][backend]['time']
                    times.append(time_val if time_val < np.inf else np.nan)
                
                plt.loglog(grid_sizes, times, 'o-', label=backend, linewidth=2, markersize=8)
            
            plt.xlabel('Grid Size N')
            plt.ylabel('Execution Time (s)')
            plt.title('Optimization Backends Performance Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('benchmarks/plots/optimization_backends_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 3: Memory usage
        if 'memory_usage' in self.results:
            plt.figure(figsize=(12, 8))
            
            grid_sizes = list(self.results['memory_usage'].keys())
            methods = list(self.results['memory_usage'][grid_sizes[0]].keys())
            
            for method in methods:
                memory_usage = []
                for N in grid_sizes:
                    memory_val = self.results['memory_usage'][N][method]['memory_used']
                    memory_usage.append(memory_val)
                
                plt.loglog(grid_sizes, memory_usage, 'o-', label=method, linewidth=2, markersize=8)
            
            plt.xlabel('Grid Size N')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('benchmarks/plots/memory_usage_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 4: Accuracy vs speed trade-off
        if 'accuracy_vs_speed' in self.results:
            plt.figure(figsize=(12, 8))
            
            grid_sizes = list(self.results['accuracy_vs_speed'].keys())
            methods = list(self.results['accuracy_vs_speed'][grid_sizes[0]].keys())
            
            for method in methods:
                times = []
                errors = []
                for N in grid_sizes:
                    time_val = self.results['accuracy_vs_speed'][N][method]['time']
                    error_val = self.results['accuracy_vs_speed'][N][method]['max_error']
                    if time_val < np.inf and error_val < np.inf:
                        times.append(time_val)
                        errors.append(error_val)
                
                if times:
                    plt.loglog(times, errors, 'o-', label=method, linewidth=2, markersize=8)
            
            plt.xlabel('Execution Time (s)')
            plt.ylabel('Maximum Error')
            plt.title('Accuracy vs Speed Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('benchmarks/plots/accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("✅ Performance plots generated and saved to 'benchmarks/plots/' directory")


def main():
    """Run comprehensive performance benchmarks."""
    print("🚀 Fractional Calculus Library - Performance Benchmarks")
    print("=" * 70)
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_derivative_methods()
    benchmark.benchmark_optimization_backends()
    benchmark.benchmark_memory_usage()
    benchmark.benchmark_accuracy_vs_speed()
    
    # Generate report and plots
    benchmark.generate_performance_report()
    benchmark.plot_performance_results()
    
    print("\n🎉 All performance benchmarks completed!")
    print("\n📁 Results saved to:")
    print("  - benchmarks/performance_report.txt")
    print("  - benchmarks/plots/")


if __name__ == "__main__":
    main()
