"""
Comprehensive Performance Benchmarking for Fractional Calculus Library

This module provides comprehensive performance benchmarking for all methods:
- Special Methods (Fractional Laplacian, FrFT, Z-Transform)
- Optimized Methods (Special Optimized versions)
- Standard Methods (for comparison)
- Performance analysis across different problem sizes and scenarios
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all methods
from src.algorithms.special_methods import (
    FractionalLaplacian,
    FractionalFourierTransform,
    FractionalZTransform,
    fractional_laplacian,
    fractional_fourier_transform,
    fractional_z_transform,
)

from src.algorithms.special_optimized_methods import (
    SpecialOptimizedWeylDerivative,
    SpecialOptimizedMarchaudDerivative,
    SpecialOptimizedReizFellerDerivative,
    UnifiedSpecialMethods,
    special_optimized_weyl_derivative,
    special_optimized_marchaud_derivative,
    special_optimized_reiz_feller_derivative,
    unified_special_derivative,
)

from src.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    ReizFellerDerivative,
)

from src.algorithms.advanced_optimized_methods import (
    OptimizedWeylDerivative,
    OptimizedMarchaudDerivative,
    OptimizedReizFellerDerivative,
)

from src.core.definitions import FractionalOrder


class ComprehensiveBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self):
        """Initialize benchmark system."""
        self.results = {}
        self.test_functions = {
            'gaussian': lambda x: np.exp(-x**2),
            'sine': lambda x: np.sin(x),
            'cosine': lambda x: np.cos(x),
            'polynomial': lambda x: x**3 - 2*x**2 + x,
            'exponential': lambda x: np.exp(-abs(x)),
        }
        
        self.problem_sizes = [50, 100, 200, 500, 1000, 2000]
        self.alpha_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        
    def benchmark_special_methods(self) -> Dict:
        """Benchmark special methods performance."""
        print("🔬 Benchmarking Special Methods...")
        
        results = {
            'fractional_laplacian': {},
            'fractional_fourier_transform': {},
            'fractional_z_transform': {},
        }
        
        # Test different problem sizes
        for size in self.problem_sizes:
            print(f"  Testing size: {size}")
            x = np.linspace(-5, 5, size)
            
            # Fractional Laplacian benchmarks
            laplacian_results = self._benchmark_laplacian(x, size)
            results['fractional_laplacian'][size] = laplacian_results
            
            # Fractional Fourier Transform benchmarks
            frft_results = self._benchmark_frft(x, size)
            results['fractional_fourier_transform'][size] = frft_results
            
            # Fractional Z-Transform benchmarks
            z_transform_results = self._benchmark_z_transform(size)
            results['fractional_z_transform'][size] = z_transform_results
        
        return results
    
    def _benchmark_laplacian(self, x: np.ndarray, size: int) -> Dict:
        """Benchmark fractional Laplacian methods."""
        alpha = 1.0
        f = self.test_functions['gaussian'](x)
        
        results = {}
        
        # Test different methods
        methods = ['spectral', 'finite_difference', 'integral']
        
        for method in methods:
            laplacian = FractionalLaplacian(alpha)
            
            # Warm up
            _ = laplacian.compute(f, x, method=method)
            
            # Benchmark
            times = []
            for _ in range(5):  # 5 runs for averaging
                start_time = time.perf_counter()
                result = laplacian.compute(f, x, method=method)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
            }
        
        return results
    
    def _benchmark_frft(self, x: np.ndarray, size: int) -> Dict:
        """Benchmark fractional Fourier transform methods."""
        alpha = np.pi / 4
        f = self.test_functions['gaussian'](x)
        
        results = {}
        
        # Test different methods
        methods = ['discrete', 'spectral', 'fast', 'auto']
        
        for method in methods:
            frft = FractionalFourierTransform(alpha)
            
            # Warm up
            _ = frft.transform(f, x, method=method)
            
            # Benchmark
            times = []
            for _ in range(5):  # 5 runs for averaging
                start_time = time.perf_counter()
                result = frft.transform(f, x, method=method)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
            }
        
        return results
    
    def _benchmark_z_transform(self, size: int) -> Dict:
        """Benchmark fractional Z-transform methods."""
        alpha = 0.5
        f = np.random.random(size)
        z_values = np.exp(1j * np.linspace(0, 2*np.pi, min(size, 100), endpoint=False))
        
        results = {}
        
        # Test different methods
        methods = ['direct', 'fft']
        
        for method in methods:
            z_transform = FractionalZTransform(alpha)
            
            # Warm up
            _ = z_transform.transform(f, z_values, method=method)
            
            # Benchmark
            times = []
            for _ in range(5):  # 5 runs for averaging
                start_time = time.perf_counter()
                result = z_transform.transform(f, z_values, method=method)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            results[method] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
            }
        
        return results
    
    def benchmark_optimized_methods(self) -> Dict:
        """Benchmark optimized methods vs standard methods."""
        print("⚡ Benchmarking Optimized vs Standard Methods...")
        
        results = {
            'weyl_derivative': {},
            'marchaud_derivative': {},
            'reiz_feller_derivative': {},
        }
        
        # Test different problem sizes
        for size in self.problem_sizes:
            print(f"  Testing size: {size}")
            x = np.linspace(0, 10, size)
            
            # Weyl Derivative benchmarks
            weyl_results = self._benchmark_weyl_derivative(x, size)
            results['weyl_derivative'][size] = weyl_results
            
            # Marchaud Derivative benchmarks
            marchaud_results = self._benchmark_marchaud_derivative(x, size)
            results['marchaud_derivative'][size] = marchaud_results
            
            # Reiz-Feller Derivative benchmarks
            reiz_results = self._benchmark_reiz_feller_derivative(x, size)
            results['reiz_feller_derivative'][size] = reiz_results
        
        return results
    
    def _benchmark_weyl_derivative(self, x: np.ndarray, size: int) -> Dict:
        """Benchmark Weyl derivative methods."""
        alpha = 0.5
        f = self.test_functions['gaussian'](x)
        
        results = {}
        
        # Standard method
        weyl_std = WeylDerivative(alpha)
        times_std = []
        for _ in range(3):  # Fewer runs for slower methods
            start_time = time.perf_counter()
            result = weyl_std.compute(f, x, use_parallel=False)
            end_time = time.perf_counter()
            times_std.append(end_time - start_time)
        
        results['standard'] = {
            'mean_time': np.mean(times_std),
            'std_time': np.std(times_std),
        }
        
        # Optimized method
        weyl_opt = OptimizedWeylDerivative(alpha)
        times_opt = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = weyl_opt.compute(f, x)
            end_time = time.perf_counter()
            times_opt.append(end_time - start_time)
        
        results['optimized'] = {
            'mean_time': np.mean(times_opt),
            'std_time': np.std(times_opt),
        }
        
        # Special optimized method
        weyl_special = SpecialOptimizedWeylDerivative(alpha)
        times_special = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = weyl_special.compute(f, x)
            end_time = time.perf_counter()
            times_special.append(end_time - start_time)
        
        results['special_optimized'] = {
            'mean_time': np.mean(times_special),
            'std_time': np.std(times_special),
        }
        
        return results
    
    def _benchmark_marchaud_derivative(self, x: np.ndarray, size: int) -> Dict:
        """Benchmark Marchaud derivative methods."""
        alpha = 0.5
        f = self.test_functions['sine'](x)
        
        results = {}
        
        # Standard method
        marchaud_std = MarchaudDerivative(alpha)
        times_std = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = marchaud_std.compute(f, x, use_parallel=False, memory_optimized=True)
            end_time = time.perf_counter()
            times_std.append(end_time - start_time)
        
        results['standard'] = {
            'mean_time': np.mean(times_std),
            'std_time': np.std(times_std),
        }
        
        # Optimized method
        marchaud_opt = OptimizedMarchaudDerivative(alpha)
        times_opt = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = marchaud_opt.compute(f, x)
            end_time = time.perf_counter()
            times_opt.append(end_time - start_time)
        
        results['optimized'] = {
            'mean_time': np.mean(times_opt),
            'std_time': np.std(times_opt),
        }
        
        # Special optimized method
        marchaud_special = SpecialOptimizedMarchaudDerivative(alpha)
        times_special = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = marchaud_special.compute(f, x)
            end_time = time.perf_counter()
            times_special.append(end_time - start_time)
        
        results['special_optimized'] = {
            'mean_time': np.mean(times_special),
            'std_time': np.std(times_special),
        }
        
        return results
    
    def _benchmark_reiz_feller_derivative(self, x: np.ndarray, size: int) -> Dict:
        """Benchmark Reiz-Feller derivative methods."""
        alpha = 0.5
        f = self.test_functions['gaussian'](x)
        
        results = {}
        
        # Standard method
        reiz_std = ReizFellerDerivative(alpha)
        times_std = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = reiz_std.compute(f, x, use_parallel=False)
            end_time = time.perf_counter()
            times_std.append(end_time - start_time)
        
        results['standard'] = {
            'mean_time': np.mean(times_std),
            'std_time': np.std(times_std),
        }
        
        # Optimized method
        reiz_opt = OptimizedReizFellerDerivative(alpha)
        times_opt = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = reiz_opt.compute(f, x)
            end_time = time.perf_counter()
            times_opt.append(end_time - start_time)
        
        results['optimized'] = {
            'mean_time': np.mean(times_opt),
            'std_time': np.std(times_opt),
        }
        
        # Special optimized method
        reiz_special = SpecialOptimizedReizFellerDerivative(alpha)
        times_special = []
        for _ in range(5):
            start_time = time.perf_counter()
            result = reiz_special.compute(f, x)
            end_time = time.perf_counter()
            times_special.append(end_time - start_time)
        
        results['special_optimized'] = {
            'mean_time': np.mean(times_special),
            'std_time': np.std(times_special),
        }
        
        return results
    
    def benchmark_unified_methods(self) -> Dict:
        """Benchmark unified special methods."""
        print("🎯 Benchmarking Unified Special Methods...")
        
        results = {}
        
        # Test different problem sizes
        for size in self.problem_sizes:
            print(f"  Testing size: {size}")
            x = np.linspace(-5, 5, size)
            f = self.test_functions['gaussian'](x)
            alpha = 0.5
            
            unified = UnifiedSpecialMethods()
            
            # Test different problem types
            problem_types = ['general', 'periodic', 'discrete', 'spectral']
            
            for problem_type in problem_types:
                times = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    result = unified.compute_derivative(f, x, alpha, h=0.1, problem_type=problem_type)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                if problem_type not in results:
                    results[problem_type] = {}
                
                results[problem_type][size] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                }
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark suite."""
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("=" * 60)
        
        all_results = {}
        
        # Run all benchmarks
        all_results['special_methods'] = self.benchmark_special_methods()
        all_results['optimized_methods'] = self.benchmark_optimized_methods()
        all_results['unified_methods'] = self.benchmark_unified_methods()
        
        print("\n✅ Comprehensive benchmark completed!")
        
        return all_results
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("# Comprehensive Performance Benchmark Report")
        report.append("=" * 60)
        report.append("")
        
        # Special Methods Report
        report.append("## 🔬 Special Methods Performance")
        report.append("")
        
        special_results = results['special_methods']
        
        # Fractional Laplacian
        report.append("### Fractional Laplacian")
        laplacian_data = special_results['fractional_laplacian']
        for size in self.problem_sizes:
            if size in laplacian_data:
                report.append(f"**Size {size}:**")
                for method, data in laplacian_data[size].items():
                    report.append(f"  - {method}: {data['mean_time']:.6f}s ± {data['std_time']:.6f}s")
                report.append("")
        
        # Fractional Fourier Transform
        report.append("### Fractional Fourier Transform")
        frft_data = special_results['fractional_fourier_transform']
        for size in self.problem_sizes:
            if size in frft_data:
                report.append(f"**Size {size}:**")
                for method, data in frft_data[size].items():
                    report.append(f"  - {method}: {data['mean_time']:.6f}s ± {data['std_time']:.6f}s")
                report.append("")
        
        # Optimized Methods Report
        report.append("## ⚡ Optimized vs Standard Methods")
        report.append("")
        
        optimized_results = results['optimized_methods']
        
        for method_name, method_data in optimized_results.items():
            report.append(f"### {method_name.replace('_', ' ').title()}")
            for size in self.problem_sizes:
                if size in method_data:
                    report.append(f"**Size {size}:**")
                    for impl, data in method_data[size].items():
                        report.append(f"  - {impl}: {data['mean_time']:.6f}s ± {data['std_time']:.6f}s")
                    
                    # Calculate speedup
                    if 'standard' in method_data[size] and 'special_optimized' in method_data[size]:
                        std_time = method_data[size]['standard']['mean_time']
                        opt_time = method_data[size]['special_optimized']['mean_time']
                        speedup = std_time / opt_time if opt_time > 0 else float('inf')
                        report.append(f"  - **Speedup: {speedup:.2f}x**")
                    report.append("")
        
        # Unified Methods Report
        report.append("## 🎯 Unified Special Methods")
        report.append("")
        
        unified_results = results['unified_methods']
        for problem_type, type_data in unified_results.items():
            report.append(f"### {problem_type.title()} Problems")
            for size in self.problem_sizes:
                if size in type_data:
                    data = type_data[size]
                    report.append(f"  - Size {size}: {data['mean_time']:.6f}s ± {data['std_time']:.6f}s")
            report.append("")
        
        return "\n".join(report)
    
    def create_performance_plots(self, results: Dict, save_path: str = "benchmarks/performance_plots.png"):
        """Create performance visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Performance Benchmark Results', fontsize=16)
        
        # Plot 1: Fractional Laplacian Methods
        ax1 = axes[0, 0]
        special_results = results['special_methods']
        laplacian_data = special_results['fractional_laplacian']
        
        methods = ['spectral', 'finite_difference', 'integral']
        colors = ['blue', 'red', 'green']
        
        for method, color in zip(methods, colors):
            times = []
            sizes = []
            for size in self.problem_sizes:
                if size in laplacian_data and method in laplacian_data[size]:
                    times.append(laplacian_data[size][method]['mean_time'])
                    sizes.append(size)
            
            if times:
                ax1.loglog(sizes, times, 'o-', color=color, label=method, linewidth=2, markersize=6)
        
        ax1.set_title('Fractional Laplacian Performance')
        ax1.set_xlabel('Problem Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fractional Fourier Transform Methods
        ax2 = axes[0, 1]
        frft_data = special_results['fractional_fourier_transform']
        
        methods = ['discrete', 'spectral', 'fast', 'auto']
        colors = ['blue', 'red', 'green', 'orange']
        
        for method, color in zip(methods, colors):
            times = []
            sizes = []
            for size in self.problem_sizes:
                if size in frft_data and method in frft_data[size]:
                    times.append(frft_data[size][method]['mean_time'])
                    sizes.append(size)
            
            if times:
                ax2.loglog(sizes, times, 'o-', color=color, label=method, linewidth=2, markersize=6)
        
        ax2.set_title('Fractional Fourier Transform Performance')
        ax2.set_xlabel('Problem Size')
        ax2.set_ylabel('Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Weyl Derivative Comparison
        ax3 = axes[0, 2]
        optimized_results = results['optimized_methods']
        weyl_data = optimized_results['weyl_derivative']
        
        implementations = ['standard', 'optimized', 'special_optimized']
        colors = ['red', 'blue', 'green']
        
        for impl, color in zip(implementations, colors):
            times = []
            sizes = []
            for size in self.problem_sizes:
                if size in weyl_data and impl in weyl_data[size]:
                    times.append(weyl_data[size][impl]['mean_time'])
                    sizes.append(size)
            
            if times:
                ax3.loglog(sizes, times, 'o-', color=color, label=impl, linewidth=2, markersize=6)
        
        ax3.set_title('Weyl Derivative Performance Comparison')
        ax3.set_xlabel('Problem Size')
        ax3.set_ylabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Marchaud Derivative Comparison
        ax4 = axes[1, 0]
        marchaud_data = optimized_results['marchaud_derivative']
        
        for impl, color in zip(implementations, colors):
            times = []
            sizes = []
            for size in self.problem_sizes:
                if size in marchaud_data and impl in marchaud_data[size]:
                    times.append(marchaud_data[size][impl]['mean_time'])
                    sizes.append(size)
            
            if times:
                ax4.loglog(sizes, times, 'o-', color=color, label=impl, linewidth=2, markersize=6)
        
        ax4.set_title('Marchaud Derivative Performance Comparison')
        ax4.set_xlabel('Problem Size')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Reiz-Feller Derivative Comparison
        ax5 = axes[1, 1]
        reiz_data = optimized_results['reiz_feller_derivative']
        
        for impl, color in zip(implementations, colors):
            times = []
            sizes = []
            for size in self.problem_sizes:
                if size in reiz_data and impl in reiz_data[size]:
                    times.append(reiz_data[size][impl]['mean_time'])
                    sizes.append(size)
            
            if times:
                ax5.loglog(sizes, times, 'o-', color=color, label=impl, linewidth=2, markersize=6)
        
        ax5.set_title('Reiz-Feller Derivative Performance Comparison')
        ax5.set_xlabel('Problem Size')
        ax5.set_ylabel('Time (seconds)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Unified Methods
        ax6 = axes[1, 2]
        unified_results = results['unified_methods']
        
        problem_types = ['general', 'periodic', 'discrete', 'spectral']
        colors = ['blue', 'red', 'green', 'orange']
        
        for problem_type, color in zip(problem_types, colors):
            times = []
            sizes = []
            if problem_type in unified_results:
                for size in self.problem_sizes:
                    if size in unified_results[problem_type]:
                        times.append(unified_results[problem_type][size]['mean_time'])
                        sizes.append(size)
            
            if times:
                ax6.loglog(sizes, times, 'o-', color=color, label=problem_type, linewidth=2, markersize=6)
        
        ax6.set_title('Unified Special Methods Performance')
        ax6.set_xlabel('Problem Size')
        ax6.set_ylabel('Time (seconds)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance plots saved to: {save_path}")


def main():
    """Run comprehensive benchmark."""
    benchmark = ComprehensiveBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate report
    report = benchmark.generate_performance_report(results)
    
    # Save report
    with open('benchmarks/comprehensive_benchmark_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n📄 Performance report saved to: benchmarks/comprehensive_benchmark_report.md")
    
    # Create plots
    benchmark.create_performance_plots(results)
    
    # Print summary
    print("\n🎯 Performance Summary:")
    print("=" * 40)
    
    # Calculate best speedups
    optimized_results = results['optimized_methods']
    
    for method_name, method_data in optimized_results.items():
        best_speedup = 0
        best_size = 0
        
        for size in benchmark.problem_sizes:
            if size in method_data:
                if 'standard' in method_data[size] and 'special_optimized' in method_data[size]:
                    std_time = method_data[size]['standard']['mean_time']
                    opt_time = method_data[size]['special_optimized']['mean_time']
                    speedup = std_time / opt_time if opt_time > 0 else float('inf')
                    
                    if speedup > best_speedup and speedup != float('inf'):
                        best_speedup = speedup
                        best_size = size
        
        if best_speedup > 0:
            print(f"{method_name.replace('_', ' ').title()}: {best_speedup:.1f}x speedup at size {best_size}")


if __name__ == "__main__":
    main()
