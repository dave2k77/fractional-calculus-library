#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite

This script runs comprehensive performance benchmarks comparing HPFRACC
against external libraries and establishing performance baselines.
"""

import numpy as np
import time
import sys
import json
import psutil
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

# Add the library to the path
sys.path.insert(0, str(Path(__file__).parent))

from hpfracc.validation import BenchmarkSuite, PerformanceBenchmark, AccuracyBenchmark
from hpfracc.algorithms.optimized_methods import (
    OptimizedRiemannLiouville as RiemannLiouvilleDerivative,
    OptimizedCaputo as CaputoDerivative,
    OptimizedGrunwaldLetnikov as GrunwaldLetnikovDerivative,
)
from hpfracc.special import mittag_leffler_function as mittag_leffler
from hpfracc.special.binomial_coeffs import BinomialCoefficients
# from hpfracc.ml.spectral_autograd import SpectralFractionalLayer


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    warmup_runs: int = 3
    benchmark_runs: int = 10
    test_sizes: List[int] = None
    fractional_orders: List[float] = None
    
    def __post_init__(self):
        if self.test_sizes is None:
            self.test_sizes = [100, 500, 1000, 2000]
        if self.fractional_orders is None:
            self.fractional_orders = [0.25, 0.5, 0.75, 1.0]


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    method_name: str
    test_size: int
    fractional_order: float
    execution_time: float
    throughput: float
    memory_usage: float
    accuracy: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class ComprehensiveBenchmarker:
    """Comprehensive benchmarker for HPFRACC library."""
    
    def __init__(self, config: BenchmarkConfig = None, verbose: bool = True):
        self.config = config or BenchmarkConfig()
        self.verbose = verbose
        self.results = {
            'derivative_methods': {},
            'special_functions': {},
            'ml_layers': {},
            'scalability': {},
            'comparison': {},
            'summary': {}
        }
        self.start_time = time.time()
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[BENCHMARK] {message}")
    
    def benchmark_derivative_methods(self) -> Dict[str, Any]:
        """Benchmark fractional derivative methods."""
        self.log("Starting derivative methods benchmarking...")
        
        results = {
            'riemann_liouville': {},
            'caputo': {},
            'grunwald_letnikov': {},
            'comparison': {}
        }
        
        try:
            methods = {
                'riemann_liouville': RiemannLiouvilleDerivative,
                'caputo': CaputoDerivative,
                'grunwald_letnikov': GrunwaldLetnikovDerivative
            }
            
            # Test function: f(t) = t^2
            test_func = lambda t: t**2
            
            for method_name, method_class in methods.items():
                self.log(f"  Benchmarking {method_name}...")
                method_results = []
                
                for order in self.config.fractional_orders:
                    for size in self.config.test_sizes:
                        try:
                            x = np.linspace(0.1, 2.0, size)
                            deriv = method_class(order)
                            
                            # Warmup runs
                            for _ in range(self.config.warmup_runs):
                                try:
                                    deriv.compute(test_func, x)
                                except Exception:
                                    pass
                            
                            # Benchmark runs
                            execution_times = []
                            memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
                            
                            for _ in range(self.config.benchmark_runs):
                                start_time = time.perf_counter()
                                result = deriv.compute(test_func, x)
                                end_time = time.perf_counter()
                                execution_times.append(end_time - start_time)
                            
                            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
                            memory_usage = memory_after - memory_before
                            
                            avg_time = np.mean(execution_times)
                            throughput = size / avg_time
                            
                            benchmark_result = BenchmarkResult(
                                method_name=method_name,
                                test_size=size,
                                fractional_order=order,
                                execution_time=avg_time,
                                throughput=throughput,
                                memory_usage=memory_usage,
                                success=True
                            )
                            
                            method_results.append(benchmark_result)
                            
                        except Exception as e:
                            self.log(f"    Warning: {method_name} failed for order={order}, size={size}: {e}")
                            benchmark_result = BenchmarkResult(
                                method_name=method_name,
                                test_size=size,
                                fractional_order=order,
                                execution_time=0.0,
                                throughput=0.0,
                                memory_usage=0.0,
                                success=False,
                                error_message=str(e)
                            )
                            method_results.append(benchmark_result)
                
                results[method_name] = method_results
            
            # Generate comparison summary
            self._generate_derivative_comparison(results)
            
        except Exception as e:
            self.log(f"  Error in derivative methods benchmarking: {e}")
            results['error'] = str(e)
        
        self.log("  Derivative methods benchmarking completed.")
        return results
    
    def benchmark_special_functions(self) -> Dict[str, Any]:
        """Benchmark special functions."""
        self.log("Starting special functions benchmarking...")
        
        results = {
            'mittag_leffler': {},
            'binomial_coefficients': {},
            'comparison': {}
        }
        
        try:
            # Benchmark Mittag-Leffler function
            self.log("  Benchmarking Mittag-Leffler function...")
            ml_results = []
            
            test_points = [0.1, 0.5, 1.0, 2.0, 5.0]
            
            for z in test_points:
                for alpha in [0.25, 0.5, 0.75, 1.0]:
                    for beta in [0.5, 1.0, 1.5]:
                        try:
                            # Warmup runs
                            for _ in range(self.config.warmup_runs):
                                try:
                                    mittag_leffler(z, alpha, beta)
                                except Exception:
                                    pass
                            
                            # Benchmark runs
                            execution_times = []
                            memory_before = psutil.Process().memory_info().rss / (1024**2)
                            
                            for _ in range(self.config.benchmark_runs):
                                start_time = time.perf_counter()
                                result = mittag_leffler(z, alpha, beta)
                                end_time = time.perf_counter()
                                execution_times.append(end_time - start_time)
                            
                            memory_after = psutil.Process().memory_info().rss / (1024**2)
                            memory_usage = memory_after - memory_before
                            
                            avg_time = np.mean(execution_times)
                            throughput = 1.0 / avg_time
                            
                            benchmark_result = BenchmarkResult(
                                method_name='mittag_leffler',
                                test_size=1,
                                fractional_order=alpha,
                                execution_time=avg_time,
                                throughput=throughput,
                                memory_usage=memory_usage,
                                success=True
                            )
                            
                            ml_results.append(benchmark_result)
                            
                        except Exception as e:
                            self.log(f"    Warning: Mittag-Leffler failed for z={z}, alpha={alpha}, beta={beta}: {e}")
            
            results['mittag_leffler'] = ml_results
            
            # Benchmark binomial coefficients
            self.log("  Benchmarking binomial coefficients...")
            bc_results = []
            
            bc = BinomialCoefficients()
            test_cases = [(10, 5), (20, 10), (50, 25), (100, 50)]
            
            for n, k in test_cases:
                try:
                    # Warmup runs
                    for _ in range(self.config.warmup_runs):
                        try:
                            bc.compute(n, k)
                        except Exception:
                            pass
                    
                    # Benchmark runs
                    execution_times = []
                    memory_before = psutil.Process().memory_info().rss / (1024**2)
                    
                    for _ in range(self.config.benchmark_runs):
                        start_time = time.perf_counter()
                        result = bc.compute(n, k)
                        end_time = time.perf_counter()
                        execution_times.append(end_time - start_time)
                    
                    memory_after = psutil.Process().memory_info().rss / (1024**2)
                    memory_usage = memory_after - memory_before
                    
                    avg_time = np.mean(execution_times)
                    throughput = 1.0 / avg_time
                    
                    benchmark_result = BenchmarkResult(
                        method_name='binomial_coefficients',
                        test_size=n,
                        fractional_order=k,
                        execution_time=avg_time,
                        throughput=throughput,
                        memory_usage=memory_usage,
                        success=True
                    )
                    
                    bc_results.append(benchmark_result)
                    
                except Exception as e:
                    self.log(f"    Warning: Binomial coefficients failed for n={n}, k={k}: {e}")
            
            results['binomial_coefficients'] = bc_results
            
            # Generate comparison summary
            self._generate_special_functions_comparison(results)
            
        except Exception as e:
            self.log(f"  Error in special functions benchmarking: {e}")
            results['error'] = str(e)
        
        self.log("  Special functions benchmarking completed.")
        return results
    
    def benchmark_ml_layers(self) -> Dict[str, Any]:
        """Benchmark ML layers."""
        self.log("Starting ML layers benchmarking...")
        
        results = {
            'spectral_fractional_layer': {},
            'comparison': {}
        }
        
        try:
            import torch
            
            # Benchmark SpectralFractionalLayer
            self.log("  Benchmarking SpectralFractionalLayer...")
            spectral_results = []
            
            batch_sizes = [1, 10, 50, 100]
            feature_sizes = [10, 50, 100, 200]
            
            for batch_size in batch_sizes:
                for feature_size in feature_sizes:
                    for alpha in [0.25, 0.5, 0.75]:
                        try:
                            layer = SpectralFractionalLayer(alpha=alpha)
                            x = torch.randn(batch_size, feature_size)
                            
                            # Warmup runs
                            for _ in range(self.config.warmup_runs):
                                try:
                                    layer(x)
                                except Exception:
                                    pass
                            
                            # Benchmark runs
                            execution_times = []
                            memory_before = psutil.Process().memory_info().rss / (1024**2)
                            
                            for _ in range(self.config.benchmark_runs):
                                start_time = time.perf_counter()
                                output = layer(x)
                                end_time = time.perf_counter()
                                execution_times.append(end_time - start_time)
                            
                            memory_after = psutil.Process().memory_info().rss / (1024**2)
                            memory_usage = memory_after - memory_before
                            
                            avg_time = np.mean(execution_times)
                            throughput = (batch_size * feature_size) / avg_time
                            
                            benchmark_result = BenchmarkResult(
                                method_name='spectral_fractional_layer',
                                test_size=batch_size * feature_size,
                                fractional_order=alpha,
                                execution_time=avg_time,
                                throughput=throughput,
                                memory_usage=memory_usage,
                                success=True
                            )
                            
                            spectral_results.append(benchmark_result)
                            
                        except Exception as e:
                            self.log(f"    Warning: SpectralFractionalLayer failed for batch={batch_size}, features={feature_size}, alpha={alpha}: {e}")
            
            results['spectral_fractional_layer'] = spectral_results
            
            # Generate comparison summary
            self._generate_ml_layers_comparison(results)
            
        except Exception as e:
            self.log(f"  Error in ML layers benchmarking: {e}")
            results['error'] = str(e)
        
        self.log("  ML layers benchmarking completed.")
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with increasing problem sizes."""
        self.log("Starting scalability benchmarking...")
        
        results = {
            'derivative_scalability': {},
            'special_functions_scalability': {},
            'comparison': {}
        }
        
        try:
            # Test derivative methods scalability
            self.log("  Testing derivative methods scalability...")
            deriv = RiemannLiouvilleDerivative(0.5)
            test_func = lambda t: t**2
            
            scalability_sizes = [100, 500, 1000, 2000, 5000, 10000]
            deriv_scalability = []
            
            for size in scalability_sizes:
                try:
                    x = np.linspace(0.1, 2.0, size)
                    
                    # Warmup runs
                    for _ in range(self.config.warmup_runs):
                        try:
                            deriv.compute(test_func, x)
                        except Exception:
                            pass
                    
                    # Benchmark runs
                    execution_times = []
                    memory_before = psutil.Process().memory_info().rss / (1024**2)
                    
                    for _ in range(self.config.benchmark_runs):
                        start_time = time.perf_counter()
                        result = deriv.compute(test_func, x)
                        end_time = time.perf_counter()
                        execution_times.append(end_time - start_time)
                    
                    memory_after = psutil.Process().memory_info().rss / (1024**2)
                    memory_usage = memory_after - memory_before
                    
                    avg_time = np.mean(execution_times)
                    throughput = size / avg_time
                    
                    benchmark_result = BenchmarkResult(
                        method_name='riemann_liouville_scalability',
                        test_size=size,
                        fractional_order=0.5,
                        execution_time=avg_time,
                        throughput=throughput,
                        memory_usage=memory_usage,
                        success=True
                    )
                    
                    deriv_scalability.append(benchmark_result)
                    
                except Exception as e:
                    self.log(f"    Warning: Scalability test failed for size={size}: {e}")
            
            results['derivative_scalability'] = deriv_scalability
            
            # Test special functions scalability
            self.log("  Testing special functions scalability...")
            bc = BinomialCoefficients()
            
            bc_scalability = []
            n_values = [10, 50, 100, 200, 500, 1000]
            
            for n in n_values:
                k = n // 2  # Test middle value
                try:
                    # Warmup runs
                    for _ in range(self.config.warmup_runs):
                        try:
                            bc.compute(n, k)
                        except Exception:
                            pass
                    
                    # Benchmark runs
                    execution_times = []
                    memory_before = psutil.Process().memory_info().rss / (1024**2)
                    
                    for _ in range(self.config.benchmark_runs):
                        start_time = time.perf_counter()
                        result = bc.compute(n, k)
                        end_time = time.perf_counter()
                        execution_times.append(end_time - start_time)
                    
                    memory_after = psutil.Process().memory_info().rss / (1024**2)
                    memory_usage = memory_after - memory_before
                    
                    avg_time = np.mean(execution_times)
                    throughput = 1.0 / avg_time
                    
                    benchmark_result = BenchmarkResult(
                        method_name='binomial_coefficients_scalability',
                        test_size=n,
                        fractional_order=k,
                        execution_time=avg_time,
                        throughput=throughput,
                        memory_usage=memory_usage,
                        success=True
                    )
                    
                    bc_scalability.append(benchmark_result)
                    
                except Exception as e:
                    self.log(f"    Warning: Binomial coefficients scalability failed for n={n}: {e}")
            
            results['special_functions_scalability'] = bc_scalability
            
            # Generate scalability comparison
            self._generate_scalability_comparison(results)
            
        except Exception as e:
            self.log(f"  Error in scalability benchmarking: {e}")
            results['error'] = str(e)
        
        self.log("  Scalability benchmarking completed.")
        return results
    
    def _generate_derivative_comparison(self, results: Dict[str, Any]):
        """Generate comparison summary for derivative methods."""
        comparison = {
            'fastest_method': None,
            'most_memory_efficient': None,
            'best_throughput': None,
            'summary_stats': {}
        }
        
        # Aggregate results by method
        method_stats = {}
        
        for method_name in ['riemann_liouville', 'caputo', 'grunwald_letnikov']:
            if method_name in results:
                method_results = [r for r in results[method_name] if r.success]
                
                if method_results:
                    avg_throughput = np.mean([r.throughput for r in method_results])
                    avg_memory = np.mean([r.memory_usage for r in method_results])
                    avg_time = np.mean([r.execution_time for r in method_results])
                    
                    method_stats[method_name] = {
                        'avg_throughput': avg_throughput,
                        'avg_memory_usage': avg_memory,
                        'avg_execution_time': avg_time,
                        'total_tests': len(method_results),
                        'success_rate': len(method_results) / len(results[method_name]) if results[method_name] else 0
                    }
        
        # Find best performers
        if method_stats:
            comparison['fastest_method'] = max(method_stats.keys(), 
                                            key=lambda k: method_stats[k]['avg_throughput'])
            comparison['most_memory_efficient'] = min(method_stats.keys(), 
                                                    key=lambda k: method_stats[k]['avg_memory_usage'])
            comparison['best_throughput'] = max(method_stats.values(), 
                                             key=lambda v: v['avg_throughput'])['avg_throughput']
        
        comparison['summary_stats'] = method_stats
        results['comparison'] = comparison
    
    def _generate_special_functions_comparison(self, results: Dict[str, Any]):
        """Generate comparison summary for special functions."""
        comparison = {
            'mittag_leffler_stats': {},
            'binomial_coefficients_stats': {},
            'overall_summary': {}
        }
        
        # Mittag-Leffler stats
        if results.get('mittag_leffler'):
            ml_results = [r for r in results['mittag_leffler'] if r.success]
            if ml_results:
                comparison['mittag_leffler_stats'] = {
                    'avg_throughput': np.mean([r.throughput for r in ml_results]),
                    'avg_memory_usage': np.mean([r.memory_usage for r in ml_results]),
                    'avg_execution_time': np.mean([r.execution_time for r in ml_results]),
                    'total_tests': len(ml_results)
                }
        
        # Binomial coefficients stats
        if results.get('binomial_coefficients'):
            bc_results = [r for r in results['binomial_coefficients'] if r.success]
            if bc_results:
                comparison['binomial_coefficients_stats'] = {
                    'avg_throughput': np.mean([r.throughput for r in bc_results]),
                    'avg_memory_usage': np.mean([r.memory_usage for r in bc_results]),
                    'avg_execution_time': np.mean([r.execution_time for r in bc_results]),
                    'total_tests': len(bc_results)
                }
        
        results['comparison'] = comparison
    
    def _generate_ml_layers_comparison(self, results: Dict[str, Any]):
        """Generate comparison summary for ML layers."""
        comparison = {
            'spectral_layer_stats': {},
            'scalability_analysis': {}
        }
        
        if results.get('spectral_fractional_layer'):
            spectral_results = [r for r in results['spectral_fractional_layer'] if r.success]
            if spectral_results:
                comparison['spectral_layer_stats'] = {
                    'avg_throughput': np.mean([r.throughput for r in spectral_results]),
                    'avg_memory_usage': np.mean([r.memory_usage for r in spectral_results]),
                    'avg_execution_time': np.mean([r.execution_time for r in spectral_results]),
                    'total_tests': len(spectral_results)
                }
                
                # Analyze scalability by batch size
                batch_sizes = sorted(set(r.test_size for r in spectral_results))
                scalability_data = {}
                for batch_size in batch_sizes:
                    batch_results = [r for r in spectral_results if r.test_size == batch_size]
                    if batch_results:
                        scalability_data[batch_size] = {
                            'avg_throughput': np.mean([r.throughput for r in batch_results]),
                            'avg_memory_usage': np.mean([r.memory_usage for r in batch_results])
                        }
                comparison['scalability_analysis'] = scalability_data
        
        results['comparison'] = comparison
    
    def _generate_scalability_comparison(self, results: Dict[str, Any]):
        """Generate scalability comparison summary."""
        comparison = {
            'derivative_scalability_trend': {},
            'special_functions_scalability_trend': {},
            'scalability_summary': {}
        }
        
        # Derivative scalability trend
        if results.get('derivative_scalability'):
            deriv_results = [r for r in results['derivative_scalability'] if r.success]
            if deriv_results:
                sizes = [r.test_size for r in deriv_results]
                throughputs = [r.throughput for r in deriv_results]
                memory_usage = [r.memory_usage for r in deriv_results]
                
                comparison['derivative_scalability_trend'] = {
                    'sizes': sizes,
                    'throughputs': throughputs,
                    'memory_usage': memory_usage,
                    'scalability_factor': throughputs[-1] / throughputs[0] if len(throughputs) > 1 else 1.0
                }
        
        # Special functions scalability trend
        if results.get('special_functions_scalability'):
            bc_results = [r for r in results['special_functions_scalability'] if r.success]
            if bc_results:
                sizes = [r.test_size for r in bc_results]
                throughputs = [r.throughput for r in bc_results]
                memory_usage = [r.memory_usage for r in bc_results]
                
                comparison['special_functions_scalability_trend'] = {
                    'sizes': sizes,
                    'throughputs': throughputs,
                    'memory_usage': memory_usage,
                    'scalability_factor': throughputs[-1] / throughputs[0] if len(throughputs) > 1 else 1.0
                }
        
        results['comparison'] = comparison
    
    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        self.log("=" * 70)
        self.log("STARTING COMPREHENSIVE PERFORMANCE BENCHMARKING")
        self.log("=" * 70)
        
        try:
            # Run all benchmark categories
            self.results['derivative_methods'] = self.benchmark_derivative_methods()
            self.results['special_functions'] = self.benchmark_special_functions()
            # self.results['ml_layers'] = self.benchmark_ml_layers()
            self.results['scalability'] = self.benchmark_scalability()
            
            # Generate overall summary
            self.results['summary'] = self._generate_overall_summary()
            
            # Print final results
            self._print_summary()
            
            # Save results to file
            self._save_results()
            
            return self.results
            
        except Exception as e:
            self.log(f"FATAL ERROR in comprehensive benchmarking: {e}")
            self.log(traceback.format_exc())
            return {'error': str(e), 'success': False}
    
    def _generate_overall_summary(self) -> Dict[str, Any]:
        """Generate overall benchmark summary."""
        summary = {
            'total_benchmarks': 0,
            'successful_benchmarks': 0,
            'failed_benchmarks': 0,
            'success_rate': 0.0,
            'execution_time': time.time() - self.start_time,
            'best_performers': {},
            'performance_insights': {}
        }
        
        # Count total benchmarks
        total_count = 0
        success_count = 0
        
        for category in ['derivative_methods', 'special_functions', 'ml_layers', 'scalability']:
            if category in self.results:
                category_data = self.results[category]
                for method_name, method_results in category_data.items():
                    if isinstance(method_results, list):
                        total_count += len(method_results)
                        success_count += len([r for r in method_results if isinstance(r, BenchmarkResult) and r.success])
        
        summary['total_benchmarks'] = total_count
        summary['successful_benchmarks'] = success_count
        summary['failed_benchmarks'] = total_count - success_count
        
        if total_count > 0:
            summary['success_rate'] = success_count / total_count
        
        # Find best performers
        self._find_best_performers(summary)
        
        return summary
    
    def _find_best_performers(self, summary: Dict[str, Any]):
        """Find best performing methods."""
        best_performers = {}
        
        # Find best derivative method
        if 'derivative_methods' in self.results:
            deriv_results = self.results['derivative_methods']
            method_throughputs = {}
            
            for method_name in ['riemann_liouville', 'caputo', 'grunwald_letnikov']:
                if method_name in deriv_results:
                    method_data = deriv_results[method_name]
                    if isinstance(method_data, list):
                        successful_results = [r for r in method_data if isinstance(r, BenchmarkResult) and r.success]
                        if successful_results:
                            avg_throughput = np.mean([r.throughput for r in successful_results])
                            method_throughputs[method_name] = avg_throughput
            
            if method_throughputs:
                best_derivative = max(method_throughputs.keys(), key=lambda k: method_throughputs[k])
                best_performers['best_derivative_method'] = {
                    'method': best_derivative,
                    'throughput': method_throughputs[best_derivative]
                }
        
        summary['best_performers'] = best_performers
    
    def _print_summary(self):
        """Print benchmark summary."""
        self.log("=" * 70)
        self.log("COMPREHENSIVE BENCHMARKING SUMMARY")
        self.log("=" * 70)
        
        summary = self.results['summary']
        
        self.log(f"Total Benchmarks: {summary['total_benchmarks']}")
        self.log(f"Successful: {summary['successful_benchmarks']}")
        self.log(f"Failed: {summary['failed_benchmarks']}")
        self.log(f"Success Rate: {summary['success_rate']:.1%}")
        self.log(f"Execution Time: {summary['execution_time']:.2f} seconds")
        
        if summary['best_performers']:
            self.log("\nBest Performers:")
            for category, data in summary['best_performers'].items():
                self.log(f"  {category}: {data}")
        
        self.log("\n" + "=" * 70)
        if summary['success_rate'] > 0.8:
            self.log("🎉 BENCHMARKING: SUCCESS")
            self.log("Performance benchmarks completed successfully!")
        else:
            self.log("⚠️ BENCHMARKING: PARTIAL SUCCESS")
            self.log("Some benchmarks failed, but core performance data collected.")
        self.log("=" * 70)
    
    def _save_results(self):
        """Save benchmark results to file."""
        try:
            # Convert BenchmarkResult objects to dictionaries for JSON serialization
            serializable_results = {}
            
            for category, category_data in self.results.items():
                if category == 'summary':
                    serializable_results[category] = category_data
                else:
                    serializable_results[category] = {}
                    for method_name, method_data in category_data.items():
                        if isinstance(method_data, list):
                            serializable_results[category][method_name] = [
                                {
                                    'method_name': r.method_name,
                                    'test_size': r.test_size,
                                    'fractional_order': r.fractional_order,
                                    'execution_time': r.execution_time,
                                    'throughput': r.throughput,
                                    'memory_usage': r.memory_usage,
                                    'accuracy': r.accuracy,
                                    'success': r.success,
                                    'error_message': r.error_message
                                } for r in method_data
                            ]
                        else:
                            serializable_results[category][method_name] = method_data
            
            # Add metadata
            serializable_results['metadata'] = {
                'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'warmup_runs': self.config.warmup_runs,
                    'benchmark_runs': self.config.benchmark_runs,
                    'test_sizes': self.config.test_sizes,
                    'fractional_orders': self.config.fractional_orders
                },
                'platform': {
                    'python_version': sys.version,
                    'numpy_version': np.__version__,
                    'cpu_count': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / (1024**3)
                }
            }
            
            # Save to file
            output_file = Path(__file__).parent / 'comprehensive_benchmark_results.json'
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.log(f"Benchmark results saved to: {output_file}")
            
        except Exception as e:
            self.log(f"Warning: Could not save results to file: {e}")


def main():
    """Main function to run comprehensive benchmarks."""
    config = BenchmarkConfig(
        warmup_runs=3,
        benchmark_runs=5,  # Reduced for faster execution
        test_sizes=[100, 500, 1000],
        fractional_orders=[0.25, 0.5, 0.75]
    )
    
    benchmarker = ComprehensiveBenchmarker(config=config, verbose=True)
    results = benchmarker.run_comprehensive_benchmarks()
    
    # Exit with appropriate code
    if results.get('summary', {}).get('success_rate', 0) > 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Partial success or failure


if __name__ == "__main__":
    main()

