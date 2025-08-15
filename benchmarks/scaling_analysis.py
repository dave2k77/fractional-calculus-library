#!/usr/bin/env python3
"""
Scaling Analysis for Fractional Calculus Library

This module provides comprehensive scaling analysis benchmarks to
understand how the library performs with different problem sizes,
hardware configurations, and parallel processing capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Updated imports for consolidated structure
from src.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    OptimizedGrunwaldLetnikov,
    AdvancedFFTMethods,
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov,
)
from src.algorithms.gpu_optimized_methods import (
    GPUOptimizedCaputo,
    GPUOptimizedRiemannLiouville,
    GPUOptimizedGrunwaldLetnikov,
    JAXAutomaticDifferentiation,
    JAXOptimizer,
)
from src.algorithms.parallel_optimized_methods import (
    ParallelOptimizedCaputo,
    ParallelOptimizedRiemannLiouville,
    ParallelOptimizedGrunwaldLetnikov,
    NumbaOptimizer,
    NumbaFractionalKernels,
    NumbaParallelManager,
)


class ScalingAnalysis:
    """Comprehensive scaling analysis for fractional calculus."""

    def __init__(self):
        """Initialize scaling analysis suite."""
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information."""
        return {
            "cpu_count": mp.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "platform": os.name,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        }

    def analyze_computational_scaling(
        self, grid_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Analyze computational scaling with problem size."""
        print("📊 Computational Scaling Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

        # Test parameters
        alpha = 0.5
        t_max = 2.0

        # Initialize methods (will be recreated for each alpha)
        method_classes = {
            "Optimized Caputo": OptimizedCaputo,
            "Optimized Riemann-Liouville": OptimizedRiemannLiouville,
            "Optimized Grünwald-Letnikov": OptimizedGrunwaldLetnikov,
            "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
            "FFT Convolution": lambda: AdvancedFFTMethods(method="convolution"),
        }

        results = {}

        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")

            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t / 2)

            method_results = {}

            for method_name, method_class in method_classes.items():
                try:
                    # Initialize method with current alpha
                    if method_name.startswith("FFT"):
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
                        "time": end_time - start_time,
                        "memory_usage": self._get_memory_usage(),
                        "result_shape": (
                            result.shape if hasattr(result, "shape") else None
                        ),
                        "success": True,
                    }

                    print(
                        f"  ✅ {method_name}: {method_results[method_name]['time']:.4f}s"
                    )

                except Exception as e:
                    method_results[method_name] = {
                        "time": None,
                        "memory_usage": None,
                        "error": str(e),
                        "success": False,
                    }
                    print(f"  ❌ {method_name}: Error - {e}")

            results[N] = method_results

        return results

    def analyze_parallel_scaling(
        self, grid_sizes: List[int] = None, worker_counts: List[int] = None
    ) -> Dict[str, Any]:
        """Analyze parallel scaling with different numbers of workers."""
        print("\n🔄 Parallel Scaling Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [1000, 2000, 5000]

        if worker_counts is None:
            worker_counts = [1, 2, 4, 8, 16]

        # Test parameters
        alpha = 0.5
        t_max = 2.0

        results = {}

        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")
            results[N] = {}

            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t / 2)

            # Test different parallel backends
            backends = ["joblib", "multiprocessing", "threading"]

            for backend in backends:
                print(f"  🧪 Testing {backend} backend...")
                backend_results = {}

                for n_workers in worker_counts:
                    try:
                        # Create parallel computing instance
                        parallel_computing = ParallelOptimizedCaputo(
                            num_workers=n_workers, backend=backend
                        )

                        # Prepare multiple datasets for parallel processing
                        datasets = []
                        for i in range(20):  # 20 different datasets
                            f_shifted = f * (1 + 0.05 * i)
                            datasets.append(f_shifted)

                        # Time parallel computation
                        start_time = time.time()
                        results_parallel = (
                            parallel_computing.parallel_fractional_derivative(
                                OptimizedCaputo().compute,
                                [datasets] * len(datasets),
                                [t] * len(datasets),
                                alpha,
                                h,
                            )
                        )
                        end_time = time.time()

                        computation_time = end_time - start_time

                        # Calculate scaling metrics
                        speedup = 1.0  # Will be calculated relative to single worker
                        efficiency = 1.0  # Will be calculated as speedup / n_workers

                        backend_results[n_workers] = {
                            "time": computation_time,
                            "speedup": speedup,
                            "efficiency": efficiency,
                            "throughput": len(datasets) / computation_time,
                        }

                        print(f"    {n_workers} workers: {computation_time:.4f}s")

                    except Exception as e:
                        print(f"    ❌ {n_workers} workers: Failed - {e}")
                        backend_results[n_workers] = {
                            "time": np.inf,
                            "speedup": 0,
                            "efficiency": 0,
                            "throughput": 0,
                        }

                results[N][backend] = backend_results

        self.results["parallel_scaling"] = results
        return results

    def analyze_memory_scaling(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Analyze memory usage scaling with problem size."""
        print("\n💾 Memory Scaling Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [100, 500, 1000, 2000, 5000, 10000]

        # Test parameters
        alpha = 0.5
        t_max = 2.0

        # Initialize methods (will be recreated for each alpha)
        method_classes = {
            "Optimized Caputo": OptimizedCaputo,
            "Optimized Riemann-Liouville": OptimizedRiemannLiouville,
            "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
        }

        results = {}

        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")

            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t / 2)

            method_results = {}

            for method_name, method_class in method_classes.items():
                try:
                    # Initialize method with current alpha
                    if method_name.startswith("FFT"):
                        method = method_class()
                    else:
                        method = method_class(alpha)

                    # Monitor memory usage
                    process = psutil.Process()
                    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

                    # Clear memory
                    import gc

                    gc.collect()

                    # Measure memory before computation
                    memory_before = process.memory_info().rss / 1024 / 1024

                    # Run computation
                    if method_name.startswith("FFT"):
                        result = method.compute_derivative(f, t, alpha)
                    else:
                        result = method.compute(f, t, h)

                    # Measure memory after computation
                    memory_after = process.memory_info().rss / 1024 / 1024

                    # Calculate memory metrics
                    memory_used = memory_after - memory_before
                    result_memory = result.nbytes / 1024 / 1024
                    peak_memory = memory_after - initial_memory
                    memory_efficiency = (
                        result_memory / memory_used if memory_used > 0 else 0
                    )

                    method_results[method_name] = {
                        "memory_used": memory_used,
                        "result_memory": result_memory,
                        "peak_memory": peak_memory,
                        "memory_efficiency": memory_efficiency,
                        "memory_per_point": memory_used / N if N > 0 else 0,
                    }

                    print(
                        f"  ✅ {method_name}: {memory_used:.2f} MB, "
                        f"efficiency: {memory_efficiency:.2f}"
                    )

                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {e}")
                    method_results[method_name] = {
                        "memory_used": 0,
                        "result_memory": 0,
                        "peak_memory": 0,
                        "memory_efficiency": 0,
                        "memory_per_point": 0,
                    }

            results[N] = method_results

        self.results["memory_scaling"] = results
        return results

    def analyze_optimization_scaling(
        self, grid_sizes: List[int] = None
    ) -> Dict[str, Any]:
        """Analyze scaling of optimization backends (JAX, Numba)."""
        print("\n⚡ Optimization Backend Scaling Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [100, 500, 1000, 2000, 5000]

        # Test parameters
        alpha = 0.5
        t_max = 2.0

        results = {}

        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")

            # Create test data
            t = np.linspace(0, t_max, N)
            h = t[1] - t[0]
            f = np.sin(t) * np.exp(-t / 2)

            backend_results = {}

            # Test JAX backend
            try:
                print("  🧪 Testing JAX backend...")
                start_time = time.time()
                result_jax = JAXAutomaticDifferentiation.caputo_derivative_gpu(
                    f, t, alpha, h
                )
                end_time = time.time()

                backend_results["JAX GPU"] = {
                    "time": end_time - start_time,
                    "memory": result_jax.nbytes / 1024 / 1024,
                    "throughput": (
                        N / (end_time - start_time)
                        if (end_time - start_time) > 0
                        else 0
                    ),
                }
                print(f"    ✅ JAX GPU: {backend_results['JAX GPU']['time']:.4f}s")

            except Exception as e:
                print(f"    ❌ JAX GPU: Failed - {e}")
                backend_results["JAX GPU"] = {
                    "time": np.inf,
                    "memory": 0,
                    "throughput": 0,
                }

            # Test Numba backend
            try:
                print("  🧪 Testing Numba backend...")
                start_time = time.time()
                result_numba = NumbaFractionalKernels.caputo_l1_kernel(f, alpha, h)
                end_time = time.time()

                backend_results["Numba"] = {
                    "time": end_time - start_time,
                    "memory": result_numba.nbytes / 1024 / 1024,
                    "throughput": (
                        N / (end_time - start_time)
                        if (end_time - start_time) > 0
                        else 0
                    ),
                }
                print(f"    ✅ Numba: {backend_results['Numba']['time']:.4f}s")

            except Exception as e:
                print(f"    ❌ Numba: Failed - {e}")
                backend_results["Numba"] = {
                    "time": np.inf,
                    "memory": 0,
                    "throughput": 0,
                }

            results[N] = backend_results

        self.results["optimization_scaling"] = results
        return results

    def analyze_accuracy_scaling(self, grid_sizes: List[int] = None) -> Dict[str, Any]:
        """Analyze how accuracy scales with problem size."""
        print("\n🎯 Accuracy Scaling Analysis")
        print("=" * 60)

        if grid_sizes is None:
            grid_sizes = [25, 50, 100, 200, 500, 1000]

        # Test parameters
        alpha = 0.5
        t_max = 1.0

        # Analytical solution for comparison
        from scipy.special import gamma

        def analytical_solution(t, alpha):
            return t ** (1 - alpha) / gamma(2 - alpha)

        # Initialize methods (will be recreated for each alpha)
        method_classes = {
            "Optimized Caputo": OptimizedCaputo,
            "Optimized Riemann-Liouville": OptimizedRiemannLiouville,
            "Grünwald-Letnikov": OptimizedGrunwaldLetnikov,
            "FFT Spectral": lambda: AdvancedFFTMethods(method="spectral"),
        }

        results = {}

        for N in grid_sizes:
            print(f"\n📊 Testing grid size: {N}")

            # Create test data
            t = np.linspace(0.1, t_max, N)
            h = t[1] - t[0]
            f = t  # Simple linear function

            # Analytical solution
            analytical = analytical_solution(t, alpha)

            method_results = {}

            for method_name, method_class in method_classes.items():
                try:
                    # Initialize method with current alpha
                    if method_name.startswith("FFT"):
                        method = method_class()
                        numerical = method.compute_derivative(f, t, alpha)
                    else:
                        method = method_class(alpha)
                        numerical = method.compute(f, t, h)

                    # Calculate accuracy metrics
                    error = np.abs(numerical - analytical)
                    max_error = np.max(error)
                    mean_error = np.mean(error)
                    l2_error = np.sqrt(np.mean(error**2))
                    relative_error = np.mean(np.abs(error / (analytical + 1e-10)))

                    # Calculate convergence rate (if we have multiple grid sizes)
                    convergence_rate = -np.log10(max_error) if max_error > 0 else np.inf

                    method_results[method_name] = {
                        "max_error": max_error,
                        "mean_error": mean_error,
                        "l2_error": l2_error,
                        "relative_error": relative_error,
                        "convergence_rate": convergence_rate,
                        "accuracy_per_point": (
                            -np.log10(mean_error) if mean_error > 0 else np.inf
                        ),
                    }

                    print(
                        f"  ✅ {method_name}: max error {max_error:.2e}, "
                        f"convergence rate {convergence_rate:.2f}"
                    )

                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {e}")
                    method_results[method_name] = {
                        "max_error": np.inf,
                        "mean_error": np.inf,
                        "l2_error": np.inf,
                        "relative_error": np.inf,
                        "convergence_rate": 0,
                        "accuracy_per_point": 0,
                    }

            results[N] = method_results

        self.results["accuracy_scaling"] = results
        return results

    def generate_scaling_report(self) -> str:
        """Generate comprehensive scaling analysis report."""
        print("\n📊 Generating Scaling Analysis Report")
        print("=" * 60)

        report = []
        report.append("FRACTIONAL CALCULUS LIBRARY - SCALING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # System information
        report.append("SYSTEM INFORMATION:")
        report.append(f"  CPU Count: {self.system_info['cpu_count']}")
        report.append(f"  CPU Frequency: {self.system_info['cpu_freq']}")
        report.append(
            f"  Total Memory: {self.system_info['memory_total'] / 1024**3:.2f} GB"
        )
        report.append(
            f"  Available Memory: {self.system_info['memory_available'] / 1024**3:.2f} GB"
        )
        report.append(f"  Platform: {self.system_info['platform']}")
        report.append(f"  Python Version: {self.system_info['python_version']}")
        report.append("")

        # Computational scaling
        if "computational_scaling" in self.results:
            report.append("COMPUTATIONAL SCALING ANALYSIS:")
            for N, methods in self.results["computational_scaling"].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    if data["time"] is not None:
                        report.append(
                            f"    {method}: {data['time']:.4f}s, "
                            f"{data['memory_usage']:.2f} MB"
                        )
            report.append("")

        # Parallel scaling
        if "parallel_scaling" in self.results:
            report.append("PARALLEL SCALING ANALYSIS:")
            for N, backends in self.results["parallel_scaling"].items():
                report.append(f"  Grid Size {N}:")
                for backend, workers in backends.items():
                    for n_workers, data in workers.items():
                        if data["time"] < np.inf:
                            report.append(
                                f"    {backend} ({n_workers} workers): "
                                f"{data['time']:.4f}s, efficiency: {data['efficiency']:.2f}"
                            )
            report.append("")

        # Memory scaling
        if "memory_scaling" in self.results:
            report.append("MEMORY SCALING ANALYSIS:")
            for N, methods in self.results["memory_scaling"].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    report.append(
                        f"    {method}: {data['memory_used']:.2f} MB, "
                        f"efficiency: {data['memory_efficiency']:.2f}"
                    )
            report.append("")

        # Optimization scaling
        if "optimization_scaling" in self.results:
            report.append("OPTIMIZATION BACKEND SCALING:")
            for N, backends in self.results["optimization_scaling"].items():
                report.append(f"  Grid Size {N}:")
                for backend, data in backends.items():
                    if data["time"] < np.inf:
                        report.append(
                            f"    {backend}: {data['time']:.4f}s, "
                            f"throughput: {data['throughput']:.0f} ops/s"
                        )
            report.append("")

        # Accuracy scaling
        if "accuracy_scaling" in self.results:
            report.append("ACCURACY SCALING ANALYSIS:")
            for N, methods in self.results["accuracy_scaling"].items():
                report.append(f"  Grid Size {N}:")
                for method, data in methods.items():
                    if data["max_error"] < np.inf:
                        report.append(
                            f"    {method}: max error {data['max_error']:.2e}, "
                            f"convergence rate {data['convergence_rate']:.2f}"
                        )
            report.append("")

        report_text = "\n".join(report)

        # Save report to file
        with open("benchmarks/scaling_analysis_report.txt", "w") as f:
            f.write(report_text)

        print(
            "✅ Scaling analysis report generated and saved to 'benchmarks/scaling_analysis_report.txt'"
        )
        return report_text

    def plot_scaling_results(self):
        """Generate scaling visualization plots."""
        print("\n📈 Generating Scaling Analysis Plots")
        print("=" * 60)

        # Create plots directory
        os.makedirs("benchmarks/plots", exist_ok=True)

        # Plot 1: Computational scaling
        if "computational_scaling" in self.results:
            plt.figure(figsize=(15, 10))

            grid_sizes = list(self.results["computational_scaling"].keys())
            methods = list(self.results["computational_scaling"][grid_sizes[0]].keys())

            # Time scaling
            plt.subplot(2, 2, 1)
            for method in methods:
                times = []
                for N in grid_sizes:
                    time_val = self.results["computational_scaling"][N][method]["time"]
                    times.append(time_val if time_val is not None else np.nan)

                plt.loglog(
                    grid_sizes, times, "o-", label=method, linewidth=2, markersize=8
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Execution Time (s)")
            plt.title("Computational Time Scaling")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Memory scaling
            plt.subplot(2, 2, 2)
            for method in methods:
                memory_usage = []
                for N in grid_sizes:
                    memory_val = self.results["computational_scaling"][N][method][
                        "memory_usage"
                    ]
                    memory_usage.append(memory_val)

                plt.loglog(
                    grid_sizes,
                    memory_usage,
                    "o-",
                    label=method,
                    linewidth=2,
                    markersize=8,
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Memory Usage (MB)")
            plt.title("Memory Usage Scaling")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Throughput scaling
            plt.subplot(2, 2, 3)
            for method in methods:
                throughput = []
                for N in grid_sizes:
                    ops_val = self.results["computational_scaling"][N][method][
                        "ops_per_second"
                    ]
                    throughput.append(ops_val if ops_val > 0 else np.nan)

                plt.loglog(
                    grid_sizes,
                    throughput,
                    "o-",
                    label=method,
                    linewidth=2,
                    markersize=8,
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Operations per Second")
            plt.title("Throughput Scaling")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Complexity analysis
            plt.subplot(2, 2, 4)
            for method in methods:
                complexity = []
                for N in grid_sizes:
                    comp_val = self.results["computational_scaling"][N][method][
                        "complexity_ratio"
                    ]
                    complexity.append(comp_val if comp_val < np.inf else np.nan)

                plt.loglog(
                    grid_sizes,
                    complexity,
                    "o-",
                    label=method,
                    linewidth=2,
                    markersize=8,
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Time / (N log N)")
            plt.title("Computational Complexity")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "benchmarks/plots/computational_scaling.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

        # Plot 2: Parallel scaling
        if "parallel_scaling" in self.results:
            plt.figure(figsize=(15, 10))

            grid_sizes = list(self.results["parallel_scaling"].keys())
            backends = list(self.results["parallel_scaling"][grid_sizes[0]].keys())
            worker_counts = list(
                self.results["parallel_scaling"][grid_sizes[0]][backends[0]].keys()
            )

            for i, backend in enumerate(backends):
                plt.subplot(2, 2, i + 1)

                for N in grid_sizes:
                    times = []
                    for n_workers in worker_counts:
                        time_val = self.results["parallel_scaling"][N][backend][
                            n_workers
                        ]["time"]
                        times.append(time_val if time_val < np.inf else np.nan)

                    plt.semilogy(
                        worker_counts,
                        times,
                        "o-",
                        label=f"N={N}",
                        linewidth=2,
                        markersize=8,
                    )

                plt.xlabel("Number of Workers")
                plt.ylabel("Execution Time (s)")
                plt.title(f"Parallel Scaling: {backend}")
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                "benchmarks/plots/parallel_scaling.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        # Plot 3: Memory scaling
        if "memory_scaling" in self.results:
            plt.figure(figsize=(12, 8))

            grid_sizes = list(self.results["memory_scaling"].keys())
            methods = list(self.results["memory_scaling"][grid_sizes[0]].keys())

            for method in methods:
                memory_usage = []
                for N in grid_sizes:
                    memory_val = self.results["memory_scaling"][N][method][
                        "memory_used"
                    ]
                    memory_usage.append(memory_val)

                plt.loglog(
                    grid_sizes,
                    memory_usage,
                    "o-",
                    label=method,
                    linewidth=2,
                    markersize=8,
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Memory Usage (MB)")
            plt.title("Memory Scaling Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                "benchmarks/plots/memory_scaling.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        # Plot 4: Accuracy scaling
        if "accuracy_scaling" in self.results:
            plt.figure(figsize=(12, 8))

            grid_sizes = list(self.results["accuracy_scaling"].keys())
            methods = list(self.results["accuracy_scaling"][grid_sizes[0]].keys())

            for method in methods:
                errors = []
                for N in grid_sizes:
                    error_val = self.results["accuracy_scaling"][N][method]["max_error"]
                    errors.append(error_val if error_val < np.inf else np.nan)

                plt.loglog(
                    grid_sizes, errors, "o-", label=method, linewidth=2, markersize=8
                )

            plt.xlabel("Grid Size N")
            plt.ylabel("Maximum Error")
            plt.title("Accuracy Scaling Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                "benchmarks/plots/accuracy_scaling.png", dpi=300, bbox_inches="tight"
            )
            plt.show()

        print(
            "✅ Scaling analysis plots generated and saved to 'benchmarks/plots/' directory"
        )


def main():
    """Run comprehensive scaling analysis."""
    print("📊 Fractional Calculus Library - Scaling Analysis")
    print("=" * 70)

    # Initialize scaling analysis suite
    scaling = ScalingAnalysis()

    # Run analyses
    scaling.analyze_computational_scaling()
    scaling.analyze_parallel_scaling()
    scaling.analyze_memory_scaling()
    scaling.analyze_optimization_scaling()
    scaling.analyze_accuracy_scaling()

    # Generate report and plots
    scaling.generate_scaling_report()
    scaling.plot_scaling_results()

    print("\n🎉 All scaling analyses completed!")
    print("\n📁 Results saved to:")
    print("  - benchmarks/scaling_analysis_report.txt")
    print("  - benchmarks/plots/")


if __name__ == "__main__":
    main()
