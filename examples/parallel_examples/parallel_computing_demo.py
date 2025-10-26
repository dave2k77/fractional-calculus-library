#!/usr/bin/env python3
"""
Parallel Computing Demo for Fractional Calculus

This example demonstrates the use of parallel computing backends (Joblib, Dask, Ray)
for accelerating fractional calculus computations.
"""

from hpfracc.algorithms.optimized_methods import OptimizedCaputo, JAX_AVAILABLE
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from joblib import Parallel, delayed

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_derivative(data):
    """Compute fractional derivative for parallel processing."""
    f_data, t_data, order_data, h_data = data
    
    # Temporarily disable JAX for multiprocessing safety
    original_jax_status = JAX_AVAILABLE
    if original_jax_status:
        from hpfracc.algorithms import optimized_methods
        optimized_methods.JAX_AVAILABLE = False

    caputo = OptimizedCaputo(order=order_data)
    result = caputo.compute(f_data, t_data, h_data)

    # Restore JAX status
    if original_jax_status:
        optimized_methods.JAX_AVAILABLE = True
        
    return result


def parallel_backend_comparison():
    """Compare different parallel computing backends."""
    print("🔄 Parallel Backend Comparison")
    print("=" * 50)

    # Check available backends
    print("Available parallel computing backends:")
    backends = ["loky", "multiprocessing", "threading"]
    available = {}

    # Check joblib availability
    try:
        pass

        available["loky"] = True
        print("  ✅ loky")
    except ImportError:
        available["loky"] = False
        print("  ❌ loky")

    # Check multiprocessing availability
    try:
        pass

        available["multiprocessing"] = True
        print("  ✅ multiprocessing")
    except ImportError:
        available["multiprocessing"] = False
        print("  ❌ multiprocessing")

    # Check threading availability
    try:
        pass

        available["threading"] = True
        print("  ✅ threading")
    except ImportError:
        available["threading"] = False
        print("  ❌ threading")

    # Test different backends
    backends = ["loky", "multiprocessing", "threading"]
    grid_sizes = [100, 500, 1000]
    alpha = 0.5

    results = {}

    for backend in backends:
        if not available.get(backend, False):
            print(f"⚠️  {backend} not available, skipping...")
            continue

        print(f"\n🧪 Testing {backend} backend...")

        backend_results = {}

        for N in grid_sizes:
            print(f"  📊 Grid size: {N}")

            # Create test data (avoid t=0 to prevent interpolation issues)
            t = np.linspace(0.01, 2, N)
            h = t[1] - t[0]
            f = np.sin(t)

            # Prepare work items (multiple datasets)
            work_items = []
            for i in range(10):  # 10 different datasets
                f_shifted = f * (1 + 0.1 * i)  # Slightly different functions
                work_items.append((f_shifted, t, alpha, h))

            # Time parallel computation
            start_time = time.time()
            Parallel(n_jobs=-1, backend=backend)(delayed(compute_derivative)(item) for item in work_items)
            end_time = time.time()

            backend_results[N] = end_time - start_time
            print(f"    ⏱️  Time: {backend_results[N]:.4f}s")

        results[backend] = backend_results

    # Plot comparison
    plt.figure(figsize=(12, 8))

    colors = ["r", "b", "g"]
    for i, (backend, backend_results) in enumerate(results.items()):
        grid_sizes_available = list(backend_results.keys())
        times = list(backend_results.values())
        plt.loglog(
            grid_sizes_available,
            times,
            "o-",
            color=colors[i],
            label=backend,
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Grid Size N")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Backend Performance Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    import os

    output_dir = os.path.join("examples", "parallel_examples")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "backend_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Parallel backend comparison completed!")


def joblib_optimization_demo():
    """Demonstrate Joblib optimization features."""
    print("\n⚡ Joblib Optimization Demo")
    print("=" * 50)

    # Create test data (avoid t=0 to prevent interpolation issues)
    t = np.linspace(0.01, 3, 500)
    h = t[1] - t[0]
    f = np.sin(t) * np.exp(-t / 2)
    alpha = 0.5

    print(f"Testing Joblib optimization for fractional derivatives")
    print(f"Grid size: {len(t)} points")

    # Test different numbers of workers
    worker_counts = [1, 2, 4, 8, 16]

    timings = {}

    for n_workers in worker_counts:
        print(f"\n🧪 Testing with {n_workers} workers...")

        # Prepare multiple datasets
        datasets = []
        for i in range(20):  # 20 different datasets
            f_shifted = f * (1 + 0.05 * i)
            datasets.append((f_shifted, t, alpha, h))

        # Time computation
        start_time = time.time()

        # Use parallel optimized Caputo for each dataset
        Parallel(n_jobs=n_workers, backend="loky")(delayed(compute_derivative)(item) for item in datasets)
        end_time = time.time()

        timings[n_workers] = end_time - start_time
        print(f"  ⏱️  Time: {timings[n_workers]:.4f}s")

    # Plot scaling
    plt.figure(figsize=(10, 6))

    workers = list(timings.keys())
    times = list(timings.values())

    plt.plot(
        workers, times, "bo-", linewidth=2, markersize=8, label="Actual Performance"
    )

    # Ideal scaling (linear speedup)
    ideal_times = [times[0] / w for w in workers]
    plt.plot(workers, ideal_times, "r--", linewidth=2,
             label="Ideal Scaling", alpha=0.7)

    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (s)")
    plt.title("Joblib Scaling Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/parallel_examples/joblib_scaling.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Joblib optimization demo completed!")


def vectorized_parallel_demo():
    """Demonstrate vectorized parallel processing."""
    print("\n📊 Vectorized Parallel Processing Demo")
    print("=" * 50)

    # Create test data (avoid t=0 to prevent interpolation issues)
    t = np.linspace(0.01, 2, 300)
    h = t[1] - t[0]
    f = np.sin(t)

    # Test different alpha values
    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    print(
        f"Computing vectorized fractional derivatives for {len(alphas)} α values")

    # Time vectorized computation
    start_time = time.time()

    # Use parallel optimized Caputo for each alpha value
    work_items = [(f, t, alpha_val, h) for alpha_val in alphas]
    results = Parallel(n_jobs=-1, backend="loky")(delayed(compute_derivative)(item) for item in work_items)
    end_time = time.time()

    print(f"⏱️  Vectorized computation time: {end_time - start_time:.4f}s")

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, f, "k-", linewidth=2, label="Original: f(t) = sin(t)")
    plt.xlabel("Time t")
    plt.ylabel("Function Value")
    plt.title("Original Function")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    for i, alpha in enumerate(alphas):
        plt.plot(t, results[i], linewidth=2,
                 label=f"Caputo Derivative (α={alpha:.1f})")

    plt.xlabel("Time t")
    plt.ylabel("Derivative Value")
    plt.title("Vectorized Parallel Caputo Derivatives")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/parallel_examples/vectorized_parallel.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Vectorized parallel processing demo completed!")


def load_balancing_demo():
    """Demonstrate load balancing strategies."""
    print("\n⚖️ Load Balancing Demo")
    print("=" * 50)

    from hpfracc.algorithms.optimized_methods import (
        ParallelLoadBalancer as LoadBalancer,
    )

    # Create test data with varying computational complexity (avoid t=0)
    t = np.linspace(0.01, 2, 200)
    h = t[1] - t[0]

    # Create work items with different complexities
    work_items = []
    for i in range(50):
        # Vary the complexity by changing the function
        complexity_factor = 1 + 0.5 * np.sin(i * np.pi / 10)
        f = np.sin(complexity_factor * t) * np.exp(-t / complexity_factor)
        work_items.append((f, t, 0.5, h))  # alpha = 0.5

    print(f"Created {len(work_items)} work items with varying complexity")

    # Test different load balancing strategies
    strategies = ["static", "dynamic", "adaptive"]
    num_workers = 8

    results = {}

    for strategy in strategies:
        print(f"\n🧪 Testing {strategy} load balancing...")

        # Create load balancer
        config = ParallelConfig(n_jobs=num_workers)
        load_balancer = LoadBalancer(config)

        # Create work chunks
        if strategy == "static":
            chunks = load_balancer.create_work_chunks(work_items)
        elif strategy == "dynamic":
            chunks = load_balancer.balance_work(work_items)
        else:  # adaptive
            estimated_time = 0.01  # Rough estimate
            chunk_size = load_balancer.adaptive_chunk_size(
                work_items, estimated_time)
            chunks = load_balancer.create_work_chunks(work_items, chunk_size)

        # Simulate work distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        load_imbalance = max(chunk_sizes) - min(chunk_sizes)

        results[strategy] = {
            "chunk_sizes": chunk_sizes,
            "load_imbalance": load_imbalance,
            "num_chunks": len(chunks),
        }

        print(f"  📊 Number of chunks: {len(chunks)}")
        print(f"  ⚖️  Load imbalance: {load_imbalance}")
        print(f"  📈 Chunk sizes: {chunk_sizes}")

    # Plot load balancing comparison
    plt.figure(figsize=(15, 5))

    for i, (strategy, result) in enumerate(results.items()):
        plt.subplot(1, 3, i + 1)
        plt.bar(range(len(result["chunk_sizes"])), result["chunk_sizes"])
        plt.xlabel("Worker ID")
        plt.ylabel("Work Items")
        plt.title(
            f'{strategy.capitalize()} Load Balancing\nImbalance: {result["load_imbalance"]}'
        )
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/parallel_examples/load_balancing.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Load balancing demo completed!")


def memory_optimization_demo():
    """Demonstrate memory optimization in parallel computing."""
    print("\n💾 Memory Optimization Demo")
    print("=" * 50)

    from hpfracc.optimisation.parallel_computing import NumbaMemoryOptimizer

    # Create large test data
    grid_sizes = [1000, 2000, 5000]
    alpha = 0.5

    results = {}

    for N in grid_sizes:
        print(f"\n📊 Testing grid size: {N}")

        t = np.linspace(0.01, 2, N)
        h = t[1] - t[0]
        f = np.sin(t) * np.exp(-t / 2)

        # Test memory-efficient computation
        memory_optimizer = NumbaMemoryOptimizer()

        # Time memory-efficient computation
        start_time = time.time()
        memory_optimizer.memory_efficient_caputo(f, alpha, h)
        end_time = time.time()

        results[N] = end_time - start_time
        print(f"  ⏱️  Memory-efficient time: {results[N]:.4f}s")
        print(f"  💾 Memory usage: ~{N * 8 / 1024 / 1024:.2f} MB")

    # Plot memory scaling
    plt.figure(figsize=(10, 6))

    grid_sizes_list = list(results.keys())
    times = list(results.values())

    plt.loglog(grid_sizes_list, times, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Grid Size N")
    plt.ylabel("Execution Time (s)")
    plt.title("Memory-Efficient Parallel Computing Scaling")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/parallel_examples/memory_optimization.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Memory optimization demo completed!")


def system_info_demo():
    """Display system information for parallel computing."""
    print("\n🖥️ System Information Demo")
    print("=" * 50)

    from hpfracc.optimisation.parallel_computing import get_system_info

    # Get system information
    system_info = get_system_info()

    print("System Information for Parallel Computing:")
    print(f"  🖥️  CPU Count: {system_info['cpu_count']}")
    print(f"  ⚡ CPU Frequency: {system_info['cpu_freq']}")
    print(f"  💾 Total Memory: {system_info['memory_total'] / 1024**3:.2f} GB")
    print(
        f"  📊 Available Memory: {system_info['memory_available'] / 1024**3:.2f} GB")
    print(f"  🖥️  Platform: {system_info['platform']}")
    print(f"  🐍 Python Version: {system_info['python_version']}")

    # Recommend optimal backend
    recommended = recommend_parallel_backend("general")
    print(f"\n🎯 Recommended backend: {recommended}")

    print("✅ System information demo completed!")


def main():
    """Run all parallel computing examples."""
    print("🚀 Parallel Computing Demo for Fractional Calculus")
    print("=" * 60)

    # Run examples
    parallel_backend_comparison()
    joblib_optimization_demo()
    vectorized_parallel_demo()

    print("\n🎉 All parallel computing examples completed!")
    print("\n📁 Generated plots saved in 'examples/parallel_examples/' directory")
    print("\n💡 Note: Performance may vary depending on your system configuration")


if __name__ == "__main__":
    main()
