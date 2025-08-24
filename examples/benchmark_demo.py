#!/usr/bin/env python3
"""
Demo script for the hpfracc benchmarking module.

This script demonstrates how to use the dedicated benchmarking system
to run different types of benchmarks.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hpfracc.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig


def demo_quick_benchmark():
    """Run a quick performance benchmark."""
    print("🚀 Running Quick Performance Benchmark")
    print("=" * 50)
    
    config = BenchmarkConfig(
        array_sizes=[100, 500, 1000],
        methods=["RL", "GL", "Caputo"],
        fractional_orders=[0.5],
        test_functions=["polynomial"],
        iterations=2,
        warmup_runs=1,
        save_results=True,
        output_dir="demo_benchmark_results"
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_performance_benchmark()
    
    print("\nQuick Benchmark Results:")
    print("-" * 30)
    for result in results:
        print(f"{result.method_name:8s} | Size: {result.array_size:4d} | "
              f"Time: {result.execution_time:.6f}s | "
              f"Memory: {result.memory_usage:.2f}MB")
    
    return results


def demo_accuracy_benchmark():
    """Run an accuracy benchmark."""
    print("\n🔬 Running Accuracy Benchmark")
    print("=" * 50)
    
    config = BenchmarkConfig(
        array_sizes=[50, 100, 200],
        methods=["RL", "GL", "Caputo"],
        fractional_orders=[0.5],
        test_functions=["polynomial"],
        iterations=2,
        warmup_runs=1,
        save_results=True,
        output_dir="demo_benchmark_results"
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_accuracy_benchmark()
    
    print("\nAccuracy Benchmark Results:")
    print("-" * 30)
    for result in results:
        if result.accuracy is not None and result.accuracy != float('inf'):
            print(f"{result.method_name:8s} | Size: {result.array_size:4d} | "
                  f"Error: {result.accuracy:.2e}")
        else:
            print(f"{result.method_name:8s} | Size: {result.array_size:4d} | "
                  f"Error: Failed")
    
    return results


def demo_scaling_benchmark():
    """Run a scaling benchmark."""
    print("\n📈 Running Scaling Benchmark")
    print("=" * 50)
    
    config = BenchmarkConfig(
        array_sizes=[100, 200, 500, 1000, 2000],
        methods=["RL", "GL", "Caputo"],
        fractional_orders=[0.5],
        test_functions=["polynomial"],
        iterations=2,
        warmup_runs=1,
        save_results=True,
        output_dir="demo_benchmark_results"
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_scaling_benchmark()
    
    print("\nScaling Benchmark Results:")
    print("-" * 30)
    for result in results:
        print(f"{result.method_name:8s} | Size: {result.array_size:4d} | "
              f"Time: {result.execution_time:.6f}s")
    
    return results


def demo_comprehensive_benchmark():
    """Run all benchmark types with report generation."""
    print("\n🌟 Running Comprehensive Benchmark Suite")
    print("=" * 50)
    
    config = BenchmarkConfig(
        array_sizes=[100, 500, 1000],
        methods=["RL", "GL", "Caputo"],
        fractional_orders=[0.5],
        test_functions=["polynomial"],
        iterations=2,
        warmup_runs=1,
        save_results=True,
        output_dir="demo_benchmark_results"
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_all_benchmarks()
    
    # Generate comprehensive report
    print("\n📊 Generating Comprehensive Report...")
    runner.generate_report(results)
    
    print(f"\n✅ All benchmarks completed!")
    print(f"📁 Results saved to: {config.output_dir}/")
    print(f"📈 Plots generated: performance, accuracy, scaling, memory")
    print(f"📋 Summary report: benchmark_summary.txt")
    
    return results


def main():
    """Main demo function."""
    print("🎯 HPFRACC Benchmarking Module Demo")
    print("=" * 60)
    print("This demo shows how to use the dedicated benchmarking system")
    print("to run different types of benchmarks for the hpfracc library.")
    print()
    
    try:
        # Run quick benchmark
        demo_quick_benchmark()
        
        # Run accuracy benchmark
        demo_accuracy_benchmark()
        
        # Run scaling benchmark
        demo_scaling_benchmark()
        
        # Run comprehensive benchmark suite
        demo_comprehensive_benchmark()
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Check the 'demo_benchmark_results' directory for results")
        print("2. Use the command-line script: python scripts/run_benchmarks.py --help")
        print("3. Customize benchmarks using BenchmarkConfig class")
        print("4. Integrate benchmarking into your CI/CD pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure the hpfracc library is properly installed and accessible.")
        sys.exit(1)


if __name__ == "__main__":
    main()
