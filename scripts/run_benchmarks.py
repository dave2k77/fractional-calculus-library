#!/usr/bin/env python3
"""
Benchmark runner script for hpfracc library.

This script provides a command-line interface to run different types of benchmarks
using the dedicated benchmarking module.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hpfracc.benchmarks.benchmark_runner import BenchmarkRunner, BenchmarkConfig


def main():
    """Main function to run benchmarks based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for hpfracc library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick performance benchmark
  python scripts/run_benchmarks.py --type performance --quick
  
  # Run all benchmarks with custom configuration
  python scripts/run_benchmarks.py --type all --array-sizes 100 500 1000
  
  # Run accuracy benchmark only
  python scripts/run_benchmarks.py --type accuracy
  
  # Run scaling benchmark with specific methods
  python scripts/run_benchmarks.py --type scaling --methods RL GL Caputo
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["performance", "accuracy", "scaling", "memory", "all"],
        default="performance",
        help="Type of benchmark to run (default: performance)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick benchmark with reduced iterations and array sizes"
    )
    
    parser.add_argument(
        "--array-sizes", "-s",
        nargs="+",
        type=int,
        help="Custom array sizes for benchmarking"
    )
    
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        choices=["RL", "GL", "Caputo", "Weyl", "Marchaud"],
        help="Specific methods to benchmark"
    )
    
    parser.add_argument(
        "--fractional-orders", "-a",
        nargs="+",
        type=float,
        help="Fractional orders to test"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of iterations for each benchmark (default: 3)"
    )
    
    parser.add_argument(
        "--warmup-runs", "-w",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate comprehensive report with plots"
    )
    
    args = parser.parse_args()
    
    # Configure benchmark settings
    if args.quick:
        config = BenchmarkConfig(
            array_sizes=[100, 500, 1000],
            methods=args.methods or ["RL", "GL", "Caputo"],
            fractional_orders=args.fractional_orders or [0.5],
            test_functions=["polynomial"],
            iterations=2,
            warmup_runs=1,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
    else:
        config = BenchmarkConfig(
            array_sizes=args.array_sizes or [100, 500, 1000, 2000, 5000],
            methods=args.methods or ["RL", "GL", "Caputo", "Weyl", "Marchaud"],
            fractional_orders=args.fractional_orders or [0.25, 0.5, 0.75],
            test_functions=["polynomial", "exponential", "trigonometric"],
            iterations=args.iterations,
            warmup_runs=args.warmup_runs,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
    
    print(f"Running {args.type} benchmark...")
    print(f"Configuration: {config}")
    print("-" * 50)
    
    # Create benchmark runner
    runner = BenchmarkRunner(config)
    
    try:
        if args.type == "all":
            results = runner.run_all_benchmarks()
            if args.generate_report:
                runner.generate_report(results)
        elif args.type == "performance":
            results = {"performance": runner.run_performance_benchmark()}
        elif args.type == "accuracy":
            results = {"accuracy": runner.run_accuracy_benchmark()}
        elif args.type == "scaling":
            results = {"scaling": runner.run_scaling_benchmark()}
        elif args.type == "memory":
            results = {"memory": runner.run_memory_benchmark()}
        
        # Print summary
        print("\nBenchmark Results Summary:")
        print("=" * 50)
        
        for benchmark_type, result_list in results.items():
            print(f"\n{benchmark_type.upper()} BENCHMARK:")
            print("-" * 30)
            
            if not result_list:
                print("No results available")
                continue
            
            # Group by method
            method_stats = {}
            for result in result_list:
                method = result.method_name
                if method not in method_stats:
                    method_stats[method] = []
                method_stats[method].append(result)
            
            for method, method_results in method_stats.items():
                print(f"\n{method}:")
                
                # Performance stats
                times = [r.execution_time for r in method_results if r.execution_time != float('inf')]
                if times:
                    print(f"  Execution Time: {sum(times)/len(times):.6f}s (avg)")
                
                # Memory stats
                memory = [r.memory_usage for r in method_results if r.memory_usage > 0]
                if memory:
                    print(f"  Memory Usage: {sum(memory)/len(memory):.2f}MB (avg)")
                
                # Accuracy stats
                accuracy = [r.accuracy for r in method_results if r.accuracy is not None and r.accuracy != float('inf')]
                if accuracy:
                    print(f"  Relative Error: {sum(accuracy)/len(accuracy):.2e} (avg)")
        
        if args.generate_report:
            print(f"\nComprehensive report generated in {args.output_dir}/")
        
        if not args.no_save:
            print(f"\nResults saved to {args.output_dir}/")
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
