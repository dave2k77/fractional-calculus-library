"""
ML Performance Benchmarking System

This module provides comprehensive benchmarking for the ML integration system,
measuring performance across different components and configurations.
"""

import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager

from ..ml import (
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    LayerConfig
)
from ..core.definitions import FractionalOrder


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs"""
    # Data dimensions
    batch_sizes: List[int] = None
    input_sizes: List[int] = None
    hidden_sizes: List[List[int]] = None
    sequence_lengths: List[int] = None

    # Fractional orders to test
    fractional_orders: List[float] = None

    # Methods to test
    methods: List[str] = None

    # Number of warmup and measurement runs
    warmup_runs: int = 5
    measurement_runs: int = 10

    # Device configuration
    device: str = "cpu"

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 32, 128]
        if self.input_sizes is None:
            self.input_sizes = [10, 50, 100, 500]
        if self.hidden_sizes is None:
            self.hidden_sizes = [[32], [64, 32], [
                128, 64, 32], [256, 128, 64, 32]]
        if self.sequence_lengths is None:
            self.sequence_lengths = [10, 50, 100, 500]
        if self.fractional_orders is None:
            self.fractional_orders = [0.1, 0.5, 0.9, 1.0]
        if self.methods is None:
            self.methods = ["RL", "Caputo", "GL"]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    component: str
    configuration: Dict[str, Any]
    execution_time: float
    memory_usage: float
    accuracy: Optional[float] = None
    throughput: Optional[float] = None
    error: Optional[str] = None


class MLPerformanceBenchmark:
    """Comprehensive ML performance benchmarking system"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.device = torch.device(config.device)

        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    @contextmanager
    def measure_performance(self, component: str, config: Dict[str, Any]):
        """Context manager for measuring performance metrics"""
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            try:
                self._warmup_run(component, config)
            except Exception:
                pass

        # Synchronize if using CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Measurement runs
        start_time = time.time()
        for _ in range(self.config.measurement_runs):
            try:
                self._measurement_run(component, config)
            except Exception as e:
                yield BenchmarkResult(
                    component=component,
                    configuration=config,
                    execution_time=0.0,
                    memory_usage=0.0,
                    error=str(e)
                )
                return

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        execution_time = (time.time() - start_time) / \
            self.config.measurement_runs

        # Memory after
        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before

        yield BenchmarkResult(
            component=component,
            configuration=config,
            execution_time=execution_time,
            memory_usage=memory_usage
        )

    def _warmup_run(self, component: str, config: Dict[str, Any]):
        """Perform a warmup run"""
        if component == "FractionalNeuralNetwork":
            self._benchmark_fractional_network_warmup(config)
        elif component == "FractionalAttention":
            self._benchmark_fractional_attention_warmup(config)
        elif component == "FractionalLayers":
            self._benchmark_fractional_layers_warmup(config)

    def _measurement_run(self, component: str, config: Dict[str, Any]):
        """Perform a measurement run"""
        if component == "FractionalNeuralNetwork":
            self._benchmark_fractional_network_measurement(config)
        elif component == "FractionalAttention":
            self._benchmark_fractional_attention_measurement(config)
        elif component == "FractionalLayers":
            self._benchmark_fractional_layers_measurement(config)

    def benchmark_fractional_networks(self):
        """Benchmark fractional neural networks"""
        print("🧠 Benchmarking Fractional Neural Networks...")

        for batch_size in self.config.batch_sizes:
            for input_size in self.config.input_sizes:
                for hidden_sizes in self.config.hidden_sizes:
                    for alpha in self.config.fractional_orders:
                        for method in self.config.methods:
                            config = {
                                "batch_size": batch_size,
                                "input_size": input_size,
                                "hidden_sizes": hidden_sizes,
                                "output_size": 3,
                                "fractional_order": alpha,
                                "method": method
                            }

                            with self.measure_performance("FractionalNeuralNetwork", config) as result:
                                if result.error is None:
                                    self.results.append(result)
                                    print(
                                        f"   ✅ {batch_size}x{input_size} → {hidden_sizes} (α={alpha}, {method})")
                                else:
                                    print(
                                        f"   ❌ {batch_size}x{input_size} → {hidden_sizes} (α={alpha}, {method}): {result.error}")

    def benchmark_fractional_attention(self):
        """Benchmark fractional attention mechanisms"""
        print("🧠 Benchmarking Fractional Attention...")

        for seq_len in self.config.sequence_lengths:
            for batch_size in self.config.batch_sizes:
                for d_model in [32, 64, 128]:
                    for alpha in self.config.fractional_orders:
                        for method in self.config.methods:
                            config = {
                                "sequence_length": seq_len,
                                "batch_size": batch_size,
                                "d_model": d_model,
                                "n_heads": 8,
                                "fractional_order": alpha,
                                "method": method
                            }

                            with self.measure_performance("FractionalAttention", config) as result:
                                if result.error is None:
                                    self.results.append(result)
                                    print(
                                        f"   ✅ {seq_len}x{batch_size}x{d_model} (α={alpha}, {method})")
                                else:
                                    print(
                                        f"   ❌ {seq_len}x{batch_size}x{d_model} (α={alpha}, {method}): {result.error}")

    def benchmark_fractional_layers(self):
        """Benchmark individual fractional layers"""
        print("🔧 Benchmarking Fractional Layers...")

        # Test different layer types
        layer_configs = [
            ("FractionalConv1D", {"in_channels": 3,
             "out_channels": 16, "kernel_size": 3}),
            ("FractionalConv2D", {"in_channels": 3,
             "out_channels": 16, "kernel_size": 3}),
            ("FractionalLSTM", {"input_size": 64, "hidden_size": 128}),
            ("FractionalTransformer", {"d_model": 64, "nhead": 8}),
            ("FractionalPooling", {"kernel_size": 2}),
            ("FractionalBatchNorm1d", {"num_features": 64})
        ]

        for layer_name, layer_params in layer_configs:
            for batch_size in self.config.batch_sizes:
                for alpha in self.config.fractional_orders:
                    for method in self.config.methods:
                        config = {
                            "layer_type": layer_name,
                            "layer_params": layer_params,
                            "batch_size": batch_size,
                            "fractional_order": alpha,
                            "method": method
                        }

                        with self.measure_performance("FractionalLayers", config) as result:
                            if result.error is None:
                                self.results.append(result)
                                print(
                                    f"   ✅ {layer_name} {batch_size} (α={alpha}, {method})")
                            else:
                                print(
                                    f"   ❌ {layer_name} {batch_size} (α={alpha}, {method}): {result.error}")

    def _benchmark_fractional_network_warmup(self, config: Dict[str, Any]):
        """Warmup run for fractional neural network"""
        net = FractionalNeuralNetwork(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            output_size=config["output_size"],
            fractional_order=config["fractional_order"]
        ).to(self.device)

        x = torch.randn(config["batch_size"],
                        config["input_size"], device=self.device)
        _ = net(x, use_fractional=True, method=config["method"])

    def _benchmark_fractional_network_measurement(
            self, config: Dict[str, Any]):
        """Measurement run for fractional neural network"""
        net = FractionalNeuralNetwork(
            input_size=config["input_size"],
            hidden_sizes=config["hidden_sizes"],
            output_size=config["output_size"],
            fractional_order=config["fractional_order"]
        ).to(self.device)

        x = torch.randn(config["batch_size"],
                        config["input_size"], device=self.device)
        output = net(x, use_fractional=True, method=config["method"])

        # Calculate throughput
        throughput = config["batch_size"] / config.get("execution_time", 1.0)

        # Calculate accuracy (simple regression R²)
        target = torch.randn_like(output)
        nn.MSELoss()(output, target)
        ss_res = torch.sum((target - output) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return output, throughput, r2.item()

    def _benchmark_fractional_attention_warmup(self, config: Dict[str, Any]):
        """Warmup run for fractional attention"""
        attention = FractionalAttention(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            fractional_order=config["fractional_order"]
        ).to(self.device)

        x = torch.randn(config["sequence_length"], config["batch_size"],
                        config["d_model"], device=self.device)
        _ = attention(x, method=config["method"])

    def _benchmark_fractional_attention_measurement(
            self, config: Dict[str, Any]):
        """Measurement run for fractional attention"""
        attention = FractionalAttention(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            fractional_order=config["fractional_order"]
        ).to(self.device)

        x = torch.randn(config["sequence_length"], config["batch_size"],
                        config["d_model"], device=self.device)
        output = attention(x, method=config["method"])

        # Calculate throughput
        throughput = config["batch_size"] * \
            config["sequence_length"] / config.get("execution_time", 1.0)

        return output, throughput

    def _benchmark_fractional_layers_warmup(self, config: Dict[str, Any]):
        """Warmup run for fractional layers"""
        layer_config = LayerConfig(
            fractional_order=FractionalOrder(config["fractional_order"]),
            method=config["method"],
            use_fractional=True
        )

        if config["layer_type"] == "FractionalConv1D":
            layer = FractionalConv1D(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["in_channels"], 32, device=self.device)
        elif config["layer_type"] == "FractionalConv2D":
            layer = FractionalConv2D(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["in_channels"], 16, 16, device=self.device)
        elif config["layer_type"] == "FractionalLSTM":
            layer = FractionalLSTM(config=layer_config,
                                   **config["layer_params"]).to(self.device)
            x = torch.randn(
                32,
                config["batch_size"],
                config["layer_params"]["input_size"],
                device=self.device)
        elif config["layer_type"] == "FractionalTransformer":
            layer = FractionalTransformer(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(
                config["batch_size"],
                32,
                config["layer_params"]["d_model"],
                device=self.device)
        elif config["layer_type"] == "FractionalPooling":
            layer = FractionalPooling(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], 16,
                            16, 16, device=self.device)
        elif config["layer_type"] == "FractionalBatchNorm1d":
            layer = FractionalBatchNorm1d(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["num_features"], 32, device=self.device)

        _ = layer(x)

    def _benchmark_fractional_layers_measurement(self, config: Dict[str, Any]):
        """Measurement run for fractional layers"""
        layer_config = LayerConfig(
            fractional_order=FractionalOrder(config["fractional_order"]),
            method=config["method"],
            use_fractional=True
        )

        if config["layer_type"] == "FractionalConv1D":
            layer = FractionalConv1D(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["in_channels"], 32, device=self.device)
        elif config["layer_type"] == "FractionalConv2D":
            layer = FractionalConv2D(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["in_channels"], 16, 16, device=self.device)
        elif config["layer_type"] == "FractionalLSTM":
            layer = FractionalLSTM(config=layer_config,
                                   **config["layer_params"]).to(self.device)
            x = torch.randn(
                32,
                config["batch_size"],
                config["layer_params"]["input_size"],
                device=self.device)
        elif config["layer_type"] == "FractionalTransformer":
            layer = FractionalTransformer(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(
                config["batch_size"],
                32,
                config["layer_params"]["d_model"],
                device=self.device)
        elif config["layer_type"] == "FractionalPooling":
            layer = FractionalPooling(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], 16,
                            16, 16, device=self.device)
        elif config["layer_type"] == "FractionalBatchNorm1d":
            layer = FractionalBatchNorm1d(
                config=layer_config, **config["layer_params"]).to(self.device)
            x = torch.randn(config["batch_size"], config["layer_params"]
                            ["num_features"], 32, device=self.device)

        output = layer(x)
        return output

    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("🚀 Starting Comprehensive ML Performance Benchmark")
        print("=" * 60)

        # Run all benchmark categories
        self.benchmark_fractional_networks()
        self.benchmark_fractional_attention()
        self.benchmark_fractional_layers()

        print("\n✅ Benchmarking completed!")
        print(f"📊 Total benchmark runs: {len(self.results)}")

        return self.results

    def generate_report(self, output_dir: str = "benchmark_results"):
        """Generate comprehensive benchmark report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save raw results
        results_file = output_path / "ml_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump([result.__dict__ for result in self.results],
                      f, indent=2, default=str)

        # Generate summary statistics
        self._generate_summary_statistics(output_path)

        # Generate visualizations
        self._generate_visualizations(output_path)

        print(f"📊 Benchmark report generated in: {output_path.absolute()}")

    def _generate_summary_statistics(self, output_path: Path):
        """Generate summary statistics"""
        if not self.results:
            return

        # Group by component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)

        # Calculate statistics
        stats = {}
        for component, results in components.items():
            if not results or any(r.error for r in results):
                continue

            times = [r.execution_time for r in results if r.error is None]
            memories = [r.memory_usage for r in results if r.error is None]

            if times:
                stats[component] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times),
                    "avg_memory": np.mean(memories),
                    "min_memory": np.min(memories),
                    "max_memory": np.max(memories)
                }

        # Save statistics
        stats_file = output_path / "ml_benchmark_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\n📊 Benchmark Summary Statistics:")
        print("-" * 40)
        for component, stat in stats.items():
            print(f"{component}:")
            print(f"  Runs: {stat['count']}")
            print(f"  Time: {stat['avg_time']:.6f}s ± {stat['std_time']:.6f}s")
            print(
                f"  Memory: {stat['avg_memory']:.2f}MB ± {stat['max_memory'] - stat['min_memory']:.2f}MB")
            print()

    def _generate_visualizations(self, output_path: Path):
        """Generate performance visualizations"""
        if not self.results:
            return

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Execution time by component
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("ML Performance Benchmark Results", fontsize=16)

        # Filter out error results
        valid_results = [r for r in self.results if r.error is None]

        if valid_results:
            # Time distribution by component
            [r.component for r in valid_results]
            times = [r.execution_time for r in valid_results]

            axes[0, 0].hist(times, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel("Execution Time (s)")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Execution Time Distribution")

            # Memory usage by component
            memories = [r.memory_usage for r in valid_results]
            axes[0, 1].hist(memories, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel("Memory Usage (MB)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Memory Usage Distribution")

            # Time vs Memory scatter
            axes[1, 0].scatter(times, memories, alpha=0.6)
            axes[1, 0].set_xlabel("Execution Time (s)")
            axes[1, 0].set_ylabel("Memory Usage (MB)")
            axes[1, 0].set_title("Time vs Memory Usage")

            # Component performance comparison
            component_times = {}
            for r in valid_results:
                if r.component not in component_times:
                    component_times[r.component] = []
                component_times[r.component].append(r.execution_time)

            component_avg_times = {comp: np.mean(
                times) for comp, times in component_times.items()}
            axes[1, 1].bar(component_avg_times.keys(),
                           component_avg_times.values())
            axes[1, 1].set_xlabel("Component")
            axes[1, 1].set_ylabel("Average Execution Time (s)")
            axes[1, 1].set_title("Component Performance Comparison")
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "ml_benchmark_visualizations.png",
                    dpi=300, bbox_inches='tight')
        plt.close()


def run_ml_benchmark():
    """Run the ML performance benchmark"""
    # Configuration
    config = BenchmarkConfig(
        batch_sizes=[1, 8, 32, 128],
        input_sizes=[10, 50, 100, 500],
        hidden_sizes=[[32], [64, 32], [128, 64, 32]],
        sequence_lengths=[10, 50, 100, 500],
        fractional_orders=[0.1, 0.5, 0.9, 1.0],
        methods=["RL", "Caputo", "GL"],
        warmup_runs=3,
        measurement_runs=5,
        device="cpu"
    )

    # Create benchmark instance
    benchmark = MLPerformanceBenchmark(config)

    # Run benchmarks
    results = benchmark.run_comprehensive_benchmark()

    # Generate report
    benchmark.generate_report()

    return results


if __name__ == "__main__":
    run_ml_benchmark()
