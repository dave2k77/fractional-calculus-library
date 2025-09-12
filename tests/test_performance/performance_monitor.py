"""
Performance Monitoring Utilities

This module provides utilities for monitoring and analyzing performance
during regression testing.
"""

import time
import psutil
import gc
import torch
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    gpu_memory_usage: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class PerformanceMonitor:
    """Monitor performance during code execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.metrics_history = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None
    
    @contextmanager
    def monitor(self, operation_name: str = "operation"):
        """Context manager for monitoring performance."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Record initial state
        start_time = time.time()
        start_memory = self.get_memory_usage()
        start_cpu = self.get_cpu_usage()
        start_gpu_memory = self.get_gpu_memory_usage()
        
        try:
            yield
        finally:
            # Record final state
            end_time = time.time()
            end_memory = self.get_memory_usage()
            end_cpu = self.get_cpu_usage()
            end_gpu_memory = self.get_gpu_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2  # Average CPU usage
            gpu_memory_usage = None
            if start_gpu_memory is not None and end_gpu_memory is not None:
                gpu_memory_usage = end_gpu_memory - start_gpu_memory
            
            # Create metrics object
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gpu_memory_usage=gpu_memory_usage
            )
            
            # Store in history
            self.metrics_history.append({
                'operation': operation_name,
                'metrics': metrics
            })
    
    def measure_function(self, func: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of a function call."""
        with self.monitor(func.__name__):
            result = func(*args, **kwargs)
        
        # Return the most recent metrics
        return self.metrics_history[-1]['metrics']
    
    def get_average_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """Get average metrics for a specific operation."""
        operation_metrics = [
            entry['metrics'] for entry in self.metrics_history
            if entry['operation'] == operation_name
        ]
        
        if not operation_metrics:
            return None
        
        # Calculate averages
        avg_execution_time = np.mean([m.execution_time for m in operation_metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in operation_metrics])
        avg_cpu_usage = np.mean([m.cpu_usage for m in operation_metrics])
        
        gpu_memories = [m.gpu_memory_usage for m in operation_metrics if m.gpu_memory_usage is not None]
        avg_gpu_memory_usage = np.mean(gpu_memories) if gpu_memories else None
        
        return PerformanceMetrics(
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            cpu_usage=avg_cpu_usage,
            gpu_memory_usage=avg_gpu_memory_usage
        )
    
    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
    
    def export_metrics(self, filepath: str):
        """Export metrics to a file."""
        import json
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_history': [
                {
                    'operation': entry['operation'],
                    'metrics': {
                        'execution_time': entry['metrics'].execution_time,
                        'memory_usage': entry['metrics'].memory_usage,
                        'cpu_usage': entry['metrics'].cpu_usage,
                        'gpu_memory_usage': entry['metrics'].gpu_memory_usage,
                        'throughput': entry['metrics'].throughput,
                        'timestamp': entry['metrics'].timestamp
                    }
                }
                for entry in self.metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class PerformanceProfiler:
    """Advanced performance profiler for detailed analysis."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.profiles = {}
    
    def profile_derivative_computation(self, derivative_class, alpha: float, 
                                     test_func: Callable, x_vals: np.ndarray,
                                     iterations: int = 5) -> Dict[str, Any]:
        """Profile derivative computation performance."""
        profile_name = f"derivative_{derivative_class.__name__}"
        
        # Warm up
        deriv = derivative_class(alpha)
        deriv.compute(test_func, x_vals)
        
        # Profile multiple iterations
        times = []
        memory_usage = []
        
        for i in range(iterations):
            with self.monitor.monitor(f"{profile_name}_iter_{i}"):
                deriv = derivative_class(alpha)
                result = deriv.compute(test_func, x_vals)
            
            # Get metrics for this iteration
            metrics = self.monitor.metrics_history[-1]['metrics']
            times.append(metrics.execution_time)
            memory_usage.append(metrics.memory_usage)
        
        # Calculate statistics
        profile_data = {
            'class_name': derivative_class.__name__,
            'alpha': alpha,
            'input_size': len(x_vals),
            'iterations': iterations,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'throughput': len(x_vals) / np.mean(times),
            'timestamp': datetime.now().isoformat()
        }
        
        self.profiles[profile_name] = profile_data
        return profile_data
    
    def profile_neural_network(self, network_class, input_size: int, 
                             hidden_dims: List[int], output_size: int,
                             alpha: float, batch_size: int = 32,
                             epochs: int = 5) -> Dict[str, Any]:
        """Profile neural network performance."""
        profile_name = f"neural_network_{network_class.__name__}"
        
        # Create test data
        x_data = torch.randn(batch_size, input_size)
        y_data = torch.randn(batch_size, output_size)
        
        # Warm up
        network = network_class(input_size, hidden_dims, output_size, alpha)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Single forward pass for warm up
        output = network(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        
        # Profile training
        times = []
        memory_usage = []
        
        for epoch in range(epochs):
            with self.monitor.monitor(f"{profile_name}_epoch_{epoch}"):
                network = network_class(input_size, hidden_dims, output_size, alpha)
                optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
                
                for _ in range(3):  # 3 iterations per epoch
                    optimizer.zero_grad()
                    output = network(x_data)
                    loss = criterion(output, y_data)
                    loss.backward()
                    optimizer.step()
            
            # Get metrics for this epoch
            metrics = self.monitor.metrics_history[-1]['metrics']
            times.append(metrics.execution_time)
            memory_usage.append(metrics.memory_usage)
        
        # Calculate statistics
        profile_data = {
            'class_name': network_class.__name__,
            'input_size': input_size,
            'hidden_dims': hidden_dims,
            'output_size': output_size,
            'alpha': alpha,
            'batch_size': batch_size,
            'epochs': epochs,
            'avg_time_per_epoch': np.mean(times),
            'std_time_per_epoch': np.std(times),
            'total_time': np.sum(times),
            'avg_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'throughput': epochs / np.sum(times),
            'timestamp': datetime.now().isoformat()
        }
        
        self.profiles[profile_name] = profile_data
        return profile_data
    
    def profile_tensor_operations(self, operation_name: str, operation_func: Callable,
                                tensor_size: int, iterations: int = 100) -> Dict[str, Any]:
        """Profile tensor operations performance."""
        profile_name = f"tensor_ops_{operation_name}"
        
        # Create test tensors
        a = torch.randn(tensor_size, tensor_size)
        b = torch.randn(tensor_size, tensor_size)
        
        # Warm up
        operation_func(a, b)
        
        # Profile multiple iterations
        times = []
        memory_usage = []
        
        for i in range(iterations):
            with self.monitor.monitor(f"{profile_name}_iter_{i}"):
                result = operation_func(a, b)
            
            # Get metrics for this iteration
            metrics = self.monitor.metrics_history[-1]['metrics']
            times.append(metrics.execution_time)
            memory_usage.append(metrics.memory_usage)
        
        # Calculate statistics
        profile_data = {
            'operation_name': operation_name,
            'tensor_size': tensor_size,
            'iterations': iterations,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memory_usage),
            'std_memory': np.std(memory_usage),
            'throughput': iterations / np.sum(times),
            'timestamp': datetime.now().isoformat()
        }
        
        self.profiles[profile_name] = profile_data
        return profile_data
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        if not self.profiles:
            return {'message': 'No profiles available'}
        
        summary = {
            'total_profiles': len(self.profiles),
            'profile_types': list(set(profile['class_name'] if 'class_name' in profile 
                                    else profile['operation_name'] 
                                    for profile in self.profiles.values())),
            'profiles': self.profiles
        }
        
        return summary
    
    def export_profiles(self, filepath: str):
        """Export all profiles to a file."""
        import json
        
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'profiles': self.profiles
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Global instances
performance_monitor = PerformanceMonitor()
performance_profiler = PerformanceProfiler()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


def get_performance_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    return performance_profiler

