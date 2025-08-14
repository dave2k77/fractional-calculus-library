"""
Advanced Parallel Computing for Fractional Calculus

This module provides advanced parallel computing strategies including
multi-core processing, distributed computing, load balancing, and
performance optimization for fractional calculus operations.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, Manager, Queue, Process
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import time
import psutil
from typing import Union, Optional, Tuple, Callable, Dict, Any, List
import os
import pickle

# Optional imports for alternative parallel computing backends
try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import dask.array as da
    import dask.distributed as dd
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    from celery import Celery

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from src.core.definitions import FractionalOrder


class ParallelComputingManager:
    """
    Advanced parallel computing manager for fractional calculus.

    Provides multi-core processing, distributed computing, and load
    balancing for high-performance fractional calculus computations.
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        backend: str = "joblib",  # Changed default to joblib
        n_jobs: int = -1,
    ):
        """
        Initialize parallel computing manager.

        Args:
            num_workers: Number of worker processes/threads
            backend: Backend for parallel processing ("multiprocessing", "threading", "joblib")
            n_jobs: Number of jobs for joblib (-1 for all cores)
        """
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers

        self.backend = backend
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()

        # Initialize pools
        self.process_pool = None
        self.thread_pool = None
        self.joblib_pool = None

    def __enter__(self):
        """Context manager entry."""
        if self.backend == "multiprocessing":
            self.process_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        elif self.backend == "threading":
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.process_pool:
            self.process_pool.shutdown()
        if self.thread_pool:
            self.thread_pool.shutdown()

    def parallel_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """
        Apply function in parallel to iterable.

        Args:
            func: Function to apply
            iterable: Items to process
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self.backend == "multiprocessing":
            return self._multiprocessing_map(func, iterable, **kwargs)
        elif self.backend == "threading":
            return self._threading_map(func, iterable, **kwargs)
        elif self.backend == "joblib":
            return self._joblib_map(func, iterable, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _multiprocessing_map(
        self, func: Callable, iterable: List[Any], **kwargs
    ) -> List[Any]:
        """Multiprocessing implementation."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            results = [future.result() for future in as_completed(futures)]
        return results

    def _threading_map(
        self, func: Callable, iterable: List[Any], **kwargs
    ) -> List[Any]:
        """Threading implementation."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(func, item) for item in iterable]
            results = [future.result() for future in as_completed(futures)]
        return results

    def _joblib_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Joblib implementation."""
        if JOBLIB_AVAILABLE:
            return Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(func)(item, **kwargs) for item in iterable
            )
        else:
            # Fallback to sequential processing
            return [func(item, **kwargs) for item in iterable]


class DistributedComputing:
    """
    Distributed computing utilities for fractional calculus.

    Provides distributed processing capabilities for large-scale
    fractional calculus computations.
    """

    def __init__(
        self, num_nodes: int = 1, node_id: int = 0, communication_backend: str = "mpi"
    ):
        """
        Initialize distributed computing.

        Args:
            num_nodes: Number of compute nodes
            node_id: Current node ID
            communication_backend: Communication backend ("mpi", "socket", "file")
        """
        self.num_nodes = num_nodes
        self.node_id = node_id
        self.communication_backend = communication_backend

        # Initialize communication
        self.comm = None
        self.dask_client = None
        self.ray_initialized = False

        if communication_backend == "mpi":
            try:
                from mpi4py import MPI

                self.comm = MPI.COMM_WORLD
                self.num_nodes = self.comm.Get_size()
                self.node_id = self.comm.Get_rank()
            except ImportError:
                print("Warning: mpi4py not available, trying alternative backends")
                self._initialize_alternative_backends()
        elif communication_backend == "dask":
            self._initialize_dask_backend()
        elif communication_backend == "ray":
            self._initialize_ray_backend()
        else:
            self._initialize_alternative_backends()

    def _initialize_alternative_backends(self):
        """Initialize alternative distributed computing backends."""
        if DASK_AVAILABLE:
            print("Using Dask for distributed computing")
            self._initialize_dask_backend()
        elif RAY_AVAILABLE:
            print("Using Ray for distributed computing")
            self._initialize_ray_backend()
        else:
            print("No distributed computing backend available, using single node")
            self.num_nodes = 1
            self.node_id = 0

    def _initialize_dask_backend(self):
        """Initialize Dask distributed computing backend."""
        if DASK_AVAILABLE:
            try:
                # Create local cluster for single machine
                cluster = LocalCluster(n_workers=self.num_workers, threads_per_worker=2)
                self.dask_client = Client(cluster)
                self.num_nodes = len(self.dask_client.scheduler_info()["workers"])
                self.node_id = 0  # Single machine setup
                print(f"Dask cluster initialized with {self.num_nodes} workers")
            except Exception as e:
                print(f"Failed to initialize Dask: {e}")
                self.num_nodes = 1
                self.node_id = 0
        else:
            print("Dask not available")
            self.num_nodes = 1
            self.node_id = 0

    def _initialize_ray_backend(self):
        """Initialize Ray distributed computing backend."""
        if RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(num_cpus=self.num_workers)
                self.ray_initialized = True
                self.num_nodes = 1  # Single machine setup
                self.node_id = 0
                print(f"Ray initialized with {self.num_workers} CPUs")
            except Exception as e:
                print(f"Failed to initialize Ray: {e}")
                self.num_nodes = 1
                self.node_id = 0
        else:
            print("Ray not available")
            self.num_nodes = 1
            self.node_id = 0

    def distribute_work(
        self, work_items: List[Any], strategy: str = "round_robin"
    ) -> List[Any]:
        """
        Distribute work across nodes.

        Args:
            work_items: Items to distribute
            strategy: Distribution strategy ("round_robin", "chunk", "dynamic")

        Returns:
            Work items for current node
        """
        if self.num_nodes == 1:
            return work_items

        if strategy == "round_robin":
            return work_items[self.node_id :: self.num_nodes]
        elif strategy == "chunk":
            chunk_size = len(work_items) // self.num_nodes
            start_idx = self.node_id * chunk_size
            end_idx = (
                start_idx + chunk_size
                if self.node_id < self.num_nodes - 1
                else len(work_items)
            )
            return work_items[start_idx:end_idx]
        elif strategy == "dynamic":
            # Dynamic load balancing
            return self._dynamic_distribution(work_items)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _dynamic_distribution(self, work_items: List[Any]) -> List[Any]:
        """Dynamic load balancing."""
        # Simplified dynamic distribution
        # In practice, you would implement more sophisticated load balancing
        chunk_size = max(1, len(work_items) // (self.num_nodes * 4))
        start_idx = self.node_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(work_items))
        return work_items[start_idx:end_idx]

    def gather_results(self, local_results: List[Any]) -> List[Any]:
        """
        Gather results from all nodes.

        Args:
            local_results: Results from current node

        Returns:
            Combined results from all nodes
        """
        if self.num_nodes == 1:
            return local_results

        if self.comm:
            # MPI communication
            all_results = self.comm.gather(local_results, root=0)
            if self.node_id == 0:
                # Flatten results
                combined_results = []
                for node_results in all_results:
                    combined_results.extend(node_results)
                return combined_results
            else:
                return []
        elif self.dask_client:
            # Dask communication (single machine setup)
            return local_results
        elif self.ray_initialized:
            # Ray communication (single machine setup)
            return local_results
        else:
            # Fallback to local results
            return local_results


class AlternativeParallelBackends:
    """
    Alternative parallel computing backends for fractional calculus.

    Provides implementations using Dask, Ray, Joblib, and other
    parallel computing frameworks as alternatives to MPI.
    """

    def __init__(self, backend: str = "auto", num_workers: Optional[int] = None):
        """
        Initialize alternative parallel backends.

        Args:
            backend: Backend to use ("dask", "ray", "joblib", "multiprocessing", "auto")
            num_workers: Number of workers
        """
        if num_workers is None:
            num_workers = mp.cpu_count()

        self.backend = backend
        self.num_workers = num_workers
        self.client = None

        if backend == "auto":
            self._auto_select_backend()
        else:
            self._initialize_backend(backend)

    def _auto_select_backend(self):
        """Automatically select the best available backend."""
        # Joblib is the best choice for scientific computing
        if JOBLIB_AVAILABLE:
            self._initialize_backend("joblib")
        elif DASK_AVAILABLE:
            self._initialize_backend("dask")
        elif RAY_AVAILABLE:
            self._initialize_backend("ray")
        else:
            self._initialize_backend("multiprocessing")

    def _initialize_backend(self, backend: str):
        """Initialize the specified backend."""
        if backend == "dask" and DASK_AVAILABLE:
            self._init_dask()
        elif backend == "ray" and RAY_AVAILABLE:
            self._init_ray()
        elif backend == "joblib" and JOBLIB_AVAILABLE:
            self._init_joblib()
        else:
            self._init_multiprocessing()

    def _init_dask(self):
        """Initialize Dask backend."""
        try:
            cluster = LocalCluster(n_workers=self.num_workers, threads_per_worker=2)
            self.client = Client(cluster)
            self.backend = "dask"
            print(f"Dask initialized with {self.num_workers} workers")
        except Exception as e:
            print(f"Failed to initialize Dask: {e}")
            self._init_multiprocessing()

    def _init_ray(self):
        """Initialize Ray backend."""
        try:
            if not ray.is_initialized():
                ray.init(num_cpus=self.num_workers)
            self.backend = "ray"
            print(f"Ray initialized with {self.num_workers} CPUs")
        except Exception as e:
            print(f"Failed to initialize Ray: {e}")
            self._init_multiprocessing()

    def _init_joblib(self):
        """Initialize Joblib backend."""
        self.backend = "joblib"
        print(f"Joblib initialized with {self.num_workers} workers")

    def _init_multiprocessing(self):
        """Initialize multiprocessing backend."""
        self.backend = "multiprocessing"
        print(f"Multiprocessing initialized with {self.num_workers} workers")

    def parallel_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """
        Apply function in parallel using the selected backend.

        Args:
            func: Function to apply
            iterable: Items to process
            **kwargs: Additional arguments

        Returns:
            List of results
        """
        if self.backend == "dask":
            return self._dask_map(func, iterable, **kwargs)
        elif self.backend == "ray":
            return self._ray_map(func, iterable, **kwargs)
        elif self.backend == "joblib":
            return self._joblib_map(func, iterable, **kwargs)
        else:
            return self._multiprocessing_map(func, iterable, **kwargs)

    def _dask_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Dask parallel mapping."""
        if self.client:
            # Convert to Dask array and map
            dask_array = da.from_array(
                iterable, chunks=len(iterable) // self.num_workers
            )
            futures = [self.client.submit(func, item, **kwargs) for item in iterable]
            results = self.client.gather(futures)
            return results
        else:
            return [func(item, **kwargs) for item in iterable]

    def _ray_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Ray parallel mapping."""
        if ray.is_initialized():
            # Remote function execution
            @ray.remote
            def remote_func(item):
                return func(item, **kwargs)

            futures = [remote_func.remote(item) for item in iterable]
            results = ray.get(futures)
            return results
        else:
            return [func(item, **kwargs) for item in iterable]

    def _joblib_map(self, func: Callable, iterable: List[Any], **kwargs) -> List[Any]:
        """Joblib parallel mapping."""
        results = Parallel(n_jobs=self.num_workers)(
            delayed(func)(item, **kwargs) for item in iterable
        )
        return results

    def _multiprocessing_map(
        self, func: Callable, iterable: List[Any], **kwargs
    ) -> List[Any]:
        """Multiprocessing parallel mapping."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(func, item, **kwargs) for item in iterable]
            results = [future.result() for future in as_completed(futures)]
        return results

    def shutdown(self):
        """Shutdown the backend."""
        if self.backend == "dask" and self.client:
            self.client.close()
        elif self.backend == "ray" and ray.is_initialized():
            ray.shutdown()


class LoadBalancer:
    """
    Load balancer for parallel fractional calculus computations.

    Provides intelligent load balancing strategies for optimal
    resource utilization.
    """

    def __init__(self, num_workers: int, strategy: str = "adaptive"):
        """
        Initialize load balancer.

        Args:
            num_workers: Number of worker processes
            strategy: Load balancing strategy ("static", "dynamic", "adaptive")
        """
        self.num_workers = num_workers
        self.strategy = strategy
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.worker_processes = []

    def create_work_chunks(
        self, work_items: List[Any], chunk_size: Optional[int] = None
    ) -> List[List[Any]]:
        """
        Create work chunks for distribution.

        Args:
            work_items: Items to chunk
            chunk_size: Size of each chunk (None for auto)

        Returns:
            List of work chunks
        """
        if chunk_size is None:
            chunk_size = max(1, len(work_items) // (self.num_workers * 4))

        chunks = []
        for i in range(0, len(work_items), chunk_size):
            chunks.append(work_items[i : i + chunk_size])

        return chunks

    def adaptive_chunk_size(
        self, work_items: List[Any], estimated_time_per_item: float
    ) -> int:
        """
        Calculate adaptive chunk size based on work characteristics.

        Args:
            work_items: Items to process
            estimated_time_per_item: Estimated time per item in seconds

        Returns:
            Optimal chunk size
        """
        total_work = len(work_items)
        total_estimated_time = total_work * estimated_time_per_item

        # Target: each worker gets work that takes ~1 second
        target_time_per_worker = 1.0
        optimal_chunk_size = max(
            1, int(target_time_per_worker / estimated_time_per_item)
        )

        # Ensure we don't have too many chunks
        max_chunks = self.num_workers * 10
        min_chunk_size = max(1, total_work // max_chunks)

        return max(min_chunk_size, optimal_chunk_size)

    def balance_work(
        self, work_items: List[Any], worker_capacities: Optional[List[float]] = None
    ) -> List[List[Any]]:
        """
        Balance work across workers based on their capacities.

        Args:
            work_items: Items to distribute
            worker_capacities: Relative capacities of workers (None for equal)

        Returns:
            Work distribution for each worker
        """
        if worker_capacities is None:
            worker_capacities = [1.0] * self.num_workers

        # Normalize capacities
        total_capacity = sum(worker_capacities)
        normalized_capacities = [cap / total_capacity for cap in worker_capacities]

        # Distribute work proportionally
        work_distribution = []
        start_idx = 0

        for i, capacity in enumerate(normalized_capacities):
            end_idx = start_idx + int(len(work_items) * capacity)
            if i == self.num_workers - 1:
                end_idx = len(work_items)  # Last worker gets remaining work

            work_distribution.append(work_items[start_idx:end_idx])
            start_idx = end_idx

        return work_distribution


class PerformanceOptimizer:
    """
    Performance optimization utilities for parallel computing.

    Provides tools for monitoring, analyzing, and optimizing
    parallel fractional calculus computations.
    """

    def __init__(self):
        """Initialize performance optimizer."""
        self.metrics = {}
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "timings": [],
            "throughput": [],
        }

    def record_metrics(
        self, cpu_usage: float, memory_usage: float, timing: float, throughput: float
    ):
        """
        Record performance metrics.

        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in MB
            timing: Execution time in seconds
            throughput: Operations per second
        """
        self.metrics["cpu_usage"].append(cpu_usage)
        self.metrics["memory_usage"].append(memory_usage)
        self.metrics["timings"].append(timing)
        self.metrics["throughput"].append(throughput)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Dictionary with performance metrics
        """
        if not self.metrics["timings"]:
            return {}

        return {
            "total_time": time.time() - self.start_time if self.start_time else 0,
            "avg_cpu_usage": np.mean(self.metrics["cpu_usage"]),
            "max_cpu_usage": np.max(self.metrics["cpu_usage"]),
            "avg_memory_usage": np.mean(self.metrics["memory_usage"]),
            "max_memory_usage": np.max(self.metrics["memory_usage"]),
            "avg_timing": np.mean(self.metrics["timings"]),
            "min_timing": np.min(self.metrics["timings"]),
            "max_timing": np.max(self.metrics["timings"]),
            "avg_throughput": np.mean(self.metrics["throughput"]),
            "total_operations": len(self.metrics["timings"]),
        }

    def optimize_parallel_parameters(
        self, work_size: int, num_workers: int, estimated_time_per_item: float
    ) -> Dict[str, Any]:
        """
        Optimize parallel processing parameters.

        Args:
            work_size: Number of work items
            num_workers: Number of available workers
            estimated_time_per_item: Estimated time per item

        Returns:
            Dictionary with optimized parameters
        """
        # Calculate optimal chunk size
        total_time = work_size * estimated_time_per_item
        target_time_per_chunk = 0.1  # 100ms per chunk
        optimal_chunk_size = max(
            1, int(target_time_per_chunk / estimated_time_per_item)
        )

        # Calculate optimal number of workers
        overhead_per_worker = 0.01  # 10ms overhead per worker
        optimal_workers = min(num_workers, int(total_time / overhead_per_worker))

        # Calculate expected performance
        expected_time = (
            total_time / optimal_workers + overhead_per_worker * optimal_workers
        )
        expected_throughput = work_size / expected_time

        return {
            "optimal_chunk_size": optimal_chunk_size,
            "optimal_workers": optimal_workers,
            "expected_time": expected_time,
            "expected_throughput": expected_throughput,
            "efficiency": total_time / (expected_time * optimal_workers),
        }


class ParallelFractionalComputing:
    """
    High-level parallel computing interface for fractional calculus.

    Provides easy-to-use parallel processing for fractional derivative
    and integral computations.
    """

    def __init__(
        self, num_workers: Optional[int] = None, backend: str = "joblib"
    ):  # Changed default to joblib
        """
        Initialize parallel fractional computing.

        Args:
            num_workers: Number of worker processes
            backend: Processing backend
        """
        self.parallel_manager = ParallelComputingManager(
            num_workers=num_workers, backend=backend
        )
        self.load_balancer = LoadBalancer(num_workers=num_workers or mp.cpu_count())
        self.performance_optimizer = PerformanceOptimizer()

    def parallel_fractional_derivative(
        self,
        derivative_func: Callable,
        f_values_list: List[np.ndarray],
        t_values_list: List[np.ndarray],
        alpha: float,
        h: float,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Compute fractional derivatives in parallel.

        Args:
            derivative_func: Derivative function to apply
            f_values_list: List of function value arrays
            t_values_list: List of time value arrays
            alpha: Fractional order
            h: Step size
            **kwargs: Additional arguments

        Returns:
            List of derivative results
        """
        # Prepare work items
        work_items = []
        for f_values, t_values in zip(f_values_list, t_values_list):
            work_items.append((derivative_func, f_values, t_values, alpha, h, kwargs))

        # Execute in parallel
        with self.parallel_manager:
            results = self.parallel_manager.parallel_map(
                self._compute_derivative_worker, work_items
            )

        return results

    def _compute_derivative_worker(self, work_item: Tuple) -> np.ndarray:
        """
        Worker function for derivative computation.

        Args:
            work_item: Tuple containing (func, f_values, t_values, alpha, h, kwargs)

        Returns:
            Derivative result
        """
        derivative_func, f_values, t_values, alpha, h, kwargs = work_item
        return derivative_func(f_values, t_values, alpha, h, **kwargs)

    def parallel_vectorized_derivative(
        self,
        derivative_func: Callable,
        f_values: np.ndarray,
        t_values: np.ndarray,
        alphas: np.ndarray,
        h: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute vectorized fractional derivatives in parallel.

        Args:
            derivative_func: Derivative function to apply
            f_values: Function values array
            t_values: Time values array
            alphas: Array of fractional orders
            h: Step size
            **kwargs: Additional arguments

        Returns:
            Array of derivative results
        """
        # Prepare work items for different alpha values
        work_items = []
        for alpha in alphas:
            work_items.append((derivative_func, f_values, t_values, alpha, h, kwargs))

        # Execute in parallel
        with self.parallel_manager:
            results = self.parallel_manager.parallel_map(
                self._compute_derivative_worker, work_items
            )

        return np.array(results)


# Convenience functions
def parallel_fractional_derivative(
    derivative_func: Callable,
    f_values_list: List[np.ndarray],
    t_values_list: List[np.ndarray],
    alpha: float,
    h: float,
    num_workers: Optional[int] = None,
    **kwargs,
) -> List[np.ndarray]:
    """
    Compute fractional derivatives in parallel.

    Args:
        derivative_func: Derivative function to apply
        f_values_list: List of function value arrays
        t_values_list: List of time value arrays
        alpha: Fractional order
        h: Step size
        num_workers: Number of worker processes
        **kwargs: Additional arguments

    Returns:
        List of derivative results
    """
    parallel_computing = ParallelFractionalComputing(num_workers=num_workers)
    return parallel_computing.parallel_fractional_derivative(
        derivative_func, f_values_list, t_values_list, alpha, h, **kwargs
    )


def optimize_parallel_parameters(
    work_size: int, num_workers: int, estimated_time_per_item: float
) -> Dict[str, Any]:
    """
    Optimize parallel processing parameters.

    Args:
        work_size: Number of work items
        num_workers: Number of available workers
        estimated_time_per_item: Estimated time per item

    Returns:
        Dictionary with optimized parameters
    """
    optimizer = PerformanceOptimizer()
    return optimizer.optimize_parallel_parameters(
        work_size, num_workers, estimated_time_per_item
    )


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for parallel computing.

    Returns:
        Dictionary with system information
    """
    return {
        "cpu_count": mp.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "platform": os.name,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
    }


def get_available_parallel_backends() -> Dict[str, bool]:
    """
    Get information about available parallel computing backends.

    Returns:
        Dictionary mapping backend names to availability status
    """
    return {
        "mpi4py": False,  # We know it's not available
        "dask": DASK_AVAILABLE,
        "ray": RAY_AVAILABLE,
        "joblib": JOBLIB_AVAILABLE,
        "multiprocessing": True,  # Always available
        "threading": True,  # Always available
        "celery": CELERY_AVAILABLE,
    }


def recommend_parallel_backend(use_case: str = "general") -> str:
    """
    Recommend the best parallel computing backend for a given use case.

    Args:
        use_case: Type of computation ("general", "large_data", "distributed", "simple")

    Returns:
        Recommended backend name
    """
    available = get_available_parallel_backends()

    # Joblib is the best choice for most use cases
    if available["joblib"]:
        return "joblib"
    elif use_case == "large_data" and available["dask"]:
        return "dask"
    elif use_case == "distributed" and available["ray"]:
        return "ray"
    else:
        return "multiprocessing"


def create_parallel_backend(
    backend: str = "auto", num_workers: Optional[int] = None
) -> AlternativeParallelBackends:
    """
    Create a parallel computing backend instance.

    Args:
        backend: Backend to use ("auto", "dask", "ray", "joblib", "multiprocessing")
        num_workers: Number of workers

    Returns:
        AlternativeParallelBackends instance
    """
    return AlternativeParallelBackends(backend=backend, num_workers=num_workers)
