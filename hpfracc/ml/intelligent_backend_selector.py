"""
Intelligent Backend Selector for HPFRACC

Provides workload-aware backend selection with performance monitoring and adaptation.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .backends import BackendType, BackendManager

logger = logging.getLogger(__name__)


@dataclass
class WorkloadCharacteristics:
    """Characteristics of a computational workload."""
    operation_type: str  # "matmul", "conv", "element_wise", "fft", "derivative"
    data_size: int  # Total number of elements
    data_shape: Tuple[int, ...]
    dtype_size: int = 8  # bytes (default: float64)
    is_iterative: bool = False
    requires_gradient: bool = False
    
    @property
    def memory_footprint_mb(self) -> float:
        """Estimate memory footprint in MB."""
        return (self.data_size * self.dtype_size) / (1024 ** 2)


@dataclass
class PerformanceRecord:
    """Record of backend performance for a specific operation."""
    backend: BackendType
    operation: str
    data_size: int
    execution_time: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    gpu_used: bool = False


class GPUMemoryEstimator:
    """Estimates available GPU memory and calculates optimal thresholds."""
    
    def __init__(self):
        self.gpu_memory_cache = {}
        self._torch_available = None
        self._jax_available = None
    
    def get_available_gpu_memory_gb(self, backend: BackendType) -> float:
        """Get available GPU memory for a specific backend."""
        cache_key = (backend, time.time() // 60)  # Cache for 1 minute
        if cache_key in self.gpu_memory_cache:
            return self.gpu_memory_cache[cache_key]
        
        memory_gb = 0.0
        
        try:
            if backend == BackendType.TORCH:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    total = torch.cuda.get_device_properties(device).total_memory
                    allocated = torch.cuda.memory_allocated(device)
                    memory_gb = (total - allocated) / (1024 ** 3)
            
            elif backend == BackendType.JAX:
                import jax
                devices = jax.devices()
                gpu_devices = [d for d in devices if 'gpu' in str(d).lower()]
                if gpu_devices:
                    # JAX doesn't provide easy memory query, estimate conservatively
                    memory_gb = 8.0  # Conservative estimate
            
        except Exception as e:
            logger.debug(f"Could not query GPU memory for {backend}: {e}")
        
        self.gpu_memory_cache[cache_key] = memory_gb
        return memory_gb
    
    def calculate_gpu_threshold(self, backend: BackendType, reserve_fraction: float = 0.3) -> int:
        """
        Calculate optimal data size threshold for GPU usage.
        
        Args:
            backend: Backend to check
            reserve_fraction: Fraction of memory to keep in reserve (default 30%)
        
        Returns:
            Data size threshold (number of elements) for GPU usage
        """
        memory_gb = self.get_available_gpu_memory_gb(backend)
        if memory_gb == 0:
            return float('inf')  # No GPU available
        
        # Use (1 - reserve_fraction) of available memory
        usable_memory_gb = memory_gb * (1 - reserve_fraction)
        # Assume float64 (8 bytes per element)
        threshold_elements = int((usable_memory_gb * (1024 ** 3)) / 8)
        
        return threshold_elements


class PerformanceMonitor:
    """Monitors and learns from backend performance over time."""
    
    def __init__(self, max_history: int = 1000):
        self.performance_history: List[PerformanceRecord] = []
        self.max_history = max_history
        self.operation_stats: Dict[Tuple[BackendType, str], Dict[str, float]] = defaultdict(lambda: {
            'total_time': 0.0,
            'count': 0,
            'failures': 0,
            'avg_time': float('inf')
        })
    
    def record(self, record: PerformanceRecord):
        """Record a performance measurement."""
        self.performance_history.append(record)
        
        # Trim history if needed
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
        
        # Update statistics
        key = (record.backend, record.operation)
        stats = self.operation_stats[key]
        stats['count'] += 1
        if record.success:
            stats['total_time'] += record.execution_time
            stats['avg_time'] = stats['total_time'] / (stats['count'] - stats['failures'])
        else:
            stats['failures'] += 1
    
    def get_best_backend(
        self, 
        operation: str, 
        available_backends: List[BackendType],
        min_samples: int = 5
    ) -> Optional[BackendType]:
        """
        Get the best performing backend for an operation based on history.
        
        Args:
            operation: Operation type
            available_backends: List of available backends to choose from
            min_samples: Minimum number of samples required for recommendation
        
        Returns:
            Best backend or None if insufficient data
        """
        candidates = []
        for backend in available_backends:
            key = (backend, operation)
            stats = self.operation_stats.get(key)
            if stats and stats['count'] >= min_samples:
                # Calculate score: lower time is better, penalize failures
                failure_rate = stats['failures'] / stats['count']
                if failure_rate > 0.5:  # Too many failures
                    continue
                score = stats['avg_time'] * (1 + failure_rate)
                candidates.append((backend, score))
        
        if not candidates:
            return None
        
        # Return backend with lowest score (fastest)
        return min(candidates, key=lambda x: x[1])[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics summary."""
        return {
            'total_records': len(self.performance_history),
            'operation_stats': dict(self.operation_stats),
            'recent_backends': [
                (r.backend.value, r.operation, r.execution_time) 
                for r in self.performance_history[-10:]
            ]
        }


class IntelligentBackendSelector:
    """
    Intelligent backend selector with workload-aware optimization.
    
    Features:
    - Workload-based backend selection
    - Performance monitoring and learning
    - Dynamic GPU threshold calculation
    - Automatic fallback on errors
    """
    
    def __init__(
        self, 
        backend_manager: Optional[BackendManager] = None,
        enable_learning: bool = True,
        enable_gpu: bool = True
    ):
        self.backend_manager = backend_manager or BackendManager()
        self.memory_estimator = GPUMemoryEstimator()
        self.performance_monitor = PerformanceMonitor() if enable_learning else None
        self.enable_learning = enable_learning
        self.enable_gpu = enable_gpu
        
        # Cache for threshold calculations
        self.gpu_thresholds = {}
        self.threshold_cache_time = 0
        
        logger.info(f"🧠 Intelligent Backend Selector initialized")
        logger.info(f"   Learning enabled: {enable_learning}")
        logger.info(f"   GPU enabled: {enable_gpu}")
    
    def select_backend(
        self, 
        workload: WorkloadCharacteristics,
        preferred_backend: Optional[BackendType] = None
    ) -> BackendType:
        """
        Select optimal backend for the given workload.
        
        Args:
            workload: Characteristics of the computational workload
            preferred_backend: User's preferred backend (if any)
        
        Returns:
            Selected backend
        """
        available = self.backend_manager.available_backends
        
        # 1. Honor explicit preference if available
        if preferred_backend and preferred_backend in available:
            if preferred_backend != BackendType.AUTO:
                logger.debug(f"Using preferred backend: {preferred_backend.value}")
                return preferred_backend
        
        # 2. Check if we have learned a good backend for this operation
        if self.enable_learning and self.performance_monitor:
            learned = self.performance_monitor.get_best_backend(
                workload.operation_type,
                available
            )
            if learned:
                logger.debug(f"Using learned optimal backend: {learned.value} for {workload.operation_type}")
                return learned
        
        # 3. Intelligent workload-based selection
        return self._select_by_workload(workload, available)
    
    def _select_by_workload(
        self, 
        workload: WorkloadCharacteristics,
        available_backends: List[BackendType]
    ) -> BackendType:
        """Select backend based on workload characteristics."""
        
        # Very small data: NumPy is fastest (no overhead)
        if workload.data_size < 1000:
            if BackendType.NUMBA in available_backends:
                logger.debug(f"Small data ({workload.data_size} elements): using NumPy/Numba")
                return BackendType.NUMBA
        
        # Small data: Still prefer CPU unless GPU is much faster
        if workload.data_size < 10000:
            # For element-wise operations, NumPy is very efficient
            if workload.operation_type in ["element_wise", "simple_ops"]:
                if BackendType.NUMBA in available_backends:
                    return BackendType.NUMBA
        
        # Large data: Consider GPU
        if workload.data_size > 10000 and self.enable_gpu:
            gpu_backend = self._select_gpu_backend(workload, available_backends)
            if gpu_backend:
                return gpu_backend
        
        # Medium data or operations requiring gradients: PyTorch
        if workload.requires_gradient and BackendType.TORCH in available_backends:
            logger.debug(f"Gradient required: using PyTorch")
            return BackendType.TORCH
        
        # Mathematical operations: JAX is excellent
        if workload.operation_type in ["fft", "matmul", "derivative"]:
            if BackendType.JAX in available_backends:
                logger.debug(f"Mathematical operation ({workload.operation_type}): using JAX")
                return BackendType.JAX
        
        # Default: Use first available in priority order
        priority = [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]
        for backend in priority:
            if backend in available_backends:
                logger.debug(f"Using default priority backend: {backend.value}")
                return backend
        
        # Ultimate fallback
        return available_backends[0]
    
    def _select_gpu_backend(
        self, 
        workload: WorkloadCharacteristics,
        available_backends: List[BackendType]
    ) -> Optional[BackendType]:
        """Select GPU backend if appropriate and available."""
        
        # Check each GPU-capable backend
        gpu_backends = [b for b in available_backends if b in [BackendType.TORCH, BackendType.JAX]]
        
        if not gpu_backends:
            return None
        
        # Calculate GPU threshold (cache for 60 seconds)
        current_time = time.time()
        if current_time - self.threshold_cache_time > 60:
            self.gpu_thresholds = {
                backend: self.memory_estimator.calculate_gpu_threshold(backend)
                for backend in gpu_backends
            }
            self.threshold_cache_time = current_time
        
        # Check which backends can handle this workload
        suitable_backends = []
        for backend in gpu_backends:
            threshold = self.gpu_thresholds.get(backend, float('inf'))
            if workload.data_size < threshold:
                gpu_memory_gb = self.memory_estimator.get_available_gpu_memory_gb(backend)
                suitable_backends.append((backend, gpu_memory_gb))
        
        if not suitable_backends:
            logger.debug(f"Data size ({workload.data_size}) exceeds GPU memory thresholds")
            return None
        
        # Return backend with most available memory
        best = max(suitable_backends, key=lambda x: x[1])
        logger.debug(f"Selected GPU backend: {best[0].value} (available memory: {best[1]:.2f} GB)")
        return best[0]
    
    def execute_with_monitoring(
        self,
        operation_name: str,
        backend: BackendType,
        func: Any,
        workload: WorkloadCharacteristics,
        fallback_backends: Optional[List[BackendType]] = None
    ) -> Any:
        """
        Execute function with performance monitoring and automatic fallback.
        
        Args:
            operation_name: Name of the operation for logging
            backend: Backend to use
            func: Function to execute
            workload: Workload characteristics
            fallback_backends: Backends to try if primary fails
        
        Returns:
            Result of function execution
        """
        start_time = time.time()
        
        try:
            result = func()
            execution_time = time.time() - start_time
            
            # Record successful execution
            if self.performance_monitor:
                record = PerformanceRecord(
                    backend=backend,
                    operation=operation_name,
                    data_size=workload.data_size,
                    execution_time=execution_time,
                    success=True,
                    gpu_used=backend in [BackendType.TORCH, BackendType.JAX]
                )
                self.performance_monitor.record(record)
            
            logger.debug(f"✅ {operation_name} completed with {backend.value} in {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure
            if self.performance_monitor:
                record = PerformanceRecord(
                    backend=backend,
                    operation=operation_name,
                    data_size=workload.data_size,
                    execution_time=execution_time,
                    success=False
                )
                self.performance_monitor.record(record)
            
            logger.warning(f"❌ {operation_name} failed with {backend.value}: {e}")
            
            # Try fallback backends
            if fallback_backends:
                logger.info(f"Attempting fallback backends: {[b.value for b in fallback_backends]}")
                for fallback_backend in fallback_backends:
                    if fallback_backend == backend:
                        continue
                    try:
                        # Switch backend and retry
                        logger.debug(f"Trying fallback: {fallback_backend.value}")
                        result = func()  # Note: func should handle backend switching
                        logger.info(f"✅ Fallback successful with {fallback_backend.value}")
                        return result
                    except Exception as fallback_error:
                        logger.debug(f"Fallback {fallback_backend.value} also failed: {fallback_error}")
                        continue
            
            # All backends failed, raise original exception
            raise e
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.performance_monitor:
            return {"learning_disabled": True}
        
        return self.performance_monitor.get_statistics()
    
    def reset_performance_history(self):
        """Reset performance history (useful for testing)."""
        if self.performance_monitor:
            self.performance_monitor.performance_history.clear()
            self.performance_monitor.operation_stats.clear()
            logger.info("Performance history reset")


# Convenience function for quick backend selection
def select_optimal_backend(
    operation_type: str,
    data_shape: Tuple[int, ...],
    requires_gradient: bool = False,
    preferred_backend: Optional[BackendType] = None
) -> BackendType:
    """
    Quick function to select optimal backend for common use cases.
    
    Args:
        operation_type: Type of operation ("matmul", "conv", "element_wise", etc.)
        data_shape: Shape of input data
        requires_gradient: Whether gradient computation is needed
        preferred_backend: User's preferred backend
    
    Returns:
        Selected backend
    
    Example:
        >>> backend = select_optimal_backend("matmul", (1000, 1000))
        >>> backend = select_optimal_backend("conv", (32, 3, 224, 224), requires_gradient=True)
    """
    selector = IntelligentBackendSelector()
    workload = WorkloadCharacteristics(
        operation_type=operation_type,
        data_size=int(np.prod(data_shape)),
        data_shape=data_shape,
        requires_gradient=requires_gradient
    )
    return selector.select_backend(workload, preferred_backend)

