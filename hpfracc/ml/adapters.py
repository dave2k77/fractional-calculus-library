"""
High-Performance Backend Tensor Adapters

This module provides zero-overhead abstractions over different tensor libraries
with intelligent backend selection and performance optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict, Callable, Union
import importlib
import functools
import time
import os

from .backends import BackendType

# Module-level import function that can be mocked in tests
_import_module = importlib.import_module


# Global cache for imported libraries and capabilities
_LIB_CACHE: Dict[BackendType, Any] = {}
_CAPABILITIES_CACHE: Dict[BackendType, 'Capabilities'] = {}
_PERFORMANCE_PROFILES: Dict[BackendType, 'PerformanceProfile'] = {}


@dataclass(frozen=True)
class Capabilities:
    """Backend capability flags with performance characteristics."""
    device_kind: str  # 'cpu' | 'gpu' | 'auto'
    has_fft: bool
    has_autograd: bool
    supports_amp: bool
    supports_jit: bool
    memory_limit_gb: float = 0.0  # 0 = unlimited
    preferred_operations: frozenset = frozenset()


@dataclass(frozen=True)
class PerformanceProfile:
    """Performance characteristics for backend selection."""
    small_data_threshold: int = 1000  # elements
    large_data_threshold: int = 100000  # elements
    gpu_memory_threshold: int = 1000000  # elements
    preferred_for_math: bool = False
    preferred_for_nn: bool = False
    preferred_for_arrays: bool = False
    jit_compilation_time: float = 0.0  # seconds


class HighPerformanceAdapter:
    """
    High-performance adapter with intelligent backend selection.
    
    This adapter:
    1. Caches imports and capabilities
    2. Selects optimal backend based on operation and data
    3. Provides zero-overhead access to native libraries
    4. Optimizes for specific use cases
    """

    def __init__(self, backend: Optional[BackendType] = None):
        self.backend = backend or self._select_optimal_backend()
        self._lib = self._get_cached_lib()
        self._capabilities = self._get_cached_capabilities()
        self._performance_profile = self._get_performance_profile()

    def _select_optimal_backend(self) -> BackendType:
        """Select optimal backend based on environment and capabilities."""
        # Check environment variables for forced backend
        if os.getenv("HPFRACC_FORCE_TORCH", "0") == "1":
            return BackendType.TORCH
        if os.getenv("HPFRACC_FORCE_JAX", "0") == "1":
            return BackendType.JAX
        if os.getenv("HPFRACC_FORCE_NUMPY", "0") == "1":
            return BackendType.NUMBA

        # Check for disabled backends
        disabled = {
            BackendType.TORCH: os.getenv("HPFRACC_DISABLE_TORCH", "0") == "1",
            BackendType.JAX: os.getenv("HPFRACC_DISABLE_JAX", "0") == "1",
            BackendType.NUMBA: os.getenv("HPFRACC_DISABLE_NUMBA", "0") == "1",
        }

        # Priority order: TORCH (best ecosystem) -> JAX (best math) -> NUMBA (fallback)
        for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
            if not disabled.get(backend, False) and self._is_available(backend):
                return backend

        # Fallback to NUMBA (NumPy) if nothing else available
        return BackendType.NUMBA

    def _is_available(self, backend: BackendType) -> bool:
        """Check if backend is available without importing."""
        try:
            if backend == BackendType.TORCH:
                return importlib.util.find_spec("torch") is not None
            elif backend == BackendType.JAX:
                return (importlib.util.find_spec("jax") is not None and 
                       importlib.util.find_spec("jax.numpy") is not None)
            elif backend == BackendType.NUMBA:
                return importlib.util.find_spec("numpy") is not None
        except Exception:
            return False
        return False

    def _get_cached_lib(self) -> Any:
        """Get cached library or import and cache it."""
        if self.backend not in _LIB_CACHE:
            _LIB_CACHE[self.backend] = self._import_lib()
        return _LIB_CACHE[self.backend]

    def _import_lib(self) -> Any:
        """Import the appropriate library."""
        if self.backend == BackendType.TORCH:
            return _import_module("torch")
        elif self.backend == BackendType.JAX:
            return _import_module("jax.numpy")
        elif self.backend == BackendType.NUMBA:
            return _import_module("numpy")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_cached_capabilities(self) -> Capabilities:
        """Get cached capabilities or detect and cache them."""
        if self.backend not in _CAPABILITIES_CACHE:
            _CAPABILITIES_CACHE[self.backend] = self._detect_capabilities()
        return _CAPABILITIES_CACHE[self.backend]

    def _detect_capabilities(self) -> Capabilities:
        """Detect backend capabilities with performance characteristics."""
        lib = self._get_cached_lib()
        
        if self.backend == BackendType.TORCH:
            has_cuda = hasattr(lib, "cuda") and lib.cuda.is_available()
            return Capabilities(
                device_kind="gpu" if has_cuda else "cpu",
                has_fft=hasattr(lib, "fft"),
                has_autograd=hasattr(lib, "autograd"),
                supports_amp=True,
                supports_jit=hasattr(lib, "compile") or hasattr(lib, "jit"),
                memory_limit_gb=8.0 if has_cuda else 0.0,
                preferred_operations=frozenset(["neural_networks", "autograd", "gpu_ops"])
            )
        elif self.backend == BackendType.JAX:
            try:
                jax = importlib.import_module("jax")
                devices = jax.devices()
                has_gpu = any("gpu" in str(d).lower() for d in devices)
            except Exception:
                has_gpu = False
            return Capabilities(
                device_kind="gpu" if has_gpu else "cpu",
                has_fft=True,
                has_autograd=True,
                supports_amp=False,
                supports_jit=True,
                memory_limit_gb=16.0 if has_gpu else 0.0,
                preferred_operations=frozenset(["mathematical", "jit", "functional"])
            )
        else:  # NUMBA
            return Capabilities(
                device_kind="cpu",
                has_fft=True,
                has_autograd=False,
                supports_amp=False,
                supports_jit=False,
                memory_limit_gb=0.0,
                preferred_operations=frozenset(["arrays", "simple_ops"])
            )

    def _get_performance_profile(self) -> PerformanceProfile:
        """Get performance profile for this backend."""
        if self.backend not in _PERFORMANCE_PROFILES:
            _PERFORMANCE_PROFILES[self.backend] = self._create_performance_profile()
        return _PERFORMANCE_PROFILES[self.backend]

    def _create_performance_profile(self) -> PerformanceProfile:
        """Create performance profile based on backend characteristics."""
        if self.backend == BackendType.TORCH:
            return PerformanceProfile(
                small_data_threshold=1000,
                large_data_threshold=100000,
                gpu_memory_threshold=1000000,
                preferred_for_math=False,
                preferred_for_nn=True,
                preferred_for_arrays=False,
                jit_compilation_time=0.1
            )
        elif self.backend == BackendType.JAX:
            return PerformanceProfile(
                small_data_threshold=100,
                large_data_threshold=10000,
                gpu_memory_threshold=100000,
                preferred_for_math=True,
                preferred_for_nn=False,
                preferred_for_arrays=False,
                jit_compilation_time=0.5
            )
        else:  # NUMBA
            return PerformanceProfile(
                small_data_threshold=10000,
                large_data_threshold=1000000,
                gpu_memory_threshold=0,
                preferred_for_math=False,
                preferred_for_nn=False,
                preferred_for_arrays=True,
                jit_compilation_time=0.0
            )

    def get_lib(self) -> Any:
        """Get the underlying library (zero overhead)."""
        return self._lib

    def get_capabilities(self) -> Capabilities:
        """Get backend capabilities."""
        return self._capabilities
    
    @property
    def capabilities(self) -> Capabilities:
        """Get backend capabilities as property."""
        return self._capabilities

    def get_performance_profile(self) -> PerformanceProfile:
        """Get performance profile."""
        return self._performance_profile

    def is_optimal_for(self, operation_type: str, data_size: int) -> bool:
        """Check if this backend is optimal for the given operation and data size."""
        profile = self._performance_profile
        caps = self._capabilities

        # Check if operation type is preferred
        if operation_type in caps.preferred_operations:
            return True

        # Check data size thresholds
        if data_size < profile.small_data_threshold:
            return self.backend == BackendType.NUMBA  # NumPy is fastest for small data
        elif data_size > profile.large_data_threshold:
            return caps.device_kind == "gpu"  # GPU is best for large data
        else:
            return True  # Any backend is fine for medium data

    def optimize_operation(self, operation: Callable, *args, **kwargs) -> Callable:
        """Optimize operation for this backend."""
        if self.backend == BackendType.JAX and self._capabilities.supports_jit:
            # JAX JIT compilation
            jax = importlib.import_module("jax")
            return jax.jit(operation)
        elif self.backend == BackendType.TORCH and self._capabilities.supports_jit:
            # PyTorch compilation (if available)
            if hasattr(self._lib, "compile"):
                return self._lib.compile(operation)
        return operation

    def create_tensor(self, data: Any, **kwargs) -> Any:
        """Create tensor optimized for this backend."""
        if self.backend == BackendType.TORCH:
            # PyTorch tensor creation with optimal settings
            if 'device' not in kwargs and self._capabilities.device_kind == "gpu":
                kwargs['device'] = 'cuda'
            if 'dtype' not in kwargs:
                kwargs['dtype'] = self._lib.float32
            return self._lib.tensor(data, **kwargs)
        elif self.backend == BackendType.JAX:
            # JAX array creation
            if 'dtype' not in kwargs:
                kwargs['dtype'] = self._lib.float32
            return self._lib.array(data, **kwargs)
        else:  # NUMBA (NumPy)
            return self._lib.array(data, **kwargs)

    def benchmark_operation(self, operation: Callable, *args, **kwargs) -> float:
        """Benchmark operation and return execution time."""
        # Warmup
        for _ in range(3):
            operation(*args, **kwargs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            operation(*args, **kwargs)
        end_time = time.time()
        
        return (end_time - start_time) / 10


def get_optimal_adapter(operation_type: str = "general", data_size: int = 1000) -> HighPerformanceAdapter:
    """
    Get the optimal adapter for the given operation and data size.
    
    Args:
        operation_type: Type of operation ("mathematical", "neural_networks", "arrays")
        data_size: Size of data to process
    
    Returns:
        Optimal adapter for the task
    """
    # Check all available backends
    available_backends = []
    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        if not os.getenv(f"HPFRACC_DISABLE_{backend.value.upper()}", "0") == "1":
            try:
                adapter = HighPerformanceAdapter(backend)
                available_backends.append(adapter)
            except Exception:
                continue

    if not available_backends:
        # Fallback to NUMBA (NumPy)
        return HighPerformanceAdapter(BackendType.NUMBA)

    # Find optimal backend
    optimal_adapter = available_backends[0]
    best_score = 0

    for adapter in available_backends:
        score = 0
        
        # Score based on operation type preference
        if operation_type in adapter.get_capabilities().preferred_operations:
            score += 100
        
        # Score based on data size optimization
        if adapter.is_optimal_for(operation_type, data_size):
            score += 50
        
        # Score based on capabilities
        caps = adapter.get_capabilities()
        if caps.device_kind == "gpu" and data_size > 100000:
            score += 30
        if caps.supports_jit and operation_type == "mathematical":
            score += 20
        
        if score > best_score:
            best_score = score
            optimal_adapter = adapter

    return optimal_adapter


def benchmark_backends(operation: Callable, *args, **kwargs) -> Dict[BackendType, float]:
    """
    Benchmark operation across all available backends.
    
    Returns:
        Dictionary mapping backend to execution time
    """
    results = {}
    
    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        if not os.getenv(f"HPFRACC_DISABLE_{backend.value.upper()}", "0") == "1":
            try:
                adapter = HighPerformanceAdapter(backend)
                time_taken = adapter.benchmark_operation(operation, *args, **kwargs)
                results[backend] = time_taken
            except Exception:
                continue
    
    return results


# Convenience functions for direct access
def get_torch_adapter() -> HighPerformanceAdapter:
    """Get PyTorch adapter."""
    return HighPerformanceAdapter(BackendType.TORCH)


def get_jax_adapter() -> HighPerformanceAdapter:
    """Get JAX adapter."""
    return HighPerformanceAdapter(BackendType.JAX)


def get_numpy_adapter() -> HighPerformanceAdapter:
    """Get NumPy adapter."""
    return HighPerformanceAdapter(BackendType.NUMBA)


def _spec_available(name: str) -> bool:
    """Check if a module is available for import."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def get_adapter(backend: BackendType) -> HighPerformanceAdapter:
    """Get adapter for specific backend type."""
    # Check if backend is available before creating adapter
    if not _spec_available(_get_backend_module_name(backend)):
        raise ImportError(f"Backend {backend} is not available")
    
    if backend == BackendType.TORCH:
        return get_torch_adapter()
    elif backend == BackendType.JAX:
        return get_jax_adapter()
    elif backend == BackendType.NUMBA:
        return get_numpy_adapter()
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _get_backend_module_name(backend: BackendType) -> str:
    """Get the module name for a backend type."""
    if backend == BackendType.TORCH:
        return "torch"
    elif backend == BackendType.JAX:
        return "jax"
    elif backend == BackendType.NUMBA:
        return "numpy"  # NUMBA backend uses numpy
    else:
        return "unknown"


def detect_capabilities(backend: BackendType) -> Capabilities:
    """Detect capabilities for a specific backend."""
    try:
        adapter = get_adapter(backend)
        return adapter.get_capabilities()
    except ImportError:
        # Return fallback capabilities when backend is not available
        return Capabilities(
            device_kind="cpu",
            has_fft=False,
            has_autograd=False,
            supports_amp=False,
            supports_jit=False,
            memory_limit_gb=0.0,
            preferred_operations=frozenset()
        )
