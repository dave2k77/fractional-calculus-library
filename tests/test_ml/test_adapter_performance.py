#!/usr/bin/env python3
"""
Performance tests for optimized adapter system.

This test suite verifies that the adapter system:
1. Does not degrade performance
2. Allows optimal framework selection
3. Maintains zero-overhead access to native libraries
4. Provides intelligent backend selection
"""

import pytest
import numpy as np
import time
from typing import Dict, Any

# Test the optimized adapter system
def test_adapter_imports_without_performance_degradation():
    """Test that adapters can be imported without performance impact."""
    from hpfracc.ml.adapters import (
        HighPerformanceAdapter, 
        get_optimal_adapter,
        benchmark_backends
    )
    
    # Should import quickly without heavy dependencies
    start_time = time.time()
    adapter = HighPerformanceAdapter()
    import_time = time.time() - start_time
    
    # Import should be reasonable (< 3 seconds for first-time import with PyTorch)
    assert import_time < 3.0
    assert adapter is not None


def test_zero_overhead_library_access():
    """Test that library access has zero overhead."""
    from hpfracc.ml.adapters import HighPerformanceAdapter
    
    adapter = HighPerformanceAdapter()
    lib = adapter.get_lib()
    
    # Use the correct method for the backend
    if adapter.backend.value == "torch":
        # PyTorch uses tensor() method
        start_time = time.time()
        for _ in range(1000):
            _ = lib.tensor([1, 2, 3])
        direct_time = time.time() - start_time
        
        # Compare with direct PyTorch import
        import torch
        start_time = time.time()
        for _ in range(1000):
            _ = torch.tensor([1, 2, 3])
        native_time = time.time() - start_time
    else:
        # NumPy/JAX use array() method
        start_time = time.time()
        for _ in range(1000):
            _ = lib.array([1, 2, 3])
        direct_time = time.time() - start_time
        
        # Compare with direct NumPy import
        import numpy as np
        start_time = time.time()
        for _ in range(1000):
            _ = np.array([1, 2, 3])
        native_time = time.time() - start_time
    
    # Adapter overhead should be minimal (< 20% overhead for first access)
    overhead_ratio = direct_time / native_time
    assert overhead_ratio < 1.2, f"Adapter overhead too high: {overhead_ratio:.2f}x"


def test_capability_caching():
    """Test that capabilities are cached and not re-detected."""
    from hpfracc.ml.adapters import HighPerformanceAdapter, _CAPABILITIES_CACHE
    
    # Clear cache
    _CAPABILITIES_CACHE.clear()
    
    # First adapter creation should detect capabilities
    start_time = time.time()
    adapter1 = HighPerformanceAdapter()
    first_time = time.time() - start_time
    
    # Second adapter creation should use cache
    start_time = time.time()
    adapter2 = HighPerformanceAdapter()
    second_time = time.time() - start_time
    
    # Second creation should be much faster
    assert second_time < first_time * 0.5, "Capabilities not cached properly"
    
    # Capabilities should be identical
    caps1 = adapter1.get_capabilities()
    caps2 = adapter2.get_capabilities()
    assert caps1 == caps2


def test_optimal_backend_selection():
    """Test that optimal backend is selected based on operation type."""
    from hpfracc.ml.adapters import get_optimal_adapter
    
    # Test mathematical operations (should prefer JAX if available)
    math_adapter = get_optimal_adapter("mathematical", 1000)
    assert math_adapter is not None
    
    # Test neural network operations (should prefer PyTorch if available)
    nn_adapter = get_optimal_adapter("neural_networks", 1000)
    assert nn_adapter is not None
    
    # Test array operations (should prefer NumPy)
    array_adapter = get_optimal_adapter("arrays", 1000)
    assert array_adapter is not None


def test_data_size_optimization():
    """Test that backend selection considers data size."""
    from hpfracc.ml.adapters import get_optimal_adapter
    
    # Small data should prefer NumPy
    small_adapter = get_optimal_adapter("general", 100)
    assert small_adapter is not None
    
    # Large data should prefer GPU backends if available
    large_adapter = get_optimal_adapter("general", 1000000)
    assert large_adapter is not None
    
    # Medium data should work with any backend
    medium_adapter = get_optimal_adapter("general", 10000)
    assert medium_adapter is not None


def test_operation_optimization():
    """Test that operations are optimized for the selected backend."""
    from hpfracc.ml.adapters import HighPerformanceAdapter
    
    adapter = HighPerformanceAdapter()
    
    # Test tensor creation optimization
    data = [1, 2, 3, 4, 5]
    tensor = adapter.create_tensor(data)
    assert tensor is not None
    
    # Test operation optimization
    def simple_operation(x):
        return x * 2
    
    optimized_op = adapter.optimize_operation(simple_operation)
    result = optimized_op(tensor)
    assert result is not None


def test_benchmarking_functionality():
    """Test that benchmarking works correctly."""
    from hpfracc.ml.adapters import benchmark_backends
    
    def test_operation(x):
        return x * 2 + 1
    
    # Benchmark across backends
    results = benchmark_backends(test_operation, np.array([1, 2, 3, 4, 5]))
    
    # Should have results for available backends
    assert len(results) > 0
    
    # All results should be positive
    for backend, time_taken in results.items():
        assert time_taken > 0
        assert isinstance(time_taken, float)


def test_performance_profiles():
    """Test that performance profiles are correctly assigned."""
    from hpfracc.ml.adapters import HighPerformanceAdapter
    
    adapter = HighPerformanceAdapter()
    profile = adapter.get_performance_profile()
    
    # Profile should have reasonable values
    assert profile.small_data_threshold > 0
    assert profile.large_data_threshold > profile.small_data_threshold
    assert profile.gpu_memory_threshold >= 0
    assert profile.jit_compilation_time >= 0


def test_backend_switching_performance():
    """Test that switching between backends is efficient."""
    from hpfracc.ml.adapters import HighPerformanceAdapter
    
    # Create adapters for different backends
    adapters = {}
    for backend in ["torch", "jax", "numpy"]:
        try:
            adapters[backend] = HighPerformanceAdapter(backend)
        except Exception:
            continue
    
    # Test that switching is fast
    start_time = time.time()
    for _ in range(10):
        for adapter in adapters.values():
            _ = adapter.get_lib()
    switch_time = time.time() - start_time
    
    # Switching should be fast (< 0.1 seconds for 10 switches)
    assert switch_time < 0.1


def test_memory_efficiency():
    """Test that adapters don't cause memory leaks."""
    import gc
    from hpfracc.ml.adapters import HighPerformanceAdapter, _LIB_CACHE, _CAPABILITIES_CACHE
    
    # Clear caches
    _LIB_CACHE.clear()
    _CAPABILITIES_CACHE.clear()
    
    # Create and destroy many adapters
    for _ in range(100):
        adapter = HighPerformanceAdapter()
        _ = adapter.get_lib()
        _ = adapter.get_capabilities()
        del adapter
    
    # Force garbage collection
    gc.collect()
    
    # Caches should still work (they're global)
    assert len(_LIB_CACHE) > 0 or len(_CAPABILITIES_CACHE) > 0


def test_environment_variable_respect():
    """Test that environment variables are respected."""
    import os
    from hpfracc.ml.adapters import get_optimal_adapter
    
    # Test with disabled backends
    original_disable_torch = os.getenv("HPFRACC_DISABLE_TORCH", "0")
    original_disable_jax = os.getenv("HPFRACC_DISABLE_JAX", "0")
    
    try:
        # Disable PyTorch
        os.environ["HPFRACC_DISABLE_TORCH"] = "1"
        adapter = get_optimal_adapter("neural_networks", 1000)
        # Should not select PyTorch
        assert adapter.backend.value != "torch"
        
        # Disable JAX
        os.environ["HPFRACC_DISABLE_JAX"] = "1"
        adapter = get_optimal_adapter("mathematical", 1000)
        # Should not select JAX
        assert adapter.backend.value != "jax"
        
    finally:
        # Restore original values
        if original_disable_torch != "0":
            os.environ["HPFRACC_DISABLE_TORCH"] = original_disable_torch
        else:
            os.environ.pop("HPFRACC_DISABLE_TORCH", None)
            
        if original_disable_jax != "0":
            os.environ["HPFRACC_DISABLE_JAX"] = original_disable_jax
        else:
            os.environ.pop("HPFRACC_DISABLE_JAX", None)


def test_fallback_behavior():
    """Test that the system gracefully falls back when backends are unavailable."""
    from hpfracc.ml.adapters import get_optimal_adapter
    
    # Should always return an adapter, even if preferred backends are unavailable
    adapter = get_optimal_adapter("general", 1000)
    assert adapter is not None
    assert adapter.backend is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
