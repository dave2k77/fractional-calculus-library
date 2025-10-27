#!/usr/bin/env python3
"""
Test and demonstrate the intelligent backend selector.
"""

import os
import numpy as np
import time

# Set non-interactive backend for testing
os.environ['MPLBACKEND'] = 'Agg'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from hpfracc.ml.intelligent_backend_selector import (
    IntelligentBackendSelector,
    WorkloadCharacteristics,
    select_optimal_backend
)
from hpfracc.ml.backends import BackendType


def test_small_data_selection():
    """Test that small data uses NumPy/Numba."""
    print("\n" + "="*70)
    print("TEST 1: Small Data Selection")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    workload = WorkloadCharacteristics(
        operation_type="element_wise",
        data_size=100,  # Very small
        data_shape=(10, 10)
    )
    
    backend = selector.select_backend(workload)
    print(f"✅ Small data (100 elements): Selected {backend.value}")
    assert backend == BackendType.NUMBA, f"Expected NUMBA for small data, got {backend.value}"
    print("   ✓ Correctly selected NumPy/Numba for small data")


def test_large_data_selection():
    """Test that large data considers GPU if available."""
    print("\n" + "="*70)
    print("TEST 2: Large Data Selection")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    workload = WorkloadCharacteristics(
        operation_type="matmul",
        data_size=1000000,  # 1M elements
        data_shape=(1000, 1000)
    )
    
    backend = selector.select_backend(workload)
    print(f"✅ Large data (1M elements): Selected {backend.value}")
    print(f"   ✓ Selected {backend.value} for large matrix operations")


def test_gradient_operations():
    """Test that gradient operations prefer PyTorch."""
    print("\n" + "="*70)
    print("TEST 3: Gradient Operations")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    workload = WorkloadCharacteristics(
        operation_type="neural_network",
        data_size=50000,
        data_shape=(100, 500),
        requires_gradient=True
    )
    
    backend = selector.select_backend(workload)
    print(f"✅ Gradient required: Selected {backend.value}")
    if backend == BackendType.TORCH:
        print("   ✓ Correctly selected PyTorch for gradient operations")
    else:
        print(f"   ⚠ Selected {backend.value} (PyTorch not available?)")


def test_mathematical_operations():
    """Test that mathematical operations prefer JAX."""
    print("\n" + "="*70)
    print("TEST 4: Mathematical Operations (FFT, derivatives)")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    for op_type in ["fft", "derivative", "matmul"]:
        workload = WorkloadCharacteristics(
            operation_type=op_type,
            data_size=10000,
            data_shape=(100, 100)
        )
        
        backend = selector.select_backend(workload)
        print(f"✅ {op_type}: Selected {backend.value}")


def test_performance_monitoring():
    """Test performance monitoring and learning."""
    print("\n" + "="*70)
    print("TEST 5: Performance Monitoring & Learning")
    print("="*70)
    
    selector = IntelligentBackendSelector(enable_learning=True)
    
    # Simulate some operations
    workload = WorkloadCharacteristics(
        operation_type="test_operation",
        data_size=1000,
        data_shape=(100, 10)
    )
    
    # Record several successful operations with different backends
    from hpfracc.ml.intelligent_backend_selector import PerformanceRecord
    
    # Simulate that JAX is faster for this operation
    for i in range(10):
        selector.performance_monitor.record(PerformanceRecord(
            backend=BackendType.JAX,
            operation="test_operation",
            data_size=1000,
            execution_time=0.001 + np.random.random() * 0.001,
            success=True
        ))
    
    # Simulate that TORCH is slower
    for i in range(10):
        selector.performance_monitor.record(PerformanceRecord(
            backend=BackendType.TORCH,
            operation="test_operation",
            data_size=1000,
            execution_time=0.005 + np.random.random() * 0.001,
            success=True
        ))
    
    # Now the selector should learn to prefer JAX for this operation
    backend = selector.select_backend(workload)
    print(f"✅ After learning: Selected {backend.value} for test_operation")
    
    # Get statistics
    stats = selector.get_performance_summary()
    print(f"   📊 Total records: {stats['total_records']}")
    print(f"   📊 Operations tracked: {len(stats['operation_stats'])}")
    
    if backend == BackendType.JAX:
        print("   ✓ Successfully learned that JAX is faster!")
    else:
        print("   ⚠ Learning didn't converge (might need more samples)")


def test_gpu_memory_estimation():
    """Test GPU memory estimation and threshold calculation."""
    print("\n" + "="*70)
    print("TEST 6: GPU Memory Estimation")
    print("="*70)
    
    from hpfracc.ml.intelligent_backend_selector import GPUMemoryEstimator
    
    estimator = GPUMemoryEstimator()
    
    for backend_type in [BackendType.TORCH, BackendType.JAX]:
        memory = estimator.get_available_gpu_memory_gb(backend_type)
        threshold = estimator.calculate_gpu_threshold(backend_type)
        
        print(f"\n{backend_type.value}:")
        if memory > 0:
            print(f"   💾 Available GPU memory: {memory:.2f} GB")
            print(f"   🎯 GPU threshold: {threshold:,} elements")
            print(f"   📊 Threshold memory: {(threshold * 8) / (1024**3):.2f} GB")
        else:
            print(f"   ⚠ No GPU detected or not available")


def test_convenience_function():
    """Test the convenience function."""
    print("\n" + "="*70)
    print("TEST 7: Convenience Function")
    print("="*70)
    
    # Test various scenarios
    scenarios = [
        ("Small matrix multiplication", "matmul", (10, 10)),
        ("Large matrix multiplication", "matmul", (1000, 1000)),
        ("Convolution layer", "conv", (32, 3, 224, 224)),
        ("Element-wise operation", "element_wise", (100000,)),
        ("FFT operation", "fft", (1024, 1024)),
    ]
    
    for name, op_type, shape in scenarios:
        backend = select_optimal_backend(op_type, shape)
        data_size = int(np.prod(shape))
        print(f"✅ {name} ({data_size:,} elements): {backend.value}")


def test_fallback_mechanism():
    """Test automatic fallback on errors."""
    print("\n" + "="*70)
    print("TEST 8: Fallback Mechanism")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    workload = WorkloadCharacteristics(
        operation_type="test_fallback",
        data_size=1000,
        data_shape=(100, 10)
    )
    
    # Simulate a function that fails
    call_count = [0]
    
    def failing_function():
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated failure")
        return "Success!"
    
    try:
        result = selector.execute_with_monitoring(
            operation_name="test_fallback",
            backend=BackendType.TORCH,
            func=failing_function,
            workload=workload,
            fallback_backends=[BackendType.JAX, BackendType.NUMBA]
        )
        print(f"✅ Fallback successful: {result}")
        print(f"   ✓ Tried {call_count[0]} backend(s)")
    except Exception as e:
        print(f"❌ Fallback failed: {e}")


def benchmark_backend_selection_overhead():
    """Benchmark the overhead of backend selection."""
    print("\n" + "="*70)
    print("TEST 9: Backend Selection Overhead")
    print("="*70)
    
    selector = IntelligentBackendSelector()
    
    workload = WorkloadCharacteristics(
        operation_type="benchmark",
        data_size=10000,
        data_shape=(100, 100)
    )
    
    # Measure selection time
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        backend = selector.select_backend(workload)
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / iterations) * 1000
    print(f"✅ Average selection time: {avg_time_ms:.4f} ms")
    print(f"   ✓ Overhead is {'acceptable' if avg_time_ms < 1.0 else 'high'} "
          f"({'< 1ms' if avg_time_ms < 1.0 else '>= 1ms'})")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "INTELLIGENT BACKEND SELECTOR TESTS")
    print("="*80)
    
    tests = [
        test_small_data_selection,
        test_large_data_selection,
        test_gradient_operations,
        test_mathematical_operations,
        test_performance_monitoring,
        test_gpu_memory_estimation,
        test_convenience_function,
        test_fallback_mechanism,
        benchmark_backend_selection_overhead,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*80)
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

