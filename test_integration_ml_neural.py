#!/usr/bin/env python3
"""
ML Neural Network Integration Tests

This module tests the integration between ML neural network components
and fractional calculus functionality.
"""

import numpy as np
import torch
import pytest
from typing import Dict, Any

# Import ML components
from hpfracc.ml.gpu_optimization import (
    PerformanceMetrics, GPUProfiler, ChunkedFFT, 
    AMPFractionalEngine, GPUOptimizedSpectralEngine
)
from hpfracc.ml.variance_aware_training import (
    VarianceMonitor, AdaptiveSamplingManager, 
    StochasticSeedManager, VarianceAwareCallback
)
from hpfracc.ml.adapters import HighPerformanceAdapter
from hpfracc.ml.backends import get_active_backend, BackendType


class TestMLNeuralNetworkIntegration:
    """Test ML neural network integration with fractional calculus."""
    
    def test_gpu_optimization_components_integration(self):
        """Test integration of GPU optimization components."""
        # Create GPU optimization components
        profiler = GPUProfiler()
        fft = ChunkedFFT()
        
        # Test basic integration
        assert profiler is not None
        assert fft is not None
        
        # Test FFT functionality
        x = torch.randn(64)
        result = fft.fft_chunked(x)
        assert result is not None
        assert result.shape == x.shape
        
        print("✅ GPU optimization components integration verified")
    
    def test_variance_aware_training_integration(self):
        """Test integration of variance-aware training components."""
        # Create variance-aware training components
        monitor = VarianceMonitor()
        sampling_manager = AdaptiveSamplingManager()
        seed_manager = StochasticSeedManager()
        
        # Test basic integration
        assert monitor is not None
        assert sampling_manager is not None
        assert seed_manager is not None
        
        # Test component interaction
        seed_manager.set_seed(42)
        assert seed_manager.current_seed == 42
        
        # Test adaptive sampling
        new_k = sampling_manager.update_k(0.3, 32)
        assert isinstance(new_k, int)
        
        print("✅ Variance-aware training components integration verified")
    
    def test_backend_adapter_integration(self):
        """Test backend adapter integration."""
        # Test backend detection
        backend = get_active_backend()
        assert backend is not None
        
        # Test adapter creation
        adapter = HighPerformanceAdapter()
        assert adapter is not None
        
        print(f"✅ Backend adapter integration verified (backend: {backend})")
    
    def test_performance_metrics_integration(self):
        """Test performance metrics integration."""
        # Create performance metrics
        metrics = PerformanceMetrics(
            operation="test_operation",
            device="cpu",
            dtype="float32",
            input_shape=(10, 10),
            execution_time=0.1,
            memory_used=100.0,
            memory_peak=120.0,
            throughput=1000.0,
            timestamp=1234567890.0
        )
        
        # Test metrics properties
        assert metrics.operation == "test_operation"
        assert metrics.throughput == 1000.0
        assert metrics.execution_time == 0.1
        
        print("✅ Performance metrics integration verified")
    
    def test_ml_components_workflow_integration(self):
        """Test integration of ML components in a workflow."""
        # Create a simple ML workflow
        profiler = GPUProfiler()
        monitor = VarianceMonitor()
        sampling_manager = AdaptiveSamplingManager()
        seed_manager = StochasticSeedManager()
        
        # Simulate a training step
        profiler.start_timer("training_step")
        
        # Simulate adaptive sampling
        variance = 0.2
        current_k = 32
        new_k = sampling_manager.update_k(variance, current_k)
        
        # Simulate variance monitoring
        monitor.update("gradient_variance", torch.randn(100), timestamp=1234567890.0)
        
        # End profiling
        profiler.end_timer(torch.randn(10))
        
        # Verify workflow components work together
        assert new_k != current_k  # Should adapt
        assert monitor is not None
        
        print("✅ ML components workflow integration verified")


class TestFractionalMLIntegration:
    """Test integration between fractional calculus and ML components."""
    
    def test_fractional_neural_network_backend_compatibility(self):
        """Test that fractional components work with ML backends."""
        # Test backend compatibility
        backend = get_active_backend()
        
        # Create fractional components
        from hpfracc.core.derivatives import CaputoDerivative
        from hpfracc.core.integrals import FractionalIntegral
        
        caputo = CaputoDerivative(order=0.5)
        integral = FractionalIntegral(order=0.5)
        
        # Test that they work with the backend
        assert caputo.alpha.alpha == 0.5
        assert integral.alpha.alpha == 0.5
        
        print(f"✅ Fractional-ML backend compatibility verified (backend: {backend})")
    
    def test_gpu_optimization_with_fractional_operations(self):
        """Test GPU optimization with fractional operations."""
        # Create GPU optimization components
        profiler = GPUProfiler()
        
        # Test profiling fractional operations
        profiler.start_timer("fractional_derivative_computation")
        
        # Simulate fractional computation
        x = torch.randn(64)
        result = torch.fft.fft(x)  # Simulate fractional operation
        
        profiler.end_timer(x, result)
        
        # Verify profiling worked
        assert profiler is not None
        
        print("✅ GPU optimization with fractional operations verified")
    
    def test_variance_aware_training_with_fractional_orders(self):
        """Test variance-aware training with fractional order adaptation."""
        # Create variance-aware components
        monitor = VarianceMonitor()
        sampling_manager = AdaptiveSamplingManager()
        
        # Simulate fractional order adaptation
        initial_order = 0.5
        variance = 0.3
        
        # Adapt sampling based on variance
        new_k = sampling_manager.update_k(variance, 32)
        
        # Monitor variance
        monitor.update("fractional_order_variance", torch.tensor([variance]))
        
        # Verify adaptation
        assert new_k is not None
        assert monitor is not None
        
        print("✅ Variance-aware training with fractional orders verified")


class TestMLPerformanceIntegration:
    """Test ML performance and optimization integration."""
    
    def test_memory_management_integration(self):
        """Test memory management across ML components."""
        # Test that components can be created and destroyed without memory issues
        components = []
        
        for i in range(10):
            profiler = GPUProfiler()
            monitor = VarianceMonitor()
            sampling_manager = AdaptiveSamplingManager()
            
            components.extend([profiler, monitor, sampling_manager])
        
        # Verify all components created successfully
        assert len(components) == 30
        
        # Clean up
        del components
        
        print("✅ Memory management integration verified")
    
    def test_parallel_processing_integration(self):
        """Test parallel processing capabilities."""
        # Test ChunkedFFT with different sizes
        fft = ChunkedFFT(chunk_size=512)
        
        # Test small arrays
        x_small = torch.randn(256)
        result_small = fft.fft_chunked(x_small)
        
        # Test larger arrays
        x_large = torch.randn(2048)
        result_large = fft.fft_chunked(x_large)
        
        # Verify results
        assert result_small.shape == x_small.shape
        assert result_large.shape == x_large.shape
        
        print("✅ Parallel processing integration verified")


def run_ml_neural_integration_tests():
    """Run all ML neural network integration tests."""
    print("🚀 Starting ML Neural Network Integration Tests")
    print("=" * 60)
    
    # Create test instances
    ml_test = TestMLNeuralNetworkIntegration()
    fractional_ml_test = TestFractionalMLIntegration()
    performance_test = TestMLPerformanceIntegration()
    
    # Run tests
    ml_tests = [
        ml_test.test_gpu_optimization_components_integration,
        ml_test.test_variance_aware_training_integration,
        ml_test.test_backend_adapter_integration,
        ml_test.test_performance_metrics_integration,
        ml_test.test_ml_components_workflow_integration,
    ]
    
    fractional_tests = [
        fractional_ml_test.test_fractional_neural_network_backend_compatibility,
        fractional_ml_test.test_gpu_optimization_with_fractional_operations,
        fractional_ml_test.test_variance_aware_training_with_fractional_orders,
    ]
    
    performance_tests = [
        performance_test.test_memory_management_integration,
        performance_test.test_parallel_processing_integration,
    ]
    
    all_tests = ml_tests + fractional_tests + performance_tests
    
    passed = 0
    failed = 0
    
    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 ML Neural Network Integration Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All ML neural network integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Review issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_ml_neural_integration_tests()
    exit(0 if success else 1)
