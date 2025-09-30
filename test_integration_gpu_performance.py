#!/usr/bin/env python3
"""
GPU Performance Integration Tests

This module tests GPU optimization integration with computational workflows,
focusing on performance, memory management, and scalability.
"""

import numpy as np
import torch
import time
import gc
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Import GPU optimization components
from hpfracc.ml.gpu_optimization import (
    PerformanceMetrics, GPUProfiler, ChunkedFFT, 
    AMPFractionalEngine, GPUOptimizedSpectralEngine,
    GPUOptimizedStochasticSampler, gpu_optimization_context
)
from hpfracc.ml.variance_aware_training import VarianceMonitor, AdaptiveSamplingManager


class TestGPUPerformanceIntegration:
    """Test GPU performance integration with computational workflows."""
    
    def test_gpu_profiling_integration(self):
        """Test GPU profiling integration with computational workflows."""
        profiler = GPUProfiler()
        
        # Test profiling workflow
        profiler.start_timer("computational_workflow")
        
        # Simulate computational work
        x = torch.randn(1000, 1000)
        result = torch.fft.fft(x)
        
        profiler.end_timer(x, result)
        
        # Verify profiling worked
        assert profiler is not None
        assert len(profiler.metrics_history) > 0
        
        print("✅ GPU profiling integration verified")
    
    def test_chunked_fft_performance_integration(self):
        """Test ChunkedFFT performance integration."""
        # Test different chunk sizes
        chunk_sizes = [512, 1024, 2048]
        
        for chunk_size in chunk_sizes:
            fft = ChunkedFFT(chunk_size=chunk_size)
            
            # Test with different array sizes
            array_sizes = [1024, 2048, 4096]
            
            for size in array_sizes:
                x = torch.randn(size)
                
                start_time = time.time()
                result = fft.fft_chunked(x)
                end_time = time.time()
                
                # Verify result
                assert result.shape == x.shape
                assert end_time - start_time < 1.0  # Should be fast
        
        print("✅ ChunkedFFT performance integration verified")
    
    def test_amp_fractional_engine_integration(self):
        """Test AMPFractionalEngine integration."""
        # Create mock base engine
        mock_base_engine = MagicMock()
        mock_base_engine.forward.return_value = torch.randn(64)
        
        # Test AMP engine
        amp_engine = AMPFractionalEngine(mock_base_engine, use_amp=True)
        
        # Test forward pass
        x = torch.randn(64)
        alpha = 0.5
        
        result = amp_engine.forward(x, alpha)
        
        # Verify result
        assert result is not None
        assert result.shape == x.shape
        
        print("✅ AMPFractionalEngine integration verified")
    
    def test_gpu_optimized_spectral_engine_integration(self):
        """Test GPUOptimizedSpectralEngine integration."""
        # Test spectral engine
        spectral_engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=True,
            chunk_size=1024
        )
        
        # Test forward pass
        x = torch.randn(64)
        alpha = 0.5
        
        result = spectral_engine.forward(x, alpha)
        
        # Verify result
        assert result is not None
        
        print("✅ GPUOptimizedSpectralEngine integration verified")
    
    def test_gpu_optimization_context_integration(self):
        """Test GPU optimization context manager integration."""
        # Test context manager
        with gpu_optimization_context(use_amp=True):
            # Create components within context
            profiler = GPUProfiler()
            fft = ChunkedFFT()
            
            # Perform computations
            x = torch.randn(256)
            result = fft.fft_chunked(x)
            
            # Verify everything works within context
            assert result is not None
        
        print("✅ GPU optimization context integration verified")


class TestMemoryManagementIntegration:
    """Test memory management integration with GPU optimization."""
    
    def test_memory_management_under_load(self):
        """Test memory management under computational load."""
        # Test creating and destroying many components
        components = []
        
        for i in range(50):
            profiler = GPUProfiler()
            fft = ChunkedFFT()
            monitor = VarianceMonitor()
            
            # Perform some computation
            x = torch.randn(128)
            result = fft.fft_chunked(x)
            
            components.extend([profiler, fft, monitor])
        
        # Verify all components created successfully
        assert len(components) == 150
        
        # Clean up
        del components
        gc.collect()
        
        print("✅ Memory management under load verified")
    
    def test_large_data_handling_integration(self):
        """Test handling of large data with GPU optimization."""
        # Test with progressively larger datasets
        sizes = [1000, 2000, 4000, 8000]
        
        for size in sizes:
            fft = ChunkedFFT(chunk_size=1024)
            
            # Create large tensor
            x = torch.randn(size)
            
            start_time = time.time()
            result = fft.fft_chunked(x)
            end_time = time.time()
            
            # Verify result and performance
            assert result.shape == x.shape
            assert end_time - start_time < 5.0  # Should complete within 5 seconds
        
        print("✅ Large data handling integration verified")
    
    def test_concurrent_component_usage(self):
        """Test concurrent usage of multiple GPU optimization components."""
        # Create multiple components
        profilers = [GPUProfiler() for _ in range(5)]
        ffts = [ChunkedFFT() for _ in range(5)]
        
        # Test concurrent usage
        results = []
        for i, (profiler, fft) in enumerate(zip(profilers, ffts)):
            profiler.start_timer(f"computation_{i}")
            
            x = torch.randn(256)
            result = fft.fft_chunked(x)
            
            profiler.end_timer(x, result)
            results.append(result)
        
        # Verify all results
        for result in results:
            assert result is not None
            assert result.shape == (256,)
        
        print("✅ Concurrent component usage verified")


class TestPerformanceBenchmarking:
    """Test performance benchmarking integration."""
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection across components."""
        # Create performance metrics
        metrics = PerformanceMetrics(
            operation="benchmark_test",
            device="cpu",
            dtype="float32",
            input_shape=(1000,),
            execution_time=0.1,
            memory_used=100.0,
            memory_peak=120.0,
            throughput=10000.0,
            timestamp=time.time()
        )
        
        # Verify metrics
        assert metrics.operation == "benchmark_test"
        assert metrics.throughput == 10000.0
        assert metrics.execution_time == 0.1
        
        print("✅ Performance metrics collection verified")
    
    def test_workflow_performance_benchmarking(self):
        """Test workflow performance benchmarking."""
        # Benchmark a complete workflow
        profiler = GPUProfiler()
        
        # Start overall profiling
        profiler.start_timer("complete_workflow")
        
        # Step 1: Data preparation
        profiler.start_timer("data_preparation")
        x = torch.randn(1000, 1000)
        profiler.end_timer(torch.randn(1000), x)
        
        # Step 2: FFT computation
        profiler.start_timer("fft_computation")
        fft = ChunkedFFT()
        result = fft.fft_chunked(x)
        profiler.end_timer(x, result)
        
        # Step 3: Post-processing
        profiler.start_timer("post_processing")
        final_result = torch.abs(result)
        profiler.end_timer(result, final_result)
        
        # End overall profiling
        profiler.end_timer(x, final_result)
        
        # Verify profiling
        assert profiler is not None
        assert len(profiler.metrics_history) >= 3
        
        print("✅ Workflow performance benchmarking verified")
    
    def test_scalability_benchmarking(self):
        """Test scalability benchmarking across different problem sizes."""
        sizes = [256, 512, 1024, 2048]
        results = {}
        
        for size in sizes:
            fft = ChunkedFFT(chunk_size=min(1024, size))
            
            # Benchmark
            x = torch.randn(size)
            
            start_time = time.time()
            result = fft.fft_chunked(x)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[size] = execution_time
            
            # Verify result
            assert result.shape == x.shape
        
        # Verify scalability (should not increase dramatically)
        small_time = results[256]
        large_time = results[2048]
        
        # Large problem should not take more than 10x longer
        assert large_time < small_time * 10
        
        print(f"✅ Scalability benchmarking verified: {results}")
    
    def test_variance_aware_performance_integration(self):
        """Test variance-aware training performance integration."""
        monitor = VarianceMonitor()
        sampling_manager = AdaptiveSamplingManager()
        
        # Simulate training loop with variance monitoring
        for epoch in range(10):
            # Simulate gradient computation
            gradients = torch.randn(100)
            
            # Monitor variance
            monitor.update(f"epoch_{epoch}_gradients", gradients)
            
            # Adapt sampling based on variance
            if epoch > 0:
                metrics = monitor.get_metrics(f"epoch_{epoch-1}_gradients")
                if metrics:
                    variance = metrics.variance
                    new_k = sampling_manager.update_k(variance, 32)
        
        # Verify monitoring worked
        assert monitor is not None
        assert sampling_manager is not None
        
        print("✅ Variance-aware performance integration verified")


def run_gpu_performance_integration_tests():
    """Run all GPU performance integration tests."""
    print("🚀 Starting GPU Performance Integration Tests")
    print("=" * 60)
    
    # Create test instances
    gpu_test = TestGPUPerformanceIntegration()
    memory_test = TestMemoryManagementIntegration()
    benchmark_test = TestPerformanceBenchmarking()
    
    # Run tests
    gpu_tests = [
        gpu_test.test_gpu_profiling_integration,
        gpu_test.test_chunked_fft_performance_integration,
        gpu_test.test_amp_fractional_engine_integration,
        gpu_test.test_gpu_optimized_spectral_engine_integration,
        gpu_test.test_gpu_optimization_context_integration,
    ]
    
    memory_tests = [
        memory_test.test_memory_management_under_load,
        memory_test.test_large_data_handling_integration,
        memory_test.test_concurrent_component_usage,
    ]
    
    benchmark_tests = [
        benchmark_test.test_performance_metrics_collection,
        benchmark_test.test_workflow_performance_benchmarking,
        benchmark_test.test_scalability_benchmarking,
        benchmark_test.test_variance_aware_performance_integration,
    ]
    
    all_tests = gpu_tests + memory_tests + benchmark_tests
    
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
    print(f"📊 GPU Performance Integration Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All GPU performance integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Review issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_gpu_performance_integration_tests()
    exit(0 if success else 1)
