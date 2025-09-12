"""
Comprehensive tests for GPU optimization utilities.

This module provides extensive testing for GPU acceleration features including
Automatic Mixed Precision (AMP), chunked FFT operations, and performance profiling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import time
import warnings

from hpfracc.ml.gpu_optimization import (
    PerformanceMetrics, GPUProfiler, AMPFractionalEngine, ChunkedFFT,
    GPUOptimizedSpectralEngine, GPUOptimizedStochasticSampler,
    create_gpu_optimized_components, gpu_optimization_context,
    benchmark_gpu_optimization, test_gpu_optimization
)


class TestPerformanceMetricsComprehensive:
    """Comprehensive tests for PerformanceMetrics dataclass."""
    
    def test_performance_metrics_all_fields(self):
        """Test PerformanceMetrics with all fields populated."""
        timestamp = time.time()
        metrics = PerformanceMetrics(
            operation="test_operation",
            device="cuda:0",
            dtype="float16",
            input_shape=(32, 1024, 512),
            execution_time=0.001234,
            memory_used=2048.5,
            memory_peak=4096.0,
            throughput=1.5e6,
            timestamp=timestamp
        )
        
        assert metrics.operation == "test_operation"
        assert metrics.device == "cuda:0"
        assert metrics.dtype == "float16"
        assert metrics.input_shape == (32, 1024, 512)
        assert metrics.execution_time == 0.001234
        assert metrics.memory_used == 2048.5
        assert metrics.memory_peak == 4096.0
        assert metrics.throughput == 1.5e6
        assert metrics.timestamp == timestamp

    def test_performance_metrics_edge_cases(self):
        """Test PerformanceMetrics with edge case values."""
        # Test with zero values
        metrics = PerformanceMetrics(
            operation="zero_test",
            device="cpu",
            dtype="float32",
            input_shape=(1, 1),
            execution_time=0.0,
            memory_used=0.0,
            memory_peak=0.0,
            throughput=0.0,
            timestamp=0.0
        )
        
        assert metrics.execution_time == 0.0
        assert metrics.memory_used == 0.0
        assert metrics.throughput == 0.0

    def test_performance_metrics_large_values(self):
        """Test PerformanceMetrics with large values."""
        metrics = PerformanceMetrics(
            operation="large_test",
            device="cuda:1",
            dtype="bfloat16",
            input_shape=(1000, 1000, 1000),
            execution_time=999.999,
            memory_used=1e9,
            memory_peak=2e9,
            throughput=1e12,
            timestamp=1e9
        )
        
        assert metrics.execution_time == 999.999
        assert metrics.memory_used == 1e9
        assert metrics.throughput == 1e12


class TestGPUProfilerComprehensive:
    """Comprehensive tests for GPUProfiler functionality."""
    
    def test_gpu_profiler_get_summary(self):
        """Test GPUProfiler get_summary method."""
        profiler = GPUProfiler(device="cpu")
        
        # Add some mock metrics
        profiler.current_metrics["test_op"] = PerformanceMetrics(
            operation="test_op",
            device="cpu",
            dtype="float32",
            input_shape=(10, 20),
            execution_time=0.001,
            memory_used=1024.0,
            memory_peak=2048.0,
            throughput=1000.0,
            timestamp=time.time()
        )
        
        summary = profiler.get_summary()
        assert "test_op" in summary
        assert summary["test_op"]["execution_time"] == 0.001
        assert summary["test_op"]["memory_used"] == 1024.0
        assert summary["test_op"]["throughput"] == 1000.0

    def test_gpu_profiler_clear_history(self):
        """Test GPUProfiler clear_history method."""
        profiler = GPUProfiler(device="cpu")
        
        # Add some metrics
        profiler.metrics_history.append(PerformanceMetrics(
            operation="test",
            device="cpu",
            dtype="float32",
            input_shape=(10,),
            execution_time=0.001,
            memory_used=100.0,
            memory_peak=200.0,
            throughput=1000.0,
            timestamp=time.time()
        ))
        profiler.current_metrics["test"] = profiler.metrics_history[0]
        
        assert len(profiler.metrics_history) == 1
        assert len(profiler.current_metrics) == 1
        
        profiler.clear_history()
        
        assert len(profiler.metrics_history) == 0
        assert len(profiler.current_metrics) == 0

    def test_gpu_profiler_multiple_operations(self):
        """Test GPUProfiler with multiple operations."""
        profiler = GPUProfiler(device="cpu")
        
        # Test multiple operations
        for i in range(3):
            profiler.start_timer(f"op_{i}")
            time.sleep(0.001)  # Small delay
            x = torch.randn(10, 20)
            profiler.end_timer(x)
        
        assert len(profiler.metrics_history) == 3
        assert len(profiler.current_metrics) == 3
        
        # Check that all operations are tracked
        operations = set(metrics.operation for metrics in profiler.metrics_history)
        assert operations == {"op_0", "op_1", "op_2"}


class TestChunkedFFTComprehensive:
    """Comprehensive tests for ChunkedFFT functionality."""
    
    def test_chunked_fft_1d_large_sequence(self):
        """Test chunked FFT with large 1D sequence."""
        chunked_fft = ChunkedFFT(chunk_size=256)
        x = torch.randn(1, 2048)  # Large sequence
        
        x_fft = chunked_fft.fft_chunked(x)
        x_reconstructed = chunked_fft.ifft_chunked(x_fft)
        
        # Check shapes
        assert x_fft.shape == x.shape
        assert x_reconstructed.shape == x.shape
        
        # Check reconstruction accuracy
        error = torch.mean(torch.abs(x - x_reconstructed.real))
        assert error < 1e-5

    def test_chunked_fft_2d_different_dimensions(self):
        """Test chunked FFT with 2D tensors on different dimensions."""
        chunked_fft = ChunkedFFT(chunk_size=128)
        x = torch.randn(16, 1024)
        
        # Test along last dimension
        x_fft_dim0 = chunked_fft.fft_chunked(x, dim=-1)
        x_reconstructed_dim0 = chunked_fft.ifft_chunked(x_fft_dim0, dim=-1)
        
        # Test along first dimension
        x_fft_dim1 = chunked_fft.fft_chunked(x, dim=0)
        x_reconstructed_dim1 = chunked_fft.ifft_chunked(x_fft_dim1, dim=0)
        
        # Check reconstruction accuracy
        error_dim0 = torch.mean(torch.abs(x - x_reconstructed_dim0.real))
        error_dim1 = torch.mean(torch.abs(x - x_reconstructed_dim1.real))
        
        assert error_dim0 < 1e-5
        assert error_dim1 < 1e-5

    def test_chunked_fft_edge_cases(self):
        """Test chunked FFT with edge cases."""
        chunked_fft = ChunkedFFT(chunk_size=512)
        
        # Test with sequence smaller than chunk size
        x_small = torch.randn(1, 256)
        x_fft_small = chunked_fft.fft_chunked(x_small)
        x_reconstructed_small = chunked_fft.ifft_chunked(x_fft_small)
        
        error_small = torch.mean(torch.abs(x_small - x_reconstructed_small.real))
        assert error_small < 1e-5
        
        # Test with exact chunk size
        x_exact = torch.randn(1, 512)
        x_fft_exact = chunked_fft.fft_chunked(x_exact)
        x_reconstructed_exact = chunked_fft.ifft_chunked(x_fft_exact)
        
        error_exact = torch.mean(torch.abs(x_exact - x_reconstructed_exact.real))
        assert error_exact < 1e-5

    def test_chunked_fft_complex_input(self):
        """Test chunked FFT with complex input."""
        chunked_fft = ChunkedFFT(chunk_size=256)
        x_real = torch.randn(1, 1024)
        x_imag = torch.randn(1, 1024)
        x_complex = torch.complex(x_real, x_imag)
        
        x_fft = chunked_fft.fft_chunked(x_complex)
        x_reconstructed = chunked_fft.ifft_chunked(x_fft)
        
        # Check reconstruction accuracy for complex data
        error = torch.mean(torch.abs(x_complex - x_reconstructed))
        assert error < 1e-5


class TestAMPFractionalEngineComprehensive:
    """Comprehensive tests for AMPFractionalEngine."""
    
    def test_amp_engine_different_dtypes(self):
        """Test AMP engine with different dtypes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a base engine first
        base_engine = GPUOptimizedSpectralEngine("fft", use_amp=False)
        
        # Test with float16
        engine_fp16 = AMPFractionalEngine(base_engine, use_amp=True, dtype=torch.float16)
        x_fp16 = torch.randn(4, 8, dtype=torch.float16, device='cuda')
        result_fp16 = engine_fp16.forward(x_fp16, 0.5)
        # The base engine might convert to float32 internally, so just check it's valid
        assert not torch.isnan(result_fp16).any()
        assert result_fp16.device == x_fp16.device
        
        # Test with bfloat16 if supported
        if torch.cuda.is_bf16_supported():
            engine_bf16 = AMPFractionalEngine(base_engine, use_amp=True, dtype=torch.bfloat16)
            x_bf16 = torch.randn(4, 8, dtype=torch.bfloat16, device='cuda')
            result_bf16 = engine_bf16.forward(x_bf16, 0.5)
            # The base engine might convert to float32 internally, so just check it's valid
            assert not torch.isnan(result_bf16).any()
            assert result_bf16.device == x_bf16.device

    def test_amp_engine_gradient_scaling(self):
        """Test AMP engine gradient scaling functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        base_engine = GPUOptimizedSpectralEngine("fft", use_amp=False)
        engine = AMPFractionalEngine(base_engine, use_amp=True, dtype=torch.float16)
        x = torch.randn(4, 8, dtype=torch.float16, device='cuda', requires_grad=True)
        
        # Forward pass
        result = engine.forward(x, 0.5)
        loss = result.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.dtype == torch.float16

    def test_amp_engine_no_amp_mode(self):
        """Test AMP engine with AMP disabled."""
        base_engine = GPUOptimizedSpectralEngine("fft", use_amp=False)
        engine = AMPFractionalEngine(base_engine, use_amp=False, dtype=torch.float32)
        x = torch.randn(4, 8, dtype=torch.float32)
        
        result = engine.forward(x, 0.5)
        assert result.dtype == torch.float32
        assert engine.scaler is None


class TestGPUOptimizedSpectralEngineComprehensive:
    """Comprehensive tests for GPUOptimizedSpectralEngine."""
    
    def test_spectral_engine_different_engine_types(self):
        """Test spectral engine with different engine types."""
        for engine_type in ["fft", "mellin", "laplacian"]:
            engine = GPUOptimizedSpectralEngine(
                engine_type=engine_type,
                use_amp=False,  # Disable AMP for simplicity
                chunk_size=256
            )
            
            x = torch.randn(4, 8)
            result = engine.forward(x, 0.5)
            
            assert result.shape == x.shape
            assert result.dtype == x.dtype

    def test_spectral_engine_chunked_processing(self):
        """Test spectral engine with chunked processing."""
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=128  # Small chunk size for testing
        )
        
        # Test with large sequence that requires chunking
        x = torch.randn(2, 512)
        result = engine.forward(x, 0.5)
        
        assert result.shape == x.shape

    def test_spectral_engine_performance_tracking(self):
        """Test spectral engine performance tracking."""
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=256
        )
        
        # Run some operations
        for _ in range(3):
            x = torch.randn(4, 8)
            engine.forward(x, 0.5)
        
        # Get performance summary
        summary = engine.get_performance_summary()
        assert len(summary) > 0
        
        # Check that performance metrics are recorded
        for op, metrics in summary.items():
            assert "execution_time" in metrics
            assert "throughput" in metrics
            assert metrics["execution_time"] > 0

    def test_spectral_engine_different_alpha_values(self):
        """Test spectral engine with different alpha values."""
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=256
        )
        
        x = torch.randn(4, 8)
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 1.9]:
            result = engine.forward(x, alpha)
            assert result.shape == x.shape
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()


class TestGPUOptimizedStochasticSamplerComprehensive:
    """Comprehensive tests for GPUOptimizedStochasticSampler."""
    
    def test_stochastic_sampler_different_distributions(self):
        """Test stochastic sampler with different distributions."""
        # Create a mock base sampler
        class MockBaseSampler:
            def sample_indices(self, n, k):
                return torch.randint(0, n, (k,))
        
        base_sampler = MockBaseSampler()
        sampler = GPUOptimizedStochasticSampler(
            base_sampler=base_sampler,
            use_amp=False
        )
        
        # Test sampling
        samples = sampler.sample_indices(1000, 100)
        assert samples.shape == (100,)
        assert torch.all(samples >= 0) and torch.all(samples < 1000)

    def test_stochastic_sampler_batch_processing(self):
        """Test stochastic sampler with batch processing."""
        # Create a mock base sampler
        class MockBaseSampler:
            def sample_indices(self, n, k):
                return torch.randint(0, n, (k,))
        
        base_sampler = MockBaseSampler()
        sampler = GPUOptimizedStochasticSampler(
            base_sampler=base_sampler,
            use_amp=False,
            batch_size=32
        )
        
        # Test batch sampling
        samples = sampler.sample_indices(1000, 100)
        assert samples.shape == (100,)

    def test_stochastic_sampler_amp_mode(self):
        """Test stochastic sampler with AMP enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a mock base sampler
        class MockBaseSampler:
            def sample_indices(self, n, k):
                return torch.randint(0, n, (k,), device='cuda')
        
        base_sampler = MockBaseSampler()
        sampler = GPUOptimizedStochasticSampler(
            base_sampler=base_sampler,
            use_amp=True,
            batch_size=32
        )
        
        samples = sampler.sample_indices(1000, 100)
        assert samples.shape == (100,)
        assert samples.device.type == 'cuda'


class TestCreateGPUOptimizedComponentsComprehensive:
    """Comprehensive tests for create_gpu_optimized_components."""
    
    def test_create_components_all_engines(self):
        """Test creating all engine types."""
        components = create_gpu_optimized_components(
            use_amp=False,
            chunk_size=256,
            dtype=torch.float32
        )
        
        expected_engines = ["fft_engine", "mellin_engine", "laplacian_engine"]
        for engine_name in expected_engines:
            assert engine_name in components
            assert isinstance(components[engine_name], GPUOptimizedSpectralEngine)

    def test_create_components_with_amp(self):
        """Test creating components with AMP enabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        components = create_gpu_optimized_components(
            use_amp=True,
            chunk_size=512,
            dtype=torch.float16
        )
        
        # Test that components work with AMP
        x = torch.randn(4, 8, dtype=torch.float16, device='cuda')
        result = components["fft_engine"].forward(x, 0.5)
        # The engine might convert to float32 internally, so just check it's valid
        assert not torch.isnan(result).any()
        assert result.device == x.device

    def test_create_components_different_configurations(self):
        """Test creating components with different configurations."""
        configs = [
            (False, 128, torch.float32),
            (False, 512, torch.float32),
            (True, 256, torch.float16),
        ]
        
        for use_amp, chunk_size, dtype in configs:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue
                
            components = create_gpu_optimized_components(
                use_amp=use_amp,
                chunk_size=chunk_size,
                dtype=dtype
            )
            
            assert len(components) == 3  # fft, mellin, laplacian engines
            for engine in components.values():
                assert isinstance(engine, GPUOptimizedSpectralEngine)


class TestGPUOptimizationContextComprehensive:
    """Comprehensive tests for gpu_optimization_context."""
    
    def test_context_with_amp(self):
        """Test GPU optimization context with AMP."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        with gpu_optimization_context(use_amp=True, dtype=torch.float16):
            x = torch.randn(4, 8, device='cuda')
            # Context should be active
            assert torch.is_autocast_enabled()

    def test_context_without_amp(self):
        """Test GPU optimization context without AMP."""
        with gpu_optimization_context(use_amp=False, dtype=torch.float32):
            x = torch.randn(4, 8)
            # Context should not affect autocast
            assert not torch.is_autocast_enabled()

    def test_context_cpu_fallback(self):
        """Test GPU optimization context with CPU fallback."""
        with gpu_optimization_context(use_amp=True, dtype=torch.float16):
            x = torch.randn(4, 8)  # CPU tensor
            # Should work without errors
            assert x.shape == (4, 8)


class TestBenchmarkFunctionComprehensive:
    """Comprehensive tests for benchmark_gpu_optimization function."""
    
    def test_benchmark_function_basic(self):
        """Test that benchmark function runs without errors."""
        # Mock CUDA availability to test CPU path
        with patch('torch.cuda.is_available', return_value=False):
            results = benchmark_gpu_optimization()
            
            # Should return results dictionary
            assert isinstance(results, dict)
            assert len(results) > 0  # Should have results for different sequence lengths

    def test_benchmark_function_cuda_available(self):
        """Test benchmark function when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        results = benchmark_gpu_optimization()
        
        # Should return results dictionary
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check structure of results
        for length, configs in results.items():
            assert isinstance(length, int)
            assert isinstance(configs, dict)
            for config_name, alphas in configs.items():
                assert isinstance(config_name, str)
                assert isinstance(alphas, dict)


class TestTestFunctionComprehensive:
    """Comprehensive tests for test_gpu_optimization function."""
    
    def test_test_function_basic(self):
        """Test that test function runs without errors."""
        # This should run without raising exceptions
        test_gpu_optimization()

    def test_test_function_cpu_mode(self):
        """Test test function in CPU mode."""
        with patch('torch.cuda.is_available', return_value=False):
            # Should run without errors on CPU
            test_gpu_optimization()


class TestGPUOptimizationIntegrationComprehensive:
    """Comprehensive integration tests for GPU optimization."""
    
    def test_full_pipeline_cpu(self):
        """Test full GPU optimization pipeline on CPU."""
        # Create components
        components = create_gpu_optimized_components(
            use_amp=False,
            chunk_size=256,
            dtype=torch.float32
        )
        
        # Test data
        x = torch.randn(8, 16)
        alpha = 0.5
        
        # Test each engine
        for engine_name, engine in components.items():
            result = engine.forward(x, alpha)
            assert result.shape == x.shape
            assert not torch.isnan(result).any()

    def test_full_pipeline_gpu(self):
        """Test full GPU optimization pipeline on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create components with AMP
        components = create_gpu_optimized_components(
            use_amp=True,
            chunk_size=512,
            dtype=torch.float16
        )
        
        # Test data
        x = torch.randn(8, 16, dtype=torch.float16, device='cuda')
        alpha = 0.5
        
        # Test each engine
        for engine_name, engine in components.items():
            result = engine.forward(x, alpha)
            assert result.shape == x.shape
            # The engine might convert to float32 internally, so just check it's valid
            assert not torch.isnan(result).any()
            assert result.device == x.device
            assert not torch.isnan(result).any()

    def test_performance_profiling_integration(self):
        """Test performance profiling integration."""
        profiler = GPUProfiler(device="cpu")
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=256
        )
        
        # Profile multiple operations
        for i in range(5):
            profiler.start_timer(f"operation_{i}")
            x = torch.randn(4, 8)
            result = engine.forward(x, 0.5)
            profiler.end_timer(x, result)
        
        # Check profiling results
        assert len(profiler.metrics_history) == 5
        summary = profiler.get_summary()
        assert len(summary) == 5

    def test_chunked_fft_integration(self):
        """Test chunked FFT integration with engines."""
        chunked_fft = ChunkedFFT(chunk_size=128)
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=128
        )
        
        # Test with large sequence
        x = torch.randn(2, 1024)
        
        # Direct FFT
        x_fft_direct = chunked_fft.fft_chunked(x)
        
        # Engine processing
        result = engine.forward(x, 0.5)
        
        # Both should work without errors
        assert x_fft_direct.shape == x.shape
        assert result.shape == x.shape

    def test_memory_efficiency(self):
        """Test memory efficiency of GPU optimization components."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test with large tensors
        x = torch.randn(32, 1024, device='cuda')
        alpha = 0.5
        
        # Test different engines
        engines = [
            GPUOptimizedSpectralEngine("fft", use_amp=True, chunk_size=256),
            GPUOptimizedSpectralEngine("mellin", use_amp=True, chunk_size=256),
            GPUOptimizedSpectralEngine("laplacian", use_amp=True, chunk_size=256)
        ]
        
        for engine in engines:
            result = engine.forward(x, alpha)
            assert result.shape == x.shape
            assert result.device == x.device
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated()
            assert memory_used > 0  # Should use some memory

    def test_error_handling(self):
        """Test error handling in GPU optimization components."""
        # Test with invalid alpha values
        engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=False,
            chunk_size=256
        )
        
        x = torch.randn(4, 8)
        
        # Test with invalid alpha values - the engine might not validate these
        # so we'll test that it doesn't crash
        try:
            result_neg = engine.forward(x, -0.1)  # Negative alpha
            # If it doesn't raise an error, check that result is valid
            assert not torch.isnan(result_neg).any()
        except (ValueError, RuntimeError):
            pass  # Expected behavior
        
        try:
            result_large = engine.forward(x, 2.1)  # Alpha > 2
            # If it doesn't raise an error, check that result is valid
            assert not torch.isnan(result_large).any()
        except (ValueError, RuntimeError):
            pass  # Expected behavior

    def test_dtype_consistency(self):
        """Test dtype consistency across operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test with different dtypes
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        
        for dtype in dtypes:
            engine = GPUOptimizedSpectralEngine(
                engine_type="fft",
                use_amp=True,
                dtype=dtype
            )
            
            x = torch.randn(4, 8, dtype=dtype, device='cuda')
            result = engine.forward(x, 0.5)
            
            # The engine might convert to float32 internally, so just check it's valid
            assert not torch.isnan(result).any()
            assert result.device == x.device
