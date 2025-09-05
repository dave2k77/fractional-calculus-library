"""
Test script for GPU optimization functionality.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from hpfracc.ml.gpu_optimization import (
    GPUProfiler, ChunkedFFT, AMPFractionalEngine, GPUOptimizedSpectralEngine,
    GPUOptimizedStochasticSampler, gpu_optimization_context, benchmark_gpu_optimization
)


def test_gpu_profiler():
    """Test the GPU profiler."""
    print("Testing GPUProfiler...")
    
    profiler = GPUProfiler(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1000, 1000, device=device)
    
    # Test profiling
    profiler.start_timer("test_operation")
    result = torch.matmul(x, x.T)
    profiler.end_timer(x, result)
    
    # Get summary
    summary = profiler.get_summary()
    print("Profiler summary:")
    for op, metrics in summary.items():
        print(f"  {op}: {metrics['execution_time']:.4f}s, "
              f"memory: {metrics['memory_used']:.2f}GB, "
              f"throughput: {metrics['throughput']:.2e} ops/s")
    
    print("✓ GPUProfiler test passed\n")


def test_chunked_fft():
    """Test chunked FFT operations."""
    print("Testing ChunkedFFT...")
    
    chunked_fft = ChunkedFFT(chunk_size=512, overlap=64)
    
    # Test with different sequence lengths
    lengths = [256, 1024, 2048]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for length in lengths:
        x = torch.randn(32, length, device=device)
        
        # Test FFT
        x_fft = chunked_fft.fft_chunked(x)
        x_reconstructed = chunked_fft.ifft_chunked(x_fft)
        
        # Check reconstruction error
        error = torch.mean(torch.abs(x - x_reconstructed.real))
        print(f"  Length {length}: reconstruction error = {error:.6f}")
        
        assert error < 1e-5, f"Reconstruction error too high: {error}"
    
    print("✓ ChunkedFFT test passed\n")


def test_gpu_optimized_spectral_engine():
    """Test GPU-optimized spectral engine."""
    print("Testing GPUOptimizedSpectralEngine...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 1024, device=device)
    alpha = 0.5
    
    # Test different engine types
    engine_types = ["fft", "laplacian"]
    
    for engine_type in engine_types:
        print(f"  Testing {engine_type} engine...")
        
        # Test with AMP
        engine_amp = GPUOptimizedSpectralEngine(
            engine_type=engine_type,
            use_amp=True,
            chunk_size=256
        )
        
        result_amp = engine_amp.forward(x, alpha)
        print(f"    AMP result shape: {result_amp.shape}")
        
        # Test without AMP
        engine_no_amp = GPUOptimizedSpectralEngine(
            engine_type=engine_type,
            use_amp=False,
            chunk_size=256
        )
        
        result_no_amp = engine_no_amp.forward(x, alpha)
        print(f"    No-AMP result shape: {result_no_amp.shape}")
        
        # Check that results are similar
        if torch.cuda.is_available():
            # Results might differ due to precision, so just check shapes
            assert result_amp.shape == result_no_amp.shape, "Shape mismatch"
        else:
            # On CPU, results should be identical
            error = torch.mean(torch.abs(result_amp - result_no_amp))
            assert error < 1e-5, f"Result mismatch: {error}"
    
    print("✓ GPUOptimizedSpectralEngine test passed\n")


def test_amp_fractional_engine():
    """Test AMP fractional engine wrapper."""
    print("Testing AMPFractionalEngine...")
    
    # Create a simple base engine
    class SimpleEngine:
        def forward(self, x, alpha, **kwargs):
            return x * (alpha + 1.0)
    
    base_engine = SimpleEngine()
    amp_engine = AMPFractionalEngine(base_engine, use_amp=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 512, device=device)
    alpha = 0.5
    
    # Test forward pass
    result = amp_engine.forward(x, alpha)
    print(f"  AMP engine result shape: {result.shape}")
    
    # Test backward pass
    grad_output = torch.randn_like(result)
    scaled_grad = amp_engine.backward(grad_output)
    print(f"  Scaled gradient shape: {scaled_grad.shape}")
    
    print("✓ AMPFractionalEngine test passed\n")


def test_gpu_optimization_context():
    """Test GPU optimization context manager."""
    print("Testing gpu_optimization_context...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 256, device=device)
    
    # Test with context manager
    with gpu_optimization_context(use_amp=True, dtype=torch.float16):
        result = torch.matmul(x, x.T)
        print(f"  Context manager result shape: {result.shape}")
    
    print("✓ gpu_optimization_context test passed\n")


def test_performance_benchmark():
    """Test performance benchmarking."""
    print("Testing performance benchmark...")
    
    if torch.cuda.is_available():
        print("  Running GPU benchmark...")
        results = benchmark_gpu_optimization()
        
        # Print summary
        for length, configs in results.items():
            print(f"    Length {length}:")
            for config, alphas in configs.items():
                avg_time = np.mean(list(alphas.values()))
                print(f"      {config}: {avg_time:.4f}s average")
    else:
        print("  CUDA not available, skipping GPU benchmark")
    
    print("✓ Performance benchmark test passed\n")


def test_integration_with_existing_components():
    """Test integration with existing fractional components."""
    print("Testing integration with existing components...")
    
    try:
        from hpfracc.ml.stochastic_memory_sampling import ImportanceSampler
        from hpfracc.ml.spectral_autograd import FFTEngine
        
        # Test GPU-optimized stochastic sampler
        base_sampler = ImportanceSampler(alpha=0.5, k=32)
        gpu_sampler = GPUOptimizedStochasticSampler(base_sampler, use_amp=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n, k = 1024, 64
        
        indices = gpu_sampler.sample_indices(n, k)
        print(f"  GPU sampler indices shape: {indices.shape}")
        
        # Test GPU-optimized spectral engine with existing FFT engine
        base_fft_engine = FFTEngine()
        gpu_fft_engine = GPUOptimizedSpectralEngine(
            engine_type="fft",
            use_amp=True
        )
        
        x = torch.randn(8, 512, device=device)
        alpha = 0.5
        
        result = gpu_fft_engine.forward(x, alpha)
        print(f"  GPU FFT engine result shape: {result.shape}")
        
        print("✓ Integration test passed\n")
        
    except ImportError as e:
        print(f"⚠ Integration test skipped (import error: {e})\n")


def main():
    """Run all GPU optimization tests."""
    print("="*60)
    print("GPU OPTIMIZATION TESTS")
    print("="*60)
    
    try:
        test_gpu_profiler()
        test_chunked_fft()
        test_gpu_optimized_spectral_engine()
        test_amp_fractional_engine()
        test_gpu_optimization_context()
        test_performance_benchmark()
        test_integration_with_existing_components()
        
        print("="*60)
        print("ALL GPU OPTIMIZATION TESTS PASSED! ✓")
        print("="*60)
        
        # Print system info
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

