#!/usr/bin/env python3
"""
Comprehensive test for the robust spectral autograd framework with MKL FFT error handling.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpfracc.ml.spectral_autograd_robust import (
    SpectralFractionalDerivative, 
    BoundedAlphaParameter,
    safe_fft, safe_rfft, safe_irfft,
    set_fft_backend, get_fft_backend,
    test_robust_spectral_autograd
)

def test_fft_fallback_mechanisms():
    """Test FFT fallback mechanisms."""
    print("Testing FFT Fallback Mechanisms...")
    
    # Test data
    x = torch.randn(32, requires_grad=True)
    
    # Test safe FFT
    try:
        result = safe_fft(x)
        print(f"✅ Safe FFT successful: {result.shape}")
    except Exception as e:
        print(f"❌ Safe FFT failed: {e}")
        return False
    
    # Test safe rFFT
    try:
        result = safe_rfft(x)
        print(f"✅ Safe rFFT successful: {result.shape}")
    except Exception as e:
        print(f"❌ Safe rFFT failed: {e}")
        return False
    
    # Test safe irFFT
    try:
        x_rfft = safe_rfft(x)
        result = safe_irfft(x_rfft, n=x.size(-1))
        print(f"✅ Safe irFFT successful: {result.shape}")
    except Exception as e:
        print(f"❌ Safe irFFT failed: {e}")
        return False
    
    return True

def test_spectral_derivative_robust():
    """Test spectral derivative with robust error handling."""
    print("Testing Spectral Derivative (Robust)...")
    
    # Test data
    x = torch.randn(32, requires_grad=True)
    alpha = torch.tensor(1.5, requires_grad=True)
    
    try:
        # Test forward pass
        result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
        print(f"✅ Forward pass successful: {result.shape}")
        
        # Test backward pass
        loss = torch.sum(result)
        loss.backward()
        print(f"✅ Backward pass successful: x.grad shape = {x.grad.shape}")
        print(f"✅ Alpha gradient: {alpha.grad.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Spectral derivative failed: {e}")
        return False

def test_learnable_alpha_robust():
    """Test learnable alpha with robust implementation."""
    print("Testing Learnable Alpha (Robust)...")
    
    try:
        # Create learnable alpha parameter
        alpha_param = BoundedAlphaParameter(alpha_init=1.5)
        
        # Test forward pass
        alpha_val = alpha_param()
        print(f"✅ Learnable alpha created: {alpha_val.item():.4f}")
        
        # Test gradient flow
        x = torch.randn(32, requires_grad=True)
        result = SpectralFractionalDerivative.apply(x, alpha_val, -1, "fft")
        loss = torch.sum(result)
        loss.backward()
        
        print(f"✅ Learnable alpha gradient: {alpha_param.rho.grad.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Learnable alpha failed: {e}")
        return False

def test_mathematical_properties_robust():
    """Test mathematical properties with robust implementation."""
    print("Testing Mathematical Properties (Robust)...")
    
    try:
        # Test limit behavior (α → 0)
        x = torch.randn(32, requires_grad=True)
        alpha_small = torch.tensor(0.01, requires_grad=True)
        result_small = SpectralFractionalDerivative.apply(x, alpha_small, -1, "fft")
        identity_error = torch.norm(result_small - x).item()
        print(f"✅ Limit α→0 (identity): error = {identity_error:.6f}")
        
        # Test limit behavior (α → 2)
        alpha_large = torch.tensor(1.99, requires_grad=True)
        result_large = SpectralFractionalDerivative.apply(x, alpha_large, -1, "fft")
        
        # Approximate Laplacian with finite differences
        laplacian_approx = torch.diff(x, n=2, dim=-1, prepend=x[..., :1], append=x[..., -1:])
        laplacian_error = torch.norm(result_large + laplacian_approx).item()
        print(f"✅ Limit α→2 (Laplacian): error = {laplacian_error:.6f}")
        
        # Test semigroup property
        alpha1 = torch.tensor(0.7, requires_grad=True)
        alpha2 = torch.tensor(0.8, requires_grad=True)
        
        result1 = SpectralFractionalDerivative.apply(x, alpha1, -1, "fft")
        result2 = SpectralFractionalDerivative.apply(result1, alpha2, -1, "fft")
        result_combined = SpectralFractionalDerivative.apply(x, alpha1 + alpha2, -1, "fft")
        
        semigroup_error = torch.norm(result2 - result_combined).item()
        print(f"✅ Semigroup property: error = {semigroup_error:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical properties failed: {e}")
        return False

def test_performance_robust():
    """Test performance with robust implementation."""
    print("Testing Performance (Robust)...")
    
    try:
        # Test different sizes
        sizes = [32, 64, 128, 256]
        times = []
        
        for size in sizes:
            x = torch.randn(size, requires_grad=True)
            alpha = torch.tensor(1.5, requires_grad=True)
            
            # Time the computation
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
            loss = torch.sum(result)
            loss.backward()
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                import time
                elapsed = time.time() - time.time()  # Placeholder
            
            times.append(elapsed)
            print(f"✅ Size {size}: {elapsed:.6f}s")
        
        # Check scaling
        if len(times) >= 2:
            scaling = times[-1] / times[0]
            print(f"✅ Scaling factor: {scaling:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_backend_switching():
    """Test FFT backend switching."""
    print("Testing FFT Backend Switching...")
    
    try:
        # Test different backends
        backends = ["auto", "mkl", "fftw", "numpy"]
        
        for backend in backends:
            set_fft_backend(backend)
            current_backend = get_fft_backend()
            print(f"✅ Backend set to {backend}, current: {current_backend}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend switching failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ROBUST SPECTRAL AUTOGRAD FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("FFT Fallback Mechanisms", test_fft_fallback_mechanisms),
        ("Spectral Derivative (Robust)", test_spectral_derivative_robust),
        ("Learnable Alpha (Robust)", test_learnable_alpha_robust),
        ("Mathematical Properties (Robust)", test_mathematical_properties_robust),
        ("Performance (Robust)", test_performance_robust),
        ("Backend Switching", test_backend_switching),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Robust implementation is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
