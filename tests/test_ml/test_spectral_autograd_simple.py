#!/usr/bin/env python3
"""
Simple coverage tests for spectral autograd module.
"""

import pytest
import torch
import numpy as np

from hpfracc.ml.spectral_autograd import (
    fractional_derivative,
    spectral_fractional_derivative,
    robust_fft,
    robust_ifft,
    safe_fft,
    safe_ifft
)


class TestSpectralAutogradSimple:
    """Simple tests for spectral autograd functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_input = torch.randn(10)
        self.alpha = 0.5
        
    def test_fractional_derivative_basic(self):
        """Test basic fractional derivative computation."""
        result = fractional_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_fractional_derivative_different_alphas(self):
        """Test fractional derivative with different alpha values."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for alpha in alphas:
            result = fractional_derivative(self.test_input, alpha)
            assert torch.is_tensor(result)
            assert result.shape == self.test_input.shape
            
    def test_spectral_fractional_derivative(self):
        """Test spectral fractional derivative function."""
        result = spectral_fractional_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_robust_fft_operations(self):
        """Test robust FFT operations."""
        # Forward FFT
        fft_result = robust_fft(self.test_input)
        assert torch.is_tensor(fft_result)
        assert torch.is_complex(fft_result)
        
        # Inverse FFT
        ifft_result = robust_ifft(fft_result)
        assert torch.is_tensor(ifft_result)
        assert torch.allclose(ifft_result.real, self.test_input, atol=1e-6)
        
    def test_safe_fft_operations(self):
        """Test safe FFT operations."""
        # Forward FFT
        fft_result = safe_fft(self.test_input)
        assert torch.is_tensor(fft_result)
        assert torch.is_complex(fft_result)
        
        # Inverse FFT
        ifft_result = safe_ifft(fft_result)
        assert torch.is_tensor(ifft_result)
        assert torch.allclose(ifft_result.real, self.test_input, atol=1e-6)
        
    def test_different_input_shapes(self):
        """Test with different input shapes."""
        shapes = [(5,), (5, 10), (3, 5, 10)]
        
        for shape in shapes:
            test_input = torch.randn(shape)
            result = fractional_derivative(test_input, self.alpha)
            assert result.shape == test_input.shape
            
    def test_gradient_computation(self):
        """Test gradient computation through fractional derivative."""
        input_tensor = torch.randn(10, requires_grad=True)
        result = fractional_derivative(input_tensor, self.alpha)
        loss = result.sum()
        
        loss.backward()
        assert input_tensor.grad is not None
        
    def test_different_dtypes(self):
        """Test with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            input_tensor = torch.randn(10, dtype=dtype)
            result = fractional_derivative(input_tensor, self.alpha)
            assert torch.is_tensor(result)
            
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero input
        zero_input = torch.zeros(10)
        zero_result = fractional_derivative(zero_input, self.alpha)
        assert torch.all(torch.isfinite(zero_result))
        
        # Single element
        single_input = torch.tensor([1.0])
        single_result = fractional_derivative(single_input, self.alpha)
        assert torch.is_tensor(single_result)
        
    def test_numerical_stability(self):
        """Test numerical stability."""
        # Very small values
        small_input = torch.full((10,), 1e-8)
        small_result = fractional_derivative(small_input, self.alpha)
        assert torch.all(torch.isfinite(small_result))
        
        # Very large values
        large_input = torch.full((10,), 1e6)
        large_result = fractional_derivative(large_input, self.alpha)
        assert torch.all(torch.isfinite(large_result))
        
    def test_batch_processing(self):
        """Test batch processing."""
        batch_input = torch.randn(32, 10)
        result = fractional_derivative(batch_input, self.alpha)
        assert result.shape == batch_input.shape
        
    def test_device_compatibility(self):
        """Test device compatibility."""
        cpu_input = torch.randn(10)
        cpu_result = fractional_derivative(cpu_input, self.alpha)
        assert cpu_result.device == cpu_input.device
        
        if torch.cuda.is_available():
            cuda_input = cpu_input.cuda()
            cuda_result = fractional_derivative(cuda_input, self.alpha)
            assert cuda_result.device == cuda_input.device

















