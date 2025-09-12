"""
Comprehensive tests for ML spectral_autograd module.

This module tests all spectral autograd functionality including fractional derivatives,
FFT operations, and neural network integration to ensure high coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.spectral_autograd import (
    set_fft_backend, get_fft_backend, safe_fft, safe_ifft,
    _get_fractional_kernel, SpectralFractionalDerivative,
    SpectralFractionalLayer, SpectralFractionalNetwork,
    spectral_fractional_derivative, create_fractional_layer
)


class TestFFTBackend:
    """Test FFT backend configuration."""
    
    def test_set_fft_backend(self):
        """Test setting FFT backend."""
        original_backend = get_fft_backend()
        
        set_fft_backend("mkl")
        assert get_fft_backend() == "mkl"
        
        set_fft_backend("fftw")
        assert get_fft_backend() == "fftw"
        
        set_fft_backend("numpy")
        assert get_fft_backend() == "numpy"
        
        # Restore original
        set_fft_backend(original_backend)
    
    def test_get_fft_backend(self):
        """Test getting FFT backend."""
        backend = get_fft_backend()
        assert isinstance(backend, str)
        assert backend in ["auto", "mkl", "fftw", "numpy"]


class TestSafeFFT:
    """Test safe FFT operations."""
    
    def test_safe_fft_basic(self):
        """Test basic safe FFT operation."""
        x = torch.randn(2, 3, 4)
        result = safe_fft(x)
        
        assert result.shape == x.shape
        assert result.dtype == torch.complex64 or result.dtype == torch.complex128
    
    def test_safe_fft_different_dims(self):
        """Test safe FFT with different dimensions."""
        x = torch.randn(2, 3, 4)
        
        for dim in [-1, 0, 1, 2]:
            result = safe_fft(x, dim=dim)
            assert result.shape == x.shape
    
    def test_safe_fft_different_norms(self):
        """Test safe FFT with different normalization modes."""
        x = torch.randn(2, 3, 4)
        
        for norm in ["ortho", "backward", "forward"]:
            result = safe_fft(x, norm=norm)
            assert result.shape == x.shape
    
    def test_safe_fft_complex_input(self):
        """Test safe FFT with complex input."""
        x = torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)
        result = safe_fft(x)
        
        assert result.shape == x.shape
        assert result.dtype == torch.complex64 or result.dtype == torch.complex128
    
    def test_safe_fft_empty_tensor(self):
        """Test safe FFT with empty tensor."""
        x = torch.randn(0, 3, 4)
        result = safe_fft(x)
        
        assert result.shape == x.shape


class TestSafeIFFT:
    """Test safe IFFT operations."""
    
    def test_safe_ifft_basic(self):
        """Test basic safe IFFT operation."""
        x = torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)
        result = safe_ifft(x)
        
        assert result.shape == x.shape
        assert result.dtype == torch.complex64 or result.dtype == torch.complex128
    
    def test_safe_ifft_different_dims(self):
        """Test safe IFFT with different dimensions."""
        x = torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)
        
        for dim in [-1, 0, 1, 2]:
            result = safe_ifft(x, dim=dim)
            assert result.shape == x.shape
    
    def test_safe_ifft_different_norms(self):
        """Test safe IFFT with different normalization modes."""
        x = torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)
        
        for norm in ["ortho", "backward", "forward"]:
            result = safe_ifft(x, norm=norm)
            assert result.shape == x.shape


class TestFractionalKernel:
    """Test fractional kernel generation."""
    
    def test_get_fractional_kernel_riesz(self):
        """Test Riesz fractional kernel generation."""
        kernel = _get_fractional_kernel(0.5, 10, "riesz")
        
        assert kernel.shape == (10,)
        assert kernel.dtype == torch.float32 or kernel.dtype == torch.float64
    
    def test_get_fractional_kernel_weyl(self):
        """Test Weyl fractional kernel generation."""
        kernel = _get_fractional_kernel(0.5, 10, "weyl")
        
        assert kernel.shape == (10,)
        # Weyl kernel is complex by nature
        assert kernel.dtype == torch.complex64 or kernel.dtype == torch.complex128
    
    def test_get_fractional_kernel_different_orders(self):
        """Test fractional kernel with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            kernel = _get_fractional_kernel(alpha, 10, "riesz")
            assert kernel.shape == (10,)
    
    def test_get_fractional_kernel_different_sizes(self):
        """Test fractional kernel with different sizes."""
        for n in [5, 10, 20, 50]:
            kernel = _get_fractional_kernel(0.5, n, "riesz")
            assert kernel.shape == (n,)


class TestSpectralFractionalDerivative:
    """Test SpectralFractionalDerivative autograd function."""
    
    def test_spectral_derivative_forward(self):
        """Test spectral derivative forward pass."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad == x.requires_grad
    
    def test_spectral_derivative_backward(self):
        """Test spectral derivative backward pass."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_spectral_derivative_different_orders(self):
        """Test spectral derivative with different orders."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
            assert result.shape == x.shape
    
    def test_spectral_derivative_different_kernels(self):
        """Test spectral derivative with different kernel types."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        for kernel_type in ["riesz", "weyl"]:
            result = SpectralFractionalDerivative.apply(x, alpha, kernel_type)
            assert result.shape == x.shape


class TestSpectralFractionalLayer:
    """Test SpectralFractionalLayer neural network layer."""
    
    def test_spectral_layer_initialization(self):
        """Test SpectralFractionalLayer initialization."""
        layer = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz")
        
        assert layer.alpha == 0.5
        assert layer.kernel_type == "riesz"
        assert layer.learnable is False
    
    def test_spectral_layer_learnable(self):
        """Test SpectralFractionalLayer with learnable alpha."""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        assert layer.learnable is True
        assert hasattr(layer, 'alpha_param')
    
    def test_spectral_layer_forward(self):
        """Test SpectralFractionalLayer forward pass."""
        layer = SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(2, 3, 4)
        
        result = layer(x)
        
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    
    def test_spectral_layer_learnable_forward(self):
        """Test SpectralFractionalLayer with learnable alpha forward pass."""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        x = torch.randn(2, 3, 4)
        
        result = layer(x)
        
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    
    def test_spectral_layer_different_orders(self):
        """Test SpectralFractionalLayer with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            layer = SpectralFractionalLayer(alpha=alpha)
            x = torch.randn(2, 3, 4)
            result = layer(x)
            assert result.shape == x.shape
    
    def test_spectral_layer_different_kernels(self):
        """Test SpectralFractionalLayer with different kernel types."""
        for kernel_type in ["riesz", "weyl"]:
            layer = SpectralFractionalLayer(alpha=0.5, kernel_type=kernel_type)
            x = torch.randn(2, 3, 4)
            result = layer(x)
            assert result.shape == x.shape


class TestSpectralFractionalNetwork:
    """Test SpectralFractionalNetwork neural network."""
    
    def test_spectral_network_initialization(self):
        """Test SpectralFractionalNetwork initialization."""
        network = SpectralFractionalNetwork(
            input_dim=10,
            hidden_dims=[20, 30],
            output_dim=5,
            alpha=0.5
        )
        
        assert network.input_size == 10
        assert network.hidden_sizes == [20, 30]
        assert network.output_size == 5
        assert network.alpha == 0.5
    
    def test_spectral_network_forward(self):
        """Test SpectralFractionalNetwork forward pass."""
        network = SpectralFractionalNetwork(
            input_dim=10,
            hidden_dims=[20, 30],
            output_dim=5,
            alpha=0.5
        )
        x = torch.randn(2, 10)
        
        result = network(x)
        
        assert result.shape == (2, 5)
        assert result.dtype == x.dtype
    
    def test_spectral_network_different_sizes(self):
        """Test SpectralFractionalNetwork with different sizes."""
        for hidden_sizes in [[10], [20, 30], [50, 40, 30]]:
            network = SpectralFractionalNetwork(
                input_dim=10,
                hidden_dims=hidden_sizes,
                output_dim=5,
                alpha=0.5
            )
            x = torch.randn(2, 10)
            result = network(x)
            assert result.shape == (2, 5)
    
    def test_spectral_network_learnable_alpha(self):
        """Test SpectralFractionalNetwork with learnable alpha."""
        network = SpectralFractionalNetwork(
            input_dim=10,
            hidden_dims=[20],
            output_dim=5,
            alpha=0.5,
            learnable_alpha=True
        )
        x = torch.randn(2, 10)
        
        result = network(x)
        
        assert result.shape == (2, 5)
        assert result.dtype == x.dtype


class TestSpectralFractionalDerivativeFunction:
    """Test spectral_fractional_derivative function."""
    
    def test_spectral_derivative_function(self):
        """Test spectral_fractional_derivative function."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad == x.requires_grad
    
    def test_spectral_derivative_function_different_orders(self):
        """Test spectral_fractional_derivative with different orders."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = spectral_fractional_derivative(x, alpha, "riesz")
            assert result.shape == x.shape
    
    def test_spectral_derivative_function_different_kernels(self):
        """Test spectral_fractional_derivative with different kernel types."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        for kernel_type in ["riesz", "weyl"]:
            result = spectral_fractional_derivative(x, alpha, kernel_type)
            assert result.shape == x.shape


class TestCreateFractionalLayer:
    """Test create_fractional_layer function."""
    
    def test_create_fractional_layer(self):
        """Test create_fractional_layer function."""
        layer = create_fractional_layer(alpha=0.5, kernel_type="riesz")
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert layer.alpha == 0.5
        assert layer.kernel_type == "riesz"
    
    def test_create_fractional_layer_learnable(self):
        """Test create_fractional_layer with learnable alpha."""
        layer = create_fractional_layer(alpha=0.5, learnable_alpha=True)
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert layer.learnable is True
    
    def test_create_fractional_layer_different_orders(self):
        """Test create_fractional_layer with different orders."""
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            layer = create_fractional_layer(alpha=alpha)
            assert layer.alpha == alpha
    
    def test_create_fractional_layer_different_kernels(self):
        """Test create_fractional_layer with different kernel types."""
        for kernel_type in ["riesz", "weyl"]:
            layer = create_fractional_layer(alpha=0.5, kernel_type=kernel_type)
            assert layer.kernel_type == kernel_type


class TestSpectralIntegration:
    """Test spectral autograd integration scenarios."""
    
    def test_fft_ifft_roundtrip(self):
        """Test FFT-IFFT roundtrip."""
        x = torch.randn(2, 3, 4)
        
        fft_result = safe_fft(x)
        ifft_result = safe_ifft(fft_result)
        
        # Should be close to original (within numerical precision)
        assert torch.allclose(x, ifft_result.real, atol=1e-6)
    
    def test_spectral_derivative_gradient_flow(self):
        """Test gradient flow through spectral derivative."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_spectral_layer_in_network(self):
        """Test spectral layer in a neural network."""
        layer = SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(2, 3, 4, requires_grad=True)
        
        # Test forward pass
        result = layer(x)
        assert result.shape == x.shape
        
        # Test backward pass
        loss = result.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_spectral_network_training(self):
        """Test spectral network training scenario."""
        network = SpectralFractionalNetwork(
            input_dim=10,
            hidden_dims=[20],
            output_dim=5,
            alpha=0.5
        )
        x = torch.randn(2, 10, requires_grad=True)
        target = torch.randn(2, 5)
        
        # Forward pass
        output = network(x)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestSpectralEdgeCases:
    """Test spectral autograd edge cases and error handling."""
    
    def test_spectral_derivative_empty_tensor(self):
        """Test spectral derivative with empty tensor."""
        x = torch.randn(0, 3, 4, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        assert result.shape == x.shape
    
    def test_spectral_derivative_single_element(self):
        """Test spectral derivative with single element tensor."""
        x = torch.randn(1, 1, 1, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        assert result.shape == x.shape
    
    def test_spectral_derivative_large_tensor(self):
        """Test spectral derivative with large tensor."""
        x = torch.randn(10, 10, 10, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        assert result.shape == x.shape
    
    def test_spectral_derivative_extreme_orders(self):
        """Test spectral derivative with extreme orders."""
        x = torch.randn(2, 3, 4, requires_grad=True)
        
        for alpha in [0.01, 0.99]:
            result = spectral_fractional_derivative(x, alpha, "riesz")
            assert result.shape == x.shape
    
    def test_spectral_derivative_complex_input(self):
        """Test spectral derivative with complex input."""
        x = torch.randn(2, 3, 4) + 1j * torch.randn(2, 3, 4)
        x.requires_grad_(True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        assert result.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
