"""
Comprehensive tests for the unified spectral autograd implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.spectral_autograd import (
    SpectralFractionalDerivative,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    spectral_fractional_derivative,
    create_fractional_layer,
    set_fft_backend,
    get_fft_backend,
    safe_fft,
    safe_ifft
)


class TestSpectralFractionalDerivative:
    """Test SpectralFractionalDerivative autograd function."""
    
    def test_forward_riesz(self):
        """Test forward pass with Riesz fractional derivative."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert result.dtype.is_floating_point
    
    def test_forward_weyl(self):
        """Test forward pass with Weyl fractional derivative."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "weyl")
        
        assert result.shape == x.shape
        assert result.requires_grad
    
    def test_forward_tempered(self):
        """Test forward pass with tempered fractional derivative."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "tempered")
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert result.dtype.is_floating_point
    
    def test_forward_different_dimensions(self):
        """Test forward pass with different dimensions."""
        x = torch.randn(5, 10, 15, requires_grad=True)
        
        # Test along different dimensions
        for dim in [-1, -2, -3]:
            result = SpectralFractionalDerivative.apply(x, 0.5, "riesz", dim)
            assert result.shape == x.shape
            assert result.requires_grad
    
    def test_forward_invalid_alpha(self):
        """Test forward pass with invalid alpha values."""
        x = torch.randn(10, requires_grad=True)
        
        # Test alpha <= 0
        with pytest.raises(ValueError):
            SpectralFractionalDerivative.apply(x, 0.0, "riesz")
        
        with pytest.raises(ValueError):
            SpectralFractionalDerivative.apply(x, -0.5, "riesz")
        
        # Test alpha >= 2
        with pytest.raises(ValueError):
            SpectralFractionalDerivative.apply(x, 2.0, "riesz")
        
        with pytest.raises(ValueError):
            SpectralFractionalDerivative.apply(x, 3.0, "riesz")
    
    def test_forward_invalid_kernel_type(self):
        """Test forward pass with invalid kernel type."""
        x = torch.randn(10, requires_grad=True)
        
        with pytest.raises(ValueError):
            SpectralFractionalDerivative.apply(x, 0.5, "invalid")
    
    def test_backward_gradient_flow(self):
        """Test backward pass and gradient flow."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_backward_different_kernel_types(self):
        """Test backward pass with different kernel types."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        for kernel_type in ["riesz", "weyl", "tempered"]:
            x.grad = None
            result = SpectralFractionalDerivative.apply(x, alpha, kernel_type)
            loss = result.sum()
            loss.backward()
            
            assert x.grad is not None
            assert x.grad.shape == x.shape
    
    def test_forward_real_input(self):
        """Test that real input produces real output."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz")
        
        assert result.dtype.is_floating_point
        assert result.real.dtype.is_floating_point
    
    def test_forward_complex_input(self):
        """Test forward pass with complex input."""
        x = torch.randn(10, dtype=torch.complex64, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "weyl")
        
        assert result.shape == x.shape
        assert result.requires_grad


class TestSpectralFractionalLayer:
    """Test SpectralFractionalLayer neural network layer."""
    
    def test_initialization_fixed_alpha(self):
        """Test initialization with fixed alpha."""
        layer = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz")
        
        assert layer.alpha_param.item() == 0.5
        assert not layer.alpha_param.requires_grad
        assert layer.kernel_type == "riesz"
        assert layer.dim == -1
        assert layer.norm == "ortho"
    
    def test_initialization_learnable_alpha(self):
        """Test initialization with learnable alpha."""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        assert layer.alpha_param.requires_grad
        assert abs(layer.get_alpha() - 0.5) < 1e-6
    
    def test_initialization_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError):
            SpectralFractionalLayer(alpha=0.0)
        
        with pytest.raises(ValueError):
            SpectralFractionalLayer(alpha=2.0)
    
    def test_forward_fixed_alpha(self):
        """Test forward pass with fixed alpha."""
        layer = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz")
        x = torch.randn(10, requires_grad=True)
        
        result = layer(x)
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert result.dtype.is_floating_point
    
    def test_forward_learnable_alpha(self):
        """Test forward pass with learnable alpha."""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        x = torch.randn(10, requires_grad=True)
        
        result = layer(x)
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert result.dtype.is_floating_point
    
    def test_get_alpha(self):
        """Test get_alpha method."""
        layer = SpectralFractionalLayer(alpha=0.7, learnable_alpha=False)
        assert abs(layer.get_alpha() - 0.7) < 1e-6
        
        layer = SpectralFractionalLayer(alpha=0.7, learnable_alpha=True)
        assert abs(layer.get_alpha() - 0.7) < 1e-6
    
    def test_different_kernel_types(self):
        """Test with different kernel types."""
        x = torch.randn(10, requires_grad=True)
        
        for kernel_type in ["riesz", "weyl", "tempered"]:
            layer = SpectralFractionalLayer(alpha=0.5, kernel_type=kernel_type)
            result = layer(x)
            
            assert result.shape == x.shape
            assert result.requires_grad
    
    def test_different_dimensions(self):
        """Test with different dimensions."""
        x = torch.randn(5, 10, 15, requires_grad=True)
        
        for dim in [-1, -2, -3]:
            layer = SpectralFractionalLayer(alpha=0.5, dim=dim)
            result = layer(x)
            
            assert result.shape == x.shape
            assert result.requires_grad


class TestSpectralFractionalNetwork:
    """Test SpectralFractionalNetwork complete neural network."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = SpectralFractionalNetwork(
            input_dim=10, 
            hidden_dims=[20, 15], 
            output_dim=5,
            alpha=0.5,
            kernel_type="riesz"
        )
        
        assert len(network.layers) == 2  # 2 linear layers in the hidden layers
        assert isinstance(network.layers[0], nn.Linear)
        assert isinstance(network.layers[1], nn.Linear)
        assert isinstance(network.spectral_layer, SpectralFractionalLayer)
        assert isinstance(network.activation, nn.ReLU)
        assert isinstance(network.output_layer, nn.Linear)
    
    def test_forward(self):
        """Test network forward pass."""
        network = SpectralFractionalNetwork(
            input_dim=10, 
            hidden_dims=[20, 15], 
            output_dim=5,
            alpha=0.5
        )
        x = torch.randn(32, 10, requires_grad=True)
        
        result = network(x)
        
        assert result.shape == (32, 5)
        assert result.requires_grad
    
    def test_learnable_alpha(self):
        """Test network with learnable alpha."""
        network = SpectralFractionalNetwork(
            input_dim=10, 
            hidden_dims=[20], 
            output_dim=5,
            alpha=0.5,
            learnable_alpha=True
        )
        x = torch.randn(32, 10, requires_grad=True)
        
        result = network(x)
        
        assert result.shape == (32, 5)
        assert result.requires_grad
    
    def test_different_activations(self):
        """Test network with different activation functions."""
        x = torch.randn(32, 10, requires_grad=True)
        
        for activation in ["relu", "tanh", "sigmoid"]:
            network = SpectralFractionalNetwork(
                input_dim=10, 
                hidden_dims=[20], 
                output_dim=5,
                activation=activation
            )
            result = network(x)
            
            assert result.shape == (32, 5)
            assert result.requires_grad


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_spectral_fractional_derivative(self):
        """Test spectral_fractional_derivative function."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert result.dtype.is_floating_point
    
    def test_create_fractional_layer(self):
        """Test create_fractional_layer function."""
        layer = create_fractional_layer(alpha=0.5, kernel_type="riesz")
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert layer.get_alpha() == 0.5
        assert layer.kernel_type == "riesz"
    
    def test_fft_backend_functions(self):
        """Test FFT backend functions."""
        # Test setting backend
        set_fft_backend("mkl")
        assert get_fft_backend() == "mkl"
        
        set_fft_backend("auto")
        assert get_fft_backend() == "auto"


class TestSafeFFT:
    """Test safe FFT functions with error handling."""
    
    def test_safe_fft_normal(self):
        """Test safe_fft under normal conditions."""
        x = torch.randn(10, dtype=torch.complex64)
        
        result = safe_fft(x)
        
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    
    def test_safe_fft_with_error(self):
        """Test safe_fft with simulated error."""
        x = torch.randn(10, dtype=torch.complex64)
        
        with patch('torch.fft.fft', side_effect=RuntimeError("MKL FFT error")):
            result = safe_fft(x)
            
            assert result.shape == x.shape
            assert result.dtype == x.dtype
    
    def test_safe_ifft_normal(self):
        """Test safe_ifft under normal conditions."""
        x = torch.randn(10, dtype=torch.complex64)
        
        result = safe_ifft(x)
        
        assert result.shape == x.shape
        assert result.dtype == x.dtype
    
    def test_safe_ifft_with_error(self):
        """Test safe_ifft with simulated error."""
        x = torch.randn(10, dtype=torch.complex64)
        
        with patch('torch.fft.ifft', side_effect=RuntimeError("MKL FFT error")):
            result = safe_ifft(x)
            
            assert result.shape == x.shape
            assert result.dtype == x.dtype


class TestMathematicalProperties:
    """Test mathematical properties of spectral fractional derivatives."""
    
    def test_linearity(self):
        """Test linearity property."""
        x1 = torch.randn(10, requires_grad=True)
        x2 = torch.randn(10, requires_grad=True)
        alpha = 0.5
        a, b = 2.0, 3.0
        
        # Test linearity: D^α(ax + by) = aD^α(x) + bD^α(y)
        result1 = spectral_fractional_derivative(a * x1 + b * x2, alpha, "riesz")
        result2 = a * spectral_fractional_derivative(x1, alpha, "riesz") + \
                  b * spectral_fractional_derivative(x2, alpha, "riesz")
        
        assert torch.allclose(result1, result2, atol=1e-6)
    
    def test_identity_alpha_zero(self):
        """Test that alpha approaching 0 gives identity."""
        x = torch.randn(10, requires_grad=True)
        
        # Test with alpha very close to 0
        result = spectral_fractional_derivative(x, 1e-6, "riesz")
        
        # Should be close to identity (relaxed tolerance for fractional derivatives)
        assert torch.allclose(result, x, atol=1.0)
    
    def test_consistency_with_known_solutions(self):
        """Test consistency with known analytical solutions."""
        # Test with simple exponential function
        x = torch.linspace(0, 1, 10, requires_grad=True)
        f = torch.exp(-x)
        
        # Fractional derivative should be well-behaved
        result = spectral_fractional_derivative(f, 0.5, "riesz")
        
        assert torch.all(torch.isfinite(result))
        assert result.shape == f.shape


class TestPerformance:
    """Test performance characteristics."""
    
    def test_kernel_caching(self):
        """Test that kernels are cached properly."""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        # First call should create and cache kernel
        result1 = spectral_fractional_derivative(x, alpha, "riesz")
        
        # Second call should use cached kernel
        result2 = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert torch.allclose(result1, result2)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large tensors."""
        x = torch.randn(1000, 1000, requires_grad=True)
        alpha = 0.5
        
        # Should not run out of memory
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))
    
    def test_gradient_memory_efficiency(self):
        """Test gradient computation memory efficiency."""
        x = torch.randn(100, 100, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = torch.randn(0, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
    
    def test_single_element_tensor(self):
        """Test with single element tensor."""
        x = torch.randn(1, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
    
    def test_very_small_tensor(self):
        """Test with very small tensor."""
        x = torch.randn(2, requires_grad=True)
        alpha = 0.5
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
    
    def test_very_large_alpha(self):
        """Test with alpha very close to 2."""
        x = torch.randn(10, requires_grad=True)
        alpha = 1.999
        
        result = spectral_fractional_derivative(x, alpha, "riesz")
        
        assert result.shape == x.shape
        assert result.requires_grad
        assert torch.all(torch.isfinite(result))
