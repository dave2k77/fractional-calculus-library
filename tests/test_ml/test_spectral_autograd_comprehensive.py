"""
Comprehensive tests for hpfracc.ml.spectral_autograd module

This module tests spectral fractional calculus utilities for ML integration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple

from hpfracc.ml.spectral_autograd import (
    set_fft_backend,
    get_fft_backend,
    robust_fft,
    robust_ifft,
    safe_fft,
    safe_ifft,
    spectral_fractional_derivative,
    SpectralFractionalDerivative,
    SpectralFractionalFunction,
    fractional_derivative,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    BoundedAlphaParameter,
    create_fractional_layer,
    benchmark_backends
)


class TestFFTBackendManagement:
    """Test FFT backend management functions"""

    def test_set_fft_backend_valid(self):
        """Test setting valid FFT backend"""
        result = set_fft_backend("torch")
        
        assert result == "torch"
        assert get_fft_backend() == "torch"

    def test_set_fft_backend_invalid(self):
        """Test setting invalid FFT backend"""
        with pytest.raises(ValueError):
            set_fft_backend("invalid_backend")

    def test_get_fft_backend_default(self):
        """Test getting default FFT backend"""
        backend = get_fft_backend()
        
        assert backend in ["torch", "numpy"]
        assert isinstance(backend, str)

    def test_set_fft_backend_numpy(self):
        """Test setting numpy FFT backend"""
        result = set_fft_backend("numpy")
        
        assert result == "numpy"
        assert get_fft_backend() == "numpy"

    def test_set_fft_backend_torch(self):
        """Test setting torch FFT backend"""
        result = set_fft_backend("torch")
        
        assert result == "torch"
        assert get_fft_backend() == "torch"


class TestRobustFFT:
    """Test robust FFT functions"""

    def test_robust_fft_basic(self):
        """Test basic robust FFT operation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result = robust_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.complex64

    def test_robust_fft_empty(self):
        """Test robust FFT with empty tensor"""
        x = torch.tensor([])
        
        result = robust_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_robust_fft_2d(self):
        """Test robust FFT with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = robust_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.complex64

    def test_robust_fft_different_dims(self):
        """Test robust FFT with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        result = robust_fft(x, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = robust_fft(x, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_robust_fft_different_norms(self):
        """Test robust FFT with different normalization modes"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        norms = ["ortho", "forward", "backward"]
        
        for norm in norms:
            result = robust_fft(x, norm=norm)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_robust_ifft_basic(self):
        """Test basic robust IFFT operation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = robust_fft(x)
        
        # Inverse FFT
        result = robust_ifft(fft_result)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32
        
        # Should be close to original
        assert torch.allclose(result, x, atol=1e-5)

    def test_robust_ifft_empty(self):
        """Test robust IFFT with empty tensor"""
        x = torch.tensor([], dtype=torch.complex64)
        
        result = robust_ifft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_robust_ifft_2d(self):
        """Test robust IFFT with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Forward FFT
        fft_result = robust_fft(x)
        
        # Inverse FFT
        result = robust_ifft(fft_result)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_robust_ifft_different_dims(self):
        """Test robust IFFT with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = robust_fft(x)
        
        # Test dim=0
        result = robust_ifft(fft_result, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = robust_ifft(fft_result, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_robust_ifft_different_norms(self):
        """Test robust IFFT with different normalization modes"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = robust_fft(x)
        
        norms = ["ortho", "forward", "backward"]
        
        for norm in norms:
            result = robust_ifft(fft_result, norm=norm)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_roundtrip_consistency(self):
        """Test roundtrip consistency of robust FFT and IFFT"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward then inverse
        fft_result = robust_fft(x)
        reconstructed = robust_ifft(fft_result)
        
        # Should be close to original
        assert torch.allclose(reconstructed, x, atol=1e-5)


class TestSafeFFT:
    """Test safe FFT functions"""

    def test_safe_fft_basic(self):
        """Test basic safe FFT operation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result = safe_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.complex64

    def test_safe_fft_with_backend(self):
        """Test safe FFT with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        result = safe_fft(x, backend="torch")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        result = safe_fft(x, backend="numpy")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_fft_empty(self):
        """Test safe FFT with empty tensor"""
        x = torch.tensor([])
        
        result = safe_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_fft_2d(self):
        """Test safe FFT with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = safe_fft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.complex64

    def test_safe_fft_different_dims(self):
        """Test safe FFT with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        result = safe_fft(x, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = safe_fft(x, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_fft_different_norms(self):
        """Test safe FFT with different normalization modes"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        norms = ["ortho", "forward", "backward"]
        
        for norm in norms:
            result = safe_fft(x, norm=norm)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_safe_ifft_basic(self):
        """Test basic safe IFFT operation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = safe_fft(x)
        
        # Inverse FFT
        result = safe_ifft(fft_result)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32
        
        # Should be close to original
        assert torch.allclose(result, x, atol=1e-5)

    def test_safe_ifft_with_backend(self):
        """Test safe IFFT with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = safe_fft(x)
        
        # Test with torch backend
        result = safe_ifft(fft_result, backend="torch")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        result = safe_ifft(fft_result, backend="numpy")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_ifft_empty(self):
        """Test safe IFFT with empty tensor"""
        x = torch.tensor([], dtype=torch.complex64)
        
        result = safe_ifft(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_ifft_2d(self):
        """Test safe IFFT with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Forward FFT
        fft_result = safe_fft(x)
        
        # Inverse FFT
        result = safe_ifft(fft_result)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_safe_ifft_different_dims(self):
        """Test safe IFFT with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = safe_fft(x)
        
        # Test dim=0
        result = safe_ifft(fft_result, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = safe_ifft(fft_result, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_safe_ifft_different_norms(self):
        """Test safe IFFT with different normalization modes"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward FFT
        fft_result = safe_fft(x)
        
        norms = ["ortho", "forward", "backward"]
        
        for norm in norms:
            result = safe_ifft(fft_result, norm=norm)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_roundtrip_consistency(self):
        """Test roundtrip consistency of safe FFT and IFFT"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward then inverse
        fft_result = safe_fft(x)
        reconstructed = safe_ifft(fft_result)
        
        # Should be close to original
        assert torch.allclose(reconstructed, x, atol=1e-5)


class TestSpectralFractionalDerivative:
    """Test spectral fractional derivative functions"""

    def test_spectral_fractional_derivative_basic(self):
        """Test basic spectral fractional derivative"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_spectral_fractional_derivative_empty(self):
        """Test spectral fractional derivative with empty tensor"""
        x = torch.tensor([])
        
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_spectral_fractional_derivative_different_alpha(self):
        """Test spectral fractional derivative with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            result = spectral_fractional_derivative(x, alpha=alpha)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_spectral_fractional_derivative_2d(self):
        """Test spectral fractional derivative with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_spectral_fractional_derivative_different_dims(self):
        """Test spectral fractional derivative with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        result = spectral_fractional_derivative(x, alpha=0.5, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = spectral_fractional_derivative(x, alpha=0.5, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_spectral_fractional_derivative_with_backend(self):
        """Test spectral fractional derivative with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        result = spectral_fractional_derivative(x, alpha=0.5, backend="torch")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        result = spectral_fractional_derivative(x, alpha=0.5, backend="numpy")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_spectral_fractional_derivative_gradient(self):
        """Test spectral fractional derivative gradient computation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_spectral_fractional_derivative_alpha_gradient(self):
        """Test spectral fractional derivative with learnable alpha"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        alpha = torch.tensor(0.5, requires_grad=True)
        
        result = spectral_fractional_derivative(x, alpha=alpha)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert alpha.grad is not None
        assert alpha.grad.shape == alpha.shape


class TestSpectralFractionalDerivativeClass:
    """Test SpectralFractionalDerivative class"""

    def test_initialization_default(self):
        """Test SpectralFractionalDerivative initialization with default parameters"""
        deriv = SpectralFractionalDerivative()
        
        assert deriv.alpha == 0.5
        assert deriv.dim == -1
        assert deriv.backend is None

    def test_initialization_custom(self):
        """Test SpectralFractionalDerivative initialization with custom parameters"""
        deriv = SpectralFractionalDerivative(alpha=0.7, dim=0, backend="torch")
        
        assert deriv.alpha == 0.7
        assert deriv.dim == 0
        assert deriv.backend == "torch"

    def test_call_basic(self):
        """Test basic call operation"""
        deriv = SpectralFractionalDerivative(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = deriv(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_call_empty(self):
        """Test call with empty tensor"""
        deriv = SpectralFractionalDerivative(alpha=0.5)
        
        x = torch.tensor([])
        result = deriv(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_2d(self):
        """Test call with 2D tensor"""
        deriv = SpectralFractionalDerivative(alpha=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = deriv(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_call_different_alpha(self):
        """Test call with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            deriv = SpectralFractionalDerivative(alpha=alpha)
            result = deriv(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_call_different_dims(self):
        """Test call with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        deriv = SpectralFractionalDerivative(dim=0)
        result = deriv(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        deriv = SpectralFractionalDerivative(dim=-1)
        result = deriv(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_with_backend(self):
        """Test call with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        deriv = SpectralFractionalDerivative(backend="torch")
        result = deriv(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        deriv = SpectralFractionalDerivative(backend="numpy")
        result = deriv(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_gradient(self):
        """Test call gradient computation"""
        deriv = SpectralFractionalDerivative(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        result = deriv(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSpectralFractionalFunction:
    """Test SpectralFractionalFunction class"""

    def test_initialization_default(self):
        """Test SpectralFractionalFunction initialization with default parameters"""
        func = SpectralFractionalFunction()
        
        assert func.alpha == 0.5
        assert func.dim == -1
        assert func.backend is None

    def test_initialization_custom(self):
        """Test SpectralFractionalFunction initialization with custom parameters"""
        func = SpectralFractionalFunction(alpha=0.7, dim=0, backend="torch")
        
        assert func.alpha == 0.7
        assert func.dim == 0
        assert func.backend == "torch"

    def test_call_basic(self):
        """Test basic call operation"""
        func = SpectralFractionalFunction(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = func(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_call_empty(self):
        """Test call with empty tensor"""
        func = SpectralFractionalFunction(alpha=0.5)
        
        x = torch.tensor([])
        result = func(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_2d(self):
        """Test call with 2D tensor"""
        func = SpectralFractionalFunction(alpha=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = func(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_call_different_alpha(self):
        """Test call with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            func = SpectralFractionalFunction(alpha=alpha)
            result = func(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_call_different_dims(self):
        """Test call with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        func = SpectralFractionalFunction(dim=0)
        result = func(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        func = SpectralFractionalFunction(dim=-1)
        result = func(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_with_backend(self):
        """Test call with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        func = SpectralFractionalFunction(backend="torch")
        result = func(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        func = SpectralFractionalFunction(backend="numpy")
        result = func(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_call_gradient(self):
        """Test call gradient computation"""
        func = SpectralFractionalFunction(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        result = func(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestFractionalDerivative:
    """Test fractional_derivative function"""

    def test_fractional_derivative_basic(self):
        """Test basic fractional derivative"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result = fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_fractional_derivative_empty(self):
        """Test fractional derivative with empty tensor"""
        x = torch.tensor([])
        
        result = fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_fractional_derivative_different_alpha(self):
        """Test fractional derivative with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            result = fractional_derivative(x, alpha=alpha)
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_fractional_derivative_2d(self):
        """Test fractional derivative with 2D tensor"""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_fractional_derivative_different_dims(self):
        """Test fractional derivative with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        result = fractional_derivative(x, alpha=0.5, dim=0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        result = fractional_derivative(x, alpha=0.5, dim=-1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_fractional_derivative_with_backend(self):
        """Test fractional derivative with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        result = fractional_derivative(x, alpha=0.5, backend="torch")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        result = fractional_derivative(x, alpha=0.5, backend="numpy")
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_fractional_derivative_gradient(self):
        """Test fractional derivative gradient computation"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        
        result = fractional_derivative(x, alpha=0.5)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_fractional_derivative_alpha_gradient(self):
        """Test fractional derivative with learnable alpha"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        alpha = torch.tensor(0.5, requires_grad=True)
        
        result = fractional_derivative(x, alpha=alpha)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert alpha.grad is not None
        assert alpha.grad.shape == alpha.shape


class TestSpectralFractionalLayer:
    """Test SpectralFractionalLayer class"""

    def test_initialization_default(self):
        """Test SpectralFractionalLayer initialization with default parameters"""
        layer = SpectralFractionalLayer()
        
        assert layer.alpha == 0.5
        assert layer.dim == -1
        assert layer.backend is None
        assert layer.activation is None

    def test_initialization_custom(self):
        """Test SpectralFractionalLayer initialization with custom parameters"""
        layer = SpectralFractionalLayer(
            alpha=0.7,
            dim=0,
            backend="torch",
            activation="relu"
        )
        
        assert layer.alpha == 0.7
        assert layer.dim == 0
        assert layer.backend == "torch"
        assert layer.activation is not None

    def test_forward_basic(self):
        """Test basic forward pass"""
        layer = SpectralFractionalLayer(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_forward_empty(self):
        """Test forward pass with empty tensor"""
        layer = SpectralFractionalLayer(alpha=0.5)
        
        x = torch.tensor([])
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_2d(self):
        """Test forward pass with 2D tensor"""
        layer = SpectralFractionalLayer(alpha=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_forward_different_alpha(self):
        """Test forward pass with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            layer = SpectralFractionalLayer(alpha=alpha)
            result = layer(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_forward_different_dims(self):
        """Test forward pass with different dimensions"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test dim=0
        layer = SpectralFractionalLayer(dim=0)
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test dim=-1
        layer = SpectralFractionalLayer(dim=-1)
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_with_backend(self):
        """Test forward pass with specific backend"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with torch backend
        layer = SpectralFractionalLayer(backend="torch")
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with numpy backend
        layer = SpectralFractionalLayer(backend="numpy")
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_with_activation(self):
        """Test forward pass with activation function"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with relu activation
        layer = SpectralFractionalLayer(activation="relu")
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with sigmoid activation
        layer = SpectralFractionalLayer(activation="sigmoid")
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with tanh activation
        layer = SpectralFractionalLayer(activation="tanh")
        result = layer(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_gradient(self):
        """Test forward pass gradient computation"""
        layer = SpectralFractionalLayer(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        result = layer(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_batch(self):
        """Test forward pass with batch input"""
        layer = SpectralFractionalLayer(alpha=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_forward_learnable_alpha(self):
        """Test forward pass with learnable alpha"""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Check that alpha is learnable
        assert hasattr(layer, 'alpha_param')
        assert layer.alpha_param.requires_grad

    def test_forward_learnable_alpha_gradient(self):
        """Test forward pass gradient with learnable alpha"""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = layer(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert layer.alpha_param.grad is not None
        assert layer.alpha_param.grad.shape == layer.alpha_param.shape


class TestSpectralFractionalNetwork:
    """Test SpectralFractionalNetwork class"""

    def test_initialization_default(self):
        """Test SpectralFractionalNetwork initialization with default parameters"""
        network = SpectralFractionalNetwork()
        
        assert network.input_size == 10
        assert network.hidden_sizes == [64, 32]
        assert network.output_size == 1
        assert network.alpha == 0.5
        assert network.activation == "relu"

    def test_initialization_custom(self):
        """Test SpectralFractionalNetwork initialization with custom parameters"""
        network = SpectralFractionalNetwork(
            input_size=20,
            hidden_sizes=[128, 64, 32],
            output_size=5,
            alpha=0.7,
            activation="sigmoid"
        )
        
        assert network.input_size == 20
        assert network.hidden_sizes == [128, 64, 32]
        assert network.output_size == 5
        assert network.alpha == 0.7
        assert network.activation == "sigmoid"

    def test_forward_basic(self):
        """Test basic forward pass"""
        network = SpectralFractionalNetwork()
        
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        result = network(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == network.output_size

    def test_forward_empty(self):
        """Test forward pass with empty tensor"""
        network = SpectralFractionalNetwork()
        
        x = torch.tensor([])
        result = network(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0
        assert result.shape[1] == network.output_size

    def test_forward_batch(self):
        """Test forward pass with batch input"""
        network = SpectralFractionalNetwork()
        
        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        ])
        result = network(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == network.output_size

    def test_forward_different_alpha(self):
        """Test forward pass with different alpha values"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            network = SpectralFractionalNetwork(alpha=alpha)
            result = network(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == x.shape[0]
            assert result.shape[1] == network.output_size

    def test_forward_different_activation(self):
        """Test forward pass with different activation functions"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        activations = ["relu", "sigmoid", "tanh"]
        
        for activation in activations:
            network = SpectralFractionalNetwork(activation=activation)
            result = network(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == x.shape[0]
            assert result.shape[1] == network.output_size

    def test_forward_different_hidden_sizes(self):
        """Test forward pass with different hidden sizes"""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        hidden_sizes_list = [[32], [64, 32], [128, 64, 32]]
        
        for hidden_sizes in hidden_sizes_list:
            network = SpectralFractionalNetwork(hidden_sizes=hidden_sizes)
            result = network(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == x.shape[0]
            assert result.shape[1] == network.output_size

    def test_forward_gradient(self):
        """Test forward pass gradient computation"""
        network = SpectralFractionalNetwork()
        
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], requires_grad=True)
        result = network(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_learnable_alpha(self):
        """Test forward pass with learnable alpha"""
        network = SpectralFractionalNetwork(learnable_alpha=True)
        
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        result = network(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == network.output_size
        
        # Check that alpha is learnable
        assert hasattr(network, 'alpha_param')
        assert network.alpha_param.requires_grad

    def test_forward_learnable_alpha_gradient(self):
        """Test forward pass gradient with learnable alpha"""
        network = SpectralFractionalNetwork(learnable_alpha=True)
        
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        result = network(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert network.alpha_param.grad is not None
        assert network.alpha_param.grad.shape == network.alpha_param.shape


class TestBoundedAlphaParameter:
    """Test BoundedAlphaParameter class"""

    def test_initialization_default(self):
        """Test BoundedAlphaParameter initialization with default parameters"""
        param = BoundedAlphaParameter()
        
        assert param.alpha == 0.5
        assert param.min_alpha == 0.0
        assert param.max_alpha == 2.0

    def test_initialization_custom(self):
        """Test BoundedAlphaParameter initialization with custom parameters"""
        param = BoundedAlphaParameter(alpha=0.7, min_alpha=0.1, max_alpha=1.5)
        
        assert param.alpha == 0.7
        assert param.min_alpha == 0.1
        assert param.max_alpha == 1.5

    def test_forward_basic(self):
        """Test basic forward pass"""
        param = BoundedAlphaParameter(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = param(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_forward_empty(self):
        """Test forward pass with empty tensor"""
        param = BoundedAlphaParameter(alpha=0.5)
        
        x = torch.tensor([])
        result = param(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_forward_2d(self):
        """Test forward pass with 2D tensor"""
        param = BoundedAlphaParameter(alpha=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = param(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    def test_forward_different_alpha(self):
        """Test forward pass with different alpha values"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            param = BoundedAlphaParameter(alpha=alpha)
            result = param(x)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_forward_gradient(self):
        """Test forward pass gradient computation"""
        param = BoundedAlphaParameter(alpha=0.5)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        result = param(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_learnable_alpha(self):
        """Test forward pass with learnable alpha"""
        param = BoundedAlphaParameter(alpha=0.5, learnable_alpha=True)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = param(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Check that alpha is learnable
        assert hasattr(param, 'alpha_param')
        assert param.alpha_param.requires_grad

    def test_forward_learnable_alpha_gradient(self):
        """Test forward pass gradient with learnable alpha"""
        param = BoundedAlphaParameter(alpha=0.5, learnable_alpha=True)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = param(x)
        
        # Compute gradient
        loss = result.sum()
        loss.backward()
        
        assert param.alpha_param.grad is not None
        assert param.alpha_param.grad.shape == param.alpha_param.shape

    def test_alpha_bounds(self):
        """Test alpha bounds enforcement"""
        param = BoundedAlphaParameter(alpha=0.5, min_alpha=0.1, max_alpha=1.0)
        
        # Test that alpha is within bounds
        assert param.alpha >= param.min_alpha
        assert param.alpha <= param.max_alpha

    def test_alpha_bounds_clamping(self):
        """Test alpha bounds clamping"""
        # Test with alpha below minimum
        param = BoundedAlphaParameter(alpha=0.05, min_alpha=0.1, max_alpha=1.0)
        assert param.alpha >= param.min_alpha
        
        # Test with alpha above maximum
        param = BoundedAlphaParameter(alpha=1.5, min_alpha=0.1, max_alpha=1.0)
        assert param.alpha <= param.max_alpha


class TestCreateFractionalLayer:
    """Test create_fractional_layer function"""

    def test_create_fractional_layer_default(self):
        """Test create_fractional_layer with default parameters"""
        layer = create_fractional_layer()
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert layer.alpha == 0.5
        assert layer.dim == -1
        assert layer.backend is None

    def test_create_fractional_layer_custom(self):
        """Test create_fractional_layer with custom parameters"""
        layer = create_fractional_layer(
            alpha=0.7,
            dim=0,
            backend="torch",
            activation="relu"
        )
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert layer.alpha == 0.7
        assert layer.dim == 0
        assert layer.backend == "torch"
        assert layer.activation is not None

    def test_create_fractional_layer_different_alpha(self):
        """Test create_fractional_layer with different alpha values"""
        alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
        
        for alpha in alpha_values:
            layer = create_fractional_layer(alpha=alpha)
            
            assert isinstance(layer, SpectralFractionalLayer)
            assert layer.alpha == alpha

    def test_create_fractional_layer_different_activation(self):
        """Test create_fractional_layer with different activation functions"""
        activations = ["relu", "sigmoid", "tanh"]
        
        for activation in activations:
            layer = create_fractional_layer(activation=activation)
            
            assert isinstance(layer, SpectralFractionalLayer)
            assert layer.activation is not None

    def test_create_fractional_layer_different_backend(self):
        """Test create_fractional_layer with different backends"""
        backends = ["torch", "numpy"]
        
        for backend in backends:
            layer = create_fractional_layer(backend=backend)
            
            assert isinstance(layer, SpectralFractionalLayer)
            assert layer.backend == backend

    def test_create_fractional_layer_learnable_alpha(self):
        """Test create_fractional_layer with learnable alpha"""
        layer = create_fractional_layer(learnable_alpha=True)
        
        assert isinstance(layer, SpectralFractionalLayer)
        assert hasattr(layer, 'alpha_param')
        assert layer.alpha_param.requires_grad


class TestBenchmarkBackends:
    """Test benchmark_backends function"""

    def test_benchmark_backends_basic(self):
        """Test basic benchmark_backends operation"""
        results = benchmark_backends()
        
        assert isinstance(results, dict)
        assert 'torch' in results
        assert 'numpy' in results
        
        for backend, metrics in results.items():
            assert isinstance(metrics, dict)
            assert 'execution_time' in metrics
            assert 'memory_used' in metrics
            assert 'accuracy' in metrics

    def test_benchmark_backends_custom(self):
        """Test benchmark_backends with custom parameters"""
        results = benchmark_backends(
            test_size=100,
            num_iterations=5,
            backends=["torch", "numpy"]
        )
        
        assert isinstance(results, dict)
        assert 'torch' in results
        assert 'numpy' in results
        
        for backend, metrics in results.items():
            assert isinstance(metrics, dict)
            assert 'execution_time' in metrics
            assert 'memory_used' in metrics
            assert 'accuracy' in metrics

    def test_benchmark_backends_single_backend(self):
        """Test benchmark_backends with single backend"""
        results = benchmark_backends(backends=["torch"])
        
        assert isinstance(results, dict)
        assert 'torch' in results
        assert len(results) == 1
        
        metrics = results['torch']
        assert isinstance(metrics, dict)
        assert 'execution_time' in metrics
        assert 'memory_used' in metrics
        assert 'accuracy' in metrics

    def test_benchmark_backends_performance_metrics(self):
        """Test benchmark_backends performance metrics"""
        results = benchmark_backends()
        
        for backend, metrics in results.items():
            assert isinstance(metrics, dict)
            
            # Check execution time
            assert isinstance(metrics['execution_time'], (int, float))
            assert metrics['execution_time'] > 0
            
            # Check memory usage
            assert isinstance(metrics['memory_used'], (int, float))
            assert metrics['memory_used'] >= 0
            
            # Check accuracy
            assert isinstance(metrics['accuracy'], (int, float))
            assert 0 <= metrics['accuracy'] <= 1

    def test_benchmark_backends_consistency(self):
        """Test benchmark_backends consistency"""
        results1 = benchmark_backends()
        results2 = benchmark_backends()
        
        assert isinstance(results1, dict)
        assert isinstance(results2, dict)
        
        # Results should have same structure
        assert set(results1.keys()) == set(results2.keys())
        
        for backend in results1.keys():
            assert set(results1[backend].keys()) == set(results2[backend].keys())


# Integration tests
class TestSpectralAutogradIntegration:
    """Integration tests for spectral autograd module"""

    def test_full_workflow(self):
        """Test complete spectral autograd workflow"""
        # Create network
        network = SpectralFractionalNetwork()
        
        # Create test data
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        
        # Forward pass
        result = network(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == x.shape[0]
        assert result.shape[1] == network.output_size

    def test_backend_consistency(self):
        """Test backend consistency across operations"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with different backends
        backends = ["torch", "numpy"]
        
        for backend in backends:
            result = spectral_fractional_derivative(x, alpha=0.5, backend=backend)
            
            assert isinstance(result, torch.Tensor)
            assert result.shape == x.shape

    def test_gradient_consistency(self):
        """Test gradient consistency"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        
        # Test gradient computation
        result = spectral_fractional_derivative(x, alpha=0.5)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_learnable_alpha_workflow(self):
        """Test learnable alpha workflow"""
        # Create layer with learnable alpha
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        # Create test data
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Forward pass
        result = layer(x)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test gradient computation
        loss = result.sum()
        loss.backward()
        
        assert layer.alpha_param.grad is not None
        assert layer.alpha_param.grad.shape == layer.alpha_param.shape

    def test_performance_profiling(self):
        """Test performance profiling integration"""
        # Benchmark backends
        results = benchmark_backends()
        
        assert isinstance(results, dict)
        assert 'torch' in results
        assert 'numpy' in results
        
        for backend, metrics in results.items():
            assert isinstance(metrics, dict)
            assert 'execution_time' in metrics
            assert 'memory_used' in metrics
            assert 'accuracy' in metrics

    def test_error_handling(self):
        """Test error handling"""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Test with invalid backend
        with pytest.raises(ValueError):
            spectral_fractional_derivative(x, alpha=0.5, backend="invalid")
        
        # Test with invalid alpha
        with pytest.raises(ValueError):
            spectral_fractional_derivative(x, alpha=-1.0)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test with zero-dimensional tensor
        x = torch.tensor(5.0)
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with very large tensor
        x = torch.tensor([1.0] * 1000)
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
        
        # Test with very small tensor
        x = torch.tensor([1.0])
        result = spectral_fractional_derivative(x, alpha=0.5)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape
