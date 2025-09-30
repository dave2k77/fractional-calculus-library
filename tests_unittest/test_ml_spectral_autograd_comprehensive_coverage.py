"""
Comprehensive coverage tests for hpfracc/ml/spectral_autograd.py

This module provides extensive tests to achieve maximum coverage of the
spectral fractional calculus utilities for ML integration.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Tuple, Union


class TestSpectralAutogradUtilities(unittest.TestCase):
    """Comprehensive tests for spectral autograd utilities"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import (
            _is_complex_dtype, _get_backend, _safe_fft, _safe_ifft,
            spectral_fractional_derivative, SpectralFractionalLayer,
            LearnableAlpha, SpectralFractionalNetwork
        )
        
        self._is_complex_dtype = _is_complex_dtype
        self._get_backend = _get_backend
        self._safe_fft = _safe_fft
        self._safe_ifft = _safe_ifft
        self.spectral_fractional_derivative = spectral_fractional_derivative
        self.SpectralFractionalLayer = SpectralFractionalLayer
        self.LearnableAlpha = LearnableAlpha
        self.SpectralFractionalNetwork = SpectralFractionalNetwork

    def test_is_complex_dtype_true(self):
        """Test _is_complex_dtype with complex dtypes"""
        self.assertTrue(self._is_complex_dtype(torch.complex64))
        self.assertTrue(self._is_complex_dtype(torch.complex128))

    def test_is_complex_dtype_false(self):
        """Test _is_complex_dtype with non-complex dtypes"""
        self.assertFalse(self._is_complex_dtype(torch.float32))
        self.assertFalse(self._is_complex_dtype(torch.float64))
        self.assertFalse(self._is_complex_dtype(torch.int32))
        self.assertFalse(self._is_complex_dtype(torch.int64))

    def test_get_backend_torch(self):
        """Test _get_backend with torch backend"""
        x = torch.randn(10)
        backend = self._get_backend(x)
        self.assertEqual(backend, "torch")

    def test_get_backend_numpy(self):
        """Test _get_backend with numpy backend"""
        x = np.random.randn(10)
        backend = self._get_backend(x)
        self.assertEqual(backend, "numpy")

    def test_get_backend_jax_available(self):
        """Test _get_backend with JAX when available"""
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            backend = self._get_backend(x)
            self.assertEqual(backend, "jax")
        except ImportError:
            self.skipTest("JAX not available")

    def test_safe_fft_torch_real(self):
        """Test _safe_fft with torch real tensor"""
        x = torch.randn(10)
        result = self._safe_fft(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.is_complex(result))

    def test_safe_fft_torch_complex(self):
        """Test _safe_fft with torch complex tensor"""
        x = torch.randn(10, dtype=torch.complex64)
        result = self._safe_fft(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.is_complex(result))

    def test_safe_fft_numpy_real(self):
        """Test _safe_fft with numpy real array"""
        x = np.random.randn(10)
        result = self._safe_fft(x)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))

    def test_safe_fft_numpy_complex(self):
        """Test _safe_fft with numpy complex array"""
        x = np.random.randn(10) + 1j * np.random.randn(10)
        result = self._safe_fft(x)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))

    def test_safe_fft_jax_available(self):
        """Test _safe_fft with JAX when available"""
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            result = self._safe_fft(x)
            self.assertTrue(hasattr(result, 'shape'))
        except ImportError:
            self.skipTest("JAX not available")

    def test_safe_ifft_torch_complex(self):
        """Test _safe_ifft with torch complex tensor"""
        x = torch.randn(10, dtype=torch.complex64)
        result = self._safe_ifft(x)
        self.assertIsInstance(result, torch.Tensor)
        # Result should be complex for complex input
        self.assertTrue(torch.is_complex(result))

    def test_safe_ifft_numpy_complex(self):
        """Test _safe_ifft with numpy complex array"""
        x = np.random.randn(10) + 1j * np.random.randn(10)
        result = self._safe_ifft(x)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.iscomplexobj(result))

    def test_safe_ifft_jax_available(self):
        """Test _safe_ifft with JAX when available"""
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
            result = self._safe_ifft(x)
            self.assertTrue(hasattr(result, 'shape'))
        except ImportError:
            self.skipTest("JAX not available")

    def test_spectral_fractional_derivative_torch_real(self):
        """Test spectral_fractional_derivative with torch real tensor"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(result.requires_grad)

    def test_spectral_fractional_derivative_torch_complex(self):
        """Test spectral_fractional_derivative with torch complex tensor"""
        x = torch.randn(10, dtype=torch.complex64, requires_grad=True)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.is_complex(result))

    def test_spectral_fractional_derivative_numpy_real(self):
        """Test spectral_fractional_derivative with numpy real array"""
        x = np.random.randn(10)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_numpy_complex(self):
        """Test spectral_fractional_derivative with numpy complex array"""
        x = np.random.randn(10) + 1j * np.random.randn(10)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.iscomplexobj(result))

    def test_spectral_fractional_derivative_jax_available(self):
        """Test spectral_fractional_derivative with JAX when available"""
        try:
            import jax.numpy as jnp
            x = jnp.array([1.0, 2.0, 3.0])
            alpha = 0.5
            
            result = self.spectral_fractional_derivative(x, alpha)
            self.assertTrue(hasattr(result, 'shape'))
        except ImportError:
            self.skipTest("JAX not available")

    def test_spectral_fractional_derivative_alpha_zero(self):
        """Test spectral_fractional_derivative with alpha=0"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.0
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        # Should return input unchanged for alpha=0
        torch.testing.assert_close(result, x)

    def test_spectral_fractional_derivative_alpha_one(self):
        """Test spectral_fractional_derivative with alpha=1"""
        x = torch.randn(10, requires_grad=True)
        alpha = 1.0
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_alpha_half(self):
        """Test spectral_fractional_derivative with alpha=0.5"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_alpha_negative(self):
        """Test spectral_fractional_derivative with negative alpha"""
        x = torch.randn(10, requires_grad=True)
        alpha = -0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_alpha_large(self):
        """Test spectral_fractional_derivative with large alpha"""
        x = torch.randn(10, requires_grad=True)
        alpha = 2.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)


class TestSpectralFractionalLayer(unittest.TestCase):
    """Comprehensive tests for SpectralFractionalLayer"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import SpectralFractionalLayer
        
        self.SpectralFractionalLayer = SpectralFractionalLayer

    def test_spectral_fractional_layer_initialization_default(self):
        """Test SpectralFractionalLayer initialization with default parameters"""
        layer = self.SpectralFractionalLayer()
        
        # Test default attributes
        self.assertEqual(layer.alpha, 0.5)
        self.assertTrue(layer.learnable_alpha)
        self.assertIsInstance(layer.alpha_param, nn.Parameter)
        self.assertEqual(layer.alpha_param.item(), 0.5)

    def test_spectral_fractional_layer_initialization_custom_alpha(self):
        """Test SpectralFractionalLayer initialization with custom alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.7)
        
        self.assertEqual(layer.alpha, 0.7)
        self.assertTrue(layer.learnable_alpha)
        self.assertEqual(layer.alpha_param.item(), 0.7)

    def test_spectral_fractional_layer_initialization_fixed_alpha(self):
        """Test SpectralFractionalLayer initialization with fixed alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.3, learnable_alpha=False)
        
        self.assertEqual(layer.alpha, 0.3)
        self.assertFalse(layer.learnable_alpha)
        self.assertIsNone(layer.alpha_param)

    def test_spectral_fractional_layer_forward_pass_learnable(self):
        """Test SpectralFractionalLayer forward pass with learnable alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        x = torch.randn(10, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(result.requires_grad)

    def test_spectral_fractional_layer_forward_pass_fixed(self):
        """Test SpectralFractionalLayer forward pass with fixed alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=False)
        x = torch.randn(10, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(result.requires_grad)

    def test_spectral_fractional_layer_forward_pass_complex(self):
        """Test SpectralFractionalLayer forward pass with complex input"""
        layer = self.SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(10, dtype=torch.complex64, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.is_complex(result))

    def test_spectral_fractional_layer_forward_pass_batch(self):
        """Test SpectralFractionalLayer forward pass with batch input"""
        layer = self.SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(32, 10, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_layer_forward_pass_2d(self):
        """Test SpectralFractionalLayer forward pass with 2D input"""
        layer = self.SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(32, 10, 20, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_layer_get_alpha_learnable(self):
        """Test SpectralFractionalLayer get_alpha with learnable alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        alpha = layer.get_alpha()
        
        self.assertIsInstance(alpha, torch.Tensor)
        self.assertTrue(alpha.requires_grad)

    def test_spectral_fractional_layer_get_alpha_fixed(self):
        """Test SpectralFractionalLayer get_alpha with fixed alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=False)
        
        alpha = layer.get_alpha()
        
        self.assertIsInstance(alpha, (int, float))
        self.assertEqual(alpha, 0.5)

    def test_spectral_fractional_layer_set_alpha_learnable(self):
        """Test SpectralFractionalLayer set_alpha with learnable alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        
        layer.set_alpha(0.7)
        
        self.assertEqual(layer.alpha_param.item(), 0.7)

    def test_spectral_fractional_layer_set_alpha_fixed(self):
        """Test SpectralFractionalLayer set_alpha with fixed alpha"""
        layer = self.SpectralFractionalLayer(alpha=0.5, learnable_alpha=False)
        
        layer.set_alpha(0.7)
        
        self.assertEqual(layer.alpha, 0.7)


class TestLearnableAlpha(unittest.TestCase):
    """Comprehensive tests for LearnableAlpha"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import LearnableAlpha
        
        self.LearnableAlpha = LearnableAlpha

    def test_learnable_alpha_initialization_default(self):
        """Test LearnableAlpha initialization with default parameters"""
        alpha_param = self.LearnableAlpha()
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.5)

    def test_learnable_alpha_initialization_custom_value(self):
        """Test LearnableAlpha initialization with custom value"""
        alpha_param = self.LearnableAlpha(0.7)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.7)

    def test_learnable_alpha_initialization_bounds(self):
        """Test LearnableAlpha initialization with bounds"""
        alpha_param = self.LearnableAlpha(0.3, min_val=0.1, max_val=0.9)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.3)

    def test_learnable_alpha_initialization_min_bound(self):
        """Test LearnableAlpha initialization at min bound"""
        alpha_param = self.LearnableAlpha(0.1, min_val=0.1, max_val=0.9)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.1)

    def test_learnable_alpha_initialization_max_bound(self):
        """Test LearnableAlpha initialization at max bound"""
        alpha_param = self.LearnableAlpha(0.9, min_val=0.1, max_val=0.9)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.9)

    def test_learnable_alpha_initialization_clamp_below_min(self):
        """Test LearnableAlpha initialization with value below min (should be clamped)"""
        alpha_param = self.LearnableAlpha(0.05, min_val=0.1, max_val=0.9)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.1)

    def test_learnable_alpha_initialization_clamp_above_max(self):
        """Test LearnableAlpha initialization with value above max (should be clamped)"""
        alpha_param = self.LearnableAlpha(0.95, min_val=0.1, max_val=0.9)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.9)


class TestSpectralFractionalNetwork(unittest.TestCase):
    """Comprehensive tests for SpectralFractionalNetwork"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork
        
        self.SpectralFractionalNetwork = SpectralFractionalNetwork

    def test_spectral_fractional_network_initialization_default(self):
        """Test SpectralFractionalNetwork initialization with default parameters"""
        network = self.SpectralFractionalNetwork(input_size=10, hidden_sizes=[20, 30], output_size=5)
        
        # Test basic attributes
        self.assertEqual(network.input_size, 10)
        self.assertEqual(network.hidden_sizes, [20, 30])
        self.assertEqual(network.output_size, 5)
        self.assertEqual(network.alpha, 0.5)
        self.assertTrue(network.learnable_alpha)
        
        # Test layer structure
        self.assertIsInstance(network.layers, nn.ModuleList)
        self.assertIsInstance(network.spectral_layer, nn.Module)
        self.assertIsInstance(network.activation, nn.Module)
        self.assertIsInstance(network.output_layer, nn.Linear)

    def test_spectral_fractional_network_initialization_custom_alpha(self):
        """Test SpectralFractionalNetwork initialization with custom alpha"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            alpha=0.7, learnable_alpha=False
        )
        
        self.assertEqual(network.alpha, 0.7)
        self.assertFalse(network.learnable_alpha)

    def test_spectral_fractional_network_initialization_custom_activation(self):
        """Test SpectralFractionalNetwork initialization with custom activation"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            activation=nn.Tanh()
        )
        
        self.assertIsInstance(network.activation, nn.Tanh)

    def test_spectral_fractional_network_forward_pass(self):
        """Test SpectralFractionalNetwork forward pass"""
        network = self.SpectralFractionalNetwork(input_size=10, hidden_sizes=[20], output_size=5)
        x = torch.randn(32, 10)
        
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (32, 5))

    def test_spectral_fractional_network_forward_pass_batch(self):
        """Test SpectralFractionalNetwork forward pass with different batch size"""
        network = self.SpectralFractionalNetwork(input_size=10, hidden_sizes=[20], output_size=5)
        x = torch.randn(16, 10)
        
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (16, 5))

    def test_spectral_fractional_network_forward_pass_single(self):
        """Test SpectralFractionalNetwork forward pass with single sample"""
        network = self.SpectralFractionalNetwork(input_size=10, hidden_sizes=[20], output_size=5)
        x = torch.randn(1, 10)
        
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 5))

    def test_spectral_fractional_network_forward_pass_requires_grad(self):
        """Test SpectralFractionalNetwork forward pass with requires_grad"""
        network = self.SpectralFractionalNetwork(input_size=10, hidden_sizes=[20], output_size=5)
        x = torch.randn(32, 10, requires_grad=True)
        
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (32, 5))
        self.assertTrue(result.requires_grad)

    def test_spectral_fractional_network_get_alpha_learnable(self):
        """Test SpectralFractionalNetwork get_alpha with learnable alpha"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=True
        )
        
        alpha = network.get_alpha()
        
        self.assertIsInstance(alpha, torch.Tensor)
        self.assertTrue(alpha.requires_grad)

    def test_spectral_fractional_network_get_alpha_fixed(self):
        """Test SpectralFractionalNetwork get_alpha with fixed alpha"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=False
        )
        
        alpha = network.get_alpha()
        
        self.assertIsInstance(alpha, (int, float))
        self.assertEqual(alpha, 0.5)

    def test_spectral_fractional_network_set_alpha_learnable(self):
        """Test SpectralFractionalNetwork set_alpha with learnable alpha"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=True
        )
        
        network.set_alpha(0.7)
        
        alpha = network.get_alpha()
        self.assertEqual(alpha.item(), 0.7)

    def test_spectral_fractional_network_set_alpha_fixed(self):
        """Test SpectralFractionalNetwork set_alpha with fixed alpha"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=False
        )
        
        network.set_alpha(0.7)
        
        alpha = network.get_alpha()
        self.assertEqual(alpha, 0.7)

    def test_spectral_fractional_network_multiple_hidden_layers(self):
        """Test SpectralFractionalNetwork with multiple hidden layers"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20, 30, 40], output_size=5
        )
        
        x = torch.randn(32, 10)
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (32, 5))

    def test_spectral_fractional_network_no_hidden_layers(self):
        """Test SpectralFractionalNetwork with no hidden layers"""
        network = self.SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[], output_size=5
        )
        
        x = torch.randn(32, 10)
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (32, 5))


class TestSpectralAutogradEdgeCases(unittest.TestCase):
    """Comprehensive tests for edge cases in spectral autograd"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import (
            spectral_fractional_derivative, SpectralFractionalLayer,
            LearnableAlpha, SpectralFractionalNetwork
        )
        
        self.spectral_fractional_derivative = spectral_fractional_derivative
        self.SpectralFractionalLayer = SpectralFractionalLayer
        self.LearnableAlpha = LearnableAlpha
        self.SpectralFractionalNetwork = SpectralFractionalNetwork

    def test_spectral_fractional_derivative_empty_tensor(self):
        """Test spectral_fractional_derivative with empty tensor"""
        x = torch.randn(0, requires_grad=True)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_single_element(self):
        """Test spectral_fractional_derivative with single element tensor"""
        x = torch.randn(1, requires_grad=True)
        alpha = 0.5
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_very_small_alpha(self):
        """Test spectral_fractional_derivative with very small alpha"""
        x = torch.randn(10, requires_grad=True)
        alpha = 1e-10
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_very_large_alpha(self):
        """Test spectral_fractional_derivative with very large alpha"""
        x = torch.randn(10, requires_grad=True)
        alpha = 100.0
        
        result = self.spectral_fractional_derivative(x, alpha)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_layer_zero_alpha(self):
        """Test SpectralFractionalLayer with alpha=0"""
        layer = self.SpectralFractionalLayer(alpha=0.0)
        x = torch.randn(10, requires_grad=True)
        
        result = layer.forward(x)
        
        # Should return input unchanged for alpha=0
        torch.testing.assert_close(result, x)

    def test_spectral_fractional_layer_alpha_one(self):
        """Test SpectralFractionalLayer with alpha=1"""
        layer = self.SpectralFractionalLayer(alpha=1.0)
        x = torch.randn(10, requires_grad=True)
        
        result = layer.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_learnable_alpha_extreme_bounds(self):
        """Test LearnableAlpha with extreme bounds"""
        alpha_param = self.LearnableAlpha(0.5, min_val=-10.0, max_val=10.0)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), 0.5)

    def test_learnable_alpha_negative_value(self):
        """Test LearnableAlpha with negative value"""
        alpha_param = self.LearnableAlpha(-0.5, min_val=-1.0, max_val=1.0)
        
        self.assertIsInstance(alpha_param, nn.Parameter)
        self.assertEqual(alpha_param.item(), -0.5)

    def test_spectral_fractional_network_extreme_sizes(self):
        """Test SpectralFractionalNetwork with extreme sizes"""
        network = self.SpectralFractionalNetwork(
            input_size=1, hidden_sizes=[2], output_size=1
        )
        
        x = torch.randn(1, 1)
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 1))

    def test_spectral_fractional_network_large_sizes(self):
        """Test SpectralFractionalNetwork with large sizes"""
        network = self.SpectralFractionalNetwork(
            input_size=100, hidden_sizes=[200, 300], output_size=50
        )
        
        x = torch.randn(16, 100)
        result = network.forward(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (16, 50))


if __name__ == '__main__':
    unittest.main()
