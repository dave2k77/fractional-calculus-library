"""
Fixed tests for hpfracc/ml/spectral_autograd.py
This version handles API mismatches and focuses on achievable coverage
"""

import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import the actual modules
from hpfracc.ml.spectral_autograd import (
    SpectralFractionalDerivative,
    SpectralFractionalFunction,
    spectral_fractional_derivative,
    fractional_derivative,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    BoundedAlphaParameter,
    create_fractional_layer,
    benchmark_backends,
    _resolve_activation_module,
    _is_complex_dtype,
    set_fft_backend,
    get_fft_backend,
    _complex_dtype_for,
    _real_dtype_for,
    _resolve_backend,
    _effective_backend,
    _normalize_dims,
    _ensure_alpha_tensor,
    _validate_alpha,
    _frequency_grid,
    _build_kernel_from_freqs,
    _to_complex,
    _reshape_kernel,
    _get_fractional_kernel,
    _apply_fractional_along_dim,
    _spectral_fractional_impl
)


class TestSpectralAutogradCore(unittest.TestCase):
    """Test core spectral autograd functionality"""

    def test_spectral_fractional_derivative_basic(self):
        """Test basic spectral_fractional_derivative functionality"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5

        result = spectral_fractional_derivative(x, alpha)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_derivative_alpha_validation(self):
        """Test alpha validation in spectral_fractional_derivative"""
        x = torch.randn(10, requires_grad=True)
        
        # Test invalid alpha values
        with self.assertRaises(ValueError):
            spectral_fractional_derivative(x, 0.0)  # alpha=0
            
        with self.assertRaises(ValueError):
            spectral_fractional_derivative(x, 2.0)  # alpha=2
            
        with self.assertRaises(ValueError):
            spectral_fractional_derivative(x, -0.5)  # negative alpha

    def test_spectral_fractional_derivative_complex_input(self):
        """Test spectral_fractional_derivative with complex input"""
        x = torch.randn(10, dtype=torch.complex64, requires_grad=True)
        alpha = 0.5

        result = spectral_fractional_derivative(x, alpha)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.is_complex(result))

    def test_fractional_derivative_wrapper(self):
        """Test fractional_derivative wrapper function"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5

        result = fractional_derivative(x, alpha)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)


class TestSpectralFractionalLayer(unittest.TestCase):
    """Test SpectralFractionalLayer functionality"""

    def test_spectral_fractional_layer_initialization_default(self):
        """Test SpectralFractionalLayer initialization with default parameters"""
        layer = SpectralFractionalLayer()

        # Test default attributes
        self.assertEqual(layer.alpha, 0.5)
        self.assertFalse(layer.learnable_alpha)

    def test_spectral_fractional_layer_initialization_custom_alpha(self):
        """Test SpectralFractionalLayer initialization with custom alpha"""
        layer = SpectralFractionalLayer(alpha=0.7)

        # Test custom attributes - use approximate equality for float comparison
        self.assertAlmostEqual(layer.alpha, 0.7, places=5)
        self.assertFalse(layer.learnable_alpha)

    def test_spectral_fractional_layer_initialization_learnable_alpha(self):
        """Test SpectralFractionalLayer initialization with learnable alpha"""
        layer = SpectralFractionalLayer(alpha=0.3, learnable_alpha=True)

        self.assertAlmostEqual(layer.alpha, 0.3, places=5)
        self.assertTrue(layer.learnable_alpha)

    def test_spectral_fractional_layer_alpha_validation(self):
        """Test SpectralFractionalLayer alpha validation"""
        # Test invalid alpha values
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=0.0)
            
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=2.0)
            
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=-0.5)

    def test_spectral_fractional_layer_forward(self):
        """Test SpectralFractionalLayer forward pass"""
        layer = SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(5, 10, requires_grad=True)

        result = layer(x)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_layer_get_alpha(self):
        """Test SpectralFractionalLayer get_alpha method"""
        layer = SpectralFractionalLayer(alpha=0.6)

        alpha = layer.get_alpha()

        self.assertIsInstance(alpha, (float, torch.Tensor))
        if isinstance(alpha, torch.Tensor):
            self.assertAlmostEqual(alpha.item(), 0.6, places=5)
        else:
            self.assertAlmostEqual(alpha, 0.6, places=5)


class TestSpectralFractionalNetwork(unittest.TestCase):
    """Test SpectralFractionalNetwork functionality"""

    def test_spectral_fractional_network_initialization_default(self):
        """Test SpectralFractionalNetwork initialization with default parameters"""
        network = SpectralFractionalNetwork()

        self.assertIsInstance(network, SpectralFractionalNetwork)
        self.assertIsInstance(network.layers, torch.nn.ModuleList)

    def test_spectral_fractional_network_initialization_custom(self):
        """Test SpectralFractionalNetwork initialization with custom parameters"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20, 30], output_size=5
        )

        self.assertEqual(network.input_size, 10)
        self.assertEqual(network.hidden_sizes, [20, 30])
        self.assertEqual(network.output_size, 5)

    def test_spectral_fractional_network_forward(self):
        """Test SpectralFractionalNetwork forward pass"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        x = torch.randn(3, 10, requires_grad=True)

        result = network(x)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 5))

    def test_spectral_fractional_network_no_hidden_layers(self):
        """Test SpectralFractionalNetwork with no hidden layers"""
        # Use unified mode with proper parameters
        network = SpectralFractionalNetwork(
            input_dim=10, hidden_dims=[], output_dim=5,
            mode="unified"
        )

        self.assertEqual(network.input_size, 10)
        self.assertEqual(network.hidden_sizes, [])
        self.assertEqual(network.output_size, 5)


class TestBoundedAlphaParameter(unittest.TestCase):
    """Test BoundedAlphaParameter functionality"""

    def test_bounded_alpha_parameter_initialization_default(self):
        """Test BoundedAlphaParameter initialization with default parameters"""
        alpha_param = BoundedAlphaParameter()

        self.assertIsInstance(alpha_param, torch.nn.Module)

    def test_bounded_alpha_parameter_initialization_custom_value(self):
        """Test BoundedAlphaParameter initialization with custom value"""
        alpha_param = BoundedAlphaParameter(0.7)

        self.assertIsInstance(alpha_param, torch.nn.Module)

    def test_bounded_alpha_parameter_forward(self):
        """Test BoundedAlphaParameter forward pass"""
        alpha_param = BoundedAlphaParameter(0.5)

        result = alpha_param.forward()

        self.assertIsInstance(result, torch.Tensor)
        # Use approximate equality for float comparison
        self.assertAlmostEqual(result.item(), 0.5, places=5)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_resolve_activation_module(self):
        """Test _resolve_activation_module function"""
        # Test string activation
        activation = _resolve_activation_module("relu")
        self.assertIsInstance(activation, torch.nn.Module)

        # Test None activation - returns default ReLU
        activation = _resolve_activation_module(None)
        self.assertIsInstance(activation, torch.nn.Module)

        # Test existing module
        existing_module = torch.nn.ReLU()
        activation = _resolve_activation_module(existing_module)
        self.assertEqual(activation, existing_module)

    def test_is_complex_dtype(self):
        """Test _is_complex_dtype function"""
        # Test complex dtype
        complex_tensor = torch.randn(5, dtype=torch.complex64)
        self.assertTrue(_is_complex_dtype(complex_tensor.dtype))

        # Test real dtype
        real_tensor = torch.randn(5, dtype=torch.float32)
        self.assertFalse(_is_complex_dtype(real_tensor.dtype))

    def test_fft_backend_functions(self):
        """Test FFT backend functions"""
        # Test set/get FFT backend
        original_backend = get_fft_backend()
        
        set_fft_backend("numpy")
        self.assertEqual(get_fft_backend(), "numpy")
        
        set_fft_backend("torch")
        self.assertEqual(get_fft_backend(), "torch")
        
        # Restore original backend
        set_fft_backend(original_backend)

    def test_dtype_conversion_functions(self):
        """Test dtype conversion functions"""
        # Test _complex_dtype_for
        complex_dtype = _complex_dtype_for(torch.float32)
        self.assertEqual(complex_dtype, torch.complex64)

        # Test _real_dtype_for
        real_dtype = _real_dtype_for(torch.complex64)
        self.assertEqual(real_dtype, torch.float32)

    def test_backend_resolution(self):
        """Test backend resolution functions"""
        # Test _resolve_backend
        backend = _resolve_backend("torch")
        self.assertIsNotNone(backend)

        # Test _effective_backend
        effective = _effective_backend("torch")
        self.assertIsNotNone(effective)

    def test_validation_functions(self):
        """Test validation functions"""
        # Test _validate_alpha with valid alpha
        valid_alpha = torch.tensor(0.5)
        try:
            _validate_alpha(valid_alpha)
        except ValueError:
            self.fail("_validate_alpha raised ValueError for valid alpha")

        # Test _validate_alpha with invalid alpha
        invalid_alpha = torch.tensor(0.0)
        with self.assertRaises(ValueError):
            _validate_alpha(invalid_alpha)

    def test_tensor_manipulation_functions(self):
        """Test tensor manipulation functions"""
        # Test _normalize_dims
        x = torch.randn(5, 10, 15)
        normalized = _normalize_dims(x, -1)
        self.assertIsInstance(normalized, tuple)

        # Test _ensure_alpha_tensor
        alpha_tensor = _ensure_alpha_tensor(0.5, x)
        self.assertIsInstance(alpha_tensor, torch.Tensor)

    def test_kernel_functions(self):
        """Test kernel-related functions"""
        # Test _frequency_grid
        freqs = _frequency_grid(10, torch.device('cpu'), torch.float32)
        self.assertIsInstance(freqs, torch.Tensor)
        self.assertEqual(freqs.shape, (10,))

        # Test _to_complex
        real_tensor = torch.randn(5)
        complex_tensor = _to_complex(real_tensor, torch.complex64)
        self.assertTrue(torch.is_complex(complex_tensor))

        # Test _reshape_kernel
        kernel = torch.randn(10)
        reshaped = _reshape_kernel(kernel, 2, 0)
        self.assertIsInstance(reshaped, torch.Tensor)


class TestSpectralAutogradEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    def test_spectral_fractional_derivative_edge_cases(self):
        """Test spectral_fractional_derivative edge cases"""
        # Test with very small alpha
        x = torch.randn(10, requires_grad=True)
        result = spectral_fractional_derivative(x, 0.001)
        self.assertIsInstance(result, torch.Tensor)

        # Test with alpha close to 2
        result = spectral_fractional_derivative(x, 1.999)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_layer_edge_cases(self):
        """Test SpectralFractionalLayer edge cases"""
        # Test with different kernel types
        layer = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz")
        self.assertEqual(layer.kernel_type, "riesz")

        # Test with different dimensions
        layer = SpectralFractionalLayer(alpha=0.5, dim=0)
        self.assertEqual(layer.dim, 0)

    def test_network_edge_cases(self):
        """Test network edge cases"""
        # Test with single hidden layer
        network = SpectralFractionalNetwork(
            input_size=5, hidden_sizes=[10], output_size=3
        )
        x = torch.randn(2, 5)
        result = network(x)
        self.assertEqual(result.shape, (2, 3))

        # Test with multiple hidden layers
        network = SpectralFractionalNetwork(
            input_size=5, hidden_sizes=[10, 15, 20], output_size=3
        )
        result = network(x)
        self.assertEqual(result.shape, (2, 3))


class TestSpectralAutogradIntegration(unittest.TestCase):
    """Test integration between components"""

    def test_layer_network_integration(self):
        """Test integration between SpectralFractionalLayer and SpectralFractionalNetwork"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        
        # Test that the network contains the expected components
        self.assertIsInstance(network.spectral_layer, SpectralFractionalLayer)
        self.assertIsInstance(network.activation, torch.nn.Module)
        self.assertIsInstance(network.output_layer, torch.nn.Linear)

    def test_forward_backward_integration(self):
        """Test forward and backward pass integration"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        x = torch.randn(3, 10, requires_grad=True)

        # Forward pass
        result = network(x)
        self.assertEqual(result.shape, (3, 5))

        # Backward pass
        loss = result.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertTrue(x.grad is not None)

    def test_learnable_alpha_integration(self):
        """Test learnable alpha integration"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=True
        )
        
        # Test that alpha is learnable
        self.assertTrue(network.learnable_alpha)

    def test_different_activations(self):
        """Test with different activation functions"""
        activations = ["relu", "tanh", "sigmoid"]
        
        for activation in activations:
            network = SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[20], output_size=5,
                activation=activation
            )
            
            x = torch.randn(3, 10)
            result = network(x)
            self.assertEqual(result.shape, (3, 5))


if __name__ == '__main__':
    unittest.main()
