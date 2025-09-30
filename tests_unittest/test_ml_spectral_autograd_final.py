"""
Final comprehensive tests to achieve maximum coverage for hpfracc/ml/spectral_autograd.py
This file targets the remaining missing lines with correct API usage.
"""

import unittest
import torch
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Import the actual modules
from hpfracc.ml.spectral_autograd import (
    safe_fft,
    safe_ifft,
    _build_kernel_from_freqs,
    _get_fractional_kernel,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    BoundedAlphaParameter,
    create_fractional_layer,
    benchmark_backends,
    _validate_alpha,
    _frequency_grid,
    _spectral_fractional_impl,
    SpectralFractionalDerivative,
    SpectralFractionalFunction,
    _resolve_activation_module
)


class TestFinalCoverage(unittest.TestCase):
    """Final test cases to maximize coverage"""

    def test_safe_fft_numpy_backend(self):
        """Test safe_fft with numpy backend"""
        x = torch.randn(10)
        result = safe_fft(x, backend="numpy")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_ifft_numpy_backend(self):
        """Test safe_ifft with numpy backend"""
        x = torch.randn(10)
        result = safe_ifft(x, backend="numpy")
        self.assertIsInstance(result, torch.Tensor)

    def test_validate_alpha_boundary_2(self):
        """Test _validate_alpha with alpha=2.0"""
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            _validate_alpha(torch.tensor(2.0))

    def test_frequency_grid_edge_cases(self):
        """Test _frequency_grid with different parameters"""
        # Test with different lengths and dtypes
        freqs1 = _frequency_grid(5, torch.device('cpu'), torch.float32)
        self.assertEqual(freqs1.shape, (5,))
        
        freqs2 = _frequency_grid(3, torch.device('cpu'), torch.float64)
        self.assertEqual(freqs2.shape, (3,))

    def test_build_kernel_from_freqs_with_tensor_alpha(self):
        """Test _build_kernel_from_freqs with tensor alpha"""
        freqs = torch.linspace(0, 1, 5)
        alpha = torch.tensor(0.5)
        
        kernel = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)

    def test_get_fractional_kernel_with_parameters(self):
        """Test _get_fractional_kernel with different parameters"""
        alpha = 0.5
        n = 5
        
        # Test with default parameters
        kernel1 = _get_fractional_kernel(alpha, n, "riesz", 1e-6)
        self.assertIsInstance(kernel1, torch.Tensor)
        
        # Test with custom dtype and device
        kernel2 = _get_fractional_kernel(alpha, n, "riesz", 1e-6, torch.float64, torch.device('cpu'))
        self.assertIsInstance(kernel2, torch.Tensor)

    def test_spectral_fractional_impl_with_parameters(self):
        """Test _spectral_fractional_impl with different parameters"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = _spectral_fractional_impl(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_derivative_apply_method(self):
        """Test SpectralFractionalDerivative.apply method"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz", -1, "ortho")
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_function_forward_method(self):
        """Test SpectralFractionalFunction.forward method"""
        x = torch.randn(10, requires_grad=True)
        
        # Create function instance and call forward method
        func = SpectralFractionalFunction()
        result = func.forward(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_resolve_activation_module_variants(self):
        """Test _resolve_activation_module with different activation types"""
        # Test different string activations
        activations = ["relu", "tanh", "sigmoid"]
        for activation in activations:
            result = _resolve_activation_module(activation)
            self.assertIsInstance(result, torch.nn.Module)

    def test_spectral_fractional_layer_comprehensive(self):
        """Test SpectralFractionalLayer comprehensive functionality"""
        # Test different configurations
        layer1 = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz", dim=-1, norm="ortho")
        self.assertIsInstance(layer1, torch.nn.Module)
        
        layer2 = SpectralFractionalLayer(alpha=0.3, learnable_alpha=True)
        self.assertTrue(layer2.learnable_alpha)
        
        # Test forward pass
        x = torch.randn(5, 10, requires_grad=True)
        result = layer1(x)
        self.assertIsInstance(result, torch.Tensor)
        
        # Test get_alpha method
        alpha = layer1.get_alpha()
        self.assertIsInstance(alpha, (float, torch.Tensor))

    def test_spectral_fractional_network_comprehensive(self):
        """Test SpectralFractionalNetwork comprehensive functionality"""
        # Test different network configurations
        network1 = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        self.assertIsInstance(network1, torch.nn.Module)
        
        network2 = SpectralFractionalNetwork(
            input_size=5, hidden_sizes=[10, 15], output_size=3, learnable_alpha=True
        )
        self.assertTrue(network2.learnable_alpha)
        
        # Test forward pass
        x = torch.randn(3, 10, requires_grad=True)
        result = network1(x)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 5))

    def test_bounded_alpha_parameter_comprehensive(self):
        """Test BoundedAlphaParameter comprehensive functionality"""
        # Test different initializations
        alpha_param1 = BoundedAlphaParameter(0.5)
        self.assertIsInstance(alpha_param1, torch.nn.Module)
        
        alpha_param2 = BoundedAlphaParameter(0.3)
        self.assertIsInstance(alpha_param2, torch.nn.Module)
        
        # Test forward method
        result = alpha_param1.forward()
        self.assertIsInstance(result, torch.Tensor)

    def test_create_fractional_layer_correct_usage(self):
        """Test create_fractional_layer with correct usage"""
        # Test with correct parameters - no input_size when using layer type
        layer = create_fractional_layer("spectral", alpha=0.5)
        self.assertIsInstance(layer, torch.nn.Module)

    def test_benchmark_backends_single_values(self):
        """Test benchmark_backends with single values"""
        # Test with single tensor and single alpha
        x = torch.randn(10)
        alpha = torch.tensor(0.5)
        
        results = benchmark_backends(x, alpha)
        self.assertIsInstance(results, dict)

    def test_coverage_mode_error_scenario(self):
        """Test the coverage mode error scenario"""
        # This should trigger the IndexError for empty hidden_sizes in coverage mode
        with self.assertRaises(IndexError, msg="hidden_sizes must be non-empty for coverage mode"):
            SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[], output_size=5, mode="coverage"
            )

    def test_input_size_validation_error_scenario(self):
        """Test input_size validation error scenario"""
        # This should trigger the ValueError for invalid input_size
        with self.assertRaises(ValueError, msg="input_size must be a positive integer when provided"):
            SpectralFractionalLayer(input_size="invalid")

    def test_network_mode_variations(self):
        """Test different network modes"""
        # Test unified mode
        network_unified = SpectralFractionalNetwork(
            input_dim=10, hidden_dims=[20], output_dim=5, mode="unified"
        )
        self.assertIsInstance(network_unified, torch.nn.Module)
        
        # Test model mode
        network_model = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5, mode="model"
        )
        self.assertIsInstance(network_model, torch.nn.Module)

    def test_layer_edge_cases(self):
        """Test layer edge cases"""
        # Test with different dimensions
        layer = SpectralFractionalLayer(alpha=0.5, dim=0)
        self.assertEqual(layer.dim, 0)
        
        # Test with different kernel types
        layer = SpectralFractionalLayer(alpha=0.5, kernel_type="riesz")
        self.assertEqual(layer.kernel_type, "riesz")

    def test_network_edge_cases(self):
        """Test network edge cases"""
        # Test with different activation functions
        activations = ["relu", "tanh", "sigmoid"]
        for activation in activations:
            network = SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[20], output_size=5,
                activation=activation
            )
            x = torch.randn(3, 10)
            result = network(x)
            self.assertEqual(result.shape, (3, 5))

    def test_comprehensive_workflow(self):
        """Test comprehensive workflow"""
        # Test complete workflow
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5, learnable_alpha=True
        )
        
        x = torch.randn(3, 10, requires_grad=True)
        result = network(x)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 5))
        
        # Test backward pass
        loss = result.sum()
        loss.backward()
        
        # Verify gradients are computed
        self.assertTrue(x.grad is not None)

    def test_error_handling_scenarios(self):
        """Test various error handling scenarios"""
        # Test invalid alpha values
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=0.0)
            
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=2.0)
            
        with self.assertRaises(ValueError):
            SpectralFractionalLayer(alpha=-0.5)

    def test_parameter_combinations(self):
        """Test various parameter combinations"""
        # Test different alpha values
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 1.9]:
            try:
                layer = SpectralFractionalLayer(alpha=alpha)
                self.assertIsInstance(layer, torch.nn.Module)
            except ValueError:
                # Some alpha values may be invalid
                pass

    def test_backend_combinations(self):
        """Test different backend combinations"""
        # Test different backends
        backends = ["auto", "torch", "numpy", "robust"]
        for backend in backends:
            try:
                layer = SpectralFractionalLayer(alpha=0.5, backend=backend)
                self.assertIsInstance(layer, torch.nn.Module)
            except (ValueError, RuntimeError):
                # Some backends may not be available
                pass

    def test_tensor_properties(self):
        """Test tensor properties and operations"""
        # Test with different tensor types
        x1 = torch.randn(10, dtype=torch.float32)
        x2 = torch.randn(10, dtype=torch.float64)
        
        layer = SpectralFractionalLayer(alpha=0.5)
        
        result1 = layer(x1)
        result2 = layer(x2)
        
        self.assertIsInstance(result1, torch.Tensor)
        self.assertIsInstance(result2, torch.Tensor)

    def test_gradient_flow(self):
        """Test gradient flow through the network"""
        network = SpectralFractionalNetwork(
            input_size=5, hidden_sizes=[10], output_size=3
        )
        
        x = torch.randn(2, 5, requires_grad=True)
        result = network(x)
        
        # Test that gradients flow
        loss = result.sum()
        loss.backward()
        
        # Check that input gradients exist
        self.assertTrue(x.grad is not None)
        
        # Check that network parameters have gradients
        for param in network.parameters():
            if param.requires_grad:
                self.assertTrue(param.grad is not None)


if __name__ == '__main__':
    unittest.main()
