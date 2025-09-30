"""
Additional tests to cover the remaining missing lines in spectral_autograd.py
Targeting the specific missing lines identified in coverage report.
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
    benchmark_backends
)


class TestMissingLinesCoverage(unittest.TestCase):
    """Test cases to cover the remaining missing lines"""

    def test_safe_fft_numpy_backend(self):
        """Test safe_fft with numpy backend (line 223)"""
        x = torch.randn(10)
        result = safe_fft(x, backend="numpy")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_ifft_numpy_backend(self):
        """Test safe_ifft with numpy backend (line 243)"""
        x = torch.randn(10)
        result = safe_ifft(x, backend="numpy")
        self.assertIsInstance(result, torch.Tensor)

    def test_validate_alpha_boundary_cases(self):
        """Test _validate_alpha with boundary cases (line 289)"""
        # Test alpha = 2.0 (should raise error)
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            from hpfracc.ml.spectral_autograd import _validate_alpha
            _validate_alpha(torch.tensor(2.0))

    def test_frequency_grid_edge_cases(self):
        """Test _frequency_grid edge cases (lines 300, 308-309, 311-315)"""
        from hpfracc.ml.spectral_autograd import _frequency_grid
        
        # Test with different devices and dtypes
        freqs1 = _frequency_grid(5, torch.device('cpu'), torch.float32)
        self.assertEqual(freqs1.shape, (5,))
        
        freqs2 = _frequency_grid(3, torch.device('cpu'), torch.float64)
        self.assertEqual(freqs2.shape, (3,))

    def test_build_kernel_from_freqs_edge_cases(self):
        """Test _build_kernel_from_freqs edge cases (lines 321, 324, 332)"""
        freqs = torch.linspace(0, 1, 5)
        alpha = 0.5
        
        # Test different kernel types and edge cases
        kernel1 = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-6)
        self.assertIsInstance(kernel1, torch.Tensor)
        
        # Test with different epsilon values
        kernel2 = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-3)
        self.assertIsInstance(kernel2, torch.Tensor)

    def test_get_fractional_kernel_edge_cases(self):
        """Test _get_fractional_kernel edge cases (lines 348, 359-360, 379)"""
        alpha = 0.5
        n = 5
        
        # Test with different parameters
        kernel1 = _get_fractional_kernel(alpha, n, "riesz", 1e-6)
        self.assertIsInstance(kernel1, torch.Tensor)
        
        # Test with different kernel types
        kernel2 = _get_fractional_kernel(alpha, n, "riesz", 1e-6, torch.float64, torch.device('cpu'))
        self.assertIsInstance(kernel2, torch.Tensor)

    def test_spectral_fractional_impl_edge_cases(self):
        """Test _spectral_fractional_impl edge cases (line 413)"""
        from hpfracc.ml.spectral_autograd import _spectral_fractional_impl
        
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        # Test with different parameters
        result = _spectral_fractional_impl(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_derivative_apply_edge_cases(self):
        """Test SpectralFractionalDerivative.apply edge cases (lines 463, 467)"""
        from hpfracc.ml.spectral_autograd import SpectralFractionalDerivative
        
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        
        # Test apply method
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz", -1, "ortho")
        self.assertIsInstance(result, torch.Tensor)

    def test_resolve_activation_module_edge_cases(self):
        """Test _resolve_activation_module edge cases (lines 522-528)"""
        from hpfracc.ml.spectral_autograd import _resolve_activation_module
        
        # Test different activation types
        activation1 = _resolve_activation_module("relu")
        self.assertIsInstance(activation1, torch.nn.Module)
        
        activation2 = _resolve_activation_module("tanh")
        self.assertIsInstance(activation2, torch.nn.Module)
        
        activation3 = _resolve_activation_module("sigmoid")
        self.assertIsInstance(activation3, torch.nn.Module)

    def test_spectral_fractional_layer_edge_cases(self):
        """Test SpectralFractionalLayer edge cases (lines 555, 557, 566, 583-585, 593, 599, 611)"""
        # Test with different parameters
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

    def test_spectral_fractional_network_edge_cases(self):
        """Test SpectralFractionalNetwork edge cases (lines 651, 658, 665-668, 686-701, 707-708, 716, 739-745, 751)"""
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

    def test_bounded_alpha_parameter_edge_cases(self):
        """Test BoundedAlphaParameter edge cases (lines 770, 847-850, 859, 863, 872, 881, 892, 911, 944-953)"""
        # Test different initializations
        alpha_param1 = BoundedAlphaParameter(0.5)
        self.assertIsInstance(alpha_param1, torch.nn.Module)
        
        alpha_param2 = BoundedAlphaParameter(0.3)
        self.assertIsInstance(alpha_param2, torch.nn.Module)
        
        # Test forward method
        result = alpha_param1.forward()
        self.assertIsInstance(result, torch.Tensor)

    def test_create_fractional_layer_edge_cases(self):
        """Test create_fractional_layer edge cases (line 911)"""
        # Test different layer types
        layer = create_fractional_layer("spectral", alpha=0.5)
        self.assertIsInstance(layer, torch.nn.Module)

    def test_benchmark_backends_edge_cases(self):
        """Test benchmark_backends edge cases (lines 944-953)"""
        # Test with different inputs
        x = torch.randn(10)
        alphas = torch.tensor([0.3, 0.7])
        
        results = benchmark_backends(x, alphas)
        self.assertIsInstance(results, dict)
        
        # Test with single values
        results2 = benchmark_backends(torch.randn(5), torch.tensor([0.5]))
        self.assertIsInstance(results2, dict)

    def test_coverage_mode_error_path(self):
        """Test the coverage mode error path (lines 689-690)"""
        # This should trigger the IndexError for empty hidden_sizes in coverage mode
        with self.assertRaises(IndexError, msg="hidden_sizes must be non-empty for coverage mode"):
            SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[], output_size=5, mode="coverage"
            )

    def test_input_size_validation_error(self):
        """Test input_size validation error (lines 707-708)"""
        # This should trigger the ValueError for invalid input_size
        with self.assertRaises(ValueError, msg="input_size must be a positive integer when provided"):
            SpectralFractionalLayer(input_size="invalid")

    def test_comprehensive_integration(self):
        """Test comprehensive integration scenarios"""
        # Test a complete workflow
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


if __name__ == '__main__':
    unittest.main()
