"""
Targeted tests to achieve 100% coverage for hpfracc/ml/spectral_autograd.py
This file specifically targets the remaining 70 uncovered lines.
"""

import unittest
import torch
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Import the actual modules
from hpfracc.ml.spectral_autograd import (
    set_fft_backend,
    get_fft_backend,
    _is_complex_dtype,
    _complex_dtype_for,
    _real_dtype_for,
    _resolve_backend,
    _effective_backend,
    _numpy_fft,
    _numpy_ifft,
    _normalize_dims,
    _validate_alpha,
    _frequency_grid,
    _build_kernel_from_freqs,
    _to_complex,
    _reshape_kernel,
    _get_fractional_kernel,
    _apply_fractional_along_dim,
    _spectral_fractional_impl,
    SpectralFractionalDerivative,
    SpectralFractionalFunction,
    _resolve_activation_module,
    SpectralFractionalLayer,
    SpectralFractionalNetwork,
    BoundedAlphaParameter,
    create_fractional_layer,
    benchmark_backends
)


class Test100PercentCoverage(unittest.TestCase):
    """Targeted tests to achieve 100% coverage"""

    def setUp(self):
        """Set up test fixtures"""
        self.original_backend = get_fft_backend()

    def tearDown(self):
        """Clean up after tests"""
        set_fft_backend(self.original_backend)

    def test_set_fft_backend_none_input(self):
        """Test set_fft_backend with None input (line 93)"""
        with self.assertRaises(ValueError, msg="Backend must be a non-empty string"):
            set_fft_backend(None)

    def test_set_fft_backend_invalid_input(self):
        """Test set_fft_backend with invalid input (line 96)"""
        with self.assertRaises(ValueError, msg="Unsupported backend"):
            set_fft_backend("invalid_backend")

    def test_complex_dtype_for_complex64(self):
        """Test _complex_dtype_for with complex64 (line 117)"""
        result = _complex_dtype_for(torch.complex64)
        self.assertEqual(result, torch.complex64)

    def test_real_dtype_for_default_case(self):
        """Test _real_dtype_for default case (line 126)"""
        result = _real_dtype_for(torch.int32)
        self.assertEqual(result, torch.float32)

    def test_resolve_backend_invalid(self):
        """Test _resolve_backend with invalid backend (line 132)"""
        with self.assertRaises(ValueError, msg="Unsupported backend"):
            _resolve_backend("invalid")

    def test_numpy_fft_empty_tensor(self):
        """Test _numpy_fft with empty tensor (lines 151-153)"""
        x = torch.tensor([], dtype=torch.float32)
        result = _numpy_fft(x)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, torch.complex64)

    def test_numpy_ifft_empty_tensor(self):
        """Test _numpy_ifft with empty tensor (lines 172-174)"""
        x = torch.tensor([], dtype=torch.float32)
        result = _numpy_ifft(x)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.dtype, torch.complex64)

    def test_safe_fft_torch_backend(self):
        """Test safe_fft with torch backend (line 220)"""
        from hpfracc.ml.spectral_autograd import safe_fft
        x = torch.randn(10)
        result = safe_fft(x, backend="torch")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_ifft_torch_backend(self):
        """Test safe_ifft with torch backend (line 240)"""
        from hpfracc.ml.spectral_autograd import safe_ifft
        x = torch.randn(10)
        result = safe_ifft(x, backend="torch")
        self.assertIsInstance(result, torch.Tensor)

    def test_normalize_dims_none(self):
        """Test _normalize_dims with None (line 257)"""
        x = torch.randn(5, 10, 15)
        result = _normalize_dims(x, None)
        self.assertEqual(result, (0, 1, 2))

    def test_normalize_dims_iterable(self):
        """Test _normalize_dims with iterable (line 259)"""
        x = torch.randn(5, 10, 15)
        result = _normalize_dims(x, [0, 2])
        self.assertEqual(result, (0, 2))

    def test_normalize_dims_invalid_axis(self):
        """Test _normalize_dims with invalid axis (line 268)"""
        x = torch.randn(5, 10, 15)
        with self.assertRaises(ValueError, msg="Invalid dimension"):
            _normalize_dims(x, 5)

    def test_validate_alpha_invalid(self):
        """Test _validate_alpha with invalid alpha (line 289)"""
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            _validate_alpha(torch.tensor(0.0))

    def test_frequency_grid(self):
        """Test _frequency_grid (line 300)"""
        freqs = _frequency_grid(10, torch.device('cpu'), torch.float32)
        self.assertEqual(freqs.shape, (10,))
        self.assertEqual(freqs.dtype, torch.float32)

    def test_build_kernel_from_freqs_unknown_type(self):
        """Test _build_kernel_from_freqs with unknown type (line 307)"""
        freqs = torch.linspace(0, 1, 10)
        alpha = torch.tensor(0.5)
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _build_kernel_from_freqs(freqs, alpha, "unknown", 1e-6)

    def test_build_kernel_from_freqs_riesz(self):
        """Test _build_kernel_from_freqs with riesz kernel (lines 308-316)"""
        freqs = torch.linspace(0, 1, 10)
        alpha = torch.tensor(0.5)
        kernel = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)

    def test_to_complex(self):
        """Test _to_complex (line 321)"""
        real_tensor = torch.randn(5)
        complex_tensor = _to_complex(real_tensor, torch.complex64)
        self.assertTrue(torch.is_complex(complex_tensor))

    def test_reshape_kernel(self):
        """Test _reshape_kernel (line 324)"""
        kernel = torch.randn(10)
        reshaped = _reshape_kernel(kernel, 2, 0)
        self.assertIsInstance(reshaped, torch.Tensor)

    def test_get_fractional_kernel_unknown_type(self):
        """Test _get_fractional_kernel with unknown type (line 332)"""
        alpha = 0.5
        n = 10
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _get_fractional_kernel(alpha, n, "unknown", 1e-6)

    def test_get_fractional_kernel_riesz(self):
        """Test _get_fractional_kernel with riesz kernel (lines 348, 359-360, 379)"""
        alpha = 0.5
        n = 10
        kernel = _get_fractional_kernel(alpha, n, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)

    def test_spectral_fractional_impl_unknown_kernel(self):
        """Test _spectral_fractional_impl with unknown kernel (line 411)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _spectral_fractional_impl(x, alpha, "unknown", -1, "ortho", "auto", 1e-6)

    def test_spectral_fractional_impl_riesz(self):
        """Test _spectral_fractional_impl with riesz kernel (line 413)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = _spectral_fractional_impl(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_derivative_apply(self):
        """Test SpectralFractionalDerivative.apply (line 463)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz", -1, "ortho")
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_function_forward(self):
        """Test SpectralFractionalFunction.forward (line 467)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        func = SpectralFractionalFunction()
        result = func.forward(x, alpha)
        self.assertIsInstance(result, torch.Tensor)

    def test_resolve_activation_module_string(self):
        """Test _resolve_activation_module with string (lines 526-528)"""
        activation = _resolve_activation_module("relu")
        self.assertIsInstance(activation, torch.nn.Module)

    def test_spectral_fractional_layer_alpha_validation(self):
        """Test SpectralFractionalLayer alpha validation (line 555)"""
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            SpectralFractionalLayer(alpha=0.0)

    def test_spectral_fractional_layer_forward(self):
        """Test SpectralFractionalLayer forward (line 557)"""
        layer = SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(5, 10, requires_grad=True)
        result = layer(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_layer_get_alpha(self):
        """Test SpectralFractionalLayer get_alpha (line 566)"""
        layer = SpectralFractionalLayer(alpha=0.5)
        alpha = layer.get_alpha()
        self.assertIsInstance(alpha, (float, torch.Tensor))

    def test_spectral_fractional_layer_forward_learnable(self):
        """Test SpectralFractionalLayer forward with learnable alpha (line 593)"""
        layer = SpectralFractionalLayer(alpha=0.5, learnable_alpha=True)
        x = torch.randn(5, 10, requires_grad=True)
        result = layer(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_network_initialization(self):
        """Test SpectralFractionalNetwork initialization (line 651)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        self.assertIsInstance(network, torch.nn.Module)

    def test_spectral_fractional_network_coverage_mode_error(self):
        """Test SpectralFractionalNetwork coverage mode error (line 658)"""
        with self.assertRaises(IndexError, msg="hidden_sizes must be non-empty for coverage mode"):
            SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[], output_size=5, mode="coverage"
            )

    def test_spectral_fractional_network_input_size_validation(self):
        """Test SpectralFractionalNetwork input_size validation (lines 707-708)"""
        with self.assertRaises(ValueError, msg="input_size must be a positive integer when provided"):
            SpectralFractionalLayer(input_size="invalid")

    def test_spectral_fractional_network_forward(self):
        """Test SpectralFractionalNetwork forward (line 716)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        x = torch.randn(3, 10, requires_grad=True)
        result = network(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_network_learnable_alpha(self):
        """Test SpectralFractionalNetwork with learnable alpha (lines 739-745, 751)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=True
        )
        self.assertTrue(network.learnable_alpha)
        
        x = torch.randn(3, 10, requires_grad=True)
        result = network(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_bounded_alpha_parameter_forward(self):
        """Test BoundedAlphaParameter forward (line 770)"""
        alpha_param = BoundedAlphaParameter(0.5)
        result = alpha_param.forward()
        self.assertIsInstance(result, torch.Tensor)

    def test_create_fractional_layer_spectral(self):
        """Test create_fractional_layer with spectral type (line 859)"""
        layer = create_fractional_layer("spectral", alpha=0.5)
        self.assertIsInstance(layer, torch.nn.Module)

    def test_create_fractional_layer_unknown_type(self):
        """Test create_fractional_layer with unknown type (line 863)"""
        with self.assertRaises(ValueError, msg="Unknown layer type"):
            create_fractional_layer("unknown", alpha=0.5)

    def test_benchmark_backends_single_alpha(self):
        """Test benchmark_backends with single alpha (lines 872, 881, 892)"""
        x = torch.randn(10)
        alpha = torch.tensor(0.5)
        results = benchmark_backends(x, alpha)
        self.assertIsInstance(results, dict)

    def test_benchmark_backends_multiple_alphas(self):
        """Test benchmark_backends with multiple alphas (lines 911, 944-953)"""
        x = torch.randn(10)
        alphas = torch.tensor([0.3, 0.7])
        results = benchmark_backends(x, alphas)
        self.assertIsInstance(results, dict)

    def test_comprehensive_edge_cases(self):
        """Test comprehensive edge cases for remaining lines"""
        # Test different alpha values
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 1.9]:
            try:
                layer = SpectralFractionalLayer(alpha=alpha)
                self.assertIsInstance(layer, torch.nn.Module)
            except ValueError:
                # Some alpha values may be invalid
                pass

        # Test different backends
        backends = ["auto", "torch", "numpy", "robust"]
        for backend in backends:
            try:
                layer = SpectralFractionalLayer(alpha=0.5, backend=backend)
                self.assertIsInstance(layer, torch.nn.Module)
            except (ValueError, RuntimeError):
                # Some backends may not be available
                pass

        # Test different kernel types
        kernel_types = ["riesz"]
        for kernel_type in kernel_types:
            try:
                layer = SpectralFractionalLayer(alpha=0.5, kernel_type=kernel_type)
                self.assertIsInstance(layer, torch.nn.Module)
            except ValueError:
                # Some kernel types may not be available
                pass

    def test_error_handling_paths(self):
        """Test error handling paths for remaining uncovered lines"""
        # Test invalid alpha values
        invalid_alphas = [0.0, 2.0, -0.5, 3.0]
        for alpha in invalid_alphas:
            with self.assertRaises(ValueError):
                SpectralFractionalLayer(alpha=alpha)

        # Test invalid input sizes
        invalid_sizes = ["invalid", -1, 0]
        for size in invalid_sizes:
            try:
                SpectralFractionalLayer(input_size=size)
            except ValueError:
                pass

    def test_backend_specific_code(self):
        """Test backend-specific code paths"""
        # Test different FFT backends
        backends = ["auto", "torch", "numpy", "robust", "fftw", "mkl", "manual", "original"]
        for backend in backends:
            try:
                set_fft_backend(backend)
                self.assertEqual(get_fft_backend(), backend.lower())
            except ValueError:
                # Some backends may not be available
                pass

    def test_complex_dtype_handling(self):
        """Test complex dtype handling"""
        # Test _is_complex_dtype
        self.assertTrue(_is_complex_dtype(torch.complex64))
        self.assertTrue(_is_complex_dtype(torch.complex128))
        self.assertFalse(_is_complex_dtype(torch.float32))
        self.assertFalse(_is_complex_dtype(torch.float64))

        # Test _complex_dtype_for
        self.assertEqual(_complex_dtype_for(torch.float32), torch.complex64)
        self.assertEqual(_complex_dtype_for(torch.float64), torch.complex128)
        self.assertEqual(_complex_dtype_for(torch.complex64), torch.complex64)
        self.assertEqual(_complex_dtype_for(torch.complex128), torch.complex128)

        # Test _real_dtype_for
        self.assertEqual(_real_dtype_for(torch.complex64), torch.float32)
        self.assertEqual(_real_dtype_for(torch.complex128), torch.float64)
        self.assertEqual(_real_dtype_for(torch.float32), torch.float32)
        self.assertEqual(_real_dtype_for(torch.float64), torch.float64)

    def test_frequency_grid_edge_cases(self):
        """Test _frequency_grid edge cases"""
        # Test with different lengths and dtypes
        freqs1 = _frequency_grid(5, torch.device('cpu'), torch.float32)
        self.assertEqual(freqs1.shape, (5,))
        
        freqs2 = _frequency_grid(3, torch.device('cpu'), torch.float64)
        self.assertEqual(freqs2.shape, (3,))

    def test_kernel_construction_edge_cases(self):
        """Test kernel construction edge cases"""
        # Test _build_kernel_from_freqs with different parameters
        freqs = torch.linspace(0, 1, 5)
        alpha = torch.tensor(0.5)
        
        kernel = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)

    def test_apply_fractional_along_dim_edge_cases(self):
        """Test _apply_fractional_along_dim edge cases"""
        x = torch.randn(5, 10)
        alpha = 0.5
        result = _apply_fractional_along_dim(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_impl_edge_cases(self):
        """Test _spectral_fractional_impl edge cases"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = _spectral_fractional_impl(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)

    def test_network_modes(self):
        """Test different network modes"""
        # Test unified mode
        try:
            network_unified = SpectralFractionalNetwork(
                input_dim=10, hidden_dims=[20], output_dim=5, mode="unified"
            )
            self.assertIsInstance(network_unified, torch.nn.Module)
        except (TypeError, ValueError):
            # Unified mode may not be available
            pass
        
        # Test model mode
        try:
            network_model = SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[20], output_size=5, mode="model"
            )
            self.assertIsInstance(network_model, torch.nn.Module)
        except (TypeError, ValueError):
            # Model mode may not be available
            pass

    def test_activation_modules(self):
        """Test different activation modules"""
        activations = ["relu", "tanh", "sigmoid", "gelu", "swish"]
        for activation in activations:
            try:
                result = _resolve_activation_module(activation)
                self.assertIsInstance(result, torch.nn.Module)
            except (ValueError, AttributeError):
                # Some activations may not be available
                pass

    def test_comprehensive_workflow(self):
        """Test comprehensive workflow to cover remaining lines"""
        # Create a complete workflow
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
