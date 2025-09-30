"""
Comprehensive tests to achieve 100% coverage for hpfracc/ml/spectral_autograd.py
This file targets all missing lines identified in the coverage report.
"""

import unittest
import torch
import numpy as np
import warnings
from unittest.mock import patch, MagicMock, mock_open

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
    _spectral_fractional_impl,
    _numpy_fft,
    _numpy_ifft,
    robust_fft,
    robust_ifft,
    safe_fft,
    safe_ifft
)


class TestSpectralAutograd100PercentCoverage(unittest.TestCase):
    """Test cases to achieve 100% coverage of spectral_autograd.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Store original backend for cleanup
        self.original_backend = get_fft_backend()

    def tearDown(self):
        """Clean up after tests"""
        # Restore original backend
        set_fft_backend(self.original_backend)

    def test_set_fft_backend_none_backend(self):
        """Test set_fft_backend with None backend (line 93)"""
        with self.assertRaises(ValueError, msg="Backend must be a non-empty string"):
            set_fft_backend(None)

    def test_set_fft_backend_invalid_backend(self):
        """Test set_fft_backend with invalid backend (lines 96-98)"""
        with self.assertRaises(ValueError, msg="Unsupported backend"):
            set_fft_backend("invalid_backend")

    def test_complex_dtype_for_float64(self):
        """Test _complex_dtype_for with float64 (line 116)"""
        result = _complex_dtype_for(torch.float64)
        self.assertEqual(result, torch.complex128)

    def test_complex_dtype_for_complex128(self):
        """Test _complex_dtype_for with complex128 (line 116)"""
        result = _complex_dtype_for(torch.complex128)
        self.assertEqual(result, torch.complex128)

    def test_real_dtype_for_float32(self):
        """Test _real_dtype_for with float32 (line 122)"""
        result = _real_dtype_for(torch.float32)
        self.assertEqual(result, torch.float32)

    def test_real_dtype_for_complex64(self):
        """Test _real_dtype_for with complex64 (line 122)"""
        result = _real_dtype_for(torch.complex64)
        self.assertEqual(result, torch.float32)

    def test_real_dtype_for_float64(self):
        """Test _real_dtype_for with float64 (line 124)"""
        result = _real_dtype_for(torch.float64)
        self.assertEqual(result, torch.float64)

    def test_real_dtype_for_complex128(self):
        """Test _real_dtype_for with complex128 (line 124)"""
        result = _real_dtype_for(torch.complex128)
        self.assertEqual(result, torch.float64)

    def test_real_dtype_for_default(self):
        """Test _real_dtype_for with default case (line 126)"""
        result = _real_dtype_for(torch.int32)
        self.assertEqual(result, torch.float32)

    def test_resolve_backend_invalid(self):
        """Test _resolve_backend with invalid backend (lines 132-134)"""
        with self.assertRaises(ValueError, msg="Unsupported backend"):
            _resolve_backend("invalid")

    def test_effective_backend_default(self):
        """Test _effective_backend with default case (line 139)"""
        result = _effective_backend("unknown")
        self.assertEqual(result, "torch")

    def test_numpy_fft_empty_tensor(self):
        """Test _numpy_fft with empty tensor (lines 150-153)"""
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

    @patch('torch.fft.fft')
    def test_robust_fft_fallback(self, mock_fft):
        """Test robust_fft fallback to NumPy (lines 189-193)"""
        mock_fft.side_effect = Exception("FFT failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = torch.randn(10)
            result = robust_fft(x)
            
            # Should have warned and used NumPy fallback
            self.assertEqual(len(w), 1)
            self.assertIn("PyTorch FFT failed", str(w[0].message))
            self.assertIsInstance(result, torch.Tensor)

    @patch('torch.fft.ifft')
    def test_robust_ifft_fallback(self, mock_ifft):
        """Test robust_ifft fallback to NumPy (lines 197-201)"""
        mock_ifft.side_effect = Exception("IFFT failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = torch.randn(10)
            result = robust_ifft(x)
            
            # Should have warned and used NumPy fallback
            self.assertEqual(len(w), 1)
            self.assertIn("PyTorch IFFT failed", str(w[0].message))
            self.assertIsInstance(result, torch.Tensor)

    @patch('torch.fft.fft')
    def test_safe_fft_auto_fallback(self, mock_fft):
        """Test safe_fft with auto backend fallback (lines 210-218)"""
        mock_fft.side_effect = Exception("FFT failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = torch.randn(10)
            result = safe_fft(x, backend="auto")
            
            # Should have warned and used NumPy fallback
            self.assertEqual(len(w), 1)
            self.assertIn("Torch FFT failed under 'auto' backend", str(w[0].message))
            self.assertIsInstance(result, torch.Tensor)

    @patch('torch.fft.ifft')
    def test_safe_ifft_auto_fallback(self, mock_ifft):
        """Test safe_ifft with auto backend fallback (lines 230-238)"""
        mock_ifft.side_effect = Exception("IFFT failed")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x = torch.randn(10)
            result = safe_ifft(x, backend="auto")
            
            # Should have warned and used NumPy fallback
            self.assertEqual(len(w), 1)
            self.assertIn("Torch IFFT failed under 'auto' backend", str(w[0].message))
            self.assertIsInstance(result, torch.Tensor)

    def test_safe_fft_torch_backend(self):
        """Test safe_fft with torch backend (line 220)"""
        x = torch.randn(10)
        result = safe_fft(x, backend="torch")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_fft_robust_backend(self):
        """Test safe_fft with robust backend (line 222)"""
        x = torch.randn(10)
        result = safe_fft(x, backend="robust")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_ifft_torch_backend(self):
        """Test safe_ifft with torch backend (line 240)"""
        x = torch.randn(10)
        result = safe_ifft(x, backend="torch")
        self.assertIsInstance(result, torch.Tensor)

    def test_safe_ifft_robust_backend(self):
        """Test safe_ifft with robust backend (line 242)"""
        x = torch.randn(10)
        result = safe_ifft(x, backend="robust")
        self.assertIsInstance(result, torch.Tensor)

    def test_normalize_dims_none(self):
        """Test _normalize_dims with None (lines 256-257)"""
        x = torch.randn(5, 10, 15)
        result = _normalize_dims(x, None)
        self.assertEqual(result, (0, 1, 2))

    def test_normalize_dims_iterable(self):
        """Test _normalize_dims with iterable (lines 258-261)"""
        x = torch.randn(5, 10, 15)
        result = _normalize_dims(x, [0, 2])
        self.assertEqual(result, (0, 2))

    def test_normalize_dims_negative_index(self):
        """Test _normalize_dims with negative index (lines 265-266)"""
        x = torch.randn(5, 10, 15)
        result = _normalize_dims(x, -1)
        self.assertEqual(result, (2,))

    def test_normalize_dims_invalid_axis(self):
        """Test _normalize_dims with invalid axis (lines 267-268)"""
        x = torch.randn(5, 10, 15)
        with self.assertRaises(ValueError, msg="Invalid dimension"):
            _normalize_dims(x, 5)

    def test_ensure_alpha_tensor_tensor_input(self):
        """Test _ensure_alpha_tensor with tensor input (lines 273-274)"""
        x = torch.randn(10)
        alpha_tensor = torch.tensor(0.5)
        result = _ensure_alpha_tensor(alpha_tensor, x)
        self.assertIsInstance(result, torch.Tensor)

    def test_validate_alpha_invalid(self):
        """Test _validate_alpha with invalid alpha (line 289)"""
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            _validate_alpha(torch.tensor(0.0))

    def test_frequency_grid(self):
        """Test _frequency_grid function (lines 287-292)"""
        freqs = _frequency_grid(10, torch.device('cpu'), torch.float32)
        self.assertEqual(freqs.shape, (10,))
        self.assertEqual(freqs.dtype, torch.float32)

    def test_build_kernel_from_freqs(self):
        """Test _build_kernel_from_freqs function (lines 293-318)"""
        freqs = torch.linspace(0, 1, 10)
        alpha = 0.5
        kernel = _build_kernel_from_freqs(freqs, alpha, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)
        self.assertEqual(kernel.shape, freqs.shape)

    def test_build_kernel_from_freqs_unknown_type(self):
        """Test _build_kernel_from_freqs with unknown kernel type (line 307)"""
        freqs = torch.linspace(0, 1, 10)
        alpha = 0.5
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _build_kernel_from_freqs(freqs, alpha, "unknown", 1e-6)

    def test_to_complex(self):
        """Test _to_complex function (lines 319-328)"""
        real_tensor = torch.randn(5)
        complex_tensor = _to_complex(real_tensor, torch.complex64)
        self.assertTrue(torch.is_complex(complex_tensor))
        self.assertEqual(complex_tensor.dtype, torch.complex64)

    def test_reshape_kernel(self):
        """Test _reshape_kernel function (lines 329-336)"""
        kernel = torch.randn(10)
        reshaped = _reshape_kernel(kernel, 2, 0)
        self.assertIsInstance(reshaped, torch.Tensor)

    def test_get_fractional_kernel(self):
        """Test _get_fractional_kernel function (lines 337-368)"""
        alpha = 0.5
        n = 10
        kernel = _get_fractional_kernel(alpha, n, "riesz", 1e-6)
        self.assertIsInstance(kernel, torch.Tensor)

    def test_get_fractional_kernel_unknown_type(self):
        """Test _get_fractional_kernel with unknown kernel type (line 348)"""
        alpha = 0.5
        n = 10
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _get_fractional_kernel(alpha, n, "unknown", 1e-6)

    def test_apply_fractional_along_dim(self):
        """Test _apply_fractional_along_dim function (lines 369-400)"""
        x = torch.randn(5, 10)
        alpha = 0.5
        result = _apply_fractional_along_dim(x, alpha, "riesz", -1, "ortho", "auto", 1e-6)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_impl(self):
        """Test _spectral_fractional_impl function (lines 401-433)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = _spectral_fractional_impl(x, alpha, "riesz", -1, "ortho")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, x.shape)

    def test_spectral_fractional_impl_unknown_kernel(self):
        """Test _spectral_fractional_impl with unknown kernel (line 411)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        with self.assertRaises(ValueError, msg="Unknown kernel type"):
            _spectral_fractional_impl(x, alpha, "unknown", -1, "ortho")

    def test_spectral_fractional_derivative_apply(self):
        """Test SpectralFractionalDerivative.apply method (lines 447-457)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = SpectralFractionalDerivative.apply(x, alpha, "riesz", -1, "ortho")
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_function_forward(self):
        """Test SpectralFractionalFunction.forward method (lines 463-467)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        func = SpectralFractionalFunction()
        result = func(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_derivative_function(self):
        """Test spectral_fractional_derivative function (lines 470-479)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = spectral_fractional_derivative(x, alpha)
        self.assertIsInstance(result, torch.Tensor)

    def test_fractional_derivative_function(self):
        """Test fractional_derivative function (lines 490-499)"""
        x = torch.randn(10, requires_grad=True)
        alpha = 0.5
        result = fractional_derivative(x, alpha)
        self.assertIsInstance(result, torch.Tensor)

    def test_resolve_activation_module_string(self):
        """Test _resolve_activation_module with string (lines 517-528)"""
        activation = _resolve_activation_module("relu")
        self.assertIsInstance(activation, torch.nn.Module)

    def test_resolve_activation_module_none(self):
        """Test _resolve_activation_module with None (lines 526-528)"""
        activation = _resolve_activation_module(None)
        self.assertIsInstance(activation, torch.nn.Module)

    def test_resolve_activation_module_module(self):
        """Test _resolve_activation_module with existing module (lines 526-528)"""
        existing_module = torch.nn.ReLU()
        activation = _resolve_activation_module(existing_module)
        self.assertEqual(activation, existing_module)

    def test_spectral_fractional_layer_initialization(self):
        """Test SpectralFractionalLayer initialization (lines 531-614)"""
        layer = SpectralFractionalLayer(alpha=0.5)
        self.assertIsInstance(layer, torch.nn.Module)

    def test_spectral_fractional_layer_alpha_validation(self):
        """Test SpectralFractionalLayer alpha validation (lines 566-570)"""
        with self.assertRaises(ValueError, msg="Alpha must be in (0, 2)"):
            SpectralFractionalLayer(alpha=0.0)

    def test_spectral_fractional_layer_forward(self):
        """Test SpectralFractionalLayer forward method (lines 593-599)"""
        layer = SpectralFractionalLayer(alpha=0.5)
        x = torch.randn(5, 10, requires_grad=True)
        result = layer(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_layer_get_alpha(self):
        """Test SpectralFractionalLayer get_alpha method (lines 611-614)"""
        layer = SpectralFractionalLayer(alpha=0.5)
        alpha = layer.get_alpha()
        self.assertIsInstance(alpha, (float, torch.Tensor))

    def test_spectral_fractional_network_initialization(self):
        """Test SpectralFractionalNetwork initialization (lines 615-757)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        self.assertIsInstance(network, torch.nn.Module)

    def test_spectral_fractional_network_coverage_mode_error(self):
        """Test SpectralFractionalNetwork coverage mode error (lines 689-690)"""
        with self.assertRaises(IndexError, msg="hidden_sizes must be non-empty for coverage mode"):
            SpectralFractionalNetwork(
                input_size=10, hidden_sizes=[], output_size=5, mode="coverage"
            )

    def test_spectral_fractional_network_forward(self):
        """Test SpectralFractionalNetwork forward method (lines 739-745)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5
        )
        x = torch.randn(3, 10, requires_grad=True)
        result = network(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_spectral_fractional_network_learnable_alpha(self):
        """Test SpectralFractionalNetwork with learnable alpha (lines 751-757)"""
        network = SpectralFractionalNetwork(
            input_size=10, hidden_sizes=[20], output_size=5,
            learnable_alpha=True
        )
        self.assertTrue(network.learnable_alpha)

    def test_bounded_alpha_parameter_initialization(self):
        """Test BoundedAlphaParameter initialization (lines 759-796)"""
        alpha_param = BoundedAlphaParameter(0.5)
        self.assertIsInstance(alpha_param, torch.nn.Module)

    def test_bounded_alpha_parameter_forward(self):
        """Test BoundedAlphaParameter forward method (lines 770-796)"""
        alpha_param = BoundedAlphaParameter(0.5)
        result = alpha_param.forward()
        self.assertIsInstance(result, torch.Tensor)

    def test_create_fractional_layer(self):
        """Test create_fractional_layer function (lines 797-819)"""
        layer = create_fractional_layer("spectral", alpha=0.5)
        self.assertIsInstance(layer, torch.nn.Module)

    def test_create_fractional_layer_unknown_type(self):
        """Test create_fractional_layer with unknown type (lines 808-819)"""
        with self.assertRaises(ValueError, msg="Unknown layer type"):
            create_fractional_layer("unknown", alpha=0.5)

    def test_benchmark_backends(self):
        """Test benchmark_backends function (lines 820-857)"""
        # Test with tensor inputs instead of lists
        x = torch.randn(10)
        alphas = torch.tensor([0.3, 0.7])
        results = benchmark_backends(x, alphas)
        self.assertIsInstance(results, dict)

    # Note: Original functions are not available in the current version
    # They may have been removed or renamed


if __name__ == '__main__':
    unittest.main()
