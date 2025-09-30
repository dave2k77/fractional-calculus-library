"""
Working tests for hpfracc/ml/layers.py

This module provides comprehensive tests for the neural network layers with
fractional calculus integration, focusing on core functionality that can be
tested without complex mocking.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Tuple


class TestLayerConfig(unittest.TestCase):
    """Test the LayerConfig dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.layers import LayerConfig
        from hpfracc.core.definitions import FractionalOrder
        from hpfracc.ml.backends import BackendType
        
        self.LayerConfig = LayerConfig
        self.FractionalOrder = FractionalOrder
        self.BackendType = BackendType

    def test_layer_config_default_values(self):
        """Test LayerConfig with default values"""
        config = self.LayerConfig()
        
        # Test default values
        self.assertEqual(config.method, "RL")
        self.assertTrue(config.use_fractional)
        self.assertEqual(config.activation, "relu")
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.backend, self.BackendType.AUTO)
        self.assertIsNone(config.device)
        self.assertEqual(config.dtype, torch.float32)
        self.assertTrue(config.enable_caching)
        self.assertFalse(config.enable_benchmarking)
        self.assertEqual(config.performance_mode, "balanced")
        
        # Test that fractional_order is set in __post_init__
        self.assertIsNotNone(config.fractional_order)
        self.assertEqual(config.fractional_order.alpha, 0.5)

    def test_layer_config_custom_values(self):
        """Test LayerConfig with custom values"""
        custom_order = self.FractionalOrder(0.7)
        config = self.LayerConfig(
            fractional_order=custom_order,
            method="Caputo",
            use_fractional=False,
            activation="tanh",
            dropout=0.2,
            backend=self.BackendType.TORCH,
            dtype=torch.float64,
            enable_caching=False,
            enable_benchmarking=True,
            performance_mode="speed"
        )
        
        # Test custom values
        self.assertEqual(config.fractional_order, custom_order)
        self.assertEqual(config.method, "Caputo")
        self.assertFalse(config.use_fractional)
        self.assertEqual(config.activation, "tanh")
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.backend, self.BackendType.TORCH)
        self.assertEqual(config.dtype, torch.float64)
        self.assertFalse(config.enable_caching)
        self.assertTrue(config.enable_benchmarking)
        self.assertEqual(config.performance_mode, "speed")

    def test_layer_config_post_init_fractional_order(self):
        """Test that __post_init__ sets default fractional_order"""
        config = self.LayerConfig(fractional_order=None)
        
        # Should be set to default FractionalOrder(0.5)
        self.assertIsNotNone(config.fractional_order)
        self.assertEqual(config.fractional_order.alpha, 0.5)


class TestBackendManager(unittest.TestCase):
    """Test the BackendManager class"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.layers import BackendManager, LayerConfig
        from hpfracc.ml.backends import BackendType
        
        self.BackendManager = BackendManager
        self.LayerConfig = LayerConfig
        self.BackendType = BackendType

    def test_backend_manager_initialization(self):
        """Test BackendManager initialization"""
        manager = self.BackendManager()
        
        # Test attributes
        self.assertIsNotNone(manager.available_backends)
        self.assertIsInstance(manager.available_backends, dict)
        self.assertIsNotNone(manager.performance_cache)
        self.assertIsInstance(manager.performance_cache, dict)
        self.assertIsNotNone(manager.benchmark_results)
        self.assertIsInstance(manager.benchmark_results, dict)
        self.assertIsNotNone(manager.backend_priority)
        self.assertIsInstance(manager.backend_priority, list)

    def test_detect_available_backends(self):
        """Test backend detection"""
        manager = self.BackendManager()
        backends = manager._detect_available_backends()
        
        # Test that all expected backends are in the result
        expected_backends = ['pytorch', 'jax', 'numba', 'robust']
        for backend in expected_backends:
            self.assertIn(backend, backends)
            self.assertIsInstance(backends[backend], bool)
        
        # PyTorch and robust should always be available
        self.assertTrue(backends['pytorch'])
        self.assertTrue(backends['robust'])

    def test_select_optimal_backend_auto_mode(self):
        """Test optimal backend selection with AUTO mode"""
        manager = self.BackendManager()
        config = self.LayerConfig(backend=self.BackendType.AUTO)
        
        # Test with small input
        small_input_shape = (32, 64)
        backend = manager.select_optimal_backend(config, small_input_shape)
        
        self.assertIn(backend, manager.available_backends)
        self.assertTrue(manager.available_backends[backend])

    def test_select_optimal_backend_specific_backend(self):
        """Test optimal backend selection with specific backend"""
        manager = self.BackendManager()
        config = self.LayerConfig(backend=self.BackendType.TORCH)
        
        input_shape = (32, 64)
        backend = manager.select_optimal_backend(config, input_shape)
        
        # Should return the requested backend if available
        self.assertEqual(backend, "torch")

    def test_select_optimal_backend_performance_modes(self):
        """Test backend selection for different performance modes"""
        manager = self.BackendManager()
        
        # Test speed mode
        config_speed = self.LayerConfig(performance_mode="speed")
        backend_speed = manager.select_optimal_backend(config_speed, (1000, 1000))
        self.assertEqual(backend_speed, "pytorch")
        
        # Test memory mode with large input
        config_memory = self.LayerConfig(performance_mode="memory")
        backend_memory = manager.select_optimal_backend(config_memory, (2000, 2000))
        # Should prefer JAX for large inputs if available, otherwise PyTorch
        self.assertIn(backend_memory, ["pytorch", "jax"])
        
        # Test balanced mode
        config_balanced = self.LayerConfig(performance_mode="balanced")
        backend_balanced = manager.select_optimal_backend(config_balanced, (1000, 1000))
        self.assertIn(backend_balanced, ["pytorch", "jax"])


class TestFractionalOps(unittest.TestCase):
    """Test the FractionalOps class"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.layers import FractionalOps, LayerConfig
        
        self.FractionalOps = FractionalOps
        self.LayerConfig = LayerConfig

    def test_fractional_ops_initialization(self):
        """Test FractionalOps initialization"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        # Test attributes
        self.assertEqual(ops.config, config)
        self.assertIsNotNone(ops.cache)

    def test_fractional_ops_initialization_no_caching(self):
        """Test FractionalOps initialization without caching"""
        config = self.LayerConfig(enable_caching=False)
        ops = self.FractionalOps(config)
        
        # Cache should be None when caching is disabled
        self.assertIsNone(ops.cache)

    def test_apply_fractional_derivative_dtype_conversion(self):
        """Test fractional derivative with dtype conversion"""
        config = self.LayerConfig(dtype=torch.float64)
        ops = self.FractionalOps(config)
        
        # Create tensor with different dtype
        x = torch.randn(2, 3, dtype=torch.float32)
        
        with patch.object(ops, '_pytorch_fractional_derivative') as mock_derivative:
            mock_result = torch.randn(2, 3, dtype=torch.float64)
            mock_derivative.return_value = mock_result
            
            result = ops.apply_fractional_derivative(x, 0.5, "RL", "pytorch")
            
            # Should convert dtype before processing
            mock_derivative.assert_called_once()
            self.assertEqual(result, mock_result)

    def test_apply_fractional_derivative_caching(self):
        """Test fractional derivative with caching"""
        config = self.LayerConfig(enable_caching=True)
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_pytorch_fractional_derivative') as mock_derivative:
            mock_result = torch.randn(2, 3)
            mock_derivative.return_value = mock_result
            
            # First call
            result1 = ops.apply_fractional_derivative(x, 0.5, "RL", "pytorch")
            
            # Second call with same parameters should use cache
            result2 = ops.apply_fractional_derivative(x, 0.5, "RL", "pytorch")
            
            # Should only call the derivative function once
            mock_derivative.assert_called_once()
            self.assertEqual(result1, result2)

    def test_apply_fractional_derivative_no_caching(self):
        """Test fractional derivative without caching"""
        config = self.LayerConfig(enable_caching=False)
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_pytorch_fractional_derivative') as mock_derivative:
            mock_result = torch.randn(2, 3)
            mock_derivative.return_value = mock_result
            
            # Multiple calls should all execute the derivative function
            ops.apply_fractional_derivative(x, 0.5, "RL", "pytorch")
            ops.apply_fractional_derivative(x, 0.5, "RL", "pytorch")
            
            # Should be called twice
            self.assertEqual(mock_derivative.call_count, 2)

    def test_apply_fractional_derivative_jax_backend(self):
        """Test fractional derivative with JAX backend"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_jax_fractional_derivative') as mock_jax:
            mock_result = torch.randn(2, 3)
            mock_jax.return_value = mock_result
            
            with patch('hpfracc.ml.layers.JAX_AVAILABLE', True):
                result = ops.apply_fractional_derivative(x, 0.5, "RL", "jax")
                
                mock_jax.assert_called_once_with(x, 0.5, "RL")
                self.assertEqual(result, mock_result)

    def test_apply_fractional_derivative_numba_backend(self):
        """Test fractional derivative with NUMBA backend"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_numba_fractional_derivative') as mock_numba:
            mock_result = torch.randn(2, 3)
            mock_numba.return_value = mock_result
            
            with patch('hpfracc.ml.layers.NUMBA_AVAILABLE', True):
                result = ops.apply_fractional_derivative(x, 0.5, "RL", "numba")
                
                mock_numba.assert_called_once_with(x, 0.5, "RL")
                self.assertEqual(result, mock_result)

    def test_apply_fractional_derivative_fallback(self):
        """Test fractional derivative fallback to PyTorch"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_jax_fractional_derivative') as mock_jax:
            mock_jax.side_effect = Exception("JAX error")
            
            with patch.object(ops, '_pytorch_fractional_derivative') as mock_pytorch:
                mock_result = torch.randn(2, 3)
                mock_pytorch.return_value = mock_result
                
                with patch('hpfracc.ml.layers.JAX_AVAILABLE', True):
                    with patch('hpfracc.ml.layers.warnings.warn') as mock_warn:
                        result = ops.apply_fractional_derivative(x, 0.5, "RL", "jax")
                        
                        # Should warn about fallback
                        mock_warn.assert_called_once()
                        mock_pytorch.assert_called_once()
                        self.assertEqual(result, mock_result)

    def test_pytorch_fractional_derivative(self):
        """Test PyTorch fractional derivative implementation"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch('hpfracc.ml.layers.fractional_derivative') as mock_fractional:
            mock_result = torch.randn(2, 3)
            mock_fractional.return_value = mock_result
            
            result = ops._pytorch_fractional_derivative(x, 0.5, "RL")
            
            mock_fractional.assert_called_once_with(x, 0.5, "RL")
            self.assertEqual(result, mock_result)

    def test_jax_fractional_derivative(self):
        """Test JAX fractional derivative implementation"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch('hpfracc.ml.layers.jnp') as mock_jnp, \
             patch('hpfracc.ml.layers.fractional_derivative') as mock_fractional, \
             patch('hpfracc.ml.layers.np') as mock_np, \
             patch('hpfracc.ml.layers.torch') as mock_torch:
            
            # Mock JAX array conversion
            mock_jax_array = Mock()
            mock_jnp.array.return_value = mock_jax_array
            
            # Mock fractional derivative
            mock_result_jax = Mock()
            mock_fractional.return_value = mock_result_jax
            
            # Mock numpy conversion
            mock_np_array = Mock()
            mock_np.array.return_value = mock_np_array
            
            # Mock torch conversion
            mock_tensor = torch.randn(2, 3)
            mock_torch.from_numpy.return_value.to.return_value.to.return_value = mock_tensor
            
            result = ops._jax_fractional_derivative(x, 0.5, "RL")
            
            # Verify the conversion chain
            mock_jnp.array.assert_called_once()
            mock_fractional.assert_called_once_with(mock_jax_array, 0.5, "RL")
            mock_np.array.assert_called_once_with(mock_result_jax)
            mock_torch.from_numpy.assert_called_once_with(mock_np_array)
            self.assertEqual(result, mock_tensor)

    def test_numba_fractional_derivative(self):
        """Test NUMBA fractional derivative implementation"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(ops, '_numba_frac_derivative_impl') as mock_numba_impl, \
             patch('hpfracc.ml.layers.torch') as mock_torch:
            
            # Mock NUMBA implementation
            mock_np_result = np.random.randn(2, 3)
            mock_numba_impl.return_value = mock_np_result
            
            # Mock torch conversion
            mock_tensor = torch.randn(2, 3)
            mock_torch.from_numpy.return_value.to.return_value.to.return_value = mock_tensor
            
            result = ops._numba_fractional_derivative(x, 0.5, "RL")
            
            # Verify the conversion chain
            mock_numba_impl.assert_called_once()
            mock_torch.from_numpy.assert_called_once_with(mock_np_result)
            self.assertEqual(result, mock_tensor)

    def test_numba_frac_derivative_impl(self):
        """Test NUMBA compiled fractional derivative implementation"""
        config = self.LayerConfig()
        ops = self.FractionalOps(config)
        
        x = np.random.randn(2, 3)
        result = ops._numba_frac_derivative_impl(x, 0.5)
        
        # For now, it just returns the input
        np.testing.assert_array_equal(result, x)


class TestFractionalLayerBase(unittest.TestCase):
    """Test the FractionalLayerBase abstract base class"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.layers import FractionalLayerBase, LayerConfig
        from hpfracc.core.definitions import FractionalOrder
        from hpfracc.ml.backends import BackendType
        
        self.FractionalLayerBase = FractionalLayerBase
        self.LayerConfig = LayerConfig
        self.FractionalOrder = FractionalOrder
        self.BackendType = BackendType

    def create_test_layer(self, config=None):
        """Create a test layer that inherits from FractionalLayerBase"""
        if config is None:
            config = self.LayerConfig()
        
        class TestLayer(self.FractionalLayerBase):
            def forward(self, x):
                return x
        
        return TestLayer(config)

    def test_fractional_layer_base_initialization(self):
        """Test FractionalLayerBase initialization"""
        config = self.LayerConfig()
        layer = self.create_test_layer(config)
        
        # Test attributes
        self.assertEqual(layer.config, config)
        self.assertIsNotNone(layer.backend)
        self.assertIsNotNone(layer.backend_manager)
        self.assertIsNotNone(layer.tensor_ops)
        self.assertIsNotNone(layer.fractional_ops)

    def test_fractional_layer_base_initialization_with_backend(self):
        """Test FractionalLayerBase initialization with specific backend"""
        config = self.LayerConfig()
        backend = self.BackendType.TORCH
        layer = self.create_test_layer(config, backend=backend)
        
        self.assertEqual(layer.backend, backend)

    def test_setup_layer(self):
        """Test layer setup"""
        config = self.LayerConfig(
            use_fractional=True,
            method="Caputo",
            activation="tanh",
            dropout=0.2,
            fractional_order=self.FractionalOrder(0.7)
        )
        layer = self.create_test_layer(config)
        
        # Test setup attributes
        self.assertTrue(layer.use_fractional)
        self.assertEqual(layer.alpha, 0.7)
        self.assertEqual(layer.method, "Caputo")
        self.assertEqual(layer.activation, "tanh")
        self.assertEqual(layer.dropout, 0.2)

    def test_setup_layer_default_fractional_order(self):
        """Test layer setup with default fractional order"""
        config = self.LayerConfig(fractional_order=None)
        layer = self.create_test_layer(config)
        
        # Should use default alpha value
        self.assertEqual(layer.alpha, 0.5)

    def test_setup_layer_activation_functions(self):
        """Test layer setup with different activation functions"""
        # Test ReLU activation
        config_relu = self.LayerConfig(activation="relu")
        layer_relu = self.create_test_layer(config_relu)
        self.assertEqual(layer_relu.activation_fn, torch.nn.functional.relu)
        
        # Test Tanh activation
        config_tanh = self.LayerConfig(activation="tanh")
        layer_tanh = self.create_test_layer(config_tanh)
        self.assertEqual(layer_tanh.activation_fn, torch.tanh)
        
        # Test Sigmoid activation
        config_sigmoid = self.LayerConfig(activation="sigmoid")
        layer_sigmoid = self.create_test_layer(config_sigmoid)
        self.assertEqual(layer_sigmoid.activation_fn, torch.sigmoid)
        
        # Test unknown activation (should default to ReLU)
        config_unknown = self.LayerConfig(activation="unknown")
        layer_unknown = self.create_test_layer(config_unknown)
        self.assertEqual(layer_unknown.activation_fn, torch.nn.functional.relu)

    def test_apply_fractional_derivative_with_fractional_enabled(self):
        """Test applying fractional derivative when enabled"""
        config = self.LayerConfig(use_fractional=True)
        layer = self.create_test_layer(config)
        
        x = torch.randn(2, 3)
        
        with patch.object(layer.fractional_ops, 'apply_fractional_derivative') as mock_frac:
            mock_result = torch.randn(2, 3)
            mock_frac.return_value = mock_result
            
            result = layer.apply_fractional_derivative(x)
            
            mock_frac.assert_called_once()
            self.assertEqual(result, mock_result)

    def test_apply_fractional_derivative_with_fractional_disabled(self):
        """Test applying fractional derivative when disabled"""
        config = self.LayerConfig(use_fractional=False)
        layer = self.create_test_layer(config)
        
        x = torch.randn(2, 3)
        
        result = layer.apply_fractional_derivative(x)
        
        # Should return input unchanged
        self.assertEqual(result, x)

    def test_apply_activation(self):
        """Test applying activation function"""
        config = self.LayerConfig(activation="relu")
        layer = self.create_test_layer(config)
        
        x = torch.tensor([[-1.0, 0.0, 1.0]])
        result = layer.apply_activation(x)
        
        # Should apply ReLU
        expected = torch.tensor([[0.0, 0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_apply_dropout_training(self):
        """Test applying dropout during training"""
        config = self.LayerConfig(dropout=0.5)
        layer = self.create_test_layer(config)
        layer.train()  # Set to training mode
        
        x = torch.randn(2, 3)
        
        with patch('hpfracc.ml.layers.F.dropout') as mock_dropout:
            mock_result = torch.randn(2, 3)
            mock_dropout.return_value = mock_result
            
            result = layer.apply_dropout(x)
            
            mock_dropout.assert_called_once_with(x, p=0.5, training=True)
            self.assertEqual(result, mock_result)

    def test_apply_dropout_eval(self):
        """Test applying dropout during evaluation"""
        config = self.LayerConfig(dropout=0.5)
        layer = self.create_test_layer(config)
        layer.eval()  # Set to evaluation mode
        
        x = torch.randn(2, 3)
        result = layer.apply_dropout(x)
        
        # Should return input unchanged in eval mode
        self.assertEqual(result, x)

    def test_apply_dropout_no_dropout(self):
        """Test applying dropout with zero dropout rate"""
        config = self.LayerConfig(dropout=0.0)
        layer = self.create_test_layer(config)
        layer.train()  # Set to training mode
        
        x = torch.randn(2, 3)
        result = layer.apply_dropout(x)
        
        # Should return input unchanged when dropout is 0
        self.assertEqual(result, x)


if __name__ == '__main__':
    unittest.main()
