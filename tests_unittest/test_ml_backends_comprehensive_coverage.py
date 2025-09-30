"""
Comprehensive coverage tests for hpfracc/ml/backends.py

This module provides extensive tests to achieve maximum coverage of the backend
management system, including edge cases and error conditions.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import warnings
from typing import Any, Dict, List


class TestBackendManagerComprehensiveCoverage(unittest.TestCase):
    """Comprehensive coverage tests for BackendManager"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendManager, BackendType
        self.BackendManager = BackendManager
        self.BackendType = BackendType

    def test_detect_available_backends_with_cuda_torch(self):
        """Test backend detection with CUDA PyTorch"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
             patch('hpfracc.ml.backends.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = True
            
            # Test with force_cpu=False (default)
            manager = self.BackendManager(force_cpu=False)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.TORCH, available)
            
            # Test with force_cpu=True
            manager = self.BackendManager(force_cpu=True)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.TORCH, available)

    def test_detect_available_backends_with_gpu_jax(self):
        """Test backend detection with GPU JAX"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
             patch('hpfracc.ml.backends.jax') as mock_jax:
            
            # Mock GPU device
            mock_gpu_device = Mock()
            mock_gpu_device.__str__ = Mock(return_value="gpu:0")
            mock_jax.devices.return_value = [mock_gpu_device]
            
            # Test with force_cpu=False (default)
            manager = self.BackendManager(force_cpu=False)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.JAX, available)
            
            # Test with force_cpu=True
            manager = self.BackendManager(force_cpu=True)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.JAX, available)

    def test_detect_available_backends_with_gpu_numba(self):
        """Test backend detection with GPU NUMBA"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True), \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            # Mock CUDA availability
            mock_numba.cuda.is_available.return_value = True
            
            # Test with force_cpu=False (default)
            manager = self.BackendManager(force_cpu=False)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.NUMBA, available)
            
            # Test with force_cpu=True
            manager = self.BackendManager(force_cpu=True)
            available = manager._detect_available_backends()
            self.assertIn(self.BackendType.NUMBA, available)

    def test_detect_available_backends_jax_devices_exception(self):
        """Test backend detection when JAX devices() raises exception"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
             patch('hpfracc.ml.backends.jax') as mock_jax:
            
            # Mock jax.devices to raise exception
            mock_jax.devices.side_effect = Exception("Device detection failed")
            
            manager = self.BackendManager()
            available = manager._detect_available_backends()
            
            # Should still include JAX even if device detection fails
            self.assertIn(self.BackendType.JAX, available)

    def test_detect_available_backends_numba_cuda_exception(self):
        """Test backend detection when NUMBA CUDA check raises exception"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True), \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            # Mock numba.cuda.is_available to raise exception
            mock_numba.cuda.is_available.side_effect = Exception("CUDA check failed")
            
            manager = self.BackendManager()
            available = manager._detect_available_backends()
            
            # Should still include NUMBA even if CUDA check fails
            self.assertIn(self.BackendType.NUMBA, available)

    def test_detect_available_backends_numba_no_cuda_attribute(self):
        """Test backend detection when NUMBA has no CUDA attribute"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True), \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            # Remove cuda attribute from numba
            del mock_numba.cuda
            
            manager = self.BackendManager()
            available = manager._detect_available_backends()
            
            # Should still include NUMBA even without CUDA
            self.assertIn(self.BackendType.NUMBA, available)

    def test_select_optimal_backend_auto_fallback(self):
        """Test automatic backend selection with fallback"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Test with only JAX available
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
                
                manager = self.BackendManager(
                    preferred_backend=self.BackendType.AUTO,
                    enable_gpu=True
                )
                self.assertEqual(manager.active_backend, self.BackendType.JAX)
            
            # Test with only NUMBA available
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
                
                manager = self.BackendManager(
                    preferred_backend=self.BackendType.AUTO
                )
                self.assertEqual(manager.active_backend, self.BackendType.NUMBA)

    def test_select_optimal_backend_preferred_not_available(self):
        """Test backend selection when preferred backend is not available"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Only TORCH available, but prefer JAX
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    manager = self.BackendManager(
                        preferred_backend=self.BackendType.JAX
                    )
                    
                    # Should fallback to available backend
                    self.assertEqual(manager.active_backend, self.BackendType.TORCH)
                    # Should issue warning
                    self.assertTrue(len(w) > 0)

    def test_initialize_backend_configs_torch(self):
        """Test backend config initialization for PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float32 = MagicMock()
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Test with CUDA available
            manager = self.BackendManager(force_cpu=False)
            torch_config = manager.backend_configs.get(self.BackendType.TORCH, {})
            
            if self.BackendType.TORCH in manager.available_backends:
                self.assertEqual(torch_config['device'], 'cuda')
                self.assertEqual(torch_config['dtype'], mock_torch.float32)
                self.assertTrue(torch_config['enable_amp'])
            
            # Test with force_cpu=True
            manager = self.BackendManager(force_cpu=True)
            torch_config = manager.backend_configs.get(self.BackendType.TORCH, {})
            
            if self.BackendType.TORCH in manager.available_backends:
                self.assertEqual(torch_config['device'], 'cpu')

    def test_initialize_backend_configs_torch_no_compile(self):
        """Test backend config initialization for PyTorch without compile"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_torch.float32 = MagicMock()
            # Remove compile attribute
            if hasattr(mock_torch, 'compile'):
                delattr(mock_torch, 'compile')
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            torch_config = manager.backend_configs.get(self.BackendType.TORCH, {})
            
            if self.BackendType.TORCH in manager.available_backends:
                self.assertFalse(torch_config['enable_compile'])

    def test_initialize_backend_configs_jax(self):
        """Test backend config initialization for JAX"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_jnp.float32 = MagicMock()
            mock_numba.cuda.is_available.return_value = False
            
            # Test with GPU enabled
            manager = self.BackendManager(enable_gpu=True, force_cpu=False)
            jax_config = manager.backend_configs.get(self.BackendType.JAX, {})
            
            if self.BackendType.JAX in manager.available_backends:
                self.assertEqual(jax_config['device'], 'gpu')
                self.assertEqual(jax_config['dtype'], mock_jnp.float32)
                self.assertTrue(jax_config['enable_jit'])
                self.assertFalse(jax_config['enable_x64'])
                self.assertTrue(jax_config['enable_amp'])
            
            # Test with GPU disabled
            manager = self.BackendManager(enable_gpu=False, force_cpu=True)
            jax_config = manager.backend_configs.get(self.BackendType.JAX, {})
            
            if self.BackendType.JAX in manager.available_backends:
                self.assertEqual(jax_config['device'], 'cpu')

    def test_initialize_backend_configs_numba(self):
        """Test backend config initialization for NUMBA"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.float32 = MagicMock()
            mock_numba.cuda.is_available.return_value = True
            
            # Test with GPU available
            manager = self.BackendManager(enable_gpu=True, force_cpu=False)
            numba_config = manager.backend_configs.get(self.BackendType.NUMBA, {})
            
            if self.BackendType.NUMBA in manager.available_backends:
                self.assertEqual(numba_config['device'], 'gpu')
                self.assertEqual(numba_config['dtype'], mock_numba.float32)
                self.assertTrue(numba_config['enable_jit'])
                self.assertTrue(numba_config['enable_parallel'])
                self.assertTrue(numba_config['enable_fastmath'])
            
            # Test with GPU unavailable
            mock_numba.cuda.is_available.return_value = False
            manager = self.BackendManager(enable_gpu=True, force_cpu=True)
            numba_config = manager.backend_configs.get(self.BackendType.NUMBA, {})
            
            if self.BackendType.NUMBA in manager.available_backends:
                self.assertEqual(numba_config['device'], 'cpu')

    def test_initialize_backend_configs_numba_cuda_exception(self):
        """Test backend config initialization for NUMBA with CUDA exception"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.float32 = MagicMock()
            # Mock CUDA check to raise exception
            mock_numba.cuda.is_available.side_effect = Exception("CUDA error")
            
            manager = self.BackendManager()
            numba_config = manager.backend_configs.get(self.BackendType.NUMBA, {})
            
            if self.BackendType.NUMBA in manager.available_backends:
                self.assertEqual(numba_config['device'], 'cpu')

    def test_to_device_torch(self):
        """Test tensor device placement for PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            # Mock tensor with to method
            mock_tensor = MagicMock()
            mock_tensor.to.return_value = mock_tensor
            
            # Test device placement
            result = manager.to_device(mock_tensor)
            self.assertEqual(result, mock_tensor)
            mock_tensor.to.assert_called_once()
            
            # Test with specific device
            result = manager.to_device(mock_tensor, "cpu")
            self.assertEqual(result, mock_tensor)
            mock_tensor.to.assert_called_with("cpu")

    def test_to_device_jax(self):
        """Test tensor device placement for JAX"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            mock_tensor = MagicMock()
            
            # JAX handles device placement differently
            result = manager.to_device(mock_tensor)
            self.assertEqual(result, mock_tensor)

    def test_to_device_numba(self):
        """Test tensor device placement for NUMBA"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.NUMBA
            
            mock_tensor = MagicMock()
            
            # NUMBA handles device placement differently
            result = manager.to_device(mock_tensor)
            self.assertEqual(result, mock_tensor)

    def test_to_device_unknown_backend(self):
        """Test tensor device placement with unknown backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = Mock()  # Unknown backend
            
            mock_tensor = MagicMock()
            
            with self.assertRaises(RuntimeError) as context:
                manager.to_device(mock_tensor)
            
            self.assertIn("Unknown backend", str(context.exception))

    def test_compile_function_torch_with_compile(self):
        """Test function compilation for PyTorch with compile support"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock torch.compile
            mock_compiled_func = MagicMock()
            mock_torch.compile.return_value = mock_compiled_func
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, mock_compiled_func)
            mock_torch.compile.assert_called_once_with(test_func)

    def test_compile_function_torch_without_compile(self):
        """Test function compilation for PyTorch without compile support"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Remove compile attribute
            if hasattr(mock_torch, 'compile'):
                delattr(mock_torch, 'compile')
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, test_func)

    def test_compile_function_jax_with_jit(self):
        """Test function compilation for JAX with JIT enabled"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock jax.jit
            mock_jitted_func = MagicMock()
            mock_jax.jit.return_value = mock_jitted_func
            
            manager = self.BackendManager(enable_jit=True)
            manager.active_backend = self.BackendType.JAX
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, mock_jitted_func)
            mock_jax.jit.assert_called_once_with(test_func)

    def test_compile_function_jax_without_jit(self):
        """Test function compilation for JAX with JIT disabled"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager(enable_jit=False)
            manager.active_backend = self.BackendType.JAX
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, test_func)

    def test_compile_function_numba_with_jit(self):
        """Test function compilation for NUMBA with JIT enabled"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock numba.jit
            mock_jitted_func = MagicMock()
            mock_numba.jit.return_value = mock_jitted_func
            
            manager = self.BackendManager(enable_jit=True)
            manager.active_backend = self.BackendType.NUMBA
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, mock_jitted_func)
            mock_numba.jit.assert_called_once_with(test_func)

    def test_compile_function_numba_without_jit(self):
        """Test function compilation for NUMBA with JIT disabled"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager(enable_jit=False)
            manager.active_backend = self.BackendType.NUMBA
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, test_func)

    def test_compile_function_unknown_backend(self):
        """Test function compilation with unknown backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = Mock()  # Unknown backend
            
            def test_func(x):
                return x * 2
            
            result = manager.compile_function(test_func)
            self.assertEqual(result, test_func)

    def test_create_tensor_with_integer_dtype_torch(self):
        """Test tensor creation with integer dtype for PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock torch.tensor and torch.long
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor
            mock_torch.long = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            # Create data with integer dtype
            mock_data = MagicMock()
            mock_data.dtype = 'int32'
            
            result = manager.create_tensor(mock_data)
            self.assertEqual(result, mock_tensor)
            
            # Should call torch.tensor with dtype=torch.long
            mock_torch.tensor.assert_called_once()

    def test_create_tensor_with_integer_dtype_jax(self):
        """Test tensor creation with integer dtype for JAX"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock jnp.array and jnp.int32
            mock_array = MagicMock()
            mock_jnp.array.return_value = mock_array
            mock_jnp.int32 = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            # Create data with integer dtype
            mock_data = MagicMock()
            mock_data.dtype = 'int32'
            
            result = manager.create_tensor(mock_data)
            self.assertEqual(result, mock_array)
            
            # Should call jnp.array with dtype=jnp.int32
            mock_jnp.array.assert_called_once()

    def test_create_tensor_with_integer_dtype_numba(self):
        """Test tensor creation with integer dtype for NUMBA"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.NUMBA
            
            # Create data with integer dtype
            mock_data = MagicMock()
            mock_data.dtype = 'int32'
            
            # Mock numpy array creation
            with patch('numpy.array') as mock_np_array:
                mock_array = MagicMock()
                mock_np_array.return_value = mock_array
                
                result = manager.create_tensor(mock_data)
                self.assertEqual(result, mock_array)
                
                # Should call numpy.array with dtype=np.int32
                mock_np_array.assert_called_once()

    def test_create_tensor_with_float_dtype_torch(self):
        """Test tensor creation with float dtype for PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock torch.tensor and torch.float32
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor
            mock_torch.float32 = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            # Create data with float dtype
            mock_data = MagicMock()
            mock_data.dtype = 'float64'
            
            result = manager.create_tensor(mock_data)
            self.assertEqual(result, mock_tensor)
            
            # Should call torch.tensor with dtype=torch.float32
            mock_torch.tensor.assert_called_once()

    def test_create_tensor_with_float_dtype_jax(self):
        """Test tensor creation with float dtype for JAX"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock jnp.array and jnp.float32
            mock_array = MagicMock()
            mock_jnp.array.return_value = mock_array
            mock_jnp.float32 = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            # Create data with float dtype
            mock_data = MagicMock()
            mock_data.dtype = 'float64'
            
            result = manager.create_tensor(mock_data)
            self.assertEqual(result, mock_array)
            
            # Should call jnp.array with dtype=jnp.float32
            mock_jnp.array.assert_called_once()

    def test_create_tensor_with_float_dtype_numba(self):
        """Test tensor creation with float dtype for NUMBA"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.NUMBA
            
            # Create data with float dtype
            mock_data = MagicMock()
            mock_data.dtype = 'float64'
            
            # Mock numpy array creation
            with patch('numpy.array') as mock_np_array:
                mock_array = MagicMock()
                mock_np_array.return_value = mock_array
                
                result = manager.create_tensor(mock_data)
                self.assertEqual(result, mock_array)
                
                # Should call numpy.array with dtype=np.float32
                mock_np_array.assert_called_once()

    def test_create_tensor_with_explicit_dtype_torch(self):
        """Test tensor creation with explicit dtype for PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock torch.tensor and torch.float64
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor
            mock_torch.float64 = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            result = manager.create_tensor([1, 2, 3], dtype=mock_torch.float64)
            self.assertEqual(result, mock_tensor)
            
            # Should call torch.tensor with explicit dtype
            mock_torch.tensor.assert_called_once()

    def test_create_tensor_with_explicit_dtype_jax(self):
        """Test tensor creation with explicit dtype for JAX"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock jnp.array and jnp.float64
            mock_array = MagicMock()
            mock_jnp.array.return_value = mock_array
            mock_jnp.float64 = MagicMock()
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            result = manager.create_tensor([1, 2, 3], dtype=mock_jnp.float64)
            self.assertEqual(result, mock_array)
            
            # Should call jnp.array with explicit dtype
            mock_jnp.array.assert_called_once()

    def test_create_tensor_with_explicit_dtype_numba(self):
        """Test tensor creation with explicit dtype for NUMBA"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.NUMBA
            
            # Mock numpy array creation
            with patch('numpy.array') as mock_np_array:
                mock_array = MagicMock()
                mock_np_array.return_value = mock_array
                mock_np_float64 = MagicMock()
                
                result = manager.create_tensor([1, 2, 3], dtype=mock_np_float64)
                self.assertEqual(result, mock_array)
                
                # Should call numpy.array with explicit dtype
                mock_np_array.assert_called_once()


if __name__ == '__main__':
    unittest.main()
