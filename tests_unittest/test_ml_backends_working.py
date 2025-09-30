"""
Working tests for hpfracc/ml/backends.py

This module provides comprehensive tests for the backend management system,
focusing on core functionality that can be tested without complex mocking.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import warnings
from typing import Any, Dict, List


class TestBackendType(unittest.TestCase):
    """Test the BackendType enum"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendType
        self.BackendType = BackendType

    def test_backend_type_values(self):
        """Test that BackendType has correct values"""
        self.assertEqual(self.BackendType.TORCH.value, "torch")
        self.assertEqual(self.BackendType.JAX.value, "jax")
        self.assertEqual(self.BackendType.NUMBA.value, "numba")
        self.assertEqual(self.BackendType.AUTO.value, "auto")

    def test_backend_type_enumeration(self):
        """Test that all expected backend types exist"""
        expected_backends = ["torch", "jax", "numba", "auto"]
        actual_backends = [bt.value for bt in self.BackendType]
        
        for expected in expected_backends:
            self.assertIn(expected, actual_backends)

    def test_backend_type_comparison(self):
        """Test backend type comparison"""
        self.assertEqual(self.BackendType.TORCH, self.BackendType.TORCH)
        self.assertNotEqual(self.BackendType.TORCH, self.BackendType.JAX)


class TestBackendManagerWorking(unittest.TestCase):
    """Test BackendManager with working scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the availability flags
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
            from hpfracc.ml.backends import BackendManager, BackendType
            self.BackendManager = BackendManager
            self.BackendType = BackendType

    def test_backend_manager_initialization_defaults(self):
        """Test BackendManager initialization with default parameters"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            # Mock torch.cuda.is_available to return False
            mock_torch.cuda.is_available.return_value = False
            
            # Mock jax.devices to return CPU devices
            mock_jax.devices.return_value = [Mock()]
            
            # Mock numba.cuda.is_available to return False
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Verify basic attributes
            self.assertIsNotNone(manager.preferred_backend)
            self.assertIsNotNone(manager.available_backends)
            self.assertIsNotNone(manager.active_backend)
            self.assertIsNotNone(manager.backend_configs)

    def test_backend_manager_initialization_with_preferences(self):
        """Test BackendManager initialization with specific preferences"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            # Mock torch.cuda.is_available to return False
            mock_torch.cuda.is_available.return_value = False
            
            # Mock jax.devices to return CPU devices
            mock_jax.devices.return_value = [Mock()]
            
            # Mock numba.cuda.is_available to return False
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager(
                preferred_backend=self.BackendType.JAX,
                force_cpu=True,
                enable_jit=False,
                enable_gpu=False
            )
            
            # Verify preferences were set
            self.assertEqual(manager.preferred_backend, self.BackendType.JAX)
            self.assertTrue(manager.force_cpu)
            self.assertFalse(manager.enable_jit)
            self.assertFalse(manager.enable_gpu)

    def test_detect_available_backends_torch_only(self):
        """Test backend detection with only PyTorch available"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
             patch('hpfracc.ml.backends.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            available = manager._detect_available_backends()
            
            self.assertIn(self.BackendType.TORCH, available)
            self.assertNotIn(self.BackendType.JAX, available)
            self.assertNotIn(self.BackendType.NUMBA, available)

    def test_detect_available_backends_no_backends(self):
        """Test backend detection with no backends available"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
            
            with self.assertRaises(RuntimeError) as context:
                self.BackendManager()
            
            self.assertIn("No computation backends available", str(context.exception))

    def test_select_optimal_backend_auto_torch_preferred(self):
        """Test automatic backend selection preferring PyTorch"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager(preferred_backend=self.BackendType.AUTO)
            
            # Should prefer PyTorch when available
            if self.BackendType.TORCH in manager.available_backends:
                self.assertEqual(manager.active_backend, self.BackendType.TORCH)

    def test_select_optimal_backend_specific_preference(self):
        """Test backend selection with specific preference"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager(preferred_backend=self.BackendType.JAX)
            
            # Should select JAX if available
            if self.BackendType.JAX in manager.available_backends:
                self.assertEqual(manager.active_backend, self.BackendType.JAX)

    def test_get_backend_config(self):
        """Test getting backend configuration"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Test getting config for active backend
            config = manager.get_backend_config()
            self.assertIsInstance(config, dict)
            
            # Test getting config for specific backend
            if self.BackendType.TORCH in manager.available_backends:
                torch_config = manager.get_backend_config(self.BackendType.TORCH)
                self.assertIsInstance(torch_config, dict)

    def test_switch_backend_success(self):
        """Test successful backend switching"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Try to switch to an available backend
            for backend in manager.available_backends:
                if backend != manager.active_backend:
                    result = manager.switch_backend(backend)
                    self.assertTrue(result)
                    self.assertEqual(manager.active_backend, backend)
                    break

    def test_switch_backend_unavailable(self):
        """Test switching to unavailable backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Create a mock unavailable backend
            unavailable_backend = Mock()
            unavailable_backend.value = "unavailable"
            
            # Should return False and issue warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = manager.switch_backend(unavailable_backend)
                self.assertFalse(result)
                self.assertTrue(len(w) > 0)

    def test_get_tensor_lib_torch(self):
        """Test getting PyTorch tensor library"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            tensor_lib = manager.get_tensor_lib()
            self.assertEqual(tensor_lib, mock_torch)

    def test_get_tensor_lib_jax(self):
        """Test getting JAX tensor library"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            tensor_lib = manager.get_tensor_lib()
            self.assertEqual(tensor_lib, mock_jnp)

    def test_get_tensor_lib_numba(self):
        """Test getting NUMBA tensor library"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.NUMBA
            
            tensor_lib = manager.get_tensor_lib()
            self.assertEqual(tensor_lib, mock_numba)

    def test_get_tensor_lib_unknown(self):
        """Test getting tensor library for unknown backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = Mock()  # Unknown backend
            
            with self.assertRaises(RuntimeError) as context:
                manager.get_tensor_lib()
            
            self.assertIn("Unknown backend", str(context.exception))


class TestBackendManagerGlobalFunctions(unittest.TestCase):
    """Test global backend management functions"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendManager, BackendType
        self.BackendManager = BackendManager
        self.BackendType = BackendType

    def test_get_backend_manager_singleton(self):
        """Test that get_backend_manager returns singleton"""
        from hpfracc.ml.backends import get_backend_manager
        
        # Mock the availability flags to avoid import issues
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
             patch('hpfracc.ml.backends.torch') as mock_torch:
            
            mock_torch.cuda.is_available.return_value = False
            
            manager1 = get_backend_manager()
            manager2 = get_backend_manager()
            
            self.assertIs(manager1, manager2)

    def test_set_backend_manager(self):
        """Test setting custom backend manager"""
        from hpfracc.ml.backends import set_backend_manager, get_backend_manager
        
        # Create a mock manager
        mock_manager = MagicMock()
        
        # Set the mock manager
        set_backend_manager(mock_manager)
        
        # Verify it's returned
        manager = get_backend_manager()
        self.assertIs(manager, mock_manager)

    def test_get_active_backend(self):
        """Test getting active backend"""
        from hpfracc.ml.backends import get_active_backend
        
        with patch('hpfracc.ml.backends.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = self.BackendType.TORCH
            mock_get_manager.return_value = mock_manager
            
            active_backend = get_active_backend()
            self.assertEqual(active_backend, self.BackendType.TORCH)

    def test_switch_backend_global(self):
        """Test global backend switching"""
        from hpfracc.ml.backends import switch_backend
        
        with patch('hpfracc.ml.backends.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.switch_backend.return_value = True
            mock_get_manager.return_value = mock_manager
            
            result = switch_backend(self.BackendType.JAX)
            self.assertTrue(result)
            mock_manager.switch_backend.assert_called_once_with(self.BackendType.JAX)


class TestBackendManagerTensorCreation(unittest.TestCase):
    """Test tensor creation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendManager, BackendType
        self.BackendManager = BackendManager
        self.BackendType = BackendType

    def test_create_tensor_torch(self):
        """Test tensor creation with PyTorch backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock torch.tensor
            mock_tensor = MagicMock()
            mock_torch.tensor.return_value = mock_tensor
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.TORCH
            
            # Test tensor creation
            result = manager.create_tensor([1, 2, 3])
            self.assertEqual(result, mock_tensor)
            mock_torch.tensor.assert_called_once()

    def test_create_tensor_torch_with_dtype(self):
        """Test tensor creation with specific dtype"""
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
            
            # Test tensor creation with dtype
            result = manager.create_tensor([1, 2, 3], dtype=mock_torch.float64)
            self.assertEqual(result, mock_tensor)
            mock_torch.tensor.assert_called_once()

    def test_create_tensor_jax(self):
        """Test tensor creation with JAX backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock jnp.array
            mock_array = MagicMock()
            mock_jnp.array.return_value = mock_array
            
            manager = self.BackendManager()
            manager.active_backend = self.BackendType.JAX
            
            # Test tensor creation
            result = manager.create_tensor([1, 2, 3])
            self.assertEqual(result, mock_array)
            mock_jnp.array.assert_called_once()

    def test_create_tensor_numba(self):
        """Test tensor creation with NUMBA backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Mock numpy array creation
            with patch('numpy.array') as mock_np_array:
                mock_array = MagicMock()
                mock_np_array.return_value = mock_array
                
                manager = self.BackendManager()
                manager.active_backend = self.BackendType.NUMBA
                
                # Test tensor creation
                result = manager.create_tensor([1, 2, 3])
                self.assertEqual(result, mock_array)
                mock_np_array.assert_called_once()

    def test_create_tensor_unknown_backend(self):
        """Test tensor creation with unknown backend"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            manager.active_backend = Mock()  # Unknown backend
            
            with self.assertRaises(RuntimeError) as context:
                manager.create_tensor([1, 2, 3])
            
            self.assertIn("Unknown backend", str(context.exception))


if __name__ == '__main__':
    unittest.main()
