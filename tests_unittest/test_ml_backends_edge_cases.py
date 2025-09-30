"""
Edge case tests for hpfracc/ml/backends.py to achieve 100% coverage

This module tests the remaining uncovered lines including import failures
and edge cases in backend selection.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import warnings
from typing import Any, Dict, List


class TestBackendManagerEdgeCases(unittest.TestCase):
    """Edge case tests for BackendManager to achieve 100% coverage"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendManager, BackendType
        self.BackendManager = BackendManager
        self.BackendType = BackendType

    def test_import_error_torch_unavailable(self):
        """Test behavior when PyTorch import fails (lines 17-18)"""
        # Mock the import to fail
        with patch.dict('sys.modules', {'torch': None}):
            # Force reimport to trigger ImportError
            if 'hpfracc.ml.backends' in sys.modules:
                del sys.modules['hpfracc.ml.backends']
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.jax') as mock_jax:
                
                mock_jax.devices.return_value = [Mock()]
                
                # This should work without PyTorch
                manager = self.BackendManager()
                available = manager._detect_available_backends()
                
                self.assertNotIn(self.BackendType.TORCH, available)
                self.assertIn(self.BackendType.JAX, available)

    def test_import_error_jax_unavailable(self):
        """Test behavior when JAX import fails (lines 24-25)"""
        # Mock the import to fail
        with patch.dict('sys.modules', {'jax': None}):
            # Force reimport to trigger ImportError
            if 'hpfracc.ml.backends' in sys.modules:
                del sys.modules['hpfracc.ml.backends']
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.torch') as mock_torch:
                
                mock_torch.cuda.is_available.return_value = False
                
                # This should work without JAX
                manager = self.BackendManager()
                available = manager._detect_available_backends()
                
                self.assertIn(self.BackendType.TORCH, available)
                self.assertNotIn(self.BackendType.JAX, available)

    def test_import_error_numba_unavailable(self):
        """Test behavior when NUMBA import fails (lines 30-31)"""
        # Mock the import to fail
        with patch.dict('sys.modules', {'numba': None}):
            # Force reimport to trigger ImportError
            if 'hpfracc.ml.backends' in sys.modules:
                del sys.modules['hpfracc.ml.backends']
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.torch') as mock_torch:
                
                mock_torch.cuda.is_available.return_value = False
                
                # This should work without NUMBA
                manager = self.BackendManager()
                available = manager._detect_available_backends()
                
                self.assertIn(self.BackendType.TORCH, available)
                self.assertNotIn(self.BackendType.NUMBA, available)

    def test_select_optimal_backend_fallback_case(self):
        """Test the fallback case in _select_optimal_backend (line 125)"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Create a scenario where none of the preferred backends are available
            # but we have at least one backend available
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
                
                manager = self.BackendManager(preferred_backend=self.BackendType.AUTO)
                
                # Mock the available_backends to have only JAX
                manager.available_backends = [self.BackendType.JAX]
                
                # This should trigger the fallback case (line 125)
                optimal_backend = manager._select_optimal_backend()
                
                # Should return the first available backend
                self.assertEqual(optimal_backend, self.BackendType.JAX)

    def test_select_optimal_backend_fallback_case_multiple_backends(self):
        """Test the fallback case with multiple backends available"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Create a scenario where none of the preferred backends are available
            # but we have multiple backends available
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
                
                manager = self.BackendManager(
                    preferred_backend=self.BackendType.AUTO,
                    enable_gpu=False  # This will make JAX not preferred
                )
                
                # Mock the available_backends to have both TORCH and JAX
                manager.available_backends = [self.BackendType.TORCH, self.BackendType.JAX]
                
                # This should trigger the fallback case (line 125)
                optimal_backend = manager._select_optimal_backend()
                
                # Should return the first available backend (TORCH)
                self.assertEqual(optimal_backend, self.BackendType.TORCH)

    def test_select_optimal_backend_fallback_case_single_backend(self):
        """Test the fallback case with only one backend available"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Create a scenario where only NUMBA is available
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
                
                manager = self.BackendManager(preferred_backend=self.BackendType.AUTO)
                
                # Mock the available_backends to have only NUMBA
                manager.available_backends = [self.BackendType.NUMBA]
                
                # This should trigger the fallback case (line 125)
                optimal_backend = manager._select_optimal_backend()
                
                # Should return the first (and only) available backend
                self.assertEqual(optimal_backend, self.BackendType.NUMBA)

    def test_detect_available_backends_all_imports_fail(self):
        """Test detection when all backend imports fail"""
        # Mock all imports to fail
        with patch.dict('sys.modules', {'torch': None, 'jax': None, 'numba': None}):
            # Force reimport to trigger ImportErrors
            if 'hpfracc.ml.backends' in sys.modules:
                del sys.modules['hpfracc.ml.backends']
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
                
                # This should raise RuntimeError
                with self.assertRaises(RuntimeError) as context:
                    self.BackendManager()
                
                self.assertIn("No computation backends available", str(context.exception))

    def test_detect_available_backends_mixed_import_failures(self):
        """Test detection with mixed import failures"""
        # Mock PyTorch and NUMBA imports to fail, but JAX to succeed
        with patch.dict('sys.modules', {'torch': None, 'numba': None}):
            # Force reimport to trigger ImportErrors
            if 'hpfracc.ml.backends' in sys.modules:
                del sys.modules['hpfracc.ml.backends']
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False), \
                 patch('hpfracc.ml.backends.jax') as mock_jax:
                
                mock_jax.devices.return_value = [Mock()]
                
                # This should work with only JAX
                manager = self.BackendManager()
                available = manager._detect_available_backends()
                
                self.assertNotIn(self.BackendType.TORCH, available)
                self.assertIn(self.BackendType.JAX, available)
                self.assertNotIn(self.BackendType.NUMBA, available)

    def test_backend_manager_with_all_backends_unavailable(self):
        """Test BackendManager when no backends are available"""
        with patch('hpfracc.ml.backends.TORCH_AVAILABLE', False), \
             patch('hpfracc.ml.backends.JAX_AVAILABLE', False), \
             patch('hpfracc.ml.backends.NUMBA_AVAILABLE', False):
            
            with self.assertRaises(RuntimeError) as context:
                self.BackendManager()
            
            self.assertIn("No computation backends available", str(context.exception))

    def test_backend_manager_initialization_print_statements(self):
        """Test that initialization print statements are covered"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba, \
             patch('builtins.print') as mock_print:
            
            mock_torch.cuda.is_available.return_value = True
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Verify that print statements were called
            self.assertTrue(mock_print.called)

    def test_backend_manager_switch_backend_print_statement(self):
        """Test that backend switching print statement is covered"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba, \
             patch('builtins.print') as mock_print:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            manager = self.BackendManager()
            
            # Switch to an available backend
            for backend in manager.available_backends:
                if backend != manager.active_backend:
                    result = manager.switch_backend(backend)
                    self.assertTrue(result)
                    break
            
            # Verify that switch print statement was called
            self.assertTrue(mock_print.called)

    def test_backend_manager_gpu_detection_print_statements(self):
        """Test that GPU detection print statements are covered"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba, \
             patch('builtins.print') as mock_print:
            
            # Mock CUDA availability
            mock_torch.cuda.is_available.return_value = True
            
            # Mock JAX GPU device
            mock_gpu_device = Mock()
            mock_gpu_device.__str__ = Mock(return_value="gpu:0")
            mock_jax.devices.return_value = [mock_gpu_device]
            
            # Mock NUMBA CUDA availability
            mock_numba.cuda.is_available.return_value = True
            
            manager = self.BackendManager(force_cpu=False)
            
            # Verify that GPU detection print statements were called
            self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()
