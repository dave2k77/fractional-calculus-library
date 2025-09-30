"""
Final coverage test for hpfracc/ml/backends.py line 125

This module tests the specific fallback case in _select_optimal_backend.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from typing import Any, Dict, List


class TestBackendManagerFinalCoverage(unittest.TestCase):
    """Final coverage test for BackendManager line 125"""

    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import BackendManager, BackendType
        self.BackendManager = BackendManager
        self.BackendType = BackendType

    def test_select_optimal_backend_fallback_line_125(self):
        """Test the specific fallback case on line 125"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            # Create a scenario where AUTO is selected but none of the preferred
            # backends are available, triggering the fallback case
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
                
                manager = self.BackendManager(preferred_backend=self.BackendType.AUTO)
                
                # Manually set available_backends to force the fallback path
                # This simulates a scenario where the preferred backends are not available
                # but we have at least one backend available
                manager.available_backends = [self.BackendType.NUMBA]
                
                # Force the _select_optimal_backend to use the fallback case
                # by mocking the preferred backend logic
                with patch.object(manager, '_select_optimal_backend') as mock_select:
                    # Create a new instance to test the actual method
                    manager2 = self.BackendManager(preferred_backend=self.BackendType.AUTO)
                    manager2.available_backends = [self.BackendType.NUMBA]
                    
                    # Test the actual fallback case
                    result = manager2._select_optimal_backend()
                    
                    # Should return the first available backend (NUMBA)
                    self.assertEqual(result, self.BackendType.NUMBA)

    def test_select_optimal_backend_fallback_with_multiple_backends(self):
        """Test the fallback case with multiple backends available"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
                
                manager = self.BackendManager(preferred_backend=self.BackendType.AUTO)
                
                # Set available backends in a specific order
                manager.available_backends = [
                    self.BackendType.JAX, 
                    self.BackendType.TORCH, 
                    self.BackendType.NUMBA
                ]
                
                # Test the fallback case
                result = manager._select_optimal_backend()
                
                # Should return the first available backend (JAX)
                self.assertEqual(result, self.BackendType.JAX)

    def test_select_optimal_backend_fallback_edge_case(self):
        """Test the fallback case with edge conditions"""
        with patch('hpfracc.ml.backends.torch') as mock_torch, \
             patch('hpfracc.ml.backends.jax') as mock_jax, \
             patch('hpfracc.ml.backends.jnp') as mock_jnp, \
             patch('hpfracc.ml.backends.numba') as mock_numba:
            
            mock_torch.cuda.is_available.return_value = False
            mock_jax.devices.return_value = [Mock()]
            mock_numba.cuda.is_available.return_value = False
            
            with patch('hpfracc.ml.backends.TORCH_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.JAX_AVAILABLE', True), \
                 patch('hpfracc.ml.backends.NUMBA_AVAILABLE', True):
                
                manager = self.BackendManager(
                    preferred_backend=self.BackendType.AUTO,
                    enable_gpu=False  # This affects JAX preference
                )
                
                # Set available backends to test the fallback
                manager.available_backends = [self.BackendType.TORCH]
                
                # Test the fallback case
                result = manager._select_optimal_backend()
                
                # Should return the first available backend (TORCH)
                self.assertEqual(result, self.BackendType.TORCH)


if __name__ == '__main__':
    unittest.main()
