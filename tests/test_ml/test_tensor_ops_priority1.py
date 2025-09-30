#!/usr/bin/env python3
"""
Priority 1: Comprehensive TensorOps tests to achieve 80%+ coverage

This test suite focuses on the critical missing coverage areas:
- Backend resolution and error handling
- Tensor creation and conversion methods
- Array constructors and mathematical operations
- Backend-specific functionality
- Edge cases and error conditions
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import warnings

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor, switch_backend
from hpfracc.ml.backends import BackendType, BackendManager


class TestTensorOpsPriority1:
    """Priority 1 tests for TensorOps to achieve 80%+ coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock backend manager to avoid import issues
        self.mock_manager = MagicMock()
        self.mock_manager.active_backend = BackendType.TORCH
        
    # ------------------------ Backend Resolution Tests ------------------------
    
    def test_backend_resolution_explicit_torch(self):
        """Test explicit TORCH backend resolution."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                assert ops.backend == BackendType.TORCH
                assert ops.tensor_lib == mock_torch
    
    def test_backend_resolution_explicit_jax(self):
        """Test explicit JAX backend resolution."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                assert ops.backend == BackendType.JAX
                assert ops.tensor_lib == mock_jax
    
    def test_backend_resolution_explicit_numba(self):
        """Test explicit NUMBA backend resolution."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            assert ops.backend == BackendType.NUMBA
            assert ops.tensor_lib == np
    
    def test_backend_resolution_from_manager_active(self):
        """Test backend resolution from manager's active backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.JAX
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps()  # No explicit backend
                assert ops.backend == BackendType.JAX
    
    def test_backend_resolution_fallback_order(self):
        """Test backend resolution fallback order."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.AUTO  # Will trigger fallback
            mock_get_manager.return_value = mock_manager
            
            # Mock TORCH import to fail, JAX to succeed
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                def side_effect(module_name):
                    if module_name == "torch":
                        raise ImportError("torch not available")
                    elif module_name == "jax.numpy":
                        return MagicMock()  # Mock JAX
                    else:
                        return MagicMock()
                
                mock_import.side_effect = side_effect
                
                ops = TensorOps()
                assert ops.backend == BackendType.JAX
    
    def test_backend_resolution_no_usable_backend(self):
        """Test error when no usable backend is found."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.AUTO
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.side_effect = ImportError("No backend available")
                
                with pytest.raises(RuntimeError, match="No usable backend found"):
                    TensorOps()
    
    def test_get_tensor_lib_for_backend_edge_cases(self):
        """Test edge cases in _get_tensor_lib_for_backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(BackendType.TORCH)
                
                # Test unknown backend (should fall back to TORCH)
                result = ops._get_tensor_lib_for_backend("unknown_backend")
                assert result == torch
    
    # ------------------------ Tensor Creation Tests ------------------------
    
    def test_create_tensor_torch_backend(self):
        """Test tensor creation with TORCH backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3], requires_grad=True)
    
    def test_create_tensor_jax_backend(self):
        """Test tensor creation with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.create_tensor.return_value = np.array([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = MagicMock()
                
                ops = TensorOps(BackendType.JAX)
                result = ops.create_tensor([1, 2, 3], requires_grad=True)
                
                # JAX should filter out requires_grad
                mock_manager.create_tensor.assert_called_once_with([1, 2, 3])
    
    def test_create_tensor_numba_backend(self):
        """Test tensor creation with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.create_tensor.return_value = np.array([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            
            # NUMBA should filter out requires_grad
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3])
    
    def test_create_tensor_unknown_backend(self):
        """Test tensor creation with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"  # Force unknown backend
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])
    
    # ------------------------ Conversion Tests ------------------------
    
    def test_from_numpy_torch(self):
        """Test from_numpy with TORCH backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(BackendType.TORCH)
                arr = np.array([1, 2, 3])
                result = ops.from_numpy(arr)
                
                # Should call torch.from_numpy
                assert torch.is_tensor(result)
    
    def test_from_numpy_jax(self):
        """Test from_numpy with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.array.return_value = np.array([1, 2, 3])
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                arr = np.array([1, 2, 3])
                result = ops.from_numpy(arr)
                
                mock_jax.array.assert_called_once_with(arr)
    
    def test_from_numpy_numba(self):
        """Test from_numpy with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            arr = np.array([1, 2, 3])
            result = ops.from_numpy(arr)
            
            # NUMBA should return the array as-is
            assert result is arr
    
    def test_from_numpy_unknown_backend(self):
        """Test from_numpy with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.from_numpy(np.array([1, 2, 3]))
    
    def test_to_numpy_torch(self):
        """Test to_numpy with TORCH backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(BackendType.TORCH)
                tensor = torch.tensor([1, 2, 3])
                result = ops.to_numpy(tensor)
                
                # Should call detach().cpu().numpy()
                assert isinstance(result, np.ndarray)
    
    def test_to_numpy_jax(self):
        """Test to_numpy with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.device_get.return_value = np.array([1, 2, 3])
                mock_import.side_effect = lambda x: mock_jax if x == "jax" else MagicMock()
                
                ops = TensorOps(BackendType.JAX)
                jax_array = np.array([1, 2, 3])  # Mock JAX array
                result = ops.to_numpy(jax_array)
                
                mock_jax.device_get.assert_called_once_with(jax_array)
    
    def test_to_numpy_numba(self):
        """Test to_numpy with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            arr = np.array([1, 2, 3])
            result = ops.to_numpy(arr)
            
            # NUMBA should return the array as-is
            assert result is arr
    
    def test_to_numpy_unknown_backend(self):
        """Test to_numpy with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.to_numpy(np.array([1, 2, 3]))
    
    # ------------------------ Context Manager Tests ------------------------
    
    def test_no_grad_torch(self):
        """Test no_grad context manager with TORCH backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(BackendType.TORCH)
                context = ops.no_grad()
                
                # Should return torch.no_grad() context
                assert hasattr(context, '__enter__')
                assert hasattr(context, '__exit__')
    
    def test_no_grad_jax(self):
        """Test no_grad context manager with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.disable_jit.return_value = MagicMock()
                mock_import.side_effect = lambda x: mock_jax if x == "jax" else MagicMock()
                
                ops = TensorOps(BackendType.JAX)
                context = ops.no_grad()
                
                mock_jax.disable_jit.assert_called_once()
    
    def test_no_grad_numba(self):
        """Test no_grad context manager with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            context = ops.no_grad()
            
            # Should return nullcontext
            from contextlib import nullcontext
            assert isinstance(context, nullcontext)
    
    def test_no_grad_unknown_backend(self):
        """Test no_grad context manager with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.no_grad()
    
    # ------------------------ Array Constructor Tests ------------------------
    
    def test_zeros_torch_jax(self):
        """Test zeros with TORCH/JAX backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.zeros.return_value = torch.zeros((2, 3))
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.zeros((2, 3))
                
                mock_torch.zeros.assert_called_once_with((2, 3))
    
    def test_zeros_numba(self):
        """Test zeros with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.zeros.return_value = np.zeros((2, 3))
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.zeros((2, 3))
                
                mock_np.zeros.assert_called_once_with((2, 3))
    
    def test_zeros_unknown_backend(self):
        """Test zeros with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.zeros((2, 3))
    
    def test_ones_torch_jax(self):
        """Test ones with TORCH/JAX backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.ones.return_value = torch.ones((2, 3))
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.ones((2, 3))
                
                mock_torch.ones.assert_called_once_with((2, 3))
    
    def test_ones_numba(self):
        """Test ones with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.ones.return_value = np.ones((2, 3))
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.ones((2, 3))
                
                mock_np.ones.assert_called_once_with((2, 3))
    
    def test_ones_unknown_backend(self):
        """Test ones with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.ones((2, 3))
    
    def test_eye_torch_jax(self):
        """Test eye with TORCH/JAX backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.eye.return_value = torch.eye(3)
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.eye(3)
                
                mock_torch.eye.assert_called_once_with(3)
    
    def test_eye_numba(self):
        """Test eye with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.eye.return_value = np.eye(3)
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.eye(3)
                
                mock_np.eye.assert_called_once_with(3)
    
    def test_eye_unknown_backend(self):
        """Test eye with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.eye(3)
    
    # ------------------------ Arange Tests ------------------------
    
    def test_arange_torch_with_dtype(self):
        """Test arange with TORCH backend including dtype handling."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.float32 = torch.float32
                mock_torch.arange.return_value = torch.arange(0, 5, dtype=torch.float32)
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                
                # Test without dtype (should default to float32)
                result1 = ops.arange(0, 5)
                mock_torch.arange.assert_called_with(0, 5, 1, dtype=torch.float32)
                
                # Test with explicit dtype
                result2 = ops.arange(0, 5, dtype=torch.float64)
                mock_torch.arange.assert_called_with(0, 5, 1, dtype=torch.float64)
    
    def test_arange_jax(self):
        """Test arange with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.arange.return_value = np.arange(0, 5)
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.arange(0, 5, 2)
                
                mock_jax.arange.assert_called_once_with(0, 5, 2)
    
    def test_arange_numba(self):
        """Test arange with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.arange.return_value = np.arange(0, 5)
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.arange(0, 5, 2)
                
                mock_np.arange.assert_called_once_with(0, 5, 2)
    
    def test_arange_unknown_backend(self):
        """Test arange with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.arange(0, 5)
    
    # ------------------------ Module-level Function Tests ------------------------
    
    def test_get_tensor_ops(self):
        """Test get_tensor_ops module-level function."""
        with patch('hpfracc.ml.tensor_ops.TensorOps') as mock_tensor_ops:
            mock_instance = MagicMock()
            mock_tensor_ops.return_value = mock_instance
            
            result = get_tensor_ops(BackendType.TORCH)
            
            mock_tensor_ops.assert_called_once_with(BackendType.TORCH)
            assert result == mock_instance
    
    def test_create_tensor_module_level(self):
        """Test create_tensor module-level function."""
        with patch('hpfracc.ml.tensor_ops.get_tensor_ops') as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_ops.return_value = mock_ops
            
            result = create_tensor([1, 2, 3], backend=BackendType.TORCH)
            
            mock_get_ops.assert_called_once_with(BackendType.TORCH)
            mock_ops.create_tensor.assert_called_once_with([1, 2, 3])
    
    def test_switch_backend(self):
        """Test switch_backend module-level function."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            result = switch_backend(BackendType.JAX)
            
            mock_manager.set_backend.assert_called_once_with(BackendType.JAX)


class TestTensorOpsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_backend_resolution_with_auto_backend(self):
        """Test backend resolution when manager has AUTO backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.AUTO
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps()
                # Should fall back to standard fallback order
                assert ops.backend == BackendType.TORCH
    
    def test_backend_resolution_none_backend(self):
        """Test backend resolution with None backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = None
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(None)
                # Should fall back to standard fallback order
                assert ops.backend == BackendType.TORCH
    
    def test_tensor_method_alias(self):
        """Test that tensor method is an alias for create_tensor."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_import.return_value = torch
                
                ops = TensorOps(BackendType.TORCH)
                
                # Both methods should call the same underlying function
                result1 = ops.create_tensor([1, 2, 3])
                result2 = ops.tensor([1, 2, 3])
                
                assert mock_manager.create_tensor.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__])
