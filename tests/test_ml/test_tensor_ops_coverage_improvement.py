#!/usr/bin/env python3
"""
Coverage improvement tests for TensorOps

This test suite focuses on improving coverage for tensor_ops.py
without causing PyTorch import issues.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.tensor_ops import TensorOps
from hpfracc.ml.backends import BackendType


class TestTensorOpsCoverageImprovement:
    """Tests to improve tensor_ops.py coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_manager = MagicMock()
        self.mock_manager.active_backend = BackendType.TORCH
        
    def test_stack_operations(self):
        """Test stack operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_tensor1 = MagicMock()
                mock_tensor2 = MagicMock()
                mock_torch.stack.return_value = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.stack([mock_tensor1, mock_tensor2], dim=0)
                
                mock_torch.stack.assert_called_once_with([mock_tensor1, mock_tensor2], dim=0)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.stack.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.stack([mock_tensor1, mock_tensor2], dim=0)
                
                mock_jax.stack.assert_called_once_with([mock_tensor1, mock_tensor2], axis=0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.stack.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.stack([mock_tensor1, mock_tensor2], dim=0)
                
                mock_np.stack.assert_called_once_with([mock_tensor1, mock_tensor2], axis=0)
            
            # Test unknown backend
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.stack([mock_tensor1, mock_tensor2])
    
    def test_cat_operations(self):
        """Test concatenation operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor1 = MagicMock()
            mock_tensor2 = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.cat.return_value = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.cat([mock_tensor1, mock_tensor2], dim=0)
                
                mock_torch.cat.assert_called_once_with([mock_tensor1, mock_tensor2], dim=0)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.concatenate.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.cat([mock_tensor1, mock_tensor2], dim=0)
                
                mock_jax.concatenate.assert_called_once_with([mock_tensor1, mock_tensor2], axis=0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.concatenate.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.cat([mock_tensor1, mock_tensor2], dim=0)
                
                mock_np.concatenate.assert_called_once_with([mock_tensor1, mock_tensor2], axis=0)
    
    def test_reshape_operation(self):
        """Test reshape operation."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            mock_tensor.reshape.return_value = MagicMock()
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.reshape(mock_tensor, (2, 3))
            
            mock_tensor.reshape.assert_called_once_with((2, 3))
    
    def test_clip_operations(self):
        """Test clip operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.clamp.return_value = MagicMock()
                result = ops.clip(mock_tensor, 0.0, 1.0)
                
                mock_tensor.clamp.assert_called_once_with(0.0, 1.0)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.clip.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.clip(mock_tensor, 0.0, 1.0)
                
                mock_jax.clip.assert_called_once_with(mock_tensor, 0.0, 1.0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.clip.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.clip(mock_tensor, 0.0, 1.0)
                
                mock_np.clip.assert_called_once_with(mock_tensor, 0.0, 1.0)
            
            # Test unknown backend
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.clip(mock_tensor, 0.0, 1.0)
    
    def test_unsqueeze_operations(self):
        """Test unsqueeze operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.unsqueeze.return_value = MagicMock()
                result = ops.unsqueeze(mock_tensor, 0)
                
                mock_tensor.unsqueeze.assert_called_once_with(0)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.expand_dims.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.unsqueeze(mock_tensor, 0)
                
                mock_jax.expand_dims.assert_called_once_with(mock_tensor, 0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.expand_dims.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.unsqueeze(mock_tensor, 0)
                
                mock_np.expand_dims.assert_called_once_with(mock_tensor, 0)
    
    def test_expand_operations(self):
        """Test expand operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.expand.return_value = MagicMock()
                result = ops.expand(mock_tensor, 2, 3)
                
                mock_tensor.expand.assert_called_once_with(2, 3)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.broadcast_to.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.expand(mock_tensor, 2, 3)
                
                mock_jax.broadcast_to.assert_called_once_with(mock_tensor, (2, 3))
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.broadcast_to.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.expand(mock_tensor, 2, 3)
                
                mock_np.broadcast_to.assert_called_once_with(mock_tensor, (2, 3))
    
    def test_gather_operations(self):
        """Test gather operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            mock_index = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.gather.return_value = MagicMock()
                result = ops.gather(mock_tensor, 0, mock_index)
                
                mock_tensor.gather.assert_called_once_with(0, mock_index)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.take_along_axis.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.gather(mock_tensor, 0, mock_index)
                
                mock_jax.take_along_axis.assert_called_once_with(mock_tensor, mock_index, axis=0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.take_along_axis.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.gather(mock_tensor, 0, mock_index)
                
                mock_np.take_along_axis.assert_called_once_with(mock_tensor, mock_index, axis=0)
    
    def test_squeeze_operations(self):
        """Test squeeze operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.squeeze.return_value = MagicMock()
                result = ops.squeeze(mock_tensor, 0)
                
                mock_tensor.squeeze.assert_called_once_with(0)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.squeeze.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.squeeze(mock_tensor, 0)
                
                mock_jax.squeeze.assert_called_once_with(mock_tensor, axis=0)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.squeeze.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.squeeze(mock_tensor, 0)
                
                mock_np.squeeze.assert_called_once_with(mock_tensor, axis=0)
    
    def test_tile_operations(self):
        """Test tile operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            
            # Test TORCH backend with integer reps
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.repeat.return_value = MagicMock()
                result = ops.tile(mock_tensor, 2)
                
                mock_tensor.repeat.assert_called_once_with(2)
            
            # Test TORCH backend with tuple reps
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                mock_tensor.repeat.return_value = MagicMock()
                result = ops.tile(mock_tensor, (2, 3))
                
                mock_tensor.repeat.assert_called_once_with(2, 3)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.array.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                # Mock to_numpy method
                ops.to_numpy = MagicMock(return_value=np.array([1, 2, 3]))
                
                with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                    mock_np.tile.return_value = MagicMock()
                    result = ops.tile(mock_tensor, (2, 3))
                    
                    ops.to_numpy.assert_called_once_with(mock_tensor)
                    mock_np.tile.assert_called_once()
                    mock_jax.array.assert_called_once()
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.tile.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.tile(mock_tensor, (2, 3))
                
                mock_np.tile.assert_called_once_with(mock_tensor, (2, 3))
            
            # Test unknown backend
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.tile(mock_tensor, 2)
    
    def test_ones_like_operations(self):
        """Test ones_like operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            mock_tensor.shape = (2, 3)
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.ones_like.return_value = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.ones_like(mock_tensor)
                
                mock_torch.ones_like.assert_called_once_with(mock_tensor)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.ones_like.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.ones_like(mock_tensor)
                
                mock_jax.ones_like.assert_called_once_with(mock_tensor)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.ones_like.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.ones_like(mock_tensor)
                
                mock_np.ones_like.assert_called_once_with(mock_tensor)
    
    def test_eye_like_operations(self):
        """Test eye_like operations across backends."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_get_manager.return_value = self.mock_manager
            
            mock_tensor = MagicMock()
            mock_tensor.shape = (3, 3)
            
            # Test TORCH backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_torch = MagicMock()
                mock_torch.eye.return_value = MagicMock()
                mock_import.return_value = mock_torch
                
                ops = TensorOps(BackendType.TORCH)
                result = ops.eye_like(mock_tensor)
                
                mock_torch.eye.assert_called_once_with(3, 3)
            
            # Test JAX backend
            with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
                mock_jax = MagicMock()
                mock_jax.eye.return_value = MagicMock()
                mock_import.return_value = mock_jax
                
                ops = TensorOps(BackendType.JAX)
                result = ops.eye_like(mock_tensor)
                
                mock_jax.eye.assert_called_once_with(3, 3)
            
            # Test NUMBA backend
            with patch('hpfracc.ml.tensor_ops.np') as mock_np:
                mock_np.eye.return_value = MagicMock()
                
                ops = TensorOps(BackendType.NUMBA)
                result = ops.eye_like(mock_tensor)
                
                mock_np.eye.assert_called_once_with(3, 3)


if __name__ == "__main__":
    pytest.main([__file__])

