"""
Comprehensive unittest tests for hpfracc/ml/tensor_ops.py
Testing all methods to achieve maximum coverage
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys


class TestTensorOpsComprehensiveCoverage(unittest.TestCase):
    """Test TensorOps comprehensively for maximum coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        # Create actual enum-like objects
        self.BackendType = Mock()
        self.BackendType.TORCH = "torch"
        self.BackendType.JAX = "jax"
        self.BackendType.NUMBA = "numba"
        self.BackendType.AUTO = "auto"
        
        # Mock tensor libraries
        self.mock_torch = MagicMock()
        self.mock_jax_numpy = MagicMock()
        self.mock_numpy = MagicMock()
        
        # Mock tensor objects
        self.mock_torch_tensor = MagicMock()
        self.mock_jax_tensor = MagicMock()
        self.mock_numpy_array = MagicMock()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_like_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test zeros_like with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.zeros_like.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test zeros_like
        input_tensor = MagicMock()
        result = ops.zeros_like(input_tensor)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.zeros_like.assert_called_once_with(input_tensor)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_like_numba_with_shape(self, mock_import, mock_backend_type, mock_get_backend):
        """Test zeros_like with NUMBA backend and tensor with shape"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock tensor with shape attribute
        mock_tensor = MagicMock()
        mock_tensor.shape = (2, 3)
        
        # Mock numpy import inside the method
        with patch('numpy.zeros_like') as mock_np_zeros_like:
            mock_result = MagicMock()
            mock_np_zeros_like.return_value = mock_result
            
            # Test zeros_like
            result = ops.zeros_like(mock_tensor)
            
            mock_np_zeros_like.assert_called_once_with(mock_tensor)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_like_numba_without_shape(self, mock_import, mock_backend_type, mock_get_backend):
        """Test zeros_like with NUMBA backend and tensor without shape"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock tensor without shape attribute
        mock_tensor = MagicMock()
        del mock_tensor.shape  # Remove shape attribute
        
        # Mock numpy import inside the method
        with patch('numpy.zeros') as mock_np_zeros:
            mock_result = MagicMock()
            mock_np_zeros.return_value = mock_result
            
            # Test zeros_like
            result = ops.zeros_like(mock_tensor)
            
            mock_np_zeros.assert_called_once_with(1)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_sqrt_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test sqrt with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.sqrt.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test sqrt
        input_tensor = MagicMock()
        result = ops.sqrt(input_tensor)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.sqrt.assert_called_once_with(input_tensor)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_sqrt_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test sqrt with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.sqrt.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test sqrt
        input_tensor = MagicMock()
        result = ops.sqrt(input_tensor)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.sqrt.assert_called_once_with(input_tensor)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_sqrt_numba(self, mock_backend_type, mock_get_backend):
        """Test sqrt with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.sqrt') as mock_np_sqrt:
            mock_result = MagicMock()
            mock_np_sqrt.return_value = mock_result
            
            # Test sqrt
            input_tensor = MagicMock()
            result = ops.sqrt(input_tensor)
            
            mock_np_sqrt.assert_called_once_with(input_tensor)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_stack_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test stack with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.stack.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test stack
        tensors = [MagicMock(), MagicMock()]
        result = ops.stack(tensors, dim=1)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.stack.assert_called_once_with(tensors, dim=1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_stack_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test stack with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.stack.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test stack
        tensors = [MagicMock(), MagicMock()]
        result = ops.stack(tensors, dim=1)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.stack.assert_called_once_with(tensors, axis=1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_stack_numba(self, mock_backend_type, mock_get_backend):
        """Test stack with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.stack') as mock_np_stack:
            mock_result = MagicMock()
            mock_np_stack.return_value = mock_result
            
            # Test stack
            tensors = [MagicMock(), MagicMock()]
            result = ops.stack(tensors, dim=1)
            
            mock_np_stack.assert_called_once_with(tensors, axis=1)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tile_torch_int(self, mock_import, mock_backend_type, mock_get_backend):
        """Test tile with Torch backend and int reps"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with repeat method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.repeat.return_value = mock_result
        
        # Test tile with int reps
        result = ops.tile(mock_tensor, 3)
        
        mock_tensor.repeat.assert_called_once_with(3)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tile_torch_tuple(self, mock_import, mock_backend_type, mock_get_backend):
        """Test tile with Torch backend and tuple reps"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with repeat method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.repeat.return_value = mock_result
        
        # Test tile with tuple reps
        result = ops.tile(mock_tensor, (2, 3))
        
        mock_tensor.repeat.assert_called_once_with(2, 3)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tile_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test tile with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Mock tensor and to_numpy method
        mock_tensor = MagicMock()
        mock_numpy_array = MagicMock()
        mock_result = MagicMock()
        
        # Mock to_numpy method
        ops.to_numpy = MagicMock(return_value=mock_numpy_array)
        self.mock_jax_numpy.array.return_value = mock_result
        
        # Mock numpy import inside the method
        with patch('numpy.tile') as mock_np_tile:
            mock_tiled = MagicMock()
            mock_np_tile.return_value = mock_tiled
            
            # Test tile
            result = ops.tile(mock_tensor, (2, 3))
            
            mock_np_tile.assert_called_once_with(mock_numpy_array, (2, 3))
            self.mock_jax_numpy.array.assert_called_once_with(mock_tiled)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_tile_numba(self, mock_backend_type, mock_get_backend):
        """Test tile with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.tile') as mock_np_tile:
            mock_result = MagicMock()
            mock_np_tile.return_value = mock_result
            
            # Test tile
            mock_tensor = MagicMock()
            result = ops.tile(mock_tensor, (2, 3))
            
            mock_np_tile.assert_called_once_with(mock_tensor, (2, 3))
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_clip_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test clip with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with clamp method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.clamp.return_value = mock_result
        
        # Test clip
        result = ops.clip(mock_tensor, 0.0, 1.0)
        
        mock_tensor.clamp.assert_called_once_with(0.0, 1.0)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_clip_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test clip with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.clip.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test clip
        input_tensor = MagicMock()
        result = ops.clip(input_tensor, 0.0, 1.0)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.clip.assert_called_once_with(input_tensor, 0.0, 1.0)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_clip_numba(self, mock_backend_type, mock_get_backend):
        """Test clip with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.clip') as mock_np_clip:
            mock_result = MagicMock()
            mock_np_clip.return_value = mock_result
            
            # Test clip
            input_tensor = MagicMock()
            result = ops.clip(input_tensor, 0.0, 1.0)
            
            mock_np_clip.assert_called_once_with(input_tensor, 0.0, 1.0)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_unsqueeze_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test unsqueeze with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with unsqueeze method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.unsqueeze.return_value = mock_result
        
        # Test unsqueeze
        result = ops.unsqueeze(mock_tensor, 1)
        
        mock_tensor.unsqueeze.assert_called_once_with(1)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_unsqueeze_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test unsqueeze with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.expand_dims.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test unsqueeze
        input_tensor = MagicMock()
        result = ops.unsqueeze(input_tensor, 1)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.expand_dims.assert_called_once_with(input_tensor, 1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_unsqueeze_numba(self, mock_backend_type, mock_get_backend):
        """Test unsqueeze with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.expand_dims') as mock_np_expand_dims:
            mock_result = MagicMock()
            mock_np_expand_dims.return_value = mock_result
            
            # Test unsqueeze
            input_tensor = MagicMock()
            result = ops.unsqueeze(input_tensor, 1)
            
            mock_np_expand_dims.assert_called_once_with(input_tensor, 1)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_expand_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test expand with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with expand method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.expand.return_value = mock_result
        
        # Test expand
        result = ops.expand(mock_tensor, 2, 3)
        
        mock_tensor.expand.assert_called_once_with(2, 3)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_expand_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test expand with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.broadcast_to.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test expand
        input_tensor = MagicMock()
        result = ops.expand(input_tensor, 2, 3)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.broadcast_to.assert_called_once_with(input_tensor, (2, 3))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_expand_numba(self, mock_backend_type, mock_get_backend):
        """Test expand with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.broadcast_to') as mock_np_broadcast_to:
            mock_result = MagicMock()
            mock_np_broadcast_to.return_value = mock_result
            
            # Test expand
            input_tensor = MagicMock()
            result = ops.expand(input_tensor, 2, 3)
            
            mock_np_broadcast_to.assert_called_once_with(input_tensor, (2, 3))
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_gather_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test gather with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with gather method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.gather.return_value = mock_result
        
        # Test gather
        mock_index = MagicMock()
        result = ops.gather(mock_tensor, 1, mock_index)
        
        mock_tensor.gather.assert_called_once_with(1, mock_index)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_gather_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test gather with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.take_along_axis.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test gather
        input_tensor = MagicMock()
        mock_index = MagicMock()
        result = ops.gather(input_tensor, 1, mock_index)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.take_along_axis.assert_called_once_with(input_tensor, mock_index, axis=1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_gather_numba(self, mock_backend_type, mock_get_backend):
        """Test gather with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.take_along_axis') as mock_np_take_along_axis:
            mock_result = MagicMock()
            mock_np_take_along_axis.return_value = mock_result
            
            # Test gather
            input_tensor = MagicMock()
            mock_index = MagicMock()
            result = ops.gather(input_tensor, 1, mock_index)
            
            mock_np_take_along_axis.assert_called_once_with(input_tensor, mock_index, axis=1)
            self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_squeeze_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test squeeze with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Mock tensor with squeeze method
        mock_tensor = MagicMock()
        mock_result = MagicMock()
        mock_tensor.squeeze.return_value = mock_result
        
        # Test squeeze
        result = ops.squeeze(mock_tensor, 1)
        
        mock_tensor.squeeze.assert_called_once_with(1)
        self.assertEqual(result, mock_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_squeeze_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test squeeze with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.squeeze.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test squeeze
        input_tensor = MagicMock()
        result = ops.squeeze(input_tensor, 1)
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.squeeze.assert_called_once_with(input_tensor, axis=1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_squeeze_numba(self, mock_backend_type, mock_get_backend):
        """Test squeeze with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Mock numpy import inside the method
        with patch('numpy.squeeze') as mock_np_squeeze:
            mock_result = MagicMock()
            mock_np_squeeze.return_value = mock_result
            
            # Test squeeze
            input_tensor = MagicMock()
            result = ops.squeeze(input_tensor, 1)
            
            mock_np_squeeze.assert_called_once_with(input_tensor, axis=1)
            self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()
