"""
Working unittest tests for hpfracc/ml/tensor_ops.py
Testing actual functionality with proper mocking
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys


class TestTensorOpsWorking(unittest.TestCase):
    """Test TensorOps with working mocks"""
    
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
    def test_tensor_ops_initialization_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test TensorOps initialization with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test initialization with torch backend
        ops = TensorOps("torch")
        
        # Verify basic attributes exist
        self.assertTrue(hasattr(ops, 'backend'))
        self.assertTrue(hasattr(ops, 'tensor_lib'))
        self.assertTrue(hasattr(ops, 'backend_manager'))
        
        # Verify backend manager was called
        mock_get_backend.assert_called_once()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_create_tensor_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test create_tensor with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        self.mock_torch.tensor.return_value = self.mock_torch_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test tensor creation
        result = ops.create_tensor([1, 2, 3])
        
        # Verify tensor creation was called
        self.mock_torch.tensor.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_add_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test add operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test add operation
        a = MagicMock()
        b = MagicMock()
        result = ops.add(a, b)
        
        # Should call torch.add
        self.mock_torch.add.assert_called_once_with(a, b)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_multiply_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test multiply operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test multiply operation
        a = MagicMock()
        b = MagicMock()
        result = ops.multiply(a, b)
        
        # Should call torch.multiply
        self.mock_torch.multiply.assert_called_once_with(a, b)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_matmul_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test matmul operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test matmul operation
        a = MagicMock()
        b = MagicMock()
        result = ops.matmul(a, b)
        
        # Should call torch.matmul
        self.mock_torch.matmul.assert_called_once_with(a, b)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_sum_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test sum operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Create a mock tensor with sum method
        mock_tensor = MagicMock()
        mock_sum_result = MagicMock()
        mock_tensor.sum.return_value = mock_sum_result
        
        # Test sum operation
        result = ops.sum(mock_tensor)
        
        # Should call tensor.sum()
        mock_tensor.sum.assert_called_once_with(dim=None, keepdim=False)
        self.assertEqual(result, mock_sum_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_mean_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test mean operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Create a mock tensor with mean method
        mock_tensor = MagicMock()
        mock_mean_result = MagicMock()
        mock_tensor.mean.return_value = mock_mean_result
        
        # Test mean operation
        result = ops.mean(mock_tensor)
        
        # Should call tensor.mean()
        mock_tensor.mean.assert_called_once_with(dim=None, keepdim=False)
        self.assertEqual(result, mock_mean_result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test zeros operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.zeros.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test zeros operation
        result = ops.zeros((3, 4))
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.zeros.assert_called_once_with((3, 4))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_ones_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test ones operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.ones.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test ones operation
        result = ops.ones((2, 3))
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.ones.assert_called_once_with((2, 3))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_eye_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test eye operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.eye.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test eye operation
        result = ops.eye(3)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.eye.assert_called_once_with(3)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_arange_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test arange operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.arange.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test arange operation
        result = ops.arange(0, 5)
        
        self.assertEqual(result, mock_tensor)
        # Check that arange was called with correct arguments (dtype may vary)
        call_args = self.mock_torch.arange.call_args
        self.assertEqual(call_args[0], (0, 5, 1))
        self.assertIn('dtype', call_args[1])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_no_grad_context_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test no_grad context manager with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_context = MagicMock()
        self.mock_torch.no_grad.return_value = mock_context
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test no_grad context manager
        result = ops.no_grad()
        
        self.assertEqual(result, mock_context)
        self.mock_torch.no_grad.assert_called_once()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_from_numpy_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test from_numpy with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.from_numpy.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test from_numpy
        numpy_array = np.array([1, 2, 3])
        result = ops.from_numpy(numpy_array)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.from_numpy.assert_called_once_with(numpy_array)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_to_numpy_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test to_numpy with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Create a mock tensor with detach, cpu, and numpy methods
        mock_tensor = MagicMock()
        mock_detached = MagicMock()
        mock_cpu = MagicMock()
        mock_numpy_array = np.array([1, 2, 3])
        
        mock_tensor.detach.return_value = mock_detached
        mock_detached.cpu.return_value = mock_cpu
        mock_cpu.numpy.return_value = mock_numpy_array
        
        # Test to_numpy
        result = ops.to_numpy(mock_tensor)
        
        # Verify the chain of calls
        mock_tensor.detach.assert_called_once()
        mock_detached.cpu.assert_called_once()
        mock_cpu.numpy.assert_called_once()
        self.assertEqual(result, mock_numpy_array)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_stack_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test stack operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.stack.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test stack operation
        tensors = [MagicMock(), MagicMock()]
        result = ops.stack(tensors, dim=0)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.stack.assert_called_once_with(tensors, dim=0)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_concatenate_operation_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test concatenate operation with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        mock_tensor = MagicMock()
        self.mock_torch.cat.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test concatenate operation
        tensors = [MagicMock(), MagicMock()]
        result = ops.concatenate(tensors, dim=0)
        
        self.assertEqual(result, mock_tensor)
        self.mock_torch.cat.assert_called_once_with(tensors, dim=0)


class TestTensorOpsErrorHandling(unittest.TestCase):
    """Test error handling in TensorOps"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_invalid_backend_string(self, mock_backend_type, mock_get_backend):
        """Test initialization with invalid backend string"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        # Mock BackendType to raise ValueError for invalid backend
        def side_effect(value):
            if value == "invalid":
                raise ValueError("Unknown backend")
            return value
        
        mock_backend_type.side_effect = side_effect
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test that ValueError is raised for invalid backend
        with self.assertRaises(ValueError) as context:
            TensorOps("invalid")
        
        self.assertIn("Unknown backend", str(context.exception))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_unknown_backend_in_operation(self, mock_import, mock_backend_type, mock_get_backend):
        """Test operation with unknown backend"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = MagicMock()
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        # Force an unknown backend
        ops.backend = "unknown"
        
        # Test that RuntimeError is raised for unknown backend
        with self.assertRaises(RuntimeError) as context:
            ops.create_tensor([1, 2, 3])
        
        self.assertIn("Unknown backend", str(context.exception))


class TestTensorOpsJAXBackend(unittest.TestCase):
    """Test TensorOps with JAX backend"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_create_tensor_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test create_tensor with JAX backend"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        mock_backend_type.JAX = "jax"
        
        mock_jax_numpy = MagicMock()
        mock_tensor = MagicMock()
        mock_jax_numpy.array.return_value = mock_tensor
        mock_import.return_value = mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test tensor creation
        result = ops.create_tensor([1, 2, 3])
        
        # Verify jax numpy array creation was called
        mock_jax_numpy.array.assert_called_once_with([1, 2, 3])
        
        # Test that requires_grad is removed for JAX
        mock_jax_numpy.array.reset_mock()
        result = ops.create_tensor([1, 2, 3], requires_grad=True)
        
        mock_jax_numpy.array.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_sum_operation_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test sum operation with JAX backend"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        mock_backend_type.JAX = "jax"
        
        mock_jax_numpy = MagicMock()
        mock_import.return_value = mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Create a mock tensor
        mock_tensor = MagicMock()
        mock_sum_result = MagicMock()
        mock_jax_numpy.sum.return_value = mock_sum_result
        
        # Test sum operation
        result = ops.sum(mock_tensor)
        
        # Should call jax numpy sum
        mock_jax_numpy.sum.assert_called_once_with(mock_tensor, axis=None, keepdims=False)
        self.assertEqual(result, mock_sum_result)


class TestTensorOpsNUMBABackend(unittest.TestCase):
    """Test TensorOps with NUMBA backend"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_create_tensor_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test create_tensor with NUMBA backend"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        mock_array = MagicMock()
        mock_np.array.return_value = mock_array
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Test tensor creation
        result = ops.create_tensor([1, 2, 3])
        
        # Verify numpy array creation was called
        mock_np.array.assert_called_once_with([1, 2, 3])
        
        # Test that requires_grad is removed for NUMBA
        mock_np.array.reset_mock()
        result = ops.create_tensor([1, 2, 3], requires_grad=True)
        
        mock_np.array.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_sum_operation_numba(self, mock_backend_type, mock_get_backend):
        """Test sum operation with NUMBA backend"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Create a mock tensor
        mock_tensor = MagicMock()
        
        # Mock numpy import inside the method
        with patch('numpy.sum') as mock_np_sum:
            mock_sum_result = MagicMock()
            mock_np_sum.return_value = mock_sum_result
            
            # Test sum operation
            result = ops.sum(mock_tensor)
            
            # Should call numpy sum
            mock_np_sum.assert_called_once_with(mock_tensor, axis=None, keepdims=False)
            self.assertEqual(result, mock_sum_result)


if __name__ == '__main__':
    unittest.main()
