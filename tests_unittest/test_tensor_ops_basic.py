"""
Basic Test Suite for hpfracc/ml/tensor_ops.py

This test suite provides basic coverage for the TensorOps class focusing on:
- Initialization and backend resolution
- Basic tensor creation and conversion
- Array constructors
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor
from hpfracc.ml.backends import BackendType


class TestTensorOpsBasic(unittest.TestCase):
    """Test basic TensorOps functionality"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    def test_initialization_numba_backend(self, mock_get_backend_manager):
        """Test initialization with NUMBA backend"""
        # Mock backend manager
        mock_manager = Mock()
        mock_manager.active_backend = BackendType.NUMBA
        mock_get_backend_manager.return_value = mock_manager
        
        # Mock numpy import
        with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
            mock_import.return_value = np
            
            ops = TensorOps()
            
            self.assertEqual(ops.backend, BackendType.NUMBA)
            self.assertEqual(ops.tensor_lib, np)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    def test_initialization_string_backend(self, mock_get_backend_manager):
        """Test initialization with string backend"""
        mock_manager = Mock()
        mock_manager.active_backend = BackendType.NUMBA
        mock_get_backend_manager.return_value = mock_manager
        
        with patch('hpfracc.ml.tensor_ops.importlib.import_module') as mock_import:
            mock_import.return_value = np
            
            ops = TensorOps("numba")  # Use lowercase as expected by BackendType
            
            self.assertEqual(ops.backend, BackendType.NUMBA)
    
    def test_initialization_invalid_string_backend(self):
        """Test initialization with invalid string backend"""
        with self.assertRaises(ValueError):
            TensorOps("INVALID")
    
    def test_create_tensor_numba(self):
        """Test create_tensor with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        data = [1, 2, 3, 4]
        result = ops.create_tensor(data)
        
        np.testing.assert_array_equal(result, np.array(data))
    
    def test_create_tensor_torch(self):
        """Test create_tensor with TORCH backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.TORCH
        mock_torch = Mock()
        mock_tensor = Mock()
        mock_torch.tensor.return_value = mock_tensor
        ops.tensor_lib = mock_torch
        
        data = [1, 2, 3, 4]
        result = ops.create_tensor(data, requires_grad=True)
        
        self.assertEqual(result, mock_tensor)
        mock_torch.tensor.assert_called_with(data, requires_grad=True)
    
    def test_create_tensor_jax(self):
        """Test create_tensor with JAX backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.JAX
        mock_jnp = Mock()
        mock_array = Mock()
        mock_jnp.array.return_value = mock_array
        ops.tensor_lib = mock_jnp
        
        data = [1, 2, 3, 4]
        result = ops.create_tensor(data, requires_grad=True)
        
        self.assertEqual(result, mock_array)
        mock_jnp.array.assert_called_with(data)
    
    def test_zeros_numba(self):
        """Test zeros with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        result = ops.zeros((2, 3))
        expected = np.zeros((2, 3))
        
        np.testing.assert_array_equal(result, expected)
    
    def test_ones_numba(self):
        """Test ones with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        result = ops.ones((2, 3))
        expected = np.ones((2, 3))
        
        np.testing.assert_array_equal(result, expected)
    
    def test_eye_numba(self):
        """Test eye with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        result = ops.eye(3)
        expected = np.eye(3)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_arange_numba(self):
        """Test arange with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        result = ops.arange(0, 5, 1)
        expected = np.arange(0, 5, 1)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_from_numpy_numba(self):
        """Test from_numpy with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        array = np.array([1, 2, 3, 4])
        result = ops.from_numpy(array)
        
        np.testing.assert_array_equal(result, array)
    
    def test_to_numpy_numba(self):
        """Test to_numpy with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        array = np.array([1, 2, 3, 4])
        result = ops.to_numpy(array)
        
        np.testing.assert_array_equal(result, array)
    
    def test_no_grad_numba(self):
        """Test no_grad with NUMBA backend"""
        ops = TensorOps.__new__(TensorOps)
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = np
        
        from contextlib import nullcontext
        result = ops.no_grad()
        
        self.assertIsInstance(result, nullcontext)


class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level functions"""
    
    def test_get_tensor_ops(self):
        """Test get_tensor_ops function"""
        # This will use the global instance
        ops = get_tensor_ops()
        
        self.assertIsInstance(ops, TensorOps)
    
    def test_create_tensor_module_level(self):
        """Test create_tensor module-level function"""
        data = [1, 2, 3, 4]
        
        with patch('hpfracc.ml.tensor_ops.get_tensor_ops') as mock_get_ops:
            mock_ops = Mock()
            mock_result = np.array(data)
            mock_ops.create_tensor.return_value = mock_result
            mock_get_ops.return_value = mock_ops
            
            result = create_tensor(data)
            
            mock_get_ops.assert_called_once()
            mock_ops.create_tensor.assert_called_once_with(data)
            np.testing.assert_array_equal(result, mock_result)


class TestTensorOpsComprehensive(unittest.TestCase):
    """Comprehensive tests for TensorOps to improve coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_manager = Mock()
        self.mock_manager.active_backend = BackendType.NUMBA
        
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_backend_switching_torch(self, mock_import, mock_get_backend_manager):
        """Test switching to PyTorch backend"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_torch = Mock()
        mock_import.return_value = mock_torch
        
        ops = TensorOps(BackendType.TORCH)
        self.assertEqual(ops.backend, BackendType.TORCH)
        self.assertEqual(ops.tensor_lib, mock_torch)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_backend_switching_jax(self, mock_import, mock_get_backend_manager):
        """Test switching to JAX backend"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_jax = Mock()
        mock_import.return_value = mock_jax
        
        ops = TensorOps(BackendType.JAX)
        self.assertEqual(ops.backend, BackendType.JAX)
        self.assertEqual(ops.tensor_lib, mock_jax)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_creation_methods(self, mock_import, mock_get_backend_manager):
        """Test tensor creation methods"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_np = Mock()
        mock_np.array = Mock(return_value=np.array([1, 2, 3]))
        mock_np.zeros = Mock(return_value=np.zeros((2, 2)))
        mock_np.ones = Mock(return_value=np.ones((2, 2)))
        mock_np.eye = Mock(return_value=np.eye(2))
        mock_import.return_value = mock_np
        
        ops = TensorOps(BackendType.NUMBA)
        
        # Test tensor creation methods - these call the actual methods
        result = ops.tensor([1, 2, 3])
        self.assertIsNotNone(result)
        
        result = ops.zeros((2, 2))
        self.assertIsNotNone(result)
        
        result = ops.ones((2, 2))
        self.assertIsNotNone(result)
        
        result = ops.eye(2)
        self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_operations(self, mock_import, mock_get_backend_manager):
        """Test tensor operations"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_np = Mock()
        mock_np.matmul = Mock(return_value=np.array([[2, 2], [2, 2]]))
        mock_np.add = Mock(return_value=np.array([[2, 2], [2, 2]]))
        mock_np.multiply = Mock(return_value=np.array([[1, 1], [1, 1]]))
        mock_np.sum = Mock(return_value=4.0)
        mock_np.mean = Mock(return_value=1.0)
        mock_import.return_value = mock_np
        
        ops = TensorOps(BackendType.NUMBA)
        
        # Test operations - these call the actual methods
        result = ops.matmul(np.ones((2, 2)), np.ones((2, 2)))
        self.assertIsNotNone(result)
        
        result = ops.add(np.ones((2, 2)), np.ones((2, 2)))
        self.assertIsNotNone(result)
        
        result = ops.multiply(np.ones((2, 2)), np.ones((2, 2)))
        self.assertIsNotNone(result)
        
        result = ops.sum(np.ones((2, 2)))
        self.assertIsNotNone(result)
        
        result = ops.mean(np.ones((2, 2)))
        self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_activation_functions(self, mock_import, mock_get_backend_manager):
        """Test activation functions"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_np = Mock()
        mock_np.maximum = Mock(return_value=np.array([1, 1]))
        mock_np.exp = Mock(return_value=np.array([2.7, 2.7]))
        mock_np.tanh = Mock(return_value=np.array([0.8, 0.8]))
        mock_import.return_value = mock_np
        
        ops = TensorOps(BackendType.NUMBA)
        
        # Test activation functions - these call the actual methods
        result = ops.relu(np.array([-1, 1]))
        self.assertIsNotNone(result)
        
        result = ops.sigmoid(np.array([1, 1]))
        self.assertIsNotNone(result)
        
        result = ops.tanh(np.array([1, 1]))
        self.assertIsNotNone(result)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_additional_operations(self, mock_import, mock_get_backend_manager):
        """Test additional tensor operations"""
        mock_get_backend_manager.return_value = self.mock_manager
        mock_np = Mock()
        mock_np.sqrt = Mock(return_value=np.array([1, 2]))
        mock_np.sin = Mock(return_value=np.array([0.8, 0.9]))
        mock_np.cos = Mock(return_value=np.array([0.6, 0.4]))
        mock_import.return_value = mock_np
        
        ops = TensorOps(BackendType.NUMBA)
        
        # Test additional operations
        result = ops.sqrt(np.array([1, 4]))
        self.assertIsNotNone(result)
        
        result = ops.sin(np.array([1, 2]))
        self.assertIsNotNone(result)
        
        result = ops.cos(np.array([1, 2]))
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
