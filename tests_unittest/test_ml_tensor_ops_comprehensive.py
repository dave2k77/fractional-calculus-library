"""
Comprehensive unittest tests for hpfracc/ml/tensor_ops.py
Testing all tensor operations, backend management, and edge cases using mocks
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys
import importlib


class TestTensorOpsComprehensive(unittest.TestCase):
    """Test the TensorOps class comprehensively"""
    
    def setUp(self):
        """Set up test fixtures with mocked backends"""
        # Mock the backend manager and BackendType
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        # Create mock backend types
        self.BackendType = MagicMock()
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
        
        self.assertEqual(ops.backend, "torch")
        self.assertEqual(ops.tensor_lib, self.mock_torch)
        mock_import.assert_called_with("torch")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_ops_initialization_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test TensorOps initialization with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test initialization with jax backend
        ops = TensorOps("jax")
        
        self.assertEqual(ops.backend, "jax")
        self.assertEqual(ops.tensor_lib, self.mock_jax_numpy)
        mock_import.assert_called_with("jax.numpy")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_tensor_ops_initialization_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test TensorOps initialization with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test initialization with numba backend
        ops = TensorOps("numba")
        
        self.assertEqual(ops.backend, "numba")
        self.assertEqual(ops.tensor_lib, mock_np)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_tensor_ops_initialization_invalid_backend(self, mock_backend_type, mock_get_backend):
        """Test TensorOps initialization with invalid backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.side_effect = ValueError("Unknown backend")
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test initialization with invalid backend
        with self.assertRaises(ValueError):
            TensorOps("invalid")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_resolve_backend_with_explicit_backend(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _resolve_backend with explicit backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that explicit backend is used
        backend, tensor_lib = ops._resolve_backend("torch")
        self.assertEqual(backend, "torch")
        self.assertEqual(tensor_lib, self.mock_torch)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_resolve_backend_with_active_backend(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _resolve_backend with active backend from manager"""
        # Setup mocks
        self.mock_backend_manager.active_backend = "jax"
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test that active backend is used
        backend, tensor_lib = ops._resolve_backend(None)
        self.assertEqual(backend, "jax")
        self.assertEqual(tensor_lib, self.mock_jax_numpy)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_resolve_backend_fallback_order(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _resolve_backend fallback order"""
        # Setup mocks - torch fails, jax succeeds
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        
        def side_effect(module_name):
            if module_name == "torch":
                raise ImportError("torch not available")
            elif module_name == "jax.numpy":
                return self.mock_jax_numpy
            else:
                return MagicMock()
        
        mock_import.side_effect = side_effect
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test fallback order
        backend, tensor_lib = ops._resolve_backend(None)
        self.assertEqual(backend, "jax")
        self.assertEqual(tensor_lib, self.mock_jax_numpy)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_resolve_backend_no_usable_backend(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _resolve_backend when no backend is available"""
        # Setup mocks - all backends fail
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        mock_import.side_effect = ImportError("No backend available")
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test that RuntimeError is raised
        with self.assertRaises(RuntimeError) as context:
            ops._resolve_backend(None)
        
        self.assertIn("No usable backend found", str(context.exception))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_get_tensor_lib_for_backend_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _get_tensor_lib_for_backend with Torch"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test torch backend
        lib = ops._get_tensor_lib_for_backend("torch")
        self.assertEqual(lib, self.mock_torch)
        mock_import.assert_called_with("torch")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_get_tensor_lib_for_backend_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _get_tensor_lib_for_backend with JAX"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test jax backend
        lib = ops._get_tensor_lib_for_backend("jax")
        self.assertEqual(lib, self.mock_jax_numpy)
        mock_import.assert_called_with("jax.numpy")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_get_tensor_lib_for_backend_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test _get_tensor_lib_for_backend with NUMBA"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test numba backend
        lib = ops._get_tensor_lib_for_backend("numba")
        self.assertEqual(lib, mock_np)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_get_tensor_lib_for_backend_edge_case(self, mock_import, mock_backend_type, mock_get_backend):
        """Test _get_tensor_lib_for_backend with edge case"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps()
        
        # Test edge case - falls back to torch
        lib = ops._get_tensor_lib_for_backend("unknown")
        self.assertEqual(lib, self.mock_torch)
        mock_import.assert_called_with("torch")
    
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
        
        self.assertEqual(result, self.mock_torch_tensor)
        self.mock_torch.tensor.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_create_tensor_torch_with_requires_grad(self, mock_import, mock_backend_type, mock_get_backend):
        """Test create_tensor with Torch backend and requires_grad"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        self.mock_torch.tensor.return_value = self.mock_torch_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test tensor creation with requires_grad=True
        result = ops.create_tensor([1, 2, 3], requires_grad=True)
        
        self.assertEqual(result, self.mock_torch_tensor)
        self.mock_torch.tensor.assert_called_once_with([1, 2, 3], requires_grad=True)
        
        # Test tensor creation with requires_grad=False (should be removed)
        self.mock_torch.tensor.reset_mock()
        result = ops.create_tensor([1, 2, 3], requires_grad=False)
        
        self.assertEqual(result, self.mock_torch_tensor)
        self.mock_torch.tensor.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_create_tensor_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test create_tensor with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        self.mock_jax_numpy.array.return_value = self.mock_jax_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test tensor creation
        result = ops.create_tensor([1, 2, 3])
        
        self.assertEqual(result, self.mock_jax_tensor)
        self.mock_jax_numpy.array.assert_called_once_with([1, 2, 3])
        
        # Test that requires_grad is removed for JAX
        self.mock_jax_numpy.array.reset_mock()
        result = ops.create_tensor([1, 2, 3], requires_grad=True)
        
        self.assertEqual(result, self.mock_jax_tensor)
        self.mock_jax_numpy.array.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_create_tensor_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test create_tensor with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        mock_np.array.return_value = self.mock_numpy_array
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Test tensor creation
        result = ops.create_tensor([1, 2, 3])
        
        self.assertEqual(result, self.mock_numpy_array)
        mock_np.array.assert_called_once_with([1, 2, 3])
        
        # Test that requires_grad is removed for NUMBA
        mock_np.array.reset_mock()
        result = ops.create_tensor([1, 2, 3], requires_grad=True)
        
        self.assertEqual(result, self.mock_numpy_array)
        mock_np.array.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_create_tensor_unknown_backend(self, mock_import, mock_backend_type, mock_get_backend):
        """Test create_tensor with unknown backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        ops.backend = "unknown"  # Force unknown backend
        
        # Test that RuntimeError is raised
        with self.assertRaises(RuntimeError) as context:
            ops.create_tensor([1, 2, 3])
        
        self.assertIn("Unknown backend", str(context.exception))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_alias(self, mock_import, mock_backend_type, mock_get_backend):
        """Test tensor method alias"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        self.mock_torch.tensor.return_value = self.mock_torch_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test tensor alias
        result = ops.tensor([1, 2, 3])
        
        self.assertEqual(result, self.mock_torch_tensor)
        self.mock_torch.tensor.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_from_numpy_torch(self, mock_import, mock_backend_type, mock_get_backend):
        """Test from_numpy with Torch backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        self.mock_torch.from_numpy.return_value = self.mock_torch_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test from_numpy
        result = ops.from_numpy(self.mock_numpy_array)
        
        self.assertEqual(result, self.mock_torch_tensor)
        self.mock_torch.from_numpy.assert_called_once_with(self.mock_numpy_array)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_from_numpy_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test from_numpy with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        self.mock_jax_numpy.array.return_value = self.mock_jax_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test from_numpy
        result = ops.from_numpy(self.mock_numpy_array)
        
        self.assertEqual(result, self.mock_jax_tensor)
        self.mock_jax_numpy.array.assert_called_once_with(self.mock_numpy_array)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_from_numpy_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test from_numpy with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Test from_numpy - should return array directly
        result = ops.from_numpy(self.mock_numpy_array)
        
        self.assertEqual(result, self.mock_numpy_array)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_from_numpy_unknown_backend(self, mock_import, mock_backend_type, mock_get_backend):
        """Test from_numpy with unknown backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.TORCH = "torch"
        mock_import.return_value = self.mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        ops.backend = "unknown"  # Force unknown backend
        
        # Test that ValueError is raised
        with self.assertRaises(ValueError) as context:
            ops.from_numpy(self.mock_numpy_array)
        
        self.assertIn("Unknown backend", str(context.exception))


class TestTensorOpsMathematicalOperations(unittest.TestCase):
    """Test mathematical operations in TensorOps"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        self.BackendType = MagicMock()
        self.BackendType.TORCH = "torch"
        self.BackendType.JAX = "jax"
        self.BackendType.NUMBA = "numba"
        
        self.mock_torch = MagicMock()
        self.mock_jax_numpy = MagicMock()
        self.mock_numpy = MagicMock()
    
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
    def test_multiply_operation_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test multiply operation with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test multiply operation
        a = MagicMock()
        b = MagicMock()
        result = ops.multiply(a, b)
        
        # Should call jax numpy multiply
        self.mock_jax_numpy.multiply.assert_called_once_with(a, b)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_matmul_operation_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test matmul operation with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Test matmul operation
        a = MagicMock()
        b = MagicMock()
        result = ops.matmul(a, b)
        
        # Should call numpy matmul
        mock_np.matmul.assert_called_once_with(a, b)


class TestTensorOpsReductionOperations(unittest.TestCase):
    """Test reduction operations in TensorOps"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        self.BackendType = MagicMock()
        self.BackendType.TORCH = "torch"
        self.BackendType.JAX = "jax"
        self.BackendType.NUMBA = "numba"
        
        self.mock_torch = MagicMock()
        self.mock_jax_numpy = MagicMock()
        self.mock_numpy = MagicMock()
    
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
        
        # Test sum operation
        tensor = MagicMock()
        result = ops.sum(tensor)
        
        # Should call torch.sum
        self.mock_torch.sum.assert_called_once_with(tensor)
        
        # Test sum with dimension
        result = ops.sum(tensor, dim=1)
        self.mock_torch.sum.assert_called_with(tensor, dim=1)
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_mean_operation_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test mean operation with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test mean operation
        tensor = MagicMock()
        result = ops.mean(tensor)
        
        # Should call jax numpy mean
        self.mock_jax_numpy.mean.assert_called_once_with(tensor)
        
        # Test mean with axis
        result = ops.mean(tensor, axis=1)
        self.mock_jax_numpy.mean.assert_called_with(tensor, axis=1)


class TestTensorOpsUtilityOperations(unittest.TestCase):
    """Test utility operations in TensorOps"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        self.BackendType = MagicMock()
        self.BackendType.TORCH = "torch"
        self.BackendType.JAX = "jax"
        self.BackendType.NUMBA = "numba"
        
        self.mock_torch = MagicMock()
        self.mock_jax_numpy = MagicMock()
        self.mock_numpy = MagicMock()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_torch(self, mock_import, mock_backend_type, mock_get_backend):
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
    def test_ones_jax(self, mock_import, mock_backend_type, mock_get_backend):
        """Test ones operation with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_tensor = MagicMock()
        self.mock_jax_numpy.ones.return_value = mock_tensor
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test ones operation
        result = ops.ones((2, 3))
        
        self.assertEqual(result, mock_tensor)
        self.mock_jax_numpy.ones.assert_called_once_with((2, 3))
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops._np')
    def test_arange_numba(self, mock_np, mock_backend_type, mock_get_backend):
        """Test arange operation with NUMBA backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.NUMBA = "numba"
        mock_array = MagicMock()
        mock_np.arange.return_value = mock_array
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("numba")
        
        # Test arange operation
        result = ops.arange(5)
        
        self.assertEqual(result, mock_array)
        mock_np.arange.assert_called_once_with(5)


class TestTensorOpsContextManagers(unittest.TestCase):
    """Test context managers in TensorOps"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_backend_manager = MagicMock()
        self.mock_backend_manager.active_backend = None
        
        self.BackendType = MagicMock()
        self.BackendType.TORCH = "torch"
        self.BackendType.JAX = "jax"
        self.BackendType.NUMBA = "numba"
        
        self.mock_torch = MagicMock()
        self.mock_jax_numpy = MagicMock()
        self.mock_numpy = MagicMock()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_no_grad_torch(self, mock_import, mock_backend_type, mock_get_backend):
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
    @patch('hpfracc.ml.tensor_ops.nullcontext')
    def test_no_grad_jax(self, mock_nullcontext, mock_import, mock_backend_type, mock_get_backend):
        """Test no_grad context manager with JAX backend"""
        # Setup mocks
        mock_get_backend.return_value = self.mock_backend_manager
        mock_backend_type.JAX = "jax"
        mock_import.return_value = self.mock_jax_numpy
        mock_context = MagicMock()
        mock_nullcontext.return_value = mock_context
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("jax")
        
        # Test no_grad context manager
        result = ops.no_grad()
        
        self.assertEqual(result, mock_context)
        mock_nullcontext.assert_called_once()


if __name__ == '__main__':
    unittest.main()
