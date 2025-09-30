"""
Simple unittest tests for hpfracc/ml/tensor_ops.py
Testing basic functionality without complex mocking
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys


class TestTensorOpsBasic(unittest.TestCase):
    """Test basic TensorOps functionality"""
    
    def test_tensor_ops_import(self):
        """Test that TensorOps can be imported"""
        try:
            from hpfracc.ml.tensor_ops import TensorOps
            self.assertTrue(True, "TensorOps imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import TensorOps: {e}")
    
    def test_tensor_ops_class_exists(self):
        """Test that TensorOps class exists and has expected methods"""
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Check that class exists
        self.assertTrue(hasattr(TensorOps, '__init__'))
        
        # Check for key methods
        expected_methods = [
            'create_tensor', 'tensor', 'from_numpy', 'to_numpy',
            'add', 'subtract', 'multiply', 'divide', 'matmul',
            'sum', 'mean', 'max', 'min', 'norm', 'softmax',
            'zeros', 'ones', 'eye', 'arange', 'randn_like',
            'no_grad', 'stack', 'concatenate'
        ]
        
        for method in expected_methods:
            self.assertTrue(hasattr(TensorOps, method), 
                          f"TensorOps missing method: {method}")
    
    def test_tensor_ops_docstring(self):
        """Test that TensorOps has proper documentation"""
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Check class docstring
        self.assertIsNotNone(TensorOps.__doc__)
        self.assertIn("Unified tensor operations", TensorOps.__doc__)
        self.assertIn("backend", TensorOps.__doc__)
    
    def test_tensor_ops_init_method_exists(self):
        """Test that __init__ method exists and has correct signature"""
        from hpfracc.ml.tensor_ops import TensorOps
        import inspect
        
        # Check init signature
        sig = inspect.signature(TensorOps.__init__)
        params = list(sig.parameters.keys())
        
        # Should have 'self' and optional 'backend'
        self.assertIn('self', params)
        self.assertIn('backend', params)
    
    def test_tensor_ops_method_signatures(self):
        """Test that key methods have correct signatures"""
        from hpfracc.ml.tensor_ops import TensorOps
        import inspect
        
        # Test create_tensor signature
        sig = inspect.signature(TensorOps.create_tensor)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('data', params)
        
        # Test add signature
        sig = inspect.signature(TensorOps.add)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('a', params)
        self.assertIn('b', params)
        
        # Test sum signature
        sig = inspect.signature(TensorOps.sum)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('tensor', params)
    
    def test_tensor_ops_backend_resolution_methods(self):
        """Test that backend resolution methods exist"""
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Check for backend resolution methods
        self.assertTrue(hasattr(TensorOps, '_resolve_backend'))
        self.assertTrue(hasattr(TensorOps, '_get_tensor_lib_for_backend'))
        
        # Check method signatures
        import inspect
        
        sig = inspect.signature(TensorOps._resolve_backend)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('backend', params)
        
        sig = inspect.signature(TensorOps._get_tensor_lib_for_backend)
        params = list(sig.parameters.keys())
        self.assertIn('self', params)
        self.assertIn('backend', params)


class TestTensorOpsWithMockBackend(unittest.TestCase):
    """Test TensorOps with mocked backend to avoid import issues"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_ops_initialization_basic(self, mock_import, mock_backend_type, mock_get_backend):
        """Test basic TensorOps initialization"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        # Mock BackendType enum
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        mock_backend_type.AUTO = "auto"
        
        # Mock tensor library
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test initialization
        ops = TensorOps("torch")
        
        # Verify basic attributes exist
        self.assertTrue(hasattr(ops, 'backend'))
        self.assertTrue(hasattr(ops, 'tensor_lib'))
        self.assertTrue(hasattr(ops, 'backend_manager'))
        
        # Verify backend manager was called
        mock_get_backend.assert_called_once()
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    def test_tensor_ops_initialization_with_invalid_backend(self, mock_backend_type, mock_get_backend):
        """Test TensorOps initialization with invalid backend"""
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
        with self.assertRaises(ValueError):
            TensorOps("invalid")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_ops_create_tensor_basic(self, mock_import, mock_backend_type, mock_get_backend):
        """Test basic create_tensor functionality"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        mock_backend_type.JAX = "jax"
        mock_backend_type.NUMBA = "numba"
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test create_tensor
        result = ops.create_tensor([1, 2, 3])
        
        # Verify tensor creation was called
        mock_torch.tensor.assert_called_once_with([1, 2, 3])
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_tensor_ops_tensor_alias(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that tensor method is an alias for create_tensor"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test tensor alias
        result = ops.tensor([1, 2, 3])
        
        # Verify tensor creation was called
        mock_torch.tensor.assert_called_once_with([1, 2, 3])


class TestTensorOpsMathematicalOperations(unittest.TestCase):
    """Test mathematical operations in TensorOps"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_add_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that add operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that add method exists and can be called
        a = MagicMock()
        b = MagicMock()
        
        # Should not raise an error
        try:
            result = ops.add(a, b)
            # Verify torch.add was called
            mock_torch.add.assert_called_once()
        except Exception as e:
            self.fail(f"add operation failed: {e}")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_multiply_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that multiply operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that multiply method exists and can be called
        a = MagicMock()
        b = MagicMock()
        
        # Should not raise an error
        try:
            result = ops.multiply(a, b)
            # Verify torch.multiply was called
            mock_torch.multiply.assert_called_once()
        except Exception as e:
            self.fail(f"multiply operation failed: {e}")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_matmul_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that matmul operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that matmul method exists and can be called
        a = MagicMock()
        b = MagicMock()
        
        # Should not raise an error
        try:
            result = ops.matmul(a, b)
            # Verify torch.matmul was called
            mock_torch.matmul.assert_called_once()
        except Exception as e:
            self.fail(f"matmul operation failed: {e}")


class TestTensorOpsReductionOperations(unittest.TestCase):
    """Test reduction operations in TensorOps"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_sum_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that sum operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that sum method exists and can be called
        tensor = MagicMock()
        
        # Should not raise an error
        try:
            result = ops.sum(tensor)
            # Verify torch.sum was called
            mock_torch.sum.assert_called_once()
        except Exception as e:
            self.fail(f"sum operation failed: {e}")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_mean_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that mean operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that mean method exists and can be called
        tensor = MagicMock()
        
        # Should not raise an error
        try:
            result = ops.mean(tensor)
            # Verify torch.mean was called
            mock_torch.mean.assert_called_once()
        except Exception as e:
            self.fail(f"mean operation failed: {e}")


class TestTensorOpsUtilityOperations(unittest.TestCase):
    """Test utility operations in TensorOps"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_zeros_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that zeros operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.zeros.return_value = mock_tensor
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that zeros method exists and can be called
        try:
            result = ops.zeros((3, 4))
            # Verify torch.zeros was called
            mock_torch.zeros.assert_called_once_with((3, 4))
        except Exception as e:
            self.fail(f"zeros operation failed: {e}")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_ones_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that ones operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.ones.return_value = mock_tensor
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that ones method exists and can be called
        try:
            result = ops.ones((2, 3))
            # Verify torch.ones was called
            mock_torch.ones.assert_called_once_with((2, 3))
        except Exception as e:
            self.fail(f"ones operation failed: {e}")
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_eye_operation_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that eye operation exists and can be called"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_torch.eye.return_value = mock_tensor
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that eye method exists and can be called
        try:
            result = ops.eye(3)
            # Verify torch.eye was called
            mock_torch.eye.assert_called_once_with(3)
        except Exception as e:
            self.fail(f"eye operation failed: {e}")


class TestTensorOpsContextManagers(unittest.TestCase):
    """Test context managers in TensorOps"""
    
    @patch('hpfracc.ml.tensor_ops.get_backend_manager')
    @patch('hpfracc.ml.tensor_ops.BackendType')
    @patch('hpfracc.ml.tensor_ops.importlib.import_module')
    def test_no_grad_exists(self, mock_import, mock_backend_type, mock_get_backend):
        """Test that no_grad context manager exists"""
        # Setup mocks
        mock_backend_manager = MagicMock()
        mock_backend_manager.active_backend = None
        mock_get_backend.return_value = mock_backend_manager
        
        mock_backend_type.TORCH = "torch"
        
        mock_torch = MagicMock()
        mock_context = MagicMock()
        mock_torch.no_grad.return_value = mock_context
        mock_import.return_value = mock_torch
        
        from hpfracc.ml.tensor_ops import TensorOps
        
        ops = TensorOps("torch")
        
        # Test that no_grad method exists and can be called
        try:
            result = ops.no_grad()
            # Verify torch.no_grad was called
            mock_torch.no_grad.assert_called_once()
        except Exception as e:
            self.fail(f"no_grad operation failed: {e}")


if __name__ == '__main__':
    unittest.main()
