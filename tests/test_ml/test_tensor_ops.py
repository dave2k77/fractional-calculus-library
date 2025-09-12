"""
Tests for unified tensor operations module.

This module tests the TensorOps class which provides consistent tensor operations
across PyTorch, JAX, and NUMBA backends.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor
from hpfracc.ml.backends import BackendType, BackendManager


class TestTensorOps:
    """Test the TensorOps class for unified tensor operations."""

    def test_init_with_torch_backend(self):
        """Test TensorOps initialization with PyTorch backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            assert ops.backend == BackendType.TORCH
            assert ops.tensor_lib == torch

    def test_init_with_jax_backend(self):
        """Test TensorOps initialization with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.JAX
            mock_manager.get_tensor_lib.return_value = MagicMock()  # Mock JAX
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.JAX)
            assert ops.backend == BackendType.JAX

    def test_init_with_numba_backend(self):
        """Test TensorOps initialization with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.get_tensor_lib.return_value = np
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            assert ops.backend == BackendType.NUMBA

    def test_init_with_auto_backend(self):
        """Test TensorOps initialization with AUTO backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.AUTO)
            assert ops.backend == BackendType.TORCH  # Should resolve to active backend

    def test_init_with_none_backend(self):
        """Test TensorOps initialization with None backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps()
            assert ops.backend == BackendType.TORCH

    def test_create_tensor_torch(self):
        """Test tensor creation with PyTorch backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_manager.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3], requires_grad=True)
            assert isinstance(result, torch.Tensor)

    def test_create_tensor_jax(self):
        """Test tensor creation with JAX backend (filters requires_grad)."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.JAX
            mock_manager.get_tensor_lib.return_value = MagicMock()
            mock_manager.create_tensor.return_value = np.array([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.JAX)
            result = ops.create_tensor([1, 2, 3], requires_grad=True, dtype='float32')
            
            # Should filter out requires_grad for JAX
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3], dtype='float32')

    def test_create_tensor_numba(self):
        """Test tensor creation with NUMBA backend (filters requires_grad)."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.get_tensor_lib.return_value = np
            mock_manager.create_tensor.return_value = np.array([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            result = ops.create_tensor([1, 2, 3], requires_grad=True, dtype='float32')
            
            # Should filter out requires_grad for NUMBA
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3], dtype='float32')

    def test_tensor_alias(self):
        """Test that tensor() is an alias for create_tensor()."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_manager.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.tensor([1, 2, 3])
            
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3])

    def test_no_grad_torch(self):
        """Test no_grad context manager for PyTorch."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            context = ops.no_grad()
            
            # Check that it returns a no_grad context manager
            assert hasattr(context, '__enter__')
            assert hasattr(context, '__exit__')

    def test_no_grad_jax(self):
        """Test no_grad context manager for JAX."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.JAX
            mock_manager.get_tensor_lib.return_value = MagicMock()
            mock_get_manager.return_value = mock_manager
            
            with patch('jax.disable_jit') as mock_disable_jit:
                mock_disable_jit.return_value = MagicMock()
                ops = TensorOps(BackendType.JAX)
                context = ops.no_grad()
                
                mock_disable_jit.assert_called_once()

    def test_no_grad_numba(self):
        """Test no_grad context manager for NUMBA."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.get_tensor_lib.return_value = np
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.NUMBA)
            context = ops.no_grad()
            
            from contextlib import nullcontext
            assert isinstance(context, nullcontext)

    def test_zeros_torch(self):
        """Test zeros creation with PyTorch."""
        with patch('torch.zeros') as mock_zeros:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
            mock_zeros.return_value = mock_result
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.zeros((2, 3), dtype=torch.float32)
            
            mock_zeros.assert_called_once_with((2, 3), dtype=torch.float32)

    def test_ones_torch(self):
        """Test ones creation with PyTorch."""
        with patch('torch.ones') as mock_ones:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
            mock_ones.return_value = mock_result
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.ones((2, 3), dtype=torch.float32)
            
            mock_ones.assert_called_once_with((2, 3), dtype=torch.float32)

    def test_eye_torch(self):
        """Test identity matrix creation with PyTorch."""
        with patch('torch.eye') as mock_eye:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
            mock_eye.return_value = mock_result
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.eye(3, dtype=torch.float32)
            
            mock_eye.assert_called_once_with(3, dtype=torch.float32)

    def test_arange_torch(self):
        """Test arange creation with PyTorch."""
        with patch('torch.arange') as mock_arange:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
            mock_arange.return_value = mock_result
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.arange(0, 5, 1, dtype=torch.float32)
            
            mock_arange.assert_called_once_with(0, 5, 1, dtype=torch.float32)

    def test_linspace_torch(self):
        """Test linspace creation with PyTorch."""
        with patch('torch.linspace') as mock_linspace:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([0.0, 0.111, 0.222, 0.333, 0.444, 0.556, 0.667, 0.778, 0.889, 1.0], dtype=torch.float32)
            mock_linspace.return_value = mock_result
            
            ops = TensorOps(BackendType.TORCH)
            result = ops.linspace(0, 1, 10, dtype=torch.float32)
            
            mock_linspace.assert_called_once_with(0, 1, 10, dtype=torch.float32)

    def test_zeros_like_torch(self):
        """Test zeros_like creation with PyTorch."""
        with patch('torch.zeros_like') as mock_zeros_like:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
            mock_zeros_like.return_value = mock_result
            
            input_tensor = torch.tensor([[1, 2], [3, 4]])
            ops = TensorOps(BackendType.TORCH)
            result = ops.zeros_like(input_tensor, dtype=torch.float32)
            
            mock_zeros_like.assert_called_once_with(input_tensor, dtype=torch.float32)

    def test_zeros_like_numba_with_shape(self):
        """Test zeros_like creation with NUMBA when tensor has shape."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.get_tensor_lib.return_value = np
            mock_get_manager.return_value = mock_manager
            
            input_tensor = np.array([[1, 2], [3, 4]])
            ops = TensorOps(BackendType.NUMBA)
            result = ops.zeros_like(input_tensor, dtype=np.float32)
            
            # For NUMBA backend, the function directly uses numpy, so we just check the result
            assert result.shape == input_tensor.shape
            assert result.dtype == np.float32

    def test_zeros_like_numba_without_shape(self):
        """Test zeros_like creation with NUMBA when tensor doesn't have shape."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.get_tensor_lib.return_value = np
            mock_get_manager.return_value = mock_manager
            
            input_tensor = 5  # Scalar without shape
            ops = TensorOps(BackendType.NUMBA)
            result = ops.zeros_like(input_tensor, dtype=np.float32)
            
            # For NUMBA backend, the function directly uses numpy, so we just check the result
            assert result.shape == (1,)
            assert result.dtype == np.float32

    def test_sqrt_torch(self):
        """Test sqrt operation with PyTorch."""
        with patch('torch.sqrt') as mock_sqrt:
            # Create a mock return value without calling the real function
            mock_result = torch.tensor([2.0, 3.0, 4.0])
            mock_sqrt.return_value = mock_result
            
            input_tensor = torch.tensor([4, 9, 16])
            ops = TensorOps(BackendType.TORCH)
            result = ops.sqrt(input_tensor)
            
            mock_sqrt.assert_called_once_with(input_tensor)

    def test_unknown_backend_error(self):
        """Test that unknown backend raises RuntimeError."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"  # Manually set invalid backend
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])

    def test_unknown_backend_error_no_grad(self):
        """Test that unknown backend raises RuntimeError in no_grad."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = TensorOps(BackendType.TORCH)
            ops.backend = "unknown_backend"  # Manually set invalid backend
            
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.no_grad()


class TestTensorOpsFunctions:
    """Test the module-level functions."""

    def test_get_tensor_ops(self):
        """Test get_tensor_ops function."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.get_tensor_lib.return_value = torch
            mock_get_manager.return_value = mock_manager
            
            ops = get_tensor_ops(BackendType.TORCH)
            assert isinstance(ops, TensorOps)
            assert ops.backend == BackendType.TORCH

    def test_create_tensor_function(self):
        """Test create_tensor function."""
        with patch('hpfracc.ml.tensor_ops.get_tensor_ops') as mock_get_ops:
            mock_ops = MagicMock()
            mock_ops.create_tensor.return_value = torch.tensor([1, 2, 3])
            mock_get_ops.return_value = mock_ops
            
            result = create_tensor([1, 2, 3])
            
            mock_get_ops.assert_called_once()
            mock_ops.create_tensor.assert_called_once_with([1, 2, 3])
            assert isinstance(result, torch.Tensor)


class TestTensorOpsIntegration:
    """Integration tests for TensorOps with real backends."""

    def test_torch_integration(self):
        """Test TensorOps with real PyTorch backend."""
        # This test uses the actual PyTorch backend
        ops = TensorOps(BackendType.TORCH)
        
        # Test basic operations
        zeros = ops.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert torch.allclose(zeros, torch.zeros(2, 3))
        
        ones = ops.ones((2, 3))
        assert ones.shape == (2, 3)
        assert torch.allclose(ones, torch.ones(2, 3))
        
        eye = ops.eye(3)
        assert eye.shape == (3, 3)
        assert torch.allclose(eye, torch.eye(3))
        
        arange = ops.arange(0, 5)
        assert arange.shape == (5,)
        assert torch.allclose(arange, torch.arange(0, 5))
        
        linspace = ops.linspace(0, 1, 5)
        assert linspace.shape == (5,)
        
        # Test with gradient computation
        x = ops.create_tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ops.sqrt(x)
        assert y.requires_grad
        
        # Test no_grad context
        with ops.no_grad():
            z = ops.sqrt(x)
            assert not z.requires_grad

    def test_numpy_integration(self):
        """Test TensorOps with NUMBA backend (using numpy)."""
        ops = TensorOps(BackendType.NUMBA)
        
        # Test basic operations
        zeros = ops.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert np.allclose(zeros, np.zeros((2, 3)))
        
        ones = ops.ones((2, 3))
        assert ones.shape == (2, 3)
        assert np.allclose(ones, np.ones((2, 3)))
        
        eye = ops.eye(3)
        assert eye.shape == (3, 3)
        assert np.allclose(eye, np.eye(3))
        
        arange = ops.arange(0, 5)
        assert arange.shape == (5,)
        assert np.allclose(arange, np.arange(0, 5))
        
        linspace = ops.linspace(0, 1, 5)
        assert linspace.shape == (5,)
        
        # Test sqrt
        x = np.array([4, 9, 16])
        y = ops.sqrt(x)
        assert np.allclose(y, np.sqrt(x))
