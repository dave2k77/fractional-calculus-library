"""
Comprehensive tests for TensorOps module.

This module tests the unified tensor operations across different backends.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor, switch_backend
from hpfracc.ml.backends import BackendType


class TestTensorOps:
    """Test cases for TensorOps class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps()
            assert ops.backend == BackendType.TORCH

    def test_initialization_custom_backend(self):
        """Test initialization with custom backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.JAX
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.JAX)
            assert ops.backend == BackendType.JAX

    def test_initialization_auto_backend(self):
        """Test initialization with AUTO backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.AUTO)
            assert ops.backend == BackendType.TORCH

    def test_create_tensor_torch(self):
        """Test creating tensor with PyTorch backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_manager.create_tensor.return_value = "torch_tensor"
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.TORCH)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            assert result == "torch_tensor"
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3], requires_grad=True)

    def test_create_tensor_jax(self):
        """Test creating tensor with JAX backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.JAX
            mock_manager.create_tensor.return_value = "jax_tensor"
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.JAX)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            assert result == "jax_tensor"
            # JAX should filter out requires_grad
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3])

    def test_create_tensor_numba(self):
        """Test creating tensor with NUMBA backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.NUMBA
            mock_manager.create_tensor.return_value = "numba_tensor"
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.NUMBA)
            result = ops.create_tensor([1, 2, 3], requires_grad=True)
            assert result == "numba_tensor"
            # NUMBA should filter out requires_grad
            mock_manager.create_tensor.assert_called_once_with([1, 2, 3])

    def test_create_tensor_unknown_backend(self):
        """Test creating tensor with unknown backend."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = "UNKNOWN"
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps("UNKNOWN")
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])

    def test_matmul_tensors(self):
        """Test matrix multiplication of tensors."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(3, 4)
        result = ops.matmul(tensor1, tensor2)
        
        # Check that result is a tensor with correct shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 4)

    def test_einsum_operation(self):
        """Test Einstein summation operation."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor1 = torch.randn(2, 3)
        tensor2 = torch.randn(3, 4)
        result = ops.einsum("ij,jk->ik", tensor1, tensor2)
        
        # Check that result is a tensor with correct shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 4)

    def test_transpose_tensor(self):
        """Test tensor transpose."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_tensor_lib = Mock()
            mock_tensor_lib.permute.return_value = "transposed_tensor"
            mock_manager.get_tensor_lib.return_value = mock_tensor_lib
            mock_backend_manager.return_value = mock_manager

            # Create a mock tensor that has the permute method
            mock_tensor = Mock()
            mock_tensor.permute.return_value = "transposed_tensor"

            ops = TensorOps(BackendType.TORCH)
            result = ops.transpose(mock_tensor, (1, 0))
            assert result == "transposed_tensor"
            mock_tensor.permute.assert_called_once_with((1, 0))

    def test_reshape_tensor(self):
        """Test tensor reshape."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_tensor_lib = Mock()
            mock_tensor_lib.reshape.return_value = "reshaped_tensor"
            mock_manager.get_tensor_lib.return_value = mock_tensor_lib
            mock_backend_manager.return_value = mock_manager

            # Create a mock tensor that has the reshape method
            mock_tensor = Mock()
            mock_tensor.reshape.return_value = "reshaped_tensor"

            ops = TensorOps(BackendType.TORCH)
            result = ops.reshape(mock_tensor, (2, 3))
            assert result == "reshaped_tensor"
            mock_tensor.reshape.assert_called_once_with((2, 3))

    def test_sum_tensor(self):
        """Test tensor sum."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_tensor_lib = Mock()
            mock_tensor_lib.sum.return_value = "sum_result"
            mock_manager.get_tensor_lib.return_value = mock_tensor_lib
            mock_backend_manager.return_value = mock_manager

            # Create a mock tensor that has the sum method
            mock_tensor = Mock()
            mock_tensor.sum.return_value = "sum_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.sum(mock_tensor)
            assert result == "sum_result"
            mock_tensor.sum.assert_called_once_with(dim=None, keepdim=False)

    def test_mean_tensor(self):
        """Test tensor mean."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_tensor_lib = Mock()
            mock_tensor_lib.mean.return_value = "mean_result"
            mock_manager.get_tensor_lib.return_value = mock_tensor_lib
            mock_backend_manager.return_value = mock_manager

            # Create a mock tensor that has the mean method
            mock_tensor = Mock()
            mock_tensor.mean.return_value = "mean_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.mean(mock_tensor)
            assert result == "mean_result"
            mock_tensor.mean.assert_called_once_with(dim=None, keepdim=False)

    def test_sqrt_tensor(self):
        """Test tensor square root."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor = torch.tensor([4.0, 9.0, 16.0])
        result = ops.sqrt(tensor)
        
        # Check that result is a tensor
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([2.0, 3.0, 4.0]))

    def test_log_tensor(self):
        """Test tensor logarithm."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor = torch.tensor([1.0, 2.718, 7.389])  # e^0, e^1, e^2
        result = ops.log(tensor)
        
        # Check that result is a tensor
        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, torch.tensor([0.0, 1.0, 2.0]), atol=1e-3)

    def test_stack_tensors(self):
        """Test tensor stacking."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5, 6])
        result = ops.stack([tensor1, tensor2], dim=0)
        
        # Check that result is a tensor with correct shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.tensor([[1, 2, 3], [4, 5, 6]]))

    def test_cat_tensors(self):
        """Test tensor concatenation."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        tensor1 = torch.tensor([[1, 2], [3, 4]])
        tensor2 = torch.tensor([[5, 6], [7, 8]])
        result = ops.cat([tensor1, tensor2], dim=0)
        
        # Check that result is a tensor with correct shape
        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 2)
        assert torch.allclose(result, torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]))

    def test_zeros_creation(self):
        """Test creating zeros tensor."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        result = ops.zeros((2, 3))
        
        # Check that result is a tensor with correct shape and values
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.zeros(2, 3))

    def test_ones_creation(self):
        """Test creating ones tensor."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        result = ops.ones((2, 3))
        
        # Check that result is a tensor with correct shape and values
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.ones(2, 3))

    def test_eye_creation(self):
        """Test creating identity matrix."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        result = ops.eye(3)
        
        # Check that result is a tensor with correct shape and values
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)
        assert torch.allclose(result, torch.eye(3))

    def test_arange_creation(self):
        """Test creating arange tensor."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        result = ops.arange(0, 5, 1)
        
        # Check that result is a tensor with correct shape and values
        assert isinstance(result, torch.Tensor)
        assert result.shape == (5,)
        assert torch.allclose(result, torch.arange(0, 5, 1))

    def test_linspace_creation(self):
        """Test creating linspace tensor."""
        import torch
        
        ops = TensorOps(BackendType.TORCH)
        result = ops.linspace(0.0, 1.0, 10)
        
        # Check that result is a tensor with correct shape and values
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)
        expected = torch.linspace(0.0, 1.0, 10)
        assert torch.allclose(result, expected)

    def test_softmax_activation(self):
        """Test softmax activation function."""
        with patch('torch.softmax') as mock_softmax:
            mock_softmax.return_value = "softmax_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.softmax("tensor", dim=0)
            assert result == "softmax_result"
            mock_softmax.assert_called_once_with("tensor", dim=0)

    def test_relu_activation(self):
        """Test ReLU activation function."""
        with patch('torch.relu') as mock_relu:
            mock_relu.return_value = "relu_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.relu("tensor")
            assert result == "relu_result"
            mock_relu.assert_called_once_with("tensor")

    def test_sigmoid_activation(self):
        """Test sigmoid activation function."""
        with patch('torch.sigmoid') as mock_sigmoid:
            mock_sigmoid.return_value = "sigmoid_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.sigmoid("tensor")
            assert result == "sigmoid_result"
            mock_sigmoid.assert_called_once_with("tensor")

    def test_tanh_activation(self):
        """Test tanh activation function."""
        with patch('torch.tanh') as mock_tanh:
            mock_tanh.return_value = "tanh_result"

            ops = TensorOps(BackendType.TORCH)
            result = ops.tanh("tensor")
            assert result == "tanh_result"
            mock_tanh.assert_called_once_with("tensor")

    def test_error_handling_unsupported_operation(self):
        """Test error handling for unsupported operations."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_tensor_lib = Mock()
            mock_tensor_lib.unknown_op.side_effect = AttributeError("Unknown operation")
            mock_manager.get_tensor_lib.return_value = mock_tensor_lib
            mock_backend_manager.return_value = mock_manager

            ops = TensorOps(BackendType.TORCH)
            with pytest.raises(AttributeError):
                ops.unknown_op("tensor")

    def test_warning_for_deprecated_operations(self):
        """Test warnings for deprecated operations."""
        # This test is a placeholder for future deprecated operations
        # Currently there are no deprecated operations in TensorOps
        pass


class TestTensorOpsFunctions:
    """Test cases for module-level functions."""

    def test_get_tensor_ops(self):
        """Test get_tensor_ops function."""
        with patch('hpfracc.ml.tensor_ops.get_backend_manager') as mock_backend_manager:
            mock_manager = Mock()
            mock_manager.active_backend = BackendType.TORCH
            mock_backend_manager.return_value = mock_manager

            ops = get_tensor_ops(BackendType.TORCH)
            assert isinstance(ops, TensorOps)
            assert ops.backend == BackendType.TORCH

    def test_create_tensor_function(self):
        """Test create_tensor function."""
        with patch('hpfracc.ml.tensor_ops.get_tensor_ops') as mock_get_tensor_ops:
            mock_ops = Mock()
            mock_ops.create_tensor.return_value = "tensor"
            mock_get_tensor_ops.return_value = mock_ops

            result = create_tensor([1, 2, 3])
            assert result == "tensor"
            mock_ops.create_tensor.assert_called_once_with([1, 2, 3])

    def test_switch_backend_function(self):
        """Test switch_backend function."""
        with patch('hpfracc.ml.backends.switch_backend') as mock_switch_backend_manager:
            mock_switch_backend_manager.return_value = True

            switch_backend(BackendType.JAX)
            # The function calls the backend manager's switch_backend function
            mock_switch_backend_manager.assert_called_once_with(BackendType.JAX)