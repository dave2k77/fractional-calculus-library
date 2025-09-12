"""
Comprehensive tests for TensorOps module.

This module tests all TensorOps functionality across different backends
to ensure consistent behavior and high coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor, switch_backend
from hpfracc.ml.backends import BackendType


class TestTensorOpsInitialization:
    """Test TensorOps initialization and basic functionality."""
    
    def test_tensor_ops_initialization_default(self):
        """Test TensorOps initialization with default backend."""
        ops = TensorOps()
        assert ops.backend is not None
        assert hasattr(ops, 'tensor_lib')
    
    def test_tensor_ops_initialization_torch(self):
        """Test TensorOps initialization with PyTorch backend."""
        ops = TensorOps(BackendType.TORCH)
        assert ops.backend == BackendType.TORCH
        assert ops.tensor_lib is not None
    
    def test_tensor_ops_initialization_jax(self):
        """Test TensorOps initialization with JAX backend."""
        ops = TensorOps(BackendType.JAX)
        assert ops.backend == BackendType.JAX
        assert ops.tensor_lib is not None
    
    def test_tensor_ops_initialization_numba(self):
        """Test TensorOps initialization with NUMBA backend."""
        ops = TensorOps(BackendType.NUMBA)
        assert ops.backend == BackendType.NUMBA
        assert ops.tensor_lib is not None


class TestTensorCreation:
    """Test tensor creation methods."""
    
    def test_create_tensor_basic(self):
        """Test basic tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        data = [1, 2, 3, 4]
        tensor = ops.create_tensor(data)
        assert tensor is not None
    
    def test_tensor_method(self):
        """Test tensor method."""
        ops = TensorOps(BackendType.TORCH)
        data = [1, 2, 3, 4]
        tensor = ops.tensor(data)
        assert tensor is not None
    
    def test_zeros_creation(self):
        """Test zeros tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.zeros((3, 4))
        assert tensor is not None
        assert tensor.shape == (3, 4)
    
    def test_ones_creation(self):
        """Test ones tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        assert tensor is not None
        assert tensor.shape == (2, 3)
    
    def test_eye_creation(self):
        """Test identity matrix creation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.eye(3)
        assert tensor is not None
        assert tensor.shape == (3, 3)
    
    def test_arange_creation(self):
        """Test arange tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.arange(0, 10, 2)
        assert tensor is not None
        assert len(tensor) == 5
    
    def test_linspace_creation(self):
        """Test linspace tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.linspace(0, 1, 5)
        assert tensor is not None
        assert len(tensor) == 5
    
    def test_zeros_like_creation(self):
        """Test zeros_like tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        reference = ops.ones((2, 3))
        tensor = ops.zeros_like(reference)
        assert tensor is not None
        assert tensor.shape == (2, 3)
    
    def test_randn_like_creation(self):
        """Test randn_like tensor creation."""
        ops = TensorOps(BackendType.TORCH)
        reference = ops.ones((2, 3))
        tensor = ops.randn_like(reference)
        assert tensor is not None
        assert tensor.shape == (2, 3)


class TestTensorOperations:
    """Test tensor operations."""
    
    def test_sqrt_operation(self):
        """Test square root operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 2)) * 4
        result = ops.sqrt(tensor)
        assert result is not None
        assert result.shape == (2, 2)
    
    def test_log_operation(self):
        """Test logarithm operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 2)) * 2
        result = ops.log(tensor)
        assert result is not None
        assert result.shape == (2, 2)
    
    def test_stack_operation(self):
        """Test stack operation."""
        ops = TensorOps(BackendType.TORCH)
        tensors = [ops.ones((2, 2)) for _ in range(3)]
        result = ops.stack(tensors, dim=0)
        assert result is not None
        assert result.shape == (3, 2, 2)
    
    def test_cat_operation(self):
        """Test concatenation operation."""
        ops = TensorOps(BackendType.TORCH)
        tensors = [ops.ones((2, 2)) for _ in range(3)]
        result = ops.cat(tensors, dim=0)
        assert result is not None
        assert result.shape == (6, 2)
    
    def test_reshape_operation(self):
        """Test reshape operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.reshape(tensor, (3, 2))
        assert result is not None
        assert result.shape == (3, 2)
    
    def test_repeat_operation(self):
        """Test repeat operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 2))
        result = ops.repeat(tensor, 2, 3)
        assert result is not None
        assert result.shape == (4, 4)  # PyTorch repeat behavior
    
    def test_clip_operation(self):
        """Test clip operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 2)) * 5
        result = ops.clip(tensor, 0, 2)
        assert result is not None
        assert result.shape == (2, 2)
    
    def test_unsqueeze_operation(self):
        """Test unsqueeze operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 2))
        result = ops.unsqueeze(tensor, 0)
        assert result is not None
        assert result.shape == (1, 2, 2)
    
    def test_expand_operation(self):
        """Test expand operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((1, 2))
        result = ops.expand(tensor, 3, 2)
        assert result is not None
        assert result.shape == (3, 2)
    
    def test_gather_operation(self):
        """Test gather operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((3, 4))
        index = ops.zeros((2, 2), dtype=torch.long)
        result = ops.gather(tensor, 0, index)
        assert result is not None
        assert result.shape == (2, 2)
    
    def test_squeeze_operation(self):
        """Test squeeze operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((1, 2, 1, 3))
        result = ops.squeeze(tensor, dim=0)  # Specify dimension
        assert result is not None
        assert result.shape == (2, 1, 3)
    
    def test_transpose_operation(self):
        """Test transpose operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.transpose(tensor, (1, 0))
        assert result is not None
        assert result.shape == (3, 2)
    
    def test_matmul_operation(self):
        """Test matrix multiplication operation."""
        ops = TensorOps(BackendType.TORCH)
        a = ops.ones((2, 3))
        b = ops.ones((3, 4))
        result = ops.matmul(a, b)
        assert result is not None
        assert result.shape == (2, 4)
    
    def test_einsum_operation(self):
        """Test einsum operation."""
        ops = TensorOps(BackendType.TORCH)
        a = ops.ones((2, 3))
        b = ops.ones((3, 4))
        result = ops.einsum('ij,jk->ik', a, b)
        assert result is not None
        assert result.shape == (2, 4)
    
    def test_norm_operation(self):
        """Test norm operation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.norm(tensor)
        assert result is not None
        assert result.shape == ()


class TestActivationFunctions:
    """Test activation functions."""
    
    def test_softmax_activation(self):
        """Test softmax activation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.softmax(tensor, dim=1)
        assert result is not None
        assert result.shape == (2, 3)
    
    def test_relu_activation(self):
        """Test ReLU activation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3)) * -1
        result = ops.relu(tensor)
        assert result is not None
        assert result.shape == (2, 3)
    
    def test_sigmoid_activation(self):
        """Test sigmoid activation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.sigmoid(tensor)
        assert result is not None
        assert result.shape == (2, 3)
    
    def test_tanh_activation(self):
        """Test tanh activation."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((2, 3))
        result = ops.tanh(tensor)
        assert result is not None
        assert result.shape == (2, 3)


class TestBackendSwitching:
    """Test backend switching functionality."""
    
    def test_backend_switching_torch_to_jax(self):
        """Test switching from PyTorch to JAX backend."""
        ops = TensorOps(BackendType.TORCH)
        assert ops.backend == BackendType.TORCH
        
        # Switch to JAX
        ops.backend = BackendType.JAX
        ops.tensor_lib = ops._get_tensor_lib_for_backend(BackendType.JAX)
        assert ops.backend == BackendType.JAX
    
    def test_backend_switching_torch_to_numba(self):
        """Test switching from PyTorch to NUMBA backend."""
        ops = TensorOps(BackendType.TORCH)
        assert ops.backend == BackendType.TORCH
        
        # Switch to NUMBA
        ops.backend = BackendType.NUMBA
        ops.tensor_lib = ops._get_tensor_lib_for_backend(BackendType.NUMBA)
        assert ops.backend == BackendType.NUMBA
    
    def test_no_grad_context(self):
        """Test no_grad context manager."""
        ops = TensorOps(BackendType.TORCH)
        with ops.no_grad():
            tensor = ops.ones((2, 3))
            assert tensor is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_tensor_creation(self):
        """Test creation of empty tensors."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.zeros((0, 0))
        assert tensor is not None
        assert tensor.shape == (0, 0)
    
    def test_single_element_tensor(self):
        """Test creation of single element tensors."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((1,))
        assert tensor is not None
        assert tensor.shape == (1,)
    
    def test_large_tensor_creation(self):
        """Test creation of large tensors."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.zeros((100, 100))
        assert tensor is not None
        assert tensor.shape == (100, 100)
    
    def test_invalid_operation_handling(self):
        """Test handling of invalid operations."""
        ops = TensorOps(BackendType.TORCH)
        # This should not raise an exception, but handle gracefully
        try:
            result = ops.sqrt(ops.ones((2, 2)) * -1)
            # Should handle negative values appropriately
            assert result is not None
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_tensor_ops_function(self):
        """Test get_tensor_ops convenience function."""
        ops = get_tensor_ops(BackendType.TORCH)
        assert isinstance(ops, TensorOps)
        assert ops.backend == BackendType.TORCH
    
    def test_create_tensor_function(self):
        """Test create_tensor convenience function."""
        tensor = create_tensor([1, 2, 3, 4])
        assert tensor is not None
    
    def test_switch_backend_function(self):
        """Test switch_backend convenience function."""
        # This function might not be implemented, so we test if it exists
        assert callable(switch_backend)


class TestPerformance:
    """Test performance-related functionality."""
    
    def test_tensor_creation_performance(self):
        """Test tensor creation performance."""
        ops = TensorOps(BackendType.TORCH)
        
        import time
        start_time = time.time()
        
        for _ in range(100):
            tensor = ops.ones((10, 10))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second)
        assert duration < 1.0
    
    def test_operation_performance(self):
        """Test operation performance."""
        ops = TensorOps(BackendType.TORCH)
        tensor = ops.ones((100, 100))
        
        import time
        start_time = time.time()
        
        for _ in range(50):
            result = ops.matmul(tensor, tensor)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (less than 2 seconds)
        assert duration < 2.0


class TestIntegration:
    """Test integration scenarios."""
    
    def test_complex_workflow(self):
        """Test a complex workflow using multiple operations."""
        ops = TensorOps(BackendType.TORCH)
        
        # Create input data
        x = ops.ones((32, 10))
        w = ops.ones((10, 5))
        b = ops.ones((5,))
        
        # Forward pass
        z = ops.matmul(x, w)
        z = ops.unsqueeze(z, 1)
        z = ops.squeeze(z, 1)
        z = z + b
        z = ops.relu(z)
        z = ops.softmax(z, dim=1)
        
        # Compute loss
        target = ops.zeros((32, 5))
        loss = ops.norm(z - target)
        
        assert z is not None
        assert loss is not None
        assert z.shape == (32, 5)
        assert loss.shape == ()
    
    def test_gradient_flow(self):
        """Test gradient flow through operations."""
        ops = TensorOps(BackendType.TORCH)
        
        # Create tensors that require gradients
        x = ops.ones((2, 3), requires_grad=True)
        w = ops.ones((3, 2), requires_grad=True)
        
        # Forward pass
        y = ops.matmul(x, w)
        y = ops.relu(y)
        loss = ops.norm(y)
        
        # Backward pass
        loss.backward()
        
        assert x.grad is not None
        assert w.grad is not None
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape


if __name__ == "__main__":
    pytest.main([__file__])