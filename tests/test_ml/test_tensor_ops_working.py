#!/usr/bin/env python3
"""Working tests for tensor_ops.py using actual API."""

import pytest
import torch
import numpy as np
from unittest.mock import patch

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor
from hpfracc.ml.backends import BackendType


class TestTensorOpsWorking:
    """Working tests for TensorOps using actual API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ops = TensorOps(backend=BackendType.TORCH)
        
    def test_create_tensor_basic(self):
        """Test basic tensor creation."""
        result = self.ops.create_tensor([1, 2, 3])
        assert torch.is_tensor(result)
        assert result.shape == (3,)
        
    def test_tensor_method(self):
        """Test tensor method."""
        result = self.ops.tensor([1, 2, 3])
        assert torch.is_tensor(result)
        
    def test_zeros_creation(self):
        """Test zeros creation."""
        result = self.ops.zeros((2, 3))
        assert torch.is_tensor(result)
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.zeros(2, 3))
        
    def test_ones_creation(self):
        """Test ones creation."""
        result = self.ops.ones((2, 3))
        assert torch.is_tensor(result)
        assert result.shape == (2, 3)
        assert torch.allclose(result, torch.ones(2, 3))
        
    def test_eye_creation(self):
        """Test identity matrix creation."""
        result = self.ops.eye(3)
        assert torch.is_tensor(result)
        assert result.shape == (3, 3)
        assert torch.allclose(result, torch.eye(3))
        
    def test_arange_creation(self):
        """Test arange creation."""
        result = self.ops.arange(0, 5, 1)
        assert torch.is_tensor(result)
        assert torch.allclose(result, torch.arange(0, 5, 1, dtype=torch.float32))
        
    def test_linspace_creation(self):
        """Test linspace creation."""
        result = self.ops.linspace(0, 1, 5)
        assert torch.is_tensor(result)
        assert torch.allclose(result, torch.linspace(0, 1, 5))
        
    def test_zeros_like(self):
        """Test zeros_like creation."""
        original = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.zeros_like(original)
        assert result.shape == original.shape
        assert torch.allclose(result, torch.zeros_like(original))
        
    def test_sqrt_operation(self):
        """Test sqrt operation."""
        tensor = self.ops.create_tensor([1, 4, 9])
        result = self.ops.sqrt(tensor)
        assert torch.allclose(result, torch.tensor([1, 2, 3], dtype=torch.float32))
        
    def test_stack_operation(self):
        """Test stack operation."""
        a = self.ops.create_tensor([1, 2])
        b = self.ops.create_tensor([3, 4])
        result = self.ops.stack([a, b], dim=0)
        assert result.shape == (2, 2)
        
    def test_cat_operation(self):
        """Test concatenation operation."""
        a = self.ops.create_tensor([[1, 2]])
        b = self.ops.create_tensor([[3, 4]])
        result = self.ops.cat([a, b], dim=0)
        assert result.shape == (2, 2)
        
    def test_reshape_operation(self):
        """Test reshape operation."""
        tensor = self.ops.create_tensor([1, 2, 3, 4])
        result = self.ops.reshape(tensor, (2, 2))
        assert result.shape == (2, 2)
        
    def test_transpose_operation(self):
        """Test transpose operation."""
        tensor = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        result = self.ops.transpose(tensor, (1, 0))
        assert result.shape == (3, 2)
        
    def test_matmul_operation(self):
        """Test matrix multiplication."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[5, 6], [7, 8]])
        result = self.ops.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
        assert torch.allclose(result, expected)
        
    def test_sum_operation(self):
        """Test sum operation."""
        tensor = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.sum(tensor)
        assert torch.allclose(result, torch.tensor(10.0))
        
    def test_sum_with_dim(self):
        """Test sum with dimension."""
        tensor = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.sum(tensor, dim=0)
        assert torch.allclose(result, torch.tensor([4, 6], dtype=torch.float32))
        
    def test_mean_operation(self):
        """Test mean operation."""
        tensor = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.mean(tensor)
        assert torch.allclose(result, torch.tensor(2.5))
        
    def test_std_operation(self):
        """Test standard deviation."""
        tensor = self.ops.create_tensor([1, 2, 3, 4, 5])
        result = self.ops.std(tensor)
        assert torch.is_tensor(result)
        
    def test_max_operation(self):
        """Test max operation."""
        tensor = self.ops.create_tensor([1, 5, 3, 2])
        result = self.ops.max(tensor)
        assert torch.allclose(result, torch.tensor(5.0))
        
    def test_min_operation(self):
        """Test min operation."""
        tensor = self.ops.create_tensor([1, 5, 3, 2])
        result = self.ops.min(tensor)
        assert torch.allclose(result, torch.tensor(1.0))
        
    def test_randn_like(self):
        """Test randn_like operation."""
        original = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.randn_like(original)
        assert result.shape == original.shape
        
    def test_norm_operation(self):
        """Test norm operation."""
        tensor = self.ops.create_tensor([3, 4])
        result = self.ops.norm(tensor)
        assert torch.allclose(result, torch.tensor(5.0))  # 3-4-5 triangle
        
    def test_softmax_operation(self):
        """Test softmax operation."""
        tensor = self.ops.create_tensor([1, 2, 3])
        result = self.ops.softmax(tensor)
        assert torch.allclose(result.sum(), torch.tensor(1.0))
        
    def test_activation_functions(self):
        """Test activation functions."""
        tensor = self.ops.create_tensor([-1, 0, 1])
        
        # ReLU
        relu_result = self.ops.relu(tensor)
        assert torch.allclose(relu_result, torch.tensor([0, 0, 1], dtype=torch.float32))
        
        # Sigmoid
        sigmoid_result = self.ops.sigmoid(tensor)
        assert torch.all((sigmoid_result >= 0) & (sigmoid_result <= 1))
        
        # Tanh
        tanh_result = self.ops.tanh(tensor)
        assert torch.all((tanh_result >= -1) & (tanh_result <= 1))
        
    def test_log_operation(self):
        """Test logarithm operation."""
        tensor = self.ops.create_tensor([1, np.e, np.e**2])
        result = self.ops.log(tensor)
        expected = torch.tensor([0, 1, 2], dtype=torch.float32)
        assert torch.allclose(result, expected, atol=1e-6)
        
    def test_dropout_operation(self):
        """Test dropout operation."""
        tensor = self.ops.create_tensor([1, 2, 3, 4, 5])
        result = self.ops.dropout(tensor, p=0.5, training=True)
        assert result.shape == tensor.shape
        
    def test_clip_operation(self):
        """Test clip operation."""
        tensor = self.ops.create_tensor([-2, -1, 0, 1, 2])
        result = self.ops.clip(tensor, -1, 1)
        assert torch.all((result >= -1) & (result <= 1))
        
    def test_unsqueeze_operation(self):
        """Test unsqueeze operation."""
        tensor = self.ops.create_tensor([1, 2, 3])
        result = self.ops.unsqueeze(tensor, 0)
        assert result.shape == (1, 3)
        
    def test_squeeze_operation(self):
        """Test squeeze operation."""
        tensor = self.ops.create_tensor([[1, 2, 3]])
        result = self.ops.squeeze(tensor, 0)
        assert result.shape == (3,)
        
    def test_expand_operation(self):
        """Test expand operation."""
        tensor = self.ops.create_tensor([[1], [2]])
        result = self.ops.expand(tensor, 2, 3)
        assert result.shape == (2, 3)
        
    def test_gather_operation(self):
        """Test gather operation."""
        tensor = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        index = self.ops.create_tensor([[0, 1], [2, 0]], dtype=torch.long)
        result = self.ops.gather(tensor, 1, index)
        assert result.shape == index.shape
        
    def test_repeat_operation(self):
        """Test repeat operation."""
        tensor = self.ops.create_tensor([1, 2])
        result = self.ops.repeat(tensor, 2, 3)
        assert result.shape == (4, 6)  # Original (2,) repeated (2, 3) times
        
    def test_einsum_operation(self):
        """Test einsum operation."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[5, 6], [7, 8]])
        result = self.ops.einsum('ij,jk->ik', a, b)
        assert result.shape == (2, 2)
        
    def test_median_operation(self):
        """Test median operation."""
        tensor = self.ops.create_tensor([1, 3, 2, 5, 4])
        result = self.ops.median(tensor)
        assert torch.allclose(result, torch.tensor(3.0))
        
    def test_quantile_operation(self):
        """Test quantile operation."""
        tensor = self.ops.create_tensor([1, 2, 3, 4, 5])
        result = self.ops.quantile(tensor, 0.5)  # Median
        assert torch.is_tensor(result)
        
    def test_no_grad_context(self):
        """Test no_grad context manager."""
        tensor = self.ops.create_tensor([1, 2, 3], requires_grad=True)
        
        with self.ops.no_grad():
            result = tensor * 2
            # Operations in no_grad should not track gradients
            assert not result.requires_grad
            
    def test_backend_consistency(self):
        """Test backend consistency."""
        # Test that operations work consistently
        tensor = self.ops.create_tensor([1, 2, 3])
        
        # Chain operations
        result = self.ops.sum(self.ops.sqrt(self.ops.create_tensor([1, 4, 9])))
        assert torch.is_tensor(result)
        
    def test_get_tensor_ops_function(self):
        """Test get_tensor_ops utility function."""
        ops = get_tensor_ops(BackendType.TORCH)
        assert isinstance(ops, TensorOps)
        assert ops.backend == BackendType.TORCH
        
    def test_create_tensor_function(self):
        """Test create_tensor utility function."""
        result = create_tensor([1, 2, 3])
        assert torch.is_tensor(result)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with unknown backend
        ops = TensorOps(backend=BackendType.TORCH)
        with patch.object(ops, 'backend', 'unknown'):
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])
                
    def test_different_backends(self):
        """Test different backend initialization."""
        backends = [BackendType.TORCH, BackendType.AUTO]
        
        for backend in backends:
            ops = TensorOps(backend=backend)
            tensor = ops.create_tensor([1, 2, 3])
            assert torch.is_tensor(tensor)  # Should work for available backends













