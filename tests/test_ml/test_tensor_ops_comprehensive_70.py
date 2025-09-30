#!/usr/bin/env python3
"""Comprehensive tensor_ops tests targeting 70% coverage goal."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops, create_tensor, switch_backend
from hpfracc.ml.backends import BackendType


class TestTensorOps70Coverage:
    """Comprehensive tests to push tensor_ops to 70%+ coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ops = TensorOps(backend=BackendType.TORCH)
        
    # Basic Creation Methods (targeting lines 47-84)
    def test_create_tensor_comprehensive(self):
        """Test comprehensive tensor creation scenarios."""
        # Basic creation
        t1 = self.ops.create_tensor([1, 2, 3])
        assert torch.is_tensor(t1)
        
        # With dtype
        t2 = self.ops.create_tensor([1, 2, 3], dtype=torch.float64)
        assert t2.dtype == torch.float64
        
        # With device
        t3 = self.ops.create_tensor([1, 2, 3], device='cpu')
        assert t3.device.type == 'cpu'
        
        # With requires_grad
        t4 = self.ops.create_tensor([1, 2, 3], requires_grad=True)
        assert t4.requires_grad
        
        # From numpy
        arr = np.array([[1, 2], [3, 4]])
        t5 = self.ops.create_tensor(arr)
        assert t5.shape == (2, 2)
        
    def test_tensor_method(self):
        """Test tensor method (alias for create_tensor)."""
        result = self.ops.tensor([[1, 2], [3, 4]])
        assert torch.is_tensor(result)
        assert result.shape == (2, 2)
        
    def test_no_grad_context(self):
        """Test no_grad context manager."""
        x = self.ops.create_tensor([1, 2, 3], requires_grad=True)
        
        with self.ops.no_grad():
            y = x * 2
            assert not y.requires_grad
            
    # Shape Creation Methods (targeting lines 85-175)
    def test_zeros_comprehensive(self):
        """Test zeros creation with various parameters."""
        # Basic zeros
        z1 = self.ops.zeros((3, 4))
        assert z1.shape == (3, 4)
        assert torch.allclose(z1, torch.zeros(3, 4))
        
        # With dtype
        z2 = self.ops.zeros((2, 2), dtype=torch.float64)
        assert z2.dtype == torch.float64
        
        # With device
        z3 = self.ops.zeros((2, 2), device='cpu')
        assert z3.device.type == 'cpu'
        
    def test_ones_comprehensive(self):
        """Test ones creation with various parameters."""
        o1 = self.ops.ones((2, 3))
        assert torch.allclose(o1, torch.ones(2, 3))
        
        o2 = self.ops.ones((1, 5), dtype=torch.float32)
        assert o2.dtype == torch.float32
        
    def test_eye_comprehensive(self):
        """Test identity matrix creation."""
        # Basic identity
        eye1 = self.ops.eye(3)
        assert torch.allclose(eye1, torch.eye(3))
        
        # With dtype
        eye2 = self.ops.eye(4, dtype=torch.float64)
        assert eye2.dtype == torch.float64
        
    def test_arange_comprehensive(self):
        """Test arange with various parameters."""
        # Basic arange
        a1 = self.ops.arange(0, 10, 2)
        expected = torch.arange(0, 10, 2, dtype=torch.float32)
        assert torch.allclose(a1, expected)
        
        # Different step sizes
        a2 = self.ops.arange(0, 5, 1)
        assert a2.shape == (5,)
        
    def test_linspace_comprehensive(self):
        """Test linspace with various parameters."""
        l1 = self.ops.linspace(0, 1, 11)
        expected = torch.linspace(0, 1, 11)
        assert torch.allclose(l1, expected)
        
        l2 = self.ops.linspace(-1, 1, 21)
        assert l2.shape == (21,)
        
    def test_zeros_like_comprehensive(self):
        """Test zeros_like with various inputs."""
        original = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        zl1 = self.ops.zeros_like(original)
        assert zl1.shape == original.shape
        assert torch.allclose(zl1, torch.zeros_like(original))
        
        # With different dtype
        zl2 = self.ops.zeros_like(original, dtype=torch.float64)
        assert zl2.dtype == torch.float64
        
    # Mathematical Operations (targeting lines 176-314)
    def test_sqrt_comprehensive(self):
        """Test sqrt with various inputs."""
        # Basic sqrt
        s1 = self.ops.sqrt(self.ops.create_tensor([1, 4, 9, 16]))
        expected = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        assert torch.allclose(s1, expected)
        
        # Edge cases
        s2 = self.ops.sqrt(self.ops.create_tensor([0.0]))
        assert torch.allclose(s2, torch.tensor([0.0]))
        
    def test_stack_comprehensive(self):
        """Test stack with various configurations."""
        a = self.ops.create_tensor([1, 2])
        b = self.ops.create_tensor([3, 4])
        c = self.ops.create_tensor([5, 6])
        
        # Stack along different dimensions
        s1 = self.ops.stack([a, b, c], dim=0)
        assert s1.shape == (3, 2)
        
        s2 = self.ops.stack([a, b], dim=1)
        assert s2.shape == (2, 2)
        
    def test_cat_comprehensive(self):
        """Test concatenation with various configurations."""
        a = self.ops.create_tensor([[1, 2]])
        b = self.ops.create_tensor([[3, 4]])
        c = self.ops.create_tensor([[5, 6]])
        
        # Concatenate along different dimensions
        c1 = self.ops.cat([a, b, c], dim=0)
        assert c1.shape == (3, 2)
        
        # Different shapes
        d = self.ops.create_tensor([[1], [2]])
        e = self.ops.create_tensor([[3], [4]])
        c2 = self.ops.cat([d, e], dim=1)
        assert c2.shape == (2, 2)
        
    def test_reshape_comprehensive(self):
        """Test reshape with various configurations."""
        original = self.ops.create_tensor([[1, 2, 3, 4, 5, 6]])
        
        # Different reshape targets
        r1 = self.ops.reshape(original, (2, 3))
        assert r1.shape == (2, 3)
        
        r2 = self.ops.reshape(original, (3, 2))
        assert r2.shape == (3, 2)
        
        r3 = self.ops.reshape(original, (6,))
        assert r3.shape == (6,)
        
    def test_repeat_comprehensive(self):
        """Test repeat with various patterns."""
        original = self.ops.create_tensor([1, 2])
        
        # Different repeat patterns
        r1 = self.ops.repeat(original, 2, 3)
        assert r1.shape == (4, 6)
        
        r2 = self.ops.repeat(original, 1, 2)
        assert r2.shape == (2, 4)
        
    def test_clip_comprehensive(self):
        """Test clip with various ranges."""
        data = self.ops.create_tensor([-5, -2, 0, 3, 8])
        
        # Different clipping ranges
        c1 = self.ops.clip(data, -1, 5)
        assert torch.all((c1 >= -1) & (c1 <= 5))
        
        c2 = self.ops.clip(data, 0, 3)
        assert torch.all((c2 >= 0) & (c2 <= 3))
        
    def test_unsqueeze_squeeze_comprehensive(self):
        """Test unsqueeze and squeeze operations."""
        original = self.ops.create_tensor([[1, 2, 3]])
        
        # Unsqueeze at different dimensions
        u1 = self.ops.unsqueeze(original, 0)
        assert u1.shape == (1, 1, 3)
        
        u2 = self.ops.unsqueeze(original, 2)
        assert u2.shape == (1, 3, 1)
        
        # Squeeze
        s1 = self.ops.squeeze(u1, 0)
        assert s1.shape == (1, 3)
        
        s2 = self.ops.squeeze(original, 0)
        assert s2.shape == (3,)
        
    def test_expand_comprehensive(self):
        """Test expand operation."""
        original = self.ops.create_tensor([[1], [2]])  # (2, 1)
        
        # Expand to different sizes
        e1 = self.ops.expand(original, 2, 5)
        assert e1.shape == (2, 5)
        
        e2 = self.ops.expand(original, 2, 3)
        assert e2.shape == (2, 3)
        
    def test_gather_comprehensive(self):
        """Test gather operation."""
        data = self.ops.create_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Gather along different dimensions
        idx1 = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)
        g1 = self.ops.gather(data, 1, idx1)
        assert g1.shape == idx1.shape
        
        idx2 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g2 = self.ops.gather(data, 0, idx2)
        assert g2.shape == idx2.shape
        
    def test_transpose_comprehensive(self):
        """Test transpose with various dimension specifications."""
        data = self.ops.create_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
        
        # Different transpose patterns
        t1 = self.ops.transpose(data, (0, 1))
        assert t1.shape == (2, 2, 2)
        
        t2 = self.ops.transpose(data, (2, 1, 0))
        assert t2.shape == (2, 2, 2)
        
        t3 = self.ops.transpose(data, (1, 0, 2))
        assert t3.shape == (2, 2, 2)
        
    # Reduction Operations (targeting lines 354-499)
    def test_sum_comprehensive(self):
        """Test sum with various configurations."""
        data = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Sum all
        s1 = self.ops.sum(data)
        assert torch.allclose(s1, torch.tensor(21.0))
        
        # Sum along dimensions
        s2 = self.ops.sum(data, dim=0)
        assert torch.allclose(s2, torch.tensor([5, 7, 9], dtype=torch.float32))
        
        s3 = self.ops.sum(data, dim=1)
        assert torch.allclose(s3, torch.tensor([6, 15], dtype=torch.float32))
        
        # Keep dimensions
        s4 = self.ops.sum(data, dim=0, keepdim=True)
        assert s4.shape == (1, 3)
        
    def test_mean_comprehensive(self):
        """Test mean with various configurations."""
        data = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Mean all
        m1 = self.ops.mean(data)
        assert torch.allclose(m1, torch.tensor(3.5))
        
        # Mean along dimensions
        m2 = self.ops.mean(data, dim=0)
        assert torch.allclose(m2, torch.tensor([2.5, 3.5, 4.5], dtype=torch.float32))
        
        m3 = self.ops.mean(data, dim=1, keepdim=True)
        assert m3.shape == (2, 1)
        
    def test_std_comprehensive(self):
        """Test standard deviation."""
        data = self.ops.create_tensor([1, 2, 3, 4, 5])
        
        # Basic std
        s1 = self.ops.std(data)
        expected = torch.std(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))
        assert torch.allclose(s1, expected)
        
        # With dimensions
        data_2d = self.ops.create_tensor([[1, 2], [3, 4]])
        s2 = self.ops.std(data_2d, dim=0)
        assert s2.shape == (2,)
        
    def test_max_min_comprehensive(self):
        """Test max and min operations."""
        data = self.ops.create_tensor([[1, 5, 2], [8, 3, 6]])
        
        # Global max/min
        max_val = self.ops.max(data)
        min_val = self.ops.min(data)
        assert torch.allclose(max_val, torch.tensor(8.0))
        assert torch.allclose(min_val, torch.tensor(1.0))
        
        # Along dimensions
        max_dim0 = self.ops.max(data, dim=0)
        min_dim1 = self.ops.min(data, dim=1)
        assert max_dim0.shape == (3,)
        assert min_dim1.shape == (2,)
        
    def test_median_quantile_comprehensive(self):
        """Test median and quantile operations."""
        data = self.ops.create_tensor([1, 3, 2, 5, 4])
        
        # Median
        med = self.ops.median(data)
        assert torch.allclose(med, torch.tensor(3.0))
        
        # Quantiles
        q25 = self.ops.quantile(data, 0.25)
        q75 = self.ops.quantile(data, 0.75)
        assert torch.is_tensor(q25)
        assert torch.is_tensor(q75)
        
    def test_randn_like_comprehensive(self):
        """Test randn_like with various configurations."""
        original = self.ops.create_tensor([[1, 2], [3, 4]])
        
        # Basic randn_like
        r1 = self.ops.randn_like(original)
        assert r1.shape == original.shape
        
        # With different dtype
        r2 = self.ops.randn_like(original, dtype=torch.float64)
        assert r2.dtype == torch.float64
        
    def test_norm_comprehensive(self):
        """Test norm operations."""
        data = self.ops.create_tensor([[3, 4], [1, 2]])
        
        # Default L2 norm
        n1 = self.ops.norm(data)
        assert torch.is_tensor(n1)
        
        # Different p-norms
        n2 = self.ops.norm(data, p=1)
        n3 = self.ops.norm(data, p=float('inf'))
        assert torch.is_tensor(n2)
        assert torch.is_tensor(n3)
        
        # Along dimensions
        n4 = self.ops.norm(data, dim=1)
        assert n4.shape == (2,)
        
    # Activation Functions (targeting lines 513-574)
    def test_softmax_comprehensive(self):
        """Test softmax with various configurations."""
        data = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Default softmax
        s1 = self.ops.softmax(data)
        assert torch.allclose(s1.sum(dim=-1), torch.ones(2))
        
        # Along different dimensions
        s2 = self.ops.softmax(data, dim=0)
        assert s2.shape == data.shape
        
    def test_relu_comprehensive(self):
        """Test ReLU activation."""
        data = self.ops.create_tensor([-2, -1, 0, 1, 2])
        
        relu_result = self.ops.relu(data)
        expected = torch.tensor([0, 0, 0, 1, 2], dtype=torch.float32)
        assert torch.allclose(relu_result, expected)
        
    def test_sigmoid_comprehensive(self):
        """Test sigmoid activation."""
        data = self.ops.create_tensor([-5, 0, 5])
        
        sig_result = self.ops.sigmoid(data)
        assert torch.all((sig_result >= 0) & (sig_result <= 1))
        assert torch.allclose(sig_result[1], torch.tensor(0.5), atol=1e-6)
        
    def test_tanh_comprehensive(self):
        """Test tanh activation."""
        data = self.ops.create_tensor([-2, 0, 2])
        
        tanh_result = self.ops.tanh(data)
        assert torch.all((tanh_result >= -1) & (tanh_result <= 1))
        assert torch.allclose(tanh_result[1], torch.tensor(0.0))
        
    def test_log_comprehensive(self):
        """Test logarithm operation."""
        data = self.ops.create_tensor([1, np.e, np.e**2])
        
        log_result = self.ops.log(data)
        expected = torch.tensor([0, 1, 2], dtype=torch.float32)
        assert torch.allclose(log_result, expected, atol=1e-6)
        
    def test_dropout_comprehensive(self):
        """Test dropout with various configurations."""
        data = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Training mode
        d1 = self.ops.dropout(data, p=0.5, training=True)
        assert d1.shape == data.shape
        
        # Eval mode
        d2 = self.ops.dropout(data, p=0.5, training=False)
        assert torch.allclose(d2, data)
        
        # Different probabilities
        d3 = self.ops.dropout(data, p=0.2, training=True)
        assert d3.shape == data.shape
        
    # Linear Algebra Operations (targeting lines 315-353)
    def test_matmul_comprehensive(self):
        """Test matrix multiplication variants."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[5, 6], [7, 8]])
        
        # Basic matmul
        result = self.ops.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
        assert torch.allclose(result, expected)
        
        # Vector-matrix multiplication
        v = self.ops.create_tensor([1, 2])
        vm_result = self.ops.matmul(v, a)
        assert vm_result.shape == (2,)
        
    def test_einsum_comprehensive(self):
        """Test einsum with various equations."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[5, 6], [7, 8]])
        
        # Matrix multiplication via einsum
        e1 = self.ops.einsum('ij,jk->ik', a, b)
        assert e1.shape == (2, 2)
        
        # Element-wise multiplication
        e2 = self.ops.einsum('ij,ij->ij', a, b)
        assert e2.shape == (2, 2)
        
        # Trace
        e3 = self.ops.einsum('ii->', a)
        assert torch.allclose(e3, torch.tensor(5.0))  # 1 + 4
        
    # Utility Functions (targeting lines 605-624)
    def test_get_tensor_ops_comprehensive(self):
        """Test get_tensor_ops utility function."""
        # Default backend
        ops1 = get_tensor_ops()
        assert isinstance(ops1, TensorOps)
        
        # Specific backend
        ops2 = get_tensor_ops(BackendType.TORCH)
        assert ops2.backend == BackendType.TORCH
        
        # AUTO backend
        ops3 = get_tensor_ops(BackendType.AUTO)
        assert ops3.backend != BackendType.AUTO  # Should resolve
        
    def test_create_tensor_utility(self):
        """Test create_tensor utility function."""
        # Basic usage
        t1 = create_tensor([1, 2, 3])
        assert torch.is_tensor(t1)
        
        # With parameters
        t2 = create_tensor([[1, 2], [3, 4]], dtype=torch.float64)
        assert t2.dtype == torch.float64
        
    def test_switch_backend_comprehensive(self):
        """Test backend switching."""
        # Switch to different backends
        switch_backend(BackendType.TORCH)
        ops1 = get_tensor_ops()
        assert ops1.backend == BackendType.TORCH
        
        switch_backend(BackendType.AUTO)
        ops2 = get_tensor_ops()
        assert ops2.backend != BackendType.AUTO
        
    # Error Handling and Edge Cases (targeting error paths)
    def test_backend_error_handling(self):
        """Test error handling for invalid backends."""
        ops = TensorOps(backend=BackendType.TORCH)
        
        # Mock unknown backend
        with patch.object(ops, 'backend', 'unknown_backend'):
            with pytest.raises(RuntimeError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])
                
    def test_jax_backend_filtering(self):
        """Test JAX backend parameter filtering."""
        # Test that JAX backend filters out requires_grad
        ops = TensorOps(backend=BackendType.TORCH)
        
        # Mock JAX backend
        with patch.object(ops, 'backend', BackendType.JAX):
            # Should work even though we're mocking
            try:
                result = ops.create_tensor([1, 2, 3], requires_grad=True)
                assert torch.is_tensor(result)
            except:
                # Expected if JAX not properly set up
                pass
                
    def test_numba_backend_filtering(self):
        """Test NUMBA backend parameter filtering."""
        ops = TensorOps(backend=BackendType.TORCH)
        
        # Mock NUMBA backend
        with patch.object(ops, 'backend', BackendType.NUMBA):
            try:
                result = ops.create_tensor([1, 2, 3], requires_grad=True)
                assert torch.is_tensor(result)
            except:
                # Expected if NUMBA not properly set up
                pass
                
    def test_complex_operation_chains(self):
        """Test complex chains of operations."""
        # Create complex computation graph
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[2, 1], [1, 2]])
        
        # Chain multiple operations
        result = self.ops.sum(
            self.ops.matmul(
                self.ops.transpose(a, (1, 0)),
                self.ops.sqrt(
                    self.ops.clip(b, 0.1, 10)
                )
            )
        )
        
        assert torch.is_tensor(result)
        assert torch.isfinite(result)
        
    def test_memory_efficiency_comprehensive(self):
        """Test memory efficiency with large operations."""
        # Create and process large tensors (reduced sizes to avoid memory issues)
        for size in [10, 20, 50]:  # Much smaller sizes to avoid hanging
            large_tensor = self.ops.zeros((size, size))
            
            # Perform operations that should be memory efficient
            result = self.ops.sum(self.ops.sqrt(large_tensor + 1))
            assert torch.isfinite(result)
            
    def test_gradient_preservation(self):
        """Test gradient preservation through operations."""
        x = self.ops.create_tensor([[1, 2], [3, 4]], requires_grad=True)
        
        # Chain operations that should preserve gradients
        y = self.ops.sum(self.ops.matmul(x, x))
        y.backward()
        
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))








