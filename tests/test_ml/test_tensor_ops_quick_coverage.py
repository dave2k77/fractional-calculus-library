#!/usr/bin/env python3
"""
Quick coverage tests for TensorOps module to reach 85% coverage.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from hpfracc.ml.tensor_ops import TensorOps
from hpfracc.ml.backends import BackendType


class TestTensorOpsQuickCoverage:
    """Quick tests to boost TensorOps coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ops = TensorOps(backend=BackendType.TORCH)
        
    def test_create_tensor_variants(self):
        """Test different tensor creation methods."""
        # Basic creation
        t1 = self.ops.create_tensor([1, 2, 3])
        assert torch.is_tensor(t1)
        
        # With dtype
        t2 = self.ops.create_tensor([1, 2, 3], dtype=torch.float32)
        assert t2.dtype == torch.float32
        
        # With device
        t3 = self.ops.create_tensor([1, 2, 3], device='cpu')
        assert t3.device.type == 'cpu'
        
        # From numpy
        arr = np.array([1, 2, 3])
        t4 = self.ops.create_tensor(arr)
        assert torch.is_tensor(t4)
        
    def test_basic_operations(self):
        """Test basic tensor operations."""
        a = self.ops.create_tensor([1, 2, 3])
        b = self.ops.create_tensor([4, 5, 6])
        
        # Arithmetic operations
        assert torch.allclose(self.ops.add(a, b), torch.tensor([5, 7, 9], dtype=torch.float32))
        assert torch.allclose(self.ops.subtract(a, b), torch.tensor([-3, -3, -3], dtype=torch.float32))
        assert torch.allclose(self.ops.multiply(a, b), torch.tensor([4, 10, 18], dtype=torch.float32))
        assert torch.allclose(self.ops.divide(b, a), torch.tensor([4, 2.5, 2], dtype=torch.float32))
        
    def test_shape_operations(self):
        """Test shape manipulation operations."""
        a = self.ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Reshape
        reshaped = self.ops.reshape(a, (3, 2))
        assert reshaped.shape == (3, 2)
        
        # Transpose
        transposed = self.ops.transpose(a)
        assert transposed.shape == (3, 2)
        
        # Squeeze/unsqueeze
        squeezed = self.ops.squeeze(self.ops.unsqueeze(a, 0), 0)
        assert torch.allclose(squeezed, a)
        
    def test_reduction_operations(self):
        """Test reduction operations."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        
        assert torch.allclose(self.ops.sum(a), torch.tensor(10.0))
        assert torch.allclose(self.ops.mean(a), torch.tensor(2.5))
        assert torch.allclose(self.ops.max(a), torch.tensor(4.0))
        assert torch.allclose(self.ops.min(a), torch.tensor(1.0))
        
    def test_mathematical_functions(self):
        """Test mathematical functions."""
        a = self.ops.create_tensor([1, 4, 9])
        
        # Power and sqrt
        powered = self.ops.power(a, 2)
        assert torch.allclose(powered, torch.tensor([1, 16, 81], dtype=torch.float32))
        
        sqrt_result = self.ops.sqrt(a)
        assert torch.allclose(sqrt_result, torch.tensor([1, 2, 3], dtype=torch.float32))
        
        # Trigonometric
        angles = self.ops.create_tensor([0, np.pi/2])
        sin_result = self.ops.sin(angles)
        cos_result = self.ops.cos(angles)
        assert torch.allclose(sin_result, torch.tensor([0, 1], dtype=torch.float32), atol=1e-6)
        assert torch.allclose(cos_result, torch.tensor([1, 0], dtype=torch.float32), atol=1e-6)
        
    def test_fft_operations(self):
        """Test FFT operations."""
        a = self.ops.create_tensor([1, 2, 3, 4])
        
        # Forward FFT
        fft_result = self.ops.fft(a)
        assert fft_result.shape == (4,)
        assert torch.is_complex(fft_result)
        
        # Inverse FFT
        ifft_result = self.ops.ifft(fft_result)
        assert torch.allclose(ifft_result.real, a, atol=1e-6)
        
    def test_tensor_utilities(self):
        """Test utility functions."""
        a = self.ops.create_tensor([1, 2, 3])
        
        # Clone and detach
        cloned = self.ops.clone(a)
        assert torch.allclose(cloned, a)
        assert cloned is not a
        
        # Numpy conversion
        numpy_array = self.ops.to_numpy(a)
        assert isinstance(numpy_array, np.ndarray)
        
        from_numpy = self.ops.from_numpy(numpy_array)
        assert torch.is_tensor(from_numpy)
        
    def test_creation_functions(self):
        """Test tensor creation functions."""
        # Zeros and ones
        zeros = self.ops.zeros((2, 3))
        assert zeros.shape == (2, 3)
        assert torch.allclose(zeros, torch.zeros(2, 3))
        
        ones = self.ops.ones((2, 3))
        assert ones.shape == (2, 3)
        assert torch.allclose(ones, torch.ones(2, 3))
        
        # Random
        randn = self.ops.randn((2, 3))
        assert randn.shape == (2, 3)
        
        # Range functions
        arange_result = self.ops.arange(0, 5, 1)
        assert torch.allclose(arange_result, torch.arange(0, 5, 1, dtype=torch.float32))
        
        linspace_result = self.ops.linspace(0, 1, 5)
        assert torch.allclose(linspace_result, torch.linspace(0, 1, 5))
        
    def test_concatenation_and_stacking(self):
        """Test concatenation and stacking operations."""
        a = self.ops.create_tensor([[1, 2]])
        b = self.ops.create_tensor([[3, 4]])
        
        # Concatenate
        concat_result = self.ops.concatenate([a, b], dim=0)
        expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        assert torch.allclose(concat_result, expected)
        
        # Stack
        stack_result = self.ops.stack([a.squeeze(), b.squeeze()], dim=0)
        assert torch.allclose(stack_result, expected)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test with unknown backend
        ops = TensorOps(backend=BackendType.TORCH)
        with patch.object(ops, 'backend', 'unknown'):
            with pytest.raises(ValueError, match="Unknown backend"):
                ops.create_tensor([1, 2, 3])
                
    def test_advanced_operations(self):
        """Test more advanced operations."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        b = self.ops.create_tensor([[5, 6], [7, 8]])
        
        # Matrix multiplication
        matmul_result = self.ops.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
        assert torch.allclose(matmul_result, expected)
        
        # Exponential and log
        exp_result = self.ops.exp(self.ops.create_tensor([0, 1]))
        log_result = self.ops.log(self.ops.create_tensor([1, np.e]))
        assert torch.allclose(exp_result, torch.tensor([1, np.e], dtype=torch.float32), atol=1e-6)
        assert torch.allclose(log_result, torch.tensor([0, 1], dtype=torch.float32), atol=1e-6)
        
        # Absolute value
        abs_result = self.ops.abs(self.ops.create_tensor([-1, -2, 3]))
        assert torch.allclose(abs_result, torch.tensor([1, 2, 3], dtype=torch.float32))
        
    def test_dimension_operations(self):
        """Test operations with specific dimensions."""
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        
        # Sum with dimension
        sum_dim0 = self.ops.sum(a, dim=0)
        assert torch.allclose(sum_dim0, torch.tensor([4, 6], dtype=torch.float32))
        
        # Mean with dimension
        mean_dim1 = self.ops.mean(a, dim=1)
        assert torch.allclose(mean_dim1, torch.tensor([1.5, 3.5], dtype=torch.float32))
        
        # Transpose with specific dimensions
        if hasattr(self.ops, 'transpose') and len(a.shape) >= 2:
            transposed = self.ops.transpose(a, dim0=0, dim1=1)
            assert transposed.shape == (2, 2)
            
    def test_backend_initialization_variants(self):
        """Test different backend initialization scenarios."""
        # Default initialization
        ops1 = TensorOps()
        assert ops1 is not None
        
        # With specific backend
        ops2 = TensorOps(backend=BackendType.TORCH)
        assert ops2.backend == BackendType.TORCH
        
        # With AUTO backend (should resolve)
        ops3 = TensorOps(backend=BackendType.AUTO)
        assert ops3.backend != BackendType.AUTO
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty tensor
        empty = self.ops.zeros((0, 5))
        assert empty.shape == (0, 5)
        
        # Single element
        single = self.ops.create_tensor([42])
        assert torch.allclose(self.ops.sum(single), torch.tensor(42.0))
        
        # Large tensor (test memory handling)
        large = self.ops.randn((100, 100))
        result = self.ops.sum(large)
        assert torch.is_tensor(result)
        
    def test_gradient_operations(self):
        """Test operations that preserve gradients."""
        a = self.ops.create_tensor([1, 2, 3], dtype=torch.float32)
        a.requires_grad = True
        
        # Operations should preserve gradient tracking
        b = self.ops.power(a, 2)
        c = self.ops.sum(b)
        
        c.backward()
        assert a.grad is not None
        
    def test_statistical_operations(self):
        """Test statistical operations."""
        data = self.ops.create_tensor([1, 2, 3, 4, 5])
        
        # Standard deviation
        std_result = self.ops.std(data)
        expected_std = torch.std(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))
        assert torch.allclose(std_result, expected_std)
        
    def test_complex_chains(self):
        """Test complex operation chains."""
        # Chain multiple operations
        a = self.ops.create_tensor([[1, 2], [3, 4]])
        result = self.ops.sum(
            self.ops.multiply(
                self.ops.transpose(a),
                self.ops.ones((2, 2))
            )
        )
        assert torch.is_tensor(result)
