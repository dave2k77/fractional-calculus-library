#!/usr/bin/env python3
"""
Simple coverage tests for fractional autograd module.
"""

import pytest
import torch
import numpy as np

from hpfracc.ml.fractional_autograd import (
    fractional_derivative,
    rl_derivative,
    caputo_derivative,
    gl_derivative
)


class TestFractionalAutogradSimple:
    """Simple tests for fractional autograd functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_input = torch.randn(10)
        self.alpha = 0.5
        
    def test_fractional_derivative_basic(self):
        """Test basic fractional derivative computation."""
        result = fractional_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_rl_derivative(self):
        """Test Riemann-Liouville derivative."""
        result = rl_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_caputo_derivative(self):
        """Test Caputo derivative."""
        result = caputo_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_gl_derivative(self):
        """Test Grünwald-Letnikov derivative."""
        result = gl_derivative(self.test_input, self.alpha)
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        
    def test_different_alpha_values(self):
        """Test with different fractional orders."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        methods = [rl_derivative, caputo_derivative, gl_derivative]
        
        for alpha in alphas:
            for method in methods:
                result = method(self.test_input, alpha)
                assert torch.is_tensor(result)
                assert result.shape == self.test_input.shape
                
    def test_gradient_computation(self):
        """Test gradient computation."""
        input_tensor = torch.randn(10, requires_grad=True)
        
        # Test with different methods
        methods = [rl_derivative, caputo_derivative, gl_derivative]
        
        for method in methods:
            input_tensor.grad = None  # Reset gradient
            result = method(input_tensor, self.alpha)
            loss = result.sum()
            
            loss.backward()
            assert input_tensor.grad is not None
            
    def test_different_input_shapes(self):
        """Test with different input shapes."""
        shapes = [(5,), (5, 10), (3, 5, 10)]
        
        for shape in shapes:
            test_input = torch.randn(shape)
            result = fractional_derivative(test_input, self.alpha)
            assert result.shape == test_input.shape
            
    def test_fractional_derivative_methods(self):
        """Test fractional derivative with different methods."""
        methods = ["RL", "Caputo", "GL"]
        
        for method in methods:
            result = fractional_derivative(self.test_input, self.alpha, method=method)
            assert torch.is_tensor(result)
            assert result.shape == self.test_input.shape
            
    def test_dtype_compatibility(self):
        """Test with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            input_tensor = torch.randn(10, dtype=dtype)
            result = rl_derivative(input_tensor, self.alpha)
            assert torch.is_tensor(result)
            
    def test_device_compatibility(self):
        """Test device compatibility."""
        cpu_input = torch.randn(10)
        cpu_result = rl_derivative(cpu_input, self.alpha)
        assert cpu_result.device == cpu_input.device
        
        if torch.cuda.is_available():
            cuda_input = cpu_input.cuda()
            cuda_result = rl_derivative(cuda_input, self.alpha)
            assert cuda_result.device == cuda_input.device
            
    def test_batch_processing(self):
        """Test batch processing."""
        batch_input = torch.randn(32, 10)
        result = rl_derivative(batch_input, self.alpha)
        assert result.shape == batch_input.shape
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero input
        zero_input = torch.zeros(10)
        zero_result = rl_derivative(zero_input, self.alpha)
        assert torch.all(torch.isfinite(zero_result))
        
        # Single element
        single_input = torch.tensor([1.0])
        single_result = rl_derivative(single_input, self.alpha)
        assert torch.is_tensor(single_result)
        
    def test_numerical_stability(self):
        """Test numerical stability."""
        # Very small values
        small_input = torch.full((10,), 1e-8)
        small_result = rl_derivative(small_input, self.alpha)
        assert torch.all(torch.isfinite(small_result))
        
        # Large values
        large_input = torch.full((10,), 1e3)
        large_result = rl_derivative(large_input, self.alpha)
        assert torch.all(torch.isfinite(large_result))
        
    def test_integration_with_networks(self):
        """Test integration with neural networks."""
        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                x = self.linear(x)
                x = rl_derivative(x, 0.5)
                return x.sum()
                
        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        
        input_data = torch.randn(10)
        loss = net(input_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters have gradients
        for param in net.parameters():
            assert param.grad is not None
            
    def test_reproducibility(self):
        """Test reproducibility with fixed seeds."""
        torch.manual_seed(42)
        input1 = torch.randn(10)
        
        torch.manual_seed(42)
        input2 = torch.randn(10)
        
        result1 = rl_derivative(input1, self.alpha)
        result2 = rl_derivative(input2, self.alpha)
        
        assert torch.allclose(result1, result2, atol=1e-6)
        
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        # Process multiple batches
        for _ in range(10):
            large_input = torch.randn(100, 50)
            result = rl_derivative(large_input, self.alpha)
            assert result.shape == large_input.shape
            
    def test_mathematical_properties(self):
        """Test mathematical properties."""
        # Test linearity: D^α[af + bg] ≈ aD^α[f] + bD^α[g]
        f = torch.randn(10)
        g = torch.randn(10)
        a, b = 2.0, 3.0
        
        # Left side
        combined = a * f + b * g
        left_side = rl_derivative(combined, self.alpha)
        
        # Right side
        df = rl_derivative(f, self.alpha)
        dg = rl_derivative(g, self.alpha)
        right_side = a * df + b * dg
        
        # Should be approximately equal (allowing for numerical errors)
        assert torch.allclose(left_side, right_side, atol=1e-3, rtol=1e-3)

















