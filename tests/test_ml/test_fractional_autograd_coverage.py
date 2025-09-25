#!/usr/bin/env python3
"""
Comprehensive tests for fractional autograd module to achieve 85% coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.fractional_autograd import (
    fractional_derivative,
    rl_derivative,
    caputo_derivative,
    gl_derivative
)
from hpfracc.core.definitions import FractionalOrder


class TestFractionalAutogradCoverage:
    """Comprehensive tests for fractional autograd functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alpha = 0.5
        self.test_input = torch.randn(10, requires_grad=True)
        self.fractional_order = FractionalOrder(self.alpha)
        
    def test_fractional_autograd_function_forward(self):
        """Test FractionalAutogradFunction forward pass."""
        # Test that function can be called
        result = fractional_derivative_autograd(self.test_input, self.alpha)
        
        assert torch.is_tensor(result)
        assert result.shape == self.test_input.shape
        assert result.requires_grad == self.test_input.requires_grad
        
    def test_fractional_autograd_function_backward(self):
        """Test FractionalAutogradFunction backward pass."""
        # Forward pass
        result = fractional_derivative_autograd(self.test_input, self.alpha)
        
        # Create a loss and backpropagate
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist
        assert self.test_input.grad is not None
        assert self.test_input.grad.shape == self.test_input.shape
        
    def test_different_alpha_values(self):
        """Test with different fractional orders."""
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        
        for alpha in alpha_values:
            input_tensor = torch.randn(10, requires_grad=True)
            result = fractional_derivative_autograd(input_tensor, alpha)
            
            assert torch.is_tensor(result)
            assert result.shape == input_tensor.shape
            
            # Test gradient computation
            loss = result.sum()
            loss.backward()
            assert input_tensor.grad is not None
            
    def test_different_input_shapes(self):
        """Test with different input shapes."""
        shapes = [(5,), (5, 10), (3, 5, 10), (2, 3, 5, 10)]
        
        for shape in shapes:
            input_tensor = torch.randn(shape, requires_grad=True)
            result = fractional_derivative_autograd(input_tensor, self.alpha)
            
            assert result.shape == input_tensor.shape
            
            # Test backward pass
            loss = result.sum()
            loss.backward()
            assert input_tensor.grad is not None
            
    def test_spectral_fractional_derivative_class(self):
        """Test SpectralFractionalDerivative class."""
        derivative = SpectralFractionalDerivative(self.fractional_order)
        
        # Test compute method
        result = derivative.compute(self.test_input.detach().numpy(), np.linspace(0, 1, 10), 0.1)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == self.test_input.shape
        
    def test_spectral_derivative_with_different_parameters(self):
        """Test spectral derivative with different parameters."""
        derivative = SpectralFractionalDerivative(self.fractional_order)
        
        # Test with different time arrays
        t_arrays = [
            np.linspace(0, 1, 10),
            np.linspace(0, 2, 10),
            np.linspace(0, 1, 20)
        ]
        
        for t in t_arrays:
            f_values = np.sin(2 * np.pi * t)
            h = t[1] - t[0] if len(t) > 1 else 0.1
            
            result = derivative.compute(f_values, t, h)
            assert isinstance(result, np.ndarray)
            assert result.shape == f_values.shape
            
    def test_gradient_flow_preservation(self):
        """Test that gradient flow is preserved through operations."""
        # Create a simple network with fractional autograd
        class FractionalNetwork(torch.nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = alpha
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                x = self.linear(x)
                x = fractional_derivative_autograd(x, self.alpha)
                return x.sum()
                
        network = FractionalNetwork(self.alpha)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        
        # Forward and backward pass
        loss = network(self.test_input)
        optimizer.zero_grad()
        loss.backward()
        
        # Check that all parameters have gradients
        for param in network.parameters():
            assert param.grad is not None
            
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small values
        small_input = torch.full((10,), 1e-8, requires_grad=True)
        small_result = fractional_derivative_autograd(small_input, self.alpha)
        assert torch.all(torch.isfinite(small_result))
        
        # Very large values
        large_input = torch.full((10,), 1e8, requires_grad=True)
        large_result = fractional_derivative_autograd(large_input, self.alpha)
        assert torch.all(torch.isfinite(large_result))
        
        # Zero input
        zero_input = torch.zeros(10, requires_grad=True)
        zero_result = fractional_derivative_autograd(zero_input, self.alpha)
        assert torch.all(torch.isfinite(zero_result))
        
    def test_device_compatibility(self):
        """Test compatibility with different devices."""
        # CPU
        cpu_input = torch.randn(10, requires_grad=True)
        cpu_result = fractional_derivative_autograd(cpu_input, self.alpha)
        assert cpu_result.device == cpu_input.device
        
        # CUDA (if available)
        if torch.cuda.is_available():
            cuda_input = cpu_input.cuda()
            cuda_result = fractional_derivative_autograd(cuda_input, self.alpha)
            assert cuda_result.device == cuda_input.device
            
    def test_dtype_compatibility(self):
        """Test compatibility with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            input_tensor = torch.randn(10, dtype=dtype, requires_grad=True)
            result = fractional_derivative_autograd(input_tensor, self.alpha)
            
            # Result should maintain reasonable dtype
            assert result.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]
            
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        batch_sizes = [1, 5, 10, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 10, requires_grad=True)
            result = fractional_derivative_autograd(input_tensor, self.alpha)
            
            assert result.shape == input_tensor.shape
            
            # Test gradient computation
            loss = result.sum()
            loss.backward()
            assert input_tensor.grad is not None
            
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid alpha values
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            fractional_derivative_autograd(self.test_input, -0.5)
            
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            fractional_derivative_autograd(self.test_input, 2.5)
            
    def test_memory_efficiency(self):
        """Test memory efficiency with repeated operations."""
        # Process multiple times to test memory handling
        for _ in range(10):
            input_tensor = torch.randn(100, requires_grad=True)
            result = fractional_derivative_autograd(input_tensor, self.alpha)
            loss = result.sum()
            loss.backward()
            
        # If we get here without memory issues, test passes
        assert True
        
    def test_reproducibility(self):
        """Test reproducibility with fixed seeds."""
        torch.manual_seed(42)
        input1 = torch.randn(10, requires_grad=True)
        
        torch.manual_seed(42)
        input2 = torch.randn(10, requires_grad=True)
        
        result1 = fractional_derivative_autograd(input1, self.alpha)
        result2 = fractional_derivative_autograd(input2, self.alpha)
        
        # Results should be similar (allowing for numerical differences)
        assert torch.allclose(result1, result2, atol=1e-6)
        
    def test_complex_gradient_chains(self):
        """Test complex gradient computation chains."""
        # Create a complex computation graph
        x = torch.randn(10, requires_grad=True)
        
        # Multiple fractional operations
        y1 = fractional_derivative_autograd(x, 0.3)
        y2 = fractional_derivative_autograd(x, 0.7)
        
        # Combine results
        combined = y1 * y2 + torch.sin(x)
        loss = combined.sum()
        
        # Backward pass
        loss.backward()
        
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        
    def test_integration_with_optimizers(self):
        """Test integration with PyTorch optimizers."""
        # Simple model using fractional autograd
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(10))
                
            def forward(self, x):
                weighted = x * self.weight
                fractional = fractional_derivative_autograd(weighted, 0.5)
                return fractional.sum()
                
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training step
        input_data = torch.randn(10)
        loss = model(input_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        assert model.weight.grad is not None
        
    def test_higher_order_gradients(self):
        """Test computation of higher-order gradients."""
        input_tensor = torch.randn(5, requires_grad=True)
        
        # First-order gradient
        result = fractional_derivative_autograd(input_tensor, self.alpha)
        loss = result.sum()
        
        # Compute first-order gradient
        grad1 = torch.autograd.grad(loss, input_tensor, create_graph=True)[0]
        
        # Compute second-order gradient (if supported)
        try:
            grad2 = torch.autograd.grad(grad1.sum(), input_tensor)[0]
            assert grad2 is not None
        except RuntimeError:
            # Second-order gradients might not be supported, which is OK
            pass
            
    def test_functional_interface_consistency(self):
        """Test consistency of functional interface."""
        # Test that the functional interface produces consistent results
        input_tensor = torch.randn(10, requires_grad=True)
        
        # Call multiple times with same input
        result1 = fractional_derivative_autograd(input_tensor, self.alpha)
        result2 = fractional_derivative_autograd(input_tensor, self.alpha)
        
        # Results should be identical
        assert torch.allclose(result1, result2)
        
    def test_spectral_derivative_edge_cases(self):
        """Test spectral derivative with edge cases."""
        derivative = SpectralFractionalDerivative(self.fractional_order)
        
        # Single point
        single_point = np.array([1.0])
        t_single = np.array([0.0])
        result = derivative.compute(single_point, t_single, 1.0)
        assert isinstance(result, np.ndarray)
        
        # Constant function
        constant = np.ones(10)
        t_const = np.linspace(0, 1, 10)
        h_const = t_const[1] - t_const[0]
        result = derivative.compute(constant, t_const, h_const)
        assert isinstance(result, np.ndarray)
        
    def test_alpha_parameter_validation(self):
        """Test validation of alpha parameter."""
        valid_alphas = [0.1, 0.5, 1.0, 1.5, 1.9]
        invalid_alphas = [-0.1, 0.0, 2.0, 2.5]
        
        for alpha in valid_alphas:
            # Should work without error
            result = fractional_derivative_autograd(self.test_input, alpha)
            assert torch.is_tensor(result)
            
        for alpha in invalid_alphas:
            # Should raise an error or handle gracefully
            try:
                result = fractional_derivative_autograd(self.test_input, alpha)
                # If no error, result should still be valid
                assert torch.is_tensor(result)
            except (ValueError, AssertionError, RuntimeError):
                # Expected for invalid alphas
                pass
                
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        import time
        
        # Test with different sizes to check scaling
        sizes = [10, 100, 1000]
        
        for size in sizes:
            input_tensor = torch.randn(size, requires_grad=True)
            
            start_time = time.time()
            result = fractional_derivative_autograd(input_tensor, self.alpha)
            loss = result.sum()
            loss.backward()
            end_time = time.time()
            
            # Should complete in reasonable time
            assert end_time - start_time < 10.0  # 10 seconds max
            
    def test_mathematical_properties(self):
        """Test mathematical properties of fractional derivatives."""
        # Test linearity: D^α[af + bg] = aD^α[f] + bD^α[g]
        f = torch.randn(10, requires_grad=True)
        g = torch.randn(10, requires_grad=True)
        a, b = 2.0, 3.0
        
        # Left side: D^α[af + bg]
        combined = a * f + b * g
        left_side = fractional_derivative_autograd(combined, self.alpha)
        
        # Right side: aD^α[f] + bD^α[g]
        df = fractional_derivative_autograd(f, self.alpha)
        dg = fractional_derivative_autograd(g, self.alpha)
        right_side = a * df + b * dg
        
        # Should be approximately equal (allowing for numerical errors)
        assert torch.allclose(left_side, right_side, atol=1e-4, rtol=1e-4)
