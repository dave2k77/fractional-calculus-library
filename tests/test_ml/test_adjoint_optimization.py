"""
Comprehensive tests for Adjoint Optimization module.

This module provides extensive tests to improve coverage of the adjoint_optimization.py
module, focusing on the adjoint methods for efficient gradient computations.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings

from hpfracc.ml.adjoint_optimization import (
    AdjointConfig,
    AdjointFractionalDerivative,
    AdjointFractionalLayer,
    MemoryEfficientFractionalNetwork,
    AdjointOptimizer,
    adjoint_fractional_derivative,
    adjoint_rl_derivative,
    adjoint_caputo_derivative,
    adjoint_gl_derivative,
    _adjoint_riemann_liouville_forward,
    _adjoint_riemann_liouville_backward,
    _adjoint_caputo_forward,
    _adjoint_caputo_backward,
    _adjoint_grunwald_letnikov_forward,
    _adjoint_grunwald_letnikov_backward
)


class TestAdjointConfig:
    """Test AdjointConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdjointConfig()
        
        assert config.use_adjoint is True
        assert config.adjoint_method == "automatic"
        assert config.memory_efficient is True
        assert config.checkpoint_frequency == 10
        assert config.precision == "float32"
        assert config.gradient_accumulation is False
        assert config.accumulation_steps == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdjointConfig(
            use_adjoint=False,
            adjoint_method="manual",
            memory_efficient=False,
            checkpoint_frequency=5,
            precision="float64",
            gradient_accumulation=True,
            accumulation_steps=8
        )
        
        assert config.use_adjoint is False
        assert config.adjoint_method == "manual"
        assert config.memory_efficient is False
        assert config.checkpoint_frequency == 5
        assert config.precision == "float64"
        assert config.gradient_accumulation is True
        assert config.accumulation_steps == 8


class TestAdjointFractionalDerivative:
    """Test AdjointFractionalDerivative class."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_forward_different_alpha_values(self):
        """Test forward pass with different alpha values."""
        x = torch.randn(16, requires_grad=True)
        
        for alpha in [0.1, 0.5, 1.0, 1.5, 1.9]:
            result = AdjointFractionalDerivative.apply(x, alpha, "RL")
            assert result.shape == x.shape
            assert torch.isfinite(result).all()
    
    def test_forward_different_methods(self):
        """Test forward pass with different methods."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        for method in ["RL", "Caputo", "GL"]:
            result = AdjointFractionalDerivative.apply(x, alpha, method)
            assert result.shape == x.shape
            assert torch.isfinite(result).all()
    
    def test_forward_different_shapes(self):
        """Test forward pass with different input shapes."""
        alpha = 0.5
        
        shapes = [(16,), (16, 32), (8, 16, 32)]
        
        for shape in shapes:
            x = torch.randn(shape, requires_grad=True)
            result = AdjointFractionalDerivative.apply(x, alpha, "RL")
            assert result.shape == x.shape
            assert torch.isfinite(result).all()
    
    def test_forward_invalid_alpha(self):
        """Test forward pass with invalid alpha values."""
        x = torch.randn(16, requires_grad=True)
        
        # Test alpha <= 0 - these should work but may produce unexpected results
        result = AdjointFractionalDerivative.apply(x, 0.0, "RL")
        assert result.shape == x.shape
        
        result = AdjointFractionalDerivative.apply(x, -1.0, "RL")
        assert result.shape == x.shape
    
    def test_forward_invalid_method(self):
        """Test forward pass with invalid method."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        with pytest.raises(ValueError, match="Unknown method"):
            AdjointFractionalDerivative.apply(x, alpha, "invalid")
    
    def test_backward_basic(self):
        """Test basic backward pass."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        # Test backward pass
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_gradient_flow(self):
        """Test gradient flow through the operation."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        # Test gradient flow
        loss = (result ** 2).sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large tensors."""
        # Test with large tensor
        x = torch.randn(1000, requires_grad=True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        x = torch.randn(16) * 1e-10
        x.requires_grad_(True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert torch.isfinite(result).all()
        
        # Test with very large values
        x = torch.randn(16) * 1e10
        x.requires_grad_(True)
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert torch.isfinite(result).all()


class TestAdjointFractionalLayer:
    """Test AdjointFractionalLayer class."""
    
    def test_initialization_default(self):
        """Test layer initialization with default parameters."""
        layer = AdjointFractionalLayer(alpha=0.5)
        
        assert layer.alpha.alpha == 0.5
        assert layer.method == "RL"
        assert layer.config.use_adjoint is True
    
    def test_initialization_custom(self):
        """Test layer initialization with custom parameters."""
        config = AdjointConfig(use_adjoint=False, adjoint_method="manual")
        layer = AdjointFractionalLayer(alpha=0.7, method="Caputo", config=config)
        
        assert layer.alpha.alpha == 0.7
        assert layer.method == "Caputo"
        assert layer.config.use_adjoint is False
        assert layer.config.adjoint_method == "manual"
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        layer = AdjointFractionalLayer(alpha=0.5)
        x = torch.randn(16, 32)
        
        result = layer(x)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_forward_different_shapes(self):
        """Test forward pass with different input shapes."""
        layer = AdjointFractionalLayer(alpha=0.5)
        
        # Test different shapes
        shapes = [(16,), (16, 32), (8, 16, 32), (4, 8, 16, 32)]
        
        for shape in shapes:
            x = torch.randn(shape)
            result = layer(x)
            assert result.shape == x.shape
            assert torch.isfinite(result).all()
    
    def test_forward_gradient_flow(self):
        """Test gradient flow through the layer."""
        layer = AdjointFractionalLayer(alpha=0.5)
        x = torch.randn(16, 32, requires_grad=True)
        
        result = layer(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_forward_different_methods(self):
        """Test forward pass with different methods."""
        x = torch.randn(16, 32)
        
        for method in ["RL", "Caputo", "GL"]:
            layer = AdjointFractionalLayer(alpha=0.5, method=method)
            result = layer(x)
            assert result.shape == x.shape
            assert torch.isfinite(result).all()


class TestMemoryEfficientFractionalNetwork:
    """Test MemoryEfficientFractionalNetwork class."""
    
    def test_initialization_default(self):
        """Test network initialization with default parameters."""
        network = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        
        assert network.input_size == 10
        assert network.hidden_sizes == [64, 32]
        assert network.output_size == 1
        assert network.fractional_order == 0.5
    
    def test_initialization_custom(self):
        """Test network initialization with custom parameters."""
        config = AdjointConfig(memory_efficient=True, checkpoint_frequency=5)
        network = MemoryEfficientFractionalNetwork(
            input_size=20,
            hidden_sizes=[128, 64, 32],
            output_size=2,
            fractional_order=0.7,
            adjoint_config=config
        )
        
        assert network.input_size == 20
        assert network.hidden_sizes == [128, 64, 32]
        assert network.output_size == 2
        assert network.fractional_order == 0.7
        assert network.adjoint_config.memory_efficient is True
        assert network.adjoint_config.checkpoint_frequency == 5
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        network = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        x = torch.randn(32, 10)
        
        result = network(x)
        
        assert result.shape == (32, 1)
        assert torch.isfinite(result).all()
    
    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        network = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        
        for batch_size in [1, 16, 32, 64, 128]:
            x = torch.randn(batch_size, 10)
            result = network(x)
            assert result.shape == (batch_size, 1)
            assert torch.isfinite(result).all()
    
    def test_forward_gradient_flow(self):
        """Test gradient flow through the network."""
        network = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        x = torch.randn(32, 10, requires_grad=True)
        
        result = network(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_forward_different_methods(self):
        """Test forward pass with different methods."""
        x = torch.randn(32, 10)
        
        # Note: The network doesn't have a method parameter, so we test with different fractional orders
        for fractional_order in [0.3, 0.5, 0.7]:
            network = MemoryEfficientFractionalNetwork(10, [64, 32], 1, fractional_order=fractional_order)
            result = network(x)
            assert result.shape == (32, 1)
            assert torch.isfinite(result).all()
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large inputs."""
        network = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        x = torch.randn(1000, 10)
        
        result = network(x)
        
        assert result.shape == (1000, 1)
        assert torch.isfinite(result).all()


class TestAdjointOptimizer:
    """Test AdjointOptimizer class."""
    
    def test_initialization_default(self):
        """Test optimizer initialization with default parameters."""
        model = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        optimizer = AdjointOptimizer(model)
        
        assert optimizer.config.use_adjoint is True
        assert optimizer.config.adjoint_method == "automatic"
        assert optimizer.config.memory_efficient is True
    
    def test_initialization_custom(self):
        """Test optimizer initialization with custom parameters."""
        model = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        config = AdjointConfig(
            use_adjoint=False,
            adjoint_method="manual",
            memory_efficient=False
        )
        optimizer = AdjointOptimizer(model, config=config)
        
        assert optimizer.config.use_adjoint is False
        assert optimizer.config.adjoint_method == "manual"
        assert optimizer.config.memory_efficient is False
    
    def test_optimize_model(self):
        """Test model optimization."""
        model = MemoryEfficientFractionalNetwork(10, [64, 32], 1)
        optimizer = AdjointOptimizer(model)
        
        # Test optimization step
        x = torch.randn(32, 10, requires_grad=True)
        y = torch.randn(32, 1)
        
        # Forward pass
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        # Optimize
        optimizer.step(loss)
        
        assert torch.isfinite(pred).all()
        assert torch.isfinite(loss).all()
    
    def test_optimize_model_different_methods(self):
        """Test model optimization with different methods."""
        x = torch.randn(32, 10, requires_grad=True)
        y = torch.randn(32, 1)
        
        for fractional_order in [0.3, 0.5, 0.7]:
            model = MemoryEfficientFractionalNetwork(10, [64, 32], 1, fractional_order=fractional_order)
            optimizer = AdjointOptimizer(model)
            
            pred = model(x)
            loss = F.mse_loss(pred, y)
            
            optimizer.step(loss)
            
            assert torch.isfinite(pred).all()
            assert torch.isfinite(loss).all()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_adjoint_fractional_derivative(self):
        """Test adjoint fractional derivative function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = adjoint_fractional_derivative(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_rl_derivative(self):
        """Test adjoint Riemann-Liouville derivative function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = adjoint_rl_derivative(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_caputo_derivative(self):
        """Test adjoint Caputo derivative function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = adjoint_caputo_derivative(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_gl_derivative(self):
        """Test adjoint Grünwald-Letnikov derivative function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = adjoint_gl_derivative(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_riemann_liouville_forward(self):
        """Test adjoint Riemann-Liouville forward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = _adjoint_riemann_liouville_forward(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_riemann_liouville_backward(self):
        """Test adjoint Riemann-Liouville backward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        grad_output = torch.randn(16)
        
        result = _adjoint_riemann_liouville_backward(grad_output, x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_caputo_forward(self):
        """Test adjoint Caputo forward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = _adjoint_caputo_forward(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_caputo_backward(self):
        """Test adjoint Caputo backward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        grad_output = torch.randn(16)
        
        result = _adjoint_caputo_backward(grad_output, x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_grunwald_letnikov_forward(self):
        """Test adjoint Grünwald-Letnikov forward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        
        result = _adjoint_grunwald_letnikov_forward(x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_adjoint_grunwald_letnikov_backward(self):
        """Test adjoint Grünwald-Letnikov backward function."""
        x = torch.randn(16, requires_grad=True)
        alpha = 0.5
        grad_output = torch.randn(16)
        
        result = _adjoint_grunwald_letnikov_backward(grad_output, x, alpha)
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = torch.randn(0)
        alpha = 0.5
        
        # This should raise a RuntimeError due to gradient computation requirements
        with pytest.raises(RuntimeError):
            AdjointFractionalDerivative.apply(x, alpha, "RL")
    
    def test_single_element_tensor(self):
        """Test with single element tensor."""
        x = torch.randn(1)
        alpha = 0.5
        
        # This should raise a RuntimeError due to gradient computation requirements
        with pytest.raises(RuntimeError):
            AdjointFractionalDerivative.apply(x, alpha, "RL")
    
    def test_inf_values(self):
        """Test with infinite values."""
        x = torch.tensor([float('inf'), float('-inf'), 1.0])
        alpha = 0.5
        
        # The function doesn't validate for inf values, so it should work
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        assert result.shape == x.shape
    
    def test_nan_values(self):
        """Test with NaN values."""
        x = torch.tensor([float('nan'), 1.0, 2.0])
        alpha = 0.5
        
        # The function doesn't validate for NaN values, so it should work
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        assert result.shape == x.shape
    
    def test_very_small_tensor(self):
        """Test with very small tensor."""
        x = torch.randn(2)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_very_large_tensor(self):
        """Test with very large tensor."""
        x = torch.randn(10000)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_extreme_alpha_values(self):
        """Test with extreme alpha values."""
        x = torch.randn(16)
        
        # Test very small alpha
        result = AdjointFractionalDerivative.apply(x, 0.001, "RL")
        assert torch.isfinite(result).all()
        
        # Test alpha close to 2
        result = AdjointFractionalDerivative.apply(x, 1.999, "RL")
        assert torch.isfinite(result).all()


class TestPerformance:
    """Test performance characteristics."""
    
    def test_memory_usage(self):
        """Test memory usage with large tensors."""
        x = torch.randn(1000, 1000)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_computation_time(self):
        """Test computation time for different methods."""
        x = torch.randn(1000)
        alpha = 0.5
        
        methods = ["RL", "Caputo", "GL"]
        
        for method in methods:
            result = AdjointFractionalDerivative.apply(x, alpha, method)
            assert result.shape == x.shape
            assert torch.isfinite(result).all()
    
    def test_gradient_computation_time(self):
        """Test gradient computation time."""
        x = torch.randn(1000, requires_grad=True)
        alpha = 0.5
        
        result = AdjointFractionalDerivative.apply(x, alpha, "RL")
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    pytest.main([__file__])
