#!/usr/bin/env python3
"""
Comprehensive tests for spectral autograd module to achieve 85% coverage.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.spectral_autograd import (
    fractional_derivative,
    create_fractional_layer,
    spectral_fractional_derivative,
    robust_fft,
    robust_ifft,
    safe_fft,
    safe_ifft
)
from hpfracc.core.definitions import FractionalOrder


class TestSpectralAutogradCoverage:
    """Comprehensive tests for spectral autograd functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.alpha = 0.5
        self.test_input = torch.randn(5, self.input_size)
        
    def test_spectral_fractional_layer_initialization(self):
        """Test SpectralFractionalLayer initialization variants."""
        # Fixed alpha
        layer1 = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        assert layer1.input_size == self.input_size
        assert layer1.alpha_value == self.alpha
        assert not layer1.learnable_alpha
        
        # Learnable alpha
        layer2 = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        assert layer2.learnable_alpha
        assert hasattr(layer2, 'alpha_param')
        
        # Different backend
        layer3 = SpectralFractionalLayer(self.input_size, alpha=self.alpha, backend='original')
        assert layer3.backend == 'original'
        
    def test_spectral_fractional_layer_forward(self):
        """Test forward pass of SpectralFractionalLayer."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        output = layer(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert torch.is_tensor(output)
        assert not torch.allclose(output, self.test_input)  # Should transform input
        
    def test_learnable_alpha_forward(self):
        """Test forward pass with learnable alpha."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        output = layer(self.test_input)
        
        assert output.shape == self.test_input.shape
        assert layer.alpha_param.requires_grad
        
    def test_get_alpha_method(self):
        """Test get_alpha method."""
        # Fixed alpha
        layer1 = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        assert layer1.get_alpha() == self.alpha
        
        # Learnable alpha
        layer2 = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        alpha_val = layer2.get_alpha()
        assert isinstance(alpha_val, (float, torch.Tensor))
        
    def test_backend_selection(self):
        """Test different backend selections."""
        backends = ['jax', 'original', 'robust']
        
        for backend in backends:
            layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, backend=backend)
            assert layer.backend == backend
            
            # Test forward pass works with each backend
            output = layer(self.test_input)
            assert output.shape == self.test_input.shape
            
    def test_spectral_fractional_network_initialization(self):
        """Test SpectralFractionalNetwork initialization."""
        hidden_sizes = [20, 15, 10]
        network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=5,
            alpha=self.alpha
        )
        
        assert len(network.layers) == len(hidden_sizes) + 3  # hidden layers + spectral + activation + output
        assert network.input_size == self.input_size
        assert network.output_size == 5
        
    def test_spectral_fractional_network_forward(self):
        """Test SpectralFractionalNetwork forward pass."""
        network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=[20, 15],
            output_size=5,
            alpha=self.alpha
        )
        
        output = network(self.test_input)
        assert output.shape == (5, 5)  # batch_size=5, output_size=5
        
    def test_network_with_different_activations(self):
        """Test network with different activation functions."""
        activations = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
        
        for activation in activations:
            network = SpectralFractionalNetwork(
                input_size=self.input_size,
                hidden_sizes=[20],
                output_size=5,
                alpha=self.alpha,
                activation=activation
            )
            
            output = network(self.test_input)
            assert output.shape == (5, 5)
            
    def test_learnable_alpha_network(self):
        """Test network with learnable alpha."""
        network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=[20],
            output_size=5,
            alpha=self.alpha,
            learnable_alpha=True
        )
        
        # Check that alpha parameters exist and require gradients
        alpha_params = [p for p in network.parameters() if p.requires_grad and p.numel() == 1]
        assert len(alpha_params) > 0
        
        output = network(self.test_input)
        assert output.shape == (5, 5)
        
    def test_create_fractional_layer_function(self):
        """Test create_fractional_layer convenience function."""
        # Fixed alpha
        layer1 = create_fractional_layer(self.input_size, alpha=self.alpha)
        assert isinstance(layer1, SpectralFractionalLayer)
        assert not layer1.learnable_alpha
        
        # Learnable alpha
        layer2 = create_fractional_layer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        assert layer2.learnable_alpha
        
    def test_spectral_fractional_function(self):
        """Test SpectralFractionalFunction static methods."""
        # Test forward method exists and is callable
        assert hasattr(SpectralFractionalFunction, 'forward')
        assert hasattr(SpectralFractionalFunction, 'backward')
        
        # Test function application
        func = SpectralFractionalFunction()
        # Note: Direct testing of autograd functions is complex, 
        # but we can test that they exist and are properly structured
        assert callable(func.forward)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through spectral layers."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        
        # Create input with gradients
        input_tensor = self.test_input.clone().requires_grad_(True)
        
        # Forward pass
        output = layer(input_tensor)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert layer.alpha_param.grad is not None
        
    def test_different_input_sizes(self):
        """Test layers with different input sizes."""
        input_sizes = [1, 5, 10, 50, 100]
        
        for size in input_sizes:
            layer = SpectralFractionalLayer(size, alpha=self.alpha)
            test_input = torch.randn(3, size)
            output = layer(test_input)
            assert output.shape == test_input.shape
            
    def test_different_alpha_values(self):
        """Test layers with different alpha values."""
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5]
        
        for alpha in alpha_values:
            layer = SpectralFractionalLayer(self.input_size, alpha=alpha)
            output = layer(self.test_input)
            assert output.shape == self.test_input.shape
            assert layer.get_alpha() == alpha
            
    def test_batch_processing(self):
        """Test processing different batch sizes."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        batch_sizes = [1, 5, 10, 32]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, self.input_size)
            output = layer(test_input)
            assert output.shape == (batch_size, self.input_size)
            
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid alpha values
        with pytest.raises((ValueError, AssertionError)):
            SpectralFractionalLayer(self.input_size, alpha=-0.5)
            
        with pytest.raises((ValueError, AssertionError)):
            SpectralFractionalLayer(self.input_size, alpha=2.5)
            
        # Invalid input size
        with pytest.raises((ValueError, TypeError)):
            SpectralFractionalLayer(0, alpha=self.alpha)
            
    def test_device_compatibility(self):
        """Test device compatibility."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        # CPU tensor
        cpu_input = torch.randn(5, self.input_size)
        cpu_output = layer(cpu_input)
        assert cpu_output.device == cpu_input.device
        
        # Test that layer can be moved to different devices
        if torch.cuda.is_available():
            layer_cuda = layer.cuda()
            cuda_input = cpu_input.cuda()
            cuda_output = layer_cuda(cuda_input)
            assert cuda_output.device == cuda_input.device
            
    def test_dtype_compatibility(self):
        """Test dtype compatibility."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        dtypes = [torch.float32, torch.float64]
        for dtype in dtypes:
            test_input = torch.randn(5, self.input_size, dtype=dtype)
            output = layer(test_input)
            # Output dtype should be compatible with input dtype
            assert output.dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]
            
    def test_reproducibility(self):
        """Test reproducibility with fixed seeds."""
        torch.manual_seed(42)
        layer1 = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        torch.manual_seed(42)
        layer2 = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        torch.manual_seed(123)
        test_input = torch.randn(5, self.input_size)
        
        output1 = layer1(test_input)
        output2 = layer2(test_input)
        
        # Outputs should be similar (allowing for some numerical differences)
        assert torch.allclose(output1, output2, atol=1e-6)
        
    def test_network_training_mode(self):
        """Test network behavior in training vs eval mode."""
        network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=[20],
            output_size=5,
            alpha=self.alpha
        )
        
        # Training mode
        network.train()
        train_output = network(self.test_input)
        
        # Eval mode
        network.eval()
        eval_output = network(self.test_input)
        
        # Outputs should be the same for this type of layer
        # (no dropout or batch norm that changes behavior)
        assert train_output.shape == eval_output.shape
        
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        
        # Check alpha parameter is initialized correctly
        alpha_val = layer.get_alpha()
        assert 0 < alpha_val < 2  # Valid range for fractional order
        
    def test_complex_network_architectures(self):
        """Test more complex network architectures."""
        # Deep network
        deep_network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=[50, 40, 30, 20, 10],
            output_size=5,
            alpha=self.alpha
        )
        
        output = deep_network(self.test_input)
        assert output.shape == (5, 5)
        
        # Wide network
        wide_network = SpectralFractionalNetwork(
            input_size=self.input_size,
            hidden_sizes=[100, 100],
            output_size=5,
            alpha=self.alpha
        )
        
        output = wide_network(self.test_input)
        assert output.shape == (5, 5)
        
    def test_backend_fallback_behavior(self):
        """Test backend fallback behavior."""
        # Test with unavailable backend (should fallback gracefully)
        with patch('hpfracc.ml.spectral_autograd.JAX_AVAILABLE', False):
            layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, backend='jax')
            # Should still work by falling back to original backend
            output = layer(self.test_input)
            assert output.shape == self.test_input.shape
            
    def test_memory_efficiency(self):
        """Test memory efficiency with large inputs."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        # Process multiple batches to test memory handling
        for _ in range(10):
            large_input = torch.randn(50, self.input_size)
            output = layer(large_input)
            assert output.shape == large_input.shape
            
    def test_edge_case_inputs(self):
        """Test edge case inputs."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha)
        
        # Very small values
        small_input = torch.full((5, self.input_size), 1e-8)
        small_output = layer(small_input)
        assert torch.all(torch.isfinite(small_output))
        
        # Very large values
        large_input = torch.full((5, self.input_size), 1e8)
        large_output = layer(large_input)
        assert torch.all(torch.isfinite(large_output))
        
        # Zero input
        zero_input = torch.zeros(5, self.input_size)
        zero_output = layer(zero_input)
        assert torch.all(torch.isfinite(zero_output))
        
    def test_integration_with_standard_layers(self):
        """Test integration with standard PyTorch layers."""
        class HybridNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(self.input_size, 20)
                self.fractional = SpectralFractionalLayer(20, alpha=self.alpha)
                self.linear2 = nn.Linear(20, 5)
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.fractional(x)
                x = self.linear2(x)
                return x
                
        network = HybridNetwork()
        output = network(self.test_input)
        assert output.shape == (5, 5)
        
    def test_serialization_compatibility(self):
        """Test that layers can be saved and loaded."""
        layer = SpectralFractionalLayer(self.input_size, alpha=self.alpha, learnable_alpha=True)
        
        # Get original output
        original_output = layer(self.test_input)
        
        # Save and load state dict
        state_dict = layer.state_dict()
        new_layer = SpectralFractionalLayer(self.input_size, alpha=0.1, learnable_alpha=True)
        new_layer.load_state_dict(state_dict)
        
        # Should produce same output
        new_output = new_layer(self.test_input)
        assert torch.allclose(original_output, new_output, atol=1e-6)
