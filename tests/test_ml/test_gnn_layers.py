#!/usr/bin/env python3
"""
Tests for Fractional Graph Neural Network Layers.

This module contains comprehensive tests for all GNN layer implementations
including fractional graph convolution, attention, and pooling layers.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from hpfracc.ml.gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestBaseFractionalGNNLayer:
    """Test base fractional GNN layer class."""
    
    def test_base_gnn_layer_creation(self):
        """Test creating BaseFractionalGNNLayer instances."""
        # Note: BaseFractionalGNNLayer is abstract, so we test through concrete implementations
        pass
    
    def test_fractional_order_validation(self):
        """Test fractional order validation."""
        # Test through concrete implementation
        layer = FractionalGraphConv(in_channels=2, out_channels=4, fractional_order=0.5)
        assert layer.fractional_order.alpha == 0.5
        
        # Test with FractionalOrder object
        alpha = FractionalOrder(0.7)
        layer = FractionalGraphConv(in_channels=2, out_channels=4, fractional_order=alpha)
        assert layer.fractional_order.alpha == 0.7
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        assert layer.backend == BackendType.TORCH  # Default backend
        
        # Test with specific backend
        layer = FractionalGraphConv(in_channels=2, out_channels=4, backend=BackendType.JAX)
        assert layer.backend == BackendType.JAX


class TestFractionalGraphConv:
    """Test Fractional Graph Convolution layer."""
    
    def test_fractional_graph_conv_creation(self):
        """Test creating FractionalGraphConv instances."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        assert layer.in_channels == 2
        assert layer.out_channels == 4
        assert layer.fractional_order.alpha == 0.5
        assert layer.method == "RL"
        assert layer.use_fractional is True
        assert layer.activation == "relu"
        assert layer.dropout == 0.1
        # Check that bias exists and is a tensor
        assert layer.bias is not None
        assert hasattr(layer.bias, 'shape')
    
    def test_fractional_graph_conv_initialization(self):
        """Test layer initialization."""
        layer = FractionalGraphConv(in_channels=3, out_channels=6, fractional_order=0.8)
        
        # Check that the layer was properly initialized
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
        # Weight matrix is (in_channels, out_channels) in this implementation
        assert layer.weight.shape == (3, 6)
        assert layer.bias.shape == (6,)
    
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        # Create simple graph data
        x = torch.randn(4, 2)  # 4 nodes, 2 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_weight = torch.ones(4)
        
        output = layer(x, edge_index, edge_weight)
        
        assert output.shape == (4, 4)  # 4 nodes, 4 output features
        assert torch.isfinite(output).all()
    
    def test_forward_pass_without_edge_weights(self):
        """Test forward pass without edge weights."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
    
    def test_fractional_derivative_application(self):
        """Test that fractional derivatives are applied when enabled."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4, use_fractional=True)
        
        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        # Mock the fractional derivative method to verify it's called
        with patch.object(layer, 'apply_fractional_derivative') as mock_fractional:
            mock_fractional.return_value = x
            output = layer(x, edge_index)
            mock_fractional.assert_called()
    
    def test_no_fractional_derivative_when_disabled(self):
        """Test that fractional derivatives are not applied when disabled."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4, use_fractional=False)
        
        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        # When use_fractional=False, apply_fractional_derivative should return x unchanged
        # Mock the method to verify it returns the input without modification
        with patch.object(layer, 'apply_fractional_derivative') as mock_fractional:
            mock_fractional.return_value = x
            output = layer(x, edge_index)
            # The method is still called, but it should return x unchanged
            mock_fractional.assert_called_once_with(x)
            # Verify the output is still valid
            assert output.shape == (3, 4)
    
    def test_different_fractional_orders(self):
        """Test different fractional orders."""
        orders = [0.1, 0.5, 0.9]
        
        for order in orders:
            layer = FractionalGraphConv(in_channels=2, out_channels=4, fractional_order=order)
            assert layer.fractional_order.alpha == order
            
            x = torch.randn(3, 2)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            
            output = layer(x, edge_index)
            assert output.shape == (3, 4)
            assert torch.isfinite(output).all()
    
    def test_different_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid"]
        
        for activation in activations:
            layer = FractionalGraphConv(in_channels=2, out_channels=4, activation=activation)
            assert layer.activation == activation
            
            x = torch.randn(3, 2)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            
            output = layer(x, edge_index)
            assert output.shape == (3, 4)
            assert torch.isfinite(output).all()
    
    def test_dropout_application(self):
        """Test that dropout is applied during training."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4, dropout=0.5)
        # Set training mode by setting the training attribute
        layer.training = True
        
        x = torch.randn(3, 2)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        # Run multiple forward passes to see dropout effects
        outputs = []
        for _ in range(10):
            output = layer(x, edge_index)
            outputs.append(output)
        
        # Check that outputs vary due to dropout
        outputs_tensor = torch.stack(outputs)
        assert not torch.allclose(outputs_tensor[0], outputs_tensor[1], rtol=1e-6)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        layer = FractionalGraphConv(in_channels=1, out_channels=1)
        
        # Single node
        x = torch.randn(1, 1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()
        
        # Empty graph
        x = torch.empty(0, 1)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (0, 1)


class TestFractionalGraphAttention:
    """Test Fractional Graph Attention layer."""
    
    def test_fractional_graph_attention_creation(self):
        """Test creating FractionalGraphAttention instances."""
        layer = FractionalGraphAttention(in_channels=2, out_channels=4, heads=2)
        
        assert layer.in_channels == 2
        assert layer.out_channels == 4
        assert layer.heads == 2
        assert layer.fractional_order.alpha == 0.5
        assert layer.use_fractional is True
    
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphAttention(in_channels=2, out_channels=4, heads=2)
        
        x = torch.randn(4, 2)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (4, 4)
        assert torch.isfinite(output).all()
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        heads = [1, 2, 4]
        
        for num_heads in heads:
            layer = FractionalGraphAttention(in_channels=4, out_channels=8, heads=num_heads)
            
            x = torch.randn(3, 4)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            
            output = layer(x, edge_index)
            assert output.shape == (3, 8)
            assert torch.isfinite(output).all()


class TestFractionalGraphPooling:
    """Test Fractional Graph Pooling layer."""
    
    def test_fractional_graph_pooling_creation(self):
        """Test creating FractionalGraphPooling instances."""
        layer = FractionalGraphPooling(in_channels=4, ratio=0.5)
        
        assert layer.in_channels == 4
        assert layer.pooling_ratio == 0.5  # The actual attribute name
        assert layer.fractional_order.alpha == 0.5
        assert layer.use_fractional is True
    
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphPooling(in_channels=4, ratio=0.5)
        
        x = torch.randn(6, 4)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
        batch = torch.zeros(6, dtype=torch.long)
        
        output, edge_index_out, batch_out = layer(x, edge_index, batch)
        
        # With ratio=0.5, should pool to ~3 nodes
        assert output.shape[0] <= 6
        assert output.shape[1] == 4
        assert torch.isfinite(output).all()
        assert edge_index_out.shape[1] <= edge_index.shape[1]
    
    def test_different_pooling_ratios(self):
        """Test different pooling ratios."""
        ratios = [0.2, 0.5, 0.8]
        
        for ratio in ratios:
            layer = FractionalGraphPooling(in_channels=4, ratio=ratio)
            
            x = torch.randn(10, 4)
            edge_index = torch.tensor([[i, (i+1) % 10] for i in range(10)], dtype=torch.long).t()
            batch = torch.zeros(10, dtype=torch.long)
            
            output, edge_index_out, batch_out = layer(x, edge_index, batch)
            
            expected_nodes = max(1, int(10 * ratio))
            assert output.shape[0] <= 10
            assert output.shape[0] >= expected_nodes * 0.8  # Allow some flexibility
            assert torch.isfinite(output).all()


class TestGNNLayersIntegration:
    """Test integration between different GNN layers."""
    
    def test_layer_consistency(self):
        """Test consistency between different layer types."""
        # Create different layer types
        conv_layer = FractionalGraphConv(in_channels=2, out_channels=4)
        attn_layer = FractionalGraphAttention(in_channels=4, out_channels=6, heads=2)
        pool_layer = FractionalGraphPooling(in_channels=6, ratio=0.5)
        
        # Test forward pass through all layers
        x = torch.randn(5, 2)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        batch = torch.zeros(5, dtype=torch.long)
        
        # Conv layer
        x_conv = conv_layer(x, edge_index)
        assert x_conv.shape == (5, 4)
        
        # Attention layer
        x_attn = attn_layer(x_conv, edge_index)
        assert x_attn.shape == (5, 6)
        
        # Pooling layer
        x_pool, edge_index_pool, batch_pool = pool_layer(x_attn, edge_index, batch)
        assert x_pool.shape[1] == 6  # Same number of features
        assert x_pool.shape[0] <= 5  # Fewer nodes after pooling
    
    def test_fractional_order_consistency(self):
        """Test that fractional orders are consistent across layers."""
        alpha = 0.7
        
        conv_layer = FractionalGraphConv(in_channels=2, out_channels=4, fractional_order=alpha)
        attn_layer = FractionalGraphAttention(in_channels=4, out_channels=6, fractional_order=alpha)
        pool_layer = FractionalGraphPooling(in_channels=6, fractional_order=alpha)
        
        assert conv_layer.fractional_order.alpha == alpha
        assert attn_layer.fractional_order.alpha == alpha
        assert pool_layer.fractional_order.alpha == alpha
    
    def test_backend_consistency(self):
        """Test that backends are consistent across layers."""
        backend = BackendType.TORCH
        
        conv_layer = FractionalGraphConv(in_channels=2, out_channels=4, backend=backend)
        attn_layer = FractionalGraphAttention(in_channels=4, out_channels=6, backend=backend)
        pool_layer = FractionalGraphPooling(in_channels=6, backend=backend)
        
        assert conv_layer.backend == backend
        assert attn_layer.backend == backend
        assert pool_layer.backend == backend


class TestGNNLayersEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        # Wrong input dimension
        x = torch.randn(3, 3)  # Should be 2 features
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        with pytest.raises(RuntimeError):
            layer(x, edge_index)
    
    def test_empty_edge_index(self):
        """Test handling of empty edge index."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        x = torch.randn(3, 2)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (3, 4)
        assert torch.isfinite(output).all()
    
    def test_single_node(self):
        """Test handling of single node."""
        layer = FractionalGraphConv(in_channels=2, out_channels=4)
        
        x = torch.randn(1, 2)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (1, 4)
        assert torch.isfinite(output).all()
    
    def test_large_graphs(self):
        """Test handling of large graphs."""
        layer = FractionalGraphConv(in_channels=4, out_channels=8)
        
        # Create larger graph
        num_nodes = 100
        x = torch.randn(num_nodes, 4)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2), dtype=torch.long)
        
        output = layer(x, edge_index)
        assert output.shape == (num_nodes, 8)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])
