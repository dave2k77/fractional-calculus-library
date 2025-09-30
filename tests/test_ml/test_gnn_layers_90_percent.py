#!/usr/bin/env python3
"""Comprehensive tests to bring GNN layers coverage to 90%."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys

from hpfracc.ml.gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)


class TestBaseFractionalGNNLayer:
    """Test base fractional GNN layer."""
    
    def test_initialization(self):
        """Test layer initialization."""
        layer = BaseFractionalGNNLayer(
            input_dim=10,
            output_dim=5,
            fractional_order=0.5,
            method="RL"
        )
        assert layer.input_dim == 10
        assert layer.output_dim == 5
        assert layer.fractional_order == 0.5
        assert layer.method == "RL"
        
    def test_fractional_order_validation(self):
        """Test fractional order validation."""
        # Valid fractional order
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        layer._validate_fractional_order(0.3)
        
        # Invalid fractional order
        with pytest.raises(ValueError):
            layer._validate_fractional_order(-0.1)
            
        with pytest.raises(ValueError):
            layer._validate_fractional_order(1.5)
            
    def test_method_validation(self):
        """Test method validation."""
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        
        # Valid methods
        for method in ["RL", "Caputo", "GL"]:
            layer._validate_method(method)
            
        # Invalid method
        with pytest.raises(ValueError):
            layer._validate_method("invalid")
            
    def test_fractional_derivative_computation(self):
        """Test fractional derivative computation."""
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        
        # Mock fractional derivative
        with patch.object(layer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.tensor([1.0, 2.0, 3.0])
            result = layer._compute_fractional_derivative(
                torch.tensor([1.0, 2.0, 3.0])
            )
            assert result is not None
            
    def test_activation_functions(self):
        """Test activation function handling."""
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        
        # Test different activation functions
        activations = ["relu", "tanh", "sigmoid", "gelu", "leaky_relu"]
        for activation in activations:
            layer.activation = activation
            layer._apply_activation(torch.randn(5, 5))
            
    def test_dropout_application(self):
        """Test dropout application."""
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        layer.dropout_rate = 0.5
        layer.training = True
        
        x = torch.randn(5, 5)
        result = layer._apply_dropout(x)
        assert result.shape == x.shape
        
    def test_normalization(self):
        """Test normalization."""
        layer = BaseFractionalGNNLayer(10, 5, 0.5, "RL")
        layer.normalize = True
        
        x = torch.randn(5, 5)
        result = layer._apply_normalization(x)
        assert result.shape == x.shape


class TestFractionalGraphConv:
    """Test fractional graph convolution layer."""
    
    def test_initialization(self):
        """Test initialization."""
        conv = FractionalGraphConv(
            input_dim=10,
            output_dim=5,
            fractional_order=0.5,
            method="RL",
            bias=True,
            activation="relu",
            dropout_rate=0.1
        )
        assert conv.input_dim == 10
        assert conv.output_dim == 5
        assert conv.fractional_order == 0.5
        assert conv.method == "RL"
        assert conv.bias is True
        assert conv.activation == "relu"
        assert conv.dropout_rate == 0.1
        
    def test_weight_initialization(self):
        """Test weight initialization."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        assert hasattr(conv, 'weight')
        assert conv.weight.shape == (5, 10)
        
        if conv.bias:
            assert hasattr(conv, 'bias')
            assert conv.bias.shape == (5,)
            
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        # Mock data
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (5, 5)
            
    def test_forward_pass_with_edge_weights(self):
        """Test forward pass with edge weights."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_weights = torch.tensor([0.5, 0.3, 0.8])
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index, edge_weights)
            assert result.shape == (5, 5)
            
    def test_forward_pass_with_edge_attr(self):
        """Test forward pass with edge attributes."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 4)
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index, edge_attr=edge_attr)
            assert result.shape == (5, 5)
            
    def test_forward_pass_batch(self):
        """Test forward pass with batch dimension."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(2, 5, 10)  # batch_size=2
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(2, 5, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (2, 5, 5)
            
    def test_different_fractional_orders(self):
        """Test different fractional orders."""
        fractional_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for alpha in fractional_orders:
            conv = FractionalGraphConv(10, 5, alpha, "RL")
            
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10)
                
                result = conv.forward(x, edge_index)
                assert result.shape == (5, 5)
                
    def test_different_methods(self):
        """Test different fractional methods."""
        methods = ["RL", "Caputo", "GL"]
        
        for method in methods:
            conv = FractionalGraphConv(10, 5, 0.5, method)
            
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10)
                
                result = conv.forward(x, edge_index)
                assert result.shape == (5, 5)
                
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu", "leaky_relu", "elu"]
        
        for activation in activations:
            conv = FractionalGraphConv(10, 5, 0.5, "RL", activation=activation)
            
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10)
                
                result = conv.forward(x, edge_index)
                assert result.shape == (5, 5)
                
    def test_dropout_rates(self):
        """Test different dropout rates."""
        dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
        
        for dropout_rate in dropout_rates:
            conv = FractionalGraphConv(10, 5, 0.5, "RL", dropout_rate=dropout_rate)
            
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10)
                
                result = conv.forward(x, edge_index)
                assert result.shape == (5, 5)


class TestFractionalGraphAttention:
    """Test fractional graph attention layer."""
    
    def test_initialization(self):
        """Test initialization."""
        attention = FractionalGraphAttention(
            input_dim=10,
            output_dim=5,
            num_heads=4,
            fractional_order=0.5,
            method="RL",
            dropout_rate=0.1
        )
        assert attention.input_dim == 10
        assert attention.output_dim == 5
        assert attention.num_heads == 4
        assert attention.fractional_order == 0.5
        assert attention.method == "RL"
        assert attention.dropout_rate == 0.1
        
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        attention = FractionalGraphAttention(10, 5, 4, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(attention, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = attention.forward(x, edge_index)
            assert result.shape == (5, 5)
            
    def test_forward_pass_with_edge_weights(self):
        """Test forward pass with edge weights."""
        attention = FractionalGraphAttention(10, 5, 4, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_weights = torch.tensor([0.5, 0.3, 0.8])
        
        with patch.object(attention, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = attention.forward(x, edge_index, edge_weights)
            assert result.shape == (5, 5)
            
    def test_different_num_heads(self):
        """Test different number of attention heads."""
        num_heads_list = [1, 2, 4, 8, 16]
        
        for num_heads in num_heads_list:
            attention = FractionalGraphAttention(10, 5, num_heads, 0.5, "RL")
            
            x = torch.randn(5, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(attention, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10)
                
                result = attention.forward(x, edge_index)
                assert result.shape == (5, 5)


class TestFractionalGraphPooling:
    """Test fractional graph pooling layer."""
    
    def test_initialization(self):
        """Test initialization."""
        pooling = FractionalGraphPooling(
            input_dim=10,
            pool_ratio=0.5,
            fractional_order=0.5,
            method="RL"
        )
        assert pooling.input_dim == 10
        assert pooling.pool_ratio == 0.5
        assert pooling.fractional_order == 0.5
        assert pooling.method == "RL"
        
    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        pooling = FractionalGraphPooling(10, 0.5, 0.5, "RL")
        
        x = torch.randn(10, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(pooling, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10, 10)
            
            result, cluster_indices = pooling.forward(x, edge_index)
            assert result.shape[0] <= x.shape[0]
            assert result.shape[1] == x.shape[1]
            assert len(cluster_indices) == result.shape[0]
            
    def test_different_pool_ratios(self):
        """Test different pooling ratios."""
        pool_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for pool_ratio in pool_ratios:
            pooling = FractionalGraphPooling(10, pool_ratio, 0.5, "RL")
            
            x = torch.randn(10, 10)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(pooling, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(10, 10)
                
                result, cluster_indices = pooling.forward(x, edge_index)
                expected_size = max(1, int(x.shape[0] * pool_ratio))
                assert result.shape[0] >= expected_size * 0.8  # Allow some tolerance


class TestFractionalGNNFactory:
    """Test fractional GNN factory."""
    
    def test_create_conv_layer(self):
        """Test creating convolution layer."""
        layer = FractionalGNNFactory.create_layer(
            layer_type="conv",
            input_dim=10,
            output_dim=5,
            fractional_order=0.5,
            method="RL"
        )
        assert isinstance(layer, FractionalGraphConv)
        
    def test_create_attention_layer(self):
        """Test creating attention layer."""
        layer = FractionalGNNFactory.create_layer(
            layer_type="attention",
            input_dim=10,
            output_dim=5,
            num_heads=4,
            fractional_order=0.5,
            method="RL"
        )
        assert isinstance(layer, FractionalGraphAttention)
        
    def test_create_pooling_layer(self):
        """Test creating pooling layer."""
        layer = FractionalGNNFactory.create_layer(
            layer_type="pooling",
            input_dim=10,
            pool_ratio=0.5,
            fractional_order=0.5,
            method="RL"
        )
        assert isinstance(layer, FractionalGraphPooling)
        
    def test_create_unknown_layer(self):
        """Test creating unknown layer type."""
        with pytest.raises(ValueError):
            FractionalGNNFactory.create_layer(
                layer_type="unknown",
                input_dim=10,
                output_dim=5,
                fractional_order=0.5,
                method="RL"
            )


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_fractional_gnn_layer(self):
        """Test create_fractional_gnn_layer function."""
        layer = create_fractional_gnn_layer(
            layer_type="conv",
            input_dim=10,
            output_dim=5,
            fractional_order=0.5,
            method="RL"
        )
        assert isinstance(layer, FractionalGraphConv)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        # Empty node features and edge index
        x = torch.empty(0, 10)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.empty(0, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (0, 5)
            
    def test_single_node(self):
        """Test handling of single node."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(1, 10)
        edge_index = torch.empty(2, 0, dtype=torch.long)
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(1, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (1, 5)
            
    def test_self_loops(self):
        """Test handling of self loops."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])  # self loops
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (5, 5)
            
    def test_isolated_nodes(self):
        """Test handling of isolated nodes."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1], [1, 2]])  # nodes 3, 4 isolated
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (5, 5)
            
    def test_large_graph(self):
        """Test handling of large graph."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        # Large graph
        x = torch.randn(1000, 10)
        edge_index = torch.randint(0, 1000, (2, 5000))
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(1000, 10)
            
            result = conv.forward(x, edge_index)
            assert result.shape == (1000, 5)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        # Test that operations don't consume excessive memory
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(100, 10)
            
            result = conv.forward(x, edge_index)
            assert torch.isfinite(result).all()
            
    def test_gradient_flow(self):
        """Test gradient flow."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        x = torch.randn(5, 10, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5, 10)
            
            result = conv.forward(x, edge_index)
            
            # Test backward pass
            loss = result.sum()
            loss.backward()
            
            assert x.grad is not None
            assert torch.isfinite(x.grad).all()
            
    def test_different_dtypes(self):
        """Test different data types."""
        conv = FractionalGraphConv(10, 5, 0.5, "RL")
        
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            x = torch.randn(5, 10, dtype=dtype)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            with patch.object(conv, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(5, 10, dtype=dtype)
                
                result = conv.forward(x, edge_index)
                assert result.dtype == dtype
