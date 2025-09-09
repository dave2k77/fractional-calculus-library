"""
Extended tests for GNN layers module to achieve 90%+ coverage.

This module contains comprehensive tests for all GNN layer classes
including edge cases, error handling, and integration scenarios.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from hpfracc.ml.gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)
from hpfracc.core.definitions import FractionalOrder
from hpfracc.ml.backends import BackendType


class ConcreteFractionalGNNLayer(BaseFractionalGNNLayer):
    """Concrete implementation for testing the abstract base class."""
    
    def _initialize_layer(self):
        """Initialize the layer for testing."""
        pass
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """Simple forward pass for testing."""
        return x
    
    def reset_parameters(self):
        """Reset parameters for testing."""
        pass


class TestExtendedBaseFractionalGNNLayer:
    """Test extended base fractional GNN layer functionality."""
    
    def test_base_layer_initialization_edge_cases(self):
        """Test base layer initialization with edge cases."""
        # Test with different fractional orders
        for alpha in [0.0, 0.1, 0.5, 0.9, 1.0]:
            layer = ConcreteFractionalGNNLayer(
                in_channels=10,
                out_channels=5,
                fractional_order=alpha
            )
            assert layer.fractional_order.alpha == alpha
        
        # Test with different methods
        for method in ["RL", "Caputo", "GL"]:
            layer = ConcreteFractionalGNNLayer(
                in_channels=8,
                out_channels=4,
                method=method
            )
            assert layer.method == method
        
        # Test with different activations
        for activation in ["relu", "tanh", "sigmoid", "gelu"]:
            layer = ConcreteFractionalGNNLayer(
                in_channels=6,
                out_channels=3,
                activation=activation
            )
            assert layer.activation == activation
    
    def test_base_layer_backend_handling(self):
        """Test base layer backend handling."""
        # Test with different backends
        for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
            layer = ConcreteFractionalGNNLayer(
                in_channels=5,
                out_channels=2,
                backend=backend
            )
            assert layer.backend == backend
    
    def test_base_layer_fractional_order_validation(self):
        """Test fractional order validation."""
        # Test with FractionalOrder object
        frac_order = FractionalOrder(0.7)
        layer = ConcreteFractionalGNNLayer(
            in_channels=4,
            out_channels=2,
            fractional_order=frac_order
        )
        assert layer.fractional_order is frac_order
        
        # Test with float
        layer = ConcreteFractionalGNNLayer(
            in_channels=4,
            out_channels=2,
            fractional_order=0.3
        )
        assert layer.fractional_order.alpha == 0.3
    
    def test_base_layer_parameter_access(self):
        """Test base layer parameter access."""
        layer = ConcreteFractionalGNNLayer(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5,
            method="RL",
            use_fractional=True,
            activation="relu",
            dropout=0.2,
            bias=False
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 5
        assert layer.fractional_order.alpha == 0.5
        assert layer.method == "RL"
        assert layer.use_fractional is True
        assert layer.activation == "relu"
        assert layer.dropout == 0.2
        assert layer.bias is None


class TestExtendedFractionalGraphConv:
    """Test extended fractional graph convolution layer."""
    
    def test_fractional_graph_conv_initialization(self):
        """Test fractional graph conv initialization."""
        conv = FractionalGraphConv(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5,
            method="RL",
            use_fractional=True,
            activation="relu",
            dropout=0.1,
            bias=True
        )
        
        assert conv.in_channels == 10
        assert conv.out_channels == 5
        assert conv.fractional_order.alpha == 0.5
        assert conv.method == "RL"
        assert conv.use_fractional is True
        assert conv.activation == "relu"
        assert conv.dropout == 0.1
        assert conv.bias is not None
    
    def test_fractional_graph_conv_forward_basic(self):
        """Test basic forward pass."""
        conv = FractionalGraphConv(
            in_channels=5,
            out_channels=3,
            fractional_order=0.3
        )
        
        # Create test data
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        # Test forward pass
        output = conv.forward(x, edge_index)
        assert output.shape == (10, 3)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_conv_forward_with_edge_weights(self):
        """Test forward pass with edge weights."""
        conv = FractionalGraphConv(
            in_channels=4,
            out_channels=2,
            fractional_order=0.4
        )
        
        x = torch.randn(8, 4)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                  [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        edge_weight = torch.randn(8)
        
        output = conv.forward(x, edge_index, edge_weight)
        assert output.shape == (8, 2)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_conv_forward_without_fractional(self):
        """Test forward pass without fractional calculus."""
        conv = FractionalGraphConv(
            in_channels=6,
            out_channels=4,
            fractional_order=0.5,
            use_fractional=False
        )
        
        x = torch.randn(12, 6)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long)
        
        output = conv.forward(x, edge_index)
        assert output.shape == (12, 4)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_conv_different_activations(self):
        """Test with different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu"]
        
        for activation in activations:
            conv = FractionalGraphConv(
                in_channels=5,
                out_channels=3,
                activation=activation
            )
            
            x = torch.randn(10, 5)
            edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
            
            output = conv.forward(x, edge_index)
            assert output.shape == (10, 3)
            assert torch.isfinite(output).all()
    
    def test_fractional_graph_conv_dropout(self):
        """Test dropout functionality."""
        conv = FractionalGraphConv(
            in_channels=4,
            out_channels=2,
            dropout=0.5
        )
        
        x = torch.randn(8, 4)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                  [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        
        # Test in training mode
        conv.train()
        output_train = conv.forward(x, edge_index)
        
        # Test in eval mode
        conv.eval()
        output_eval = conv.forward(x, edge_index)
        
        assert output_train.shape == (8, 2)
        assert output_eval.shape == (8, 2)
        assert torch.isfinite(output_train).all()
        assert torch.isfinite(output_eval).all()
    
    def test_fractional_graph_conv_reset_parameters(self):
        """Test parameter reset functionality."""
        conv = FractionalGraphConv(
            in_channels=5,
            out_channels=3,
            fractional_order=0.5
        )
        
        # Store initial parameters
        initial_weight = conv.weight.clone()
        
        # Reset parameters
        conv.reset_parameters()
        
        # Check that parameters have been reset (should be different)
        assert not torch.equal(conv.weight, initial_weight)
    
    def test_fractional_graph_conv_edge_cases(self):
        """Test edge cases for fractional graph conv."""
        # Test with single node
        conv = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.3
        )
        
        x = torch.randn(1, 3)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        output = conv.forward(x, edge_index)
        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()
        
        # Test with empty edge index
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        output = conv.forward(x, edge_index)
        assert output.shape == (5, 2)
        assert torch.isfinite(output).all()


class TestExtendedFractionalGraphAttention:
    """Test extended fractional graph attention layer."""
    
    def test_fractional_graph_attention_initialization(self):
        """Test fractional graph attention initialization."""
        attn = FractionalGraphAttention(
            in_channels=10,
            out_channels=5,
            num_heads=4,
            fractional_order=0.5,
            method="RL",
            use_fractional=True,
            activation="relu",
            dropout=0.1,
            bias=True
        )
        
        assert attn.in_channels == 10
        assert attn.out_channels == 5
        assert attn.heads == 4
        assert attn.fractional_order.alpha == 0.5
        assert attn.method == "RL"
        assert attn.use_fractional is True
        assert attn.activation == "relu"
        assert attn.dropout == 0.1
        assert attn.bias is not None
    
    def test_fractional_graph_attention_forward_basic(self):
        """Test basic forward pass."""
        attn = FractionalGraphAttention(
            in_channels=6,
            out_channels=4,
            num_heads=2,
            fractional_order=0.4
        )
        
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        output = attn.forward(x, edge_index)
        assert output.shape == (10, 4)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_attention_forward_with_edge_weights(self):
        """Test forward pass with edge weights."""
        attn = FractionalGraphAttention(
            in_channels=5,
            out_channels=3,
            num_heads=3,
            fractional_order=0.3
        )
        
        x = torch.randn(8, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                  [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        edge_weight = torch.randn(8)
        
        output = attn.forward(x, edge_index, edge_weight)
        assert output.shape == (8, 3)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_attention_different_heads(self):
        """Test with different numbers of attention heads."""
        for num_heads in [1, 2, 4, 8]:
            attn = FractionalGraphAttention(
                in_channels=8,
                out_channels=4,
                num_heads=num_heads,
                fractional_order=0.5
            )
            
            x = torch.randn(12, 8)
            edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long)
            
            output = attn.forward(x, edge_index)
            assert output.shape == (12, 4)
            assert torch.isfinite(output).all()
    
    def test_fractional_graph_attention_without_fractional(self):
        """Test forward pass without fractional calculus."""
        attn = FractionalGraphAttention(
            in_channels=6,
            out_channels=4,
            num_heads=2,
            fractional_order=0.5,
            use_fractional=False
        )
        
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        output = attn.forward(x, edge_index)
        assert output.shape == (10, 4)
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_attention_reset_parameters(self):
        """Test parameter reset functionality."""
        attn = FractionalGraphAttention(
            in_channels=5,
            out_channels=3,
            num_heads=2,
            fractional_order=0.5
        )
        
        # Store initial parameters
        initial_query_weight = attn.query_weight.clone()
        initial_key_weight = attn.key_weight.clone()
        initial_value_weight = attn.value_weight.clone()
        initial_output_weight = attn.output_weight.clone()
        
        # Reset parameters
        attn.reset_parameters()
        
        # Check that parameters have been reset (should be different)
        assert not torch.equal(attn.query_weight, initial_query_weight)
        assert not torch.equal(attn.key_weight, initial_key_weight)
        assert not torch.equal(attn.value_weight, initial_value_weight)
        assert not torch.equal(attn.output_weight, initial_output_weight)
    
    def test_fractional_graph_attention_edge_cases(self):
        """Test edge cases for fractional graph attention."""
        # Test with single node
        attn = FractionalGraphAttention(
            in_channels=4,
            out_channels=2,
            num_heads=1,
            fractional_order=0.3
        )
        
        x = torch.randn(1, 4)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        output = attn.forward(x, edge_index)
        assert output.shape == (1, 2)
        assert torch.isfinite(output).all()


class TestExtendedFractionalGraphPooling:
    """Test extended fractional graph pooling layer."""
    
    def test_fractional_graph_pooling_initialization(self):
        """Test fractional graph pooling initialization."""
        pool = FractionalGraphPooling(
            in_channels=10,
            out_channels=5,
            pooling_ratio=0.5,
            fractional_order=0.5,
            method="RL",
            use_fractional=True,
            activation="relu",
            dropout=0.1,
            bias=True
        )
        
        assert pool.in_channels == 10
        assert pool.out_channels == 5
        assert pool.pooling_ratio == 0.5
        assert pool.fractional_order.alpha == 0.5
        assert pool.method == "RL"
        assert pool.use_fractional is True
        assert pool.activation == "relu"
        assert pool.dropout == 0.1
        assert pool.bias is True
    
    def test_fractional_graph_pooling_forward_basic(self):
        """Test basic forward pass."""
        pool = FractionalGraphPooling(
            in_channels=8,
            out_channels=4,
            pooling_ratio=0.6,
            fractional_order=0.4
        )
        
        x = torch.randn(10, 8)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index)
        assert output.shape[1] == 4  # Output channels
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_pooling_forward_with_edge_weights(self):
        """Test forward pass with edge weights."""
        pool = FractionalGraphPooling(
            in_channels=6,
            out_channels=3,
            pooling_ratio=0.7,
            fractional_order=0.3
        )
        
        x = torch.randn(8, 6)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                  [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
        edge_weight = torch.randn(8)
        
        output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index, edge_weight)
        assert output.shape[1] == 3  # Output channels
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_pooling_different_ratios(self):
        """Test with different pooling ratios."""
        for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pool = FractionalGraphPooling(
                in_channels=8,
                out_channels=4,
                pooling_ratio=ratio,
                fractional_order=0.5
            )
            
            x = torch.randn(12, 8)
            edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]], dtype=torch.long)
            
            output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index)
            assert output.shape[1] == 4  # Output channels
            assert torch.isfinite(output).all()
    
    def test_fractional_graph_pooling_without_fractional(self):
        """Test forward pass without fractional calculus."""
        pool = FractionalGraphPooling(
            in_channels=6,
            out_channels=3,
            pooling_ratio=0.5,
            fractional_order=0.5,
            use_fractional=False
        )
        
        x = torch.randn(10, 6)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index)
        assert output.shape[1] == 3  # Output channels
        assert torch.isfinite(output).all()
    
    def test_fractional_graph_pooling_reset_parameters(self):
        """Test parameter reset functionality."""
        pool = FractionalGraphPooling(
            in_channels=5,
            out_channels=3,
            pooling_ratio=0.5,
            fractional_order=0.5
        )
        
        # Store initial parameters
        initial_score_network = pool.score_network.clone()
        
        # Reset parameters
        pool.reset_parameters()
        
        # Check that parameters have been reset (should be different)
        assert not torch.equal(pool.score_network, initial_score_network)
    
    def test_fractional_graph_pooling_edge_cases(self):
        """Test edge cases for fractional graph pooling."""
        # Test with single node
        pool = FractionalGraphPooling(
            in_channels=4,
            out_channels=2,
            pooling_ratio=0.5,
            fractional_order=0.3
        )
        
        x = torch.randn(1, 4)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index)
        assert output.shape[1] == 2  # Output channels
        assert torch.isfinite(output).all()


class TestExtendedGNNLayersIntegration:
    """Test extended integration scenarios for GNN layers."""
    
    def test_layers_consistency_across_backends(self):
        """Test layer consistency across different backends."""
        layers = [
            FractionalGraphConv(5, 3, fractional_order=0.5),
            FractionalGraphAttention(5, 3, num_heads=2, fractional_order=0.5),
            FractionalGraphPooling(5, 3, pooling_ratio=0.5, fractional_order=0.5)
        ]
        
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        
        for layer in layers:
            # All layers should have these attributes
            assert hasattr(layer, 'in_channels')
            assert hasattr(layer, 'out_channels')
            assert hasattr(layer, 'fractional_order')
            assert hasattr(layer, 'forward')
            assert hasattr(layer, 'reset_parameters')
            
            # Test forward pass
            output = layer.forward(x, edge_index)
            if isinstance(output, tuple):
                # Handle pooling layers that return tuples
                output = output[0]
            assert torch.isfinite(output).all()
    
    def test_layers_fractional_order_consistency(self):
        """Test fractional order consistency across layers."""
        fractional_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for alpha in fractional_orders:
            conv = FractionalGraphConv(5, 3, fractional_order=alpha)
            attn = FractionalGraphAttention(5, 3, num_heads=2, fractional_order=alpha)
            pool = FractionalGraphPooling(5, 3, pooling_ratio=0.5, fractional_order=alpha)
            
            assert conv.fractional_order.alpha == alpha
            assert attn.fractional_order.alpha == alpha
            assert pool.fractional_order.alpha == alpha
    
    def test_layers_parameter_initialization(self):
        """Test parameter initialization across layers."""
        layers = [
            FractionalGraphConv(8, 4, fractional_order=0.5),
            FractionalGraphAttention(8, 4, num_heads=2, fractional_order=0.5),
            FractionalGraphPooling(8, 4, pooling_ratio=0.5, fractional_order=0.5)
        ]
        
        for layer in layers:
            # Test parameter reset
            layer.reset_parameters()
            
            # Check that parameters are finite
            for param in layer.parameters():
                if isinstance(param, torch.Tensor):
                    assert torch.isfinite(param).all()
    
    def test_layers_dropout_consistency(self):
        """Test dropout consistency across layers."""
        dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
        
        for dropout in dropout_rates:
            conv = FractionalGraphConv(5, 3, dropout=dropout)
            attn = FractionalGraphAttention(5, 3, num_heads=2, dropout=dropout)
            pool = FractionalGraphPooling(5, 3, pooling_ratio=0.5, dropout=dropout)
            
            assert conv.dropout == dropout
            assert attn.dropout == dropout
            assert pool.dropout == dropout


class TestExtendedGNNLayersEdgeCases:
    """Test extended edge cases for GNN layers."""
    
    def test_layers_with_extreme_fractional_orders(self):
        """Test layers with extreme fractional orders."""
        extreme_orders = [0.01, 0.99, 0.001, 0.999]
        
        for alpha in extreme_orders:
            conv = FractionalGraphConv(5, 3, fractional_order=alpha)
            attn = FractionalGraphAttention(5, 3, num_heads=2, fractional_order=alpha)
            pool = FractionalGraphPooling(5, 3, pooling_ratio=0.5, fractional_order=alpha)
            
            assert conv.fractional_order.alpha == alpha
            assert attn.fractional_order.alpha == alpha
            assert pool.fractional_order.alpha == alpha
    
    def test_layers_with_large_graphs(self):
        """Test layers with large graphs."""
        # Test with larger graph
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        
        conv = FractionalGraphConv(10, 5, fractional_order=0.5)
        attn = FractionalGraphAttention(10, 5, num_heads=4, fractional_order=0.5)
        pool = FractionalGraphPooling(10, 5, pooling_ratio=0.3, fractional_order=0.5)
        
        # Test forward passes
        conv_output = conv.forward(x, edge_index)
        attn_output = attn.forward(x, edge_index)
        pool_output, pooled_edge_index, pooled_batch = pool.forward(x, edge_index)
        
        assert conv_output.shape == (100, 5)
        assert attn_output.shape == (100, 5)
        assert pool_output.shape[1] == 5
        assert torch.isfinite(conv_output).all()
        assert torch.isfinite(attn_output).all()
        assert torch.isfinite(pool_output).all()
    
    def test_layers_with_different_activations(self):
        """Test layers with different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu", "leaky_relu"]
        
        for activation in activations:
            conv = FractionalGraphConv(5, 3, activation=activation)
            attn = FractionalGraphAttention(5, 3, num_heads=2, activation=activation)
            pool = FractionalGraphPooling(5, 3, pooling_ratio=0.5, activation=activation)
            
            assert conv.activation == activation
            assert attn.activation == activation
            assert pool.activation == activation
    
    def test_layers_without_bias(self):
        """Test layers without bias."""
        conv = FractionalGraphConv(5, 3, bias=False)
        attn = FractionalGraphAttention(5, 3, num_heads=2, bias=False)
        pool = FractionalGraphPooling(5, 3, pooling_ratio=0.5, bias=False)
        
        assert conv.bias is None
        assert attn.bias is None
        assert pool.bias is None


if __name__ == "__main__":
    pytest.main([__file__])
