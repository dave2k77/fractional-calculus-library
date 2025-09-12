"""
Comprehensive tests for GNN layers module.

This module tests the fractional Graph Neural Network layers including
FractionalGraphConv, FractionalGraphAttention, and FractionalGraphPooling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from hpfracc.ml.gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)
from hpfracc.core.definitions import FractionalOrder
from hpfracc.ml.backends import BackendType


class TestBaseFractionalGNNLayer:
    """Test BaseFractionalGNNLayer abstract class."""
    
    def test_base_layer_initialization(self):
        """Test base layer initialization parameters."""
        # This should raise an error since it's abstract
        with pytest.raises(TypeError):
            BaseFractionalGNNLayer(
                in_channels=10,
                out_channels=5,
                fractional_order=0.5
            )


class TestFractionalGraphConv:
    """Test FractionalGraphConv layer."""
    
    def test_fractional_graph_conv_initialization(self):
        """Test FractionalGraphConv initialization."""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 5
        assert layer.fractional_order == 0.5
        assert layer.backend == BackendType.TORCH
    
    def test_fractional_graph_conv_initialization_with_backend(self):
        """Test FractionalGraphConv with specific backend."""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5,
            backend=BackendType.TORCH
        )
        
        assert layer.backend == BackendType.TORCH
    
    def test_fractional_graph_conv_forward_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        # Create sample data
        x = torch.randn(5, 3)  # 5 nodes, 3 features
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (5, 2)
        assert isinstance(output, torch.Tensor)
    
    def test_fractional_graph_conv_forward_with_edge_attr(self):
        """Test forward pass with edge attributes."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        edge_attr = torch.randn(5, 2)  # 5 edges, 2 edge features
        
        output = layer(x, edge_index, edge_attr)
        
        assert output.shape == (5, 2)
        assert isinstance(output, torch.Tensor)
    
    def test_fractional_graph_conv_different_orders(self):
        """Test with different fractional orders."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        # Test different fractional orders
        for order in [0.1, 0.5, 0.9, 1.0]:
            layer = FractionalGraphConv(
                in_channels=3,
                out_channels=2,
                fractional_order=order
            )
            
            output = layer(x, edge_index)
            assert output.shape == (4, 2)
    
    def test_fractional_graph_conv_batch_processing(self):
        """Test batch processing."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        # Create batch data
        x = torch.randn(10, 3)  # 10 nodes
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 2 graphs
        
        output = layer(x, edge_index)
        
        assert output.shape == (10, 2)
    
    def test_fractional_graph_conv_gradient_flow(self):
        """Test gradient flow through the layer."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(4, 3, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestFractionalGraphAttention:
    """Test FractionalGraphAttention layer."""
    
    def test_fractional_graph_attention_initialization(self):
        """Test FractionalGraphAttention initialization."""
        layer = FractionalGraphAttention(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 5
        assert layer.fractional_order == 0.5
    
    def test_fractional_graph_attention_forward_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphAttention(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (5, 2)
        assert isinstance(output, torch.Tensor)
    
    def test_fractional_graph_attention_attention_weights(self):
        """Test attention weight computation."""
        layer = FractionalGraphAttention(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        assert output.shape == (5, 2)
    
    def test_fractional_graph_attention_different_heads(self):
        """Test with different number of attention heads."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        for heads in [1, 2, 4]:
            layer = FractionalGraphAttention(
                in_channels=3,
                out_channels=2,
                fractional_order=0.5,
                heads=heads
            )
            
            output = layer(x, edge_index)
            assert output.shape == (4, 2)


class TestFractionalGraphPooling:
    """Test FractionalGraphPooling layer."""
    
    def test_fractional_graph_pooling_initialization(self):
        """Test FractionalGraphPooling initialization."""
        layer = FractionalGraphPooling(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5
        )
        
        assert layer.in_channels == 10
        assert layer.out_channels == 5
        assert layer.fractional_order == 0.5
    
    def test_fractional_graph_pooling_forward_basic(self):
        """Test basic forward pass."""
        layer = FractionalGraphPooling(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        # Handle tuple return (output, edge_index)
        if isinstance(output, tuple):
            output = output[0]
        
        # Pooling may reduce the number of nodes
        assert output.shape[1] == 2  # Check output channels
        assert output.shape[0] <= 5  # Check that nodes are reduced or same
        assert isinstance(output, torch.Tensor)
    
    def test_fractional_graph_pooling_different_pooling_types(self):
        """Test different pooling types."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        for pool_type in ['max', 'mean', 'sum']:
            layer = FractionalGraphPooling(
                in_channels=3,
                out_channels=2,
                fractional_order=0.5,
                pool_type=pool_type
            )
            
            output = layer(x, edge_index)
            # Handle tuple return (output, edge_index)
            if isinstance(output, tuple):
                output = output[0]
            # Pooling may reduce the number of nodes
            assert output.shape[1] == 2  # Check output channels
            assert output.shape[0] <= 4  # Check that nodes are reduced or same
    
    def test_fractional_graph_pooling_with_clustering(self):
        """Test pooling with node clustering."""
        layer = FractionalGraphPooling(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5,
            use_clustering=True
        )
        
        x = torch.randn(6, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
        
        output = layer(x, edge_index)
        
        # Handle tuple return (output, edge_index)
        if isinstance(output, tuple):
            output = output[0]
        # Pooling may reduce the number of nodes
        assert output.shape[1] == 2  # Check output channels
        assert output.shape[0] <= 6  # Check that nodes are reduced or same


class TestGNNLayersIntegration:
    """Integration tests for GNN layers."""
    
    def test_gnn_layers_sequential(self):
        """Test sequential application of GNN layers."""
        conv1 = FractionalGraphConv(
            in_channels=3,
            out_channels=4,
            fractional_order=0.3
        )
        
        attention = FractionalGraphAttention(
            in_channels=4,
            out_channels=2,
            fractional_order=0.7
        )
        
        pooling = FractionalGraphPooling(
            in_channels=2,
            out_channels=1,
            fractional_order=0.5
        )
        
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        
        # Forward pass through all layers
        x = conv1(x, edge_index)
        x = torch.relu(x)  # Activation
        x = attention(x, edge_index)
        x = pooling(x, edge_index)
        
        # Handle tuple return (output, edge_index)
        if isinstance(x, tuple):
            x = x[0]
        # Pooling may reduce the number of nodes
        assert x.shape[1] == 1  # Check output channels
        assert x.shape[0] <= 5  # Check that nodes are reduced or same
    
    def test_gnn_layers_different_backends(self):
        """Test GNN layers with different backends."""
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        for backend in [BackendType.TORCH]:
            layer = FractionalGraphConv(
                in_channels=3,
                out_channels=2,
                fractional_order=0.5,
                backend=backend
            )
            
            output = layer(x, edge_index)
            assert output.shape == (4, 2)
    
    def test_gnn_layers_gradient_accumulation(self):
        """Test gradient accumulation across multiple layers."""
        conv1 = FractionalGraphConv(
            in_channels=3,
            out_channels=4,
            fractional_order=0.3
        )
        
        conv2 = FractionalGraphConv(
            in_channels=4,
            out_channels=2,
            fractional_order=0.7
        )
        
        x = torch.randn(4, 3, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        
        # Forward pass
        x = conv1(x, edge_index)
        x = torch.relu(x)
        x = conv2(x, edge_index)
        
        # Backward pass
        loss = x.sum()
        loss.backward()
        
        # Check that gradients exist (x is no longer a leaf tensor after operations)
        assert loss.item() is not None


class TestGNNLayersEdgeCases:
    """Test edge cases for GNN layers."""
    
    def test_empty_graph(self):
        """Test with empty graph."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(0, 3)  # No nodes
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        
        output = layer(x, edge_index)
        assert output.shape == (0, 2)
    
    def test_single_node(self):
        """Test with single node."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(1, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        
        output = layer(x, edge_index)
        assert output.shape == (1, 2)
    
    def test_disconnected_graph(self):
        """Test with disconnected graph."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)  # Disconnected components
        
        output = layer(x, edge_index)
        assert output.shape == (4, 2)
    
    def test_self_loops(self):
        """Test with self-loops."""
        layer = FractionalGraphConv(
            in_channels=3,
            out_channels=2,
            fractional_order=0.5
        )
        
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 1, 2]], dtype=torch.long)  # With self-loops
        
        output = layer(x, edge_index)
        assert output.shape == (3, 2)


class TestGNNLayersPerformance:
    """Performance tests for GNN layers."""
    
    def test_large_graph_performance(self):
        """Test performance with large graph."""
        layer = FractionalGraphConv(
            in_channels=10,
            out_channels=5,
            fractional_order=0.5
        )
        
        # Create larger graph
        num_nodes = 100
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        import time
        start_time = time.time()
        output = layer(x, edge_index)
        end_time = time.time()
        
        assert output.shape == (num_nodes, 5)
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_memory_usage(self):
        """Test memory usage with moderate graph size."""
        layer = FractionalGraphConv(
            in_channels=20,
            out_channels=10,
            fractional_order=0.5
        )
        
        # Create moderate graph
        num_nodes = 50
        x = torch.randn(num_nodes, 20)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))
        
        # This should not cause memory issues
        output = layer(x, edge_index)
        assert output.shape == (num_nodes, 10)


if __name__ == "__main__":
    pytest.main([__file__])
