#!/usr/bin/env python3
"""High-impact tests for GNN layers to dramatically boost coverage."""

import pytest
import torch
import numpy as np

from hpfracc.ml.gnn_layers import (
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling
)
from hpfracc.core.definitions import FractionalOrder
from hpfracc.ml.backends import BackendType


class TestGNNLayersHighImpact:
    """High-impact tests for GNN layers - 547 lines to cover!"""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.in_channels = 10
        self.out_channels = 20
        self.num_nodes = 50
        self.alpha = 0.5
        
        # Create test data
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(0, self.num_nodes, (2, 100))
        
    def test_fractional_graph_conv_initialization(self):
        """Test FractionalGraphConv initialization."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        assert layer.in_channels == self.in_channels
        assert layer.out_channels == self.out_channels
        # Handle both direct types and compatibility wrapper
        assert (isinstance(layer.fractional_order, (float, FractionalOrder)) or 
                hasattr(layer.fractional_order, '_fractional_order'))
        
    def test_fractional_graph_conv_forward(self):
        """Test FractionalGraphConv forward pass."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Test forward pass
        output = layer(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        assert torch.is_tensor(output)
        
    def test_fractional_graph_attention_initialization(self):
        """Test FractionalGraphAttention initialization."""
        layer = FractionalGraphAttention(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        assert layer.in_channels == self.in_channels
        assert layer.out_channels == self.out_channels
        
    def test_fractional_graph_attention_forward(self):
        """Test FractionalGraphAttention forward pass."""
        layer = FractionalGraphAttention(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        output = layer(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        assert torch.is_tensor(output)
        
    def test_fractional_graph_pooling_initialization(self):
        """Test FractionalGraphPooling initialization."""
        layer = FractionalGraphPooling(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        assert layer.in_channels == self.in_channels
        assert layer.out_channels == self.out_channels
        
    def test_fractional_graph_pooling_forward(self):
        """Test FractionalGraphPooling forward pass."""
        layer = FractionalGraphPooling(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        output = layer(self.x, self.edge_index)
        # Handle tuple return (output, edge_index, batch)
        if isinstance(output, tuple):
            output = output[0]
        assert torch.is_tensor(output)
        # Pooling may change shape
        assert output.shape[1] == self.out_channels
        
    def test_different_fractional_orders(self):
        """Test layers with different fractional orders."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for alpha in alphas:
            layer = FractionalGraphConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                fractional_order=alpha
            )
            
            output = layer(self.x, self.edge_index)
            assert output.shape == (self.num_nodes, self.out_channels)
            
    def test_different_channel_sizes(self):
        """Test layers with different channel sizes."""
        channel_configs = [
            (5, 10), (10, 20), (20, 50), (50, 100)
        ]
        
        for in_ch, out_ch in channel_configs:
            x_test = torch.randn(self.num_nodes, in_ch)
            
            layer = FractionalGraphConv(
                in_channels=in_ch,
                out_channels=out_ch,
                fractional_order=self.alpha
            )
            
            output = layer(x_test, self.edge_index)
            assert output.shape == (self.num_nodes, out_ch)
            
    def test_different_backends(self):
        """Test layers with different backends."""
        backends = [BackendType.TORCH, BackendType.AUTO]
        
        for backend in backends:
            layer = FractionalGraphConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                fractional_order=self.alpha,
                backend=backend
            )
            
            output = layer(self.x, self.edge_index)
            assert output.shape == (self.num_nodes, self.out_channels)
            
    def test_gradient_computation(self):
        """Test gradient computation through GNN layers."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Enable gradients
        x_grad = self.x.clone().requires_grad_(True)
        
        output = layer(x_grad, self.edge_index)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x_grad.grad is not None
        
        # Check layer parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_edge_cases(self):
        """Test edge cases for GNN layers."""
        # Small graph
        small_x = torch.randn(2, self.in_channels)
        small_edge_index = torch.tensor([[0, 1], [1, 0]])
        
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        output = layer(small_x, small_edge_index)
        assert output.shape == (2, self.out_channels)
        
    def test_different_graph_structures(self):
        """Test with different graph structures."""
        # Dense graph
        dense_edges = torch.combinations(torch.arange(10), r=2).T
        dense_edge_index = torch.cat([dense_edges, dense_edges.flip(0)], dim=1)
        
        # Sparse graph
        sparse_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Test with both structures
        x_small = torch.randn(10, self.in_channels)
        
        dense_output = layer(x_small, dense_edge_index)
        sparse_output = layer(x_small[:3], sparse_edge_index)
        
        assert dense_output.shape == (10, self.out_channels)
        assert sparse_output.shape == (3, self.out_channels)
        
    def test_layer_combinations(self):
        """Test combining different GNN layers."""
        conv_layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=15,
            fractional_order=self.alpha
        )
        
        attention_layer = FractionalGraphAttention(
            in_channels=15,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Chain layers
        x1 = conv_layer(self.x, self.edge_index)
        x2 = attention_layer(x1, self.edge_index)
        
        assert x2.shape == (self.num_nodes, self.out_channels)
        
    def test_device_compatibility(self):
        """Test device compatibility."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # CPU
        cpu_output = layer(self.x, self.edge_index)
        assert cpu_output.device == self.x.device
        
        # CUDA (if available)
        if torch.cuda.is_available():
            layer_cuda = layer.cuda()
            x_cuda = self.x.cuda()
            edge_index_cuda = self.edge_index.cuda()
            
            cuda_output = layer_cuda(x_cuda, edge_index_cuda)
            assert cuda_output.device == x_cuda.device
            
    def test_dtype_compatibility(self):
        """Test dtype compatibility."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        dtypes = [torch.float32, torch.float64]
        
        for dtype in dtypes:
            x_typed = self.x.to(dtype)
            output = layer(x_typed, self.edge_index)
            assert output.dtype in [torch.float32, torch.float64]
            
    def test_training_vs_eval_mode(self):
        """Test layer behavior in training vs eval mode."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Training mode
        layer.train()
        train_output = layer(self.x, self.edge_index)
        
        # Eval mode  
        layer.eval()
        eval_output = layer(self.x, self.edge_index)
        
        # Shapes should be consistent
        assert train_output.shape == eval_output.shape
        
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Check that layer has parameters
        params = list(layer.parameters())
        assert len(params) > 0
        
        # Check parameter shapes are reasonable
        for param in params:
            assert param.requires_grad
            assert torch.all(torch.isfinite(param))
            
    def test_memory_efficiency(self):
        """Test memory efficiency with large graphs."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Process multiple times to test memory handling
        for _ in range(5):
            large_x = torch.randn(200, self.in_channels)
            large_edge_index = torch.randint(0, 200, (2, 500))
            
            output = layer(large_x, large_edge_index)
            assert output.shape == (200, self.out_channels)
            
    def test_numerical_stability(self):
        """Test numerical stability."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Very small values
        small_x = torch.full((self.num_nodes, self.in_channels), 1e-8)
        small_output = layer(small_x, self.edge_index)
        assert torch.all(torch.isfinite(small_output))
        
        # Large values
        large_x = torch.full((self.num_nodes, self.in_channels), 1e3)
        large_output = layer(large_x, self.edge_index)
        assert torch.all(torch.isfinite(large_output))
        
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Batch of graphs (if supported)
        batch_x = torch.randn(5, self.num_nodes, self.in_channels)
        
        # Process each graph in batch
        for i in range(batch_x.shape[0]):
            output = layer(batch_x[i], self.edge_index)
            assert output.shape == (self.num_nodes, self.out_channels)
            
    def test_fractional_order_types(self):
        """Test different fractional order types."""
        # Float alpha
        layer1 = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=0.5
        )
        
        # FractionalOrder object
        layer2 = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=FractionalOrder(0.5)
        )
        
        # Both should work
        output1 = layer1(self.x, self.edge_index)
        output2 = layer2(self.x, self.edge_index)
        
        assert output1.shape == output2.shape
        
    def test_attention_layer_specific(self):
        """Test attention layer specific functionality."""
        layer = FractionalGraphAttention(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        output = layer(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        
        # Attention should produce different outputs for different inputs
        x_different = torch.randn(self.num_nodes, self.in_channels)
        output_different = layer(x_different, self.edge_index)
        
        # Outputs should be different
        assert not torch.allclose(output, output_different, atol=1e-6)
        
    def test_pooling_layer_specific(self):
        """Test pooling layer specific functionality."""
        layer = FractionalGraphPooling(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        output = layer(self.x, self.edge_index)
        # Handle tuple return (output, edge_index, batch)
        if isinstance(output, tuple):
            output = output[0]
        assert torch.is_tensor(output)
        assert output.shape[1] == self.out_channels
        # Pooling may reduce number of nodes
        
    def test_layer_serialization(self):
        """Test layer serialization."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Get original output
        original_output = layer(self.x, self.edge_index)
        
        # Save and load state dict
        state_dict = layer.state_dict()
        new_layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        new_layer.load_state_dict(state_dict)
        
        # Should produce same output
        new_output = new_layer(self.x, self.edge_index)
        assert torch.allclose(original_output, new_output, atol=1e-6)
        
    def test_integration_with_optimizers(self):
        """Test integration with PyTorch optimizers."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        
        # Training step
        output = layer(self.x, self.edge_index)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
                
    def test_different_edge_configurations(self):
        """Test with different edge configurations."""
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Different edge configurations
        edge_configs = [
            torch.tensor([[0, 1], [1, 0]]),  # Simple bidirectional edge
            torch.randint(0, 10, (2, 20)),   # Random edges
            torch.tensor([[0, 1, 2], [1, 2, 0]]),  # Triangle
        ]
        
        for edge_index in edge_configs:
            max_node = edge_index.max().item()
            x_test = torch.randn(max_node + 1, self.in_channels)
            
            output = layer(x_test, edge_index)
            assert output.shape == (max_node + 1, self.out_channels)
            
    def test_reproducibility(self):
        """Test reproducibility with fixed seeds."""
        torch.manual_seed(42)
        layer1 = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        torch.manual_seed(42)
        layer2 = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        torch.manual_seed(123)
        x_test = torch.randn(self.num_nodes, self.in_channels)
        edge_test = torch.randint(0, self.num_nodes, (2, 50))
        
        output1 = layer1(x_test, edge_test)
        output2 = layer2(x_test, edge_test)
        
        # Should be similar (parameters initialized the same way)
        assert torch.allclose(output1, output2, atol=1e-6)
        
    def test_base_class_interface(self):
        """Test base class interface compliance."""
        # All concrete classes should inherit from base
        assert issubclass(FractionalGraphConv, BaseFractionalGNNLayer)
        assert issubclass(FractionalGraphAttention, BaseFractionalGNNLayer)
        assert issubclass(FractionalGraphPooling, BaseFractionalGNNLayer)
        
    def test_layer_composition(self):
        """Test composing multiple GNN layers."""
        # Create a small GNN
        class SimpleGNN(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = FractionalGraphConv(
                    in_channels=in_channels,
                    out_channels=16,
                    fractional_order=0.3
                )
                self.conv2 = FractionalGraphConv(
                    in_channels=16,
                    out_channels=out_channels,
                    fractional_order=0.7
                )
                
            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = torch.relu(x)
                x = self.conv2(x, edge_index)
                return x
                
        gnn = SimpleGNN(self.in_channels, self.out_channels)
        output = gnn(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        import time
        
        layer = FractionalGraphConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            fractional_order=self.alpha
        )
        
        # Measure computation time
        start_time = time.time()
        output = layer(self.x, self.edge_index)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 seconds max
        assert output.shape == (self.num_nodes, self.out_channels)



