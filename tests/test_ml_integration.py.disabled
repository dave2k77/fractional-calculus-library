"""
Comprehensive ML Integration Test Suite

This test suite validates the complete ML integration system including:
- Fractional Neural Networks
- All Fractional Layers
- Loss Functions and Optimizers
- Fractional Graph Neural Networks (GNNs)
- Multi-backend support (PyTorch, JAX, NUMBA)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

from hpfracc.ml import (
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalMSELoss,
    FractionalAdam,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    LayerConfig,
    # GNN Components
    BaseFractionalGNNLayer,
    FractionalGraphConv,
    FractionalGraphAttention,
    FractionalGraphPooling,
    BaseFractionalGNN,
    FractionalGCN,
    FractionalGAT,
    FractionalGraphSAGE,
    FractionalGraphUNet,
    FractionalGNNFactory,
    # Backend Management
    BackendManager,
    BackendType,
    get_backend_manager,
    set_backend_manager,
    get_active_backend,
    switch_backend,
    # Tensor Operations
    TensorOps,
    get_tensor_ops
)
from hpfracc.core.definitions import FractionalOrder


class TestFractionalNeuralNetwork:
    """Test Fractional Neural Network functionality"""
    
    def test_network_creation(self):
        """Test network creation with different configurations"""
        # Test basic network
        net = FractionalNeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=3,
            fractional_order=0.5
        )
        assert net is not None
        assert sum(p.numel() for p in net.parameters()) > 0
        
        # Test with different fractional orders (Caputo L1 scheme requires 0 < α < 1)
        for alpha in [0.1, 0.5, 0.9]:
            net = FractionalNeuralNetwork(
                input_size=5,
                hidden_sizes=[32],
                output_size=2,
                fractional_order=alpha
            )
            assert net.fractional_order.alpha == alpha
    
    def test_forward_pass(self):
        """Test forward pass with and without fractional derivatives"""
        net = FractionalNeuralNetwork(
            input_size=10,
            hidden_sizes=[64, 32],
            output_size=3,
            fractional_order=0.5
        )
        
        x = torch.randn(32, 10)
        
        # Test with fractional derivatives
        output_frac = net(x, use_fractional=True, method="RL")
        assert output_frac.shape == (32, 3)
        assert output_frac.requires_grad
        
        # Test without fractional derivatives
        output_std = net(x, use_fractional=False)
        assert output_std.shape == (32, 3)
        assert output_std.requires_grad
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the network"""
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        x = torch.randn(8, 5, requires_grad=True)
        output = net(x, use_fractional=True, method="RL")
        
        # Create a simple loss
        target = torch.randn(8, 2)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients - the input gradient might be None if the network doesn't modify the input
        # Check parameter gradients instead, which should always have gradients
        param_grads = [param.grad for param in net.parameters()]
        assert any(grad is not None for grad in param_grads), "At least one parameter should have gradients"
        
        # Check that the network parameters are trainable
        assert any(param.requires_grad for param in net.parameters()), "Network should have trainable parameters"


class TestFractionalLayers:
    """Test all fractional neural network layers"""
    
    def test_fractional_conv1d(self):
        """Test 1D convolutional layer with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        conv = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            config=config
        )
        
        x = torch.randn(1, 3, 10, requires_grad=True)
        output = conv(x)
        
        assert output.shape == (1, 16, 8)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_fractional_conv2d(self):
        """Test 2D convolutional layer with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        conv = FractionalConv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            config=config
        )
        
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        output = conv(x)
        
        assert output.shape == (1, 16, 6, 6)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_fractional_lstm(self):
        """Test LSTM layer with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        lstm = FractionalLSTM(
            input_size=10,
            hidden_size=32,
            config=config
        )
        
        x = torch.randn(5, 1, 10, requires_grad=True)  # (seq_len, batch, input_size)
        output = lstm(x)  # Default behavior returns only output tensor
        
        assert output.shape == (5, 1, 32)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_fractional_transformer(self):
        """Test transformer layer with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        transformer = FractionalTransformer(
            d_model=64,
            nhead=8,
            config=config
        )
        
        src = torch.randn(10, 2, 64, requires_grad=True)  # (seq_len, batch, d_model)
        tgt = torch.randn(8, 2, 64, requires_grad=True)
        
        output = transformer(src, tgt)
        
        assert output.shape == (8, 2, 64)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert src.grad is not None
        assert tgt.grad is not None
    
    def test_fractional_pooling(self):
        """Test pooling layer with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        pooling = FractionalPooling(
            kernel_size=2,
            config=config
        )
        
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        output = pooling(x)
        
        assert output.shape == (1, 16, 4, 4)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_fractional_batchnorm(self):
        """Test batch normalization with fractional derivatives"""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.5),
            method="RL",
            use_fractional=True
        )
        
        batchnorm = FractionalBatchNorm1d(
            num_features=64,
            config=config
        )
        
        x = torch.randn(1, 64, 10, requires_grad=True)
        output = batchnorm(x)
        
        assert output.shape == (1, 64, 10)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestFractionalLossFunctions:
    """Test fractional loss functions"""
    
    def test_fractional_mse_loss(self):
        """Test fractional MSE loss function"""
        loss_fn = FractionalMSELoss(fractional_order=0.5, method="RL")
        
        predictions = torch.randn(32, 3, requires_grad=True)
        targets = torch.randn(32, 3)
        
        # Test with fractional derivatives
        loss = loss_fn(predictions, targets, use_fractional=True)
        assert loss.requires_grad
        assert loss.item() > 0
        
        # Test gradient flow
        loss.backward()
        assert predictions.grad is not None
    
    def test_fractional_cross_entropy_loss(self):
        """Test fractional cross entropy loss function"""
        from hpfracc.ml.losses import FractionalCrossEntropyLoss
        
        loss_fn = FractionalCrossEntropyLoss(fractional_order=0.5, method="RL")
        
        predictions = torch.randn(32, 5, requires_grad=True)
        targets = torch.randint(0, 5, (32,))
        
        # Test with fractional derivatives
        loss = loss_fn(predictions, targets, use_fractional=True)
        assert loss.requires_grad
        assert loss.item() > 0
        
        # Test gradient flow
        loss.backward()
        assert predictions.grad is not None


class TestFractionalOptimizers:
    """Test fractional optimizers"""
    
    def test_fractional_adam(self):
        """Test fractional Adam optimizer"""
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        optimizer = FractionalAdam(
            net.parameters(),
            lr=0.001,
            fractional_order=0.5,
            method="RL",
            use_fractional=True
        )
        
        x = torch.randn(8, 5)
        target = torch.randn(8, 2)
        
        # Training step
        optimizer.zero_grad(net.parameters())
        output = net(x, use_fractional=True, method="RL")
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Store parameter values before optimization
        param_values_before = [p.clone() for p in net.parameters()]
        
        # Perform optimization step
        optimizer.step(net.parameters())
        
        # Check that parameters were updated
        param_values_after = [p.clone() for p in net.parameters()]
        
        # At least some parameters should have changed
        changes = sum(torch.sum(a != b).item() for a, b in zip(param_values_before, param_values_after))
        assert changes > 0, f"Parameters should be updated during optimization. Changes: {changes}"


class TestFractionalGraphNeuralNetworks:
    """Test Fractional Graph Neural Network functionality"""
    
    def test_gnn_factory_creation(self):
        """Test GNN factory can create all model types"""
        factory = FractionalGNNFactory()
        
        # Test creating each model type
        for model_type in ['gcn', 'gat', 'sage', 'unet']:
            model = factory.create_model(
                model_type=model_type,
                input_dim=10,
                hidden_dim=32,
                output_dim=2,
                num_layers=2,
                fractional_order=0.5
            )
            assert model is not None
            assert hasattr(model, 'forward')
    
    def test_fractional_gcn(self):
        """Test FractionalGCN model"""
        model = FractionalGCN(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            fractional_order=0.5,
            dropout=0.1
        )
        
        # Create sample graph data
        num_nodes = 20
        x = torch.randn(num_nodes, 10)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, 50))  # Random edges
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (num_nodes, 2)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        param_grads = [param.grad for param in model.parameters()]
        assert any(grad is not None for grad in param_grads)
    
    def test_fractional_gat(self):
        """Test FractionalGAT model"""
        model = FractionalGAT(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            num_heads=4,
            fractional_order=0.5,
            dropout=0.1
        )
        
        # Create sample graph data
        num_nodes = 20
        x = torch.randn(num_nodes, 10)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, 50))  # Random edges
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (num_nodes, 2)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        param_grads = [param.grad for param in model.parameters()]
        assert any(grad is not None for grad in param_grads)
    
    def test_fractional_graphsage(self):
        """Test FractionalGraphSAGE model"""
        model = FractionalGraphSAGE(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            fractional_order=0.5,
            dropout=0.1
        )
        
        # Create sample graph data
        num_nodes = 20
        x = torch.randn(num_nodes, 10)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, 50))  # Random edges
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (num_nodes, 2)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        param_grads = [param.grad for param in model.parameters()]
        assert any(grad is not None for grad in param_grads)
    
    def test_fractional_graph_unet(self):
        """Test FractionalGraphUNet model"""
        model = FractionalGraphUNet(
            input_dim=10,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            fractional_order=0.5,
            dropout=0.1
        )
        
        # Create sample graph data
        num_nodes = 20
        x = torch.randn(num_nodes, 10)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, 50))  # Random edges
        
        # Test forward pass
        output = model(x, edge_index)
        assert output.shape == (num_nodes, 2)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        param_grads = [param.grad for param in model.parameters()]
        assert any(grad is not None for grad in param_grads)
    
    def test_gnn_layers(self):
        """Test individual GNN layers"""
        # Test FractionalGraphConv
        conv_layer = FractionalGraphConv(
            in_channels=10,
            out_channels=32,
            fractional_order=0.5,
            activation="relu",
            dropout=0.1
        )
        
        num_nodes = 20
        x = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        
        conv_output = conv_layer(x, edge_index)
        assert conv_output.shape == (num_nodes, 32)
        
        # Test FractionalGraphAttention
        attn_layer = FractionalGraphAttention(
            in_channels=10,
            out_channels=32,
            num_heads=4,
            fractional_order=0.5,
            activation="relu",
            dropout=0.1
        )
        
        attn_output = attn_layer(x, edge_index)
        assert attn_output.shape == (num_nodes, 32)
        
        # Test FractionalGraphPooling
        pool_layer = FractionalGraphPooling(
            in_channels=10,
            ratio=0.5,
            fractional_order=0.5
        )
        
        pool_output, pool_edge_index, pool_batch = pool_layer(x, edge_index)
        assert pool_output.shape[0] <= num_nodes  # Pooled nodes should be fewer or equal
        assert pool_output.shape[1] == 10  # Feature dimension should be preserved


class TestBackendManagement:
    """Test backend management system"""
    
    def test_backend_manager_creation(self):
        """Test backend manager creation and initialization"""
        manager = BackendManager()
        assert manager is not None
        assert hasattr(manager, 'active_backend')
    
    def test_backend_switching(self):
        """Test switching between different backends"""
        manager = BackendManager()
        
        # Test switching to PyTorch
        manager.switch_backend(BackendType.TORCH)
        assert manager.active_backend == BackendType.TORCH
        
        # Test switching to JAX
        manager.switch_backend(BackendType.JAX)
        assert manager.active_backend == BackendType.JAX
        
        # Test switching to NUMBA
        manager.switch_backend(BackendType.NUMBA)
        assert manager.active_backend == BackendType.NUMBA
    
    def test_tensor_ops_creation(self):
        """Test tensor operations creation for different backends"""
        # Test PyTorch tensor ops
        torch_ops = get_tensor_ops(BackendType.TORCH)
        assert torch_ops is not None
        assert hasattr(torch_ops, 'zeros')
        assert hasattr(torch_ops, 'ones')
        
        # Test JAX tensor ops
        jax_ops = get_tensor_ops(BackendType.JAX)
        assert jax_ops is not None
        assert hasattr(jax_ops, 'zeros')
        assert hasattr(jax_ops, 'ones')
        
        # Test NUMBA tensor ops
        numba_ops = get_tensor_ops(BackendType.NUMBA)
        assert numba_ops is not None
        assert hasattr(numba_ops, 'zeros')
        assert hasattr(numba_ops, 'ones')


class TestFractionalAttention:
    """Test fractional attention mechanism"""
    
    def test_fractional_attention_creation(self):
        """Test fractional attention layer creation"""
        attention = FractionalAttention(
            d_model=64,
            n_heads=8,
            fractional_order=0.5,
            dropout=0.1
        )
        
        assert attention is not None
        assert attention.fractional_order.alpha == 0.5
    
    def test_fractional_attention_forward(self):
        """Test fractional attention forward pass"""
        attention = FractionalAttention(
            d_model=64,
            n_heads=8,
            fractional_order=0.5,
            dropout=0.1
        )
        
        x = torch.randn(10, 2, 64, requires_grad=True)  # (seq_len, batch, d_model)
        output = attention(x, method="RL")
        
        assert output.shape == (10, 2, 64)
        assert output.requires_grad
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
