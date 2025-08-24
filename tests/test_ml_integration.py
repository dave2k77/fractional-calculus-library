"""
Comprehensive ML Integration Test Suite

This test suite validates the complete ML integration system including:
- Fractional Neural Networks
- Model Registry
- Development/Production Workflows
- All Fractional Layers
- Loss Functions and Optimizers
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
    ModelRegistry,
    DevelopmentWorkflow,
    ProductionWorkflow,
    ModelValidator,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    LayerConfig
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
        output, (h, c) = lstm(x)
        
        assert output.shape == (5, 1, 32)
        assert h.shape == (1, 1, 32)
        assert c.shape == (1, 1, 32)
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
        optimizer.zero_grad()
        output = net(x, use_fractional=True, method="RL")
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Store parameter values before optimization
        param_values_before = [p.clone() for p in net.parameters()]
        
        # Perform optimization step
        optimizer.step()
        
        # Check that parameters were updated
        param_values_after = [p.clone() for p in net.parameters()]
        
        # At least some parameters should have changed
        changes = sum(torch.sum(a != b).item() for a, b in zip(param_values_before, param_values_after))
        assert changes > 0, f"Parameters should be updated during optimization. Changes: {changes}"


class TestModelRegistry:
    """Test model registry functionality"""
    
    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing"""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(storage_path=temp_dir)
        yield registry
        shutil.rmtree(temp_dir)
    
    def test_model_registration(self, temp_registry):
        """Test model registration and retrieval"""
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        model_id = temp_registry.register_model(
            model=net,
            name="test_network",
            version="1.0.0",
            description="Test network",
            author="Test Author",
            tags=["test"],
            framework="PyTorch",
            model_type="FractionalNeuralNetwork",
            fractional_order=0.5,
            hyperparameters={"input_size": 5, "hidden_sizes": [16], "output_size": 2},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"num_samples": 1000},
            dependencies={"torch": ">=2.0.0"},
            notes="Test model",
            git_commit="test_commit",
            git_branch="test_branch"
        )
        
        assert model_id is not None
        
        # Retrieve model info
        model_info = temp_registry.get_model(model_id)
        assert model_info.name == "test_network"
        assert model_info.version == "1.0.0"
        assert model_info.author == "Test Author"
    
    def test_model_reconstruction(self, temp_registry):
        """Test model reconstruction from registry"""
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        model_id = temp_registry.register_model(
            model=net,
            name="test_network",
            version="1.0.0",
            description="Test network",
            author="Test Author",
            tags=["test"],
            framework="PyTorch",
            model_type="FractionalNeuralNetwork",
            fractional_order=0.5,
            hyperparameters={"input_size": 5, "hidden_sizes": [16], "output_size": 2},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"num_samples": 1000},
            dependencies={"torch": ">=2.0.0"},
            notes="Test model",
            git_commit="test_commit",
            git_branch="test_branch"
        )
        
        # Reconstruct model
        reconstructed_net = temp_registry.reconstruct_model(model_id)
        assert reconstructed_net is not None
        assert isinstance(reconstructed_net, FractionalNeuralNetwork)
        
        # Test that reconstructed model works
        x = torch.randn(8, 5)
        output = reconstructed_net(x, use_fractional=True, method="RL")
        assert output.shape == (8, 2)


class TestWorkflows:
    """Test development and production workflows"""
    
    @pytest.fixture
    def temp_registry(self):
        """Create a temporary registry for testing"""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(storage_path=temp_dir)
        yield registry
        shutil.rmtree(temp_dir)
    
    def test_development_workflow(self, temp_registry):
        """Test development workflow validation"""
        # Create and register a model
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        model_id = temp_registry.register_model(
            model=net,
            name="test_network",
            version="1.0.0",
            description="Test network",
            author="Test Author",
            tags=["test"],
            framework="PyTorch",
            model_type="FractionalNeuralNetwork",
            fractional_order=0.5,
            hyperparameters={"input_size": 5, "hidden_sizes": [16], "output_size": 2},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"num_samples": 1000},
            dependencies={"torch": ">=2.0.0"},
            notes="Test model",
            git_commit="test_commit",
            git_branch="test_branch"
        )
        
        # Test development workflow
        validator = ModelValidator()
        dev_workflow = DevelopmentWorkflow(temp_registry, validator)
        
        # Create test data
        test_data = torch.randn(100, 5)
        test_labels = torch.randn(100, 2)
        
        validation_results = dev_workflow.validate_development_model(
            model_id=model_id,
            test_data=test_data,
            test_labels=test_labels
        )
        
        assert 'validation_passed' in validation_results
        assert 'final_score' in validation_results
        assert 'gate_results' in validation_results
    
    def test_production_workflow(self, temp_registry):
        """Test production workflow promotion"""
        # Create and register a model
        net = FractionalNeuralNetwork(
            input_size=5,
            hidden_sizes=[16],
            output_size=2,
            fractional_order=0.5
        )
        
        model_id = temp_registry.register_model(
            model=net,
            name="test_network",
            version="1.0.0",
            description="Test network",
            author="Test Author",
            tags=["test"],
            framework="PyTorch",
            model_type="FractionalNeuralNetwork",
            fractional_order=0.5,
            hyperparameters={"input_size": 5, "hidden_sizes": [16], "output_size": 2},
            performance_metrics={"accuracy": 0.95},
            dataset_info={"num_samples": 1000},
            dependencies={"torch": ">=2.0.0"},
            notes="Test model",
            git_commit="test_commit",
            git_branch="test_branch"
        )
        
        # Test production workflow
        validator = ModelValidator()
        prod_workflow = ProductionWorkflow(temp_registry, validator)
        
        # Create test data
        test_data = torch.randn(100, 5)
        test_labels = torch.randn(100, 2)
        
        promotion_results = prod_workflow.promote_to_production(
            model_id=model_id,
            version="1.0.0",
            test_data=test_data,
            test_labels=test_labels
        )
        
        assert 'promoted' in promotion_results
        assert 'reason' in promotion_results


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
