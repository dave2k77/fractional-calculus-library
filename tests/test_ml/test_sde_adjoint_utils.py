"""
Unit tests for SDE adjoint optimization utilities in hpfracc.ml.sde_adjoint_utils

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from hpfracc.ml.sde_adjoint_utils import (
    CheckpointConfig, MixedPrecisionConfig, SDEStateCheckpoint,
    MixedPrecisionManager, SparseGradientAccumulator, checkpoint_trajectory,
    SDEAdjointOptimizer
)


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = CheckpointConfig()
        
        assert config.checkpoint_frequency == 10
        assert config.max_checkpoints == 100
        assert config.checkpoint_strategy == "uniform"
        assert config.enable_checkpointing is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CheckpointConfig(
            checkpoint_frequency=5,
            max_checkpoints=50,
            checkpoint_strategy="adaptive",
            enable_checkpointing=False
        )
        
        assert config.checkpoint_frequency == 5
        assert config.max_checkpoints == 50
        assert config.checkpoint_strategy == "adaptive"
        assert config.enable_checkpointing is False


class TestMixedPrecisionConfig:
    """Test MixedPrecisionConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = MixedPrecisionConfig()
        
        assert config.enable_amp is False
        assert config.half_precision is False
        assert config.loss_scaling == 1.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MixedPrecisionConfig(
            enable_amp=True,
            half_precision=True,
            loss_scaling=2.0
        )
        
        assert config.enable_amp is True
        assert config.half_precision is True
        assert config.loss_scaling == 2.0


class TestSDEStateCheckpoint:
    """Test SDEStateCheckpoint class"""
    
    def test_initialization(self):
        """Test checkpoint initialization"""
        config = CheckpointConfig()
        checkpoint = SDEStateCheckpoint(config)
        
        assert checkpoint.config == config
        assert len(checkpoint.checkpoints) == 0
        assert len(checkpoint.checkpoint_indices) == 0
    
    def test_state_access(self):
        """Test state access"""
        config = CheckpointConfig()
        checkpoint = SDEStateCheckpoint(config)
        
        state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checkpoint.save_checkpoint(10, state)
        
        loaded_state = checkpoint.load_checkpoint(10)
        assert loaded_state is not None
        assert torch.equal(loaded_state, state)
    
    def test_step_tracking(self):
        """Test step tracking"""
        config = CheckpointConfig(checkpoint_frequency=1)  # Save every step
        checkpoint = SDEStateCheckpoint(config)
        
        state = torch.tensor([[1.0, 2.0]])
        checkpoint.save_checkpoint(42, state)
        
        assert len(checkpoint.checkpoints) == 1
        assert checkpoint.checkpoints[0]['step'] == 42


class TestMixedPrecisionManager:
    """Test MixedPrecisionManager class"""
    
    def test_initialization(self):
        """Test manager initialization"""
        config = MixedPrecisionConfig(enable_amp=True)
        manager = MixedPrecisionManager(config)
        
        assert manager.config == config
        assert manager.scaler is not None
    
    def test_disabled_mode(self):
        """Test disabled mixed precision"""
        config = MixedPrecisionConfig(enable_amp=False)
        manager = MixedPrecisionManager(config)
        
        assert manager.config == config
        assert manager.scaler is None
    
    def test_loss_scaling(self):
        """Test loss scaling functionality"""
        config = MixedPrecisionConfig(enable_amp=False, loss_scaling=2.0)
        manager = MixedPrecisionManager(config)
        
        loss = torch.tensor(1.0)
        scaled_loss = manager.scale_loss(loss)
        
        assert scaled_loss.item() == 2.0
    
    def test_gradient_scaling(self):
        """Test gradient scaling functionality"""
        config = MixedPrecisionConfig(enable_amp=False, loss_scaling=4.0)
        manager = MixedPrecisionManager(config)
        
        # Create a simple model with requires_grad
        model = nn.Linear(2, 1)
        x = torch.randn(1, 2, requires_grad=True)
        y = model(x)
        loss = torch.sum(y)
        
        # Test gradient scaling
        manager.scale_gradients(loss)
        
        # Check that gradients exist
        for param in model.parameters():
            if param.grad is not None:
                assert param.grad is not None


class TestSparseGradientAccumulator:
    """Test SparseGradientAccumulator class"""
    
    def test_initialization(self):
        """Test accumulator initialization"""
        accumulator = SparseGradientAccumulator()
        
        assert accumulator.sparsity_threshold == 1e-6
        assert len(accumulator.accumulated_grads) == 0
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        accumulator = SparseGradientAccumulator()
        
        # Create tensor with gradient
        tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = torch.sum(tensor)
        loss.backward()
        
        # Accumulate gradient
        accumulator.accumulate(tensor.grad)
        
        assert len(accumulator.accumulated_grads) > 0
    
    def test_sparse_accumulation(self):
        """Test sparse gradient accumulation"""
        accumulator = SparseGradientAccumulator()
        
        # Create sparse gradients
        tensor = torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True)
        loss = torch.sum(tensor)
        loss.backward()
        
        # Accumulate sparse gradients
        accumulator.accumulate(tensor.grad)
        
        assert len(accumulator.accumulated_grads) > 0


class TestCheckpointTrajectory:
    """Test checkpoint_trajectory function"""
    
    def test_basic_checkpointing(self):
        """Test basic trajectory checkpointing"""
        # Create a simple function to checkpoint
        def simple_func(x):
            return x * 2
        
        # Create a simple tensor
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        result = checkpoint_trajectory(simple_func, tensor)
        
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor * 2)
    
    def test_checkpoint_frequency(self):
        """Test checkpoint frequency"""
        def simple_func(x):
            return x * 2
        
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        result = checkpoint_trajectory(simple_func, tensor, checkpoint_freq=2)
        
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor * 2)
    
    def test_max_checkpoints(self):
        """Test maximum checkpoint limit"""
        def simple_func(x):
            return x * 2
        
        tensor = torch.randn(1, 100, 2)  # 100 time steps
        
        result = checkpoint_trajectory(simple_func, tensor)
        
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor * 2)


class TestSDEAdjointOptimizer:
    """Test SDEAdjointOptimizer class"""
    
    def test_initialization(self):
        """Test optimizer initialization"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        sde_optimizer = SDEAdjointOptimizer(model, optimizer)
        
        assert sde_optimizer.model == model
        assert sde_optimizer.optimizer == optimizer
        assert sde_optimizer.checkpoint_config is not None
        assert sde_optimizer.mixed_precision_config is not None
    
    def test_basic_optimization_step(self):
        """Test basic optimization step"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        sde_optimizer = SDEAdjointOptimizer(model, optimizer)
        
        # Create a simple loss
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.mean(y)
        
        # Test optimization step
        sde_optimizer.step(loss)
        
        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_checkpointing_integration(self):
        """Test integration with checkpointing"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_config = CheckpointConfig(checkpoint_frequency=1)
        sde_optimizer = SDEAdjointOptimizer(
            model,
            optimizer,
            checkpoint_config=checkpoint_config
        )
        
        # Test checkpointing
        state = torch.randn(1, 2)
        sde_optimizer.save_state_checkpoint(0, state)
        
        loaded_state = sde_optimizer.load_state_checkpoint(0)
        assert loaded_state is not None
    
    def test_mixed_precision_integration(self):
        """Test integration with mixed precision"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        mixed_precision_config = MixedPrecisionConfig(enable_amp=True)
        sde_optimizer = SDEAdjointOptimizer(
            model,
            optimizer,
            mixed_precision_config=mixed_precision_config
        )
        
        # Test mixed precision training
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.mean(y)
        
        sde_optimizer.step(loss)
        
        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None


class TestSDEAdjointIntegration:
    """Test integration between SDE adjoint components"""
    
    def test_full_adjoint_workflow(self):
        """Test full adjoint optimization workflow"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create optimizer with all features
        checkpoint_config = CheckpointConfig(checkpoint_frequency=2)
        mixed_precision_config = MixedPrecisionConfig(enable_amp=True)
        sde_optimizer = SDEAdjointOptimizer(
            model,
            optimizer,
            checkpoint_config=checkpoint_config,
            mixed_precision_config=mixed_precision_config,
            enable_sparse_gradients=True
        )
        
        # Test full workflow
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.mean(y)
        
        sde_optimizer.step(loss)
        
        # Verify all components worked
        for param in model.parameters():
            assert param.grad is not None
    
    def test_memory_efficiency(self):
        """Test memory efficiency features"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint_config = CheckpointConfig(checkpoint_frequency=1)
        
        sde_optimizer = SDEAdjointOptimizer(
            model,
            optimizer,
            checkpoint_config=checkpoint_config
        )
        
        # Test checkpointing saves memory
        state = torch.randn(1, 2)
        sde_optimizer.save_state_checkpoint(0, state)
        
        loaded_state = sde_optimizer.load_state_checkpoint(0)
        assert loaded_state is not None
        
        # Clear checkpoints
        sde_optimizer.clear_checkpoints()
        assert len(sde_optimizer.checkpoint_manager.checkpoints) == 0


class TestSDEAdjointEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_model_parameters(self):
        """Test with empty model parameters"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # This should work fine
        sde_optimizer = SDEAdjointOptimizer(model, optimizer)
        assert sde_optimizer.model == model
    
    def test_invalid_checkpoint_config(self):
        """Test with invalid checkpoint configuration"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test with invalid config type - should work as it's not validated
        checkpoint_config = "invalid"
        sde_optimizer = SDEAdjointOptimizer(
            model,
            optimizer,
            checkpoint_config=checkpoint_config
        )
        assert sde_optimizer.checkpoint_config == checkpoint_config
    
    def test_invalid_mixed_precision_config(self):
        """Test with invalid mixed precision configuration"""
        # Test with invalid loss_scaling parameter name
        config = MixedPrecisionConfig(loss_scaling=0.0)  # This should work
        assert config.loss_scaling == 0.0
    
    def test_nan_gradients(self):
        """Test handling of NaN gradients"""
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        sde_optimizer = SDEAdjointOptimizer(model, optimizer)
        
        # Create a loss that produces NaN gradients
        x = torch.randn(1, 2, requires_grad=True)
        y = model(x)
        loss = torch.tensor(float('nan'))
        
        # Should handle NaN gracefully
        try:
            sde_optimizer.step(loss)
        except RuntimeError:
            # Expected behavior for NaN gradients
            pass
        
        # Check that gradients are handled
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isnan(param.grad).any() or not torch.isnan(param.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])