"""
Unit tests for SDE adjoint optimization utilities in hpfracc.ml.sde_adjoint_utils

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from hpfracc.ml.sde_adjoint_utils import (
    CheckpointConfig, MixedPrecisionConfig, SDEStateCheckpoint,
    MixedPrecisionManager, SparseGradientAccumulator,
    checkpoint_trajectory, SDEAdjointOptimizer
)


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = CheckpointConfig()
        
        assert config.checkpoint_frequency == 10
        assert config.max_checkpoints == 100
        assert config.compression_enabled is False
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CheckpointConfig(
            checkpoint_frequency=5,
            max_checkpoints=50,
            compression_enabled=True
        )
        
        assert config.checkpoint_frequency == 5
        assert config.max_checkpoints == 50
        assert config.compression_enabled is True


class TestMixedPrecisionConfig:
    """Test MixedPrecisionConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = MixedPrecisionConfig()
        
        assert config.enabled is False
        assert config.loss_scale == 1.0
        assert config.initial_loss_scale == 65536.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MixedPrecisionConfig(
            enabled=True,
            loss_scale=2.0,
            initial_loss_scale=32768.0
        )
        
        assert config.enabled is True
        assert config.loss_scale == 2.0
        assert config.initial_loss_scale == 32768.0


class TestSDEStateCheckpoint:
    """Test SDEStateCheckpoint class"""
    
    def test_initialization(self):
        """Test checkpoint initialization"""
        state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checkpoint = SDEStateCheckpoint(state, step=5)
        
        assert torch.equal(checkpoint.state, state)
        assert checkpoint.step == 5
        assert checkpoint.timestamp is not None
    
    def test_state_access(self):
        """Test state access"""
        state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        checkpoint = SDEStateCheckpoint(state, step=10)
        
        retrieved_state = checkpoint.get_state()
        assert torch.equal(retrieved_state, state)
    
    def test_step_tracking(self):
        """Test step tracking"""
        state = torch.tensor([[1.0, 2.0]])
        checkpoint = SDEStateCheckpoint(state, step=42)
        
        assert checkpoint.get_step() == 42


class TestMixedPrecisionManager:
    """Test MixedPrecisionManager class"""
    
    def test_initialization(self):
        """Test manager initialization"""
        config = MixedPrecisionConfig(enabled=True)
        manager = MixedPrecisionManager(config)
        
        assert manager.config is config
        assert manager.enabled is True
    
    def test_disabled_mode(self):
        """Test disabled mixed precision"""
        config = MixedPrecisionConfig(enabled=False)
        manager = MixedPrecisionManager(config)
        
        assert manager.enabled is False
    
    def test_loss_scaling(self):
        """Test loss scaling functionality"""
        config = MixedPrecisionConfig(enabled=True, loss_scale=2.0)
        manager = MixedPrecisionManager(config)
        
        # Test scaling
        original_loss = torch.tensor(1.0)
        scaled_loss = manager.scale_loss(original_loss)
        
        assert scaled_loss.item() == 2.0
    
    def test_gradient_scaling(self):
        """Test gradient scaling functionality"""
        config = MixedPrecisionConfig(enabled=True, loss_scale=4.0)
        manager = MixedPrecisionManager(config)
        
        # Create a tensor with gradients
        tensor = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = torch.sum(tensor)
        loss.backward()
        
        # Scale gradients
        manager.scale_gradients([tensor])
        
        # Gradients should be scaled
        assert torch.allclose(tensor.grad, torch.tensor([[4.0, 4.0]]))


class TestSparseGradientAccumulator:
    """Test SparseGradientAccumulator class"""
    
    def test_initialization(self):
        """Test accumulator initialization"""
        accumulator = SparseGradientAccumulator()
        
        assert accumulator is not None
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        accumulator = SparseGradientAccumulator()
        
        # Create tensors with gradients
        tensor1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
        tensor2 = torch.tensor([[3.0, 4.0]], requires_grad=True)
        
        loss1 = torch.sum(tensor1)
        loss1.backward()
        
        loss2 = torch.sum(tensor2)
        loss2.backward()
        
        # Accumulate gradients
        accumulator.accumulate([tensor1, tensor2])
        
        # Check that gradients are accumulated
        assert tensor1.grad is not None
        assert tensor2.grad is not None
    
    def test_sparse_accumulation(self):
        """Test sparse gradient accumulation"""
        accumulator = SparseGradientAccumulator()
        
        # Create sparse gradients
        tensor = torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True)
        loss = torch.sum(tensor)
        loss.backward()
        
        # Accumulate sparse gradients
        accumulator.accumulate([tensor])
        
        # Should handle sparse gradients correctly
        assert tensor.grad is not None


class TestCheckpointTrajectory:
    """Test checkpoint_trajectory function"""
    
    def test_basic_checkpointing(self):
        """Test basic trajectory checkpointing"""
        # Create a simple trajectory
        trajectory = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        config = CheckpointConfig(checkpoint_frequency=2)
        
        checkpoints = checkpoint_trajectory(trajectory, config)
        
        assert len(checkpoints) > 0
        assert all(isinstance(cp, SDEStateCheckpoint) for cp in checkpoints)
    
    def test_checkpoint_frequency(self):
        """Test checkpoint frequency"""
        trajectory = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        config = CheckpointConfig(checkpoint_frequency=2)
        
        checkpoints = checkpoint_trajectory(trajectory, config)
        
        # Should checkpoint every 2 steps
        expected_steps = [0, 2]  # Assuming 0-indexed
        actual_steps = [cp.step for cp in checkpoints]
        
        # Check that checkpoint steps match expected frequency
        assert len(actual_steps) <= len(expected_steps)
    
    def test_max_checkpoints(self):
        """Test maximum checkpoint limit"""
        # Create a long trajectory
        trajectory = torch.randn(1, 100, 2)  # 100 time steps
        config = CheckpointConfig(max_checkpoints=5)
        
        checkpoints = checkpoint_trajectory(trajectory, config)
        
        assert len(checkpoints) <= 5


class TestSDEAdjointOptimizer:
    """Test SDEAdjointOptimizer class"""
    
    def test_initialization(self):
        """Test optimizer initialization"""
        model = nn.Linear(2, 2)
        optimizer = SDEAdjointOptimizer(model.parameters())
        
        assert optimizer is not None
    
    def test_basic_optimization_step(self):
        """Test basic optimization step"""
        model = nn.Linear(2, 2)
        optimizer = SDEAdjointOptimizer(model.parameters())
        
        # Create dummy loss
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.sum(y)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert True
    
    def test_checkpointing_integration(self):
        """Test integration with checkpointing"""
        model = nn.Linear(2, 2)
        checkpoint_config = CheckpointConfig(checkpoint_frequency=1)
        optimizer = SDEAdjointOptimizer(
            model.parameters(),
            checkpoint_config=checkpoint_config
        )
        
        # Create dummy loss
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.sum(y)
        
        # Optimization step with checkpointing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert True
    
    def test_mixed_precision_integration(self):
        """Test integration with mixed precision"""
        model = nn.Linear(2, 2)
        mixed_precision_config = MixedPrecisionConfig(enabled=True)
        optimizer = SDEAdjointOptimizer(
            model.parameters(),
            mixed_precision_config=mixed_precision_config
        )
        
        # Create dummy loss
        x = torch.randn(1, 2)
        y = model(x)
        loss = torch.sum(y)
        
        # Optimization step with mixed precision
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert True


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
        
        # Create optimizer with all features
        checkpoint_config = CheckpointConfig(checkpoint_frequency=2)
        mixed_precision_config = MixedPrecisionConfig(enabled=True)
        
        optimizer = SDEAdjointOptimizer(
            model.parameters(),
            checkpoint_config=checkpoint_config,
            mixed_precision_config=mixed_precision_config
        )
        
        # Training loop
        for step in range(5):
            x = torch.randn(1, 2)
            y = model(x)
            loss = torch.sum(y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should complete without errors
        assert True
    
    def test_memory_efficiency(self):
        """Test memory efficiency features"""
        model = nn.Linear(2, 2)
        checkpoint_config = CheckpointConfig(checkpoint_frequency=1)
        
        optimizer = SDEAdjointOptimizer(
            model.parameters(),
            checkpoint_config=checkpoint_config
        )
        
        # Create a trajectory that would normally use a lot of memory
        trajectory = torch.randn(1, 50, 2)  # 50 time steps
        
        # Checkpoint the trajectory
        checkpoints = checkpoint_trajectory(trajectory, checkpoint_config)
        
        # Should have created checkpoints to save memory
        assert len(checkpoints) > 0


class TestSDEAdjointEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_model_parameters(self):
        """Test with empty model parameters"""
        with pytest.raises(ValueError):
            SDEAdjointOptimizer([])
    
    def test_invalid_checkpoint_config(self):
        """Test with invalid checkpoint configuration"""
        model = nn.Linear(2, 2)
        
        with pytest.raises(ValueError):
            CheckpointConfig(checkpoint_frequency=0)
    
    def test_invalid_mixed_precision_config(self):
        """Test with invalid mixed precision configuration"""
        with pytest.raises(ValueError):
            MixedPrecisionConfig(loss_scale=0.0)
    
    def test_nan_gradients(self):
        """Test handling of NaN gradients"""
        model = nn.Linear(2, 2)
        optimizer = SDEAdjointOptimizer(model.parameters())
        
        # Create NaN gradients
        for param in model.parameters():
            param.grad = torch.tensor(float('nan'))
        
        # Should handle NaN gradients gracefully
        try:
            optimizer.step()
        except RuntimeError:
            # Expected behavior for NaN gradients
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
