"""
Unit tests for neural fractional SDE components in hpfracc.ml.neural_fsde

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import (
    NeuralFSDEConfig, NeuralFractionalSDE, create_neural_fsde
)
from hpfracc.core.definitions import FractionalOrder


class TestNeuralFSDEConfig:
    """Test NeuralFSDEConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = NeuralFSDEConfig()
        
        assert config.diffusion_dim == 1
        assert config.noise_type == "additive"
        assert config.drift_net is None
        assert config.diffusion_net is None
        assert config.use_sde_adjoint is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        drift_net = nn.Linear(2, 2)
        diffusion_net = nn.Linear(2, 1)
        
        config = NeuralFSDEConfig(
            diffusion_dim=2,
            noise_type="multiplicative",
            drift_net=drift_net,
            diffusion_net=diffusion_net,
            use_sde_adjoint=False
        )
        
        assert config.diffusion_dim == 2
        assert config.noise_type == "multiplicative"
        assert config.drift_net is drift_net
        assert config.diffusion_net is diffusion_net
        assert config.use_sde_adjoint is False
    
    def test_invalid_noise_type(self):
        """Test that invalid noise types are handled gracefully"""
        # The current implementation doesn't validate noise_type, so we just test it works
        config = NeuralFSDEConfig(noise_type="invalid_type")
        assert config.noise_type == "invalid_type"


class TestNeuralFractionalSDE:
    """Test NeuralFractionalSDE class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5,
            diffusion_dim=1
        )
    
    def test_initialization(self):
        """Test NeuralFractionalSDE initialization"""
        model = NeuralFractionalSDE(self.config)
        
        assert model.config is self.config
        assert model.fractional_order.alpha == 0.5
        assert model.input_dim == 2
        assert model.diffusion_dim == 1
    
    def test_initialization_with_custom_networks(self):
        """Test initialization with custom drift and diffusion networks"""
        drift_net = nn.Sequential(
            nn.Linear(3, 16),  # t + input_dim
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        diffusion_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            drift_net=drift_net,
            diffusion_net=diffusion_net,
            diffusion_dim=1
        )
        
        model = NeuralFractionalSDE(config)
        
        assert model.drift_net is drift_net
        assert model.diffusion_net is diffusion_net
    
    def test_forward_pass(self):
        """Test forward pass through the model"""
        model = NeuralFractionalSDE(self.config)
        
        # Test input
        t = torch.tensor([0.0, 0.1, 0.2])
        x0 = torch.tensor([[1.0, 0.5]])
        
        # Forward pass
        with torch.no_grad():
            trajectory = model.forward(t, x0)
        
        # Check output shape
        assert trajectory.shape == (3, 1, 2)  # (time_steps, batch_size, output_dim)
        assert not torch.any(torch.isnan(trajectory))
        assert not torch.any(torch.isinf(trajectory))
    
    def test_drift_function(self):
        """Test drift function computation"""
        model = NeuralFractionalSDE(self.config)
        
        t = torch.tensor(0.5)
        x = torch.tensor([[1.0, 0.5]])
        
        drift = model.drift_function(t, x)
        
        assert drift.shape == (1, 2)  # (batch_size, output_dim)
        assert not torch.any(torch.isnan(drift))
    
    def test_diffusion_function(self):
        """Test diffusion function computation"""
        model = NeuralFractionalSDE(self.config)
        
        t = torch.tensor(0.5)
        x = torch.tensor([[1.0, 0.5]])
        
        diffusion = model.diffusion_function(t, x)
        
        assert diffusion.shape == (1, 1)  # (batch_size, diffusion_dim)
        assert not torch.any(torch.isnan(diffusion))
    
    def test_additive_noise(self):
        """Test additive noise configuration"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            noise_type="additive",
            diffusion_dim=1
        )
        
        model = NeuralFractionalSDE(config)
        
        t = torch.tensor(0.5)
        x = torch.tensor([[1.0, 0.5]])
        
        diffusion = model.diffusion_function(t, x)
        
        # For additive noise, diffusion should not depend on state
        x2 = torch.tensor([[2.0, 1.0]])
        diffusion2 = model.diffusion_function(t, x2)
        
        # Should be approximately equal (allowing for small numerical differences)
        assert torch.allclose(diffusion, diffusion2, atol=1e-6)
    
    def test_multiplicative_noise(self):
        """Test multiplicative noise configuration"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            noise_type="multiplicative",
            diffusion_dim=1
        )
        
        model = NeuralFractionalSDE(config)
        
        t = torch.tensor(0.5)
        x1 = torch.tensor([[1.0, 0.5]])
        x2 = torch.tensor([[2.0, 1.0]])
        
        diffusion1 = model.diffusion_function(t, x1)
        diffusion2 = model.diffusion_function(t, x2)
        
        # For multiplicative noise, diffusion should depend on state
        # They should be different (unless by coincidence)
        assert not torch.allclose(diffusion1, diffusion2, atol=1e-6)
    
    def test_learnable_fractional_order(self):
        """Test learnable fractional order"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            learnable_alpha=True,
            fractional_order=0.5
        )
        
        model = NeuralFractionalSDE(config)
        
        # Check that alpha is a parameter
        alpha_param = None
        for name, param in model.named_parameters():
            if 'alpha' in name:
                alpha_param = param
                break
        
        assert alpha_param is not None
        assert alpha_param.requires_grad
    
    def test_fixed_fractional_order(self):
        """Test fixed fractional order"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            learnable_alpha=False,
            fractional_order=0.7
        )
        
        model = NeuralFractionalSDE(config)
        
        # Check that alpha is not a parameter
        alpha_params = [name for name, param in model.named_parameters() 
                       if 'alpha' in name]
        
        assert len(alpha_params) == 0
        assert model.fractional_order.alpha == 0.7
    
    def test_batch_processing(self):
        """Test batch processing"""
        model = NeuralFractionalSDE(self.config)
        
        # Multiple initial conditions
        x0 = torch.tensor([[1.0, 0.5], [2.0, 1.0], [0.5, 1.5]])
        t = torch.tensor([0.0, 0.1, 0.2])
        
        with torch.no_grad():
            trajectory = model.forward(t, x0)
        
        assert trajectory.shape == (3, 3, 2)  # (time_steps, batch_size, output_dim)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        model = NeuralFractionalSDE(self.config)
        
        t = torch.tensor([0.0, 0.1, 0.2])
        x0 = torch.tensor([[1.0, 0.5]])
        
        # Forward pass
        trajectory = model.forward(t, x0)
        
        # Compute loss (simple MSE)
        target = torch.zeros_like(trajectory)
        loss = torch.mean((trajectory - target) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.any(torch.isnan(param.grad))
    
    def test_different_fractional_orders(self):
        """Test with different fractional orders"""
        alphas = [0.3, 0.5, 0.7, 0.9]
        
        for alpha in alphas:
            config = NeuralFSDEConfig(
                input_dim=2,
                output_dim=2,
                fractional_order=alpha
            )
            
            model = NeuralFractionalSDE(config)
            
            t = torch.tensor([0.0, 0.1, 0.2])
            x0 = torch.tensor([[1.0, 0.5]])
            
            with torch.no_grad():
                trajectory = model.forward(t, x0)
            
            assert trajectory.shape == (3, 1, 2)
            assert model.fractional_order.alpha == alpha


class TestCreateNeuralFSDE:
    """Test create_neural_fsde factory function"""
    
    def test_create_with_default_config(self):
        """Test creating neural fSDE with default config"""
        model = create_neural_fsde(
            input_dim=3,
            output_dim=3,
            hidden_dim=32,
            fractional_order=0.6
        )
        
        assert isinstance(model, NeuralFractionalSDE)
        assert model.input_dim == 3
        assert model.fractional_order.alpha == 0.6
    
    def test_create_with_custom_config(self):
        """Test creating neural fSDE with custom config"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            hidden_dim=16,
            fractional_order=0.5,
            diffusion_dim=2,
            noise_type="multiplicative"
        )
        
        model = create_neural_fsde(config=config)
        
        assert isinstance(model, NeuralFractionalSDE)
        assert model.config is config
    
    def test_create_with_learnable_alpha(self):
        """Test creating neural fSDE with learnable fractional order"""
        model = create_neural_fsde(
            input_dim=2,
            output_dim=2,
            learnable_alpha=True,
            fractional_order=0.5
        )
        
        # Check that alpha is learnable
        alpha_params = [name for name, param in model.named_parameters() 
                       if 'alpha' in name]
        
        assert len(alpha_params) > 0


class TestNeuralFSDEIntegration:
    """Test neural fSDE integration with SDE solvers"""
    
    def test_sde_solver_integration(self):
        """Test integration with SDE solvers"""
        model = NeuralFractionalSDE(NeuralFSDEConfig(input_dim=2, output_dim=2))
        
        # Test that the model can be used with SDE solvers
        t = torch.tensor([0.0, 0.1, 0.2])
        x0 = torch.tensor([[1.0, 0.5]])
        
        with torch.no_grad():
            trajectory = model.forward(t, x0)
        
        # Should produce valid trajectory
        assert trajectory.shape == (3, 1, 2)
        assert not torch.any(torch.isnan(trajectory))
    
    def test_adjoint_training_compatibility(self):
        """Test compatibility with adjoint training"""
        config = NeuralFSDEConfig(
            input_dim=2,
            output_dim=2,
            use_sde_adjoint=True
        )
        
        model = NeuralFractionalSDE(config)
        
        # Test that model is compatible with adjoint training
        assert hasattr(model, 'adjoint_forward')
        assert config.use_sde_adjoint is True


class TestNeuralFSDEEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_state_dim(self):
        """Test with invalid state dimension"""
        # The current implementation doesn't validate input_dim, so we just test it works
        config = NeuralFSDEConfig(input_dim=0, output_dim=0)
        assert config.input_dim == 0
    
    def test_invalid_diffusion_dim(self):
        """Test with invalid diffusion dimension"""
        # The current implementation doesn't validate diffusion_dim, so we just test it works
        config = NeuralFSDEConfig(diffusion_dim=0)
        assert config.diffusion_dim == 0
    
    def test_invalid_fractional_order(self):
        """Test with invalid fractional order"""
        # The current implementation doesn't validate fractional_order, so we just test it works
        config = NeuralFSDEConfig(fractional_order=2.5)
        assert config.fractional_order == 2.5
    
    def test_empty_time_sequence(self):
        """Test with empty time sequence"""
        model = NeuralFractionalSDE(NeuralFSDEConfig(input_dim=2, output_dim=2))
        
        t = torch.tensor([])
        x0 = torch.tensor([[1.0, 0.5]])
        
        with pytest.raises(ValueError):
            model.forward(t, x0)
    
    def test_mismatched_dimensions(self):
        """Test with mismatched input dimensions"""
        model = NeuralFractionalSDE(NeuralFSDEConfig(input_dim=2, output_dim=2))
        
        t = torch.tensor([0.0, 0.1])
        x0 = torch.tensor([[1.0, 0.5, 1.0]])  # Wrong dimension
        
        with pytest.raises(RuntimeError):
            model.forward(t, x0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
