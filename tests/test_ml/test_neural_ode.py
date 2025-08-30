"""
Tests for Neural Fractional Ordinary Differential Equations (Neural fODE).

This module contains comprehensive tests for all Neural fODE implementations
including standard Neural ODEs and fractional extensions.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from hpfracc.ml.neural_ode import (
    BaseNeuralODE,
    NeuralODE,
    NeuralFODE,
    NeuralODETrainer,
    create_neural_ode,
    create_neural_ode_trainer
)


class TestBaseNeuralODE:
    """Test base neural ODE class."""
    
    def test_base_neural_ode_creation(self):
        """Test creating BaseNeuralODE instances."""
        # Note: BaseNeuralODE is abstract, so we test through concrete implementations
        pass
    
    def test_activation_functions(self):
        """Test activation function selection."""
        # Test through concrete implementation
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        # Test tanh activation
        x = torch.tensor([[1.0, 2.0]])
        result = model._get_activation(x)
        assert torch.allclose(result, torch.tanh(x))
        
        # Test relu activation
        model.activation = "relu"
        result = model._get_activation(x)
        assert torch.allclose(result, torch.relu(x))
        
        # Test sigmoid activation
        model.activation = "sigmoid"
        result = model._get_activation(x)
        assert torch.allclose(result, torch.sigmoid(x))


class TestNeuralODE:
    """Test standard Neural ODE implementation."""
    
    def test_neural_ode_creation(self):
        """Test creating NeuralODE instances."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        assert model.input_dim == 2
        assert model.hidden_dim == 4
        assert model.output_dim == 1
        assert model.num_layers == 3
        assert model.activation == "tanh"
        assert model.use_adjoint is True
        
        # Check network architecture
        assert len(model.network) == 3  # input + hidden + output layers
        
    def test_ode_func(self):
        """Test ODE function computation."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        # Test with single state
        x = torch.tensor([1.0, 2.0])
        t = torch.tensor(0.5)
        
        result = model.ode_func(t, x)
        assert result.shape == (1,)
        assert torch.isfinite(result).all()
        
        # Test with batch
        x_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t_batch = torch.tensor([0.5, 1.0])
        
        result_batch = model.ode_func(t_batch, x_batch)
        assert result_batch.shape == (2, 1)
        assert torch.isfinite(result_batch).all()
    
    def test_forward_pass_basic(self):
        """Test forward pass with basic solver."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1, solver="euler")
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # batch_size=2
        t = torch.linspace(0, 1, 5)  # 5 time steps
        
        output = model(x, t)
        
        assert output.shape == (2, 5, 1)  # (batch_size, time_steps, output_dim)
        assert torch.isfinite(output).all()
        
        # Check initial condition
        assert torch.allclose(output[:, 0, :], x.unsqueeze(-1), atol=1e-6)
    
    @patch('hpfracc.ml.neural_ode.torchdiffeq')
    def test_forward_pass_torchdiffeq(self, mock_torchdiffeq):
        """Test forward pass with torchdiffeq solver."""
        # Mock torchdiffeq
        mock_torchdiffeq.odeint_adjoint.return_value = torch.randn(5, 2, 1)
        
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1, solver="dopri5")
        model.has_torchdiffeq = True
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.linspace(0, 1, 5)
        
        output = model(x, t)
        
        assert output.shape == (2, 5, 1)
        mock_torchdiffeq.odeint_adjoint.assert_called_once()


class TestNeuralFODE:
    """Test Neural Fractional ODE implementation."""
    
    def test_neural_fode_creation(self):
        """Test creating NeuralFODE instances."""
        model = NeuralFODE(input_dim=2, hidden_dim=4, output_dim=1, fractional_order=0.5)
        
        assert model.input_dim == 2
        assert model.hidden_dim == 4
        assert model.output_dim == 1
        assert model.alpha.alpha == 0.5
        assert model.solver == "fractional_euler"
        
    def test_fractional_order_validation(self):
        """Test fractional order validation."""
        # Valid fractional order
        model = NeuralFODE(input_dim=2, hidden_dim=4, output_dim=1, fractional_order=0.7)
        assert model.alpha.alpha == 0.7
        
        # Invalid fractional order should raise error
        with pytest.raises(ValueError):
            NeuralFODE(input_dim=2, hidden_dim=4, output_dim=1, fractional_order=-0.5)
    
    def test_forward_pass_fractional(self):
        """Test forward pass for fractional ODE."""
        model = NeuralFODE(input_dim=2, hidden_dim=4, output_dim=1, fractional_order=0.5)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t = torch.linspace(0, 1, 5)
        
        output = model(x, t)
        
        assert output.shape == (2, 5, 1)
        assert torch.isfinite(output).all()
        
        # Check initial condition
        assert torch.allclose(output[:, 0, :], x.unsqueeze(-1), atol=1e-6)
    
    def test_get_fractional_order(self):
        """Test getting fractional order."""
        model = NeuralFODE(input_dim=2, hidden_dim=4, output_dim=1, fractional_order=0.8)
        assert model.get_fractional_order() == 0.8


class TestNeuralODETrainer:
    """Test Neural ODE trainer."""
    
    def test_trainer_creation(self):
        """Test creating NeuralODETrainer instances."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        trainer = NeuralODETrainer(model, optimizer="adam", learning_rate=1e-3)
        
        assert trainer.model == model
        assert trainer.learning_rate == 1e-3
        assert trainer.loss_function == "mse"
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.criterion, nn.MSELoss)
    
    def test_optimizer_setup(self):
        """Test optimizer setup."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        # Test Adam
        trainer = NeuralODETrainer(model, optimizer="adam")
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        
        # Test SGD
        trainer = NeuralODETrainer(model, optimizer="sgd")
        assert isinstance(trainer.optimizer, torch.optim.SGD)
        
        # Test RMSprop
        trainer = NeuralODETrainer(model, optimizer="rmsprop")
        assert isinstance(trainer.optimizer, torch.optim.RMSprop)
        
        # Test default
        trainer = NeuralODETrainer(model, optimizer="invalid")
        assert isinstance(trainer.optimizer, torch.optim.Adam)
    
    def test_loss_function_setup(self):
        """Test loss function setup."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        # Test MSE
        trainer = NeuralODETrainer(model, loss_function="mse")
        assert isinstance(trainer.criterion, nn.MSELoss)
        
        # Test MAE
        trainer = NeuralODETrainer(model, loss_function="mae")
        assert isinstance(trainer.criterion, nn.L1Loss)
        
        # Test Huber
        trainer = NeuralODETrainer(model, loss_function="huber")
        assert isinstance(trainer.criterion, nn.SmoothL1Loss)
        
        # Test default
        trainer = NeuralODETrainer(model, loss_function="invalid")
        assert isinstance(trainer.criterion, nn.MSELoss)
    
    def test_train_step(self):
        """Test single training step."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        trainer = NeuralODETrainer(model, learning_rate=1e-3)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_target = torch.randn(2, 5, 1)
        t = torch.linspace(0, 1, 5)
        
        # Record initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Perform training step
        loss = trainer.train_step(x, y_target, t)
        
        # Check that loss is finite
        assert np.isfinite(loss)
        
        # Check that parameters were updated
        current_params = [p.clone() for p in model.parameters()]
        params_changed = any(not torch.allclose(init, curr) 
                           for init, curr in zip(initial_params, current_params))
        assert params_changed
    
    def test_validation(self):
        """Test validation step."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        trainer = NeuralODETrainer(model)
        
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_target = torch.randn(2, 5, 1)
        t = torch.linspace(0, 1, 5)
        
        # Create a simple data loader
        dataset = torch.utils.data.TensorDataset(x, y_target, t)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        val_loss = trainer._validate(val_loader)
        
        assert np.isfinite(val_loss)
        assert val_loss >= 0


class TestFactoryFunctions:
    """Test factory functions for creating neural ODE models."""
    
    def test_create_neural_ode(self):
        """Test create_neural_ode factory function."""
        # Test standard Neural ODE
        model = create_neural_ode("standard", input_dim=2, hidden_dim=4, output_dim=1)
        assert isinstance(model, NeuralODE)
        assert model.input_dim == 2
        assert model.hidden_dim == 4
        assert model.output_dim == 1
        
        # Test fractional Neural ODE
        model = create_neural_ode("fractional", input_dim=2, hidden_dim=4, output_dim=1, fractional_order=0.5)
        assert isinstance(model, NeuralFODE)
        assert model.alpha.alpha == 0.5
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_neural_ode("invalid", input_dim=2, hidden_dim=4, output_dim=1)
    
    def test_create_neural_ode_trainer(self):
        """Test create_neural_ode_trainer factory function."""
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        trainer = create_neural_ode_trainer(model, optimizer="adam", learning_rate=1e-3)
        
        assert isinstance(trainer, NeuralODETrainer)
        assert trainer.model == model
        assert trainer.learning_rate == 1e-3


class TestIntegration:
    """Test integration between different components."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training workflow."""
        # Create model
        model = NeuralODE(input_dim=2, hidden_dim=4, output_dim=1)
        
        # Create trainer
        trainer = NeuralODETrainer(model, learning_rate=1e-2)
        
        # Create simple training data
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_target = torch.randn(2, 5, 1)
        t = torch.linspace(0, 1, 5)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(x, y_target, t)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Train for a few epochs
        history = trainer.train(train_loader, num_epochs=3, verbose=False)
        
        # Check training history
        assert "loss" in history
        assert "epochs" in history
        assert len(history["loss"]) == 3
        assert len(history["epochs"]) == 3
        
        # Check that loss decreased (or at least didn't explode)
        assert history["loss"][-1] < 1e6  # Loss shouldn't explode
