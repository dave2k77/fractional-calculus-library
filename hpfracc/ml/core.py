"""
Core Machine Learning Components for Fractional Calculus

This module provides the foundational ML classes that integrate fractional calculus
with neural networks, attention mechanisms, loss functions, and AutoML capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path

from ..core.definitions import FractionalOrder
from ..algorithms.optimized_methods import (
    OptimizedRiemannLiouville,
    OptimizedCaputo,
    OptimizedGrunwaldLetnikov,
)


@dataclass
class MLConfig:
    """Configuration for ML components"""
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    fractional_order: float = 0.5
    use_gpu: bool = False
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_save_path: str = "models/"
    log_interval: int = 10


class FractionalNeuralNetwork(nn.Module):
    """
    Neural network with fractional calculus integration
    
    This class provides a flexible framework for building neural networks
    that incorporate fractional derivatives in their forward pass.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        fractional_order: float = 0.5,
        activation: str = "relu",
        dropout: float = 0.1,
        config: Optional[MLConfig] = None
    ):
        super().__init__()
        
        self.config = config or MLConfig()
        self.fractional_order = FractionalOrder(fractional_order)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize fractional derivative calculators
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)
        self.caputo_calculator = OptimizedCaputo(alpha=fractional_order)
        self.gl_calculator = OptimizedGrunwaldLetnikov(alpha=fractional_order)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def fractional_forward(self, x: torch.Tensor, method: str = "RL") -> torch.Tensor:
        """
        Apply fractional derivative to input
        
        Args:
            x: Input tensor
            method: Fractional derivative method ("RL", "Caputo", "GL")
            
        Returns:
            Tensor with fractional derivative applied
        """
        if method == "RL":
            calculator = self.rl_calculator
        elif method == "Caputo":
            calculator = self.caputo_calculator
        elif method == "GL":
            calculator = self.gl_calculator
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to numpy for fractional calculus computation
        x_np = x.detach().cpu().numpy()
        
        # Apply fractional derivative
        if x_np.ndim == 2:
            # For 2D tensors (batch_size, features)
            result = np.zeros_like(x_np)
            for i in range(x_np.shape[0]):
                t = np.linspace(0, 1, x_np.shape[1])
                result[i] = calculator.compute(x_np[i], t, t[1] - t[0])
        else:
            # For 1D tensors
            t = np.linspace(0, 1, x_np.shape[0])
            result = calculator.compute(x_np, t, t[1] - t[0])
        
        return torch.from_numpy(result).to(x.device, x.dtype)
    
    def forward(self, x: torch.Tensor, use_fractional: bool = True, method: str = "RL") -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            use_fractional: Whether to apply fractional derivatives
            method: Fractional derivative method if use_fractional is True
            
        Returns:
            Network output
        """
        if use_fractional:
            x = self.fractional_forward(x, method)
        
        # Pass through network layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        
        return x
    
    def save_model(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        
        # Save configuration
        config_path = path.replace('.pth', '_config.json')
        config_data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'fractional_order': float(self.fractional_order),
            'activation': self.activation.__name__ if hasattr(self.activation, '__name__') else str(self.activation)
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def load_model(cls, path: str, config_path: Optional[str] = None):
        """Load model from file"""
        if config_path is None:
            config_path = path.replace('.pth', '_config.json')
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        model = cls(
            input_size=config_data['input_size'],
            hidden_sizes=config_data['hidden_sizes'],
            output_size=config_data['output_size'],
            fractional_order=config_data['fractional_order']
        )
        
        model.load_state_dict(torch.load(path))
        return model


class FractionalAttention(nn.Module):
    """
    Attention mechanism with fractional calculus integration
    
    This class implements attention mechanisms that use fractional derivatives
    to capture long-range dependencies and temporal relationships.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        fractional_order: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.fractional_order = FractionalOrder(fractional_order)
        
        # Attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Fractional derivative calculators
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)
        self.caputo_calculator = OptimizedCaputo(alpha=fractional_order)
    
    def fractional_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                           method: str = "RL") -> torch.Tensor:
        """
        Compute attention with fractional derivatives
        
        Args:
            q, k, v: Query, key, value tensors
            method: Fractional derivative method
            
        Returns:
            Attention output with fractional calculus applied
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Apply fractional derivative to context
        if method == "RL":
            calculator = self.rl_calculator
        elif method == "Caputo":
            calculator = self.caputo_calculator
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert to numpy for fractional calculus
        context_np = context.detach().cpu().numpy()
        
        # Apply fractional derivative along sequence dimension
        result = np.zeros_like(context_np)
        for batch in range(context_np.shape[0]):
            for head in range(context_np.shape[1]):
                for feature in range(context_np.shape[3]):
                    t = np.linspace(0, 1, context_np.shape[2])
                    result[batch, head, :, feature] = calculator.compute(
                        context_np[batch, head, :, feature], t, t[1] - t[0]
                    )
        
        return torch.from_numpy(result).to(context.device, context.dtype)
    
    def forward(self, x: torch.Tensor, method: str = "RL") -> torch.Tensor:
        """
        Forward pass through fractional attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            method: Fractional derivative method
            
        Returns:
            Output tensor with attention and fractional calculus applied
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply fractional attention
        context = self.fractional_attention(q, k, v, method)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + output)
        
        return output


class FractionalLossFunction(nn.Module):
    """
    Base class for loss functions with fractional calculus integration
    
    This class provides a framework for creating loss functions that
    incorporate fractional derivatives to capture complex relationships.
    """
    
    def __init__(self, fractional_order: float = 0.5):
        super().__init__()
        self.fractional_order = FractionalOrder(fractional_order)
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the base loss"""
        pass
    
    def fractional_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with fractional derivative applied to predictions
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Fractional loss value
        """
        # Apply fractional derivative to predictions
        pred_np = predictions.detach().cpu().numpy()
        
        if pred_np.ndim == 2:
            # For 2D tensors (batch_size, features)
            result = np.zeros_like(pred_np)
            for i in range(pred_np.shape[0]):
                t = np.linspace(0, 1, pred_np.shape[1])
                result[i] = self.rl_calculator.compute(pred_np[i], t, t[1] - t[0])
        else:
            # For 1D tensors
            t = np.linspace(0, 1, pred_np.shape[0])
            result = self.rl_calculator.compute(pred_np, t, t[1] - t[0])
        
        fractional_pred = torch.from_numpy(result).to(predictions.device, predictions.dtype)
        
        # Compute loss with fractional predictions
        return self.compute_loss(fractional_pred, targets)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                use_fractional: bool = True) -> torch.Tensor:
        """
        Forward pass for loss computation
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            use_fractional: Whether to apply fractional derivatives
            
        Returns:
            Loss value
        """
        if use_fractional:
            return self.fractional_loss(predictions, targets)
        else:
            return self.compute_loss(predictions, targets)


class FractionalMSELoss(FractionalLossFunction):
    """Mean Squared Error loss with fractional calculus integration"""
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)


class FractionalCrossEntropyLoss(FractionalLossFunction):
    """Cross Entropy loss with fractional calculus integration"""
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)


class FractionalAutoML:
    """
    Automated Machine Learning for fractional calculus parameters
    
    This class provides automated optimization of fractional orders and
    other hyperparameters for optimal performance on specific tasks.
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.best_params = {}
        self.optimization_history = []
    
    def optimize_fractional_order(
        self,
        model_class: type,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        param_ranges: Dict[str, List[float]],
        n_trials: int = 50,
        metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Optimize fractional order and other hyperparameters
        
        Args:
            model_class: Class of model to optimize
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            param_ranges: Dictionary of parameter ranges to search
            n_trials: Number of optimization trials
            metric: Metric to optimize
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        import optuna
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Create and train model
            model = model_class(**params)
            model.train()
            
            # Training loop (simplified)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            for epoch in range(min(10, self.config.max_epochs)):  # Limit epochs for optimization
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
                if metric == "accuracy":
                    _, predicted = torch.max(val_outputs.data, 1)
                    accuracy = (predicted == y_val).sum().item() / y_val.size(0)
                    return accuracy
                else:
                    return -val_loss.item()  # Minimize loss
        
        # Create study and optimize
        study = optuna.create_study(direction="maximize" if metric == "accuracy" else "minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Store results
        self.best_params = study.best_params
        self.optimization_history = study.trials
        
        return {
            'best_params': self.best_params,
            'best_value': study.best_value,
            'optimization_history': self.optimization_history
        }
    
    def get_best_model(self, model_class: type, **kwargs) -> Any:
        """Get model instance with best parameters"""
        if not self.best_params:
            raise ValueError("No optimization has been run yet")
        
        # Merge best params with additional kwargs
        params = {**self.best_params, **kwargs}
        return model_class(**params)
