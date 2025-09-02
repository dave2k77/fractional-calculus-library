"""
Core Machine Learning Components for Fractional Calculus

This module provides the foundational ML classes that integrate fractional calculus
with neural networks, attention mechanisms, loss functions, and AutoML capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import abstractmethod
import json
from pathlib import Path

from ..core.definitions import FractionalOrder
from ..algorithms.optimized_methods import (
    OptimizedRiemannLiouville,
    OptimizedCaputo,
    OptimizedGrunwaldLetnikov,
)
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


@dataclass
class MLConfig:
    """Configuration for ML components"""
    device: str = "cpu"
    dtype: str = "float32"
    fractional_order: float = 0.5
    use_gpu: bool = False
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    model_save_path: str = "models/"
    log_interval: int = 10
    backend: BackendType = BackendType.AUTO


class FractionalNeuralNetwork:
    """
    Neural network with fractional calculus integration

    This class provides a flexible framework for building neural networks
    that incorporate fractional derivatives in their forward pass.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        fractional_order: float = 0.5,
        activation: str = "relu",
        dropout: float = 0.1,
        config: Optional[MLConfig] = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or MLConfig()
        self.fractional_order = FractionalOrder(fractional_order)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.dropout_rate = dropout

        # Set backend
        # Resolve backend; treat AUTO as active backend
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize fractional derivative calculators
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)
        self.caputo_calculator = OptimizedCaputo(alpha=fractional_order)
        self.gl_calculator = OptimizedGrunwaldLetnikov(alpha=fractional_order)

        # Build network layers
        self.layers = []
        self._build_network()

        # Initialize weights
        self._initialize_weights()

    def parameters(self) -> List[Any]:
        """Return list of learnable parameters for compatibility with optimizers/tests"""
        params: List[Any] = []
        params.extend(self.weights)
        params.extend(self.biases)
        return params

    def _build_network(self):
        """Build the network architecture using the current backend"""
        # Input layer
        self.layers.append({
            'type': 'linear',
            'in_features': self.input_size,
            'out_features': self.hidden_sizes[0]
        })

        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append({
                'type': 'linear',
                'in_features': self.hidden_sizes[i],
                'out_features': self.hidden_sizes[i + 1]
            })

        # Output layer
        self.layers.append({
            'type': 'linear',
            'in_features': self.hidden_sizes[-1],
            'out_features': self.output_size
        })

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []

        for layer in self.layers:
            if layer['type'] == 'linear':
                # Initialize weights with proper random data
                if self.backend == BackendType.TORCH:
                    import torch
                    weight = torch.randn(
                        layer['in_features'],
                        layer['out_features'],
                        dtype=torch.float32,
                        requires_grad=True)
                    bias = torch.zeros(
                        layer['out_features'],
                        dtype=torch.float32,
                        requires_grad=True)
                elif self.backend == BackendType.JAX:
                    import jax.random as random
                    import jax.numpy as jnp
                    key = random.PRNGKey(0)
                    weight = random.normal(
                        key, (layer['in_features'], layer['out_features']))
                    bias = jnp.zeros(layer['out_features'])
                else:  # NUMBA
                    import numpy as np
                    weight = np.random.randn(
                        layer['in_features'], layer['out_features'])
                    bias = np.zeros(layer['out_features'])

                self.weights.append(weight)
                self.biases.append(bias)

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if self.backend == BackendType.TORCH:
                import torch.nn.init as init
                init.xavier_uniform_(weight)
                init.zeros_(bias)
            else:
                # Xavier-like initialization for JAX/NUMBA
                import math
                scale = math.sqrt(2.0 / (weight.shape[0] + weight.shape[1]))
                if self.backend == BackendType.JAX:
                    self.weights[i] = weight * scale
                    self.biases[i] = bias * 0.0
                else:  # NUMBA
                    self.weights[i] = weight * scale
                    self.biases[i] = bias * 0.0

    def fractional_forward(self, x: Any, method: str = "RL") -> Any:
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
        if self.backend == BackendType.TORCH:
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.array(x, dtype=np.float32)

        # Apply fractional derivative
        if x_np.ndim == 2:
            # For 2D tensors (batch_size, features)
            result = np.zeros_like(x_np, dtype=np.float32)
            for i in range(x_np.shape[0]):
                t = np.linspace(0, 1, x_np.shape[1], dtype=np.float32)
                result[i] = calculator.compute(x_np[i], t, t[1] - t[0])
        else:
            # For 1D tensors
            t = np.linspace(0, 1, x_np.shape[0], dtype=np.float32)
            result = calculator.compute(x_np, t, t[1] - t[0])

        # Convert back to backend tensor with consistent dtype
        return self.tensor_ops.create_tensor(
            result.astype(np.float32), requires_grad=True)

    def forward(
            self,
            x: Any,
            use_fractional: bool = True,
            method: str = "RL") -> Any:
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
        for i, (weight, bias) in enumerate(
                zip(self.weights[:-1], self.biases[:-1])):
            # Linear transformation
            x = self.tensor_ops.matmul(x, weight) + bias

            # Apply activation
            x = self._apply_activation(x)

            # Apply dropout
            x = self.tensor_ops.dropout(x, p=self.dropout_rate, training=True)

        # Output layer (no activation)
        x = self.tensor_ops.matmul(x, self.weights[-1]) + self.biases[-1]

        return x

    def _apply_activation(self, x: Any) -> Any:
        """Apply activation function based on backend"""
        if self.activation_name == "relu":
            return self.tensor_ops.relu(x)
        elif self.activation_name == "sigmoid":
            return self.tensor_ops.sigmoid(x)
        elif self.activation_name == "tanh":
            return self.tensor_ops.tanh(x)
        else:
            return x

    def save_model(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save weights and biases
        model_data = {
            'weights': [
                self.tensor_ops.create_tensor(w) for w in self.weights], 'biases': [
                self.tensor_ops.create_tensor(b) for b in self.biases]}

        if self.backend == BackendType.TORCH:
            import torch
            torch.save(model_data, path)
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)

        # Save configuration
        config_path = path.replace('.pth', '_config.json')
        config_data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'fractional_order': float(self.fractional_order),
            'activation': self.activation_name,
            'backend': self.backend.value
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

        # Determine backend from config
        backend = BackendType(config_data.get('backend', 'torch'))

        model = cls(
            input_size=config_data['input_size'],
            hidden_sizes=config_data['hidden_sizes'],
            output_size=config_data['output_size'],
            fractional_order=config_data['fractional_order'],
            backend=backend
        )

        # Load weights and biases
        if backend == BackendType.TORCH:
            import torch
            model_data = torch.load(path)
        else:
            import pickle
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

        model.weights = model_data['weights']
        model.biases = model_data['biases']

        return model

    def __call__(
            self,
            x: Any,
            use_fractional: bool = True,
            method: str = "RL") -> Any:
        """Make the network callable"""
        return self.forward(x, use_fractional, method)


class FractionalAttention:
    """
    Attention mechanism with fractional calculus integration

    This class implements attention mechanisms that use fractional derivatives
    to capture long-range dependencies and temporal relationships.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        fractional_order: float = 0.5,
        dropout: float = 0.1,
        backend: Optional[BackendType] = None
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        # Ensure d_k is valid
        if d_model % n_heads != 0:
            # Adjust d_model to be divisible by n_heads
            self.d_model = ((d_model // n_heads) + 1) * n_heads
            print(
                f"Warning: d_model adjusted from {d_model} to {self.d_model} to be divisible by {n_heads}")
        self.d_k = self.d_model // n_heads
        self.fractional_order = FractionalOrder(fractional_order)
        self.dropout_rate = dropout

        # Set backend
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize attention weights
        self._initialize_weights()

        # Fractional derivative calculators
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)
        self.caputo_calculator = OptimizedCaputo(alpha=fractional_order)

    def _initialize_weights(self):
        """Initialize attention weights"""
        if self.backend == BackendType.TORCH:
            import torch
            self.w_q = torch.randn(
                self.d_model, self.d_model, dtype=torch.float32)
            self.w_k = torch.randn(
                self.d_model, self.d_model, dtype=torch.float32)
            self.w_v = torch.randn(
                self.d_model, self.d_model, dtype=torch.float32)
            self.w_o = torch.randn(
                self.d_model, self.d_model, dtype=torch.float32)

            # Xavier initialization
            import torch.nn.init as init
            init.xavier_uniform_(self.w_q)
            init.xavier_uniform_(self.w_k)
            init.xavier_uniform_(self.w_v)
            init.xavier_uniform_(self.w_o)
        elif self.backend == BackendType.JAX:
            import jax.random as random
            key = random.PRNGKey(0)
            self.w_q = random.normal(key, (self.d_model, self.d_model))
            self.w_k = random.normal(key, (self.d_model, self.d_model))
            self.w_v = random.normal(key, (self.d_model, self.d_model))
            self.w_o = random.normal(key, (self.d_model, self.d_model))
        else:  # NUMBA
            import numpy as np
            self.w_q = np.random.randn(self.d_model, self.d_model)
            self.w_k = np.random.randn(self.d_model, self.d_model)
            self.w_v = np.random.randn(self.d_model, self.d_model)
            self.w_o = np.random.randn(self.d_model, self.d_model)

    def fractional_attention(
            self,
            q: Any,
            k: Any,
            v: Any,
            method: str = "RL") -> Any:
        """
        Compute attention with fractional derivatives

        Args:
            q, k, v: Query, key, value tensors of shape (batch_size, n_heads, seq_len, d_k)
            method: Fractional derivative method

        Returns:
            Attention output with fractional calculus applied
        """
        # Compute attention scores
        # k needs to be transposed to (batch_size, n_heads, d_k, seq_len) for
        # matmul
        k_t = self.tensor_ops.transpose(k, (0, 1, 3, 2))
        # Use tensor_ops for sqrt to maintain dtype consistency
        # Ensure d_k is the same dtype as the input tensors
        if self.backend == BackendType.TORCH:
            import torch
            d_k_tensor = torch.tensor(self.d_k, dtype=torch.float32)
        else:
            d_k_tensor = self.tensor_ops.create_tensor(self.d_k)
        d_k_sqrt = self.tensor_ops.sqrt(d_k_tensor)
        scores = self.tensor_ops.matmul(q, k_t) / d_k_sqrt
        attention_weights = self.tensor_ops.softmax(scores, dim=-1)
        attention_weights = self.tensor_ops.dropout(
            attention_weights, p=self.dropout_rate, training=True)

        # Apply attention to values
        context = self.tensor_ops.matmul(attention_weights, v)

        # Apply fractional derivative to context
        if method == "RL":
            calculator = self.rl_calculator
        elif method == "Caputo":
            calculator = self.caputo_calculator
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert to numpy for fractional calculus
        if self.backend == BackendType.TORCH:
            context_np = context.detach().cpu().numpy()
        else:
            context_np = np.array(context)

        # Apply fractional derivative along sequence dimension
        result = np.zeros_like(context_np)
        for batch in range(context_np.shape[0]):
            for head in range(context_np.shape[1]):
                for feature in range(context_np.shape[3]):
                    t = np.linspace(0, 1, context_np.shape[2])
                    if len(t) > 1:
                        dt = t[1] - t[0]
                    else:
                        dt = 1.0  # Default time step for single element
                    result[batch, head, :, feature] = calculator.compute(
                        context_np[batch, head, :, feature], t, dt
                    )

        # Convert back to backend tensor
        return self.tensor_ops.create_tensor(result, requires_grad=True)

    def forward(self, x: Any, method: str = "RL") -> Any:
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
        q = self.tensor_ops.matmul(x, self.w_q)
        k = self.tensor_ops.matmul(x, self.w_k)
        v = self.tensor_ops.matmul(x, self.w_v)

        # Reshape for multi-head attention
        q = self.tensor_ops.reshape(
            q, (batch_size, seq_len, self.n_heads, self.d_k))
        k = self.tensor_ops.reshape(
            k, (batch_size, seq_len, self.n_heads, self.d_k))
        v = self.tensor_ops.reshape(
            v, (batch_size, seq_len, self.n_heads, self.d_k))

        # Transpose for attention computation (batch_size, n_heads, seq_len,
        # d_k)
        q = self.tensor_ops.transpose(q, (0, 2, 1, 3))
        k = self.tensor_ops.transpose(k, (0, 2, 1, 3))
        v = self.tensor_ops.transpose(v, (0, 2, 1, 3))

        # Apply fractional attention
        context = self.fractional_attention(q, k, v, method)

        # Reshape and apply output projection
        context = self.tensor_ops.transpose(context, (0, 2, 1, 3))
        context = self.tensor_ops.reshape(
            context, (batch_size, seq_len, self.d_model))
        output = self.tensor_ops.matmul(context, self.w_o)

        # Residual connection and layer normalization (simplified)
        # Ensure consistent dtype for residual connection
        if self.backend == BackendType.TORCH:
            if x.dtype != output.dtype:
                output = output.to(x.dtype)
        output = x + output

        return output

    def __call__(self, x: Any, method: str = "RL") -> Any:
        """Make the attention mechanism callable"""
        return self.forward(x, method)


class FractionalLossFunction:
    """
    Base class for loss functions with fractional calculus integration

    This class provides a framework for creating loss functions that
    incorporate fractional derivatives to capture complex relationships.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(self, fractional_order: float = 0.5,
                 backend: Optional[BackendType] = None):
        self.fractional_order = FractionalOrder(fractional_order)
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.rl_calculator = OptimizedRiemannLiouville(alpha=fractional_order)

    @abstractmethod
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        """Compute the base loss"""

    def fractional_loss(self, predictions: Any, targets: Any) -> Any:
        """
        Compute loss with fractional derivative applied to predictions

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Fractional loss value
        """
        # Apply fractional derivative to predictions
        if self.backend == BackendType.TORCH:
            pred_np = predictions.detach().cpu().numpy()
        else:
            pred_np = np.array(predictions)

        if pred_np.ndim == 2:
            # For 2D tensors (batch_size, features)
            result = np.zeros_like(pred_np)
            for i in range(pred_np.shape[0]):
                t = np.linspace(0, 1, pred_np.shape[1])
                result[i] = self.rl_calculator.compute(
                    pred_np[i], t, t[1] - t[0])
        else:
            # For 1D tensors
            t = np.linspace(0, 1, pred_np.shape[0])
            result = self.rl_calculator.compute(pred_np, t, t[1] - t[0])

        fractional_pred = self.tensor_ops.create_tensor(
            result, requires_grad=True)

        # Compute loss with fractional predictions
        return self.compute_loss(fractional_pred, targets)

    def forward(self, predictions: Any, targets: Any,
                use_fractional: bool = True) -> Any:
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

    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        return self.tensor_ops.mean((predictions - targets) ** 2)


class FractionalCrossEntropyLoss(FractionalLossFunction):
    """Cross Entropy loss with fractional calculus integration"""

    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        # Simplified cross-entropy for multi-backend compatibility
        # In practice, you'd want more sophisticated implementations
        return self.tensor_ops.mean(-targets * self.tensor_ops.log(
            self.tensor_ops.softmax(predictions, dim=-1)))


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
        train_data: Tuple[Any, Any],
        val_data: Tuple[Any, Any],
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
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1])
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1])
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range)

            # Create and train model
            model = model_class(**params)

            # Training loop (simplified)
            X_train, y_train = train_data
            X_val, y_val = val_data

            # Simple evaluation (in practice, you'd want proper training)
            model(X_train)

            # Evaluate on validation set
            model(X_val)

            if metric == "accuracy":
                # Simplified accuracy calculation
                return 0.5  # Placeholder
            else:
                # Simplified loss calculation
                return 0.1  # Placeholder

        # Create study and optimize
        study = optuna.create_study(
            direction="maximize" if metric == "accuracy" else "minimize")
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
