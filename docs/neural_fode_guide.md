# Neural fODE Framework Guide

## Overview

The Neural fODE (Fractional Ordinary Differential Equation) framework in HPFRACC provides a complete implementation for learning-based solution of fractional differential equations. This framework extends the concept of Neural ODEs to fractional calculus, enabling researchers to solve complex fractional differential equations using deep learning approaches.

## 🚀 **Quick Start**

### Installation

```bash
pip install hpfracc[ml]
```

### Basic Usage

```python
import hpfracc.ml.neural_ode as nfode
import torch
import numpy as np

# Create a neural ODE model
model = nfode.NeuralODE(
    input_dim=2,      # Input dimension
    hidden_dim=32,    # Hidden layer dimension
    output_dim=1,     # Output dimension
    num_layers=3,     # Number of hidden layers
    activation="tanh" # Activation function
)

# Create input data
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Batch of initial conditions
t = torch.linspace(0, 1, 100)                # Time points

# Forward pass
solution = model(x, t)
print(f"Solution shape: {solution.shape}")  # (batch_size, time_steps, output_dim)
```

## 🏗️ **Architecture**

### BaseNeuralODE

The abstract base class that provides common functionality for all neural ODE implementations:

- **Network Architecture**: Configurable neural network with multiple layers
- **Activation Functions**: Support for tanh, relu, and sigmoid
- **Weight Initialization**: Xavier initialization for optimal training
- **Abstract Interface**: Defines the contract for neural ODE implementations

### NeuralODE

Standard neural ODE implementation for ordinary differential equations:

- **ODE Function**: Learns the dynamics dx/dt = f(x, t)
- **Multiple Solvers**: Support for dopri5 (with torchdiffeq) and basic Euler
- **Adjoint Method**: Memory-efficient gradient computation
- **Adaptive Stepping**: Configurable tolerance and step size

### NeuralFODE

Fractional neural ODE implementation extending to fractional calculus:

- **Fractional Order**: Configurable fractional order α
- **Fractional Dynamics**: Learns D^α x = f(x, t) where D^α is the fractional derivative
- **Specialized Solvers**: Fractional Euler method for fractional ODEs
- **Order Validation**: Ensures fractional order is in valid range (0 < α < 1)

### NeuralODETrainer

Comprehensive training infrastructure:

- **Multiple Optimizers**: Adam, SGD, RMSprop with configurable learning rates
- **Multiple Loss Functions**: MSE, MAE, Huber loss functions
- **Training Loops**: Complete training and validation workflows
- **History Tracking**: Monitor training progress and performance

## 🔧 **Configuration Options**

### Model Configuration

```python
# Standard Neural ODE
model = nfode.NeuralODE(
    input_dim=2,           # Required: Input dimension
    hidden_dim=32,         # Required: Hidden layer dimension
    output_dim=1,          # Required: Output dimension
    num_layers=3,          # Optional: Number of hidden layers (default: 3)
    activation="tanh",     # Optional: Activation function (default: "tanh")
    use_adjoint=True,      # Optional: Use adjoint method (default: True)
    solver="dopri5",       # Optional: ODE solver (default: "dopri5")
    rtol=1e-5,            # Optional: Relative tolerance (default: 1e-5)
    atol=1e-5             # Optional: Absolute tolerance (default: 1e-5)
)

# Fractional Neural ODE
fode_model = nfode.NeuralFODE(
    input_dim=2,           # Required: Input dimension
    hidden_dim=32,         # Required: Hidden layer dimension
    output_dim=1,          # Required: Output dimension
    fractional_order=0.5,  # Required: Fractional order α
    num_layers=3,          # Optional: Number of hidden layers (default: 3)
    activation="tanh",     # Optional: Activation function (default: "tanh")
    use_adjoint=True,      # Optional: Use adjoint method (default: True)
    solver="fractional_euler", # Optional: Solver type (default: "fractional_euler")
    rtol=1e-5,            # Optional: Relative tolerance (default: 1e-5)
    atol=1e-5             # Optional: Absolute tolerance (default: 1e-5)
)
```

### Training Configuration

```python
trainer = nfode.NeuralODETrainer(
    model=model,                    # Required: Neural ODE model
    optimizer="adam",              # Optional: Optimizer type (default: "adam")
    learning_rate=1e-3,            # Optional: Learning rate (default: 1e-3)
    loss_function="mse"            # Optional: Loss function (default: "mse")
)
```

## 📚 **Examples**

### Example 1: Simple Harmonic Oscillator

```python
import hpfracc.ml.neural_ode as nfode
import torch
import numpy as np
import matplotlib.pyplot as plt

# Create model for harmonic oscillator: d²x/dt² + ω²x = 0
model = nfode.NeuralODE(input_dim=2, hidden_dim=16, output_dim=2)

# Initial conditions: [position, velocity]
x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Two different initial conditions
t = torch.linspace(0, 10, 200)

# Forward pass
with torch.no_grad():
    solution = model(x0, t)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t.numpy(), solution[0, :, 0].numpy(), 'b-', label='Position')
plt.plot(t.numpy(), solution[0, :, 1].numpy(), 'r--', label='Velocity')
plt.title('Initial Condition: [1, 0]')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t.numpy(), solution[1, :, 0].numpy(), 'b-', label='Position')
plt.plot(t.numpy(), solution[1, :, 1].numpy(), 'r--', label='Velocity')
plt.title('Initial Condition: [0, 1]')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Example 2: Fractional Diffusion Equation

```python
# Create fractional neural ODE for diffusion: D^α x = D∇²x
alpha = 0.5  # Fractional order
model = nfode.NeuralFODE(
    input_dim=1, 
    hidden_dim=32, 
    output_dim=1, 
    fractional_order=alpha
)

# Initial condition: Gaussian pulse
x0 = torch.tensor([[1.0], [0.5], [0.1]])  # Different initial amplitudes
t = torch.linspace(0, 5, 100)

# Forward pass
with torch.no_grad():
    solution = model(x0, t)

# Plot results
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(t.numpy(), solution[i, :, 0].numpy(), 
             label=f'Initial amplitude: {x0[i, 0].item():.1f}')

plt.title(f'Fractional Diffusion (α = {alpha})')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

### Example 3: Training a Neural ODE

```python
import torch.utils.data as data

# Create training data (synthetic ODE solution)
def generate_training_data(n_samples=1000, time_steps=50):
    t = torch.linspace(0, 1, time_steps)
    x0 = torch.randn(n_samples, 2) * 2  # Random initial conditions
    
    # Simple ODE: dx/dt = -x (exponential decay)
    solution = x0.unsqueeze(1) * torch.exp(-t.unsqueeze(0).unsqueeze(-1))
    
    return x0, solution, t

# Generate data
x0, y_target, t = generate_training_data()

# Create data loader
dataset = data.TensorDataset(x0, y_target, t)
train_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)

# Create model and trainer
model = nfode.NeuralODE(input_dim=2, hidden_dim=16, output_dim=2)
trainer = nfode.NeuralODETrainer(model, learning_rate=1e-2)

# Train the model
history = trainer.train(train_loader, num_epochs=50, verbose=True)

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['epochs'], history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['epochs'], history['val_loss'])
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 🏭 **Factory Functions**

### Creating Models

```python
# Using factory functions
model = nfode.create_neural_ode(
    model_type="standard",  # or "fractional"
    input_dim=2,
    hidden_dim=32,
    output_dim=1,
    fractional_order=0.5  # Only for fractional models
)

# Creating trainers
trainer = nfode.create_neural_ode_trainer(
    model=model,
    optimizer="adam",
    learning_rate=1e-3
)
```

### Model Properties

```python
# Get model properties
properties = nfode.get_neural_ode_properties(model)
print(f"Model type: {properties['model_type']}")
print(f"Input dimension: {properties['input_dim']}")
print(f"Hidden dimension: {properties['hidden_dim']}")
print(f"Output dimension: {properties['output_dim']}")

# For fractional models
if hasattr(model, 'get_fractional_order'):
    print(f"Fractional order: {model.get_fractional_order()}")
```

## 🔬 **Research Applications**

### Physics-Informed Neural Networks (PINNs)

The Neural fODE framework is particularly well-suited for PINNs applications:

```python
# Example: Learning a fractional differential equation
# D^α x + f(x, t) = 0

class FractionalPINN(nfode.NeuralFODE):
    def __init__(self, input_dim, hidden_dim, output_dim, fractional_order, physics_func):
        super().__init__(input_dim, hidden_dim, output_dim, fractional_order)
        self.physics_func = physics_func
    
    def physics_loss(self, x, t):
        """Compute physics-informed loss"""
        # Forward pass to get solution
        solution = self(x, t)
        
        # Compute fractional derivative (simplified)
        # In practice, use proper fractional derivative computation
        alpha = self.get_fractional_order()
        
        # Physics constraint: D^α x + f(x, t) = 0
        physics_residual = self.physics_func(solution, t)
        
        return torch.mean(physics_residual**2)

# Usage
def physics_constraint(x, t):
    return x + 0.1 * torch.sin(t)  # Example constraint

model = FractionalPINN(
    input_dim=1, 
    hidden_dim=32, 
    output_dim=1, 
    fractional_order=0.5,
    physics_func=physics_constraint
)
```

### Time Series Prediction

```python
# Predict future values of a time series
def predict_future(model, x0, t_past, t_future):
    """Predict future values using trained neural ODE"""
    model.eval()
    with torch.no_grad():
        # Combine past and future time points
        t_combined = torch.cat([t_past, t_future])
        
        # Get full solution
        solution = model(x0, t_combined)
        
        # Extract future part
        future_solution = solution[:, len(t_past):, :]
        
    return future_solution

# Example usage
t_past = torch.linspace(0, 5, 100)
t_future = torch.linspace(5, 10, 100)
x0 = torch.tensor([[1.0, 0.0]])

future_prediction = predict_future(model, x0, t_past, t_future)
```

## ⚡ **Performance Optimization**

### GPU Acceleration

```python
# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move data to GPU
x0 = x0.to(device)
t = t.to(device)
```

### Batch Processing

```python
# Process multiple initial conditions simultaneously
batch_size = 1000
x0_batch = torch.randn(batch_size, 2)
t = torch.linspace(0, 1, 50)

# Efficient batch processing
solution_batch = model(x0_batch, t)
```

### Memory Management

```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class MemoryEfficientNeuralODE(nfode.NeuralODE):
    def forward(self, x, t):
        # Use gradient checkpointing to save memory
        return checkpoint(super().forward, x, t)
```

## 🧪 **Testing and Validation**

### Running Tests

```bash
# Run all neural ODE tests
python -m pytest tests/test_ml/test_neural_ode.py -v

# Run specific test categories
python -m pytest tests/test_ml/test_neural_ode.py::TestNeuralODE -v
python -m pytest tests/test_ml/test_neural_ode.py::TestNeuralFODE -v
python -m pytest tests/test_ml/test_neural_ode.py::TestNeuralODETrainer -v
```

### Validation Examples

```python
# Validate model behavior
def validate_model(model, test_cases):
    """Validate model behavior on test cases"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for x0, t, expected_shape in test_cases:
            try:
                output = model(x0, t)
                shape_correct = output.shape == expected_shape
                finite_output = torch.isfinite(output).all()
                
                results.append({
                    'test_case': (x0.shape, t.shape),
                    'output_shape': output.shape,
                    'expected_shape': expected_shape,
                    'shape_correct': shape_correct,
                    'finite_output': finite_output.item()
                })
            except Exception as e:
                results.append({
                    'test_case': (x0.shape, t.shape),
                    'error': str(e)
                })
    
    return results

# Example validation
test_cases = [
    (torch.randn(1, 2), torch.linspace(0, 1, 10), (1, 10, 2)),
    (torch.randn(5, 2), torch.linspace(0, 1, 20), (5, 20, 2)),
    (torch.randn(10, 2), torch.linspace(0, 1, 50), (10, 50, 2))
]

validation_results = validate_model(model, test_cases)
for result in validation_results:
    print(result)
```

## 🔮 **Future Developments**

### Planned Features

- **Neural fSDE**: Stochastic differential equation solving
- **Advanced Solvers**: More sophisticated ODE solvers
- **Multi-scale Methods**: Adaptive time stepping
- **Physics Constraints**: Built-in PINN capabilities
- **Uncertainty Quantification**: Bayesian neural ODEs

### Research Directions

- **Fractional PDEs**: Extension to partial differential equations
- **Graph Neural ODEs**: Dynamic graph evolution
- **Control Systems**: Optimal control with neural ODEs
- **Multi-physics**: Coupled physical systems

## 📖 **References**

1. Chen, R. T. Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
2. Podlubny, I. "Fractional Differential Equations." Academic Press, 1999.
3. Raissi, M., et al. "Physics Informed Deep Learning." JCP 2019.

## 🤝 **Contributing**

We welcome contributions to the Neural fODE framework! Areas for contribution include:

- **New Solvers**: Implementation of additional ODE solvers
- **Performance**: Optimization and GPU acceleration
- **Examples**: Additional tutorials and use cases
- **Documentation**: Improvements to this guide
- **Testing**: Additional test cases and validation

## 📞 **Support**

For questions and support:

- **Documentation**: This guide and the main HPFRACC documentation
- **GitHub Issues**: Report bugs and request features
- **Academic Contact**: d.r.chin@pgr.reading.ac.uk

---

**Neural fODE Framework v1.4.0** - *Empowering Research with Learning-Based Fractional Differential Equation Solving*
