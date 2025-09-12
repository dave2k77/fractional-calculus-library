# Neural ODE Usage Guide

## Quick Start

### Standard Neural ODE
```python
from hpfracc.ml.neural_ode import NeuralODE

# Create neural ODE
model = NeuralODE(input_dim=2, hidden_dim=64, output_dim=2, num_layers=3)

# Solve ODE
x = torch.randn(32, 2)  # Initial conditions
t = torch.linspace(0, 1, 10)  # Time points
solution = model(x, t)  # Shape: (32, 10, 2)
```

### Fractional Neural ODE
```python
from hpfracc.ml.neural_ode import NeuralFODE

# Create fractional neural ODE
model = NeuralFODE(input_dim=2, hidden_dim=64, output_dim=2, 
                   fractional_order=0.5, num_layers=3)

# Solve fractional ODE
solution = model(x, t)  # Shape: (32, 10, 2)
```

### Training
```python
from hpfracc.ml.neural_ode import NeuralODETrainer

# Create trainer
trainer = NeuralODETrainer(model, optimizer='adam', learning_rate=1e-3)

# Training step
loss = trainer.train_step(x, y_target, t)
```

## Performance Characteristics

- **NeuralODE**: Best performance (0.002884s average)
- **NeuralFODE**: Good performance with fractional calculus (0.003459s average)
- **Research implementations**: Advanced features, slower performance

## Use Case Guidelines

- **Production**: Use NeuralODE or NeuralFODE
- **Research**: Use implementations in `hpfracc.ml.research`
- **Fractional calculus**: Use NeuralFODE
- **Standard ODEs**: Use NeuralODE

## Author

Davian R. Chin, Department of Biomedical Engineering, University of Reading
