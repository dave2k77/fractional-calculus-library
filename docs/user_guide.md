# HPFRACC User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Fractional Calculus](#basic-fractional-calculus)
4. [Core Features](#core-features)
5. [Machine Learning Integration](#machine-learning-integration)
6. [Advanced Usage](#advanced-usage)
7. [Configuration and Settings](#configuration-and-settings)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (for ML features)
- CUDA (optional, for GPU acceleration)

### Basic Installation
```bash
pip install hpfracc
```

### Full Installation with ML Dependencies
```bash
pip install hpfracc[ml]
```

### Development Installation
```bash
pip install hpfracc[dev]
```

---

## Quick Start

### Basic Fractional Calculus Operations

```python
import numpy as np
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.derivatives import create_fractional_derivative
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.special import gamma, beta, mittag_leffler

# Create fractional order
alpha = FractionalOrder(0.5)

# Create derivative operator
derivative = create_fractional_derivative(alpha, method="RL")

# Create integral operator
integral = create_fractional_integral(alpha, method="RL")

# Define a function
def f(x):
    return x**2

# Compute fractional derivative
x = np.linspace(0, 5, 100)
result_derivative = derivative(f, x)

# Compute fractional integral
result_integral = integral(f, x)

# Compute special functions
gamma_val = gamma(5.5)
beta_val = beta(2.5, 3.5)
ml_val = mittag_leffler(0.5, 1.0, 2.0)
```

### Simple Fractional Neural Network
```python
import torch
from hpfracc.ml import FractionalNeuralNetwork

# Create network
net = FractionalNeuralNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5
)

# Forward pass
x = torch.randn(32, 100)  # batch_size=32, input_size=100
output = net(x)
print(f"Output shape: {output.shape}")
```

---

## Basic Fractional Calculus

### Fractional Derivatives

Fractional derivatives extend the concept of integer-order derivatives to non-integer orders. The library supports several methods:

- **Riemann-Liouville (RL)**: Most general, works for 0 < α < 2
- **Caputo**: Better for initial value problems, works for 0 < α < 1
- **Grünwald-Letnikov (GL)**: Numerical approximation, works for 0 < α < 2
- **Weyl**: For periodic functions
- **Marchaud**: For functions with specific decay properties
- **Hadamard**: Logarithmic fractional derivative

```python
from hpfracc.core.derivatives import create_fractional_derivative

# For general purposes
derivative_rl = create_fractional_derivative(0.5, method="RL")

# For initial value problems
derivative_caputo = create_fractional_derivative(0.3, method="Caputo")

# Compute derivatives
def f(x):
    return x**2

x = np.linspace(0, 5, 100)
result_rl = derivative_rl(f, x)
result_caputo = derivative_caputo(f, x)
```

### Fractional Integrals

The library supports various fractional integral definitions:

- **Riemann-Liouville**: Most general fractional integral
- **Caputo**: Related to Caputo derivative
- **Weyl**: For periodic functions
- **Hadamard**: Logarithmic fractional integral

```python
from hpfracc.core.integrals import create_fractional_integral

# Create integral operators
integral_rl = create_fractional_integral(0.5, method="RL")
integral_caputo = create_fractional_integral(0.5, method="Caputo")
integral_weyl = create_fractional_integral(0.5, method="Weyl")
integral_hadamard = create_fractional_integral(0.5, method="Hadamard")

# Compute integrals
def f(x):
    return x**2

x = np.linspace(0, 5, 100)
result_rl = integral_rl(f, x)
result_caputo = integral_caputo(f, x)
```

### Special Functions

The library provides comprehensive special function implementations:

```python
from hpfracc.special import gamma, beta, mittag_leffler, binomial

# Gamma function
gamma_val = gamma(5.5)

# Beta function
beta_val = beta(2.5, 3.5)

# Mittag-Leffler function (one-parameter)
ml_1 = mittag_leffler(0.5, 1.0)

# Mittag-Leffler function (two-parameter)
ml_2 = mittag_leffler(0.5, 1.0, 2.0)

# Binomial coefficients
binom_std = binomial(10, 5)
binom_frac = binomial(10.5, 5.2)
```

---

## Core Features

### Fractional Derivatives

The library provides a unified interface for fractional derivatives:

```python
from hpfracc.core.derivatives import create_fractional_derivative

# Create derivative operator
derivative = create_fractional_derivative(0.5, method="RL")

# Use with different functions
def f1(x):
    return x**2

def f2(x):
    return np.sin(x)

x = np.linspace(0, 5, 100)
result1 = derivative(f1, x)
result2 = derivative(f2, x)
```

### Fractional Integrals

Comprehensive fractional integral support:

```python
from hpfracc.core.integrals import create_fractional_integral

# Create integral operators for different methods
integral_rl = create_fractional_integral(0.5, method="RL")
integral_caputo = create_fractional_integral(0.5, method="Caputo")
integral_weyl = create_fractional_integral(0.5, method="Weyl")
integral_hadamard = create_fractional_integral(0.5, method="Hadamard")

# Note: Hadamard integral requires x > 1
x_hadamard = np.linspace(1.1, 5, 100)
result_hadamard = integral_hadamard(lambda x: x**2, x_hadamard)
```

### Special Functions

Complete special function library:

```python
from hpfracc.special import gamma, beta, mittag_leffler, binomial

# Gamma function with validation
gamma_val = gamma(5.5)

# Beta function
beta_val = beta(2.5, 3.5)

# Mittag-Leffler functions
ml_1 = mittag_leffler(0.5, 1.0)  # One-parameter
ml_2 = mittag_leffler(0.5, 1.0, 2.0)  # Two-parameter

# Binomial coefficients
binom_std = binomial(10, 5)  # Standard
binom_frac = binomial(10.5, 5.2)  # Fractional
```

### Fractional Green's Functions

Green's functions for fractional differential equations:

```python
from hpfracc.special.greens_function import (
    FractionalDiffusionGreenFunction,
    FractionalWaveGreenFunction,
    FractionalAdvectionGreenFunction
)

# Diffusion Green's function
diffusion_gf = FractionalDiffusionGreenFunction(alpha=0.5, D=1.0)
x, t = np.meshgrid(np.linspace(0, 5, 50), np.linspace(0, 2, 20))
G_diffusion = diffusion_gf(x, t)

# Wave Green's function
wave_gf = FractionalWaveGreenFunction(alpha=1.5, c=1.0)
G_wave = wave_gf(x, t)

# Advection Green's function
advection_gf = FractionalAdvectionGreenFunction(alpha=0.7, v=1.0)
G_advection = advection_gf(x, t)
```

### Analytical Methods

Homotopy Perturbation Method (HPM) and Variational Iteration Method (VIM):

```python
from hpfracc.solvers import HomotopyPerturbationSolver, VariationalIterationSolver

# HPM solver
hpm_solver = HomotopyPerturbationSolver()

# Define the equation: D^α u + u = f(x,t)
def equation(x, t, u):
    return u(x, t) + x**2 + t

def initial_condition(x):
    return x

# Solve with HPM
solution_hpm = hpm_solver.solve(equation, initial_condition)

# VIM solver
vim_solver = VariationalIterationSolver()

# Solve with VIM
solution_vim = vim_solver.solve(equation, initial_condition)
```

### Mathematical Utilities

Comprehensive mathematical utilities:

```python
from hpfracc.core.utilities import (
    validate_fractional_order,
    validate_function,
    factorial_fractional,
    binomial_coefficient,
    pochhammer_symbol,
    timing_decorator,
    memory_usage_decorator
)

# Validation functions
is_valid = validate_fractional_order(0.5)
is_valid_func = validate_function(lambda x: x**2)

# Mathematical functions
fact_frac = factorial_fractional(5.5)
binom_coeff = binomial_coefficient(10, 5)
poch = pochhammer_symbol(5.5, 3)

# Performance monitoring
@timing_decorator
@memory_usage_decorator
def my_function(x):
    return x**2
```

---

## Machine Learning Integration

### Fractional Neural Networks

```python
import torch
from hpfracc.ml import FractionalNeuralNetwork

# Create fractional neural network
net = FractionalNeuralNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5,
    activation='relu'
)

# Training
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Fit the model
x_train = torch.randn(1000, 100)
y_train = torch.randint(0, 10, (1000,))

net.fit(x_train, y_train, optimizer, criterion, epochs=100)

# Predictions
x_test = torch.randn(100, 100)
predictions = net.predict(x_test)
```

### Graph Neural Networks

```python
import torch
from hpfracc.ml import FractionalGraphConvolution

# Create fractional graph convolution layer
conv_layer = FractionalGraphConvolution(
    in_channels=10,
    out_channels=32,
    fractional_order=0.5
)

# Apply to graph data
x = torch.randn(100, 10)  # Node features
edge_index = torch.randint(0, 100, (2, 200))  # Edge connections
output = conv_layer(x, edge_index)
```

---

## Advanced Usage

### Error Analysis and Validation

```python
from hpfracc.core.utilities import validate_fractional_order
from hpfracc.analytics import analyze_convergence, estimate_error

# Validate parameters
is_valid = validate_fractional_order(0.5)

# Analyze convergence
convergence_analysis = analyze_convergence(solutions, analytical_solution)

# Estimate error
error_estimate = estimate_error(numerical_solution, analytical_solution)
```

### Performance Optimization

```python
from hpfracc.core.backend_manager import BackendManager

# Switch backends
backend_manager = BackendManager()
backend_manager.set_backend('pytorch')  # or 'jax', 'numba'

# GPU acceleration (if available)
if torch.cuda.is_available():
    device = torch.device('cuda')
    model = model.to(device)
```

### Signal Processing

```python
import numpy as np
from hpfracc.core.derivatives import create_fractional_derivative

# Fractional filtering
derivative = create_fractional_derivative(0.5, method="RL")

# Apply to signal
signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
filtered_signal = derivative(lambda x: signal[int(x * len(signal))], 
                           np.linspace(0, 1, 1000))
```

### Image Processing

```python
import torch
from hpfracc.ml import FractionalNeuralNetwork

# Fractional image processing
net = FractionalNeuralNetwork(
    input_size=784,  # 28x28 image
    hidden_sizes=[512, 256],
    output_size=10,
    fractional_order=0.5
)

# Process image
image = torch.randn(1, 784)  # Flattened image
processed = net(image)
```

---

## Configuration and Settings

### Precision Settings

```python
from hpfracc.core.utilities import set_precision

# Set precision for computations
set_precision('double')  # or 'single', 'extended'
```

### Logging Configuration

```python
import logging
from hpfracc.core.utilities import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
```

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Use batch processing for large datasets
3. **GPU Memory**: Implement gradient checkpointing
4. **Numerical Issues**: Use higher precision for critical calculations

### Performance Issues

- Use GPU acceleration when available
- Implement batch processing for large datasets
- Use appropriate backend for your use case
- Monitor memory usage with `memory_usage_decorator`

---

## Best Practices

### Code Organization

- Use validation functions for input parameters
- Implement proper error handling
- Use performance monitoring decorators
- Follow the library's API conventions

### Performance

- Choose appropriate fractional order and method
- Use GPU acceleration for large computations
- Implement batch processing for ML workflows
- Monitor memory usage and optimize accordingly

### Validation

- Always validate fractional orders
- Compare with analytical solutions when available
- Use convergence analysis for iterative methods
- Implement comprehensive testing

---

For more detailed information, see the [API Reference](api_reference.md) and [Examples](examples.md).
