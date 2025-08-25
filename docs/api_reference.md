# HPFRACC API Reference

## Table of Contents
1. [Core Module](#core-module)
2. [Special Functions](#special-functions)
3. [Analytical Methods](#analytical-methods)
4. [ML Module](#ml-module)
5. [Graph Neural Networks](#graph-neural-networks)
6. [Benchmarks Module](#benchmarks-module)
7. [Analytics Module](#analytics-module)

---

## Core Module

### Fractional Derivatives

#### `create_fractional_derivative(alpha, method="RL")`
Create a fractional derivative operator.

**Parameters:**
- `alpha` (FractionalOrder): Fractional order
- `method` (str): Derivative method ("RL", "Caputo", "GL", "Weyl", "Marchaud", "Hadamard")

**Returns:**
- `BaseFractionalDerivative`: Fractional derivative operator

**Example:**
```python
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.derivatives import create_fractional_derivative

alpha = FractionalOrder(0.5)
deriv = create_fractional_derivative(alpha, method="RL")
result = deriv(lambda x: x**2, np.linspace(0, 5, 100))
```

#### `FractionalDerivativeOperator`
Base class for fractional derivative operators.

**Methods:**
- `__call__(f, x)`: Compute fractional derivative of function f at points x
- `compute(f, x)`: Alias for `__call__`

### Fractional Integrals

#### `create_fractional_integral(alpha, method="RL")`
Create a fractional integral operator.

**Parameters:**
- `alpha` (FractionalOrder): Fractional order
- `method` (str): Integral method ("RL", "Caputo", "Weyl", "Hadamard")

**Returns:**
- `FractionalIntegral`: Fractional integral operator

**Example:**
```python
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.integrals import create_fractional_integral

alpha = FractionalOrder(0.5)
integral = create_fractional_integral(alpha, method="RL")
result = integral(lambda x: x**2, np.linspace(0, 5, 100))
```

#### `FractionalIntegral`
Base class for fractional integral operators.

**Methods:**
- `__call__(f, x)`: Compute fractional integral of function f at points x
- `compute(f, x)`: Alias for `__call__`

### Fractional Order

#### `FractionalOrder(alpha)`
Class representing fractional order with validation.

**Parameters:**
- `alpha` (float): Fractional order value

**Attributes:**
- `alpha` (float): The fractional order value
- `is_valid` (bool): Whether the order is valid

**Methods:**
- `validate()`: Validate the fractional order
- `__str__()`: String representation

### Mathematical Utilities

#### `validate_fractional_order(alpha)`
Validate if a fractional order is within valid range.

**Parameters:**
- `alpha` (float): Fractional order to validate

**Returns:**
- `bool`: True if valid, False otherwise

#### `validate_function(f)`
Validate if input is a callable function.

**Parameters:**
- `f`: Function to validate

**Returns:**
- `bool`: True if callable, False otherwise

#### `factorial_fractional(x)`
Compute fractional factorial.

**Parameters:**
- `x` (float): Input value

**Returns:**
- `float`: Fractional factorial result

#### `binomial_coefficient(n, k)`
Compute binomial coefficient.

**Parameters:**
- `n` (int): Upper parameter
- `k` (int): Lower parameter

**Returns:**
- `int`: Binomial coefficient

#### `pochhammer_symbol(a, n)`
Compute Pochhammer symbol.

**Parameters:**
- `a` (float): Base parameter
- `n` (int): Order parameter

**Returns:**
- `float`: Pochhammer symbol result

#### `timing_decorator(func)`
Decorator to measure function execution time.

**Parameters:**
- `func` (callable): Function to decorate

**Returns:**
- `callable`: Decorated function with timing

#### `memory_usage_decorator(func)`
Decorator to measure function memory usage.

**Parameters:**
- `func` (callable): Function to decorate

**Returns:**
- `callable`: Decorated function with memory monitoring

---

## Special Functions

### Gamma and Beta Functions

#### `gamma_function(x)`
Compute gamma function.

**Parameters:**
- `x` (float): Input value

**Returns:**
- `float`: Gamma function value

#### `beta_function(a, b)`
Compute beta function.

**Parameters:**
- `a` (float): First parameter
- `b` (float): Second parameter

**Returns:**
- `float`: Beta function value

#### `incomplete_gamma(a, x)`
Compute incomplete gamma function.

**Parameters:**
- `a` (float): Shape parameter
- `x` (float): Upper limit

**Returns:**
- `float`: Incomplete gamma function value

#### `incomplete_beta(a, b, x)`
Compute incomplete beta function.

**Parameters:**
- `a` (float): First parameter
- `b` (float): Second parameter
- `x` (float): Upper limit

**Returns:**
- `float`: Incomplete beta function value

### Mittag-Leffler Functions

#### `mittag_leffler_function(alpha, z)`
Compute one-parameter Mittag-Leffler function.

**Parameters:**
- `alpha` (float): Parameter
- `z` (float): Argument

**Returns:**
- `float`: Mittag-Leffler function value

#### `mittag_leffler_derivative(alpha, z, n=1)`
Compute derivative of Mittag-Leffler function.

**Parameters:**
- `alpha` (float): Parameter
- `z` (float): Argument
- `n` (int): Derivative order

**Returns:**
- `float`: Derivative value

### Binomial Coefficients

#### `binomial_coefficient(n, k)`
Compute standard binomial coefficient.

**Parameters:**
- `n` (int): Upper parameter
- `k` (int): Lower parameter

**Returns:**
- `int`: Binomial coefficient

#### `generalized_binomial(alpha, k)`
Compute fractional binomial coefficient.

**Parameters:**
- `alpha` (float): Fractional parameter
- `k` (int): Lower parameter

**Returns:**
- `float`: Fractional binomial coefficient

### Fractional Green's Functions

#### `FractionalDiffusionGreensFunction(alpha, D)`
Green's function for fractional diffusion equation.

**Parameters:**
- `alpha` (float): Fractional order
- `D` (float): Diffusion coefficient

**Methods:**
- `compute(x, t)`: Compute Green's function at position x and time t

#### `FractionalWaveGreensFunction(alpha, c)`
Green's function for fractional wave equation.

**Parameters:**
- `alpha` (float): Fractional order
- `c` (float): Wave speed

**Methods:**
- `compute(x, t)`: Compute Green's function at position x and time t

#### `FractionalAdvectionGreensFunction(alpha, v)`
Green's function for fractional advection equation.

**Parameters:**
- `alpha` (float): Fractional order
- `v` (float): Advection velocity

**Methods:**
- `compute(x, t)`: Compute Green's function at position x and time t

---

## Analytical Methods

### Homotopy Perturbation Method (HPM)

#### `HomotopyPerturbationMethod(alpha)`
Solver for fractional differential equations using HPM.

**Parameters:**
- `alpha` (float): Fractional order

**Methods:**
- `solve(source_function, initial_condition, t_span, max_iterations=5)`: Solve the equation
- `analyze_convergence(source_function, initial_condition, t_span, max_iterations=10)`: Analyze convergence

**Example:**
```python
from hpfracc.solvers.homotopy_perturbation import HomotopyPerturbationMethod

def source_function(t):
    return t**2

def initial_condition(t):
    return 0.0

alpha = 0.5
hpm_solver = HomotopyPerturbationMethod(alpha)
t = np.linspace(0, 2, 100)
solution = hpm_solver.solve(source_function, initial_condition, t)
```

### Variational Iteration Method (VIM)

#### `VariationalIterationMethod(alpha)`
Solver for fractional differential equations using VIM.

**Parameters:**
- `alpha` (float): Fractional order

**Methods:**
- `solve(source_function, initial_condition, nonlinear_term, t_span, max_iterations=5)`: Solve the equation
- `analyze_convergence(source_function, initial_condition, t_span, max_iterations=10)`: Analyze convergence

**Example:**
```python
from hpfracc.solvers.variational_iteration import VariationalIterationMethod

def source_function(t):
    return np.ones_like(t)

def initial_condition(t):
    return 0.0

def nonlinear_term(u):
    return u**2

alpha = 0.5
vim_solver = VariationalIterationMethod(alpha)
t = np.linspace(0, 2, 100)
solution = vim_solver.solve(source_function, initial_condition, nonlinear_term, t)
```

---

## ML Module

### Neural Networks

#### `FractionalNeuralNetwork(input_dim, hidden_dims, output_dim, fractional_order=0.5)`
Standard fractional neural network.

**Parameters:**
- `input_dim` (int): Input dimension
- `hidden_dims` (List[int]): Hidden layer dimensions
- `output_dim` (int): Output dimension
- `fractional_order` (float): Fractional order for derivatives

**Methods:**
- `forward(x)`: Forward pass
- `fit(X, y, epochs=100, batch_size=32, learning_rate=0.001)`: Train the model
- `predict(X)`: Make predictions

**Example:**
```python
from hpfracc.ml import FractionalNeuralNetwork
from hpfracc.core.definitions import FractionalOrder

model = FractionalNeuralNetwork(
    input_dim=10,
    hidden_dims=[64, 32, 16],
    output_dim=1,
    fractional_order=FractionalOrder(0.5)
)

# Train the model
history = model.fit(X_train, y_train, epochs=100)

# Make predictions
predictions = model.predict(X_test)
```

### Graph Neural Networks

#### `FractionalGraphConvolution(input_dim, output_dim, fractional_order, activation='relu')`
Fractional graph convolution layer.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `output_dim` (int): Output feature dimension
- `fractional_order` (FractionalOrder): Fractional order
- `activation` (str): Activation function

**Methods:**
- `forward(adj_matrix, node_features)`: Apply fractional graph convolution

**Example:**
```python
from hpfracc.ml.gnn_layers import FractionalGraphConvolution
from hpfracc.core.definitions import FractionalOrder

fgc_layer = FractionalGraphConvolution(
    input_dim=5,
    output_dim=3,
    fractional_order=FractionalOrder(0.5),
    activation='relu'
)

output_features = fgc_layer(adj_matrix, node_features)
```

### Backend Management

#### `BackendManager`
Manager for computation backends.

**Class Methods:**
- `set_backend(backend_type)`: Set the current backend
- `get_current_backend()`: Get the current backend
- `get_available_backends()`: Get list of available backends
- `is_backend_available(backend_type)`: Check if backend is available

**Example:**
```python
from hpfracc.ml.backends import BackendManager, BackendType

# Set backend
BackendManager.set_backend(BackendType.TORCH)

# Check available backends
available = BackendManager.get_available_backends()
print(f"Available backends: {available}")
```

---

## Benchmarks Module

### Performance Testing

#### `benchmark_fractional_derivatives(data_sizes, methods)`
Benchmark fractional derivative performance.

**Parameters:**
- `data_sizes` (List[int]): List of data sizes to test
- `methods` (List[str]): List of methods to benchmark

**Returns:**
- `dict`: Performance results

#### `benchmark_memory_usage(data_sizes)`
Benchmark memory usage for different data sizes.

**Parameters:**
- `data_sizes` (List[int]): List of data sizes to test

**Returns:**
- `dict`: Memory usage results

#### `benchmark_gpu_acceleration(data_sizes)`
Benchmark GPU acceleration performance.

**Parameters:**
- `data_sizes` (List[int]): List of data sizes to test

**Returns:**
- `dict`: GPU performance results

---

## Analytics Module

### Performance Monitoring

#### `PerformanceMonitor`
Monitor performance metrics.

**Methods:**
- `timer(name)`: Context manager for timing operations
- `get_timing(name)`: Get timing for specific operation
- `get_all_timings()`: Get all recorded timings
- `reset()`: Reset all timings

**Example:**
```python
from hpfracc.core.utilities import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.timer("computation"):
    result = expensive_computation(data)

print(f"Computation time: {monitor.get_timing('computation')}")
```

### Error Analysis

#### `analyze_numerical_error(analytical, numerical)`
Analyze numerical error between analytical and numerical solutions.

**Parameters:**
- `analytical` (np.ndarray): Analytical solution
- `numerical` (np.ndarray): Numerical solution

**Returns:**
- `dict`: Error analysis results

#### `analyze_convergence(residuals)`
Analyze convergence of iterative methods.

**Parameters:**
- `residuals` (List[float]): List of residuals

**Returns:**
- `dict`: Convergence analysis results

---

## Configuration

### Precision Settings

#### `get_default_precision()`
Get the default numerical precision.

**Returns:**
- `int`: Precision in bits

#### `set_default_precision(precision)`
Set the default numerical precision.

**Parameters:**
- `precision` (int): Precision in bits

#### `get_available_methods()`
Get list of available methods.

**Returns:**
- `List[str]`: Available methods

#### `get_method_properties(method)`
Get properties of a specific method.

**Parameters:**
- `method` (str): Method name

**Returns:**
- `dict`: Method properties

### Logging

#### `setup_logging(level="INFO", log_file=None)`
Setup logging configuration.

**Parameters:**
- `level` (str): Logging level
- `log_file` (str): Log file path

**Returns:**
- `logging.Logger`: Configured logger

#### `get_logger(name)`
Get logger for specific module.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Logger instance

---

## Error Handling

### Common Exceptions

#### `FractionalOrderError`
Raised for invalid fractional orders.

#### `MethodNotSupportedError`
Raised when method is not supported.

#### `BackendNotAvailableError`
Raised when backend is not available.

#### `ValidationError`
Raised for validation failures.

### Error Handling Examples

```python
from hpfracc.core.utilities import validate_fractional_order

try:
    if not validate_fractional_order(alpha):
        raise ValueError(f"Invalid fractional order: {alpha}")
    
    # Perform computation
    result = compute_fractional_derivative(f, x, alpha)
    
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Best Practices

### Performance Optimization

1. **Use appropriate backends**: Choose the best backend for your use case
2. **Batch processing**: Process data in batches for large datasets
3. **Memory management**: Monitor memory usage and optimize when needed
4. **GPU acceleration**: Use GPU acceleration for large-scale computations

### Code Organization

1. **Import organization**: Organize imports logically
2. **Error handling**: Use proper error handling and validation
3. **Documentation**: Document your code with clear docstrings
4. **Testing**: Write comprehensive tests for your code

### Validation

1. **Input validation**: Always validate inputs before computation
2. **Numerical stability**: Check for numerical stability issues
3. **Convergence analysis**: Analyze convergence for iterative methods
4. **Error estimation**: Estimate and monitor numerical errors

This comprehensive API reference covers all the major features of the HPFRACC library, from core fractional calculus operations to advanced machine learning integration and analytical methods.
