# HPFRACC API Reference (Updated)

## Core Module (`hpfracc`)

### Main Functions

#### `fractional_derivative(x, alpha, method="caputo")`
Compute fractional derivative with automatic backend selection.

**Parameters:**
- `x` (array-like): Input data
- `alpha` (float): Fractional order (0 < alpha < 2)
- `method` (str): Method to use ("caputo", "riemann_liouville", "grunwald_letnikov")

**Returns:**
- `array`: Fractional derivative values

**Example:**
```python
import hpfracc as hpc
import numpy as np

x = np.linspace(0, 10, 100)
result = hpc.fractional_derivative(x, alpha=0.5, method="caputo")
```

#### `fractional_integral(x, alpha, method="riemann_liouville")`
Compute fractional integral with automatic backend selection.

**Parameters:**
- `x` (array-like): Input data
- `alpha` (float): Fractional order (0 < alpha < 2)
- `method` (str): Method to use ("riemann_liouville", "caputo")

**Returns:**
- `array`: Fractional integral values

---

## Core Definitions (`hpfracc.core.definitions`)

### `FractionalOrder`
Represents a fractional order with validation and properties.

**Constructor:**
```python
FractionalOrder(alpha: float, definition_type: str = "caputo")
```

**Properties:**
- `alpha`: The fractional order value
- `definition_type`: The definition type ("caputo", "riemann_liouville", "grunwald_letnikov")
- `is_valid`: Whether the order is in valid range (0 < alpha < 2)

**Methods:**
- `validate()`: Validate the fractional order
- `get_properties()`: Get mathematical properties
- `get_definition_formula()`: Get the mathematical formula

### `DefinitionType`
Enumeration of available fractional derivative definitions.

**Values:**
- `CAPUTO`: Caputo fractional derivative
- `RIEMANN_LIOUVILLE`: Riemann-Liouville fractional derivative
- `GRUNWALD_LETNIKOV`: Grünwald-Letnikov fractional derivative

---

## Core Derivatives (`hpfracc.core.derivatives`)

### `BaseFractionalDerivative`
Abstract base class for fractional derivative implementations.

**Constructor:**
```python
BaseFractionalDerivative(
    alpha: Union[float, FractionalOrder],
    definition: Optional[FractionalDefinition] = None,
    use_jax: bool = False,
    use_numba: bool = True
)
```

**Methods:**
- `compute(func, x, **kwargs)`: Compute the fractional derivative
- `get_definition_info()`: Get definition information
- `validate_parameters()`: Validate input parameters

### `create_fractional_derivative()`
Factory function to create fractional derivative instances.

**Parameters:**
- `definition_type`: Type of derivative ("caputo", "riemann_liouville", "grunwald_letnikov")
- `alpha`: Fractional order
- `use_jax`: Whether to use JAX backend
- `use_numba`: Whether to use NUMBA optimization

**Returns:**
- `BaseFractionalDerivative`: Derivative instance

---

## Fractional Implementations (`hpfracc.core.fractional_implementations`)

### `RiemannLiouvilleDerivative`
Riemann-Liouville fractional derivative implementation.

**Constructor:**
```python
RiemannLiouvilleDerivative(alpha: Union[float, FractionalOrder])
```

**Methods:**
- `compute(func, x)`: Compute Riemann-Liouville derivative
- `get_definition_info()`: Get definition information

### `CaputoDerivative`
Caputo fractional derivative implementation.

**Constructor:**
```python
CaputoDerivative(alpha: Union[float, FractionalOrder])
```

### `GrunwaldLetnikovDerivative`
Grünwald-Letnikov fractional derivative implementation.

**Constructor:**
```python
GrunwaldLetnikovDerivative(alpha: Union[float, FractionalOrder])
```

---

## Machine Learning Module (`hpfracc.ml`)

### Spectral Autograd Framework

#### `SpectralFractionalDerivative`
Spectral fractional derivative with automatic differentiation support.

**Usage:**
```python
import torch
from hpfracc.ml import SpectralFractionalDerivative

x = torch.randn(32, requires_grad=True)
alpha = 0.5
result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
```

**Parameters:**
- `x`: Input tensor
- `alpha`: Fractional order
- `axis`: Axis along which to compute derivative (-1 for last axis)
- `method`: Spectral method ("fft", "mellin")

#### `BoundedAlphaParameter`
Learnable fractional order parameter with bounds.

**Constructor:**
```python
BoundedAlphaParameter(
    alpha_init: float = 1.0,
    alpha_min: float = 0.1,
    alpha_max: float = 1.9
)
```

**Methods:**
- `__call__()`: Get current alpha value
- `get_alpha()`: Get alpha value
- `set_alpha(alpha)`: Set alpha value

### Neural Network Layers

#### `FractionalLSTM`
Fractional Long Short-Term Memory layer.

**Constructor:**
```python
FractionalLSTM(
    input_size: int,
    hidden_size: int,
    alpha: float = 0.5,
    num_layers: int = 1,
    batch_first: bool = True
)
```

**Methods:**
- `forward(x, return_state=False)`: Forward pass

#### `FractionalPooling`
Fractional pooling layer.

**Constructor:**
```python
FractionalPooling(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    alpha: float = 0.5
)
```

#### `SpectralFractionalLayer`
Spectral fractional layer for neural networks.

**Constructor:**
```python
SpectralFractionalLayer(alpha: float = 0.5)
```

### Backend Management

#### `BackendManager`
Manages computation backends (PyTorch, JAX, NUMBA).

**Methods:**
- `set_backend(backend_type)`: Set active backend
- `get_current_backend()`: Get current backend
- `get_available_backends()`: Get available backends
- `is_backend_available(backend_type)`: Check if backend is available

#### `BackendType`
Enumeration of available backends.

**Values:**
- `TORCH`: PyTorch backend
- `JAX`: JAX backend
- `NUMBA`: NUMBA backend
- `NUMPY`: NumPy backend

---

## Special Functions (`hpfracc.special`)

### `gamma_function(x)`
Compute gamma function.

**Parameters:**
- `x`: Input value

**Returns:**
- `float`: Gamma function value

### `beta_function(a, b)`
Compute beta function.

**Parameters:**
- `a`, `b`: Input parameters

**Returns:**
- `float`: Beta function value

### `binomial_coefficient(n, k)`
Compute binomial coefficient.

**Parameters:**
- `n`, `k`: Input parameters

**Returns:**
- `int`: Binomial coefficient

### `mittag_leffler_function(alpha, z, beta=1.0)`
Compute Mittag-Leffler function.

**Parameters:**
- `alpha`: First parameter
- `z`: Argument
- `beta`: Second parameter (default: 1.0)

**Returns:**
- `float`: Mittag-Leffler function value

---

## Validation Module (`hpfracc.validation`)

### Analytical Solutions

#### `get_analytical_solution(func_type, x, **params)`
Get analytical solution for common functions.

**Parameters:**
- `func_type`: Function type ("power", "exponential", "trigonometric")
- `x`: Input array
- `**params`: Function parameters

**Returns:**
- `array`: Analytical solution

#### `validate_against_analytical(numerical_func, analytical_func, test_params)`
Validate numerical method against analytical solution.

**Parameters:**
- `numerical_func`: Numerical method function
- `analytical_func`: Analytical solution function
- `test_params`: Test parameters

**Returns:**
- `dict`: Validation results

### Convergence Tests

#### `ConvergenceTester`
Test convergence of numerical methods.

**Methods:**
- `test_convergence(numerical_func, analytical_func, grid_sizes, test_params)`
- `test_multiple_norms(numerical_func, analytical_func, grid_sizes, test_params)`

#### `run_convergence_study(numerical_func, analytical_func, test_cases, grid_sizes)`
Run comprehensive convergence study.

### Benchmarks

#### `BenchmarkSuite`
Comprehensive benchmark suite.

**Constructor:**
```python
BenchmarkSuite(tolerance: float = 1e-10, warmup_runs: int = 3)
```

**Methods:**
- `run_comprehensive_benchmark(methods, analytical_func, test_cases, n_runs=10)`

#### `PerformanceBenchmark`
Performance benchmarking.

**Methods:**
- `benchmark_method(method_func, test_params, n_runs=10)`

#### `AccuracyBenchmark`
Accuracy benchmarking.

**Methods:**
- `benchmark_method(numerical_func, analytical_func, test_params)`

---

## Utilities (`hpfracc.utils`)

### Error Analysis

#### `ErrorAnalyzer`
Analyze numerical errors.

**Methods:**
- `compute_all_errors(numerical, analytical)`
- `compute_l2_error(numerical, analytical)`
- `compute_linf_error(numerical, analytical)`

### Memory Management

#### `MemoryManager`
Manage memory usage.

**Constructor:**
```python
MemoryManager(max_memory_gb: float = 8.0)
```

**Methods:**
- `memory_context()`: Memory context manager
- `get_memory_usage()`: Get current memory usage
- `check_memory_limit()`: Check memory limits

### Performance Monitoring

#### `PerformanceMonitor`
Monitor performance metrics.

**Methods:**
- `timer(name)`: Timing context manager
- `get_timing(name)`: Get timing results
- `get_memory_usage()`: Get memory usage

---

## Core Utilities (`hpfracc.core.utilities`)

### Validation Functions

#### `validate_fractional_order(alpha)`
Validate fractional order.

**Parameters:**
- `alpha`: Fractional order to validate

**Returns:**
- `bool`: Whether order is valid

#### `validate_function(func)`
Validate function.

**Parameters:**
- `func`: Function to validate

**Returns:**
- `bool`: Whether function is valid

#### `validate_tensor_input(tensor)`
Validate tensor input.

**Parameters:**
- `tensor`: Tensor to validate

**Returns:**
- `bool`: Whether tensor is valid

### Mathematical Functions

#### `factorial_fractional(x)`
Compute fractional factorial.

#### `pochhammer_symbol(a, n)`
Compute Pochhammer symbol.

#### `hypergeometric_series(a, b, c, z)`
Compute hypergeometric series.

### Performance Utilities

#### `timing_decorator`
Decorator for timing functions.

#### `memory_usage_decorator`
Decorator for monitoring memory usage.

---

## Solver Module (`hpfracc.solvers`)

### `FractionalODESolver`
Solver for fractional ordinary differential equations.

**Constructor:**
```python
FractionalODESolver(method: str = "predictor_corrector")
```

**Methods:**
- `solve(ode_func, t_span, y0, alpha)`: Solve fractional ODE

### `PredictorCorrectorSolver`
Predictor-corrector solver for fractional ODEs.

**Constructor:**
```python
PredictorCorrectorSolver(derivative_type: str, order: float)
```

**Methods:**
- `solve(ode_func, t_span, y0)`: Solve fractional ODE

---

## Advanced Methods (`hpfracc.algorithms`)

### Optimized Methods (`hpfracc.algorithms.optimized_methods`)
High-performance implementations of fractional calculus methods.

### Advanced Methods (`hpfracc.algorithms.advanced_methods`)
Advanced numerical methods for fractional calculus.

### Special Methods (`hpfracc.algorithms.special_methods`)
Specialized methods for specific applications.

---

## Configuration

### Backend Configuration
```python
from hpfracc.ml.backends import BackendManager, BackendType

# Set preferred backend
BackendManager.set_backend(BackendType.TORCH)

# Check availability
available = BackendManager.get_available_backends()
```

### Precision Settings
```python
from hpfracc.core.utilities import set_default_precision

# Set precision
set_default_precision(64)  # 64-bit precision
```

### Logging Configuration
```python
from hpfracc.core.utilities import setup_logging

# Setup logging
logger = setup_logging(level="INFO", log_file="hpfracc.log")
```

---

## Error Handling

### Common Exceptions

#### `FractionalOrderError`
Raised when fractional order is invalid.

#### `BackendError`
Raised when backend operation fails.

#### `ValidationError`
Raised when input validation fails.

#### `MemoryError`
Raised when memory limits are exceeded.

### Error Recovery
```python
try:
    result = compute_fractional_derivative(f, x, alpha)
except FractionalOrderError as e:
    print(f"Invalid fractional order: {e}")
    # Handle error
except BackendError as e:
    print(f"Backend error: {e}")
    # Fallback to different backend
```

---

*This API reference covers the main functionality of HPFRACC. For more detailed information, see the specific module documentation and examples.*

