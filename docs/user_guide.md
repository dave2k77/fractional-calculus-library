# User Guide - Fractional Calculus Library

A comprehensive guide to using the Fractional Calculus Library for numerical methods in fractional calculus.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Fractional Derivatives](#fractional-derivatives)
4. [Fractional Integrals](#fractional-integrals)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Error Analysis and Validation](#error-analysis-and-validation)
8. [Visualization](#visualization)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Quick Start Example

```python
import numpy as np
from src.algorithms.optimized_methods import OptimizedCaputo, optimized_caputo

# Create a fractional derivative calculator
alpha = 0.5  # Fractional order
caputo = OptimizedCaputo(alpha)

# Define time points and function values
t = np.linspace(0.1, 2.0, 100)
f = t**2  # Function f(t) = t²
h = t[1] - t[0]  # Step size

# Compute fractional derivative
result = caputo.compute(f, t, h)
print(f"Caputo derivative of order {alpha}: {result[-1]:.6f}")

# Or use the function interface
result_func = optimized_caputo(f, t, alpha, h)
print(f"Function interface result: {result_func[-1]:.6f}")
```

### Import Structure

```python
# Core components
from src.core.definitions import FractionalOrder
from src.core.derivatives import FractionalDerivative

# Optimized algorithms (PRIMARY implementations)
from src.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    OptimizedGrunwaldLetnikov,
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov
)

# GPU-optimized algorithms
from src.algorithms.gpu_optimized_methods import (
    GPUOptimizedCaputo,
    gpu_optimized_caputo,
    JAXAutomaticDifferentiation
)

# Parallel-optimized algorithms
from src.algorithms.parallel_optimized_methods import (
    ParallelOptimizedCaputo,
    parallel_optimized_caputo,
    NumbaParallelManager
)

# Advanced methods
from src.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    optimized_weyl_derivative,
    optimized_marchaud_derivative
)

# Utilities
from src.utils import ErrorAnalyzer, PlotManager
from src.validation import AnalyticalSolutions, ConvergenceTester
```

---

## Basic Usage

### Understanding Fractional Orders

```python
from src.core.definitions import FractionalOrder

# Create fractional orders
alpha = FractionalOrder(0.5, "derivative")  # Half-derivative
beta = FractionalOrder(1.5, "derivative")   # 1.5-derivative
gamma = FractionalOrder(0.3, "integral")    # 0.3-integral

print(f"Alpha: {alpha}")  # FractionalOrder(0.5, derivative)
print(f"Is valid: {alpha.is_valid()}")  # True
```

### Working with Different Methods

```python
import numpy as np
from src.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    optimized_caputo,
    optimized_riemann_liouville
)

# Test parameters
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = np.sin(t)
h = t[1] - t[0]

# Compare different methods using classes
caputo = OptimizedCaputo(alpha)
riemann = OptimizedRiemannLiouville(alpha)

result_caputo = caputo.compute(f, t, h)
result_riemann = riemann.compute(f, t, h)

print(f"Caputo: {result_caputo[-1]:.6f}")
print(f"Riemann-Liouville: {result_riemann[-1]:.6f}")

# Or use function interfaces
result_caputo_func = optimized_caputo(f, t, alpha, h)
result_riemann_func = optimized_riemann_liouville(f, t, alpha, h)

print(f"Caputo (function): {result_caputo_func[-1]:.6f}")
print(f"Riemann-Liouville (function): {result_riemann_func[-1]:.6f}")
```

---

## Fractional Derivatives

### Caputo Derivative

The Caputo derivative is particularly useful for initial value problems:

```python
from src.algorithms.optimized_methods import OptimizedCaputo
import numpy as np

# Initialize with different methods
alpha = 0.5
caputo_trap = OptimizedCaputo(alpha, method="trapezoidal")
caputo_simp = OptimizedCaputo(alpha, method="simpson")
caputo_gauss = OptimizedCaputo(alpha, method="gauss")

# Test function
t = np.linspace(0.1, 2.0, 100)
f = np.exp(-t)  # Exponential function
h = t[1] - t[0]

# Compare methods
result_trap = caputo_trap.compute(f, t, h)
result_simp = caputo_simp.compute(f, t, h)
result_gauss = caputo_gauss.compute(f, t, h)

print(f"Trapezoidal: {result_trap[-1]:.6f}")
print(f"Simpson: {result_simp[-1]:.6f}")
print(f"Gauss: {result_gauss[-1]:.6f}")
```

### Riemann-Liouville Derivative

```python
from src.algorithms.optimized_methods import OptimizedRiemannLiouville

# Initialize Riemann-Liouville derivative
alpha = 0.7
riemann = OptimizedRiemannLiouville(alpha, method="trapezoidal")

# Test with power function (has known analytical solution)
t = np.linspace(0.1, 2.0, 100)
f = t**2  # f(t) = t²
h = t[1] - t[0]

result = riemann.compute(f, t, h)

# Analytical solution for D^α(t²) = Γ(3)/Γ(3-α) * t^(2-α)
import scipy.special as sp
analytical = sp.gamma(3) / sp.gamma(3 - alpha) * t**(2 - alpha)

print(f"Numerical: {result[-1]:.6f}")
print(f"Analytical: {analytical[-1]:.6f}")
```

### Grünwald-Letnikov Derivative

```python
from src.algorithms.optimized_methods import OptimizedGrunwaldLetnikov

# Initialize Grünwald-Letnikov derivative
alpha = 0.5
grunwald = OptimizedGrunwaldLetnikov(alpha)

# Test with trigonometric function
t = np.linspace(0.1, 2*np.pi, 200)
f = np.sin(t)
h = t[1] - t[0]

result = grunwald.compute(f, t, h)

print(f"Grünwald-Letnikov derivative: {result[-1]:.6f}")
```

---

## Fractional Integrals

### Riemann-Liouville Integral

```python
from src.core.integrals import RiemannLiouvilleIntegral
import numpy as np

# Initialize fractional integral
alpha = 0.5
integral = RiemannLiouvilleIntegral(alpha)

# Test function
t = np.linspace(0.1, 2.0, 100)
f = np.ones_like(t)  # Constant function f(t) = 1
h = t[1] - t[0]

result = integral.compute(f, t, h)

# Analytical solution for I^α(1) = t^α / Γ(α+1)
import scipy.special as sp
analytical = t**alpha / sp.gamma(alpha + 1)

print(f"Numerical: {result[-1]:.6f}")
print(f"Analytical: {analytical[-1]:.6f}")
```

---

## Advanced Features

### Using JAX for GPU Acceleration

```python
import jax
import jax.numpy as jnp
from src.optimisation.jax_implementations import JAXOptimizer

# Check if GPU is available
print(f"Available devices: {jax.devices()}")
print(f"GPU devices: {jax.devices('gpu')}")

# Initialize JAX optimizer
optimizer = JAXOptimizer()

# Convert data to JAX arrays
t = jnp.linspace(0.1, 2.0, 1000)
f = jnp.sin(t)

# Compute with JAX acceleration
result = optimizer.compute_caputo_derivative(f, t, alpha=0.5)
print(f"JAX result: {result[-1]:.6f}")
```

### Parallel Computing

```python
from src.optimisation.parallel_computing import ParallelComputingManager
import numpy as np

# Initialize parallel computing manager
parallel_manager = ParallelComputingManager()

# Large dataset
t = np.linspace(0.1, 10.0, 10000)
f = np.sin(t) + np.cos(2*t)

# Compute with parallel processing
result = parallel_manager.compute_parallel_derivative(
    f, t, alpha=0.5, method="caputo", n_jobs=-1
)

print(f"Parallel computation completed")
print(f"Result shape: {result.shape}")
```

### Error Analysis

```python
from src.utils.error_analysis import ErrorAnalyzer
import numpy as np

# Initialize error analyzer
analyzer = ErrorAnalyzer()

# Test with known analytical solution
t = np.linspace(0.1, 2.0, 100)
f = t**2  # f(t) = t²

# Compute numerical result
from src.algorithms.optimized_methods import OptimizedCaputo
caputo = OptimizedCaputo(0.5)
numerical = caputo.compute(f, t, t[1] - t[0])

# Analytical solution
import scipy.special as sp
analytical = sp.gamma(3) / sp.gamma(2.5) * t**1.5

# Analyze errors
absolute_error = analyzer.absolute_error(numerical, analytical)
relative_error = analyzer.relative_error(numerical, analytical)
l2_error = analyzer.l2_error(numerical, analytical)

print(f"Absolute error: {absolute_error:.2e}")
print(f"Relative error: {relative_error:.2e}")
print(f"L2 error: {l2_error:.2e}")
```

### Convergence Analysis

```python
from src.validation.convergence_tests import ConvergenceTester
import numpy as np

# Initialize convergence tester
tester = ConvergenceTester()

# Test convergence with different grid sizes
grid_sizes = [50, 100, 200, 400, 800]
alpha = 0.5

def test_function(t):
    return np.sin(t)

# Run convergence test
convergence_rate = tester.test_convergence(
    test_function, alpha, grid_sizes, method="caputo"
)

print(f"Convergence rate: {convergence_rate:.3f}")
```

---

## Performance Optimization

### Memory Management

```python
from src.utils.memory_management import MemoryManager, CacheManager

# Initialize memory manager
memory_manager = MemoryManager()
cache_manager = CacheManager()

# Monitor memory usage
initial_memory = memory_manager.get_current_memory_usage()
print(f"Initial memory: {initial_memory:.2f} MB")

# Large computation
t = np.linspace(0.1, 10.0, 50000)
f = np.sin(t)

from src.algorithms.optimized_methods import OptimizedCaputo
caputo = OptimizedCaputo(0.5)
result = caputo.compute(f, t, t[1] - t[0])

# Check memory after computation
final_memory = memory_manager.get_current_memory_usage()
print(f"Final memory: {final_memory:.2f} MB")
print(f"Memory increase: {final_memory - initial_memory:.2f} MB")

# Clear cache if needed
cache_manager.clear_cache()
```

### Benchmarking

```python
from src.validation.benchmarks import BenchmarkSuite
import numpy as np

# Initialize benchmark suite
benchmark_suite = BenchmarkSuite()

# Define test function
def test_function(t):
    return np.sin(t) + np.cos(2*t)

# Run comprehensive benchmark
results = benchmark_suite.run_comprehensive_benchmark(
    test_function, alpha=0.5, grid_sizes=[100, 500, 1000]
)

print("Benchmark Results:")
for method, metrics in results.items():
    print(f"{method}: {metrics}")
```

---

## Error Analysis and Validation

### Validation Against Analytical Solutions

```python
from src.validation.analytical_solutions import AnalyticalSolutions
import numpy as np

# Initialize analytical solutions
analytical = AnalyticalSolutions()

# Test with power function
t = np.linspace(0.1, 2.0, 100)
f = t**2

# Get analytical solution
analytical_result = analytical.power_function_derivative(t, 2, 0.5)

# Compare with numerical result
from src.algorithms.optimized_methods import OptimizedCaputo
caputo = OptimizedCaputo(0.5)
numerical_result = caputo.compute(f, t, t[1] - t[0])

# Validate
from src.validation import validate_against_analytical
validation_result = validate_against_analytical(
    numerical_result, analytical_result, tolerance=1e-6
)

print(f"Validation passed: {validation_result}")
```

### Convergence Study

```python
from src.validation.convergence_tests import run_convergence_study
import numpy as np

# Define test function
def test_function(t):
    return np.exp(-t)

# Run convergence study
grid_sizes = [50, 100, 200, 400, 800]
convergence_results = run_convergence_study(
    test_function, alpha=0.5, grid_sizes=grid_sizes
)

print("Convergence Study Results:")
for method, rate in convergence_results.items():
    print(f"{method}: {rate:.3f}")
```

---

## Visualization

### Basic Plotting

```python
from src.utils.plotting import PlotManager
import numpy as np
import matplotlib.pyplot as plt

# Initialize plot manager
plot_manager = PlotManager()
plot_manager.setup_plotting_style()

# Generate data
t = np.linspace(0.1, 2*np.pi, 100)
f = np.sin(t)

# Compute derivatives
from src.algorithms.optimized_methods import OptimizedCaputo
caputo = OptimizedCaputo(0.5)
result = caputo.compute(f, t, t[1] - t[0])

# Create comparison plot
fig, ax = plot_manager.create_comparison_plot(
    t, [f, result], 
    labels=['Original Function', 'Caputo Derivative (α=0.5)'],
    title='Fractional Derivative Example'
)

plt.show()
```

### Error Analysis Plots

```python
from src.utils.plotting import plot_error_analysis
import numpy as np

# Generate error data
grid_sizes = [50, 100, 200, 400, 800]
errors = []

for n in grid_sizes:
    t = np.linspace(0.1, 2.0, n)
    f = t**2
    
    from src.algorithms.optimized_methods import OptimizedCaputo
    caputo = OptimizedCaputo(0.5)
    numerical = caputo.compute(f, t, t[1] - t[0])
    
    # Analytical solution
    import scipy.special as sp
    analytical = sp.gamma(3) / sp.gamma(2.5) * t**1.5
    
    # Compute error
    from src.utils.error_analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    error = analyzer.l2_error(numerical, analytical)
    errors.append(error)

# Plot error analysis
plot_error_analysis(grid_sizes, errors, title='Convergence Analysis')
plt.show()
```

---

## Best Practices

### 1. Choose Appropriate Methods

```python
# For initial value problems: Use Caputo
from src.algorithms.optimized_methods import OptimizedCaputo

# For boundary value problems: Use Riemann-Liouville
from src.algorithms.optimized_methods import OptimizedRiemannLiouville

# For high-order derivatives: Use Grünwald-Letnikov
from src.algorithms.optimized_methods import OptimizedGrunwaldLetnikov
```

### 2. Grid Size Selection

```python
# Rule of thumb: Use at least 100 points for basic accuracy
# For high precision: Use 1000+ points
# For convergence studies: Use multiple grid sizes

grid_sizes = [100, 500, 1000, 2000]  # Good for convergence analysis
```

### 3. Error Control

```python
# Always validate against analytical solutions when possible
# Use error analysis tools for numerical validation
# Monitor convergence rates

from src.utils.error_analysis import ErrorAnalyzer
analyzer = ErrorAnalyzer()

# Check multiple error metrics
absolute_error = analyzer.absolute_error(numerical, analytical)
relative_error = analyzer.relative_error(numerical, analytical)
l2_error = analyzer.l2_error(numerical, analytical)
```

### 4. Performance Optimization

```python
# Use JAX for large-scale computations
# Enable parallel processing for multiple computations
# Monitor memory usage for large datasets

# For GPU acceleration
import jax
if jax.devices('gpu'):
    # Use JAX implementations
    pass

# For parallel processing
from src.optimisation.parallel_computing import ParallelComputingManager
parallel_manager = ParallelComputingManager()
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Slow Performance
**Problem**: Computations are taking too long

**Solutions**:
```python
# Use JAX acceleration
from src.optimisation.jax_implementations import JAXOptimizer

# Use parallel processing
from src.optimisation.parallel_computing import ParallelComputingManager

# Reduce grid size for initial testing
t = np.linspace(0.1, 2.0, 100)  # Start with smaller grid
```

#### Issue 2: Memory Errors
**Problem**: Out of memory errors

**Solutions**:
```python
# Monitor memory usage
from src.utils.memory_management import MemoryManager
memory_manager = MemoryManager()

# Clear cache
from src.utils.memory_management import CacheManager
cache_manager = CacheManager()
cache_manager.clear_cache()

# Use smaller datasets
t = np.linspace(0.1, 2.0, 1000)  # Reduce grid size
```

#### Issue 3: Inaccurate Results
**Problem**: Results don't match expected values

**Solutions**:
```python
# Validate against analytical solutions
from src.validation.analytical_solutions import AnalyticalSolutions

# Check convergence
from src.validation.convergence_tests import ConvergenceTester

# Use different numerical methods
caputo_trap = OptimizedCaputo(alpha, method="trapezoidal")
caputo_simp = OptimizedCaputo(alpha, method="simpson")
```

#### Issue 4: Import Errors
**Problem**: Module not found errors

**Solutions**:
```bash
# Reinstall the library
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
python scripts/run_tests.py --type fast
```

---

## Next Steps

After mastering the basics:

1. **Explore Advanced Features**: Try GPU acceleration and parallel computing
2. **Study Examples**: Check the `examples/` directory for more complex use cases
3. **Read API Documentation**: Detailed API reference in `docs/api_reference/`
4. **Contribute**: Join the development community
5. **Research Applications**: Apply to your specific domain

---

**Note**: This user guide covers the most common use cases. For advanced features and detailed API documentation, refer to the [API Reference](api_reference/) and [Examples](examples/).
