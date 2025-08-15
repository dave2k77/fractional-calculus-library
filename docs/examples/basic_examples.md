# Basic Examples - Fractional Calculus Library

This document provides basic examples demonstrating the core functionality of the Fractional Calculus Library.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Derivative Computations](#basic-derivative-computations)
3. [Comparing Different Methods](#comparing-different-methods)
4. [Working with Different Functions](#working-with-different-functions)
5. [Error Analysis](#error-analysis)
6. [Visualization](#visualization)

---

## Getting Started

### Simple Caputo Derivative

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Create a fractional derivative calculator
alpha = 0.5  # Half-derivative
caputo = CaputoDerivative(alpha)

# Define time points and function values
t = np.linspace(0.1, 2.0, 100)
f = t**2  # Function f(t) = t²
h = t[1] - t[0]  # Step size

# Compute fractional derivative
result = caputo.compute(f, t, h)

print(f"Caputo derivative of order {alpha}: {result[-1]:.6f}")
# Output: Caputo derivative of order 0.5: 1.595769
```

### Multiple Fractional Orders

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test different fractional orders
orders = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
t = np.linspace(0.1, 2.0, 100)
f = t**2
h = t[1] - t[0]

print("Caputo derivatives of f(t) = t²:")
for alpha in orders:
    caputo = CaputoDerivative(alpha)
    result = caputo.compute(f, t, h)
    print(f"α = {alpha:4.2f}: {result[-1]:8.6f}")
```

---

## Basic Derivative Computations

### Power Functions

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
import scipy.special as sp

# Test with power function f(t) = t^n
n = 2  # Power
alpha = 0.5  # Fractional order
t = np.linspace(0.1, 2.0, 100)
f = t**n
h = t[1] - t[0]

# Numerical computation
caputo = CaputoDerivative(alpha)
numerical = caputo.compute(f, t, h)

# Analytical solution: D^α(t^n) = Γ(n+1)/Γ(n+1-α) * t^(n-α)
analytical = sp.gamma(n + 1) / sp.gamma(n + 1 - alpha) * t**(n - alpha)

print(f"Numerical:  {numerical[-1]:.6f}")
print(f"Analytical: {analytical[-1]:.6f}")
print(f"Error:      {abs(numerical[-1] - analytical[-1]):.2e}")
```

### Exponential Functions

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test with exponential function f(t) = e^(-t)
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = np.exp(-t)
h = t[1] - t[0]

# Compute derivative
caputo = CaputoDerivative(alpha)
result = caputo.compute(f, t, h)

print(f"Caputo derivative of e^(-t) at t=2.0: {result[-1]:.6f}")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, f, 'b-', label='Original: e^(-t)', linewidth=2)
plt.plot(t, result, 'r-', label=f'Caputo D^{alpha}(e^(-t))', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title('Caputo Fractional Derivative of Exponential Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Trigonometric Functions

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test with trigonometric function f(t) = sin(t)
alpha = 0.5
t = np.linspace(0.1, 2*np.pi, 200)
f = np.sin(t)
h = t[1] - t[0]

# Compute derivative
caputo = CaputoDerivative(alpha)
result = caputo.compute(f, t, h)

print(f"Caputo derivative of sin(t) at t=2π: {result[-1]:.6f}")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, f, 'b-', label='Original: sin(t)', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(t, result, 'r-', label=f'Caputo D^{alpha}(sin(t))', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Derivative Value')
plt.title('Caputo Fractional Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Comparing Different Methods

### Caputo vs Riemann-Liouville

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative

# Test parameters
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = t**2
h = t[1] - t[0]

# Compare methods
caputo = CaputoDerivative(alpha)
riemann = RiemannLiouvilleDerivative(alpha)

result_caputo = caputo.compute(f, t, h)
result_riemann = riemann.compute(f, t, h)

print(f"Caputo:           {result_caputo[-1]:.6f}")
print(f"Riemann-Liouville: {result_riemann[-1]:.6f}")
print(f"Difference:       {abs(result_caputo[-1] - result_riemann[-1]):.2e}")

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, f, 'k-', label='Original: t²', linewidth=2)
plt.plot(t, result_caputo, 'b-', label='Caputo', linewidth=2)
plt.plot(t, result_riemann, 'r--', label='Riemann-Liouville', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title(f'Comparison of Fractional Derivatives (α={alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Different Numerical Methods

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test different numerical methods
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = np.sin(t)
h = t[1] - t[0]

methods = ["trapezoidal", "simpson", "gauss"]
results = {}

for method in methods:
    caputo = CaputoDerivative(alpha, method=method)
    results[method] = caputo.compute(f, t, h)

print("Comparison of numerical methods:")
for method, result in results.items():
    print(f"{method:12}: {result[-1]:.6f}")

# Plot comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, f, 'k-', label='Original: sin(t)', linewidth=2)
for method, result in results.items():
    plt.plot(t, result, '--', label=f'{method.capitalize()}', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title(f'Caputo Derivative with Different Methods (α={alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Working with Different Functions

### Piecewise Functions

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Define piecewise function
def piecewise_function(t):
    """Piecewise function: f(t) = t for t < 1, f(t) = 1 for t >= 1"""
    return np.where(t < 1, t, 1)

# Compute fractional derivative
alpha = 0.5
t = np.linspace(0.1, 2.0, 200)
f = piecewise_function(t)
h = t[1] - t[0]

caputo = CaputoDerivative(alpha)
result = caputo.compute(f, t, h)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, f, 'b-', label='Original: Piecewise', linewidth=2)
plt.plot(t, result, 'r-', label=f'Caputo D^{alpha}', linewidth=2)
plt.axvline(x=1, color='g', linestyle='--', alpha=0.7, label='Discontinuity')
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title('Caputo Derivative of Piecewise Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Polynomial Functions

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test with polynomial f(t) = t³ - 2t² + t
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = t**3 - 2*t**2 + t
h = t[1] - t[0]

caputo = CaputoDerivative(alpha)
result = caputo.compute(f, t, h)

print(f"Caputo derivative of t³ - 2t² + t at t=2.0: {result[-1]:.6f}")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(t, f, 'b-', label='Original: t³ - 2t² + t', linewidth=2)
plt.plot(t, result, 'r-', label=f'Caputo D^{alpha}', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Function Value')
plt.title('Caputo Derivative of Polynomial Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Error Analysis

### Convergence Study

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
from src.utils.error_analysis import ErrorAnalyzer

# Test convergence with different grid sizes
grid_sizes = [50, 100, 200, 400, 800]
alpha = 0.5
analyzer = ErrorAnalyzer()

# Analytical solution for f(t) = t²
def analytical_solution(t, alpha):
    import scipy.special as sp
    return sp.gamma(3) / sp.gamma(3 - alpha) * t**(2 - alpha)

errors = []
for n in grid_sizes:
    t = np.linspace(0.1, 2.0, n)
    f = t**2
    h = t[1] - t[0]
    
    # Numerical solution
    caputo = CaputoDerivative(alpha)
    numerical = caputo.compute(f, t, h)
    
    # Analytical solution
    analytical = analytical_solution(t, alpha)
    
    # Compute error
    error = analyzer.l2_error(numerical, analytical)
    errors.append(error)

print("Convergence Study:")
for n, error in zip(grid_sizes, errors):
    print(f"Grid size {n:3d}: Error = {error:.2e}")

# Plot convergence
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.loglog(grid_sizes, errors, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Grid Size')
plt.ylabel('L2 Error')
plt.title('Convergence of Caputo Derivative')
plt.grid(True, alpha=0.3)
plt.show()
```

### Method Comparison

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative
from src.utils.error_analysis import ErrorAnalyzer

# Test with known analytical solution
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = t**2
h = t[1] - t[0]

# Analytical solution
import scipy.special as sp
analytical = sp.gamma(3) / sp.gamma(3 - alpha) * t**(2 - alpha)

# Compare different methods
methods = {
    'Caputo': CaputoDerivative(alpha),
    'Riemann-Liouville': RiemannLiouvilleDerivative(alpha),
    'Grünwald-Letnikov': GrunwaldLetnikovDerivative(alpha)
}

analyzer = ErrorAnalyzer()
results = {}

for name, method in methods.items():
    numerical = method.compute(f, t, h)
    error = analyzer.l2_error(numerical, analytical)
    results[name] = error

print("Method Comparison (L2 Error):")
for name, error in results.items():
    print(f"{name:20}: {error:.2e}")
```

---

## Visualization

### Multiple Orders Comparison

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
import matplotlib.pyplot as plt

# Test multiple fractional orders
orders = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
t = np.linspace(0.1, 2.0, 100)
f = t**2
h = t[1] - t[0]

# Compute derivatives for all orders
results = {}
for alpha in orders:
    caputo = CaputoDerivative(alpha)
    results[alpha] = caputo.compute(f, t, h)

# Create subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, alpha in enumerate(orders):
    ax = axes[i]
    ax.plot(t, f, 'k-', label='Original: t²', linewidth=2)
    ax.plot(t, results[alpha], 'r-', label=f'D^{alpha}', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Caputo Derivative (α={alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3D Surface Plot

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create meshgrid for 3D plot
alpha_values = np.linspace(0.1, 1.9, 20)
t_values = np.linspace(0.1, 2.0, 50)
alpha_mesh, t_mesh = np.meshgrid(alpha_values, t_values)

# Compute derivatives for all combinations
result_mesh = np.zeros_like(alpha_mesh)
f = t_values**2  # Function values

for i, alpha in enumerate(alpha_values):
    caputo = CaputoDerivative(alpha)
    result = caputo.compute(f, t_values, t_values[1] - t_values[0])
    result_mesh[:, i] = result

# Create 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(alpha_mesh, t_mesh, result_mesh, 
                         cmap='viridis', alpha=0.8)
ax.set_xlabel('Fractional Order α')
ax.set_ylabel('Time t')
ax.set_zlabel('D^α(t²)')
ax.set_title('Caputo Derivative Surface')

# Add colorbar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
plt.show()
```

---

## Summary

These basic examples demonstrate:

1. **Simple Usage**: How to compute fractional derivatives with minimal code
2. **Method Comparison**: Differences between Caputo, Riemann-Liouville, and Grünwald-Letnikov
3. **Function Types**: Working with various function types (power, exponential, trigonometric)
4. **Error Analysis**: How to validate results and study convergence
5. **Visualization**: Creating informative plots and comparisons

For more advanced examples, see the [Advanced Examples](advanced_examples.md) and [Performance Examples](performance_examples.md) documents.
