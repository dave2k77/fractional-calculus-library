# Advanced Examples - Fractional Calculus Library

This document provides advanced examples demonstrating the new advanced fractional calculus methods and their optimized implementations.

## Table of Contents

1. [Advanced Methods Overview](#advanced-methods-overview)
2. [Weyl Derivative Examples](#weyl-derivative-examples)
3. [Marchaud Derivative Examples](#marchaud-derivative-examples)
4. [Hadamard Derivative Examples](#hadamard-derivative-examples)
5. [Reiz-Feller Derivative Examples](#reiz-feller-derivative-examples)
6. [Adomian Decomposition Examples](#adomian-decomposition-examples)
7. [Performance Comparisons](#performance-comparisons)
8. [Real-World Applications](#real-world-applications)

---

## Advanced Methods Overview

The library now includes five advanced fractional calculus methods:

1. **Weyl Derivative** - For periodic functions using FFT convolution
2. **Marchaud Derivative** - With difference quotient convolution and memory optimization
3. **Hadamard Derivative** - Using logarithmic transformation
4. **Reiz-Feller Derivative** - Via spectral method using FFT
5. **Adomian Decomposition** - For solving fractional differential equations

### Importing Advanced Methods

```python
# Standard implementations
from src.algorithms.advanced_methods import (
    WeylDerivative, MarchaudDerivative, HadamardDerivative,
    ReizFellerDerivative, AdomianDecomposition,
    weyl_derivative, marchaud_derivative, hadamard_derivative,
    reiz_feller_derivative
)

# Optimized implementations (JAX/Numba)
from src.algorithms.advanced_optimized_methods import (
    OptimizedWeylDerivative, OptimizedMarchaudDerivative,
    OptimizedHadamardDerivative, OptimizedReizFellerDerivative,
    OptimizedAdomianDecomposition,
    optimized_weyl_derivative, optimized_marchaud_derivative,
    optimized_hadamard_derivative, optimized_reiz_feller_derivative,
    optimized_adomian_solve
)
```

---

## Weyl Derivative Examples

### Basic Weyl Derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import WeylDerivative

# Weyl derivative is ideal for periodic functions
alpha = 0.5
x = np.linspace(0, 4*np.pi, 200)
f = lambda x: np.sin(x)  # Periodic function

# Compute Weyl derivative
weyl = WeylDerivative(alpha)
result = weyl.compute(f, x, h=0.1)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, f(x), 'b-', label='Original: sin(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x, result, 'r-', label=f'Weyl D^{alpha}(sin(x))', linewidth=2)
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Weyl Fractional Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Weyl derivative at x=2π: {result[100]:.6f}")
```

### Weyl vs Standard Methods for Periodic Functions

```python
import numpy as np
import time
from src.algorithms.advanced_methods import WeylDerivative
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative

# Test with periodic function
alpha = 0.5
x = np.linspace(0, 2*np.pi, 1000)
f = lambda x: np.cos(2*x) + np.sin(3*x)

# Compare methods
methods = {
    'Weyl': WeylDerivative(alpha),
    'Caputo': CaputoDerivative(alpha),
    'Riemann-Liouville': RiemannLiouvilleDerivative(alpha)
}

results = {}
times = {}

for name, method in methods.items():
    start_time = time.time()
    results[name] = method.compute(f, x, h=0.01)
    times[name] = time.time() - start_time

print("Method Comparison for Periodic Function:")
for name in methods.keys():
    print(f"{name:15}: {times[name]:.4f}s, Result at x=π: {results[name][500]:.6f}")

# Plot comparison
plt.figure(figsize=(12, 8))
plt.plot(x, f(x), 'k-', label='Original: cos(2x) + sin(3x)', linewidth=2)
for name, result in results.items():
    plt.plot(x, result, '--', label=f'{name}', linewidth=2)
plt.xlabel('x')
plt.ylabel('Function Value')
plt.title(f'Fractional Derivatives Comparison (α={alpha})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Marchaud Derivative Examples

### Basic Marchaud Derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import MarchaudDerivative

# Marchaud derivative with memory optimization
alpha = 0.5
x = np.linspace(0, 5, 200)
f = lambda x: np.exp(-x) * np.sin(x)

# Compute Marchaud derivative
marchaud = MarchaudDerivative(alpha)
result = marchaud.compute(f, x, h=0.025)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, f(x), 'b-', label='Original: e^(-x)sin(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x, result, 'r-', label=f'Marchaud D^{alpha}', linewidth=2)
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Marchaud Fractional Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Marchaud derivative at x=2: {result[80]:.6f}")
```

### Memory Optimization Demonstration

```python
import numpy as np
import time
import psutil
from src.algorithms.advanced_methods import MarchaudDerivative

# Test memory usage with different grid sizes
grid_sizes = [1000, 5000, 10000, 20000]
alpha = 0.5

print("Memory Usage Comparison:")
print("Grid Size | Time (s) | Memory (MB)")
print("-" * 35)

for n in grid_sizes:
    x = np.linspace(0, 10, n)
    f = lambda x: np.exp(-x/2) * np.sin(x)
    
    # Monitor memory usage
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    marchaud = MarchaudDerivative(alpha)
    result = marchaud.compute(f, x, h=0.01)
    computation_time = time.time() - start_time
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    print(f"{n:9d} | {computation_time:7.3f} | {mem_used:10.1f}")
```

---

## Hadamard Derivative Examples

### Basic Hadamard Derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import HadamardDerivative

# Hadamard derivative requires positive domain
alpha = 0.5
x = np.linspace(1, 10, 200)  # Must be positive
f = lambda x: np.log(x) * np.sin(x)

# Compute Hadamard derivative
hadamard = HadamardDerivative(alpha)
result = hadamard.compute(f, x, h=0.045)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, f(x), 'b-', label='Original: ln(x)sin(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x, result, 'r-', label=f'Hadamard D^{alpha}', linewidth=2)
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Hadamard Fractional Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Hadamard derivative at x=5: {result[100]:.6f}")
```

### Logarithmic Transformation Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import HadamardDerivative

# Test with different function types
alpha = 0.5
x = np.linspace(1, 5, 100)

functions = {
    'Power': lambda x: x**2,
    'Logarithmic': lambda x: np.log(x),
    'Exponential': lambda x: np.exp(-x),
    'Trigonometric': lambda x: np.sin(x)
}

hadamard = HadamardDerivative(alpha)
results = {}

for name, func in functions.items():
    results[name] = hadamard.compute(func, x, h=0.04)

# Plot all results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (name, result) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(x, functions[name](x), 'b-', label=f'Original: {name}', linewidth=2)
    ax.plot(x, result, 'r-', label=f'Hadamard D^{alpha}', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Hadamard Derivative - {name} Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Reiz-Feller Derivative Examples

### Basic Reiz-Feller Derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import ReizFellerDerivative

# Reiz-Feller derivative using spectral method
alpha = 0.5
x = np.linspace(0, 5, 200)
f = lambda x: np.exp(-x**2/2)  # Gaussian function

# Compute Reiz-Feller derivative
reiz_feller = ReizFellerDerivative(alpha)
result = reiz_feller.compute(f, x, h=0.025)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x, f(x), 'b-', label='Original: exp(-x²/2)', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Original Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x, result, 'r-', label=f'Reiz-Feller D^{alpha}', linewidth=2)
plt.xlabel('x')
plt.ylabel('Derivative')
plt.title('Reiz-Feller Fractional Derivative')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Reiz-Feller derivative at x=1: {result[40]:.6f}")
```

### Spectral Method Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import ReizFellerDerivative

# Test spectral method with different functions
alpha = 0.5
x = np.linspace(0, 4, 200)

functions = {
    'Gaussian': lambda x: np.exp(-x**2/2),
    'Lorentzian': lambda x: 1/(1 + x**2),
    'Exponential': lambda x: np.exp(-x),
    'Polynomial': lambda x: x**3 - 2*x**2 + x
}

reiz_feller = ReizFellerDerivative(alpha)
results = {}

for name, func in functions.items():
    results[name] = reiz_feller.compute(func, x, h=0.02)

# Plot all results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (name, result) in enumerate(results.items()):
    ax = axes[i]
    ax.plot(x, functions[name](x), 'b-', label=f'Original: {name}', linewidth=2)
    ax.plot(x, result, 'r-', label=f'Reiz-Feller D^{alpha}', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Function Value')
    ax.set_title(f'Reiz-Feller Derivative - {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Adomian Decomposition Examples

### Basic FDE Solving

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import AdomianDecomposition

# Solve fractional differential equation: D^α y(t) = -y(t)
alpha = 0.5
t = np.linspace(0, 5, 100)

def fractional_ode(t, y, alpha):
    """Right-hand side of the FDE: D^α y(t) = -y(t)"""
    return -y

# Solve using Adomian decomposition
adomian = AdomianDecomposition(alpha)
solution = adomian.solve(fractional_ode, t, initial_condition=1.0, terms=10)

# Plot solution
plt.figure(figsize=(10, 6))
plt.plot(t, solution, 'r-', label=f'Adomian Solution (α={alpha})', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.title('Solution of D^α y(t) = -y(t) using Adomian Decomposition')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Solution at t=5: {solution[-1]:.6f}")
```

### Nonlinear FDE Example

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import AdomianDecomposition

# Solve nonlinear FDE: D^α y(t) = -y(t)²
alpha = 0.5
t = np.linspace(0, 3, 100)

def nonlinear_fde(t, y, alpha):
    """Nonlinear FDE: D^α y(t) = -y(t)²"""
    return -y**2

# Solve using Adomian decomposition
adomian = AdomianDecomposition(alpha)
solution = adomian.solve(nonlinear_fde, t, initial_condition=1.0, terms=15)

# Plot solution
plt.figure(figsize=(10, 6))
plt.plot(t, solution, 'r-', label=f'Nonlinear FDE Solution (α={alpha})', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.title('Solution of D^α y(t) = -y(t)² using Adomian Decomposition')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Nonlinear solution at t=3: {solution[-1]:.6f}")
```

### Convergence Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import AdomianDecomposition

# Test convergence with different numbers of terms
alpha = 0.5
t = np.linspace(0, 2, 50)

def test_fde(t, y, alpha):
    """Test FDE: D^α y(t) = -y(t) + t"""
    return -y + t

# Test different numbers of terms
term_counts = [5, 10, 15, 20]
solutions = {}

adomian = AdomianDecomposition(alpha)
for terms in term_counts:
    solutions[terms] = adomian.solve(test_fde, t, initial_condition=1.0, terms=terms)

# Plot convergence
plt.figure(figsize=(10, 6))
for terms, solution in solutions.items():
    plt.plot(t, solution, '--', label=f'{terms} terms', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('y(t)')
plt.title('Adomian Decomposition Convergence Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check convergence at final time
print("Convergence at t=2:")
for terms in term_counts:
    print(f"{terms:2d} terms: {solutions[terms][-1]:.6f}")
```

---

## Performance Comparisons

### Standard vs Optimized Methods

```python
import numpy as np
import time
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import (
    WeylDerivative, MarchaudDerivative, HadamardDerivative, ReizFellerDerivative
)
from src.algorithms.advanced_optimized_methods import (
    optimized_weyl_derivative, optimized_marchaud_derivative,
    optimized_hadamard_derivative, optimized_reiz_feller_derivative
)

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 1000)
f = lambda x: np.sin(x) * np.exp(-x/3)

# Standard methods
methods_std = {
    'Weyl': WeylDerivative(alpha),
    'Marchaud': MarchaudDerivative(alpha),
    'Hadamard': HadamardDerivative(alpha),
    'Reiz-Feller': ReizFellerDerivative(alpha)
}

# Test standard methods
times_std = {}
results_std = {}

for name, method in methods_std.items():
    start_time = time.time()
    if name == 'Hadamard':
        x_test = np.linspace(1, 5, 1000)  # Positive domain
    else:
        x_test = x
    results_std[name] = method.compute(f, x_test, h=0.005)
    times_std[name] = time.time() - start_time

# Test optimized methods
times_opt = {}
results_opt = {}

start_time = time.time()
results_opt['Weyl'] = optimized_weyl_derivative(f, x, alpha, h=0.005)
times_opt['Weyl'] = time.time() - start_time

start_time = time.time()
results_opt['Marchaud'] = optimized_marchaud_derivative(f, x, alpha, h=0.005)
times_opt['Marchaud'] = time.time() - start_time

start_time = time.time()
results_opt['Hadamard'] = optimized_hadamard_derivative(f, x, alpha, h=0.005)
times_opt['Hadamard'] = time.time() - start_time

start_time = time.time()
results_opt['Reiz-Feller'] = optimized_reiz_feller_derivative(f, x, alpha, h=0.005)
times_opt['Reiz-Feller'] = time.time() - start_time

# Print performance comparison
print("Performance Comparison (1000 points):")
print("Method      | Standard (s) | Optimized (s) | Speedup")
print("-" * 50)
for method in methods_std.keys():
    speedup = times_std[method] / times_opt[method]
    print(f"{method:11} | {times_std[method]:11.4f} | {times_opt[method]:12.4f} | {speedup:7.1f}x")

# Plot performance comparison
methods = list(methods_std.keys())
std_times = [times_std[m] for m in methods]
opt_times = [times_opt[m] for m in methods]

x_pos = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_pos - width/2, std_times, width, label='Standard', alpha=0.8)
plt.bar(x_pos + width/2, opt_times, width, label='Optimized', alpha=0.8)
plt.xlabel('Method')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison: Standard vs Optimized Methods')
plt.xticks(x_pos, methods)
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

### Memory Usage Comparison

```python
import numpy as np
import psutil
import time
from src.algorithms.advanced_methods import MarchaudDerivative
from src.algorithms.advanced_optimized_methods import optimized_marchaud_derivative

# Test memory usage with different grid sizes
grid_sizes = [1000, 5000, 10000]
alpha = 0.5

print("Memory Usage Comparison (Marchaud Derivative):")
print("Grid Size | Standard (MB) | Optimized (MB) | Ratio")
print("-" * 55)

for n in grid_sizes:
    x = np.linspace(0, 10, n)
    f = lambda x: np.exp(-x/2) * np.sin(x)
    
    # Standard method
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    marchaud = MarchaudDerivative(alpha)
    result_std = marchaud.compute(f, x, h=0.01)
    time_std = time.time() - start_time
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_std = mem_after - mem_before
    
    # Optimized method
    mem_before = process.memory_info().rss / 1024 / 1024
    
    start_time = time.time()
    result_opt = optimized_marchaud_derivative(f, x, alpha, h=0.01)
    time_opt = time.time() - start_time
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_opt = mem_after - mem_before
    
    ratio = mem_std / mem_opt if mem_opt > 0 else float('inf')
    
    print(f"{n:9d} | {mem_std:12.1f} | {mem_opt:13.1f} | {ratio:5.1f}x")
```

---

## Real-World Applications

### Anomalous Diffusion Modeling

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import MarchaudDerivative

# Model anomalous diffusion using Marchaud derivative
alpha = 0.7  # Anomalous diffusion exponent
t = np.linspace(0, 10, 200)
x0 = 1.0  # Initial position

# Diffusion coefficient
D = 0.1

# Solve diffusion equation: ∂P/∂t = D * D^α P
def diffusion_equation(t, P, alpha):
    """Right-hand side of anomalous diffusion equation"""
    return D * P  # Simplified for demonstration

# Use Marchaud derivative for spatial discretization
marchaud = MarchaudDerivative(alpha)

# Initial condition (Gaussian)
P0 = np.exp(-(t - x0)**2 / (2 * 0.1))

# Time evolution (simplified)
dt = t[1] - t[0]
P = P0.copy()

for i in range(1, len(t)):
    # Compute fractional derivative
    dP_dx = marchaud.compute(P, t, dt)
    # Update solution (Euler method)
    P += dt * D * dP_dx

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, P0, 'b-', label='Initial condition', linewidth=2)
plt.plot(t, P, 'r-', label=f'Anomalous diffusion (α={alpha})', linewidth=2)
plt.xlabel('Position x')
plt.ylabel('Probability density P(x,t)')
plt.title('Anomalous Diffusion Modeling')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.semilogy(t, P0, 'b-', label='Initial condition', linewidth=2)
plt.semilogy(t, P, 'r-', label=f'Anomalous diffusion (α={alpha})', linewidth=2)
plt.xlabel('Position x')
plt.ylabel('Probability density P(x,t)')
plt.title('Anomalous Diffusion (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Fractional Wave Equation

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import WeylDerivative

# Solve fractional wave equation using Weyl derivative
alpha = 0.5
x = np.linspace(0, 4*np.pi, 100)
t = np.linspace(0, 2, 50)

# Initial condition
def initial_condition(x):
    return np.sin(x)

# Wave speed
c = 1.0

# Use Weyl derivative for spatial discretization
weyl = WeylDerivative(alpha)

# Time evolution
dx = x[1] - x[0]
dt = t[1] - t[0]

# Initialize solution
u = np.zeros((len(t), len(x)))
u[0, :] = initial_condition(x)

# First time step (using initial velocity = 0)
d2u_dx2 = weyl.compute(u[0, :], x, dx)
u[1, :] = u[0, :] + 0.5 * c**2 * dt**2 * d2u_dx2

# Time stepping
for n in range(1, len(t)-1):
    d2u_dx2 = weyl.compute(u[n, :], x, dx)
    u[n+1, :] = 2*u[n, :] - u[n-1, :] + c**2 * dt**2 * d2u_dx2

# Plot results
plt.figure(figsize=(15, 10))

# Initial condition
plt.subplot(2, 2, 1)
plt.plot(x, u[0, :], 'b-', linewidth=2)
plt.title('Initial Condition')
plt.xlabel('x')
plt.ylabel('u(x,0)')
plt.grid(True, alpha=0.3)

# Solution at different times
plt.subplot(2, 2, 2)
plt.plot(x, u[10, :], 'r-', label='t=0.4', linewidth=2)
plt.plot(x, u[20, :], 'g-', label='t=0.8', linewidth=2)
plt.plot(x, u[30, :], 'm-', label='t=1.2', linewidth=2)
plt.title('Solution at Different Times')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3D surface plot
ax = plt.subplot(2, 2, (3, 4), projection='3d')
X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, u, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Fractional Wave Equation Solution')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
```

### Financial Modeling with Hadamard Derivative

```python
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.advanced_methods import HadamardDerivative

# Model option pricing with fractional volatility
alpha = 0.6
S = np.linspace(1, 100, 200)  # Stock prices (positive)
T = 1.0  # Time to maturity
r = 0.05  # Risk-free rate
sigma = 0.3  # Volatility

# Use Hadamard derivative for fractional volatility modeling
hadamard = HadamardDerivative(alpha)

# Black-Scholes-like model with fractional volatility
def option_price(S, T, r, sigma, K=50):
    """Simplified option pricing with fractional volatility"""
    # Compute fractional derivative of volatility
    sigma_frac = hadamard.compute(lambda x: sigma * np.ones_like(x), S, h=0.5)
    
    # Simplified option pricing formula
    d1 = (np.log(S/K) + (r + 0.5*sigma_frac**2)*T) / (sigma_frac*np.sqrt(T))
    d2 = d1 - sigma_frac*np.sqrt(T)
    
    # Call option price
    C = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return C

# Compute option prices
from scipy.stats import norm
prices = option_price(S, T, r, sigma)

# Plot results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(S, prices, 'r-', linewidth=2)
plt.xlabel('Stock Price S')
plt.ylabel('Option Price C(S,T)')
plt.title('Fractional Option Pricing Model')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
# Compute fractional volatility
sigma_frac = hadamard.compute(lambda x: sigma * np.ones_like(x), S, h=0.5)
plt.plot(S, sigma_frac, 'b-', linewidth=2)
plt.xlabel('Stock Price S')
plt.ylabel('Fractional Volatility σ_α(S)')
plt.title('Fractional Volatility Term')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Option price at S=50: {prices[100]:.4f}")
```

---

## Summary

These advanced examples demonstrate:

1. **Advanced Methods Usage**: Comprehensive examples for all 5 new methods
2. **Performance Optimization**: Standard vs optimized implementations
3. **Real-World Applications**: Anomalous diffusion, wave equations, financial modeling
4. **Memory Management**: Efficient handling of large datasets
5. **Convergence Analysis**: Validation of numerical methods
6. **Visualization**: Advanced plotting and analysis techniques

The advanced methods provide powerful tools for:
- **Scientific Computing**: Complex fractional differential equations
- **Engineering Applications**: Anomalous transport phenomena
- **Financial Modeling**: Fractional volatility and option pricing
- **Physics Research**: Fractional wave equations and diffusion

For more information, see the [Advanced Methods Guide](../advanced_methods_guide.md) and [API Reference](../api_reference/).
