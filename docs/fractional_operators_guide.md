# HPFRACC Fractional Operators Guide

## Overview

The HPFRACC library provides a comprehensive collection of fractional calculus operators, from classical definitions to cutting-edge advanced methods. This guide covers all available operators, their mathematical foundations, and practical usage examples.

## Table of Contents

1. [Classical Fractional Derivatives](#classical-fractional-derivatives)
2. [Novel Fractional Derivatives](#novel-fractional-derivatives)
3. [Advanced Methods](#advanced-methods)
4. [Parallel-Optimized Methods](#parallel-optimized-methods)
5. [Special Operators](#special-operators)
6. [Fractional Integrals](#fractional-integrals)
7. [Autograd Fractional Derivatives (ML)](#autograd-fractional-derivatives-ml)
8. [Usage Examples](#usage-examples)
9. [Performance Considerations](#performance-considerations)
10. [Mathematical Theory](#mathematical-theory)

## Classical Fractional Derivatives

### 1. Riemann-Liouville Derivative

**Definition**: 
```
D^α_RL f(t) = (1/Γ(n-α)) (d/dt)^n ∫₀ᵗ (t-τ)^(n-α-1) f(τ) dτ
```

**Usage**:
```python
from hpfracc.core.derivatives import create_fractional_derivative

# Create Riemann-Liouville derivative with α = 0.5
rl_derivative = create_fractional_derivative('riemann_liouville', 0.5)

# Compute derivative of f(x) = x^2 at x = 2.0
def f(x): return x**2
result = rl_derivative.compute(f, 2.0)
```

**Characteristics**:
- Most fundamental fractional derivative definition
- Well-suited for initial value problems
- Computationally efficient with optimized algorithms

### 2. Caputo Derivative

**Definition**:
```
D^α_C f(t) = (1/Γ(n-α)) ∫₀ᵗ (t-τ)^(n-α-1) f^(n)(τ) dτ
```

**Usage**:
```python
caputo_derivative = create_fractional_derivative('caputo', 0.5)
result = caputo_derivative.compute(f, 2.0)
```

**Characteristics**:
- Better behavior for initial value problems
- Preserves classical derivative properties
- Widely used in physics and engineering

### 3. Grunwald-Letnikov Derivative

**Definition**:
```
D^α_GL f(t) = lim_{h→0} h^(-α) Σ_{k=0}^∞ (-1)^k (α choose k) f(t-kh)
```

**Usage**:
```python
gl_derivative = create_fractional_derivative('grunwald_letnikov', 0.5)
result = gl_derivative.compute(f, 2.0)
```

**Characteristics**:
- Discrete approximation approach
- Good for numerical computations
- Memory-efficient implementation

## Novel Fractional Derivatives

### 4. Caputo-Fabrizio Derivative

**Definition**:
```
CF D^α f(t) = M(α)/(1-α) ∫₀ᵗ f'(τ) exp(-α(t-τ)/(1-α)) dτ
```

**Usage**:
```python
cf_derivative = create_fractional_derivative('caputo_fabrizio', 0.5)
result = cf_derivative.compute(f, 2.0)
```

**Characteristics**:
- Non-singular exponential kernel
- Better numerical stability
- Ideal for biological systems and viscoelasticity

### 5. Atangana-Baleanu Derivative

**Definition**:
```
AB D^α f(t) = B(α)/(1-α) ∫₀ᵗ f'(τ) E_α(-α(t-τ)^α/(1-α)) dτ
```

**Usage**:
```python
ab_derivative = create_fractional_derivative('atangana_baleanu', 0.5)
result = ab_derivative.compute(f, 2.0)
```

**Characteristics**:
- Mittag-Leffler kernel
- Enhanced memory effects
- Advanced applications in complex systems

## Advanced Methods

### 6. Weyl Derivative

**Definition**:
```
D^α_W f(x) = (1/Γ(n-α)) (d/dx)^n ∫_x^∞ (τ-x)^(n-α-1) f(τ) dτ
```

**Usage**:
```python
weyl_derivative = create_fractional_derivative('weyl', 0.5)
result = weyl_derivative.compute(f, 2.0)
```

**Characteristics**:
- FFT convolution implementation
- Parallel processing optimization
- Suitable for functions on entire real line

### 7. Marchaud Derivative

**Definition**:
```
D^α_M f(x) = (α/Γ(1-α)) ∫_0^∞ (f(x) - f(x-τ))/τ^(α+1) dτ
```

**Usage**:
```python
marchaud_derivative = create_fractional_derivative('marchaud', 0.5)
result = marchaud_derivative.compute(f, 2.0)
```

**Characteristics**:
- Difference quotient convolution
- Memory optimization
- General kernel support

### 8. Hadamard Derivative

**Definition**:
```
D^α_H f(x) = (1/Γ(n-α)) (x d/dx)^n ∫₁ˣ (ln(x/t))^(n-α-1) f(t) dt/t
```

**Usage**:
```python
hadamard_derivative = create_fractional_derivative('hadamard', 0.5)
result = hadamard_derivative.compute(f, 2.0)
```

**Characteristics**:
- Logarithmic kernels
- Geometric interpretation
- Applications in geometric analysis

### 9. Reiz-Feller Derivative

**Definition**:
```
D^α_RF f(x) = (1/2π) ∫_ℝ |ξ|^α F[f](ξ) e^(iξx) dξ
```

**Usage**:
```python
rf_derivative = create_fractional_derivative('reiz_feller', 0.5)
result = rf_derivative.compute(f, 2.0)
```

**Characteristics**:
- Spectral method implementation
- Fourier domain computation
- High accuracy for smooth functions

## Parallel-Optimized Methods

### 10. Parallel-Optimized Riemann-Liouville

**Usage**:
```python
parallel_rl = create_fractional_derivative('parallel_riemann_liouville', 0.5)
result = parallel_rl.compute(f, x_array)
```

**Characteristics**:
- Multi-core parallel processing
- Load balancing optimization
- Ideal for large-scale computations

### 11. Parallel-Optimized Caputo

**Usage**:
```python
parallel_caputo = create_fractional_derivative('parallel_caputo', 0.5)
result = parallel_caputo.compute(f, x_array)
```

**Characteristics**:
- Parallel memory management
- Optimized for distributed systems
- High-performance computing ready

## Special Operators

### 12. Fractional Laplacian

**Definition**:
```
(-Δ)^(α/2) f(x) = (1/2π)^n ∫_ℝ^n |ξ|^α F[f](ξ) e^(iξ·x) dξ
```

**Usage**:
```python
laplacian = create_fractional_derivative('fractional_laplacian', 0.5)
result = laplacian.compute(f, x_array)
```

**Characteristics**:
- Spatial fractional derivatives
- Multi-dimensional support
- Applications in PDEs and image processing

### 13. Fractional Fourier Transform

**Definition**:
```
F^α[f](u) = ∫_ℝ f(x) K_α(x,u) dx
```

**Usage**:
```python
fft = create_fractional_derivative('fractional_fourier_transform', 0.5)
result = fft.compute(f, x_array)
```

**Characteristics**:
- Generalized Fourier transform
- Signal processing applications
- Time-frequency analysis

### 14. Riesz-Fisher Operator

**Definition**:
```
R^α f(x) = (1/2) [D^α_+ f(x) + D^α_- f(x)]
```

**Usage**:
```python
from hpfracc.core.fractional_implementations import create_riesz_fisher_operator

# For derivative behavior (α > 0)
rf_derivative = create_riesz_fisher_operator(0.5)
result = rf_derivative.compute(f, x)

# For integral behavior (α < 0)
rf_integral = create_riesz_fisher_operator(-0.5)
result = rf_integral.compute(f, x)

# For identity behavior (α = 0)
rf_identity = create_riesz_fisher_operator(0.0)
result = rf_identity.compute(f, x)
```

**Characteristics**:
- Unified derivative/integral operator
- Smooth transition between operations
- Perfect for signal processing and image analysis

## Autograd Fractional Derivatives (ML)

The ML module provides autograd-friendly fractional derivatives that preserve the computation graph, implemented as 1D convolutions along the last dimension with method-specific kernels.

- RL/GL/Caputo: Grünwald–Letnikov (GL) binomial weights.
- CF (Caputo–Fabrizio): normalized exponential kernel.
- AB (Atangana–Baleanu): blended kernel (GL + exponential tail).

### Usage

```python
import torch
from hpfracc.ml.fractional_autograd import fractional_derivative, FractionalDerivativeLayer

x = torch.randn(2, 64, 128, requires_grad=True)  # (batch, channels, time)

# RL/GL
y_rl = fractional_derivative(x, alpha=0.5, method="RL")

# Caputo
y_caputo = fractional_derivative(x, alpha=0.5, method="Caputo")

# Caputo–Fabrizio (exponential kernel)
y_cf = fractional_derivative(x, alpha=0.5, method="CF")

# Atangana–Baleanu (blended kernel)
y_ab = fractional_derivative(x, alpha=0.5, method="AB")

# Layer wrapper
layer = FractionalDerivativeLayer(alpha=0.5, method="RL")
out = layer(torch.randn(4, 16, 256, requires_grad=True))
```

## Fractional Integrals

### Available Integral Types

1. **Riemann-Liouville Integral** (`"RL"`)
2. **Caputo Integral** (`"Caputo"`)
3. **Weyl Integral** (`"Weyl"`)
4. **Hadamard Integral** (`"Hadamard"`)
5. **Miller-Ross Integral** (`"MillerRoss"`)
6. **Marchaud Integral** (`"Marchaud"`)

**Usage**:
```python
from hpfracc.core.integrals import create_fractional_integral

# Create Riemann-Liouville integral
rl_integral = create_fractional_integral("RL", 0.5)
result = rl_integral(f, x)

# Create Weyl integral
weyl_integral = create_fractional_integral("Weyl", 0.5)
result = weyl_integral(f, x)
```

## Usage Examples

### Basic Usage Pattern

```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Define function
def f(x): return x**2

# Create derivative
derivative = create_fractional_derivative('riemann_liouville', 0.5)

# Single point computation
result = derivative.compute(f, 2.0)

# Array computation
x_array = np.linspace(0, 5, 100)
result_array = derivative.compute(f, x_array)

# Numerical computation from function values
f_values = f(x_array)
result_numerical = derivative.compute_numerical(f_values, x_array)
```

### Advanced Usage with Parallel Processing

```python
# Use parallel-optimized methods for large computations
parallel_derivative = create_fractional_derivative('parallel_riemann_liouville', 0.5)

# Large array computation
x_large = np.linspace(0, 100, 10000)
result_large = parallel_derivative.compute(f, x_large)
```

### Riesz-Fisher Operator Examples

```python
from hpfracc.core.fractional_implementations import create_riesz_fisher_operator

# Create operators for different behaviors
rf_derivative = create_riesz_fisher_operator(0.5)    # Derivative
rf_integral = create_riesz_fisher_operator(-0.5)    # Integral
rf_identity = create_riesz_fisher_operator(0.0)     # Identity

# Test function
def f(x): return np.exp(-x**2)

# Compute results
x = np.linspace(-5, 5, 100)
derivative_result = rf_derivative.compute(f, x)
integral_result = rf_integral.compute(f, x)
identity_result = rf_identity.compute(f, x)
```

## Performance Considerations

### Method Selection Guidelines

1. **For small computations (< 1000 points)**: Use classical methods
2. **For medium computations (1000-10000 points)**: Use advanced methods
3. **For large computations (> 10000 points)**: Use parallel-optimized methods
4. **For real-time applications**: Use optimized methods with JAX/Numba
5. **For memory-constrained systems**: Use memory-optimized methods

### Optimization Tips

```python
# Enable JAX acceleration when available
derivative = create_fractional_derivative('riemann_liouville', 0.5, use_jax=True)

# Enable Numba optimization
derivative = create_fractional_derivative('riemann_liouville', 0.5, use_numba=True)

# Use parallel processing for large arrays
parallel_derivative = create_fractional_derivative('parallel_riemann_liouville', 0.5)
```

## Mathematical Theory

### Key Properties

1. **Linearity**: D^α(af + bg) = aD^αf + bD^αg
2. **Leibniz Rule**: D^α(fg) = Σ_{k=0}^∞ (α choose k) D^(α-k)f D^kg
3. **Chain Rule**: D^α(f∘g) = Σ_{k=1}^∞ (α choose k) (D^kf∘g) (D^αg)^k
4. **Semigroup Property**: D^α(D^βf) = D^(α+β)f

### Convergence and Stability

- **Riemann-Liouville**: Stable for 0 < α < 1, may have boundary effects
- **Caputo**: Better initial value behavior, stable for all α > 0
- **Grunwald-Letnikov**: Numerical stability depends on step size
- **Novel Methods**: Enhanced stability with non-singular kernels

### Error Analysis

```python
# Estimate numerical error
def estimate_error(derivative, f, x, h_values):
    errors = []
    for h in h_values:
        result_h = derivative.compute(f, x, h=h)
        # Compare with analytical solution or smaller h
        # errors.append(relative_error)
    return errors
```

## Conclusion

The HPFRACC library provides a comprehensive suite of fractional calculus operators suitable for a wide range of applications. From classical definitions to cutting-edge advanced methods, users can choose the most appropriate operator for their specific needs.

For optimal performance, consider:
- The nature of your problem (initial value, boundary value, etc.)
- Computational requirements (speed vs. accuracy)
- Available computational resources
- Required mathematical properties

The library is designed to be both mathematically rigorous and computationally efficient, making it suitable for both research and production applications.
