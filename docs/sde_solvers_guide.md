# SDE Solvers Guide

## Overview

The SDE (Stochastic Differential Equation) solvers in HPFRACC provide robust numerical methods for solving stochastic differential equations. These solvers are essential for modeling systems with random fluctuations and are the foundation for future Neural fSDE implementations.

## 🚀 **Quick Start**

### Installation

```bash
pip install hpfracc
```

### Basic Usage

```python
import hpfracc.solvers.sde_solvers as sde
import numpy as np

# Create an SDE solver
solver = sde.EulerMaruyama(
    drift_func=lambda x, t: -0.1 * x,      # Drift function
    diffusion_func=lambda x, t: 0.5,       # Diffusion function
    initial_condition=1.0,                  # Initial value
    time_span=(0, 10),                     # Time interval
    num_steps=1000                         # Number of time steps
)

# Solve the SDE
result = solver.solve()
print(f"Solution shape: {result['solution'].shape}")
print(f"Time points: {result['time_points'].shape}")
```

## 🏗️ **Architecture**

### BaseSDESolver

The abstract base class that provides common functionality for all SDE solvers:

- **Common Interface**: Unified API for all SDE methods
- **Wiener Process Generation**: Efficient Brownian motion simulation
- **Error Estimation**: Built-in error analysis and validation
- **Stability Analysis**: Numerical stability checks
- **Utility Methods**: Common operations for all SDE solvers

### EulerMaruyama

First-order explicit method for SDEs:

- **Convergence Order**: 0.5 (strong convergence)
- **Stability**: Conditionally stable
- **Memory Usage**: Low memory footprint
- **Best For**: Quick prototyping and simple SDEs

### Milstein

Second-order method with improved accuracy:

- **Convergence Order**: 1.0 (strong convergence)
- **Stability**: Better stability than Euler-Maruyama
- **Memory Usage**: Moderate memory usage
- **Best For**: Production applications requiring higher accuracy

### Heun

Predictor-corrector method with enhanced stability:

- **Convergence Order**: 1.0 (strong convergence)
- **Stability**: Excellent numerical stability
- **Memory Usage**: Moderate memory usage
- **Best For**: Stiff SDEs and long-time integration

## 🔧 **Configuration Options**

### Solver Configuration

```python
# Euler-Maruyama Solver
solver = sde.EulerMaruyama(
    drift_func=drift_function,           # Required: Drift function f(x, t)
    diffusion_func=diffusion_function,   # Required: Diffusion function g(x, t)
    initial_condition=1.0,               # Required: Initial condition
    time_span=(0, 10),                   # Required: Time interval (t_start, t_end)
    num_steps=1000,                      # Required: Number of time steps
    seed=42                              # Optional: Random seed for reproducibility
)

# Milstein Solver
solver = sde.Milstein(
    drift_func=drift_function,
    diffusion_func=diffusion_function,
    initial_condition=1.0,
    time_span=(0, 10),
    num_steps=1000,
    seed=42
)

# Heun Solver
solver = sde.Heun(
    drift_func=drift_function,
    diffusion_func=diffusion_function,
    initial_condition=1.0,
    time_span=(0, 10),
    num_steps=1000,
    seed=42
)
```

### Function Signatures

```python
def drift_function(x: np.ndarray, t: float) -> np.ndarray:
    """
    Drift function f(x, t) for the SDE: dx = f(x, t)dt + g(x, t)dW
    
    Args:
        x: Current state vector
        t: Current time
        
    Returns:
        Drift vector
    """
    return -0.1 * x  # Example: mean-reverting process

def diffusion_function(x: np.ndarray, t: float) -> np.ndarray:
    """
    Diffusion function g(x, t) for the SDE: dx = f(x, t)dt + g(x, t)dW
    
    Args:
        x: Current state vector
        t: Current time
        
    Returns:
        Diffusion matrix or vector
    """
    return 0.5 * np.ones_like(x)  # Example: constant volatility
```

## 📚 **Examples**

### Example 1: Geometric Brownian Motion

```python
import hpfracc.solvers.sde_solvers as sde
import numpy as np
import matplotlib.pyplot as plt

# Geometric Brownian Motion: dS = μS dt + σS dW
def gbm_drift(x, t):
    mu = 0.05  # Drift rate
    return mu * x

def gbm_diffusion(x, t):
    sigma = 0.2  # Volatility
    return sigma * x

# Create solver
solver = sde.Milstein(
    drift_func=gbm_drift,
    diffusion_func=gbm_diffusion,
    initial_condition=100.0,  # Initial stock price
    time_span=(0, 1),         # 1 year
    num_steps=252,            # Daily steps (252 trading days)
    seed=42
)

# Solve SDE
result = solver.solve()

# Plot results
plt.figure(figsize=(12, 6))

# Plot multiple realizations
for i in range(5):
    solver.seed = i  # Different seed for each realization
    result = solver.solve()
    plt.plot(result['time_points'], result['solution'], 
             alpha=0.7, label=f'Realization {i+1}')

plt.plot(result['time_points'], 100 * np.exp(0.05 * result['time_points']), 
         'k--', linewidth=2, label='Expected value')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Geometric Brownian Motion')
plt.legend()
plt.grid(True)
plt.show()

# Print statistics
print(f"Final price range: {result['solution'][-1]:.2f}")
print(f"Expected final price: {100 * np.exp(0.05):.2f}")
```

### Example 2: Ornstein-Uhlenbeck Process

```python
# Ornstein-Uhlenbeck process: dx = -θ(x - μ)dt + σdW
def ou_drift(x, t):
    theta = 2.0   # Mean reversion speed
    mu = 0.0      # Long-term mean
    return -theta * (x - mu)

def ou_diffusion(x, t):
    sigma = 0.5   # Volatility
    return sigma

# Create solver
solver = sde.Heun(
    drift_func=ou_drift,
    diffusion_func=ou_diffusion,
    initial_condition=2.0,    # Start away from equilibrium
    time_span=(0, 5),         # 5 time units
    num_steps=1000,           # High resolution
    seed=42
)

# Solve SDE
result = solver.solve()

# Plot results
plt.figure(figsize=(12, 6))

# Plot multiple realizations
for i in range(10):
    solver.seed = i
    result = solver.solve()
    plt.plot(result['time_points'], result['solution'], 
             alpha=0.5, linewidth=0.8)

# Plot theoretical mean
theoretical_mean = 2.0 * np.exp(-2.0 * result['time_points'])
plt.plot(result['time_points'], theoretical_mean, 
         'r--', linewidth=2, label='Theoretical mean')

plt.xlabel('Time')
plt.ylabel('State')
plt.title('Ornstein-Uhlenbeck Process')
plt.legend()
plt.grid(True)
plt.show()

# Analyze convergence to equilibrium
final_values = []
for i in range(100):
    solver.seed = i
    result = solver.solve()
    final_values.append(result['solution'][-1])

print(f"Mean final value: {np.mean(final_values):.4f}")
print(f"Std final value: {np.std(final_values):.4f}")
print(f"Theoretical std: {0.5 / np.sqrt(2 * 2.0):.4f}")
```

### Example 3: Multi-dimensional SDE

```python
# Two-dimensional SDE system
def multi_drift(x, t):
    """Drift function for 2D system"""
    x1, x2 = x[0], x[1]
    
    # Coupled oscillators with damping
    dx1_dt = x2
    dx2_dt = -x1 - 0.1 * x2
    
    return np.array([dx1_dt, dx2_dt])

def multi_diffusion(x, t):
    """Diffusion function for 2D system"""
    # Diagonal diffusion matrix
    return np.array([0.1, 0.1])

# Create solver
solver = sde.Milstein(
    drift_func=multi_drift,
    diffusion_func=multi_diffusion,
    initial_condition=np.array([1.0, 0.0]),  # Initial position and velocity
    time_span=(0, 10),
    num_steps=1000,
    seed=42
)

# Solve SDE
result = solver.solve()

# Plot phase space
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(result['solution'][:, 0], result['solution'][:, 1], 'b-', alpha=0.7)
plt.plot(result['solution'][0, 0], result['solution'][0, 1], 'go', markersize=8, label='Start')
plt.plot(result['solution'][-1, 0], result['solution'][-1, 1], 'ro', markersize=8, label='End')
plt.xlabel('Position (x1)')
plt.ylabel('Velocity (x2)')
plt.title('Phase Space Trajectory')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(result['time_points'], result['solution'][:, 0], 'b-', label='Position')
plt.plot(result['time_points'], result['solution'][:, 1], 'r-', label='Velocity')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 🏭 **Factory Functions**

### Creating Solvers

```python
# Using factory functions
solver = sde.create_sde_solver(
    solver_type="milstein",           # "euler", "milstein", or "heun"
    drift_func=drift_function,
    diffusion_func=diffusion_function,
    initial_condition=1.0,
    time_span=(0, 10),
    num_steps=1000,
    seed=42
)

# Get solver properties
properties = sde.get_sde_solver_properties(solver)
print(f"Solver type: {properties['solver_type']}")
print(f"Convergence order: {properties['convergence_order']}")
print(f"Time span: {properties['time_span']}")
print(f"Number of steps: {properties['num_steps']}")
```

### Parameter Validation

```python
# Validate SDE parameters
validation = sde.validate_sde_parameters(
    drift_func=drift_function,
    diffusion_func=diffusion_function,
    initial_condition=1.0,
    time_span=(0, 10),
    num_steps=1000
)

if validation['valid']:
    print("Parameters are valid!")
else:
    print(f"Validation errors: {validation['errors']}")
```

## 🔬 **Research Applications**

### Financial Modeling

```python
# Heston model for stochastic volatility
def heston_drift(x, t):
    """Drift for Heston model: [S, V]"""
    S, V = x[0], x[1]
    
    # Parameters
    r = 0.05      # Risk-free rate
    kappa = 2.0   # Mean reversion speed
    theta = 0.04  # Long-term variance
    rho = -0.7    # Correlation
    
    # Drift terms
    dS_dt = r * S
    dV_dt = kappa * (theta - V)
    
    return np.array([dS_dt, dV_dt])

def heston_diffusion(x, t):
    """Diffusion for Heston model"""
    S, V = x[0], x[1]
    
    # Parameters
    sigma = 0.3   # Volatility of volatility
    rho = -0.7    # Correlation
    
    # Diffusion matrix
    sqrt_V = np.sqrt(np.maximum(V, 1e-6))  # Avoid negative variance
    
    d11 = sqrt_V * S
    d12 = 0.0
    d21 = rho * sigma * sqrt_V
    d22 = sigma * sqrt_V * np.sqrt(1 - rho**2)
    
    return np.array([[d11, d12], [d21, d22]])

# Create and solve Heston model
solver = sde.Milstein(
    drift_func=heston_drift,
    diffusion_func=heston_diffusion,
    initial_condition=np.array([100.0, 0.04]),  # [S0, V0]
    time_span=(0, 1),
    num_steps=252,
    seed=42
)

result = solver.solve()

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(result['time_points'], result['solution'][:, 0])
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Heston Model: Stock Price')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(result['time_points'], result['solution'][:, 1])
plt.xlabel('Time')
plt.ylabel('Variance')
plt.title('Heston Model: Variance')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Biological Systems

```python
# Lotka-Volterra predator-prey model with noise
def lotka_volterra_drift(x, t):
    """Drift for Lotka-Volterra system: [prey, predator]"""
    prey, predator = x[0], x[1]
    
    # Parameters
    alpha = 1.0   # Prey growth rate
    beta = 0.1    # Predation rate
    gamma = 0.1   # Predator death rate
    delta = 0.02  # Predator growth rate
    
    # Drift terms
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    
    return np.array([dprey_dt, dpredator_dt])

def lotka_volterra_diffusion(x, t):
    """Diffusion for Lotka-Volterra system"""
    prey, predator = x[0], x[1]
    
    # Environmental noise
    noise_strength = 0.1
    
    return noise_strength * np.array([prey, predator])

# Create and solve Lotka-Volterra model
solver = sde.Heun(
    drift_func=lotka_volterra_drift,
    diffusion_func=lotka_volterra_diffusion,
    initial_condition=np.array([50.0, 20.0]),  # [prey0, predator0]
    time_span=(0, 100),
    num_steps=2000,
    seed=42
)

result = solver.solve()

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(result['time_points'], result['solution'][:, 0], 'g-', label='Prey')
plt.plot(result['time_points'], result['solution'][:, 1], 'r-', label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra with Noise')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(result['solution'][:, 0], result['solution'][:, 1], 'b-', alpha=0.7)
plt.plot(result['solution'][0, 0], result['solution'][0, 1], 'go', markersize=8, label='Start')
plt.plot(result['solution'][-1, 0], result['solution'][-1, 1], 'ro', markersize=8, label='End')
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Space')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## ⚡ **Performance Optimization**

### Batch Processing

```python
# Solve multiple SDEs with different initial conditions
def solve_multiple_sdes(initial_conditions, solver_class, **kwargs):
    """Solve multiple SDEs efficiently"""
    results = []
    
    for i, x0 in enumerate(initial_conditions):
        solver = solver_class(
            initial_condition=x0,
            **kwargs
        )
        solver.seed = i  # Different seed for each
        result = solver.solve()
        results.append(result)
    
    return results

# Example usage
initial_conditions = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 1.0]),
    np.array([-1.0, 0.0])
]

results = solve_multiple_sdes(
    initial_conditions,
    sde.Milstein,
    drift_func=multi_drift,
    diffusion_func=multi_diffusion,
    time_span=(0, 10),
    num_steps=1000
)

# Plot all trajectories
plt.figure(figsize=(8, 8))
for i, result in enumerate(results):
    plt.plot(result['solution'][:, 0], result['solution'][:, 1], 
             alpha=0.7, label=f'Trajectory {i+1}')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Multiple SDE Trajectories')
plt.legend()
plt.grid(True)
plt.show()
```

### Error Analysis

```python
# Analyze convergence with different step sizes
def convergence_analysis(solver_class, drift_func, diffusion_func, 
                        initial_condition, time_span, step_sizes):
    """Analyze convergence of SDE solver"""
    errors = []
    
    # Reference solution with very fine grid
    ref_solver = solver_class(
        drift_func=drift_func,
        diffusion_func=diffusion_func,
        initial_condition=initial_condition,
        time_span=time_span,
        num_steps=10000,
        seed=42
    )
    ref_solution = ref_solver.solve()
    
    # Interpolate reference to coarse grid for comparison
    for num_steps in step_sizes:
        solver = solver_class(
            drift_func=drift_func,
            diffusion_func=diffusion_func,
            initial_condition=initial_condition,
            time_span=time_span,
            num_steps=num_steps,
            seed=42
        )
        solution = solver.solve()
        
        # Compute error at final time
        error = np.abs(solution['solution'][-1] - ref_solution['solution'][-1])
        errors.append(error)
    
    return step_sizes, errors

# Example convergence analysis
step_sizes = [100, 200, 500, 1000, 2000]
dt_values = [10/ns for ns in step_sizes]

dt_values, errors = convergence_analysis(
    sde.EulerMaruyama,
    ou_drift,
    ou_diffusion,
    initial_condition=1.0,
    time_span=(0, 10),
    step_sizes=step_sizes
)

# Plot convergence
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.loglog(dt_values, errors, 'bo-', label='Euler-Maruyama')
plt.xlabel('Time step (dt)')
plt.ylabel('Error at final time')
plt.title('Convergence Analysis')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# Theoretical convergence lines
dt_theory = np.array(dt_values)
plt.loglog(dt_theory, 0.1 * dt_theory**0.5, 'r--', label='O(dt^0.5)')
plt.loglog(dt_theory, 0.1 * dt_theory**1.0, 'g--', label='O(dt^1.0)')
plt.xlabel('Time step (dt)')
plt.ylabel('Theoretical error')
plt.title('Theoretical Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 🧪 **Testing and Validation**

### Running Tests

```bash
# Run all SDE solver tests
python -m pytest tests/test_solvers/test_sde_solvers.py -v

# Run specific test categories
python -m pytest tests/test_solvers/test_sde_solvers.py::TestEulerMaruyama -v
python -m pytest tests/test_solvers/test_sde_solvers.py::TestMilstein -v
python -m pytest tests/test_solvers/test_sde_solvers.py::TestHeun -v
```

### Validation Examples

```python
# Validate solver behavior
def validate_sde_solver(solver, test_cases):
    """Validate SDE solver behavior"""
    results = []
    
    for i, (x0, expected_shape) in enumerate(test_cases):
        try:
            solver.initial_condition = x0
            solver.seed = i
            result = solver.solve()
            
            shape_correct = result['solution'].shape == expected_shape
            finite_output = np.all(np.isfinite(result['solution']))
            time_monotonic = np.all(np.diff(result['time_points']) > 0)
            
            results.append({
                'test_case': i,
                'initial_condition': x0,
                'output_shape': result['solution'].shape,
                'expected_shape': expected_shape,
                'shape_correct': shape_correct,
                'finite_output': finite_output,
                'time_monotonic': time_monotonic
            })
            
        except Exception as e:
            results.append({
                'test_case': i,
                'initial_condition': x0,
                'error': str(e)
            })
    
    return results

# Example validation
test_cases = [
    (1.0, (1001,)),
    (np.array([1.0, 2.0]), (1001, 2)),
    (np.array([0.5]), (1001, 1))
]

validation_results = validate_sde_solver(solver, test_cases)
for result in validation_results:
    print(result)
```

## 🔮 **Future Developments**

### Planned Features

- **Neural fSDE**: Learning-based stochastic differential equation solving
- **Adaptive Time Stepping**: Automatic step size selection
- **Multi-scale Methods**: Handling systems with multiple time scales
- **Parallel Solvers**: GPU acceleration and parallel processing
- **Advanced Noise Models**: Lévy processes, fractional Brownian motion

### Research Directions

- **Stochastic Control**: Optimal control of SDEs
- **Stochastic PDEs**: Extension to partial differential equations
- **Mean Field Games**: Large-scale stochastic systems
- **Stochastic Optimization**: Optimization under uncertainty

## 📖 **References**

1. Kloeden, P. E., & Platen, E. "Numerical Solution of Stochastic Differential Equations." Springer, 1992.
2. Higham, D. J. "An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations." SIAM Review, 2001.
3. Milstein, G. N. "Numerical Integration of Stochastic Differential Equations." Springer, 1995.

## 🤝 **Contributing**

We welcome contributions to the SDE solvers! Areas for contribution include:

- **New Methods**: Implementation of additional SDE solvers
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

**SDE Solvers v1.4.0** - *Robust Numerical Methods for Stochastic Differential Equations*
