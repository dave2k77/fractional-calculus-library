# Fractional Calculus Library - Class Documentation

This document provides comprehensive documentation for all classes in the Fractional Calculus Library, including their methods, attributes, and usage examples.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Algorithm Classes](#algorithm-classes)
3. [Solver Classes](#solver-classes)
4. [Optimization Classes](#optimization-classes)
5. [Special Function Classes](#special-function-classes)
6. [Utility Classes](#utility-classes)

---

## Core Classes

### FractionalOrder

**Location**: `src/core/definitions.py`

**Description**: Represents a fractional order for derivatives and integrals.

**Attributes**:
- `value` (float): The numerical value of the fractional order
- `type` (str): The type of fractional order ("derivative" or "integral")

**Methods**:
- `__init__(value, type="derivative")`: Initialize a fractional order
- `__str__()`: String representation
- `__repr__()`: Detailed string representation
- `is_valid()`: Check if the order is valid (0 < value < 2 for derivatives)

**Usage Example**:
```python
from src.core.definitions import FractionalOrder

# Create a fractional derivative order
alpha = FractionalOrder(0.5, "derivative")
print(alpha)  # Output: FractionalOrder(0.5, derivative)

# Check validity
print(alpha.is_valid())  # Output: True
```

### FractionalDerivative

**Location**: `src/core/derivatives.py`

**Description**: Base class for computing fractional derivatives using various methods.

**Attributes**:
- `order` (FractionalOrder): The fractional order
- `method` (str): The method to use ("caputo", "riemann_liouville", "grunwald_letnikov")
- `grid_size` (int): Number of grid points for discretization

**Methods**:
- `__init__(order, method="caputo", grid_size=1000)`: Initialize the derivative calculator
- `compute(x, func)`: Compute the fractional derivative of a function
- `validate_inputs(x, func)`: Validate input parameters
- `get_method_name()`: Get the name of the computation method

**Usage Example**:
```python
from src.core.derivatives import FractionalDerivative
from src.core.definitions import FractionalOrder
import numpy as np

# Create derivative calculator
alpha = FractionalOrder(0.5)
deriv_calc = FractionalDerivative(alpha, method="caputo")

# Define function
def f(x):
    return np.sin(x)

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
result = deriv_calc.compute(x, f)
```

---

## Algorithm Classes

### CaputoDerivative

**Location**: `src/algorithms/caputo.py`

**Description**: Implements Caputo fractional derivative using various numerical methods.

**Attributes**:
- `order` (float): Fractional order (0 < order < 2)
- `method` (str): Numerical method ("trapezoidal", "simpson", "gauss")
- `grid_size` (int): Number of grid points

**Methods**:
- `__init__(order, method="trapezoidal", grid_size=1000)`: Initialize Caputo derivative
- `compute(x, func)`: Compute Caputo derivative
- `_trapezoidal_rule(x, func)`: Use trapezoidal rule for integration
- `_simpson_rule(x, func)`: Use Simpson's rule for integration
- `_gauss_quadrature(x, func)`: Use Gauss quadrature for integration

**Usage Example**:
```python
from src.algorithms.caputo import CaputoDerivative
import numpy as np

# Create Caputo derivative calculator
caputo = CaputoDerivative(order=0.5, method="trapezoidal")

# Define function
def f(x):
    return np.exp(-x)

# Compute derivative
x = np.linspace(0, 5, 100)
result = caputo.compute(x, f)
```

### RiemannLiouvilleDerivative

**Location**: `src/algorithms/riemann_liouville.py`

**Description**: Implements Riemann-Liouville fractional derivative.

**Attributes**:
- `order` (float): Fractional order (0 < order < 2)
- `method` (str): Numerical method ("trapezoidal", "simpson", "gauss")
- `grid_size` (int): Number of grid points

**Methods**:
- `__init__(order, method="trapezoidal", grid_size=1000)`: Initialize Riemann-Liouville derivative
- `compute(x, func)`: Compute Riemann-Liouville derivative
- `_compute_kernel(x, t, order)`: Compute the kernel function
- `_integrate_kernel(x, func, order)`: Integrate the kernel with the function

**Usage Example**:
```python
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
import numpy as np

# Create Riemann-Liouville derivative calculator
rl_deriv = RiemannLiouvilleDerivative(order=0.7, method="simpson")

# Define function
def f(x):
    return x**2

# Compute derivative
x = np.linspace(0, 3, 100)
result = rl_deriv.compute(x, f)
```

### GrunwaldLetnikovDerivative

**Location**: `src/algorithms/grunwald_letnikov.py`

**Description**: Implements Grünwald-Letnikov fractional derivative using finite differences.

**Attributes**:
- `order` (float): Fractional order (0 < order < 2)
- `step_size` (float): Step size for finite differences
- `truncation_order` (int): Order of truncation for the series

**Methods**:
- `__init__(order, step_size=0.01, truncation_order=100)`: Initialize Grünwald-Letnikov derivative
- `compute(x, func)`: Compute Grünwald-Letnikov derivative
- `_compute_binomial_coefficients(order, n)`: Compute binomial coefficients
- `_finite_difference(x, func, step_size)`: Compute finite differences

**Usage Example**:
```python
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative
import numpy as np

# Create Grünwald-Letnikov derivative calculator
gl_deriv = GrunwaldLetnikovDerivative(order=0.5, step_size=0.01)

# Define function
def f(x):
    return np.cos(x)

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
result = gl_deriv.compute(x, f)
```

### FFTMethods

**Location**: `src/algorithms/fft_methods.py`

**Description**: Implements spectral methods for fractional derivatives using FFT.

**Attributes**:
- `order` (float): Fractional order
- `method` (str): Spectral method ("fourier", "chebyshev", "legendre")
- `grid_size` (int): Number of grid points (must be power of 2 for FFT)

**Methods**:
- `__init__(order, method="fourier", grid_size=1024)`: Initialize FFT methods
- `compute(x, func)`: Compute fractional derivative using spectral method
- `_fourier_transform(x, func)`: Apply Fourier transform
- `_chebyshev_transform(x, func)`: Apply Chebyshev transform
- `_legendre_transform(x, func)`: Apply Legendre transform

**Usage Example**:
```python
from src.algorithms.fft_methods import FFTMethods
import numpy as np

# Create FFT-based derivative calculator
fft_deriv = FFTMethods(order=0.5, method="fourier", grid_size=1024)

# Define function
def f(x):
    return np.sin(2*np.pi*x)

# Compute derivative
x = np.linspace(0, 1, 1024)
result = fft_deriv.compute(x, f)
```

### L1L2Schemes

**Location**: `src/algorithms/L1_L2_schemes.py`

**Description**: Implements L1 and L2 schemes for time-fractional PDEs.

**Attributes**:
- `scheme` (str): Scheme type ("l1" or "l2")
- `order` (float): Fractional order
- `grid_size` (int): Number of spatial grid points
- `time_steps` (int): Number of time steps

**Methods**:
- `__init__(scheme="l1", order=0.5, grid_size=100, time_steps=50)`: Initialize L1/L2 scheme
- `solve_time_fractional_pde(initial_condition, boundary_conditions, alpha, t_final, dt, dx)`: Solve time-fractional PDE
- `_l1_scheme(u_prev, u_current, dt, dx, alpha)`: Apply L1 scheme
- `_l2_scheme(u_prev, u_current, dt, dx, alpha)`: Apply L2 scheme
- `_compute_weights(alpha, n)`: Compute weights for the scheme

**Usage Example**:
```python
from src.algorithms.L1_L2_schemes import L1L2Schemes
import numpy as np

# Create L1 scheme solver
l1_solver = L1L2Schemes(scheme="l1", order=0.5)

# Define initial condition
def initial_condition(x):
    return np.sin(np.pi * x)

# Define boundary conditions
def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

# Solve PDE
x, t, u = l1_solver.solve_time_fractional_pde(
    initial_condition=initial_condition,
    boundary_conditions=(boundary_left, boundary_right),
    alpha=0.5,
    t_final=1.0,
    dt=0.01,
    dx=0.01
)
```

---

## Solver Classes

### FractionalPDESolver

**Location**: `src/solvers/pde_solvers.py`

**Description**: Base class for solving fractional partial differential equations.

**Attributes**:
- `pde_type` (str): Type of PDE ("diffusion", "advection", "reaction_diffusion", "wave")
- `method` (str): Numerical method ("finite_difference", "spectral", "finite_element")
- `spatial_order` (int): Order of spatial discretization
- `temporal_order` (int): Order of temporal discretization
- `adaptive` (bool): Whether to use adaptive step size

**Methods**:
- `__init__(pde_type="diffusion", method="finite_difference", spatial_order=2, temporal_order=1, adaptive=False)`: Initialize PDE solver
- `solve(x_span, t_span, initial_condition, boundary_conditions, **kwargs)`: Solve the PDE
- `_validate_parameters()`: Validate solver parameters
- `_setup_grid(x_span, t_span, nx, nt)`: Set up spatial and temporal grids

**Usage Example**:
```python
from src.solvers.pde_solvers import FractionalPDESolver
import numpy as np

# Create PDE solver
solver = FractionalPDESolver(
    pde_type="diffusion",
    method="finite_difference",
    spatial_order=2,
    temporal_order=1
)

# Define problem
def initial_condition(x):
    return np.sin(np.pi * x)

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

# Solve PDE
t, x, u = solver.solve(
    x_span=(0, 1),
    t_span=(0, 0.1),
    initial_condition=initial_condition,
    boundary_conditions=(boundary_left, boundary_right),
    alpha=0.5,
    beta=2.0,
    nx=50,
    nt=20
)
```

### FractionalDiffusionSolver

**Location**: `src/solvers/pde_solvers.py`

**Description**: Specialized solver for fractional diffusion equations.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative ("caputo", "riemann_liouville", "grunwald_letnikov")
- `method` (str): Numerical method
- `spatial_order` (int): Order of spatial discretization
- `temporal_order` (int): Order of temporal discretization

**Methods**:
- `__init__(method="finite_difference", spatial_order=2, temporal_order=1, derivative_type="caputo")`: Initialize diffusion solver
- `solve(x_span, t_span, initial_condition, boundary_conditions, alpha, beta, nx, nt, source_term=None)`: Solve diffusion equation
- `_compute_spatial_derivative(u, dx, beta)`: Compute spatial derivative
- `_compute_temporal_derivative(u_prev, u_current, dt, alpha)`: Compute temporal derivative

**Usage Example**:
```python
from src.solvers.pde_solvers import FractionalDiffusionSolver
import numpy as np

# Create diffusion solver
diffusion_solver = FractionalDiffusionSolver(
    method="finite_difference",
    derivative_type="caputo"
)

# Define problem
def initial_condition(x):
    return np.exp(-x**2)

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

def source_term(x, t, u):
    return np.zeros_like(x)

# Solve diffusion equation
t, x, u = diffusion_solver.solve(
    x_span=(0, 2),
    t_span=(0, 0.2),
    initial_condition=initial_condition,
    boundary_conditions=(boundary_left, boundary_right),
    alpha=0.7,
    beta=2.0,
    nx=100,
    nt=50,
    source_term=source_term
)
```

### FractionalAdvectionSolver

**Location**: `src/solvers/pde_solvers.py`

**Description**: Specialized solver for fractional advection equations.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `method` (str): Numerical method
- `velocity` (float): Advection velocity

**Methods**:
- `__init__(method="finite_difference", derivative_type="caputo")`: Initialize advection solver
- `solve(x_span, t_span, initial_condition, boundary_conditions, alpha, beta, velocity, nx, nt)`: Solve advection equation
- `_compute_advection_term(u, velocity, dx, beta)`: Compute advection term

**Usage Example**:
```python
from src.solvers.pde_solvers import FractionalAdvectionSolver
import numpy as np

# Create advection solver
advection_solver = FractionalAdvectionSolver(
    method="finite_difference",
    derivative_type="caputo"
)

# Define problem
def initial_condition(x):
    return np.exp(-(x - 0.5)**2 / 0.01)

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

# Solve advection equation
t, x, u = advection_solver.solve(
    x_span=(0, 1),
    t_span=(0, 0.1),
    initial_condition=initial_condition,
    boundary_conditions=(boundary_left, boundary_right),
    alpha=0.8,
    beta=1.5,
    velocity=1.0,
    nx=100,
    nt=50
)
```

### FractionalReactionDiffusionSolver

**Location**: `src/solvers/pde_solvers.py`

**Description**: Specialized solver for fractional reaction-diffusion equations.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `method` (str): Numerical method
- `reaction_term` (callable): Reaction term function

**Methods**:
- `__init__(method="finite_difference", derivative_type="caputo")`: Initialize reaction-diffusion solver
- `solve(x_span, t_span, initial_condition, boundary_conditions, alpha, beta, reaction_term, nx, nt)`: Solve reaction-diffusion equation
- `_compute_reaction_term(u, reaction_term)`: Compute reaction term

**Usage Example**:
```python
from src.solvers.pde_solvers import FractionalReactionDiffusionSolver
import numpy as np

# Create reaction-diffusion solver
rd_solver = FractionalReactionDiffusionSolver(
    method="finite_difference",
    derivative_type="caputo"
)

# Define problem
def initial_condition(x):
    return 0.5 * np.ones_like(x)

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

def reaction_term(u):
    return u * (1 - u)  # Fisher-KPP reaction term

# Solve reaction-diffusion equation
t, x, u = rd_solver.solve(
    x_span=(0, 1),
    t_span=(0, 0.1),
    initial_condition=initial_condition,
    boundary_conditions=(boundary_left, boundary_right),
    alpha=0.7,
    beta=2.0,
    reaction_term=reaction_term,
    nx=100,
    nt=50
)
```

### FractionalODESolver

**Location**: `src/solvers/ode_solvers.py`

**Description**: Base class for solving fractional ordinary differential equations.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `method` (str): Numerical method ("predictor_corrector", "adams_bashforth", "runge_kutta", "euler")
- `adaptive` (bool): Whether to use adaptive step size
- `tol` (float): Tolerance for adaptive methods
- `max_iter` (int): Maximum number of iterations

**Methods**:
- `__init__(derivative_type="caputo", method="predictor_corrector", adaptive=True, tol=1e-6, max_iter=1000)`: Initialize ODE solver
- `solve(f, t_span, y0, alpha, h=None)`: Solve the ODE
- `_validate_parameters()`: Validate solver parameters
- `_adaptive_step_control()`: Control adaptive step size

**Usage Example**:
```python
from src.solvers.ode_solvers import FractionalODESolver
import numpy as np

# Create ODE solver
ode_solver = FractionalODESolver(
    derivative_type="caputo",
    method="predictor_corrector",
    adaptive=True,
    tol=1e-6
)

# Define ODE
def f(t, y):
    return -y

# Solve ODE
t_span = (0, 1)
y0 = 1.0
alpha = 0.5

t, y = ode_solver.solve(f, t_span, y0, alpha)
```

### AdaptiveFractionalODESolver

**Location**: `src/solvers/ode_solvers.py`

**Description**: Adaptive solver for fractional ODEs with automatic step size control.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `method` (str): Numerical method
- `adaptive` (bool): Always True for this solver
- `tol` (float): Tolerance for step size control
- `min_h` (float): Minimum step size
- `max_h` (float): Maximum step size

**Methods**:
- `__init__(derivative_type="caputo", method="predictor_corrector", tol=1e-6, min_h=1e-8, max_h=1e-2)`: Initialize adaptive solver
- `solve(f, t_span, y0, alpha)`: Solve ODE with adaptive step size
- `_estimate_error(y_pred, y_corr)`: Estimate local error
- `_adjust_step_size(h_current, error, tol)`: Adjust step size based on error

**Usage Example**:
```python
from src.solvers.ode_solvers import AdaptiveFractionalODESolver
import numpy as np

# Create adaptive ODE solver
adaptive_solver = AdaptiveFractionalODESolver(
    derivative_type="caputo",
    method="predictor_corrector",
    tol=1e-8,
    min_h=1e-10,
    max_h=1e-1
)

# Define ODE
def f(t, y):
    return np.array([-y[0], -2*y[1]])

# Solve ODE
t_span = (0, 1)
y0 = np.array([1.0, 2.0])
alpha = 0.5

t, y = adaptive_solver.solve(f, t_span, y0, alpha)
```

### PredictorCorrectorSolver

**Location**: `src/solvers/predictor_corrector.py`

**Description**: Implements predictor-corrector methods for fractional ODEs.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `order` (int): Order of the predictor-corrector method
- `adaptive` (bool): Whether to use adaptive step size
- `tol` (float): Tolerance for adaptive methods
- `max_iter` (int): Maximum number of corrector iterations
- `min_h` (float): Minimum step size
- `max_h` (float): Maximum step size

**Methods**:
- `__init__(derivative_type="caputo", order=1, adaptive=True, tol=1e-6, max_iter=10, min_h=1e-8, max_h=1e-2)`: Initialize predictor-corrector solver
- `solve(f, t_span, y0, alpha, h0=None)`: Solve ODE using predictor-corrector method
- `_predictor_step(f, t_current, t_next, y_current, alpha, h)`: Perform predictor step
- `_corrector_step(f, t_current, t_next, y_current, y_pred, alpha, h)`: Perform corrector step
- `_estimate_error(y_pred, y_corr)`: Estimate local error
- `_solve_adaptive(f, t_span, y0, alpha)`: Solve with adaptive step size
- `_solve_fixed_step(f, t_span, y0, alpha, h0)`: Solve with fixed step size

**Usage Example**:
```python
from src.solvers.predictor_corrector import PredictorCorrectorSolver
import numpy as np

# Create predictor-corrector solver
pc_solver = PredictorCorrectorSolver(
    derivative_type="caputo",
    order=1,
    adaptive=True,
    tol=1e-6,
    max_iter=10
)

# Define ODE
def f(t, y):
    return -y

# Solve ODE
t_span = (0, 1)
y0 = 1.0
alpha = 0.5

t, y = pc_solver.solve(f, t_span, y0, alpha)
```

### AdamsBashforthMoultonSolver

**Location**: `src/solvers/predictor_corrector.py`

**Description**: Implements Adams-Bashforth-Moulton predictor-corrector method.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `order` (int): Order of the Adams method
- `adaptive` (bool): Whether to use adaptive step size

**Methods**:
- `__init__(derivative_type="caputo", order=1, adaptive=True)`: Initialize Adams-Bashforth-Moulton solver
- `solve(f, t_span, y0, alpha, h0=None)`: Solve ODE using Adams method
- `_adams_bashforth_predictor(f, t_history, y_history, alpha, h)`: Adams-Bashforth predictor
- `_adams_moulton_corrector(f, t_current, t_next, y_current, y_pred, alpha, h)`: Adams-Moulton corrector

**Usage Example**:
```python
from src.solvers.predictor_corrector import AdamsBashforthMoultonSolver
import numpy as np

# Create Adams-Bashforth-Moulton solver
abm_solver = AdamsBashforthMoultonSolver(
    derivative_type="caputo",
    order=2,
    adaptive=True
)

# Define ODE
def f(t, y):
    return np.array([y[1], -y[0] - 0.1*y[1]])

# Solve ODE
t_span = (0, 10)
y0 = np.array([1.0, 0.0])
alpha = 0.8

t, y = abm_solver.solve(f, t_span, y0, alpha)
```

### VariableStepPredictorCorrector

**Location**: `src/solvers/predictor_corrector.py`

**Description**: Variable step size predictor-corrector solver.

**Attributes**:
- `derivative_type` (str): Type of fractional derivative
- `adaptive` (bool): Always True for this solver
- `tol` (float): Tolerance for step size control

**Methods**:
- `__init__(derivative_type="caputo", tol=1e-6)`: Initialize variable step solver
- `solve(f, t_span, y0, alpha)`: Solve ODE with variable step size
- `_variable_step_control()`: Control variable step size

**Usage Example**:
```python
from src.solvers.predictor_corrector import VariableStepPredictorCorrector
import numpy as np

# Create variable step solver
vs_solver = VariableStepPredictorCorrector(
    derivative_type="caputo",
    tol=1e-6
)

# Define ODE
def f(t, y):
    return -10 * y  # Stiff problem

# Solve ODE
t_span = (0, 1)
y0 = 1.0
alpha = 0.5

t, y = vs_solver.solve(f, t_span, y0, alpha)
```

---

## Optimization Classes

### JAXImplementations

**Location**: `src/optimisation/jax_implementations.py`

**Description**: JAX-optimized implementations of fractional calculus operations.

**Attributes**:
- `order` (float): Fractional order
- `method` (str): Computation method
- `jit_enabled` (bool): Whether JIT compilation is enabled

**Methods**:
- `__init__(order, method="caputo", jit_enabled=True)`: Initialize JAX implementations
- `fractional_derivative(x, func)`: Compute fractional derivative using JAX
- `fractional_integral(x, func)`: Compute fractional integral using JAX
- `_jax_caputo_derivative(x, func, order)`: JAX implementation of Caputo derivative
- `_jax_riemann_liouville_derivative(x, func, order)`: JAX implementation of Riemann-Liouville derivative

**Usage Example**:
```python
from src.optimisation.jax_implementations import JAXImplementations
import numpy as np

# Create JAX implementation
jax_impl = JAXImplementations(order=0.5, method="caputo", jit_enabled=True)

# Define function
def f(x):
    return np.sin(x)

# Compute derivative
x = np.linspace(0, 2*np.pi, 1000)
result = jax_impl.fractional_derivative(x, f)
```

### NumbaKernels

**Location**: `src/optimisation/numba_kernels.py`

**Description**: NUMBA-optimized kernels for fractional calculus operations.

**Attributes**:
- `order` (float): Fractional order
- `parallel_enabled` (bool): Whether parallel execution is enabled
- `fastmath_enabled` (bool): Whether fast math is enabled

**Methods**:
- `__init__(order, parallel_enabled=True, fastmath_enabled=True)`: Initialize NUMBA kernels
- `fractional_derivative_kernel(x, func_values, weights)`: NUMBA kernel for fractional derivative
- `fractional_integral_kernel(x, func_values, weights)`: NUMBA kernel for fractional integral
- `_compute_weights_kernel(order, n)`: NUMBA kernel for computing weights

**Usage Example**:
```python
from src.optimisation.numba_kernels import NumbaKernels
import numpy as np

# Create NUMBA kernels
numba_kernels = NumbaKernels(
    order=0.5,
    parallel_enabled=True,
    fastmath_enabled=True
)

# Define function values
x = np.linspace(0, 1, 1000)
func_values = np.sin(x)

# Compute weights
weights = numba_kernels._compute_weights_kernel(0.5, len(x))

# Apply kernel
result = numba_kernels.fractional_derivative_kernel(x, func_values, weights)
```

### ParallelComputing

**Location**: `src/optimisation/parallel_computing.py`

**Description**: Parallel computing utilities for fractional calculus operations.

**Attributes**:
- `backend` (str): Parallel computing backend ("joblib", "multiprocessing", "threading")
- `n_jobs` (int): Number of parallel jobs
- `chunk_size` (int): Size of chunks for parallel processing

**Methods**:
- `__init__(backend="joblib", n_jobs=-1, chunk_size=1000)`: Initialize parallel computing
- `parallel_fractional_derivative(x, func, order, method="caputo")`: Compute fractional derivative in parallel
- `parallel_fractional_integral(x, func, order, method="riemann_liouville")`: Compute fractional integral in parallel
- `_setup_backend()`: Set up the parallel computing backend
- `_chunk_data(data, chunk_size)`: Split data into chunks for parallel processing

**Usage Example**:
```python
from src.optimisation.parallel_computing import ParallelComputing
import numpy as np

# Create parallel computing instance
parallel_comp = ParallelComputing(
    backend="joblib",
    n_jobs=-1,  # Use all available cores
    chunk_size=1000
)

# Define function
def f(x):
    return np.exp(-x**2)

# Compute derivative in parallel
x = np.linspace(-5, 5, 10000)
result = parallel_comp.parallel_fractional_derivative(
    x, f, order=0.5, method="caputo"
)
```

---

## Special Function Classes

### BinomialCoefficients

**Location**: `src/special/binomial_coeffs.py`

**Description**: Computes binomial coefficients for fractional calculus.

**Attributes**:
- `cache_enabled` (bool): Whether to enable caching
- `max_cache_size` (int): Maximum size of the cache

**Methods**:
- `__init__(cache_enabled=True, max_cache_size=10000)`: Initialize binomial coefficients calculator
- `compute(n, k)`: Compute binomial coefficient C(n,k)
- `fractional_binomial(alpha, k)`: Compute fractional binomial coefficient
- `_cache_key(n, k)`: Generate cache key
- `_clear_cache()`: Clear the cache

**Usage Example**:
```python
from src.special.binomial_coeffs import BinomialCoefficients

# Create binomial coefficients calculator
binom = BinomialCoefficients(cache_enabled=True)

# Compute regular binomial coefficient
result = binom.compute(10, 3)  # C(10,3) = 120

# Compute fractional binomial coefficient
alpha = 0.5
frac_result = binom.fractional_binomial(alpha, 5)
```

### GammaBeta

**Location**: `src/special/gamma_beta.py`

**Description**: Computes Gamma and Beta functions for fractional calculus.

**Attributes**:
- `precision` (int): Precision for numerical computations
- `method` (str): Computation method ("lanczos", "stirling", "numerical")

**Methods**:
- `__init__(precision=15, method="lanczos")`: Initialize Gamma/Beta calculator
- `gamma(x)`: Compute Gamma function
- `beta(x, y)`: Compute Beta function
- `log_gamma(x)`: Compute logarithm of Gamma function
- `_lanczos_gamma(x)`: Lanczos approximation for Gamma function
- `_stirling_gamma(x)`: Stirling approximation for Gamma function

**Usage Example**:
```python
from src.special.gamma_beta import GammaBeta

# Create Gamma/Beta calculator
gb = GammaBeta(precision=15, method="lanczos")

# Compute Gamma function
gamma_val = gb.gamma(0.5)  # sqrt(pi)

# Compute Beta function
beta_val = gb.beta(2, 3)  # 1/12

# Compute log Gamma
log_gamma_val = gb.log_gamma(5)  # log(4!)
```

### MittagLeffler

**Location**: `src/special/mittag_leffler.py`

**Description**: Computes Mittag-Leffler functions for fractional calculus.

**Attributes**:
- `precision` (int): Precision for numerical computations
- `max_terms` (int): Maximum number of terms in series expansion

**Methods**:
- `__init__(precision=15, max_terms=100)`: Initialize Mittag-Leffler calculator
- `e_alpha(z, alpha)`: Compute E_alpha(z)
- `e_alpha_beta(z, alpha, beta)`: Compute E_alpha,beta(z)
- `_series_expansion(z, alpha, beta=1)`: Series expansion for Mittag-Leffler function
- `_asymptotic_expansion(z, alpha, beta=1)`: Asymptotic expansion for large z

**Usage Example**:
```python
from src.special.mittag_leffler import MittagLeffler
import numpy as np

# Create Mittag-Leffler calculator
ml = MittagLeffler(precision=15, max_terms=100)

# Compute E_alpha(z)
z = np.linspace(0, 5, 100)
alpha = 0.5
result = ml.e_alpha(z, alpha)

# Compute E_alpha,beta(z)
alpha = 0.5
beta = 1.5
result_ab = ml.e_alpha_beta(z, alpha, beta)
```

---

## Utility Classes

### ErrorAnalysis

**Location**: `src/utils/error_analysis.py`

**Description**: Utilities for error analysis in fractional calculus computations.

**Methods**:
- `relative_error(computed, exact)`: Compute relative error
- `absolute_error(computed, exact)`: Compute absolute error
- `convergence_rate(errors, grid_sizes)`: Compute convergence rate
- `stability_analysis(solver, test_problem)`: Analyze numerical stability

**Usage Example**:
```python
from src.utils.error_analysis import ErrorAnalysis
import numpy as np

# Create error analysis utility
error_analysis = ErrorAnalysis()

# Compute errors
computed = np.array([1.01, 1.02, 1.03])
exact = np.array([1.0, 1.0, 1.0])

rel_error = error_analysis.relative_error(computed, exact)
abs_error = error_analysis.absolute_error(computed, exact)

# Analyze convergence
grid_sizes = [10, 20, 40, 80]
errors = [0.1, 0.05, 0.025, 0.0125]
convergence_rate = error_analysis.convergence_rate(errors, grid_sizes)
```

### MemoryManagement

**Location**: `src/utils/memory_management.py`

**Description**: Utilities for memory management in large-scale computations.

**Methods**:
- `get_memory_usage()`: Get current memory usage
- `optimize_memory_usage(data)`: Optimize memory usage for large arrays
- `clear_cache()`: Clear memory caches
- `monitor_memory(func)`: Decorator to monitor memory usage of functions

**Usage Example**:
```python
from src.utils.memory_management import MemoryManagement
import numpy as np

# Create memory management utility
mem_mgmt = MemoryManagement()

# Monitor memory usage
current_memory = mem_mgmt.get_memory_usage()

# Optimize large array
large_array = np.random.random((10000, 10000))
optimized_array = mem_mgmt.optimize_memory_usage(large_array)

# Monitor function memory usage
@mem_mgmt.monitor_memory
def expensive_computation(data):
    return np.linalg.svd(data)
```

### Plotting

**Location**: `src/utils/plotting.py`

**Description**: Utilities for plotting fractional calculus results.

**Methods**:
- `plot_solution(t, x, u, title="Solution")`: Plot 2D solution
- `plot_convergence(grid_sizes, errors, title="Convergence")`: Plot convergence analysis
- `plot_comparison(x, exact, computed, title="Comparison")`: Plot exact vs computed solutions
- `plot_3d_surface(t, x, u, title="3D Surface")`: Plot 3D surface of solution

**Usage Example**:
```python
from src.utils.plotting import Plotting
import numpy as np

# Create plotting utility
plotting = Plotting()

# Plot solution
t = np.linspace(0, 1, 50)
x = np.linspace(0, 1, 100)
u = np.random.random((50, 100))

plotting.plot_solution(t, x, u, title="Fractional Diffusion Solution")

# Plot convergence
grid_sizes = [10, 20, 40, 80]
errors = [0.1, 0.05, 0.025, 0.0125]
plotting.plot_convergence(grid_sizes, errors, title="Convergence Analysis")
```

---

## Validation Classes

### AnalyticalSolutions

**Location**: `src/validation/analytical_solutions.py`

**Description**: Provides analytical solutions for validation of numerical methods.

**Methods**:
- `fractional_diffusion_solution(x, t, alpha, beta)`: Analytical solution for fractional diffusion
- `fractional_advection_solution(x, t, alpha, velocity)`: Analytical solution for fractional advection
- `fractional_ode_solution(t, alpha, initial_condition)`: Analytical solution for fractional ODE
- `compare_numerical_analytical(numerical, analytical, tolerance=1e-6)`: Compare numerical and analytical solutions

**Usage Example**:
```python
from src.validation.analytical_solutions import AnalyticalSolutions
import numpy as np

# Create analytical solutions utility
analytical = AnalyticalSolutions()

# Get analytical solution for fractional diffusion
x = np.linspace(0, 1, 100)
t = np.linspace(0, 0.1, 50)
alpha = 0.5
beta = 2.0

exact_solution = analytical.fractional_diffusion_solution(x, t, alpha, beta)

# Compare with numerical solution
numerical_solution = np.random.random((50, 100))  # Placeholder
comparison = analytical.compare_numerical_analytical(
    numerical_solution, exact_solution, tolerance=1e-6
)
```

### Benchmarks

**Location**: `src/validation/benchmarks.py`

**Description**: Benchmarking utilities for performance analysis.

**Methods**:
- `benchmark_solver(solver, test_problem, iterations=100)`: Benchmark solver performance
- `compare_solvers(solvers, test_problem)`: Compare performance of multiple solvers
- `memory_benchmark(func, *args, **kwargs)`: Benchmark memory usage
- `accuracy_benchmark(solver, exact_solution, test_problem)`: Benchmark accuracy

**Usage Example**:
```python
from src.validation.benchmarks import Benchmarks
from src.solvers.pde_solvers import FractionalDiffusionSolver

# Create benchmarking utility
benchmarks = Benchmarks()

# Create solver
solver = FractionalDiffusionSolver()

# Define test problem
def test_problem():
    # Define problem parameters
    return {
        'x_span': (0, 1),
        't_span': (0, 0.1),
        'alpha': 0.5,
        'beta': 2.0,
        'nx': 100,
        'nt': 50
    }

# Benchmark solver
performance_results = benchmarks.benchmark_solver(
    solver, test_problem, iterations=100
)
```

### ConvergenceTests

**Location**: `src/validation/convergence_tests.py`

**Description**: Utilities for testing convergence of numerical methods.

**Methods**:
- `test_spatial_convergence(solver, exact_solution, grid_sizes)`: Test spatial convergence
- `test_temporal_convergence(solver, exact_solution, time_steps)`: Test temporal convergence
- `test_order_of_accuracy(errors, grid_sizes)`: Determine order of accuracy
- `plot_convergence_results(grid_sizes, errors, title="Convergence")`: Plot convergence results

**Usage Example**:
```python
from src.validation.convergence_tests import ConvergenceTests
from src.solvers.pde_solvers import FractionalDiffusionSolver

# Create convergence tests utility
convergence_tests = ConvergenceTests()

# Create solver
solver = FractionalDiffusionSolver()

# Test spatial convergence
grid_sizes = [10, 20, 40, 80, 160]
exact_solution = lambda x, t: np.sin(np.pi * x) * np.exp(-t)

spatial_results = convergence_tests.test_spatial_convergence(
    solver, exact_solution, grid_sizes
)

# Plot results
convergence_tests.plot_convergence_results(
    grid_sizes, spatial_results['errors'], 
    title="Spatial Convergence"
)
```

---

## Summary

This documentation provides comprehensive coverage of all classes in the Fractional Calculus Library. Each class is designed with specific functionality and can be used independently or in combination with other classes to solve complex fractional calculus problems.

### Key Features:

1. **Modular Design**: Each class has a specific purpose and can be used independently
2. **Multiple Methods**: Most classes support multiple numerical methods and algorithms
3. **Adaptive Capabilities**: Many solvers include adaptive step size control
4. **Parallel Computing**: Built-in support for parallel processing
5. **Validation Tools**: Comprehensive validation and benchmarking utilities
6. **Performance Optimization**: JAX and NUMBA integration for high performance

### Best Practices:

1. **Choose Appropriate Methods**: Select numerical methods based on problem characteristics
2. **Validate Results**: Always validate numerical results against analytical solutions when available
3. **Monitor Performance**: Use benchmarking tools to ensure optimal performance
4. **Handle Edge Cases**: Be aware of numerical stability issues with certain parameter combinations
5. **Use Adaptive Methods**: Prefer adaptive methods for problems with varying solution characteristics

For more detailed information about specific implementations and mathematical foundations, refer to the main documentation file and the individual module docstrings.
