# Advanced Methods API Reference

This document provides the complete API reference for the advanced fractional calculus methods in the library.

## Table of Contents

1. [Weyl Derivative](#weyl-derivative)
2. [Marchaud Derivative](#marchaud-derivative)
3. [Hadamard Derivative](#hadamard-derivative)
4. [Reiz-Feller Derivative](#reiz-feller-derivative)
5. [Adomian Decomposition](#adomian-decomposition)
6. [Optimized Advanced Methods](#optimized-advanced-methods)
7. [Convenience Functions](#convenience-functions)
8. [Consolidated Structure](#consolidated-structure)

---

## Consolidated Structure

The library has been consolidated to provide a cleaner, more efficient structure:

### Primary Import Structure

```python
# Core optimized methods (PRIMARY implementations)
from hpfracc.core.derivatives import (
    create_fractional_derivative,
    CaputoDerivative,
    RiemannLiouvilleDerivative,
    GrunwaldLetnikovDerivative
)

# Special functions and utilities
from hpfracc.special import (
    gamma, beta, mittag_leffler,
    FractionalDiffusionGreenFunction,
    FractionalWaveGreenFunction,
    FractionalAdvectionGreenFunction
)

# Analytical solvers
from hpfracc.solvers import (
    HomotopyPerturbationSolver,
    VariationalIterationSolver
)

# Mathematical utilities
from hpfracc.core.utilities import (
    validate_fractional_order,
    timing_decorator,
    memory_decorator
)

# Analytics and performance
from hpfracc.analytics import (
    analyze_convergence,
    estimate_error,
    benchmark_performance
)
```

### Migration Guide

**Old imports (deprecated):**
```python
# ❌ These imports are no longer available
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative
from src.algorithms.fft_methods import FFTFractionalMethods
from src.optimisation.jax_implementations import JAXFractionalDerivatives
from src.optimisation.numba_kernels import NumbaFractionalKernels
from src.optimisation.parallel_computing import ParallelFractionalComputing
```

**New imports (recommended):**
```python
# ✅ Use these optimized implementations instead
from hpfracc.core.derivatives import (
    create_fractional_derivative,  # Factory function for all derivative types
    CaputoDerivative,  # Direct class access
    RiemannLiouvilleDerivative,  # Direct class access
    GrunwaldLetnikovDerivative   # Direct class access
)

# Special functions
from hpfracc.special import gamma, beta, mittag_leffler

# Green's functions
from hpfracc.special.greens_function import (
    FractionalDiffusionGreenFunction,
    FractionalWaveGreenFunction,
    FractionalAdvectionGreenFunction
)
```

---

## Weyl Derivative

### Class: `WeylDerivative`

**Description:** Computes the Weyl fractional derivative using FFT convolution with parallelization.

**Location:** `hpfracc.core.derivatives.WeylDerivative`

#### Constructor

```python
WeylDerivative(alpha: float, n_jobs: int = -1, chunk_size: Optional[int] = None)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)
- `chunk_size` (int, optional): Size of chunks for parallel processing (default: None)

**Returns:** `WeylDerivative` instance

#### Method: `compute`

```python
compute(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Weyl fractional derivative values

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize using factory function
alpha = 0.5
weyl = create_fractional_derivative(alpha, method="Weyl")

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
result = weyl(f, x)

print(f"Weyl derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedWeylDerivative`

**Description:** Optimized version of Weyl derivative with enhanced performance.

**Location:** `hpfracc.core.derivatives.OptimizedWeylDerivative`

#### Constructor

```python
OptimizedWeylDerivative(alpha: float, method: str = "fft", n_jobs: int = -1)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Computation method ("fft", "direct", default: "fft")
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize optimized version
alpha = 0.5
weyl_opt = create_fractional_derivative(alpha, method="Weyl", optimized=True)

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
result = weyl_opt(f, x)

print(f"Optimized Weyl derivative shape: {result.shape}")
```

### Convenience Function: `optimized_weyl_derivative`

**Description:** Convenience function for computing optimized Weyl derivatives.

**Location:** `hpfracc.core.derivatives.optimized_weyl_derivative`

```python
optimized_weyl_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                         method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Compute using factory function
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
alpha = 0.5

weyl_deriv = create_fractional_derivative(alpha, method="Weyl")
result = weyl_deriv(f, x)

print(f"Weyl derivative result: {result.shape}")
```

---

## Marchaud Derivative

### Class: `MarchaudDerivative`

**Description:** Computes the Marchaud fractional derivative with difference quotient convolution and memory optimization.

**Location:** `hpfracc.core.derivatives.MarchaudDerivative`

#### Constructor

```python
MarchaudDerivative(alpha: float, n_jobs: int = -1, chunk_size: Optional[int] = None, 
                   memory_optimized: bool = True)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)
- `chunk_size` (int, optional): Size of chunks for parallel processing (default: None)
- `memory_optimized` (bool, optional): Enable memory optimization (default: True)

**Returns:** `MarchaudDerivative` instance

#### Method: `compute`

```python
compute(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Marchaud fractional derivative values

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize using factory function
alpha = 0.5
marchaud = create_fractional_derivative(alpha, method="Marchaud")

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
result = marchaud(f, x)

print(f"Marchaud derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedMarchaudDerivative`

**Description:** Optimized version of Marchaud derivative with enhanced performance.

**Location:** `hpfracc.core.derivatives.OptimizedMarchaudDerivative`

#### Constructor

```python
OptimizedMarchaudDerivative(alpha: float, method: str = "fft", n_jobs: int = -1)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Computation method ("fft", "direct", default: "fft")
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize optimized version
alpha = 0.5
marchaud_opt = create_fractional_derivative(alpha, method="Marchaud", optimized=True)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
result = marchaud_opt(f, x)

print(f"Optimized Marchaud derivative shape: {result.shape}")
```

### Convenience Function: `optimized_marchaud_derivative`

**Description:** Convenience function for computing optimized Marchaud derivatives.

**Location:** `hpfracc.core.derivatives.optimized_marchaud_derivative`

```python
optimized_marchaud_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Compute using factory function
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
alpha = 0.5

marchaud_deriv = create_fractional_derivative(alpha, method="Marchaud")
result = marchaud_deriv(f, x)

print(f"Marchaud derivative result: {result.shape}")
```

---

## Hadamard Derivative

### Class: `HadamardDerivative`

**Description:** Computes the Hadamard fractional derivative using logarithmic transformation.

**Location:** `hpfracc.core.derivatives.HadamardDerivative`

#### Constructor

```python
HadamardDerivative(alpha: float, n_jobs: int = -1, chunk_size: Optional[int] = None)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)
- `chunk_size` (int, optional): Size of chunks for parallel processing (default: None)

**Returns:** `HadamardDerivative` instance

#### Method: `compute`

```python
compute(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s) (must be positive)
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Hadamard fractional derivative values

**Note:** The domain `x` must be positive for Hadamard derivatives.

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize using factory function
alpha = 0.5
hadamard = create_fractional_derivative(alpha, method="Hadamard")

# Compute derivative (positive domain required)
x = np.linspace(1, 10, 100)  # Must be positive
f = lambda x: np.log(x) * np.sin(x)
result = hadamard(f, x)

print(f"Hadamard derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedHadamardDerivative`

**Description:** Optimized version of Hadamard derivative with enhanced performance.

**Location:** `hpfracc.core.derivatives.OptimizedHadamardDerivative`

#### Constructor

```python
OptimizedHadamardDerivative(alpha: float, method: str = "fft", n_jobs: int = -1)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Computation method ("fft", "direct", default: "fft")
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize optimized version
alpha = 0.5
hadamard_opt = create_fractional_derivative(alpha, method="Hadamard", optimized=True)

# Compute derivative
x = np.linspace(1, 10, 100)
f = lambda x: np.log(x) * np.sin(x)
result = hadamard_opt(f, x)

print(f"Optimized Hadamard derivative shape: {result.shape}")
```

### Convenience Function: `optimized_hadamard_derivative`

**Description:** Convenience function for computing optimized Hadamard derivatives.

**Location:** `hpfracc.core.derivatives.optimized_hadamard_derivative`

```python
optimized_hadamard_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Compute using factory function
x = np.linspace(1, 10, 100)
f = lambda x: np.log(x) * np.sin(x)
alpha = 0.5

hadamard_deriv = create_fractional_derivative(alpha, method="Hadamard")
result = hadamard_deriv(f, x)

print(f"Hadamard derivative result: {result.shape}")
```

---

## Reiz-Feller Derivative

### Class: `ReizFellerDerivative`

**Description:** Computes the Reiz-Feller fractional derivative using spectral method with FFT.

**Location:** `hpfracc.core.derivatives.ReizFellerDerivative`

#### Constructor

```python
ReizFellerDerivative(alpha: float, n_jobs: int = -1, chunk_size: Optional[int] = None)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)
- `chunk_size` (int, optional): Size of chunks for parallel processing (default: None)

**Returns:** `ReizFellerDerivative` instance

#### Method: `compute`

```python
compute(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Reiz-Feller fractional derivative values

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize using factory function
alpha = 0.5
reiz_feller = create_fractional_derivative(alpha, method="ReizFeller")

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
result = reiz_feller(f, x)

print(f"Reiz-Feller derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedReizFellerDerivative`

**Description:** Optimized version of Reiz-Feller derivative with enhanced performance.

**Location:** `hpfracc.core.derivatives.OptimizedReizFellerDerivative`

#### Constructor

```python
OptimizedReizFellerDerivative(alpha: float, method: str = "fft", n_jobs: int = -1)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Computation method ("fft", "direct", default: "fft")
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Initialize optimized version
alpha = 0.5
reiz_feller_opt = create_fractional_derivative(alpha, method="ReizFeller", optimized=True)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
result = reiz_feller_opt(f, x)

print(f"Optimized Reiz-Feller derivative shape: {result.shape}")
```

### Convenience Function: `optimized_reiz_feller_derivative`

**Description:** Convenience function for computing optimized Reiz-Feller derivatives.

**Location:** `hpfracc.core.derivatives.optimized_reiz_feller_derivative`

```python
optimized_reiz_feller_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                                method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from hpfracc.core.derivatives import create_fractional_derivative
import numpy as np

# Compute using factory function
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
alpha = 0.5

reiz_feller_deriv = create_fractional_derivative(alpha, method="ReizFeller")
result = reiz_feller_deriv(f, x)

print(f"Reiz-Feller derivative result: {result.shape}")
```

---

## Adomian Decomposition

### Class: `AdomianDecomposition`

**Description:** Solves fractional differential equations using the Adomian decomposition method.

**Location:** `hpfracc.solvers.AdomianDecomposition`

#### Constructor

```python
AdomianDecomposition(alpha: float, n_jobs: int = -1, chunk_size: Optional[int] = None)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)
- `chunk_size` (int, optional): Size of chunks for parallel processing (default: None)

**Returns:** `AdomianDecomposition` instance

#### Method: `solve`

```python
solve(f: Callable, t: np.ndarray, initial_condition: float = 0.0, 
      terms: int = 10, tolerance: float = 1e-6) -> np.ndarray
```

**Parameters:**
- `f` (Callable): Right-hand side function f(t, y, α)
- `t` (np.ndarray): Time points for solution
- `initial_condition` (float, optional): Initial condition y(0) (default: 0.0)
- `terms` (int, optional): Number of decomposition terms (default: 10)
- `tolerance` (float, optional): Convergence tolerance (default: 1e-6)

**Returns:** `np.ndarray` - Solution values at time points

**Example:**
```python
from hpfracc.solvers import AdomianDecomposition
import numpy as np

# Initialize
alpha = 0.5
adomian = AdomianDecomposition(alpha)

# Define FDE: D^α y(t) = -y(t)
def fractional_ode(t, y, alpha):
    return -y

# Solve
t = np.linspace(0, 5, 100)
solution = adomian.solve(fractional_ode, t, initial_condition=1.0, terms=15)

print(f"Solution shape: {solution.shape}")
```

### Optimized Class: `OptimizedAdomianDecomposition`

**Description:** Optimized version of Adomian decomposition for parallel computation.

**Location:** `hpfracc.solvers.OptimizedAdomianDecomposition`

#### Constructor

```python
OptimizedAdomianDecomposition(alpha: float, method: str = "jit", n_jobs: int = -1)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Computation method ("jit", "pmap", "vmap", default: "jit")
- `n_jobs` (int, optional): Number of parallel jobs (-1 for all cores, default: -1)

**Example:**
```python
from hpfracc.solvers import OptimizedAdomianDecomposition
import numpy as np

# Initialize optimized version
alpha = 0.5
adomian_opt = OptimizedAdomianDecomposition(alpha, method="jit", n_jobs=4)

# Define FDE: D^α y(t) = -y(t) + t
def fractional_ode(t, y, alpha):
    return -y + t

# Solve
t = np.linspace(0, 5, 100)
solution = adomian_opt.solve(fractional_ode, t, initial_condition=1.0, terms=15)

print(f"Optimized solution shape: {solution.shape}")
```

### Convenience Function: `optimized_adomian_solve`

**Description:** Convenience function for computing optimized Adomian decompositions.

**Location:** `hpfracc.solvers.optimized_adomian_solve`

```python
optimized_adomian_solve(f: Callable, t: np.ndarray, alpha: float, 
                       initial_condition: float = 0.0, terms: int = 10, 
                       method: str = "jit", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from hpfracc.solvers import optimized_adomian_solve
import numpy as np

# Define FDE: D^α y(t) = -y(t) + t
def fractional_ode(t, y, alpha):
    return -y + t

# Solve using optimized method
t = np.linspace(0, 5, 100)
alpha = 0.5
solution = optimized_adomian_solve(fractional_ode, t, alpha, initial_condition=1.0, terms=15)

print(f"Optimized Adomian solution shape: {solution.shape}")
```

---

## Convenience Functions

### Function: `weyl_derivative`

**Description:** Convenience function for computing Weyl derivative.

**Location:** `hpfracc.core.derivatives.weyl_derivative`

```python
weyl_derivative(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], 
                alpha: float, h: float, n_jobs: int = -1) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `alpha` (float): Fractional order (0 < α < 2)
- `h` (float): Step size for discretization
- `n_jobs` (int, optional): Number of parallel jobs (default: -1)

**Returns:** `np.ndarray` - Weyl derivative values

### Function: `marchaud_derivative`

**Description:** Convenience function for computing Marchaud derivative.

**Location:** `hpfracc.core.derivatives.marchaud_derivative`

```python
marchaud_derivative(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], 
                    alpha: float, h: float, n_jobs: int = -1, 
                    memory_optimized: bool = True) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `alpha` (float): Fractional order (0 < α < 2)
- `h` (float): Step size for discretization
- `n_jobs` (int, optional): Number of parallel jobs (default: -1)
- `memory_optimized` (bool, optional): Enable memory optimization (default: True)

**Returns:** `np.ndarray` - Marchaud derivative values

### Function: `hadamard_derivative`

**Description:** Convenience function for computing Hadamard derivative.

**Location:** `hpfracc.core.derivatives.hadamard_derivative`

```python
hadamard_derivative(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], 
                    alpha: float, h: float, n_jobs: int = -1) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s) (must be positive)
- `alpha` (float): Fractional order (0 < α < 2)
- `h` (float): Step size for discretization
- `n_jobs` (int, optional): Number of parallel jobs (default: -1)

**Returns:** `np.ndarray` - Hadamard derivative values

### Function: `reiz_feller_derivative`

**Description:** Convenience function for computing Reiz-Feller derivative.

**Location:** `hpfracc.core.derivatives.reiz_feller_derivative`

```python
reiz_feller_derivative(f: Union[Callable, np.ndarray], x: Union[float, np.ndarray], 
                       alpha: float, h: float, n_jobs: int = -1) -> np.ndarray
```

**Parameters:**
- `f` (Callable or np.ndarray): Function to differentiate or function values
- `x` (float or np.ndarray): Evaluation point(s)
- `alpha` (float): Fractional order (0 < α < 2)
- `h` (float): Step size for discretization
- `n_jobs` (int, optional): Number of parallel jobs (default: -1)

**Returns:** `np.ndarray` - Reiz-Feller derivative values

### Optimized Convenience Functions

#### Function: `optimized_weyl_derivative`

**Location:** `hpfracc.core.derivatives.optimized_weyl_derivative`

```python
optimized_weyl_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                         method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_marchaud_derivative`

**Location:** `hpfracc.core.derivatives.optimized_marchaud_derivative`

```python
optimized_marchaud_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_hadamard_derivative`

**Location:** `hpfracc.core.derivatives.optimized_hadamard_derivative`

```python
optimized_hadamard_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_reiz_feller_derivative`

**Location:** `hpfracc.core.derivatives.optimized_reiz_feller_derivative`

```python
optimized_reiz_feller_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                                method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_adomian_solve`

**Location:** `hpfracc.solvers.optimized_adomian_solve`

```python
optimized_adomian_solve(f: Callable, t: np.ndarray, alpha: float, 
                       initial_condition: float = 0.0, terms: int = 10, 
                       method: str = "jit", n_jobs: int = -1) -> np.ndarray
```

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from hpfracc.core.derivatives import create_fractional_derivative

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 100)
f = lambda x: np.sin(x) * np.exp(-x/3)

# Compute derivatives using factory function
weyl_result = create_fractional_derivative(alpha, method="Weyl")(f, x)
marchaud_result = create_fractional_derivative(alpha, method="Marchaud")(f, x)
hadamard_result = create_fractional_derivative(alpha, method="Hadamard")(f, np.linspace(1, 5, 100))
reiz_result = create_fractional_derivative(alpha, method="ReizFeller")(f, x)

print(f"Results shapes: {weyl_result.shape}, {marchaud_result.shape}, {hadamard_result.shape}, {reiz_result.shape}")
```

### Optimized Usage

```python
import numpy as np
from hpfracc.core.derivatives import create_fractional_derivative

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 1000)
f = lambda x: np.sin(x) * np.exp(-x/3)

# Compute optimized derivatives
weyl_opt = create_fractional_derivative(alpha, method="Weyl", optimized=True)(f, x)
marchaud_opt = create_fractional_derivative(alpha, method="Marchaud", optimized=True)(f, x)
hadamard_opt = create_fractional_derivative(alpha, method="Hadamard", optimized=True)(f, x)
reiz_opt = create_fractional_derivative(alpha, method="ReizFeller", optimized=True)(f, x)

print("Optimized computations completed successfully!")
```

### FDE Solving

```python
import numpy as np
from hpfracc.solvers import AdomianDecomposition, optimized_adomian_solve

# Define FDE: D^α y(t) = -y(t)
def fractional_ode(t, y, alpha):
    return -y

# Solve using standard method
alpha = 0.5
t = np.linspace(0, 3, 100)
adomian = AdomianDecomposition(alpha)
solution_std = adomian.solve(fractional_ode, t, initial_condition=1.0, terms=15)

# Solve using optimized method
solution_opt = optimized_adomian_solve(fractional_ode, t, alpha, initial_condition=1.0, terms=15)

print(f"Solutions computed: {solution_std.shape}, {solution_opt.shape}")
```

---

## Error Handling

All methods include comprehensive error handling:

- **Invalid fractional order**: Raises `ValueError` for α ≤ 0 or α ≥ 2
- **Invalid step size**: Raises `ValueError` for h ≤ 0
- **Domain errors**: Raises `ValueError` for negative x in Hadamard derivative
- **Memory errors**: Graceful handling of out-of-memory situations
- **GPU errors**: Automatic fallback to CPU if GPU unavailable

## Performance Notes

- **Standard methods**: Suitable for small to medium datasets (< 10,000 points)
- **Optimized methods**: Recommended for large datasets (> 1,000 points)
- **GPU acceleration**: Most effective for datasets > 10,000 points
- **Parallel processing**: Optimal with 4-16 cores depending on method
- **Memory optimization**: Critical for datasets > 50,000 points

## Dependencies

- **Standard methods**: NumPy, SciPy
- **Optimized methods**: JAX, Numba (optional but recommended)
- **GPU acceleration**: JAX with CUDA support
- **Parallel processing**: concurrent.futures, multiprocessing

For more information, see the [Advanced Methods Guide](../advanced_methods_guide.md) and [Examples](../examples/).
