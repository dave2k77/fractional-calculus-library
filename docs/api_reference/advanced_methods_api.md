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
from src.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    OptimizedGrunwaldLetnikov,
    AdvancedFFTMethods,
    L1L2Schemes,
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov
)

# GPU-optimized methods
from src.algorithms.gpu_optimized_methods import (
    GPUOptimizedCaputo,
    GPUOptimizedRiemannLiouville,
    GPUOptimizedGrunwaldLetnikov,
    JAXAutomaticDifferentiation,
    JAXOptimizer,
    gpu_optimized_caputo,
    gpu_optimized_riemann_liouville,
    gpu_optimized_grunwald_letnikov
)

# Parallel-optimized methods
from src.algorithms.parallel_optimized_methods import (
    ParallelOptimizedCaputo,
    ParallelOptimizedRiemannLiouville,
    ParallelOptimizedGrunwaldLetnikov,
    NumbaOptimizer,
    NumbaFractionalKernels,
    NumbaParallelManager,
    parallel_optimized_caputo,
    parallel_optimized_caputo,
    parallel_optimized_grunwald_letnikov
)

# Advanced methods
from src.algorithms.advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    AdomianDecomposition
)

# Optimized advanced methods
from src.algorithms.advanced_optimized_methods import (
    OptimizedWeylDerivative,
    OptimizedMarchaudDerivative,
    OptimizedHadamardDerivative,
    OptimizedReizFellerDerivative,
    OptimizedAdomianDecomposition,
    optimized_weyl_derivative,
    optimized_marchaud_derivative,
    optimized_hadamard_derivative,
    optimized_reiz_feller_derivative,
    optimized_adomian_decomposition
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
from src.algorithms.optimized_methods import (
    OptimizedCaputo,  # Instead of CaputoDerivative
    OptimizedRiemannLiouville,  # Instead of RiemannLiouvilleDerivative
    OptimizedGrunwaldLetnikov,  # Instead of GrunwaldLetnikovDerivative
    AdvancedFFTMethods,  # Instead of FFTFractionalMethods
    optimized_caputo,  # Function interface
    optimized_riemann_liouville,  # Function interface
    optimized_grunwald_letnikov   # Function interface
)
```

---

## Weyl Derivative

### Class: `WeylDerivative`

**Description:** Computes the Weyl fractional derivative using FFT convolution with parallelization.

**Location:** `src.algorithms.advanced_methods.WeylDerivative`

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
from src.algorithms.advanced_methods import WeylDerivative
import numpy as np

# Initialize
alpha = 0.5
weyl = WeylDerivative(alpha, n_jobs=4)

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
result = weyl.compute(f, x, h=0.1)

print(f"Weyl derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedWeylDerivative`

**Description:** Optimized version of Weyl derivative with enhanced performance.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedWeylDerivative`

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
from src.algorithms.advanced_optimized_methods import OptimizedWeylDerivative
import numpy as np

# Initialize optimized version
alpha = 0.5
weyl_opt = OptimizedWeylDerivative(alpha, method="fft", n_jobs=4)

# Compute derivative
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
result = weyl_opt.compute(f, x, h=0.1)

print(f"Optimized Weyl derivative shape: {result.shape}")
```

### Convenience Function: `optimized_weyl_derivative`

**Description:** Convenience function for computing optimized Weyl derivatives.

**Location:** `src.algorithms.advanced_optimized_methods.optimized_weyl_derivative`

```python
optimized_weyl_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                         method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from src.algorithms.advanced_optimized_methods import optimized_weyl_derivative
import numpy as np

# Compute using convenience function
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)
alpha = 0.5
h = 0.1

result = optimized_weyl_derivative(f, x, alpha, h, method="fft", n_jobs=4)
print(f"Weyl derivative result: {result.shape}")
```

---

## Marchaud Derivative

### Class: `MarchaudDerivative`

**Description:** Computes the Marchaud fractional derivative with difference quotient convolution and memory optimization.

**Location:** `src.algorithms.advanced_methods.MarchaudDerivative`

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
from src.algorithms.advanced_methods import MarchaudDerivative
import numpy as np

# Initialize with memory optimization
alpha = 0.5
marchaud = MarchaudDerivative(alpha, memory_optimized=True, n_jobs=4)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
result = marchaud.compute(f, x, h=0.05)

print(f"Marchaud derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedMarchaudDerivative`

**Description:** Optimized version of Marchaud derivative with enhanced performance.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedMarchaudDerivative`

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
from src.algorithms.advanced_optimized_methods import OptimizedMarchaudDerivative
import numpy as np

# Initialize optimized version
alpha = 0.5
marchaud_opt = OptimizedMarchaudDerivative(alpha, method="fft", n_jobs=4)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
result = marchaud_opt.compute(f, x, h=0.05)

print(f"Optimized Marchaud derivative shape: {result.shape}")
```

### Convenience Function: `optimized_marchaud_derivative`

**Description:** Convenience function for computing optimized Marchaud derivatives.

**Location:** `src.algorithms.advanced_optimized_methods.optimized_marchaud_derivative`

```python
optimized_marchaud_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from src.algorithms.advanced_optimized_methods import optimized_marchaud_derivative
import numpy as np

# Compute using convenience function
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x) * np.sin(x)
alpha = 0.5
h = 0.05

result = optimized_marchaud_derivative(f, x, alpha, h, method="fft", n_jobs=4)
print(f"Marchaud derivative result: {result.shape}")
```

---

## Hadamard Derivative

### Class: `HadamardDerivative`

**Description:** Computes the Hadamard fractional derivative using logarithmic transformation.

**Location:** `src.algorithms.advanced_methods.HadamardDerivative`

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
from src.algorithms.advanced_methods import HadamardDerivative
import numpy as np

# Initialize
alpha = 0.5
hadamard = HadamardDerivative(alpha)

# Compute derivative (positive domain required)
x = np.linspace(1, 10, 100)  # Must be positive
f = lambda x: np.log(x) * np.sin(x)
result = hadamard.compute(f, x, h=0.09)

print(f"Hadamard derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedHadamardDerivative`

**Description:** Optimized version of Hadamard derivative with enhanced performance.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedHadamardDerivative`

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
from src.algorithms.advanced_optimized_methods import OptimizedHadamardDerivative
import numpy as np

# Initialize optimized version
alpha = 0.5
hadamard_opt = OptimizedHadamardDerivative(alpha, method="fft", n_jobs=4)

# Compute derivative
x = np.linspace(1, 10, 100)
f = lambda x: np.log(x) * np.sin(x)
result = hadamard_opt.compute(f, x, h=0.09)

print(f"Optimized Hadamard derivative shape: {result.shape}")
```

### Convenience Function: `optimized_hadamard_derivative`

**Description:** Convenience function for computing optimized Hadamard derivatives.

**Location:** `src.algorithms.advanced_optimized_methods.optimized_hadamard_derivative`

```python
optimized_hadamard_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from src.algorithms.advanced_optimized_methods import optimized_hadamard_derivative
import numpy as np

# Compute using convenience function
x = np.linspace(1, 10, 100)
f = lambda x: np.log(x) * np.sin(x)
alpha = 0.5
h = 0.09

result = optimized_hadamard_derivative(f, x, alpha, h, method="fft", n_jobs=4)
print(f"Hadamard derivative result: {result.shape}")
```

---

## Reiz-Feller Derivative

### Class: `ReizFellerDerivative`

**Description:** Computes the Reiz-Feller fractional derivative using spectral method with FFT.

**Location:** `src.algorithms.advanced_methods.ReizFellerDerivative`

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
from src.algorithms.advanced_methods import ReizFellerDerivative
import numpy as np

# Initialize
alpha = 0.5
reiz_feller = ReizFellerDerivative(alpha)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
result = reiz_feller.compute(f, x, h=0.05)

print(f"Reiz-Feller derivative shape: {result.shape}")
```

### Optimized Class: `OptimizedReizFellerDerivative`

**Description:** Optimized version of Reiz-Feller derivative with enhanced performance.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedReizFellerDerivative`

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
from src.algorithms.advanced_optimized_methods import OptimizedReizFellerDerivative
import numpy as np

# Initialize optimized version
alpha = 0.5
reiz_feller_opt = OptimizedReizFellerDerivative(alpha, method="fft", n_jobs=4)

# Compute derivative
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
result = reiz_feller_opt.compute(f, x, h=0.05)

print(f"Optimized Reiz-Feller derivative shape: {result.shape}")
```

### Convenience Function: `optimized_reiz_feller_derivative`

**Description:** Convenience function for computing optimized Reiz-Feller derivatives.

**Location:** `src.algorithms.advanced_optimized_methods.optimized_reiz_feller_derivative`

```python
optimized_reiz_feller_derivative(f: Union[Callable, np.ndarray], x: np.ndarray, alpha: float, h: float, 
                                method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from src.algorithms.advanced_optimized_methods import optimized_reiz_feller_derivative
import numpy as np

# Compute using convenience function
x = np.linspace(0, 5, 100)
f = lambda x: np.exp(-x**2/2)  # Gaussian function
alpha = 0.5
h = 0.05

result = optimized_reiz_feller_derivative(f, x, alpha, h, method="fft", n_jobs=4)
print(f"Reiz-Feller derivative result: {result.shape}")
```

---

## Adomian Decomposition

### Class: `AdomianDecomposition`

**Description:** Solves fractional differential equations using the Adomian decomposition method.

**Location:** `src.algorithms.advanced_methods.AdomianDecomposition`

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
from src.algorithms.advanced_methods import AdomianDecomposition
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

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedAdomianDecomposition`

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
from src.algorithms.advanced_optimized_methods import OptimizedAdomianDecomposition
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

**Location:** `src.algorithms.advanced_optimized_methods.optimized_adomian_solve`

```python
optimized_adomian_solve(f: Callable, t: np.ndarray, alpha: float, 
                       initial_condition: float = 0.0, terms: int = 10, 
                       method: str = "jit", n_jobs: int = -1) -> np.ndarray
```

**Example:**
```python
from src.algorithms.advanced_optimized_methods import optimized_adomian_solve
import numpy as np

# Initialize optimized version
alpha = 0.5
adomian_opt = OptimizedAdomianDecomposition(alpha, method="jit", n_jobs=4)

# Define FDE: D^α y(t) = -y(t) + t
def fractional_ode(t, y, alpha):
    return -y + t

# Solve
t = np.linspace(0, 5, 100)
solution = optimized_adomian_solve(fractional_ode, t, alpha, initial_condition=1.0, terms=15)

print(f"Optimized Adomian solution shape: {solution.shape}")
```

---

## Convenience Functions

### Function: `weyl_derivative`

**Description:** Convenience function for computing Weyl derivative.

**Location:** `src.algorithms.advanced_methods.weyl_derivative`

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

**Location:** `src.algorithms.advanced_methods.marchaud_derivative`

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

**Location:** `src.algorithms.advanced_methods.hadamard_derivative`

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

**Location:** `src.algorithms.advanced_methods.reiz_feller_derivative`

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

**Location:** `src.algorithms.advanced_optimized_methods.optimized_weyl_derivative`

```python
optimized_weyl_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                         method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_marchaud_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_marchaud_derivative`

```python
optimized_marchaud_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_hadamard_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_hadamard_derivative`

```python
optimized_hadamard_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_reiz_feller_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_reiz_feller_derivative`

```python
optimized_reiz_feller_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                                method: str = "fft", n_jobs: int = -1) -> np.ndarray
```

#### Function: `optimized_adomian_solve`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_adomian_solve`

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
from src.algorithms.advanced_methods import (
    weyl_derivative, marchaud_derivative, hadamard_derivative, reiz_feller_derivative
)

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 100)
f = lambda x: np.sin(x) * np.exp(-x/3)
h = 0.05

# Compute derivatives
weyl_result = weyl_derivative(f, x, alpha, h)
marchaud_result = marchaud_derivative(f, x, alpha, h)
hadamard_result = hadamard_derivative(f, np.linspace(1, 5, 100), alpha, h)
reiz_result = reiz_feller_derivative(f, x, alpha, h)

print(f"Results shapes: {weyl_result.shape}, {marchaud_result.shape}, {hadamard_result.shape}, {reiz_result.shape}")
```

### Optimized Usage

```python
import numpy as np
from src.algorithms.advanced_optimized_methods import (
    optimized_weyl_derivative, optimized_marchaud_derivative,
    optimized_hadamard_derivative, optimized_reiz_feller_derivative
)

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 1000)
f = lambda x: np.sin(x) * np.exp(-x/3)
h = 0.005

# Compute optimized derivatives
weyl_opt = optimized_weyl_derivative(f, x, alpha, h, method="fft", n_jobs=4)
marchaud_opt = optimized_marchaud_derivative(f, x, alpha, h, method="fft", n_jobs=4)
hadamard_opt = optimized_hadamard_derivative(f, x, alpha, h, method="fft", n_jobs=4)
reiz_opt = optimized_reiz_feller_derivative(f, x, alpha, h, method="fft", n_jobs=4)

print("Optimized computations completed successfully!")
```

### FDE Solving

```python
import numpy as np
from src.algorithms.advanced_methods import AdomianDecomposition
from src.algorithms.advanced_optimized_methods import optimized_adomian_solve

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
