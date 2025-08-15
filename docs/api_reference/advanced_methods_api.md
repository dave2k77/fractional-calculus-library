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

---

## Optimized Advanced Methods

### Class: `OptimizedWeylDerivative`

**Description:** JAX-optimized Weyl derivative with GPU acceleration.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedWeylDerivative`

#### Constructor

```python
OptimizedWeylDerivative(alpha: float, use_gpu: bool = True, compile_mode: str = "jit")
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `use_gpu` (bool, optional): Enable GPU acceleration (default: True)
- `compile_mode` (str, optional): JAX compilation mode ("jit", "pmap", "vmap", default: "jit")

**Returns:** `OptimizedWeylDerivative` instance

#### Method: `compute`

```python
compute(f: Callable, x: np.ndarray, h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable): Function to differentiate
- `x` (np.ndarray): Evaluation points
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Optimized Weyl derivative values

### Class: `OptimizedMarchaudDerivative`

**Description:** Numba-optimized Marchaud derivative with memory-efficient streaming.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedMarchaudDerivative`

#### Constructor

```python
OptimizedMarchaudDerivative(alpha: float, parallel: bool = True, 
                           memory_efficient: bool = True)
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `parallel` (bool, optional): Enable parallel processing (default: True)
- `memory_efficient` (bool, optional): Enable memory optimization (default: True)

**Returns:** `OptimizedMarchaudDerivative` instance

#### Method: `compute`

```python
compute(f: Callable, x: np.ndarray, h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable): Function to differentiate
- `x` (np.ndarray): Evaluation points
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Optimized Marchaud derivative values

### Class: `OptimizedHadamardDerivative`

**Description:** JAX-optimized Hadamard derivative with vectorized computation.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedHadamardDerivative`

#### Constructor

```python
OptimizedHadamardDerivative(alpha: float, use_gpu: bool = True, compile_mode: str = "jit")
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `use_gpu` (bool, optional): Enable GPU acceleration (default: True)
- `compile_mode` (str, optional): JAX compilation mode (default: "jit")

**Returns:** `OptimizedHadamardDerivative` instance

#### Method: `compute`

```python
compute(f: Callable, x: np.ndarray, h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable): Function to differentiate
- `x` (np.ndarray): Evaluation points (must be positive)
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Optimized Hadamard derivative values

### Class: `OptimizedReizFellerDerivative`

**Description:** JAX-optimized Reiz-Feller derivative using spectral method.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedReizFellerDerivative`

#### Constructor

```python
OptimizedReizFellerDerivative(alpha: float, use_gpu: bool = True, compile_mode: str = "jit")
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `use_gpu` (bool, optional): Enable GPU acceleration (default: True)
- `compile_mode` (str, optional): JAX compilation mode (default: "jit")

**Returns:** `OptimizedReizFellerDerivative` instance

#### Method: `compute`

```python
compute(f: Callable, x: np.ndarray, h: float) -> np.ndarray
```

**Parameters:**
- `f` (Callable): Function to differentiate
- `x` (np.ndarray): Evaluation points
- `h` (float): Step size for discretization

**Returns:** `np.ndarray` - Optimized Reiz-Feller derivative values

### Class: `OptimizedAdomianDecomposition`

**Description:** JAX-optimized Adomian decomposition for parallel computation.

**Location:** `src.algorithms.advanced_optimized_methods.OptimizedAdomianDecomposition`

#### Constructor

```python
OptimizedAdomianDecomposition(alpha: float, use_gpu: bool = True, compile_mode: str = "jit")
```

**Parameters:**
- `alpha` (float): Fractional order (0 < α < 2)
- `use_gpu` (bool, optional): Enable GPU acceleration (default: True)
- `compile_mode` (str, optional): JAX compilation mode (default: "jit")

**Returns:** `OptimizedAdomianDecomposition` instance

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

**Returns:** `np.ndarray` - Optimized solution values

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
                         use_gpu: bool = True) -> np.ndarray
```

#### Function: `optimized_marchaud_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_marchaud_derivative`

```python
optimized_marchaud_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             parallel: bool = True) -> np.ndarray
```

#### Function: `optimized_hadamard_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_hadamard_derivative`

```python
optimized_hadamard_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                             use_gpu: bool = True) -> np.ndarray
```

#### Function: `optimized_reiz_feller_derivative`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_reiz_feller_derivative`

```python
optimized_reiz_feller_derivative(f: Callable, x: np.ndarray, alpha: float, h: float, 
                                use_gpu: bool = True) -> np.ndarray
```

#### Function: `optimized_adomian_solve`

**Location:** `src.algorithms.advanced_optimized_methods.optimized_adomian_solve`

```python
optimized_adomian_solve(f: Callable, t: np.ndarray, alpha: float, 
                       initial_condition: float = 0.0, terms: int = 10, 
                       use_gpu: bool = True) -> np.ndarray
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
weyl_opt = optimized_weyl_derivative(f, x, alpha, h, use_gpu=True)
marchaud_opt = optimized_marchaud_derivative(f, x, alpha, h, parallel=True)
hadamard_opt = optimized_hadamard_derivative(f, x, alpha, h, use_gpu=True)
reiz_opt = optimized_reiz_feller_derivative(f, x, alpha, h, use_gpu=True)

print("Optimized computations completed successfully!")
```

### FDE Solving

```python
import numpy as np
from src.algorithms.advanced_methods import AdomianDecomposition
from src.algorithms.advanced_optimized_methods import optimized_adomian_solve

# Define FDE: D^α y(t) = -y(t) + t
def fractional_ode(t, y, alpha):
    return -y + t

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
