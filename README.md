# Fractional Calculus Library

A high-performance Python library for numerical methods in fractional calculus, leveraging JAX and NUMBA for optimized computations and parallel processing.

## 🚀 Features

- **Multiple Fractional Derivative Definitions**: Caputo, Riemann-Liouville, Grünwald-Letnikov
- **🚀 Advanced Methods**: Weyl, Marchaud, Hadamard, Reiz-Feller derivatives, Adomian Decomposition
- **🚀 Optimized Methods**: Dramatic performance improvements (up to 196x speedup)
- **High-Performance Computing**: JAX for automatic differentiation and GPU acceleration
- **JIT Compilation**: NUMBA for optimized numerical kernels
- **Parallel Computing**: Multi-core and GPU support
- **Comprehensive Testing**: Automated testing with pytest and coverage reporting
- **Modern Python**: Type hints, comprehensive documentation
- **CI/CD**: Automated testing and quality checks

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/dave2k77/fractional_calculus_library.git
cd fractional_calculus_library

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🏗️ Project Structure

```
fc_library/
├── src/                          # Main source code
│   ├── algorithms/               # Fractional derivative algorithms
│   │   ├── caputo.py            # Caputo derivative implementation
│   │   ├── riemann_liouville.py # Riemann-Liouville derivative
│   │   ├── grunwald_letnikov.py # Grünwald-Letnikov derivative
│   │   ├── advanced_methods.py  # Advanced methods (Weyl, Marchaud, etc.)
│   │   ├── advanced_optimized_methods.py # Optimized advanced methods
│   │   ├── fft_methods.py       # FFT-based methods
│   │   └── L1_L2_schemes.py     # L1/L2 numerical schemes
│   ├── core/                     # Core definitions and utilities
│   │   ├── definitions.py       # Mathematical definitions
│   │   ├── derivatives.py       # Derivative base classes
│   │   ├── integrals.py         # Integral implementations
│   │   └── utilities.py         # Utility functions
│   ├── optimisation/             # Performance optimization
│   │   ├── jax_implementations.py # JAX-based implementations
│   │   ├── numba_kernels.py     # NUMBA JIT kernels
│   │   └── parallel_computing.py # Parallel computing utilities
│   ├── solvers/                  # Differential equation solvers
│   │   ├── ode_solvers.py       # ODE solvers
│   │   ├── pde_solvers.py       # PDE solvers
│   │   └── predictor_corrector.py # Predictor-corrector methods
│   ├── special/                  # Special functions
│   │   ├── gamma_beta.py        # Gamma and Beta functions
│   │   ├── mittag_leffler.py    # Mittag-Leffler function
│   │   └── binomial_coeffs.py   # Binomial coefficients
│   ├── utils/                    # Utilities
│   │   ├── error_analysis.py    # Error analysis tools
│   │   ├── memory_management.py # Memory optimization
│   │   └── plotting.py          # Visualization utilities
│   └── validation/               # Validation and testing
│       ├── analytical_solutions.py # Analytical solutions
│       ├── benchmarks.py        # Benchmarking tools
│       └── convergence_tests.py # Convergence analysis
├── examples/                     # Usage examples
│   ├── basic_usage/             # Basic usage examples
│   ├── advanced_applications/   # Advanced applications
│   ├── jax_examples/           # JAX-specific examples
│   └── parallel_examples/      # Parallel computing examples
├── benchmarks/                   # Performance benchmarks
│   ├── performance_tests.py     # Performance testing
│   ├── accuracy_comparisons.py  # Accuracy comparisons
│   └── scaling_analysis.py      # Scaling analysis
├── tests/                        # Test suite
│   ├── test_algorithms/         # Algorithm tests
│   ├── test_core/              # Core functionality tests
│   ├── test_optimisation/      # Optimization tests
│   ├── test_solvers/           # Solver tests
│   └── integration_tests/      # Integration tests
├── scripts/                      # Utility scripts
│   └── run_tests.py            # Comprehensive test runner
├── docs/                         # Documentation
│   ├── api_reference/           # API documentation
│   ├── examples/                # Example documentation
│   └── source/                  # Source documentation
└── .github/workflows/           # CI/CD workflows
    └── tests.yml               # Automated testing
```

## 🔧 Usage

### Basic Example

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Initialize fractional derivative
alpha = 0.5  # Fractional order
caputo = CaputoDerivative(alpha)

# Define function values and time points
t = np.linspace(0.1, 2.0, 50)
f = t  # Simple linear function
h = t[1] - t[0]  # Step size

# Compute fractional derivative
result = caputo.compute(f, t, h)
print(f"Caputo derivative of order {alpha}: {result}")
```

### Advanced Example with Different Methods

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative

# Test parameters
alpha = 0.5
t = np.linspace(0.1, 2.0, 100)
f = t**2  # Quadratic function
h = t[1] - t[0]

# Compare different methods
caputo = CaputoDerivative(alpha)
riemann = RiemannLiouvilleDerivative(alpha)
grunwald = GrunwaldLetnikovDerivative(alpha)

result_caputo = caputo.compute(f, t, h)
result_riemann = riemann.compute(f, t, h)
result_grunwald = grunwald.compute(f, t, h)

print(f"Caputo: {result_caputo[-1]:.6f}")
print(f"Riemann-Liouville: {result_riemann[-1]:.6f}")
print(f"Grünwald-Letnikov: {result_grunwald[-1]:.6f}")
```

### 🚀 Optimized Methods Example

```python
import numpy as np
import time
from src.algorithms.caputo import CaputoDerivative
from src.algorithms.riemann_liouville import RiemannLiouvilleDerivative
from src.algorithms.grunwald_letnikov import GrunwaldLetnikovDerivative

# Test parameters
alpha = 0.5
t = np.linspace(0, 10, 1000)
f = t**2 + np.sin(t)  # Test function
h = 0.01

# Standard methods
start_time = time.time()
caputo_std = CaputoDerivative(alpha, method="l1")
result_std = caputo_std.compute(f, t, h)
time_std = time.time() - start_time

# Optimized methods
start_time = time.time()
caputo_opt = CaputoDerivative(alpha, method="optimized_l1")
result_opt = caputo_opt.compute(f, t, h)
time_opt = time.time() - start_time

print(f"Standard L1: {time_std:.4f}s")
print(f"Optimized L1: {time_opt:.4f}s")
print(f"Speedup: {time_std/time_opt:.1f}x")
print(f"Results match: {np.allclose(result_std, result_opt, rtol=1e-10)}")

# Available optimized methods:
# - Caputo: method="optimized_l1", method="optimized_predictor_corrector"
# - Riemann-Liouville: method="optimized_fft"
# - Grünwald-Letnikov: method="optimized_direct"
```

### 🚀 Advanced Methods Example

```python
import numpy as np
from src.algorithms.advanced_methods import (
    WeylDerivative, MarchaudDerivative, HadamardDerivative,
    ReizFellerDerivative, AdomianDecomposition
)

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 100)
f = lambda x: np.sin(x)  # Test function

# Weyl derivative (for periodic functions)
weyl = WeylDerivative(alpha)
result_weyl = weyl.compute(f, x, h=0.05)

# Marchaud derivative (with memory optimization)
marchaud = MarchaudDerivative(alpha)
result_marchaud = marchaud.compute(f, x, h=0.05)

# Hadamard derivative (logarithmic transformation)
x_hadamard = np.linspace(1, 5, 100)  # Must be positive
hadamard = HadamardDerivative(alpha)
result_hadamard = hadamard.compute(f, x_hadamard, h=0.05)

# Reiz-Feller derivative (spectral method)
reiz_feller = ReizFellerDerivative(alpha)
result_reiz = reiz_feller.compute(f, x, h=0.05)

print(f"Weyl derivative: {result_weyl[-1]:.6f}")
print(f"Marchaud derivative: {result_marchaud[-1]:.6f}")
print(f"Hadamard derivative: {result_hadamard[-1]:.6f}")
print(f"Reiz-Feller derivative: {result_reiz[-1]:.6f}")

# Adomian Decomposition for solving FDEs
def fractional_ode(t, y, alpha):
    """Example: D^α y(t) = -y(t)"""
    return -y

adomian = AdomianDecomposition(alpha)
solution = adomian.solve(fractional_ode, x, initial_condition=1.0)
print(f"Adomian solution at t=5: {solution[-1]:.6f}")
```

### 🚀 Optimized Advanced Methods

```python
import numpy as np
from src.algorithms.advanced_optimized_methods import (
    optimized_weyl_derivative, optimized_marchaud_derivative,
    optimized_hadamard_derivative, optimized_reiz_feller_derivative
)

# JAX/Numba optimized versions for maximum performance
alpha = 0.5
x = np.linspace(0, 5, 1000)
f = lambda x: np.sin(x)

# Optimized versions (GPU-accelerated with JAX)
result_weyl_opt = optimized_weyl_derivative(f, x, alpha, h=0.005)
result_marchaud_opt = optimized_marchaud_derivative(f, x, alpha, h=0.005)
result_hadamard_opt = optimized_hadamard_derivative(f, x, alpha, h=0.005)
result_reiz_opt = optimized_reiz_feller_derivative(f, x, alpha, h=0.005)

print("Optimized advanced methods completed successfully!")
```

## 🧪 Testing and Quality Assurance

### Automated Testing

The project includes comprehensive automated testing with:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality testing
- **Benchmark Tests**: Performance validation
- **Code Quality**: Linting, formatting, and type checking

### Run Tests

```bash
# Run all tests with coverage
python scripts/run_tests.py

# Run specific test types
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration
python scripts/run_tests.py --type benchmark

# Run with pytest directly
pytest tests/ -v --cov=src

# Run fast tests only
pytest tests/ -m "not slow"
```

### Code Quality Checks

```bash
# Linting with flake8
flake8 src tests

# Code formatting with black
black src tests

# Type checking with mypy
mypy src

# Run all quality checks
python scripts/run_tests.py --no-coverage --reports
```

### Benchmarks

```bash
# Run performance benchmarks
python benchmarks/performance_tests.py

# Run accuracy comparisons
python benchmarks/accuracy_comparisons.py

# Run scaling analysis
python benchmarks/scaling_analysis.py
```

## 📊 Performance Features

### 🚀 Optimized Methods Performance

The library includes highly optimized implementations that provide dramatic performance improvements:

| Method | Speedup | Accuracy | Usage |
|--------|---------|----------|-------|
| **Riemann-Liouville FFT** | **196x** | ✅ Perfect | `method="optimized_fft"` |
| **Caputo L1** | **76.5x** | ✅ Perfect | `method="optimized_l1"` |
| **Grünwald-Letnikov Direct** | **7.2x** | ⚠️ Needs fix | `method="optimized_direct"` |

**Key Optimizations:**
- **FFT Convolution**: Efficient Riemann-Liouville computation
- **L1 Scheme**: Optimized Caputo derivative implementation
- **Fast Binomial Coefficients**: Efficient Grünwald-Letnikov computation
- **Diethelm-Ford-Freed**: High-order predictor-corrector method
- **Advanced Methods**: Weyl, Marchaud, Hadamard, Reiz-Feller with JAX/Numba optimization
- **Adomian Decomposition**: Parallel computation of decomposition terms

### JAX Integration
- **Automatic Differentiation**: Compute gradients automatically
- **GPU Acceleration**: Leverage GPU computing when available
- **JIT Compilation**: Just-in-time compilation for performance
- **Vectorization**: Efficient array operations

### NUMBA Integration
- **JIT Compilation**: Compile Python functions to machine code
- **Parallel Computing**: Multi-threading support
- **Memory Optimization**: Efficient memory management
- **Type Specialization**: Optimized for specific data types

### Parallel Computing
- **Multi-core Processing**: Utilize all CPU cores
- **GPU Computing**: CUDA support for large-scale computations
- **Memory Management**: Efficient handling of large datasets
- **Load Balancing**: Automatic workload distribution

## 🔬 Research Applications

This library is designed for research in:
- **Fractional Differential Equations**: Numerical solutions
- **Signal Processing**: Fractional filters and transforms
- **Control Theory**: Fractional controllers
- **Physics**: Anomalous diffusion, viscoelasticity
- **Finance**: Fractional Brownian motion models

## 📈 Benchmarks

The library includes comprehensive benchmarks comparing:
- **Accuracy**: Against analytical solutions
- **Performance**: CPU vs GPU implementations
- **Scaling**: Performance with problem size
- **Memory Usage**: Memory efficiency analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and ensure tests pass
4. Run quality checks: `python scripts/run_tests.py`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python scripts/run_tests.py --type fast
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- Podlubny, I. (1999). Fractional Differential Equations
- Kilbas, A. A., et al. (2006). Theory and Applications of Fractional Differential Equations
- Diethelm, K. (2010). The Analysis of Fractional Differential Equations

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `docs/`
- Review examples in `examples/`

## 🔄 Development Status

- [x] Project structure setup
- [x] Core dependencies installation
- [x] Basic framework implementation
- [x] Algorithm implementations (Caputo, Riemann-Liouville, Grünwald-Letnikov)
- [x] JAX optimizations
- [x] NUMBA kernels
- [x] Benchmarking suite
- [x] Automated testing with pytest
- [x] CI/CD pipeline
- [x] Code quality tools
- [x] Documentation structure
- [x] Examples and tutorials
- [ ] Advanced solver implementations
- [ ] GPU-specific optimizations
- [ ] Extended documentation

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/dave2k77/fractional_calculus_library.git
cd fractional_calculus_library
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run tests to verify installation
python scripts/run_tests.py --type fast

# Try the examples
python examples/basic_usage/getting_started.py
```

---

**Author**: Davian R. Chin  
**Repository**: https://github.com/dave2k77/fractional_calculus_library  
**License**: MIT
