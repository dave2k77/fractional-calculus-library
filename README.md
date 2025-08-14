# Fractional Calculus Library

A high-performance Python library for numerical methods in fractional calculus, leveraging JAX and NUMBA for optimized computations and parallel processing.

## 🚀 Features

- **Multiple Fractional Derivative Definitions**: Caputo, Riemann-Liouville, Grünwald-Letnikov
- **High-Performance Computing**: JAX for automatic differentiation and GPU acceleration
- **JIT Compilation**: NUMBA for optimized numerical kernels
- **Parallel Computing**: Multi-core and GPU support
- **Comprehensive Testing**: Benchmarking and validation suite
- **Modern Python**: Type hints, comprehensive documentation

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
```

## 🏗️ Project Structure

```
fc_library/
├── src/                          # Main source code
│   ├── algorithms/               # Fractional derivative algorithms
│   │   ├── caputo.py            # Caputo derivative implementation
│   │   ├── riemann_liouville.py # Riemann-Liouville derivative
│   │   ├── grunwald_letnikov.py # Grünwald-Letnikov derivative
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
└── docs/                         # Documentation
    ├── api_reference/           # API documentation
    ├── examples/                # Example documentation
    └── source/                  # Source documentation
```

## 🔧 Usage

### Basic Example

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative
from src.optimisation.jax_implementations import JAXCaputo

# Initialize fractional derivative
alpha = 0.5  # Fractional order
caputo = CaputoDerivative(alpha)

# Define function
def f(x):
    return x**2

# Compute fractional derivative
x = np.linspace(0, 1, 100)
result = caputo.compute(f, x)

# Using JAX for automatic differentiation
jax_caputo = JAXCaputo(alpha)
jax_result = jax_caputo.compute(f, x)
```

### Advanced Example with JAX and NUMBA

```python
import jax
import jax.numpy as jnp
from src.solvers.pde_solvers import hybrid_solver

# Define PDE parameters
params = {
    'alpha': 0.5,  # Fractional order
    'dt': 0.01,    # Time step
    'dx': 0.1      # Spatial step
}

# Initial conditions
u0 = jnp.sin(jnp.linspace(0, 2*jnp.pi, 100))

# Solve fractional PDE
solution = hybrid_solver(u0, params)
```

## 🧪 Testing and Benchmarking

### Run Tests
```bash
pytest tests/
```

### Run Benchmarks
```bash
python benchmarks/performance_tests.py
```

### Run Accuracy Comparisons
```bash
python benchmarks/accuracy_comparisons.py
```

## 📊 Performance Features

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
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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
- [ ] Algorithm implementations
- [ ] JAX optimizations
- [ ] NUMBA kernels
- [ ] Benchmarking suite
- [ ] Documentation
- [ ] Testing suite
- [ ] Examples and tutorials

---

**Author**: David  
**Repository**: https://github.com/dave2k77/fractional_calculus_library
