# HPFRACC Documentation

Welcome to the **HPFRACC** (High-Performance Fractional Calculus) documentation!

## What is HPFRACC?

**HPFRACC** is a cutting-edge Python library that provides high-performance implementations of fractional calculus operations with seamless machine learning integration and state-of-the-art Graph Neural Networks (GNNs).

### Key Features

* **Advanced Fractional Calculus**: Riemann-Liouville, Caputo, Grünwald-Letnikov definitions
* **Fractional Integrals**: Riemann-Liouville, Caputo, Weyl, and Hadamard integrals
* **Special Functions**: Gamma, Beta, Mittag-Leffler, and Binomial coefficients
* **Fractional Green's Functions**: For diffusion, wave, and advection equations
* **Analytical Methods**: Homotopy Perturbation Method (HPM) and Variational Iteration Method (VIM)
* **Machine Learning Integration**: Native PyTorch, JAX, and NUMBA support
* **Graph Neural Networks**: GCN, GAT, GraphSAGE, and Graph U-Net architectures
* **High Performance**: Optimized algorithms with GPU acceleration support
* **Multi-Backend**: Seamless switching between computation backends
* **Comprehensive Analytics**: Built-in performance monitoring and error analysis

## Quick Start

### Installation

```bash
# Basic installation
pip install hpfracc

# Full installation with ML dependencies
pip install hpfracc[ml]

# Development installation
pip install hpfracc[dev]
```

### Basic Usage

```python
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.special import gamma, beta, mittag_leffler
from hpfracc.solvers import HomotopyPerturbationSolver
import numpy as np

# Create a fractional integral
integral = create_fractional_integral(0.5, method="RL")

# Compute special functions
result = gamma(5.5)
beta_val = beta(2.5, 3.5)
ml_result = mittag_leffler(0.5, 1.0, 2.0)

# Solve fractional differential equation with HPM
solver = HomotopyPerturbationSolver()
solution = solver.solve(lambda x, t: x**2 + t, initial_condition=lambda x: x)
```

## Documentation Sections

### Core Concepts

* [**Model Theory**](model_theory.rst) - Mathematical foundations and theoretical background
* [**User Guide**](user_guide.md) - Getting started and basic usage patterns

### API Reference

* [**Core API**](api_reference.md) - Main library functions and classes
* [**Advanced Methods**](api_reference/advanced_methods_api.md) - Specialized algorithms and optimizations

### Examples & Tutorials

* [**Examples Gallery**](examples.md) - Comprehensive code examples and use cases
* [**Scientific Tutorials**](scientific_tutorials.md) - Advanced scientific tutorials and research applications
* [**ML Integration Guide**](ml_integration_guide.md) - Machine learning workflows and best practices

### Development & Testing

* [**Testing Status**](testing_status.md) - Current test coverage and validation status

## Why Choose HPFRACC?

### Academic Excellence

* Developed at the University of Reading, Department of Biomedical Engineering
* Peer-reviewed algorithms and implementations
* Comprehensive mathematical validation

### Production Ready

* Extensive test coverage (>95%)
* Performance benchmarking and optimization
* Multi-platform compatibility

### Active Development

* Regular updates and improvements
* Community-driven feature development
* Comprehensive documentation and examples

## Quick Links

* **GitHub Repository**: [fractional_calculus_library](https://github.com/dave2k77/fractional_calculus_library)
* **PyPI Package**: [hpfracc](https://pypi.org/project/hpfracc/)
* **Issue Tracker**: [GitHub Issues](https://github.com/dave2k77/fractional_calculus_library/issues)
* **Academic Contact**: [d.r.chin@pgr.reading.ac.uk](mailto:d.r.chin@pgr.reading.ac.uk)

## Citation

If you use HPFRACC in your research, please cite:

```bibtex
@software{hpfracc2025,
  title={HPFRACC: High-Performance Fractional Calculus Library with Machine Learning Integration},
  author={Chin, Davian R.},
  year={2025},
  url={https://github.com/dave2k77/fractional_calculus_library},
  note={Department of Biomedical Engineering, University of Reading}
}
```

## Getting Help

* **Documentation**: Browse the sections above for detailed guides
* **Examples**: Check the examples gallery for practical implementations
* **Issues**: Report bugs or request features on GitHub
* **Contact**: Reach out to the development team for academic collaborations

---

**HPFRACC v1.2.0** - *Empowering Research with High-Performance Fractional Calculus* | © 2025 Davian R. Chin
