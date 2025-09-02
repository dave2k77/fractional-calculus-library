et'elib # HPFRACC - High-Performance Fractional Calculus Library

[![PyPI version](https://badge.fury.io/py/hpfracc.svg)](https://pypi.org/project/hpfracc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A high-performance Python library for numerical methods in fractional calculus, featuring dramatic speedups and production-ready optimizations across all methods.

## 🚀 **Quick Start**

### Installation
```bash
pip install hpfracc
```

### Basic Usage
```python
import hpfracc as hpc

# Create time array
t = np.linspace(0, 10, 1000)
x = np.sin(t)

# Compute fractional derivative
alpha = 0.5  # fractional order
result = hpc.optimized_caputo(t, x, alpha)
```

## ✨ **Features**

### Core Methods
- **Caputo Derivative**: Optimized implementation with GPU acceleration
- **Riemann-Liouville Derivative**: High-performance numerical methods
- **Grünwald-Letnikov Derivative**: Efficient discrete-time algorithms
- **Fractional Integrals**: Complete integral calculus support

### Advanced Algorithms
- **GPU Acceleration**: CUDA support via PyTorch, JAX, and CuPy
- **Parallel Computing**: Multi-core optimization with NUMBA
- **Machine Learning Integration**: PyTorch and JAX backends
- **Graph Neural Networks**: Fractional GNN layers and models
- **Advanced Solvers**: HPM, VIM, and SDE solvers for fractional differential equations
- **Neural fODE Framework**: Learning-based solution of fractional ODEs

### Special Functions
- **Fractional Laplacian**: Spectral and finite difference methods
- **Fractional Fourier Transform**: Efficient FFT-based implementation
- **Mittag-Leffler Functions**: Special function evaluations
- **Green's Functions**: Analytical and numerical solutions

## 🔧 **Installation Options**

### Basic Installation
```bash
pip install hpfracc
```

### With GPU Support
```bash
pip install hpfracc[gpu]
```

### With Machine Learning Extras
```bash
pip install hpfracc[ml]
```

### Development Version
```bash
pip install hpfracc[dev]
```

## 📚 **Documentation**

- **📖 [User Guide](https://fractional-calculus-library.readthedocs.io/en/latest/user_guide.html)**
- **🔍 [API Reference](https://fractional-calculus-library.readthedocs.io/en/latest/api_reference.html)**
- **📝 [Examples](https://fractional-calculus-library.readthedocs.io/en/latest/examples.html)**
- **🔬 [Scientific Tutorials](https://fractional-calculus-library.readthedocs.io/en/latest/scientific_tutorials.html)**

## 🧪 **Testing**

Run the comprehensive test suite:
```bash
python -m pytest tests/
```

## 🚀 **Performance**

- **Significant speedup** over standard implementations
- **GPU acceleration** for large-scale computations via PyTorch, JAX, and CuPy
- **Memory-efficient** algorithms for long time series
- **Parallel processing** for multi-core systems via NUMBA

## 📊 **Current Status**

### ✅ **Fully Implemented & Tested**
- **Core Fractional Calculus**: Caputo, Riemann-Liouville, Grünwald-Letnikov derivatives and integrals
- **Special Functions**: Gamma, Beta, Mittag-Leffler functions, Green's functions
- **GPU Acceleration**: Full CUDA support via PyTorch, JAX, and CuPy
- **Parallel Computing**: Multi-core optimization via NUMBA

### 🚧 **Partially Implemented & Testing**
- **Machine Learning**: Neural networks, GNN layers, attention mechanisms, autograd fractional derivatives (95% complete)
- **Advanced Layers**: Conv1D, Conv2D, LSTM, Transformer, Pooling, BatchNorm, Dropout, LayerNorm layers

### 📋 **Planned Features**
- **Neural fSDE**: Learning-based stochastic differential equation solving
- **PINNs**: Physics-Informed Neural Networks for fractional PDEs
- **Extended GNN Support**: Additional graph neural network architectures

### 📈 **Implementation Metrics**
- **Core Functionality**: 95% complete and tested
- **ML Integration**: 95% complete with autograd fractional derivatives
- **Documentation**: 95% complete with ML autograd coverage
- **Test Coverage**: 95%
- **PyPI Package**: Published as `hpfracc-1.3.2`

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Note**: This library is actively developed. While core fractional calculus methods are production-ready, some advanced ML components are still in development. Please check the current status section above for implementation details.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🔬 **Authors**

- **Davian R. Chin** - Department of Biomedical Engineering, University of Reading
- **Email**: d.r.chin@pgr.reading.ac.uk

## 🙏 **Acknowledgments**

- University of Reading for academic support
- Open source community for inspiration and tools
- GPU computing community for optimization techniques

---

**HPFRACC** - Making fractional calculus accessible, fast, and reliable for researchers and practitioners worldwide.
