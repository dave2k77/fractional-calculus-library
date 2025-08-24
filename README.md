# HPFRACC - High-Performance Fractional Calculus Library

[![PyPI version](https://badge.fury.io/py/hpfracc.svg)](https://badge.fury.io/py/hpfracc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hpfracc.readthedocs.io/)

## 👨‍💻 **Author & Developer**

**Davian R. Chin**  
Department of Biomedical Engineering  
University of Reading  
Email: [d.r.chin@pgr.reading.ac.uk](mailto:d.r.chin@pgr.reading.ac.uk)  
GitHub: [@dave2k77](https://github.com/dave2k77)

## 🚀 **Overview**

HPFRACC (High-Performance Fractional Calculus Library) is a comprehensive Python library that provides high-performance implementations of fractional calculus operations, advanced numerical methods, and machine learning integration with fractional derivatives.

## ✨ **Key Features**

### 🔬 **Core Fractional Calculus**
- **Multiple Definitions**: Riemann-Liouville, Caputo, Grünwald-Letnikov, and more
- **High-Performance Algorithms**: Optimized implementations for speed and accuracy
- **GPU Acceleration**: CUDA support for large-scale computations
- **Advanced Methods**: Mellin transforms, fractional differential equations, and special functions

### 🤖 **Machine Learning Integration**
- **Multi-Backend Support**: Seamless integration with PyTorch, JAX, and NUMBA
- **Fractional Neural Networks**: Core networks with fractional calculus integration
- **Fractional Attention Mechanisms**: Multi-head attention with fractional derivatives
- **Graph Neural Networks**: Fractional GNN architectures for graph learning tasks
- **Comprehensive ML Components**: Loss functions, optimizers, and layers (in development)

### 🎯 **Performance & Usability**
- **Cross-Platform**: Windows, macOS, and Linux support
- **Extensive Documentation**: Comprehensive guides and examples
- **Active Development**: Regular updates and improvements
- **Research-Ready**: Designed for academic and industrial applications

## 🏗️ **Architecture**

### **Multi-Backend Support**
HPFRACC provides a unified interface across multiple computation backends:

- **PyTorch**: Full-featured deep learning with GPU acceleration
- **JAX**: High-performance numerical computing with automatic differentiation
- **NUMBA**: JIT compilation for CPU optimization

### **Core Components**
- **Backend Management**: Automatic detection and seamless switching between backends
- **Unified Tensor Operations**: Consistent API across all backends
- **Fractional Calculus Integration**: Built-in fractional derivatives in all ML components

## 📦 **Installation**

### **Basic Installation**
```bash
pip install hpfracc
```

### **Full Installation with ML Dependencies**
```bash
pip install hpfracc[ml]
```

### **Development Installation**
```bash
git clone https://github.com/dave2k77/hpfracc.git
cd hpfracc
pip install -e .
```

## 🚀 **Quick Start**

### **Basic Fractional Calculus**
```python
from hpfracc import FractionalOrder, riemann_liouville

# Create fractional order
alpha = FractionalOrder(0.5)

# Compute fractional derivative
result = riemann_liouville(function, t, alpha)
```

### **Multi-Backend Neural Networks**
```python
from hpfracc.ml import BackendType, FractionalNeuralNetwork
from hpfracc.core.definitions import FractionalOrder

# Create network with JAX backend
network = FractionalNeuralNetwork(
    input_size=10,
    hidden_sizes=[32, 16],
    output_size=2,
    fractional_order=FractionalOrder(0.5),
    backend=BackendType.JAX
)

# Forward pass with fractional derivatives
output = network(input_data, use_fractional=True, method="RL")
```

### **Fractional Attention Mechanism**
```python
from hpfracc.ml import FractionalAttention

# Create attention with fractional calculus
attention = FractionalAttention(
    d_model=64,
    n_heads=8,
    fractional_order=FractionalOrder(0.5),
    backend=BackendType.TORCH
)

# Apply fractional attention
output = attention(input_sequence, method="RL")
```

## 🔧 **Current Status**

### **✅ Fully Working**
- **Core Fractional Calculus**: All mathematical operations and algorithms
- **Backend Management**: Seamless switching between PyTorch, JAX, and NUMBA
- **Core Neural Networks**: FractionalNeuralNetwork with multi-backend support
- **Attention Mechanisms**: FractionalAttention with fractional derivatives
- **Tensor Operations**: Unified API across all backends

### **🚧 In Development**
- **Advanced Layers**: Conv1D, Conv2D, LSTM, Transformer layers
- **Loss Functions**: Comprehensive loss function library
- **Optimizers**: Fractional gradient-based optimizers
- **Graph Neural Networks**: Fractional GNN architectures

### **📋 Planned Features**
- **Advanced ML Components**: Complete layer and optimizer library
- **Performance Optimization**: Backend-specific optimizations
- **Extended GNN Support**: More graph neural network architectures
- **Research Tools**: Benchmarking and analysis utilities

## 📚 **Documentation**

- **User Guide**: [docs/user_guide.md](docs/user_guide.md)
- **API Reference**: [docs/api_reference.md](docs/api_reference.md)
- **Examples**: [examples/](examples/) directory
- **Development Guide**: [README_DEV.md](README_DEV.md)

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Development Guide](README_DEV.md) for details.

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/dave2k77/hpfracc.git
cd hpfracc

# Create virtual environment
conda create -n hpfracc_dev python=3.9
conda activate hpfracc_dev

# Install development dependencies
pip install -e .[dev]
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **University of Reading**: Department of Biomedical Engineering
- **Open Source Community**: Contributors and maintainers
- **Research Community**: Academic and industrial partners

## 📞 **Contact**

- **Email**: [d.r.chin@pgr.reading.ac.uk](mailto:d.r.chin@pgr.reading.ac.uk)
- **GitHub**: [@dave2k77](https://github.com/dave2k77)
- **Project**: [HPFRACC Repository](https://github.com/dave2k77/hpfracc)

---

**HPFRACC** - Advancing fractional calculus through high-performance computing and machine learning integration.
