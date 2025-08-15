# Fractional Calculus Library - Documentation

Welcome to the comprehensive documentation for the Fractional Calculus Library. This documentation provides everything you need to understand, install, and use the library effectively.

## 📚 Documentation Overview

The Fractional Calculus Library is a high-performance Python library for numerical methods in fractional calculus, leveraging JAX and NUMBA for optimized computations and parallel processing.

## 🗂️ Documentation Structure

### 📖 Getting Started
- **[Installation Guide](installation_guide.md)** - Complete setup instructions for all platforms
- **[User Guide](user_guide.md)** - Comprehensive guide to using the library
- **[Basic Examples](examples/basic_examples.md)** - Simple examples to get you started

### 🔧 Development
- **[Contributing Guidelines](contributing.md)** - How to contribute to the project
- **[Project Status](project_status_and_issues.md)** - Current status and known issues
- **[Class Documentation](class_documentation.md)** - Detailed API documentation

### 📊 Advanced Topics
- **[Mathematical Documentation](fractional-calculus-documentation.md)** - Mathematical foundations and theory
- **[Performance Optimization](parallel_computing_alternatives.md)** - Parallel computing strategies
- **[Joblib Implementation](joblib_implementation_summary.md)** - Implementation details

### 🔍 Reference
- **[API Reference](api_reference/)** - Auto-generated API documentation
- **[Examples](examples/)** - Code examples and tutorials
- **[Source Documentation](source/)** - Source code documentation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dave2k77/fractional_calculus_library.git
cd fractional_calculus_library

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Create fractional derivative calculator
alpha = 0.5  # Half-derivative
caputo = CaputoDerivative(alpha)

# Define function and compute derivative
t = np.linspace(0.1, 2.0, 100)
f = t**2  # Function f(t) = t²
h = t[1] - t[0]

result = caputo.compute(f, t, h)
print(f"Caputo derivative: {result[-1]:.6f}")
```

### Run Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run fast tests only
python scripts/run_tests.py --type fast

# Run with coverage
python scripts/run_tests.py --coverage
```

---

## 📋 Documentation by Topic

### For New Users
1. **[Installation Guide](installation_guide.md)** - Start here to set up the library
2. **[User Guide](user_guide.md)** - Learn how to use the library
3. **[Basic Examples](examples/basic_examples.md)** - Try simple examples

### For Researchers
1. **[Mathematical Documentation](fractional-calculus-documentation.md)** - Mathematical theory and foundations
2. **[Class Documentation](class_documentation.md)** - Detailed API reference
3. **[Performance Optimization](parallel_computing_alternatives.md)** - Optimization strategies

### For Developers
1. **[Contributing Guidelines](contributing.md)** - How to contribute
2. **[Project Status](project_status_and_issues.md)** - Current development status
3. **[API Reference](api_reference/)** - Complete API documentation

### For Advanced Users
1. **[Performance Examples](examples/performance_examples.md)** - Advanced optimization examples
2. **[Research Applications](examples/research_applications.md)** - Real-world applications
3. **[Benchmarking](examples/benchmarking.md)** - Performance benchmarking

---

## 🎯 Key Features

### Core Algorithms
- **Caputo Derivative** - Ideal for initial value problems
- **Riemann-Liouville Derivative** - Classical fractional derivative
- **Grünwald-Letnikov Derivative** - Limit-based definition
- **Fractional Integrals** - Riemann-Liouville integrals

### Performance Optimization
- **JAX Integration** - GPU acceleration and automatic differentiation
- **NUMBA Kernels** - JIT compilation for critical numerical operations
- **Parallel Computing** - Multi-core and distributed processing
- **Memory Management** - Optimized memory usage patterns

### Validation and Testing
- **Analytical Solutions** - Validation against known solutions
- **Convergence Analysis** - Error analysis and convergence studies
- **Benchmarking Suite** - Performance and accuracy benchmarks
- **Comprehensive Testing** - Unit, integration, and performance tests

---

## 📊 Current Status

### ✅ Completed Features
- Core fractional derivative algorithms
- JAX and NUMBA optimization
- Parallel computing with Joblib
- Comprehensive testing framework
- Basic documentation structure

### 🔄 In Progress
- Advanced solver implementations
- GPU-specific optimizations
- Extended documentation
- Performance benchmarking

### 📋 Planned Features
- Additional fractional derivative definitions
- Advanced PDE solvers
- Machine learning integration
- Cloud computing support

---

## 🧪 Testing and Quality

### Test Coverage
- **Total Tests**: 73/73 tests passing (100% success rate)
- **Coverage**: 49% overall code coverage
- **Test Categories**: Unit, integration, performance, and accuracy tests

### Quality Assurance
- **Code Quality**: flake8 linting, black formatting, mypy type checking
- **Performance**: Automated benchmarking and performance regression tests
- **Documentation**: Comprehensive documentation with examples
- **CI/CD**: Automated testing and quality checks

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Types of Contributions
- **Bug Fixes** - Report and fix issues
- **Feature Development** - Add new functionality
- **Documentation** - Improve guides and examples
- **Testing** - Add tests or improve coverage
- **Performance** - Optimize existing code
- **Research** - Implement new algorithms

### Getting Started
1. Read the **[Contributing Guidelines](contributing.md)**
2. Check existing issues and discussions
3. Fork the repository and create a feature branch
4. Make your changes and add tests
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python scripts/run_tests.py --type fast
```

---

## 📞 Support and Community

### Getting Help
- **Documentation**: Start with this guide and the user guide
- **Examples**: Check the examples directory for usage patterns
- **Issues**: Search existing issues or create new ones
- **Discussions**: Join GitHub discussions for questions

### Resources
- **GitHub Repository**: [fractional_calculus_library](https://github.com/dave2k77/fractional_calculus_library)
- **Issues**: [GitHub Issues](https://github.com/dave2k77/fractional_calculus_library/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dave2k77/fractional_calculus_library/discussions)

### Contact
- **Maintainer**: @dave2k77
- **Email**: [Project email]
- **Discussions**: [GitHub Discussions]

---

## 📈 Performance

### Benchmarks
- **Speed**: 10-100x improvement for large datasets
- **Scalability**: Linear scaling demonstrated
- **Memory**: Optimized memory usage patterns
- **GPU**: Basic support with JAX integration

### Optimization Features
- **JAX Acceleration**: GPU computing and automatic differentiation
- **NUMBA JIT**: Just-in-time compilation for numerical kernels
- **Parallel Processing**: Multi-core and distributed computing
- **Memory Management**: Efficient memory usage and caching

---

## 🔬 Research Applications

The library is designed for research in:
- **Fractional Differential Equations** - Numerical solutions
- **Signal Processing** - Fractional filters and transforms
- **Control Theory** - Fractional controllers
- **Physics** - Anomalous diffusion, viscoelasticity
- **Finance** - Fractional Brownian motion models

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Mathematical Foundation**: Based on established fractional calculus theory
- **Performance Libraries**: Built on JAX, NUMBA, and NumPy
- **Community**: Contributions from the open-source community
- **Research**: Inspired by research in fractional calculus applications

---

## 📚 References

### Key Papers
- Podlubny, I. (1999). Fractional Differential Equations
- Kilbas, A. A., et al. (2006). Theory and Applications of Fractional Differential Equations
- Diethelm, K. (2010). The Analysis of Fractional Differential Equations

### Software References
- JAX: Autograd and XLA for high-performance machine learning research
- NUMBA: JIT compiler for Python and NumPy
- SciPy: Scientific computing library for Python

---

**Note**: This documentation is regularly updated. For the latest version, check the [GitHub repository](https://github.com/dave2k77/fractional_calculus_library).

---

*Last updated: December 2024*
