# HPFRACC Development Guide

This document is for developers and contributors working on the HPFRACC library. For user documentation, see the main [README.md](README.md) and [docs/](docs/) directory.

## 🏗️ **Project Structure**

```
hpfracc/
├── hpfracc/                    # Main package
│   ├── core/                   # Core fractional calculus
│   │   ├── __init__.py
│   │   ├── definitions.py      # FractionalOrder class
│   │   ├── derivatives.py      # Fractional derivative implementations
│   │   └── utils.py           # Utility functions
│   ├── ml/                     # Machine learning integration
│   │   ├── __init__.py
│   │   ├── layers.py          # Fractional neural network layers
│   │   ├── networks.py        # Neural network architectures
│   │   ├── optimizers.py      # Fractional optimizers
│   │   ├── losses.py          # Fractional loss functions
│   │   ├── registry.py        # Model registry system
│   │   ├── workflow.py        # Development/production workflows
│   │   └── adjoint_optimization.py  # Adjoint method optimization
│   ├── benchmarks/             # Performance benchmarking
│   │   ├── __init__.py
│   │   └── ml_performance_benchmark.py
│   └── analytics/              # Usage analytics and monitoring
│       ├── __init__.py
│       ├── usage_analytics.py
│       ├── performance_monitor.py
│       └── error_analyzer.py
├── tests/                      # Test suite
│   ├── test_core.py
│   ├── test_ml_integration.py
│   ├── test_benchmarks.py
│   └── test_analytics.py
├── examples/                    # Example scripts
│   ├── ml_integration_demo.py
│   └── fractional_calculus_examples.py
├── docs/                       # Documentation
│   ├── api_reference.md
│   ├── user_guide.md
│   ├── ml_integration_guide.md
│   ├── model_theory.md
│   └── examples.md
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── pyproject.toml             # Project configuration
├── setup.py                   # Package setup
├── README.md                  # Main user documentation
├── README_DEV.md              # This development guide
└── LICENSE                    # MIT License
```

## 🚀 **Getting Started**

### **Prerequisites**

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)
- PyTorch 2.0+ with CUDA support (recommended)

### **Development Setup**

```bash
# Clone repository
git clone https://github.com/your-username/hpfracc.git
cd hpfracc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
python -c "import hpfracc; print('Installation successful!')"
```

### **Development Dependencies**

```bash
# Core development tools
pip install pytest pytest-cov pytest-mock
pip install black isort flake8 mypy
pip install pre-commit

# Documentation
pip install sphinx sphinx-rtd-theme
pip install myst-parser

# Additional tools
pip install jupyter notebook
pip install ipython
```

## 🔧 **Development Workflow**

### **Code Style**

We use strict code formatting and linting:

```bash
# Format code with black
black hpfracc/ tests/ examples/

# Sort imports with isort
isort hpfracc/ tests/ examples/

# Lint with flake8
flake8 hpfracc/ tests/ examples/

# Type checking with mypy
mypy hpfracc/
```

### **Pre-commit Hooks**

Set up pre-commit hooks for automatic code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### **Git Workflow**

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add new fractional derivative method"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub
   ```

### **Commit Message Format**

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## 🧪 **Testing**

### **Running Tests**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=hpfracc --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run specific test class
pytest tests/test_core.py::TestFractionalDerivatives -v

# Run specific test method
pytest tests/test_core.py::TestFractionalDerivatives::test_riemann_liouville -v
```

### **Test Structure**

```python
# Example test structure
import pytest
import torch
from hpfracc.core import fractional_derivative

class TestFractionalDerivatives:
    """Test suite for fractional derivatives"""
    
    def test_riemann_liouville_basic(self):
        """Test basic Riemann-Liouville derivative"""
        x = torch.randn(100, 50)
        result = fractional_derivative(x, alpha=0.5, method="RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
    
    def test_invalid_fractional_order(self):
        """Test error handling for invalid fractional orders"""
        x = torch.randn(100, 50)
        
        with pytest.raises(ValueError, match="Fractional order must be positive"):
            fractional_derivative(x, alpha=-0.5, method="RL")
    
    @pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9, 1.5])
    def test_fractional_orders(self, alpha):
        """Test multiple fractional orders"""
        x = torch.randn(100, 50)
        result = fractional_derivative(x, alpha=alpha, method="RL")
        
        assert result.shape == x.shape
        assert torch.isfinite(result).all()
```

### **Test Coverage**

Maintain high test coverage:

```bash
# Generate coverage report
pytest tests/ --cov=hpfracc --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
start htmlcov/index.html  # On Windows
```

**Target Coverage**: >90% for all modules

## 📊 **Benchmarking**

### **Performance Testing**

```bash
# Run ML performance benchmarks
python -m hpfracc.benchmarks.ml_performance_benchmark

# Run specific benchmarks
python examples/performance_benchmarks.py
```

### **Benchmark Structure**

```python
# Example benchmark
import time
import torch
from hpfracc.core import fractional_derivative

def benchmark_fractional_derivatives():
    """Benchmark different fractional derivative methods"""
    
    # Test data
    x = torch.randn(1000, 500)
    methods = ["RL", "Caputo", "GL"]
    alphas = [0.1, 0.5, 0.9]
    
    results = {}
    
    for method in methods:
        for alpha in alphas:
            # Warmup
            for _ in range(3):
                _ = fractional_derivative(x, alpha=alpha, method=method)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                result = fractional_derivative(x, alpha=alpha, method=method)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            results[f"{method}_α={alpha}"] = avg_time
            
            print(f"{method} α={alpha}: {avg_time:.4f}s")
    
    return results
```

## 📚 **Documentation**

### **Building Documentation**

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html  # On macOS
start _build/html/index.html  # On Windows
```

### **Documentation Standards**

- **API Reference**: Complete parameter descriptions
- **User Guide**: Step-by-step instructions
- **Examples**: Working code samples
- **Model Theory**: Mathematical foundations
- **Inline Comments**: Clear code documentation

### **Docstring Format**

```python
def fractional_derivative(x: torch.Tensor, alpha: float, method: str = "RL") -> torch.Tensor:
    """
    Compute fractional derivative of input tensor.
    
    Args:
        x: Input tensor of shape (..., seq_len)
        alpha: Fractional order (0 < α < 2)
        method: Derivative method ("RL", "Caputo", "GL", "Weyl", "Marchaud", "Hadamard")
    
    Returns:
        Fractional derivative tensor of same shape as input
        
    Raises:
        ValueError: If alpha is outside valid range for method
        RuntimeError: If method is not implemented
        
    Example:
        >>> x = torch.randn(100, 50)
        >>> result = fractional_derivative(x, alpha=0.5, method="RL")
        >>> print(result.shape)
        torch.Size([100, 50])
    """
```

## 🔍 **Code Quality**

### **Static Analysis**

```bash
# Type checking
mypy hpfracc/

# Linting
flake8 hpfracc/ tests/ examples/

# Import sorting
isort --check-only hpfracc/ tests/ examples/

# Code formatting
black --check hpfracc/ tests/ examples/
```

### **Code Review Checklist**

- [ ] **Functionality**: Code works as intended
- [ ] **Testing**: Adequate test coverage
- [ ] **Documentation**: Clear docstrings and comments
- [ ] **Style**: Follows project style guidelines
- [ ] **Performance**: No performance regressions
- [ ] **Security**: No security vulnerabilities
- [ ] **Error Handling**: Proper error handling and validation

## 🚀 **Performance Optimization**

### **Profiling**

```python
import torch.profiler as profiler

def profile_function():
    """Profile function performance"""
    
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        # Your code here
        result = fractional_derivative(x, alpha=0.5, method="RL")
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    # Save results
    prof.export_chrome_trace("trace.json")
```

### **Memory Profiling**

```python
import torch.cuda

def monitor_memory():
    """Monitor GPU memory usage"""
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Cached: {cached:.2f} GB")
        
        # Memory summary
        print(torch.cuda.memory_summary())
```

## 🔧 **Debugging**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Reinstall in development mode
   pip install -e .
   ```

2. **CUDA Issues**
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Check CUDA version
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Memory Issues**
   ```python
   # Clear GPU cache
   torch.cuda.empty_cache()
   
   # Check memory usage
   print(torch.cuda.memory_summary())
   ```

### **Debug Tools**

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Use IPython for interactive debugging
import IPython
IPython.embed()

# Print tensor shapes
def debug_shapes(*tensors, names=None):
    """Debug tensor shapes"""
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]
    
    for name, tensor in zip(names, tensors):
        print(f"{name}: {tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
```

## 📦 **Release Process**

### **Version Management**

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### **Release Checklist**

- [ ] **Tests Pass**: All tests pass
- [ ] **Documentation**: Documentation is up to date
- [ ] **Benchmarks**: Performance benchmarks pass
- [ ] **Version Update**: Update version in `pyproject.toml`
- [ ] **Changelog**: Update CHANGELOG.md
- [ ] **Tag Release**: Create git tag
- [ ] **PyPI Upload**: Upload to PyPI (if applicable)

### **Creating Release**

```bash
# Update version
# Edit pyproject.toml

# Commit version change
git add pyproject.toml
git commit -m "chore: bump version to 1.0.0"

# Create tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag
git push origin v1.0.0

# Create GitHub release
# Go to GitHub releases page and create release from tag
```

## 🤝 **Contributing Guidelines**

### **Pull Request Process**

1. **Fork** the repository
2. **Create** feature branch
3. **Implement** changes with tests
4. **Update** documentation
5. **Run** tests and benchmarks
6. **Submit** pull request

### **Review Process**

- **Code Review**: At least one maintainer must approve
- **CI Checks**: All CI checks must pass
- **Test Coverage**: Maintain or improve test coverage
- **Documentation**: Update relevant documentation

### **Issue Reporting**

When reporting issues:

1. **Use Issue Template**: Fill out the issue template
2. **Provide Details**: Include error messages, stack traces
3. **Reproduce**: Provide minimal reproduction code
4. **Environment**: Specify OS, Python version, dependencies

## 📞 **Getting Help**

### **Development Resources**

- **Code**: [GitHub Repository](https://github.com/your-username/hpfracc)
- **Issues**: [GitHub Issues](https://github.com/your-username/hpfracc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/hpfracc/discussions)
- **Documentation**: [docs/](docs/) directory

### **Contact**

- **Email**: [d.r.chin@pgr.reading.ac.uk](mailto:d.r.chin@pgr.reading.ac.uk)
- **GitHub**: [@your-username](https://github.com/your-username)

## 🎯 **Development Goals**

### **Short Term (Next 3 months)**

- [ ] **Performance Optimization**: Further optimize adjoint methods
- [ ] **GPU Support**: Improve CUDA implementations
- [ ] **Testing**: Increase test coverage to >95%
- [ ] **Documentation**: Complete API documentation

### **Medium Term (3-6 months)**

- [ ] **Distributed Training**: Multi-GPU and multi-node support
- [ ] **AutoML Integration**: Automated hyperparameter optimization
- [ ] **Cloud Deployment**: AWS, GCP, Azure integration
- [ ] **Real-time Inference**: Streaming computation capabilities

### **Long Term (6+ months)**

- [ ] **Research Integration**: Novel fractional derivative methods
- [ ] **Industry Adoption**: Production deployments in industry
- [ ] **Community Growth**: Active contributor community
- [ ] **Academic Recognition**: Publications and citations

---

**Happy Coding! 🚀**

*This development guide is maintained by the HPFRACC development team.*
