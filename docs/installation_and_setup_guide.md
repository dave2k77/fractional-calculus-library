# Installation and Setup Guide - Fractional Calculus Library

This guide provides comprehensive instructions for installing and setting up the Fractional Calculus Library on different platforms and configurations.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start Installation](#quick-start-installation)
3. [Detailed Installation Methods](#detailed-installation-methods)
4. [GPU Setup](#gpu-setup)
5. [Development Setup](#development-setup)
6. [Dependencies Management](#dependencies-management)
7. [Troubleshooting](#troubleshooting)
8. [Verification](#verification)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4 GB RAM
- **Storage**: 2 GB free space

### Recommended Requirements
- **Python**: 3.9 or higher
- **Memory**: 8 GB RAM or more
- **GPU**: NVIDIA GPU with CUDA support (for GPU acceleration)
- **Storage**: 5 GB free space

### Supported Platforms
- **Windows**: 10, 11
- **macOS**: 10.14 (Mojave) or later
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 9+

---

## Quick Start Installation

### Method 1: Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install fractional-calculus-library

# Or install from GitHub
pip install git+https://github.com/dave2k77/fractional-calculus-library.git
```

### Method 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library

# Install in development mode
pip install -e .
```

### Method 3: Using conda (Alternative)

```bash
# Create new conda environment
conda create -n fractional-calc python=3.9
conda activate fractional-calc

# Install dependencies
conda install numpy scipy matplotlib jax numba

# Install the library
pip install -e .
```

---

## Detailed Installation Methods

### Windows Installation

#### Prerequisites
1. **Python**: Download and install Python 3.8+ from [python.org](https://python.org)
2. **Git**: Download and install Git from [git-scm.com](https://git-scm.com)
3. **Visual Studio Build Tools** (for some dependencies):
   ```bash
   # Install via pip
   pip install --upgrade setuptools wheel
   ```

#### Installation Steps
```powershell
# Open PowerShell as Administrator
# Clone repository
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install library
pip install -e .
```

### macOS Installation

#### Prerequisites
1. **Homebrew** (recommended):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Python**:
   ```bash
   brew install python@3.9
   ```

#### Installation Steps
```bash
# Clone repository
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install library
pip install -e .
```

### Linux Installation

#### Ubuntu/Debian
```bash
# Update system
sudo apt update
sudo apt upgrade

# Install Python and development tools
sudo apt install python3 python3-pip python3-venv git build-essential

# Clone repository
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install library
pip install -e .
```

#### CentOS/RHEL
```bash
# Install Python and development tools
sudo yum install python3 python3-pip git gcc gcc-c++

# Follow same steps as Ubuntu
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## GPU Setup

### NVIDIA GPU Setup

#### Prerequisites
1. **NVIDIA Driver**: Install latest NVIDIA driver
2. **CUDA Toolkit**: Install CUDA 11.0 or later
3. **cuDNN**: Install cuDNN compatible with your CUDA version

#### Installation Steps

##### Windows
```powershell
# Install CUDA toolkit from NVIDIA website
# Then install JAX with CUDA support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

##### Linux
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda

# Install JAX with CUDA support
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

##### macOS
```bash
# Note: CUDA is not supported on macOS
# Use CPU-only JAX or Metal Performance Shaders (MPS)
pip install --upgrade jax
```

#### Verification
```python
import jax
import jax.numpy as jnp

# Check available devices
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test GPU computation
if len(jax.devices()) > 1:
    x = jnp.array([1., 2., 3.])
    y = jnp.sin(x)
    print(f"GPU computation successful: {y}")
else:
    print("GPU not available, using CPU")
```

### AMD GPU Setup (Experimental)

```bash
# Install ROCm (AMD's CUDA alternative)
# Follow AMD's official documentation for your system

# Install JAX with ROCm support
pip install --upgrade "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
```

---

## Development Setup

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/dave2k77/fractional-calculus-library.git
cd fractional-calculus-library

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install development dependencies
pip install -r requirements-dev.txt

# Install library in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `requirements-dev.txt` includes:
- **Testing**: pytest, pytest-cov, pytest-xdist
- **Code Quality**: flake8, black, mypy, isort
- **Documentation**: sphinx, sphinx-rtd-theme
- **Development Tools**: pre-commit, jupyter

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test categories
pytest tests/test_algorithms/
pytest tests/test_optimisation/
pytest tests/integration_tests/

# Run performance tests
pytest tests/ -m "performance"
```

### Code Quality Checks

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Type checking
mypy src

# Linting
flake8 src tests

# Run all checks
pre-commit run --all-files
```

---

## Dependencies Management

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Numerical computing |
| scipy | >=1.7.0 | Scientific computing |
| matplotlib | >=3.3.0 | Plotting and visualization |
| jax | >=0.3.0 | GPU acceleration and autodiff |
| numba | >=0.56.0 | JIT compilation |
| concurrent.futures | built-in | Parallel processing |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| jax[cuda] | >=0.3.0 | GPU acceleration (NVIDIA) |
| jax[rocm] | >=0.3.0 | GPU acceleration (AMD) |
| psutil | >=5.8.0 | System monitoring |
| seaborn | >=0.11.0 | Enhanced plotting |

### Installing Specific Dependencies

```bash
# Install with GPU support
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install with all optional dependencies
pip install -e .[all]

# Install specific optional dependencies
pip install -e .[gpu,plotting,monitoring]
```

### Environment Management

#### Using conda
```bash
# Create environment with specific Python version
conda create -n fractional-calc python=3.9

# Install packages
conda install numpy scipy matplotlib
conda install -c conda-forge jax numba

# Install the library
pip install -e .
```

#### Using pipenv
```bash
# Install pipenv
pip install pipenv

# Create environment
pipenv install

# Install development dependencies
pipenv install --dev

# Activate environment
pipenv shell
```

#### Using poetry
```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create project
poetry init

# Install dependencies
poetry install

# Activate environment
poetry shell
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. JAX Installation Issues

**Problem**: JAX installation fails

**Solution**:
```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install JAX CPU version first
pip install jax jaxlib

# Then install GPU version if needed
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 3. CUDA Issues

**Problem**: CUDA not detected

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Check JAX CUDA support
python -c "import jax; print(jax.devices())"

# Reinstall JAX with correct CUDA version
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 4. Memory Issues

**Problem**: Out of memory errors

**Solution**:
```python
# Reduce batch size or use memory-efficient methods
import jax
jax.config.update('jax_platform_name', 'cpu')  # Force CPU if needed

# Use chunked processing for large datasets
from src.algorithms.advanced_methods import MarchaudDerivative
marchaud = MarchaudDerivative(alpha=0.5, memory_optimized=True)
```

#### 5. Performance Issues

**Problem**: Slow computation

**Solution**:
```python
# Use optimized methods
from src.algorithms.advanced_optimized_methods import optimized_weyl_derivative

# Enable JIT compilation
import jax
jax.config.update('jax_enable_x64', True)

# Use GPU if available
import jax.numpy as jnp
x = jnp.array(your_data)  # Move to GPU
```

### Platform-Specific Issues

#### Windows
- **Visual Studio Build Tools**: Install for compiling extensions
- **Path Issues**: Use PowerShell or Command Prompt with proper PATH
- **Permission Issues**: Run as Administrator if needed

#### macOS
- **Homebrew**: Install for easier package management
- **Xcode**: Install command line tools: `xcode-select --install`
- **M1/M2 Macs**: Use Rosetta 2 for x86 packages if needed

#### Linux
- **Package Dependencies**: Install development packages
- **Permission Issues**: Use `sudo` for system-wide installation
- **Library Path**: Set `LD_LIBRARY_PATH` for custom CUDA installations

### Getting Help

1. **Check Documentation**: Review the [API Reference](api_reference/) and [Examples](examples/)
2. **Search Issues**: Look for similar issues on [GitHub Issues](https://github.com/dave2k77/fractional-calculus-library/issues)
3. **Create Issue**: Report bugs with detailed information
4. **Community**: Join discussions on GitHub Discussions

---

## Verification

### Basic Verification

```python
# Test basic functionality
import numpy as np
from src.algorithms.advanced_methods import WeylDerivative

# Create test data
alpha = 0.5
x = np.linspace(0, 2*np.pi, 100)
f = lambda x: np.sin(x)

# Compute derivative
weyl = WeylDerivative(alpha)
result = weyl.compute(f, x, h=0.1)

print(f"Weyl derivative computed successfully!")
print(f"Result shape: {result.shape}")
print(f"Sample values: {result[:5]}")
```

### Advanced Verification

```python
# Test all advanced methods
from src.algorithms.advanced_methods import (
    WeylDerivative, MarchaudDerivative, HadamardDerivative,
    ReizFellerDerivative, AdomianDecomposition
)

# Test parameters
alpha = 0.5
x = np.linspace(0, 5, 100)
f = lambda x: np.sin(x) * np.exp(-x/3)

# Test each method
methods = [
    WeylDerivative(alpha),
    MarchaudDerivative(alpha),
    HadamardDerivative(alpha),
    ReizFellerDerivative(alpha)
]

for i, method in enumerate(methods):
    if isinstance(method, HadamardDerivative):
        x_test = np.linspace(1, 5, 100)  # Positive domain
    else:
        x_test = x
    
    result = method.compute(f, x_test, h=0.05)
    print(f"Method {i+1} completed: {result.shape}")

print("All advanced methods working correctly!")
```

### Performance Verification

```python
# Test optimized methods
import time
from src.algorithms.advanced_optimized_methods import optimized_weyl_derivative

# Performance test
alpha = 0.5
x = np.linspace(0, 5, 1000)
f = lambda x: np.sin(x) * np.exp(-x/3)

start_time = time.time()
result = optimized_weyl_derivative(f, x, alpha, h=0.005)
computation_time = time.time() - start_time

print(f"Optimized computation completed in {computation_time:.4f} seconds")
print(f"Result shape: {result.shape}")
```

### GPU Verification

```python
# Test GPU functionality
import jax
import jax.numpy as jnp

print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test GPU computation
x = jnp.array([1., 2., 3., 4., 5.])
y = jnp.sin(x)
print(f"JAX computation successful: {y}")

if len(jax.devices()) > 1:
    print("GPU acceleration available!")
else:
    print("Using CPU computation")
```

### Complete Verification Script

```python
#!/usr/bin/env python3
"""
Complete verification script for the Fractional Calculus Library
"""

def run_verification():
    """Run complete verification suite"""
    print("=" * 60)
    print("FRACTIONAL CALCULUS LIBRARY VERIFICATION")
    print("=" * 60)
    
    try:
        # Test imports
        print("1. Testing imports...")
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        
        from src.algorithms.advanced_methods import (
            WeylDerivative, MarchaudDerivative, HadamardDerivative,
            ReizFellerDerivative, AdomianDecomposition
        )
        
        from src.algorithms.advanced_optimized_methods import (
            optimized_weyl_derivative, optimized_marchaud_derivative
        )
        
        print("   ✓ All imports successful")
        
        # Test basic functionality
        print("2. Testing basic functionality...")
        alpha = 0.5
        x = np.linspace(0, 2*np.pi, 50)
        f = lambda x: np.sin(x)
        
        weyl = WeylDerivative(alpha)
        result = weyl.compute(f, x, h=0.1)
        
        print(f"   ✓ Basic computation successful: {result.shape}")
        
        # Test optimized methods
        print("3. Testing optimized methods...")
        result_opt = optimized_weyl_derivative(f, x, alpha, h=0.1)
        print(f"   ✓ Optimized computation successful: {result_opt.shape}")
        
        # Test GPU if available
        print("4. Testing GPU support...")
        try:
            import jax
            print(f"   ✓ JAX available: {jax.default_backend()}")
            print(f"   ✓ Devices: {jax.devices()}")
        except ImportError:
            print("   ⚠ JAX not available (optional)")
        
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe Fractional Calculus Library is ready to use!")
        print("\nNext steps:")
        print("1. Check the examples in docs/examples/")
        print("2. Run the real-world applications guide")
        print("3. Explore the API reference")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
```

---

## Summary

This installation guide covers:

1. **System Requirements**: Minimum and recommended specifications
2. **Installation Methods**: Multiple approaches for different platforms
3. **GPU Setup**: NVIDIA and AMD GPU configuration
4. **Development Setup**: Complete development environment
5. **Dependencies**: Core and optional package management
6. **Troubleshooting**: Common issues and solutions
7. **Verification**: Testing installation and functionality

For additional help, see:
- [User Guide](user_guide.md)
- [API Reference](api_reference/)
- [Examples](examples/)
- [GitHub Issues](https://github.com/dave2k77/fractional-calculus-library/issues)
