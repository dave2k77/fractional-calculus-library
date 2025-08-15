# Installation Guide - Fractional Calculus Library

This guide provides detailed instructions for installing and setting up the Fractional Calculus Library on different platforms and environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Development Installation](#development-installation)
5. [GPU Support Setup](#gpu-support-setup)
6. [Troubleshooting](#troubleshooting)
7. [Verification](#verification)

---

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4 GB RAM (8 GB recommended)
- **Storage**: 2 GB free space

### Recommended Requirements
- **Python**: 3.9 or higher
- **Memory**: 16 GB RAM
- **Storage**: 5 GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)

### Dependencies
The library automatically installs these core dependencies:
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **JAX**: GPU acceleration and automatic differentiation
- **Numba**: JIT compilation
- **Matplotlib**: Plotting and visualization
- **Joblib**: Parallel computing

---

## Quick Installation

### Option 1: Using pip (Recommended)

```bash
# Install from PyPI (when available)
pip install fractional-calculus-library

# Or install from GitHub
pip install git+https://github.com/dave2k77/fractional_calculus_library.git
```

### Option 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/dave2k77/fractional_calculus_library.git
cd fractional_calculus_library

# Install in development mode
pip install -e .
```

---

## Detailed Installation Steps

### Step 1: Prepare Your Environment

#### Windows
```powershell
# Create virtual environment
python -m venv fc_env
.\fc_env\Scripts\Activate.ps1

# Update pip
python -m pip install --upgrade pip
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv fc_env
source fc_env/bin/activate

# Update pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 3: Install the Library

```bash
# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Step 4: Verify Installation

```python
# Test basic import
python -c "import src; print('Installation successful!')"

# Run tests
python scripts/run_tests.py --type fast
```

---

## Development Installation

For contributors and developers who want to work on the library:

### Step 1: Clone Repository

```bash
git clone https://github.com/dave2k77/fractional_calculus_library.git
cd fractional_calculus_library
```

### Step 2: Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Step 3: Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

### Step 4: Verify Development Setup

```bash
# Run all tests
python scripts/run_tests.py

# Run code quality checks
python scripts/run_tests.py --no-coverage --reports
```

---

## GPU Support Setup

### NVIDIA GPU with CUDA

#### Step 1: Install CUDA Toolkit
1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Follow installation instructions for your platform
3. Verify installation: `nvcc --version`

#### Step 2: Install JAX with GPU Support

```bash
# For CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 12.1
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Step 3: Verify GPU Support

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"GPU available: {jax.devices('gpu')}")
```

### Apple Silicon (M1/M2) GPU

```bash
# Install JAX for Apple Silicon
pip install --upgrade "jax[metal]" -f https://storage.googleapis.com/jax-releases/jax_metal_releases.html
```

### CPU-Only Installation

If you don't need GPU support:

```bash
# Install JAX for CPU only
pip install --upgrade jax jaxlib
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors
**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the correct directory
cd fractional_calculus_library

# Install in development mode
pip install -e .
```

#### Issue 2: JAX Installation Problems
**Error**: `ImportError: cannot import name 'jax'`

**Solution**:
```bash
# Uninstall existing JAX
pip uninstall jax jaxlib

# Reinstall JAX
pip install --upgrade jax jaxlib
```

#### Issue 3: GPU Not Detected
**Error**: `No GPU devices found`

**Solution**:
1. Check CUDA installation: `nvcc --version`
2. Check GPU drivers: `nvidia-smi`
3. Reinstall JAX with correct CUDA version
4. Restart your Python environment

#### Issue 4: Memory Issues
**Error**: `OutOfMemoryError`

**Solution**:
```python
# Reduce batch size or use CPU
import jax
jax.config.update('jax_platform_name', 'cpu')
```

#### Issue 5: Windows-specific Issues
**Error**: Compilation errors on Windows

**Solution**:
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Or use conda instead of pip
conda install -c conda-forge jax jaxlib
```

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- Install Visual Studio Build Tools for compilation
- Consider using WSL2 for better compatibility

#### macOS
- Use Homebrew for system dependencies
- For Apple Silicon, use Rosetta if needed
- Install Xcode Command Line Tools

#### Linux
- Install system dependencies: `sudo apt-get install build-essential`
- Use virtual environment for isolation
- Consider using conda for complex dependencies

---

## Verification

### Basic Functionality Test

```python
import numpy as np
from src.algorithms.caputo import CaputoDerivative

# Test basic functionality
alpha = 0.5
caputo = CaputoDerivative(alpha)

t = np.linspace(0.1, 2.0, 50)
f = t
h = t[1] - t[0]

result = caputo.compute(f, t, h)
print(f"Test successful! Result shape: {result.shape}")
```

### Performance Test

```python
import time
from src.algorithms.caputo import CaputoDerivative

# Performance test
alpha = 0.5
caputo = CaputoDerivative(alpha)

# Large dataset
t = np.linspace(0.1, 10.0, 10000)
f = np.sin(t)
h = t[1] - t[0]

start_time = time.time()
result = caputo.compute(f, t, h)
end_time = time.time()

print(f"Computation time: {end_time - start_time:.3f} seconds")
print(f"Performance test passed!")
```

### GPU Test (if available)

```python
import jax
import jax.numpy as jnp

# Test GPU functionality
if jax.devices('gpu'):
    print("GPU detected!")
    
    # Test JAX on GPU
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sin(x)
    print(f"JAX GPU test: {y}")
else:
    print("No GPU detected, using CPU")
```

### Full Test Suite

```bash
# Run comprehensive tests
python scripts/run_tests.py

# Check coverage
python scripts/run_tests.py --coverage

# Run benchmarks
python benchmarks/performance_tests.py
```

---

## Next Steps

After successful installation:

1. **Read the Documentation**: Start with the [User Guide](user_guide.md)
2. **Try Examples**: Run examples in the `examples/` directory
3. **Explore API**: Check the [API Reference](api_reference/)
4. **Join Community**: Contribute or ask questions on GitHub

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Search existing [GitHub issues](https://github.com/dave2k77/fractional_calculus_library/issues)
3. Create a new issue with detailed information
4. Include your system information and error messages

---

**Note**: This installation guide is regularly updated. For the latest version, check the [GitHub repository](https://github.com/dave2k77/fractional_calculus_library).
