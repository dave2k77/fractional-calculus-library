# HPFracc Dedicated Environment Setup

## Environment Overview
- **Environment Name**: `hpfracc-env`
- **Python Version**: 3.11.13
- **Package Manager**: Conda
- **Installation Type**: Development mode (editable)

## Quick Start
```bash
# Activate the environment
source activate_env.sh

# Or manually activate
conda activate hpfracc-env

# Deactivate when done
conda deactivate
```

## Core Dependencies
### Scientific Computing
- **NumPy**: 2.2.6 - Numerical computing
- **SciPy**: 1.16.1 - Scientific algorithms
- **Matplotlib**: 3.10.6 - Plotting and visualization
- **Pandas**: 2.3.2 - Data manipulation
- **SymPy**: 1.14.0 - Symbolic mathematics

### Machine Learning & Deep Learning
- **PyTorch**: 2.8.0+cu128 - Deep learning framework with CUDA support
- **JAX**: 0.7.1 - High-performance ML framework
- **Flax**: 0.11.2 - Neural network library for JAX
- **Optax**: 0.2.5 - Optimization library for JAX
- **Torch Geometric**: 2.6.1 - Geometric deep learning

### GPU Acceleration
- **CuPy**: 13.6.0 - CUDA-accelerated NumPy
- **JAXlib**: 0.7.1 - JAX CUDA support
- **NVIDIA CUDA**: 12.x runtime support

### Development Tools
- **Pytest**: 8.4.1 - Testing framework
- **Black**: 25.1.0 - Code formatting
- **Flake8**: 7.3.0 - Code linting
- **MyPy**: 1.17.1 - Type checking
- **Pre-commit**: 4.3.0 - Git hooks

### Documentation
- **Sphinx**: 8.2.3 - Documentation generator
- **Sphinx RTD Theme**: 3.0.2 - ReadTheDocs theme

## Environment Benefits
1. **Isolation**: Clean separation from system Python
2. **Reproducibility**: Exact dependency versions
3. **GPU Support**: Full CUDA acceleration
4. **Development Ready**: All tools for development workflow
5. **Performance**: Optimized for fractional calculus computations

## Testing the Environment
```bash
# Activate environment
conda activate hpfracc-env

# Test basic functionality
python -c "import hpfracc; print(hpfracc.__version__)"

# Test GPU support
python -c "import hpfracc.algorithms.gpu_optimized_methods as gpu_mod; print(f'CuPy: {gpu_mod.CUPY_AVAILABLE}, JAX: {gpu_mod.JAX_AVAILABLE}')"

# Run tests
python -m pytest tests/ -v

# Run examples
python examples/basic_usage/getting_started.py
```

## Environment Management
```bash
# List all environments
conda env list

# Export environment
conda env export > environment.yml

# Recreate environment from file
conda env create -f environment.yml

# Remove environment (if needed)
conda env remove -n hpfracc-env
```

## Troubleshooting
- **CUDA Issues**: Ensure NVIDIA drivers are up to date
- **Memory Issues**: Monitor GPU memory usage with `nvidia-smi`
- **Import Errors**: Verify environment is activated with `conda info --envs`

## Performance Notes
- **GPU Memory**: RTX 3050 has 4GB VRAM - suitable for medium-scale computations
- **CPU**: Leverages all available CPU cores for parallel processing
- **Memory**: Efficient memory management for large-scale fractional calculus operations
