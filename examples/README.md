# Fractional Calculus Library - Examples

This directory contains comprehensive examples demonstrating the capabilities of the fractional calculus library.

## 📁 Directory Structure

```
examples/
├── basic_usage/           # Getting started examples
├── jax_examples/         # JAX optimization examples
├── parallel_examples/    # Parallel computing examples
├── advanced_applications/ # Advanced PDE solver examples
└── README.md            # This file
```

## 🚀 Quick Start

### Basic Usage Examples

Start with the basic usage examples to understand the core functionality:

```bash
cd examples/basic_usage
python getting_started.py
```

This will demonstrate:
- Basic fractional derivative computations
- Fractional integral calculations
- Comparison with analytical solutions
- Error analysis and convergence studies

### JAX Optimization Examples

Explore GPU acceleration and automatic differentiation:

```bash
cd examples/jax_examples
python jax_optimization_demo.py
```

Features demonstrated:
- GPU acceleration with JAX
- Automatic differentiation (gradients, Jacobians, Hessians)
- Vectorization over multiple parameters
- Performance benchmarking
- FFT-based methods

### Parallel Computing Examples

Learn about parallel processing capabilities:

```bash
cd examples/parallel_examples
python parallel_computing_demo.py
```

Features demonstrated:
- Joblib backend (recommended)
- Multiprocessing and threading alternatives
- Load balancing strategies
- Memory optimization
- System information analysis

### Advanced Applications

Explore advanced PDE solving capabilities:

```bash
cd examples/advanced_applications
python fractional_pde_solver.py
```

Features demonstrated:
- Fractional diffusion equation solving
- Fractional wave equation solving
- L1/L2 scheme comparisons
- Predictor-corrector methods
- 3D visualization of solutions

## 📊 Example Outputs

Each example generates:
- **Interactive plots** showing results
- **Saved images** in the respective directory
- **Console output** with performance metrics
- **Error analysis** and convergence studies

## 🎯 Key Features Demonstrated

### 1. Basic Usage (`basic_usage/`)
- **Fractional Derivatives**: Caputo, Riemann-Liouville, Grünwald-Letnikov
- **Fractional Integrals**: Direct computation and validation
- **Analytical Comparisons**: Numerical vs analytical solutions
- **Convergence Analysis**: Error rates and grid refinement studies

### 2. JAX Optimization (`jax_examples/`)
- **GPU Acceleration**: Leveraging GPU for faster computations
- **Automatic Differentiation**: Gradients, Jacobians, and Hessians
- **Vectorization**: Processing multiple parameters simultaneously
- **Performance Monitoring**: Real-time performance analysis
- **FFT Methods**: Spectral and convolution-based approaches

### 3. Parallel Computing (`parallel_examples/`)
- **Joblib Backend**: Optimal parallel processing (default)
- **Load Balancing**: Static, dynamic, and adaptive strategies
- **Memory Optimization**: Efficient memory usage patterns
- **System Analysis**: Hardware utilization and recommendations
- **Scaling Analysis**: Performance with different worker counts

### 4. Advanced Applications (`advanced_applications/`)
- **PDE Solvers**: Fractional partial differential equations
- **Numerical Schemes**: L1, L2, and predictor-corrector methods
- **3D Visualization**: Surface plots and contour maps
- **Stability Analysis**: Numerical stability assessment
- **Convergence Studies**: Method comparison and validation

## 🔧 Requirements

### Core Dependencies
```bash
pip install numpy scipy matplotlib
```

### Optional Dependencies
```bash
# For JAX examples
pip install jax jaxlib

# For advanced visualization
pip install mpl_toolkits

# For parallel computing (usually included)
pip install joblib
```

## 📈 Performance Tips

### 1. Basic Usage
- Start with small grid sizes (N=100) for testing
- Use analytical solutions for validation
- Monitor convergence rates for accuracy

### 2. JAX Optimization
- Ensure GPU is available for best performance
- Use JIT compilation for repeated computations
- Leverage vectorization for multiple parameters

### 3. Parallel Computing
- Joblib is the recommended backend
- Adjust worker count based on your CPU cores
- Monitor memory usage for large datasets

### 4. Advanced Applications
- Use appropriate grid sizes for your problem
- Consider stability requirements
- Validate results with known solutions

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd /path/to/fc_library
   python -m examples.basic_usage.getting_started
   ```

2. **JAX GPU Issues**
   ```bash
   # Check JAX installation
   python -c "import jax; print(jax.devices())"
   ```

3. **Memory Issues**
   ```bash
   # Reduce grid sizes for large problems
   # Use memory-efficient methods
   ```

4. **Performance Issues**
   ```bash
   # Check CPU utilization
   # Monitor memory usage
   # Use appropriate backends
   ```

### Getting Help

1. **Check the documentation** in the main `docs/` directory
2. **Review error messages** for specific issues
3. **Start with basic examples** before advanced features
4. **Monitor system resources** during execution

## 📚 Learning Path

### Beginner
1. Start with `basic_usage/getting_started.py`
2. Understand fractional derivatives and integrals
3. Learn about different numerical methods
4. Practice with analytical comparisons

### Intermediate
1. Explore `jax_examples/jax_optimization_demo.py`
2. Learn GPU acceleration techniques
3. Understand automatic differentiation
4. Master vectorization strategies

### Advanced
1. Study `parallel_examples/parallel_computing_demo.py`
2. Optimize for your specific hardware
3. Implement custom parallel strategies
4. Analyze performance bottlenecks

### Expert
1. Dive into `advanced_applications/fractional_pde_solver.py`
2. Implement custom PDE solvers
3. Develop new numerical schemes
4. Contribute to the library

## 🔄 Customization

### Modifying Examples

1. **Change Parameters**: Modify alpha values, grid sizes, etc.
2. **Add Functions**: Implement your own test functions
3. **Custom Visualization**: Create specific plots for your needs
4. **Performance Tuning**: Optimize for your use case

### Example Customization

```python
# Custom test function
def my_function(t):
    return np.sin(2 * np.pi * t) * np.exp(-t)

# Custom parameters
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
grid_sizes = [50, 100, 200, 500, 1000]

# Run with custom parameters
# ... modify example code accordingly
```

## 📊 Benchmarking

For comprehensive performance analysis, see the `benchmarks/` directory:

```bash
cd ../benchmarks
python performance_tests.py
python accuracy_comparisons.py
python scaling_analysis.py
```

## 🤝 Contributing

When adding new examples:

1. **Follow the existing structure**
2. **Include comprehensive documentation**
3. **Add error handling**
4. **Provide performance metrics**
5. **Include visualization options**
6. **Test on different systems**

## 📄 License

This examples directory is part of the fractional calculus library and follows the same license terms.

---

**Happy Computing! 🚀**

For more information, see the main library documentation and the `benchmarks/` directory for performance analysis.
