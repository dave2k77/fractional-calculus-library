# HPFRACC v2.0.0 - Production Ready Examples

This directory contains comprehensive examples demonstrating the capabilities of the HPFRACC fractional calculus library.

## 🎯 **Production Ready Status**

✅ **Integration Tests**: 188/188 passed (100% success)  
✅ **Performance Benchmarks**: 151/151 passed (100% success)  
✅ **Research Validated**: Complete workflows for computational physics and biophysics  
✅ **GPU Optimized**: Accelerated computation with CUDA support  
✅ **ML Integration**: Fractional neural networks with autograd support  

**Author**: Davian R. Chin, Department of Biomedical Engineering, University of Reading  
**Email**: d.r.chin@pgr.reading.ac.uk

## 📁 Directory Structure

```
examples/
├── basic_usage/           # Getting started examples
├── jax_examples/         # JAX optimization examples
├── parallel_examples/    # Parallel computing examples
├── advanced_applications/ # Advanced PDE solver examples
├── benchmarks/           # Performance benchmarking examples
├── ml_examples/          # Machine learning and neural network examples
├── physics_examples/     # Physics and scientific computing examples
├── real_world_applications/ # Real-world application examples
├── scientific_tutorials/ # Scientific tutorials and guides
├── advanced_methods_demo.py      # Advanced methods demonstration
├── analytics_demo.py            # Analytics and monitoring demo
├── fractional_operators_demo.py # Fractional operators demo
├── minimal_fractional_demo.py   # Minimal fractional calculus demo
├── multi_backend_demo.py        # Multi-backend comparison demo
├── special_methods_examples.py  # Special methods examples
├── getting_started_production.py # NEW: Production-ready basic examples
├── research_applications_demo.py # NEW: Complete research workflows
├── PRODUCTION_READY_EXAMPLES.md  # NEW: Examples update plan
└── README.md            # This file
```

## 🚀 Quick Start

### **NEW: Production Ready Examples**

Start with the updated production-ready examples:

```bash
# Updated production-ready basic examples
cd examples/basic_usage
python getting_started_production.py
```

This demonstrates:
- ✅ Standardized `order` parameter (not `alpha`)
- ✅ Production-ready components with 100% test success
- ✅ GPU optimization and performance benchmarking
- ✅ Complete research workflows for physics and biophysics

### **Research Applications Examples**

For computational physics and biophysics research:

```bash
# Complete research applications
python research_applications_demo.py
```

This demonstrates:
- ✅ Fractional diffusion in complex media
- ✅ Viscoelastic material dynamics
- ✅ Protein folding with memory effects
- ✅ Membrane transport with anomalous diffusion
- ✅ Drug delivery pharmacokinetics
- ✅ Fractional neural networks
- ✅ GPU optimization for large-scale computations

### Basic Usage Examples (Legacy)

For backward compatibility, the original examples are still available:

```bash
cd examples/basic_usage
python getting_started.py
```

**Note**: These use the old `alpha` parameter naming. Use the production-ready examples for new development.

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

### Machine Learning Examples

#### Fractional GNN Demo

Explore fractional graph neural networks:

```bash
cd examples/ml_examples
python fractional_gnn_demo.py
```

#### ML Integration Demo

Learn about machine learning integration:

```bash
cd examples/ml_examples
python ml_integration_demo.py
```

#### Probabilistic Fractional Training

Run a minimal end-to-end training loop that combines spectral autograd, stochastic memory sampling, and probabilistic fractional orders:

```bash
cd examples/ml_examples
python minimal_probabilistic_fractional_training.py
```

#### Variance-Aware Training

Learn how to monitor and control variance in stochastic fractional derivatives:

```bash
cd examples/ml_examples
python variance_aware_training_example.py
```

### Physics Examples

#### Fractional Physics Demo

Explore fractional physics applications:

```bash
cd examples/physics_examples
python fractional_physics_demo.py
```

#### Fractional PINO Experiment

Learn about Physics-Informed Neural Operators:

```bash
cd examples/physics_examples
python fractional_pino_experiment.py
```

#### Fractional vs Integer Comparison

Compare fractional and integer calculus:

```bash
cd examples/physics_examples
python fractional_vs_integer_comparison.py
```

### Benchmarking Examples

#### Performance Benchmarking

Run comprehensive performance benchmarks:

```bash
cd examples/benchmarks
python benchmark_demo.py
```

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

### 5. Machine Learning (`ml_examples/`)
- **Fractional GNNs**: Graph neural networks with fractional derivatives
- **ML Integration**: PyTorch and JAX backends
- **Probabilistic Orders**: Uncertainty quantification in fractional orders
- **Variance-Aware Training**: Monitoring and controlling training variance
- **Spectral Autograd**: Advanced automatic differentiation

### 6. Physics Examples (`physics_examples/`)
- **Fractional Physics**: Real-world physics applications
- **PINO Experiments**: Physics-Informed Neural Operators
- **Fractional vs Integer**: Comparative analysis
- **Scientific Computing**: Domain-specific applications

### 7. Benchmarking (`benchmarks/`)
- **Performance Analysis**: Comprehensive performance testing
- **Accuracy Comparisons**: Method validation and comparison
- **Scaling Studies**: Performance across different scales
- **Resource Monitoring**: Memory and computational efficiency

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
