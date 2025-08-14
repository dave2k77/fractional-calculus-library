# Fractional Calculus Library - Benchmarks

This directory contains comprehensive benchmarking tools for analyzing the performance, accuracy, and scaling characteristics of the fractional calculus library.

## 📁 Directory Structure

```
benchmarks/
├── performance_tests.py      # Performance benchmarking suite
├── accuracy_comparisons.py   # Accuracy analysis and validation
├── scaling_analysis.py       # Scaling and complexity analysis
├── plots/                   # Generated visualization plots
├── performance_report.txt   # Performance analysis results
├── accuracy_report.txt      # Accuracy analysis results
├── scaling_analysis_report.txt # Scaling analysis results
└── README.md               # This file
```

## 🚀 Quick Start

### Run All Benchmarks

Execute the complete benchmarking suite:

```bash
# Performance tests
python performance_tests.py

# Accuracy comparisons
python accuracy_comparisons.py

# Scaling analysis
python scaling_analysis.py
```

### Individual Benchmark Categories

#### 1. Performance Tests
```bash
python performance_tests.py
```

**What it tests:**
- Execution time for different methods
- Memory usage patterns
- Optimization backend performance
- Accuracy vs speed trade-offs

**Output:**
- Performance comparison plots
- Memory usage analysis
- Throughput measurements
- Detailed performance report

#### 2. Accuracy Comparisons
```bash
python accuracy_comparisons.py
```

**What it tests:**
- Numerical vs analytical solutions
- Convergence rates
- Stability analysis
- Error propagation

**Output:**
- Accuracy comparison plots
- Convergence analysis
- Stability assessment
- Detailed accuracy report

#### 3. Scaling Analysis
```bash
python scaling_analysis.py
```

**What it tests:**
- Computational scaling with problem size
- Parallel scaling with worker count
- Memory scaling patterns
- Optimization backend scaling

**Output:**
- Scaling analysis plots
- Complexity analysis
- Parallel efficiency metrics
- Detailed scaling report

## 📊 Benchmark Categories

### 1. Performance Tests (`performance_tests.py`)

#### Derivative Methods Performance
- **Caputo Derivative**: Time-domain approach
- **Riemann-Liouville Derivative**: Classical definition
- **Grünwald-Letnikov Derivative**: Finite difference approach
- **FFT Spectral**: Spectral domain computation
- **FFT Convolution**: Convolution-based approach

#### Optimization Backends
- **JAX GPU**: GPU acceleration with automatic differentiation
- **Numba**: JIT compilation for CPU optimization
- **Parallel (Joblib)**: Multi-core parallel processing

#### Memory Usage Analysis
- Memory consumption patterns
- Memory efficiency metrics
- Peak memory usage tracking

#### Accuracy vs Speed Trade-offs
- Performance comparison with analytical solutions
- Error analysis for different methods
- Optimal method selection guidelines

### 2. Accuracy Comparisons (`accuracy_comparisons.py`)

#### Test Functions
- **Constant**: f(t) = 1
- **Linear**: f(t) = t
- **Quadratic**: f(t) = t²
- **Cubic**: f(t) = t³
- **Exponential**: f(t) = exp(-t)
- **Sine**: f(t) = sin(t)
- **Cosine**: f(t) = cos(t)
- **Power**: f(t) = t^0.5
- **Logarithmic**: f(t) = log(1+t)
- **Gaussian**: f(t) = exp(-t²)

#### Analytical Solutions
- **Caputo Derivatives**: Exact analytical solutions
- **Riemann-Liouville Derivatives**: Classical analytical forms
- **Error Metrics**: Maximum, mean, L2, and relative errors

#### Convergence Analysis
- Grid refinement studies
- Convergence rate calculation
- Method comparison

#### Stability Analysis
- Numerical stability assessment
- Condition number analysis
- Robustness testing

### 3. Scaling Analysis (`scaling_analysis.py`)

#### Computational Scaling
- Time complexity analysis
- Memory complexity analysis
- Throughput scaling
- Complexity ratio calculations

#### Parallel Scaling
- Speedup analysis
- Efficiency metrics
- Load balancing assessment
- Worker count optimization

#### Memory Scaling
- Memory usage patterns
- Memory efficiency analysis
- Peak memory tracking
- Memory per point metrics

#### Optimization Backend Scaling
- JAX GPU scaling
- Numba scaling
- Performance comparison
- Hardware utilization

## 📈 Generated Reports

### Performance Report (`performance_report.txt`)
```
FRACTIONAL CALCULUS LIBRARY - PERFORMANCE REPORT
============================================================

SYSTEM INFORMATION:
  CPU Count: 16
  Total Memory: 32.00 GB
  Platform: nt
  Python Version: 3.9.7

DERIVATIVE METHODS PERFORMANCE:
  Grid Size 100:
    Caputo: 0.0012s
    Riemann-Liouville: 0.0015s
    Grünwald-Letnikov: 0.0018s
    FFT Spectral: 0.0021s
    FFT Convolution: 0.0023s

OPTIMIZATION BACKENDS PERFORMANCE:
  Grid Size 1000:
    JAX GPU: 0.0008s
    Numba: 0.0012s
    Parallel (Joblib): 0.0015s

MEMORY USAGE ANALYSIS:
  Grid Size 1000:
    Caputo: 0.008 MB
    Riemann-Liouville: 0.008 MB
    FFT Spectral: 0.016 MB
```

### Accuracy Report (`accuracy_report.txt`)
```
FRACTIONAL CALCULUS LIBRARY - ACCURACY REPORT
============================================================

ACCURACY ANALYSIS:
  α = 0.5:
    Function: linear
      Grid 50: Caputo (max error: 1.23e-05)
      Grid 100: Caputo (max error: 3.12e-06)
      Grid 200: Caputo (max error: 7.89e-07)

CONVERGENCE RATE ANALYSIS:
  Function: linear
    Caputo: 1.987
    Riemann-Liouville: 1.945
    Grünwald-Letnikov: 1.876
    FFT Spectral: 2.123

STABILITY ANALYSIS:
  Function: oscillatory
    α=0.5, Grid 100: Stable methods: ['Caputo', 'Riemann-Liouville', 'FFT Spectral']
```

### Scaling Analysis Report (`scaling_analysis_report.txt`)
```
FRACTIONAL CALCULUS LIBRARY - SCALING ANALYSIS REPORT
============================================================

SYSTEM INFORMATION:
  CPU Count: 16
  CPU Frequency: {'current': 2400.0, 'min': 800.0, 'max': 3200.0}
  Total Memory: 32.00 GB
  Available Memory: 28.50 GB

COMPUTATIONAL SCALING ANALYSIS:
  Grid Size 100:
    Caputo: 0.0012s, 1250 ops/s
    Riemann-Liouville: 0.0015s, 1000 ops/s
    Grünwald-Letnikov: 0.0018s, 833 ops/s

PARALLEL SCALING ANALYSIS:
  Grid Size 1000:
    joblib (1 workers): 0.0150s, efficiency: 1.00
    joblib (2 workers): 0.0080s, efficiency: 0.94
    joblib (4 workers): 0.0045s, efficiency: 0.83
    joblib (8 workers): 0.0030s, efficiency: 0.63

MEMORY SCALING ANALYSIS:
  Grid Size 1000:
    Caputo: 0.008 MB, efficiency: 1.00
    Riemann-Liouville: 0.008 MB, efficiency: 1.00
    FFT Spectral: 0.016 MB, efficiency: 0.50
```

## 📊 Generated Plots

### Performance Plots (`plots/`)
- `derivative_methods_performance.png`: Execution time comparison
- `optimization_backends_performance.png`: Backend performance analysis
- `memory_usage_analysis.png`: Memory consumption patterns
- `accuracy_vs_speed.png`: Performance-accuracy trade-offs

### Accuracy Plots (`plots/`)
- `accuracy_comparison.png`: Error analysis for different methods
- `convergence_analysis.png`: Convergence rate visualization
- `stability_analysis.png`: Numerical stability assessment

### Scaling Plots (`plots/`)
- `computational_scaling.png`: Time and memory scaling
- `parallel_scaling.png`: Parallel efficiency analysis
- `memory_scaling.png`: Memory usage scaling
- `accuracy_scaling.png`: Accuracy vs problem size

## 🔧 Configuration

### Customizing Benchmark Parameters

#### Performance Tests
```python
# Modify grid sizes
grid_sizes = [50, 100, 200, 500, 1000, 2000, 5000]

# Modify test parameters
alpha = 0.5
t_max = 2.0
```

#### Accuracy Comparisons
```python
# Modify alpha values
alpha_values = [0.25, 0.5, 0.75]

# Modify grid sizes for convergence
grid_sizes = [25, 50, 100, 200, 400, 800]
```

#### Scaling Analysis
```python
# Modify worker counts
worker_counts = [1, 2, 4, 8, 16, 32]

# Modify grid sizes
grid_sizes = [100, 500, 1000, 2000, 5000]
```

### System-Specific Optimization

#### For High-Performance Systems
```python
# Increase grid sizes for better scaling analysis
grid_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]

# Test more worker counts
worker_counts = [1, 2, 4, 8, 16, 32, 64]
```

#### For Limited Resources
```python
# Reduce grid sizes for faster execution
grid_sizes = [50, 100, 200, 500]

# Limit worker counts
worker_counts = [1, 2, 4]
```

## 📈 Performance Guidelines

### Method Selection

#### For Speed
1. **JAX GPU**: Best for large datasets with GPU
2. **Numba**: Best for CPU-only systems
3. **FFT Methods**: Good for periodic functions

#### For Accuracy
1. **Caputo**: Most accurate for most functions
2. **Riemann-Liouville**: Good for classical problems
3. **Grünwald-Letnikov**: Good for finite difference approaches

#### For Memory Efficiency
1. **Caputo/Riemann-Liouville**: Most memory efficient
2. **FFT Methods**: Higher memory usage
3. **Parallel Methods**: Additional memory overhead

### Optimization Recommendations

#### Small Problems (N < 1000)
- Use standard methods (Caputo, Riemann-Liouville)
- Single-threaded execution
- Focus on accuracy over speed

#### Medium Problems (1000 ≤ N < 10000)
- Use Numba optimization
- Moderate parallel processing (4-8 workers)
- Balance accuracy and speed

#### Large Problems (N ≥ 10000)
- Use JAX GPU if available
- Full parallel processing (8-16 workers)
- Consider FFT methods for periodic functions

## 🐛 Troubleshooting

### Common Issues

#### 1. Memory Errors
```bash
# Reduce grid sizes
grid_sizes = [50, 100, 200, 500]

# Use memory-efficient methods
# Monitor system memory usage
```

#### 2. Performance Issues
```bash
# Check CPU utilization
# Monitor GPU usage (if using JAX)
# Adjust worker counts
```

#### 3. Accuracy Problems
```bash
# Use smaller step sizes
# Check analytical solutions
# Validate with known test cases
```

#### 4. Import Errors
```bash
# Ensure all dependencies are installed
pip install numpy scipy matplotlib jax jaxlib joblib psutil

# Check Python path
python -c "import src.algorithms.caputo"
```

### Performance Debugging

#### Monitor System Resources
```python
import psutil
import time

# Monitor CPU usage
cpu_percent = psutil.cpu_percent(interval=1)

# Monitor memory usage
memory = psutil.virtual_memory()

# Monitor execution time
start_time = time.time()
# ... your computation ...
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f}s")
```

#### Profile Specific Methods
```python
import cProfile
import pstats

# Profile a specific function
profiler = cProfile.Profile()
profiler.enable()
# ... your computation ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 📚 Advanced Usage

### Custom Benchmarking

#### Adding New Test Functions
```python
def custom_test_function(t):
    """Custom test function for benchmarking."""
    return np.sin(2 * np.pi * t) * np.exp(-t/2)

# Add to test functions dictionary
test_functions['custom'] = custom_test_function
```

#### Custom Analytical Solutions
```python
def custom_analytical_solution(t, alpha):
    """Custom analytical solution."""
    from scipy.special import gamma
    return t**(1-alpha) / gamma(2-alpha)

# Add to analytical solutions
analytical_solutions['caputo']['custom'] = custom_analytical_solution
```

#### Custom Performance Metrics
```python
def custom_performance_metric(result, time_taken):
    """Custom performance metric."""
    return {
        'custom_metric': len(result) / time_taken,
        'efficiency': result.nbytes / (time_taken * 1024 * 1024)
    }
```

### Integration with CI/CD

#### Automated Benchmarking
```yaml
# .github/workflows/benchmarks.yml
name: Run Benchmarks
on: [push, pull_request]

jobs:
  benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run benchmarks
        run: |
          python benchmarks/performance_tests.py
          python benchmarks/accuracy_comparisons.py
          python benchmarks/scaling_analysis.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/plots/
```

## 🤝 Contributing

### Adding New Benchmarks

1. **Follow the existing structure**
2. **Include comprehensive documentation**
3. **Add error handling**
4. **Provide performance metrics**
5. **Include visualization options**
6. **Test on different systems**

### Benchmark Guidelines

#### Performance Tests
- Test multiple grid sizes
- Include warm-up runs
- Measure both time and memory
- Compare with baseline methods

#### Accuracy Tests
- Use analytical solutions when available
- Test multiple alpha values
- Include convergence analysis
- Validate with known results

#### Scaling Tests
- Test multiple problem sizes
- Include parallel scaling
- Measure memory scaling
- Analyze complexity

## 📄 License

This benchmarks directory is part of the fractional calculus library and follows the same license terms.

---

**Happy Benchmarking! 📊**

For more information, see the main library documentation and the `examples/` directory for usage examples.
