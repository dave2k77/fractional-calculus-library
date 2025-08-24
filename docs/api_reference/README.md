# API Reference - hpfracc Library

## 📚 **Complete Function Documentation**

This section provides comprehensive documentation for all functions, classes, and methods in the hpfracc library, including complete parameter descriptions, return values, and usage examples.

## 🗂️ **Documentation Structure**

### **Core Functions**
- [**Optimized Methods**](optimized_methods.md) - Primary fractional derivative functions
- [**Advanced Methods**](advanced_methods.md) - Weyl, Marchaud, Hadamard derivatives
- [**Core Definitions**](core_definitions.md) - Mathematical foundations and base classes

### **Solvers and Applications**
- [**ODE Solvers**](ode_solvers.md) - Fractional differential equation solvers
- [**PDE Solvers**](pde_solvers.md) - Partial differential equation methods
- [**Predictor-Corrector**](predictor_corrector.md) - High-order numerical methods

### **Utilities and Special Functions**
- [**Special Functions**](special_functions.md) - Gamma, Beta, Mittag-Leffler functions
- [**Error Analysis**](error_analysis.md) - Validation and error estimation tools
- [**Plotting Utilities**](plotting_utils.md) - Visualization and analysis functions

### **Benchmarking and Validation**
- [**Benchmarking Module**](benchmarking.md) - Performance testing and analysis
- [**Validation Tools**](validation.md) - Accuracy verification and convergence testing

## 🔍 **Quick Function Lookup**

### **Primary Functions (Most Used)**

| Function | Purpose | Performance |
|----------|---------|-------------|
| `optimized_riemann_liouville()` | RL fractional derivative | **1874x speedup** |
| `optimized_caputo()` | Caputo fractional derivative | **29.6x speedup** |
| `optimized_grunwald_letnikov()` | GL fractional derivative | **113.8x speedup** |

### **Advanced Functions**

| Function | Purpose | Status |
|----------|---------|--------|
| `optimized_weyl_derivative()` | Weyl fractional derivative | ✅ Working |
| `optimized_marchaud_derivative()` | Marchaud fractional derivative | ✅ Working |
| `optimized_hadamard_derivative()` | Hadamard fractional derivative | ✅ Working |

## 📖 **Function Documentation Format**

Each function is documented with:

- **Function Signature**: Complete parameter list with types
- **Parameters**: Detailed description of each input parameter
- **Returns**: Description of output values and types
- **Raises**: Possible exceptions and error conditions
- **Examples**: Working code examples
- **Notes**: Important implementation details
- **References**: Mathematical background and citations

## 🚀 **Getting Started with API**

### **Basic Usage Pattern**

```python
from hpfracc import optimized_riemann_liouville

# Function signature:
# optimized_riemann_liouville(f, t, alpha, h, method='fft')

# Parameters:
# f: function or array - Function to differentiate
# t: array - Time points (must be uniform)
# alpha: float - Fractional order (0 < alpha < 2)
# h: float - Time step size
# method: str - Method type ('fft', 'direct')

# Returns:
# array - Fractional derivative values
```

### **Parameter Guidelines**

- **f**: Can be callable function or numpy array
- **t**: Must be uniformly spaced (use `np.linspace()`)
- **alpha**: Typically 0.25, 0.5, 0.75, 1.0
- **h**: Calculate as `t[1] - t[0]`
- **method**: Choose based on array size and accuracy needs

## 🔧 **Performance Considerations**

### **Method Selection Guide**

| Array Size | Recommended Method | Reason |
|------------|-------------------|---------|
| 100-500 | Any method | All are fast for small arrays |
| 500-2000 | RL FFT or GL Direct | Best balance of speed/accuracy |
| 2000+ | RL FFT | Fastest for large arrays |

### **Memory Requirements**

- **RL FFT**: Most memory efficient
- **GL Direct**: Moderate memory usage
- **Caputo L1**: Higher memory for large arrays

## 📊 **Error Handling**

### **Common Error Types**

1. **ValueError**: Invalid parameters (e.g., alpha ≤ 0)
2. **TypeError**: Wrong input types
3. **RuntimeError**: Numerical computation failures
4. **MemoryError**: Insufficient memory for large arrays

### **Validation Functions**

```python
from hpfracc.validation import validate_parameters

# Validate inputs before computation
is_valid, message = validate_parameters(f, t, alpha, h)
if not is_valid:
    print(f"Parameter error: {message}")
```

## 🔗 **Related Documentation**

- [**User Guide**](../user_guide.md) - Getting started and basic usage
- [**Examples**](../../examples/) - Working code examples
- [**Model Theory**](../model_theory.md) - Mathematical foundations
- [**Development Guide**](../../README_DEV.md) - For contributors

---

**Need a specific function?** Use the navigation above or search for function names in the documentation.
