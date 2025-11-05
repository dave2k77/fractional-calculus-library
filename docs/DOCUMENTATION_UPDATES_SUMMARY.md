# Documentation Updates Summary

This document summarizes all documentation and API reference updates made to reflect the critical fixes and improvements implemented in HPFRACC v3.0.1.

## Date: 2025-01-30

## Overview

All documentation has been updated to reflect:
1. Mathematical correctness fixes (binomial coefficients, fractional derivatives)
2. New feature implementations (Caputo integral for α ≥ 1, backend-agnostic support)
3. Improved error messages and limitations documentation
4. Enhanced API documentation

---

## Updated Documentation Files

### 1. Core Module Documentation

#### `docs/06_derivatives_integrals.rst`
- ✅ Updated Caputo Integral section to note support for all orders α ≥ 0
- ✅ Added explanation of decomposition method for α ≥ 1
- ✅ Clarified that Caputo equals Riemann-Liouville for 0 < α < 1

#### `docs/model_theory.rst`
- ✅ Updated Caputo Fractional Integral mathematical theory section
- ✅ Added decomposition formula for α ≥ 1: I^α = I^n ∘ I^β

### 2. Machine Learning Documentation

#### `docs/08_fractional_gnn.rst`
- ✅ Updated Key Features section:
  - Noted full multi-backend support (PyTorch, JAX, NUMBA)
  - Emphasized proper fractional derivative implementations (no placeholders)
  - Clarified backend-agnostic fractional derivatives

#### `docs/09_neural_ode_sde.rst`
- ✅ Added note on diffusion function limitations:
  - Documented supported types: scalar and vector diffusion
  - Noted matrix diffusion is not yet implemented
  - Provided workarounds for users

### 3. User Guide

#### `docs/user_guide.rst`
- ✅ Updated Caputo Integral usage example:
  - Added example for α ≥ 1 (newly implemented)
  - Updated import statements to use correct API
  - Added note about decomposition method

- ✅ Updated Binomial Coefficients section:
  - Changed to use `BinomialCoefficients` class API
  - Added example for fractional binomial coefficients
  - Noted improved implementation accuracy

### 4. Advanced Usage Documentation

#### `docs/11_advanced_usage.rst`
- ✅ Added comprehensive "Known Limitations" section:
  - **SDE Solvers**: Matrix diffusion limitations and workarounds
  - **ODE Solvers**: FFT convolution axis limitations, predictor-corrector scope
  - **PDE Solvers**: Spectral scheme range limitations
  - **Adaptive ODE Solver**: Current disabled status
  
- ✅ Added Backend Support section:
  - Documented multi-backend fractional derivative support
  - Listed GPU optimization status for all methods
  - Noted ML module backend-agnostic support

---

## Updated API Documentation

### Module Docstrings

#### `hpfracc/core/integrals.py`
- ✅ Enhanced `CaputoIntegral` class docstring:
  - Added support information for all orders α ≥ 0
  - Included usage examples for both 0 < α < 1 and α ≥ 1 cases
  - Documented decomposition method

#### `hpfracc/special/binomial_coeffs.py`
- ✅ Enhanced `_binomial_fractional_numba_scalar` method docstring:
  - Documented recursive formula: C(α,k) = C(α,k-1) * (α-k+1) / k
  - Added parameter descriptions
  - Noted accuracy improvements

#### `hpfracc/ml/gnn_layers.py`
- ✅ Enhanced `_torch_fractional_derivative` method docstring:
  - Documented proper fractional derivative approximation
  - Explained weighted combination for 0 < α < 1
  - Noted iterative approach for α ≥ 1
  - Replaced placeholder references

#### `hpfracc/algorithms/optimized_methods.py`
- ✅ Enhanced class docstrings:
  - `ParallelOptimizedRiemannLiouville`: Documented inheritance and future enhancements
  - `ParallelOptimizedCaputo`: Same as above
  - `ParallelOptimizedGrunwaldLetnikov`: Same as above
  - `AdvancedFFTMethods`: Documented FFT-based implementation
  - `NumbaOptimizer`, `NumbaFractionalKernels`: Added placeholder descriptions

---

## Key Changes Summary

### 1. Caputo Integral - Extended Support
**Before**: Only supported 0 < α < 1  
**After**: Supports all orders α ≥ 0 with decomposition method

**Documentation Updates**:
- API docs now show full order support
- Examples include α ≥ 1 cases
- Mathematical theory updated with decomposition formula

### 2. Binomial Coefficients - Improved Accuracy
**Before**: Placeholder returning 1.0 for k > 2  
**After**: Proper recursive formula implementation

**Documentation Updates**:
- Updated examples to use `BinomialCoefficients` class
- Documented recursive formula in docstrings
- Updated user guide examples

### 3. GNN Fractional Derivatives - Proper Implementation
**Before**: Placeholder `x * (alpha ** 0.5)`  
**After**: Mathematically correct fractional derivative approximation

**Documentation Updates**:
- Updated GNN guide to note "proper implementations"
- Added backend support documentation
- Enhanced method docstrings

### 4. Backend-Agnostic Support
**Before**: Only PyTorch backend supported  
**After**: Full support for PyTorch, JAX, and NumPy/NUMBA

**Documentation Updates**:
- Updated ML module documentation
- Added backend support section in advanced usage
- Documented implementation differences

### 5. Error Messages - Enhanced Clarity
**Before**: Generic `NotImplementedError` messages  
**After**: Detailed error messages with workarounds

**Documentation Updates**:
- Added "Known Limitations" section
- Documented all solver limitations
- Provided workarounds for each limitation

---

## Validation

✅ All documented classes are importable:
- `CaputoIntegral`: OK
- `BinomialCoefficients`: OK
- `AdvancedFFTMethods`: OK

✅ All API references updated:
- Core integrals API: Updated
- Special functions API: Updated
- ML modules API: Updated
- Solvers API: Updated

✅ No linting errors in documentation files

---

## Migration Notes for Users

### Caputo Integrals
```python
# Old (only worked for 0 < α < 1)
integral = CaputoIntegral(0.5)

# New (works for all α ≥ 0)
integral_05 = CaputoIntegral(0.5)  # Still works
integral_15 = CaputoIntegral(1.5)  # Now works!
```

### Binomial Coefficients
```python
# Old (placeholder)
from hpfracc.special import binomial_coefficient

# New (proper implementation)
from hpfracc.special.binomial_coeffs import BinomialCoefficients
bc = BinomialCoefficients()
result = bc.compute_fractional(0.5, 5)  # Accurate!
```

### Backend Support
```python
# Now works with all backends
from hpfracc.ml.backends import BackendType
from hpfracc.ml.losses import FractionalLossBase

# JAX backend (now fully supported)
loss = FractionalLossBase(alpha, BackendType.JAX)

# NumPy backend (now fully supported)
loss = FractionalLossBase(alpha, BackendType.NUMBA)
```

---

## Future Documentation Updates

The following are documented as limitations but may be implemented in future versions:
- Full matrix diffusion in SDE solvers
- Multi-axis FFT convolution
- Predictor-corrector for non-Caputo derivatives
- Spectral PDE schemes for α ≥ 1
- Adaptive ODE solver (currently disabled)

---

## Conclusion

All documentation and API references have been comprehensively updated to reflect:
1. Mathematical correctness improvements
2. New feature implementations
3. Enhanced error messages
4. Backend-agnostic support

The documentation now accurately represents the current state of HPFRACC v3.0.1 and provides clear guidance for users on both capabilities and limitations.

