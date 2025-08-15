# Algorithms Folder Consolidation Plan

## Current Structure Analysis

The `src/algorithms/` folder contains multiple files with overlapping functionality:

### Core Algorithm Files (STANDARD implementations):
1. `caputo.py` - Standard Caputo derivative implementations
2. `riemann_liouville.py` - Standard RL derivative implementations  
3. `grunwald_letnikov.py` - Standard GL derivative implementations
4. `fft_methods.py` - FFT-based methods
5. `L1_L2_schemes.py` - L1/L2 schemes for PDEs

### Optimized Algorithm Files (OPTIMIZED implementations):
1. `optimized_methods.py` - Highly optimized versions of core methods
2. `gpu_optimized_methods.py` - GPU-accelerated versions
3. `parallel_optimized_methods.py` - Parallel processing versions
4. `advanced_methods.py` - Advanced fractional calculus methods
5. `advanced_optimized_methods.py` - Optimized advanced methods

## Duplication Analysis

### **DUPLICATE IMPLEMENTATIONS:**
1. **Caputo Derivatives:**
   - `caputo.py` → `CaputoDerivative` (standard)
   - `optimized_methods.py` → `OptimizedCaputo` (optimized)
   - `gpu_optimized_methods.py` → `GPUOptimizedCaputo` (GPU)
   - `parallel_optimized_methods.py` → `ParallelOptimizedCaputo` (parallel)

2. **Riemann-Liouville Derivatives:**
   - `riemann_liouville.py` → `RiemannLiouvilleDerivative` (standard)
   - `optimized_methods.py` → `OptimizedRiemannLiouville` (optimized)
   - `gpu_optimized_methods.py` → `GPUOptimizedRiemannLiouville` (GPU)
   - `parallel_optimized_methods.py` → `ParallelOptimizedRiemannLiouville` (parallel)

3. **Grünwald-Letnikov Derivatives:**
   - `grunwald_letnikov.py` → `GrunwaldLetnikovDerivative` (standard)
   - `optimized_methods.py` → `OptimizedGrunwaldLetnikov` (optimized)
   - `gpu_optimized_methods.py` → `GPUOptimizedGrunwaldLetnikov` (GPU)
   - `parallel_optimized_methods.py` → `ParallelOptimizedGrunwaldLetnikov` (parallel)

4. **FFT Methods:**
   - `fft_methods.py` → `FFTFractionalMethods` (standard)
   - `optimized_methods.py` → Contains FFT convolution for RL (optimized)

5. **L1/L2 Schemes:**
   - `L1_L2_schemes.py` → `L1L2Schemes` (standard)
   - `optimized_methods.py` → Contains L1 scheme for Caputo (optimized)

## Consolidation Strategy

### **KEEP (Essential files):**
1. `optimized_methods.py` - **PRIMARY** - Contains the most efficient implementations
2. `gpu_optimized_methods.py` - **KEEP** - GPU acceleration
3. `parallel_optimized_methods.py` - **KEEP** - Parallel processing
4. `advanced_methods.py` - **KEEP** - Advanced methods (Weyl, Marchaud, etc.)
5. `advanced_optimized_methods.py` - **KEEP** - Optimized advanced methods

### **CONSOLIDATE/REMOVE (Redundant files):**
1. `caputo.py` - **REMOVE** - Functionality covered by `optimized_methods.py`
2. `riemann_liouville.py` - **REMOVE** - Functionality covered by `optimized_methods.py`
3. `grunwald_letnikov.py` - **REMOVE** - Functionality covered by `optimized_methods.py`
4. `fft_methods.py` - **CONSOLIDATE** - Extract unique FFT methods into `optimized_methods.py`
5. `L1_L2_schemes.py` - **CONSOLIDATE** - Extract unique L1/L2 schemes into `optimized_methods.py`

## Recommended Actions

### **Phase 1: Extract Unique Features**
1. Extract unique FFT methods from `fft_methods.py` into `optimized_methods.py`
2. Extract unique L1/L2 schemes from `L1_L2_schemes.py` into `optimized_methods.py`
3. Extract any unique JAX implementations from core files

### **Phase 2: Update Imports**
1. Update `__init__.py` to remove imports from deleted files
2. Update any external imports to use consolidated implementations
3. Update documentation to reflect new structure

### **Phase 3: Remove Redundant Files**
1. Delete `caputo.py`, `riemann_liouville.py`, `grunwald_letnikov.py`
2. Delete `fft_methods.py` and `L1_L2_schemes.py` after consolidation

### **Phase 4: Update Documentation**
1. Update README.md to reflect new structure
2. Update performance_report.md
3. Update any other documentation

## Benefits of Consolidation

- **Reduce codebase size** by ~50% (from 12 files to 6 files)
- **Eliminate maintenance burden** of duplicate implementations
- **Simplify import structure** - single source of truth for each method
- **Improve code quality** by keeping only the best implementations
- **Reduce confusion** for users about which implementation to use

## Final Structure

```
src/algorithms/
├── __init__.py                    # Updated imports
├── optimized_methods.py           # PRIMARY - All core optimized methods
├── gpu_optimized_methods.py       # GPU acceleration
├── parallel_optimized_methods.py  # Parallel processing
├── advanced_methods.py            # Advanced methods (Weyl, Marchaud, etc.)
└── advanced_optimized_methods.py  # Optimized advanced methods
```

This consolidation will result in a cleaner, more maintainable codebase with clear separation of concerns and no duplicate implementations.
