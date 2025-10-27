# Intelligent Backend Selector - Integration Guide

## Overview

The new **Intelligent Backend Selector** provides:
- ✅ Workload-aware backend selection (small data → NumPy, large data → GPU)
- ✅ Performance monitoring and learning (adapts over time)
- ✅ Dynamic GPU memory management
- ✅ Automatic fallback on errors
- ✅ Minimal overhead (~0.0006 ms per selection)

## Quick Start

### Basic Usage

```python
from hpfracc.ml.intelligent_backend_selector import select_optimal_backend

# Quick backend selection
backend = select_optimal_backend(
    operation_type="matmul",
    data_shape=(1000, 1000),
    requires_gradient=False
)
# Returns: BackendType.TORCH or BackendType.JAX depending on GPU availability
```

### Advanced Usage

```python
from hpfracc.ml.intelligent_backend_selector import (
    IntelligentBackendSelector,
    WorkloadCharacteristics
)

# Create selector with learning enabled
selector = IntelligentBackendSelector(enable_learning=True)

# Define your workload
workload = WorkloadCharacteristics(
    operation_type="fft",
    data_size=1000000,
    data_shape=(1000, 1000),
    requires_gradient=False
)

# Get optimal backend
backend = selector.select_backend(workload)
```

### With Automatic Fallback

```python
# Execute with automatic fallback
result = selector.execute_with_monitoring(
    operation_name="fractional_derivative",
    backend=backend,
    func=lambda: compute_derivative(data),
    workload=workload,
    fallback_backends=[BackendType.JAX, BackendType.NUMBA]
)
```

## Integration Examples

### 1. Integrating with Existing Fractional Operators

**File:** `hpfracc/core/fractional_implementations.py`

**Before:**
```python
class FractionalDerivative:
    def compute(self, f, x, **kwargs):
        # Always uses predefined backend
        return self._optimized_impl.compute(f, x, **kwargs)
```

**After:**
```python
from hpfracc.ml.intelligent_backend_selector import (
    IntelligentBackendSelector,
    WorkloadCharacteristics
)

class FractionalDerivative:
    def __init__(self, alpha, method='RL'):
        self.alpha = alpha
        self.method = method
        self.backend_selector = IntelligentBackendSelector(enable_learning=True)
    
    def compute(self, f, x, **kwargs):
        # Determine workload characteristics
        if callable(f):
            x_array = np.asarray(x)
        else:
            x_array = np.asarray(f)
        
        workload = WorkloadCharacteristics(
            operation_type="derivative",
            data_size=x_array.size,
            data_shape=x_array.shape,
            requires_gradient=kwargs.get('requires_grad', False)
        )
        
        # Select optimal backend
        backend = self.backend_selector.select_backend(workload)
        
        # Execute with monitoring and fallback
        def compute_func():
            impl = self._get_implementation(backend)
            return impl.compute(f, x, **kwargs)
        
        return self.backend_selector.execute_with_monitoring(
            operation_name=f"derivative_alpha_{self.alpha}",
            backend=backend,
            func=compute_func,
            workload=workload,
            fallback_backends=[BackendType.JAX, BackendType.NUMBA]
        )
```

### 2. Integrating with ML Layers

**File:** `hpfracc/ml/layers.py`

**Before:**
```python
class FractionalLayer(nn.Module):
    def __init__(self, alpha, config):
        self.alpha = alpha
        self.backend = "pytorch"  # Fixed
```

**After:**
```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector

class FractionalLayer(nn.Module):
    def __init__(self, alpha, config):
        super().__init__()
        self.alpha = alpha
        self.config = config
        self.backend_selector = IntelligentBackendSelector(enable_learning=True)
    
    def forward(self, x):
        # Characterize workload
        workload = WorkloadCharacteristics(
            operation_type="neural_network",
            data_size=x.numel(),
            data_shape=x.shape,
            requires_gradient=x.requires_grad
        )
        
        # Select backend (will learn over time)
        backend = self.backend_selector.select_backend(
            workload, 
            preferred_backend=self.config.backend
        )
        
        # Use selected backend for computation
        if backend == BackendType.JAX and x.requires_grad:
            # Convert to JAX, compute, convert back
            ...
        else:
            # Use PyTorch
            return self._pytorch_forward(x)
```

### 3. Integrating with GPU-Optimized Methods

**File:** `hpfracc/algorithms/gpu_optimized_methods.py`

**Enhancement:**
```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector

class GPUOptimizedCaputo:
    def __init__(self, alpha, gpu_config=None):
        self.alpha = alpha
        self.gpu_config = gpu_config or GPUConfig()
        
        # Add intelligent selector
        self.backend_selector = IntelligentBackendSelector()
        
        # Auto-detect optimal backend based on data
        if self.gpu_config.backend == "auto":
            # Will be determined per-computation based on data size
            self.gpu_config.backend = None
    
    def compute(self, f, t, h=None):
        # Prepare data
        f_array = self._prepare_data(f, t, h)
        
        # Characterize workload
        workload = WorkloadCharacteristics(
            operation_type="caputo_derivative",
            data_size=f_array.size,
            data_shape=f_array.shape,
            is_iterative=True
        )
        
        # Select backend if not explicitly set
        if self.gpu_config.backend is None:
            from hpfracc.ml.backends import BackendType
            backend_type = self.backend_selector.select_backend(workload)
            
            if backend_type == BackendType.JAX:
                self.gpu_config.backend = "jax"
            elif backend_type == BackendType.TORCH:
                self.gpu_config.backend = "torch"
            else:
                self.gpu_config.backend = "numpy"
        
        # Execute with fallback
        try:
            if self.gpu_config.backend == "jax" and JAX_AVAILABLE:
                return self._compute_jax_l1(f_array, h)
            else:
                return self._compute_cpu_fallback(f_array, h, "L1")
        except Exception as e:
            if self.gpu_config.fallback_to_cpu:
                return self._compute_cpu_fallback(f_array, h, "L1")
            raise
```

### 4. Integrating with Solvers

**File:** `hpfracc/solvers/ode_solvers.py`

**Enhancement:**
```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector

class FixedStepODESolver:
    def __init__(self, derivative_type='caputo', method='predictor_corrector', adaptive=False):
        self.derivative_type = derivative_type
        self.method = method
        self.adaptive = adaptive
        
        # Add intelligent backend selection
        self.backend_selector = IntelligentBackendSelector(enable_learning=True)
    
    def solve(self, f, t_span, y0, alpha, h, **kwargs):
        # Estimate computational workload
        n_steps = int((t_span[1] - t_span[0]) / h)
        y0_array = np.asarray(y0)
        
        workload = WorkloadCharacteristics(
            operation_type="ode_solve",
            data_size=n_steps * y0_array.size,
            data_shape=(n_steps, *y0_array.shape),
            is_iterative=True
        )
        
        # Select optimal backend
        backend = self.backend_selector.select_backend(workload)
        
        # Choose implementation based on backend
        if backend == BackendType.JAX and not self.adaptive:
            return self._solve_jax(f, t_span, y0, alpha, h, **kwargs)
        elif backend == BackendType.TORCH:
            return self._solve_torch(f, t_span, y0, alpha, h, **kwargs)
        else:
            return self._solve_numpy(f, t_span, y0, alpha, h, **kwargs)
```

## Performance Monitoring

### Enable Learning

```python
# Initialize with learning
selector = IntelligentBackendSelector(enable_learning=True)

# Use for multiple operations
for data in dataset:
    backend = selector.select_backend(workload)
    # ... compute ...

# Check what was learned
stats = selector.get_performance_summary()
print(stats)
```

### View Statistics

```python
{
    'total_records': 1000,
    'operation_stats': {
        (BackendType.JAX, 'matmul'): {
            'total_time': 5.2,
            'count': 500,
            'failures': 2,
            'avg_time': 0.0104
        },
        (BackendType.TORCH, 'matmul'): {
            'total_time': 8.7,
            'count': 500,
            'failures': 0,
            'avg_time': 0.0174
        }
    },
    'recent_backends': [...]
}
```

## Environment Variables

Control backend selection with environment variables:

```bash
# Force a specific backend
export HPFRACC_FORCE_JAX=1

# Disable a backend
export HPFRACC_DISABLE_TORCH=1

# Disable GPU
export JAX_PLATFORM_NAME=cpu
```

## Best Practices

### 1. Reuse Selector Instances

**Good:**
```python
class MyModel:
    def __init__(self):
        self.backend_selector = IntelligentBackendSelector(enable_learning=True)
    
    def forward(self, x):
        backend = self.backend_selector.select_backend(workload)
```

**Avoid:**
```python
def forward(self, x):
    # Don't create new selector each time - loses learning
    selector = IntelligentBackendSelector()
    backend = selector.select_backend(workload)
```

### 2. Use Appropriate Operation Types

Recommended operation types:
- `"matmul"` - Matrix multiplication
- `"conv"` - Convolution operations
- `"fft"` - FFT operations
- `"derivative"` - Fractional derivatives
- `"element_wise"` - Element-wise operations
- `"neural_network"` - Neural network forward/backward
- `"ode_solve"` - ODE integration
- `"iterative"` - Iterative algorithms

### 3. Enable Learning for Production

```python
# Development: Fast startup, no learning
selector = IntelligentBackendSelector(enable_learning=False)

# Production: Learns optimal backends over time
selector = IntelligentBackendSelector(enable_learning=True)
```

### 4. Specify Workload Characteristics Accurately

```python
# Good - provides all relevant information
workload = WorkloadCharacteristics(
    operation_type="derivative",
    data_size=x.size,
    data_shape=x.shape,
    dtype_size=8,  # float64
    is_iterative=True,
    requires_gradient=True
)

# Avoid - minimal information may lead to suboptimal selection
workload = WorkloadCharacteristics(
    operation_type="other",
    data_size=x.size,
    data_shape=x.shape
)
```

## Troubleshooting

### Issue: Backend selection is slow

**Solution:** Backend selection has ~0.0006 ms overhead. If this is significant:
1. Cache the backend for repeated operations on similar data
2. Disable learning: `IntelligentBackendSelector(enable_learning=False)`

### Issue: Wrong backend selected

**Reasons:**
1. Not enough learning data (< 5 samples per operation)
2. Workload characteristics not accurate
3. GPU detection failed

**Solutions:**
```python
# Force a backend temporarily
backend = selector.select_backend(workload, preferred_backend=BackendType.JAX)

# Reset learning history
selector.reset_performance_history()

# Check GPU detection
stats = selector.get_performance_summary()
```

### Issue: Fallback not working

**Check:**
```python
result = selector.execute_with_monitoring(
    ...,
    fallback_backends=[BackendType.JAX, BackendType.NUMBA]  # Must specify fallbacks
)
```

## Migration Guide

### Step 1: Add import

```python
from hpfracc.ml.intelligent_backend_selector import (
    IntelligentBackendSelector,
    WorkloadCharacteristics,
    select_optimal_backend  # For quick selection
)
```

### Step 2: Initialize selector

```python
class YourClass:
    def __init__(self):
        self.backend_selector = IntelligentBackendSelector(enable_learning=True)
```

### Step 3: Characterize workload

```python
def compute(self, data):
    workload = WorkloadCharacteristics(
        operation_type="your_operation",
        data_size=data.size,
        data_shape=data.shape,
        requires_gradient=data.requires_grad if hasattr(data, 'requires_grad') else False
    )
```

### Step 4: Select and use backend

```python
    backend = self.backend_selector.select_backend(workload)
    
    # Option A: Select implementation based on backend
    if backend == BackendType.JAX:
        result = self._compute_jax(data)
    else:
        result = self._compute_torch(data)
    
    # Option B: Use execute_with_monitoring for automatic fallback
    result = self.backend_selector.execute_with_monitoring(
        operation_name="your_operation",
        backend=backend,
        func=lambda: self._compute(data, backend),
        workload=workload,
        fallback_backends=[BackendType.JAX, BackendType.NUMBA]
    )
```

## Testing

Run the test suite:

```bash
python test_intelligent_backend.py
```

Expected output:
```
SUMMARY: 9 passed, 0 failed out of 9 tests
🎉 All tests passed!
```

## Performance Impact

Benchmark results:
- **Selection overhead:** ~0.0006 ms (negligible)
- **GPU memory detection:** Cached for 60s
- **Learning overhead:** ~0.001 ms per record
- **Total impact:** < 0.1% for operations > 10ms

## Future Enhancements

Planned improvements:
1. ✅ Workload-based selection (IMPLEMENTED)
2. ✅ Performance monitoring (IMPLEMENTED)
3. ✅ Dynamic thresholds (IMPLEMENTED)
4. 🔄 Auto-tuning of thresholds based on hardware
5. 🔄 Distributed backend selection for multi-node
6. 🔄 Power consumption optimization

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review test examples in `test_intelligent_backend.py`
3. Examine detailed analysis in `BACKEND_ANALYSIS_REPORT.md`

