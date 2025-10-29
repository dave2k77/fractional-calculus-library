# JAX-GPU Fixes Plan for hpfracc v3.0.0

## Current Issues Identified

### 1. PJRT Plugin Conflict
**Error**: `ALREADY_EXISTS: PJRT_Api already exists for device type cuda`
**Cause**: Multiple JAX plugins trying to register the same CUDA backend
**Impact**: Prevents JAX from using GPU, falls back to CPU

### 2. CuDNN Version Mismatch  
**Error**: `Loaded runtime CuDNN library: 9.10.2 but source was compiled with: 9.12.0`
**Cause**: JAXlib compiled with newer CuDNN than system has
**Impact**: Performance degradation, potential numerical issues

### 3. Environment Variable Conflicts
**Issue**: Setting `JAX_PLATFORM_NAME=gpu` manually causes plugin conflicts
**Impact**: Tests fail when trying to force GPU usage

## System Information
- **CUDA Version**: 13.0
- **JAX Version**: 0.8.0  
- **JAXlib Version**: 0.8.0
- **GPU**: NVIDIA GeForce RTX 5070
- **CuDNN Runtime**: 9.10.2
- **CuDNN Compiled**: 9.12.0

## Fix Strategy

### Phase 1: Environment and Plugin Management

#### 1.1 Update JAX-GPU Setup
**File**: `hpfracc/jax_gpu_setup.py`

**Changes**:
- Implement proper plugin conflict resolution
- Add CuDNN version checking and warnings
- Implement graceful fallback with clear messaging
- Add device memory management

**Key Features**:
```python
def setup_jax_gpu_safe() -> bool:
    """Safe JAX GPU setup with conflict resolution"""
    # Clear any existing PJRT plugins
    # Check CuDNN compatibility
    # Initialize JAX with proper error handling
    # Return success status with detailed info
```

#### 1.2 Environment Variable Management
**Strategy**: 
- Remove manual `JAX_PLATFORM_NAME` setting
- Let JAX auto-detect GPU
- Use `JAX_ENABLE_XLA=1` for better compatibility
- Set `CUDA_VISIBLE_DEVICES` if needed

### Phase 2: CuDNN Compatibility Fixes

#### 2.1 CuDNN Version Detection
**Implementation**:
- Check installed CuDNN version
- Compare with JAXlib requirements
- Provide upgrade/downgrade recommendations
- Implement compatibility warnings

#### 2.2 Fallback Strategies
**Options**:
1. **Upgrade CuDNN**: Install CuDNN 9.12.0+ to match JAXlib
2. **Downgrade JAXlib**: Use JAXlib version compatible with CuDNN 9.10.2
3. **CPU Fallback**: Graceful degradation with clear messaging

### Phase 3: Test Framework Updates

#### 3.1 GPU-Aware Testing
**Strategy**:
- Detect GPU availability before running tests
- Skip GPU-specific tests if GPU unavailable
- Run both CPU and GPU versions when possible
- Clear error messages for GPU failures

#### 3.2 Test Environment Setup
**Implementation**:
```python
def setup_test_environment():
    """Setup test environment with GPU detection"""
    gpu_available = setup_jax_gpu_safe()
    if gpu_available:
        print("✅ Running tests with GPU acceleration")
    else:
        print("⚠️  Running tests with CPU fallback")
    return gpu_available
```

### Phase 4: Library Integration

#### 4.1 Intelligent Backend Selection
**File**: `hpfracc/ml/intelligent_backend_selector.py`

**Updates**:
- Prioritize JAX-GPU when available
- Implement proper fallback chain: JAX-GPU → JAX-CPU → PyTorch → NumPy
- Add GPU memory management
- Cache backend selection results

#### 4.2 Performance Monitoring
**Implementation**:
- Monitor GPU memory usage
- Track performance differences between backends
- Implement automatic backend switching based on workload

## Implementation Steps

### Step 1: Fix JAX-GPU Setup (Priority: HIGH)

1. **Update `hpfracc/jax_gpu_setup.py`**:
   - Implement plugin conflict resolution
   - Add CuDNN compatibility checking
   - Improve error handling and messaging

2. **Test JAX-GPU functionality**:
   - Verify GPU detection works
   - Test basic JAX operations on GPU
   - Confirm fallback to CPU works

### Step 2: Update Test Framework (Priority: HIGH)

1. **Modify test execution**:
   - Remove `JAX_PLATFORM_NAME=cpu` from test commands
   - Let JAX auto-detect GPU
   - Add GPU availability checks

2. **Update failing tests**:
   - Fix tests that assume CPU-only execution
   - Add GPU-specific test cases
   - Implement proper error handling

### Step 3: CuDNN Compatibility (Priority: MEDIUM)

1. **Investigate CuDNN upgrade**:
   - Check if CuDNN 9.12.0+ is available
   - Test compatibility with current system
   - Document installation process

2. **Implement compatibility warnings**:
   - Detect version mismatches
   - Provide clear upgrade instructions
   - Implement graceful degradation

### Step 4: Performance Optimization (Priority: LOW)

1. **GPU memory management**:
   - Implement memory pooling
   - Add memory usage monitoring
   - Optimize for large computations

2. **Backend selection optimization**:
   - Cache backend decisions
   - Implement workload-based selection
   - Add performance benchmarking

## Testing Commands (Updated)

### GPU-First Testing
```bash
# Let JAX auto-detect GPU (recommended)
python -m pytest tests/test_core/test_derivatives_expanded.py -v

# Force GPU usage (may cause conflicts)
JAX_PLATFORM_NAME=gpu python -m pytest tests/test_core/test_derivatives_expanded.py -v

# Force CPU fallback (for debugging)
JAX_PLATFORM_NAME=cpu python -m pytest tests/test_core/test_derivatives_expanded.py -v
```

### GPU Detection Testing
```bash
# Test JAX-GPU setup
python -c "from hpfracc.jax_gpu_setup import setup_jax_gpu_safe, get_jax_info; print('GPU Available:', setup_jax_gpu_safe()); print('JAX Info:', get_jax_info())"
```

## Success Criteria

### Phase 1 Success:
- ✅ JAX-GPU setup works without plugin conflicts
- ✅ Clear error messages for GPU issues
- ✅ Graceful fallback to CPU when GPU unavailable

### Phase 2 Success:
- ✅ CuDNN compatibility issues resolved or clearly documented
- ✅ Performance tests show GPU acceleration working
- ✅ No test failures due to GPU/CPU differences

### Phase 3 Success:
- ✅ All tests pass with GPU auto-detection
- ✅ GPU-specific tests run when GPU available
- ✅ CPU fallback works seamlessly

### Phase 4 Success:
- ✅ Intelligent backend selector prioritizes GPU
- ✅ Performance monitoring shows GPU benefits
- ✅ Memory management prevents GPU OOM errors

## Risk Mitigation

### Risk: CuDNN upgrade breaks other libraries
**Mitigation**: 
- Test in isolated environment first
- Document rollback procedure
- Implement compatibility checking

### Risk: GPU tests are flaky due to memory issues
**Mitigation**:
- Implement proper memory cleanup
- Add memory usage monitoring
- Use smaller test datasets for GPU tests

### Risk: Plugin conflicts persist
**Mitigation**:
- Implement plugin cleanup before JAX initialization
- Use environment isolation
- Document workarounds for different systems

## Next Steps

1. **Immediate**: Update `hpfracc/jax_gpu_setup.py` with conflict resolution
2. **Short-term**: Remove `JAX_PLATFORM_NAME=cpu` from test commands
3. **Medium-term**: Investigate CuDNN upgrade options
4. **Long-term**: Implement comprehensive GPU performance monitoring

## Files to Modify

### High Priority:
- `hpfracc/jax_gpu_setup.py` - Core GPU setup fixes
- Test execution scripts - Remove CPU forcing
- `hpfracc/ml/intelligent_backend_selector.py` - GPU prioritization

### Medium Priority:
- `hpfracc/algorithms/optimized_methods.py` - GPU optimization
- `hpfracc/ml/gpu_optimization.py` - GPU memory management
- Documentation files - Update GPU setup instructions

### Low Priority:
- Performance benchmarking scripts
- Memory monitoring utilities
- GPU-specific test cases
