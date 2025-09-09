# HPFRACC Consolidation Plan

## Current State Analysis

### Multiple Spectral Autograd Implementations
- `spectral_autograd.py` (652 lines, 05:00) - **MAIN/CANONICAL**
- `spectral_autograd_production.py` (448 lines, 04:31) - Production optimizations
- `spectral_autograd_robust.py` (421 lines, 04:48) - MKL FFT error handling
- `spectral_autograd_corrected.py` (309 lines, 04:19) - Mathematical corrections

### Test Coverage Issues
- **Overall Coverage**: 45% (not 85% as claimed)
- **Untested Modules**: Analytics, Benchmarks, Advanced ML features
- **Well-tested**: Core algorithms, basic ML layers, special functions

## Phase 1: Foundation Consolidation

### 1.1 Spectral Autograd Consolidation
**Goal**: Merge the best features from all implementations into one canonical version

**Strategy**:
1. **Keep**: `spectral_autograd.py` as the base (most complete, most recent)
2. **Merge from others**:
   - Production optimizations from `spectral_autograd_production.py`
   - MKL FFT error handling from `spectral_autograd_robust.py`
   - Mathematical corrections from `spectral_autograd_corrected.py`
3. **Remove**: Duplicate implementations after merging
4. **Test**: Ensure merged version works correctly

### 1.2 Core Functionality Audit
**Goal**: Identify what's actually being used vs. what's dead code

**Strategy**:
1. **Check imports**: What's actually imported in `__init__.py` files
2. **Check usage**: What's used in tests and examples
3. **Identify dead code**: Remove unused implementations
4. **Document**: What's core vs. experimental vs. deprecated

### 1.3 Testing Strategy
**Goal**: Establish realistic testing standards and improve coverage

**Strategy**:
1. **Core functionality**: Must have 80%+ coverage
2. **Supporting modules**: Target 60%+ coverage
3. **Experimental features**: Document as experimental, lower coverage OK
4. **Testing requirements**: New code must include tests

## Phase 2: Testing Implementation

### 2.1 Priority Testing Order
1. **Core algorithms** (already well-tested)
2. **Spectral autograd** (critical for ML functionality)
3. **ML layers and neural networks** (user-facing)
4. **Analytics module** (currently 0% coverage)
5. **Benchmarks module** (currently 0% coverage)
6. **Advanced ML features** (stochastic, probabilistic)

### 2.2 Testing Standards
- **Unit tests**: Individual functions and classes
- **Integration tests**: End-to-end workflows
- **Performance tests**: Benchmarking and regression testing
- **Error handling tests**: Edge cases and error conditions

## Phase 3: Development Workflow

### 3.1 Code Review Requirements
- All new code must include tests
- Coverage must not decrease
- Performance regressions must be justified
- Documentation must be updated

### 3.2 Release Standards
- Core functionality must be tested
- Performance benchmarks must pass
- Documentation must be complete
- No known critical bugs

## Immediate Actions

1. **Consolidate spectral autograd implementations**
2. **Audit and remove dead code**
3. **Establish testing requirements**
4. **Create development workflow**
5. **Set realistic coverage targets**

## Success Metrics

- **Test coverage**: 60%+ overall, 80%+ for core functionality
- **Code quality**: No duplicate implementations
- **Documentation**: Accurate coverage claims
- **Development**: Proper testing discipline
