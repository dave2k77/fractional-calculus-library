# Failing Tests Analysis

**Date**: 2025-09-30  
**Total Failing Tests**: 765  
**Analysis Scope**: All tests except large integration tests

## Summary by Category

### Category 1: Missing Implementations (High Priority)
**Count**: ~50 failures
**Impact**: Core functionality gaps

#### 1.1 Advanced Fractional Derivatives
- `CaputoFabrizioDerivative` - Not implemented
- `AtanganaBaleanuDerivative` - Not implemented  
- `FractionalLaplacian` - Not implemented
- `FractionalFourierTransform` - Not implemented
- `WeylDerivative` - Incomplete implementation
- `MarchaudDerivative` - Incomplete implementation
- `HadamardDerivative` - Missing `order` parameter initialization
- `ReizFellerDerivative` - Incomplete implementation
- `MillerRossDerivative` - Missing `order` attribute
- `ParallelOptimizedRiemannLiouville` - Incomplete implementation
- `ParallelOptimizedCaputo` - Incomplete implementation

**Files affected**:
- `hpfracc/core/fractional_implementations.py`
- `tests/test_core/test_fractional_implementations_comprehensive.py`
- `tests/test_algorithms/test_advanced_methods_phase2.py`

**Recommended action**: Implement missing derivatives or mark as experimental/future features

---

### Category 2: API Mismatches (Medium Priority)
**Count**: ~150 failures
**Impact**: Test suite maintenance

#### 2.1 ML Adapter/Backend Issues
- `get_adapter()` function not found (replaced with specific adapters)
- `_spec_available` attribute missing
- `BackendType.NUMPY` doesn't exist (should be `NUMBA`)
- Mock issues with `backends.torch`, `backends.jax`
- `optuna` attribute not in `ml.core`

**Files affected**:
- `tests/test_ml/test_adapter_mocks.py`
- `tests/test_ml/test_adapters_lazy_imports.py`
- `tests/test_ml/test_core_numpy_lane.py`
- `tests/test_ml/test_backends_and_adapters.py`

**Recommended action**: Update tests to match current API, remove deprecated API tests

#### 2.2 GNN Layer API Issues
- `BaseFractionalGNNLayer` is abstract, cannot instantiate directly
- Initialization parameters mismatch (e.g., `input_dim` vs `in_channels`)
- `_compute_fractional_derivative` method not accessible (private/renamed)
- `FractionalGraphPooling`, `FractionalGraphAttention` parameter mismatches

**Files affected**:
- `tests/test_ml/test_gnn_layers_90_percent.py`
- `tests/test_ml/test_gnn_comprehensive.py`

**Recommended action**: Update tests to use concrete GNN classes, fix parameter names

#### 2.3 Algorithm Function Passing Issues
- Tests passing functions to compute methods when numerical arrays expected
- `compute_function()` vs `compute()` API confusion

**Files affected**:
- `tests/test_algorithms/test_algorithms_functionality.py`

**Recommended action**: Clarify API documentation, update tests

---

### Category 3: Validation/Benchmark Issues (Medium Priority)
**Count**: ~400 failures
**Impact**: Quality assurance gaps

#### 3.1 Benchmark Result Structure
- Expected keys missing: `accuracy_results`, `performance_results`
- Return type mismatches in benchmark functions
- Convergence test API changes

**Files affected**:
- `tests/test_additional_coverage.py`
- `tests/test_validation/*.py` (multiple files)
- `hpfracc/validation/benchmarks.py`
- `hpfracc/validation/convergence_tests.py`

**Recommended action**: Standardize benchmark return structures, update validation API

#### 3.2 Analytical Solutions
- `TrigonometricSolution.get_solution()` failures
- `get_analytical_solution()` utility function issues
- `validate_against_analytical()` API changes

**Files affected**:
- `tests/test_validation/test_analytical_solutions_*.py`
- `hpfracc/validation/analytical_solutions.py`

**Recommended action**: Fix analytical solution implementations, update utility functions

---

### Category 4: Import/Export Issues (Low Priority)
**Count**: ~50 failures
**Impact**: Module structure

#### 4.1 Missing __all__ Exports
- `OptimizedRiemannLiouville` not in `__all__`
- Module attribute count mismatch (4 vs expected 20)

**Files affected**:
- `tests/test_main_init_comprehensive.py`
- `hpfracc/__init__.py`

**Recommended action**: Review and update `__all__` declarations

---

### Category 5: Test Infrastructure Issues (Low Priority)
**Count**: ~100 failures
**Impact**: Test suite reliability

#### 5.1 Mock/Patch Issues
- Attempting to mock attributes that don't exist
- Mock spec issues with backend modules
- Private method mocking failures

**Files affected**:
- Multiple test files using `unittest.mock`

**Recommended action**: Update mocking strategies, use proper test doubles

#### 5.2 Coverage Measurement
- Coverage test assertion failures (expected 0, got 1)
- Test assumes no coverage for new tests

**Files affected**:
- `tests/test_coverage_measurement.py`

**Recommended action**: Update coverage expectations

---

### Category 6: Utils Module (Low Priority)
**Count**: ~15 failures
**Impact**: Utility functionality

#### 6.1 Plotting/Error Analysis
- Attribute errors in plotting functions
- Error analysis function issues

**Files affected**:
- `tests/test_utils.py`
- `hpfracc/utils/plotting.py`
- `hpfracc/utils/error_analysis.py`

**Recommended action**: Fix utility implementations

---

## Prioritized Fix Plan

### Phase 1: Quick Wins (Target: -200 failures)
1. Update all API mismatch tests (~150 tests)
   - Fix adapter/backend API tests
   - Update GNN layer tests
   - Fix BackendType.NUMPY → NUMBA

2. Fix validation benchmark structure (~50 tests)
   - Standardize return types
   - Update expected keys

### Phase 2: Core Implementations (Target: -50 failures)
1. Fix existing derivative implementations
   - Add `order` attribute to MillerRossDerivative
   - Fix HadamardDerivative initialization
   - Complete partial implementations

2. Mark unimplemented as TODO
   - CaputoFabrizio, AtanganaBaleanu
   - FractionalLaplacian, FractionalFourierTransform
   - Skip tests with proper markers

### Phase 3: Test Infrastructure (Target: -150 failures)
1. Update mocking strategies
2. Fix import/export issues
3. Update coverage tests
4. Fix utils module tests

### Phase 4: Validation Suite (Target: -350 failures)
1. Fix analytical solutions
2. Update convergence tests  
3. Standardize benchmark APIs
4. Update integration tests

---

## Metrics

**Current State**:
- Total tests: ~3,473
- Passing: 2,630 (75.7%)
- Failing: 765 (22.0%)
- Skipped: 80 (2.3%)

**Target State** (After fixes):
- Passing: >3,300 (>95%)
- Failing: <100 (<3%)
- Skipped: ~70 (2%)

**Coverage Goals**:
- Current: ~18% (ML module)
- Target: 50%+ overall

---

## Implementation Strategy

1. **Start with Phase 1** (API mismatches) - These are test-only changes, low risk
2. **Move to Phase 2** (Missing implementations) - Add missing features or skip gracefully
3. **Continue with Phase 3** (Test infrastructure) - Improve test quality
4. **Finish with Phase 4** (Validation suite) - Most complex, needs careful review

Each phase should be completed, committed, and tested before moving to the next.

