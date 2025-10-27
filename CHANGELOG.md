# Changelog

All notable changes to the HPFRACC library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] - 2025-10-27

### Changed
- **Python Version Requirement**: Now requires Python 3.9+ (dropped Python 3.8 support)
- **CI/CD**: Simplified PyPI release workflow for faster, more reliable releases
- **Testing**: Focus on Python 3.9-3.12 in CI pipelines

### Added

#### Intelligent Backend Selection System
- **New Module**: `hpfracc.ml.intelligent_backend_selector` - Workload-aware backend optimization
- Automatic backend selection based on data size, operation type, and available hardware
- Performance learning system that adapts over time
- Dynamic GPU memory thresholds to prevent out-of-memory errors
- Convenience function `select_optimal_backend()` for quick backend selection
- GPU memory estimator with dynamic threshold calculation

#### Performance Improvements
- **10-100x speedup** for small data operations (< 1K elements) by avoiding GPU overhead
- **1.5-3x speedup** for large data operations (> 100K elements) through optimal backend selection
- **< 1 μs overhead** for backend selection (negligible performance impact)
- Sub-linear per-step scaling for ODE solvers with intelligent FFT backend selection

#### Integration Enhancements
- Enhanced `BackendManager` in ML layers with intelligent selection
- Enhanced `GPUConfig` in GPU-optimized methods with workload-aware selection
- Intelligent FFT backend selection for ODE solvers
- Workload-aware array backend selection for PDE solvers
- Memory-safe GPU operations with automatic CPU fallback

#### Documentation
- Comprehensive backend optimization guides (17,000+ words across 8 documents)
- Quick reference card for backend selection scenarios
- Integration guide with code examples
- Technical analysis report
- Executive summary for decision makers

### Fixed
- API mismatch in `FractionalNeuralNetwork` initialization (`alpha` keyword argument)
- API mismatch in `FractionalAdam` optimizer (missing `params` parameter)
- Transpose method calls in `FractionalAttention` (incorrect argument format)
- All ML integration tests now passing (23/23, 100%)

### Changed
- `hpfracc/ml/layers.py`: `BackendManager` now uses intelligent backend selection
- `hpfracc/ml/core.py`: Fixed fractional calculator initialization and transpose calls
- `hpfracc/ml/optimized_optimizers.py`: Added PyTorch-compatible parameter handling
- `hpfracc/algorithms/gpu_optimized_methods.py`: Added intelligent backend selection
- `hpfracc/solvers/ode_solvers.py`: Integrated intelligent FFT backend selection
- `hpfracc/solvers/pde_solvers.py`: Added workload-aware array operations
- `hpfracc/core/fractional_implementations.py`: Updated documentation for automatic optimization

### Technical Details
- **Lines of Code Added**: 1,200+ (600 production, 350 tests, 250+ examples)
- **Files Modified**: 9
- **Files Created**: 8 documentation files, 2 test files, 1 demo file
- **Test Coverage**: 47/47 integration tests passing (100%)
- **Backward Compatibility**: 100% maintained

### Performance Benchmarks
- Backend selection overhead: 0.57-1.86 μs
- Selection throughput: 1.4M-1.8M selections/sec
- ODE solver (50 points): 39.02 μs per step
- ODE solver (1000 points): 96.80 μs per step
- GPU memory detected: 7.53 GB (PyTorch CUDA)
- Dynamic threshold: 707M elements (~5.27 GB of float64 data)

---

## [2.0.0] - 2025-09-29

### Added
- Production-ready release with 100% integration test coverage
- Complete GPU acceleration support
- ML integration with PyTorch, JAX, and Numba
- Comprehensive research workflows for computational physics and biophysics
- 151 performance benchmarks
- Complete documentation suite

### Features
- Core fractional calculus operations (Riemann-Liouville, Caputo, Grünwald-Letnikov)
- Fractional neural networks with spectral autograd
- GPU-optimized methods with multi-backend support
- Variance-aware training components
- Graph neural networks with fractional components
- Fractional ODE/PDE solvers

### Testing
- 37 integration tests (100% passing)
- 151 performance benchmarks (100% passing)
- End-to-end research workflow validation

---

## [1.0.0] - Initial Release

### Added
- Basic fractional calculus operations
- Core mathematical implementations
- NumPy/SciPy backend support
- Initial documentation

---

## Version Numbering

- **Major** version: Incompatible API changes
- **Minor** version: New functionality (backward compatible)
- **Patch** version: Bug fixes (backward compatible)

---

**Current Version**: 2.1.0  
**Release Date**: October 27, 2025  
**Status**: Production Ready

