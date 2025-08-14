"""
Optimisation Module

This module provides advanced optimization and parallel computing capabilities
for fractional calculus operations, including JAX GPU acceleration, NUMBA
CPU parallelization, and distributed computing strategies.
"""

# Import JAX optimizations
from .jax_implementations import (
    JAXOptimizer,
    JAXFractionalDerivatives,
    JAXAutomaticDifferentiation,
    JAXVectorization,
    JAXPerformanceMonitor,
    optimize_fractional_derivative_jax,
    compute_fractional_derivative_gpu,
    vectorize_fractional_derivatives
)

# Import NUMBA kernels
from .numba_kernels import (
    NumbaOptimizer,
    NumbaFractionalKernels,
    NumbaMemoryOptimizer,
    NumbaSpecializedKernels,
    NumbaParallelManager,
    gamma_approx,
    optimize_fractional_kernel_numba,
    compute_caputo_derivative_numba_optimized,
    compute_grunwald_letnikov_numba_optimized,
    compute_riemann_liouville_numba_optimized
)

# Import parallel computing
from .parallel_computing import (
    ParallelComputingManager,
    DistributedComputing,
    LoadBalancer,
    PerformanceOptimizer,
    ParallelFractionalComputing,
    parallel_fractional_derivative,
    optimize_parallel_parameters,
    get_system_info
)

# Define what gets imported with "from optimisation import *"
__all__ = [
    # JAX optimizations
    'JAXOptimizer',
    'JAXFractionalDerivatives',
    'JAXAutomaticDifferentiation',
    'JAXVectorization',
    'JAXPerformanceMonitor',
    'optimize_fractional_derivative_jax',
    'compute_fractional_derivative_gpu',
    'vectorize_fractional_derivatives',
    
    # NUMBA kernels
    'NumbaOptimizer',
    'NumbaFractionalKernels',
    'NumbaMemoryOptimizer',
    'NumbaSpecializedKernels',
    'NumbaParallelManager',
    'gamma_approx',
    'optimize_fractional_kernel_numba',
    'compute_caputo_derivative_numba_optimized',
    'compute_grunwald_letnikov_numba_optimized',
    'compute_riemann_liouville_numba_optimized',
    
    # Parallel computing
    'ParallelComputingManager',
    'DistributedComputing',
    'LoadBalancer',
    'PerformanceOptimizer',
    'ParallelFractionalComputing',
    'parallel_fractional_derivative',
    'optimize_parallel_parameters',
    'get_system_info'
]
