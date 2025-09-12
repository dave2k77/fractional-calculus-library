"""
Fractional Calculus Algorithms

This module provides comprehensive implementations of fractional calculus methods
including standard, optimized, GPU-accelerated, and parallel processing versions.
"""

# Core algorithms (consolidated into optimized_methods.py)
# from .caputo import CaputoDerivative  # REMOVED - use OptimizedCaputo instead
# from .riemann_liouville import RiemannLiouvilleDerivative  # REMOVED - use OptimizedRiemannLiouville instead
# from .grunwald_letnikov import GrunwaldLetnikovDerivative  # REMOVED -
# use OptimizedGrunwaldLetnikov instead

# Optimized algorithms (consolidated - PRIMARY implementations)
from .optimized_methods import (
    OptimizedRiemannLiouville,
    OptimizedCaputo,
    OptimizedGrunwaldLetnikov,
    OptimizedFractionalMethods,
    AdvancedFFTMethods,
    L1L2Schemes,
    optimized_riemann_liouville,
    optimized_caputo,
    optimized_grunwald_letnikov,
)

# GPU-optimized algorithms
from .gpu_optimized_methods import (
    GPUConfig,
    GPUOptimizedRiemannLiouville,
    GPUOptimizedCaputo,
    GPUOptimizedGrunwaldLetnikov,
    MultiGPUManager,
    JAXAutomaticDifferentiation,
    JAXOptimizer,
    gpu_optimized_riemann_liouville,
    gpu_optimized_caputo,
    gpu_optimized_grunwald_letnikov,
    benchmark_gpu_vs_cpu,
    optimize_fractional_derivative_jax,
    vectorize_fractional_derivatives,
)

# Parallel-optimized algorithms (now consolidated in optimized_methods)
from .optimized_methods import (
    ParallelConfig,
    ParallelOptimizedRiemannLiouville,
    ParallelOptimizedCaputo,
    ParallelOptimizedGrunwaldLetnikov,
    parallel_optimized_riemann_liouville,
    parallel_optimized_caputo,
    parallel_optimized_grunwald_letnikov,
)

# Advanced methods
from .advanced_methods import (
    WeylDerivative,
    MarchaudDerivative,
    HadamardDerivative,
    ReizFellerDerivative,
    AdomianDecomposition,
)

# Advanced optimized methods (now consolidated in advanced_methods)
from .advanced_methods import (
    OptimizedWeylDerivative,
    OptimizedMarchaudDerivative,
    OptimizedHadamardDerivative,
    OptimizedReizFellerDerivative,
    OptimizedAdomianDecomposition,
    optimized_weyl_derivative,
    optimized_marchaud_derivative,
    optimized_hadamard_derivative,
    optimized_reiz_feller_derivative,
)

# Special optimized methods (now consolidated in special_methods)
from .special_methods import (
    SpecialOptimizedWeylDerivative,
    SpecialOptimizedMarchaudDerivative,
    SpecialOptimizedReizFellerDerivative,
    UnifiedSpecialMethods,
    special_optimized_weyl_derivative,
    special_optimized_marchaud_derivative,
    special_optimized_reiz_feller_derivative,
    unified_special_derivative,
)

# FFT methods (consolidated into optimized_methods.py)
# from .fft_methods import FFTFractionalMethods  # REMOVED - use
# AdvancedFFTMethods instead

# L1/L2 schemes (consolidated into optimized_methods.py)
# from .L1_L2_schemes import L1L2Schemes  # REMOVED - now imported from
# optimized_methods

# Define what gets imported with "from algorithms import *"
__all__ = [
    # Core algorithms (consolidated - use optimized versions)
    # "CaputoDerivative",  # REMOVED
    # "RiemannLiouvilleDerivative",  # REMOVED
    # "GrunwaldLetnikovDerivative",  # REMOVED
    # Optimized algorithms (PRIMARY implementations)
    "OptimizedRiemannLiouville",
    "OptimizedCaputo",
    "OptimizedGrunwaldLetnikov",
    "OptimizedFractionalMethods",
    "AdvancedFFTMethods",
    "L1L2Schemes",
    "optimized_riemann_liouville",
    "optimized_caputo",
    "optimized_grunwald_letnikov",
    # GPU-optimized algorithms
    "GPUConfig",
    "GPUOptimizedRiemannLiouville",
    "GPUOptimizedCaputo",
    "GPUOptimizedGrunwaldLetnikov",
    "MultiGPUManager",
    "JAXAutomaticDifferentiation",
    "JAXOptimizer",
    "gpu_optimized_riemann_liouville",
    "gpu_optimized_caputo",
    "gpu_optimized_grunwald_letnikov",
    "benchmark_gpu_vs_cpu",
    "optimize_fractional_derivative_jax",
    "vectorize_fractional_derivatives",
    # Parallel-optimized algorithms
    "ParallelConfig",
    "ParallelOptimizedRiemannLiouville",
    "ParallelOptimizedCaputo",
    "ParallelOptimizedGrunwaldLetnikov",
    "ParallelLoadBalancer",
    "ParallelPerformanceMonitor",
    "NumbaOptimizer",
    "NumbaFractionalKernels",
    "NumbaParallelManager",
    "parallel_optimized_riemann_liouville",
    "parallel_optimized_caputo",
    "parallel_optimized_grunwald_letnikov",
    "benchmark_parallel_vs_serial",
    "optimize_parallel_parameters",
    "memory_efficient_caputo",
    "block_processing_kernel",
    # Advanced methods
    "WeylDerivative",
    "MarchaudDerivative",
    "HadamardDerivative",
    "ReizFellerDerivative",
    "AdomianDecomposition",
    # Advanced optimized methods
    "OptimizedWeylDerivative",
    "OptimizedMarchaudDerivative",
    "OptimizedHadamardDerivative",
    "OptimizedReizFellerDerivative",
    "OptimizedAdomianDecomposition",
    "optimized_weyl_derivative",
    "optimized_marchaud_derivative",
    "optimized_hadamard_derivative",
    "optimized_reiz_feller_derivative",
    "optimized_adomian_decomposition",
    # FFT methods (consolidated)
    # "FFTFractionalMethods",  # REMOVED - use AdvancedFFTMethods instead
    # L1/L2 schemes (consolidated)
    # "L1L2Schemes"  # REMOVED - now imported from optimized_methods
]
