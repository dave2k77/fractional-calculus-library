"""
Fractional Calculus Library

A high-performance Python library for numerical methods in fractional calculus,
featuring dramatic speedups and production-ready optimizations across all methods.

This library provides optimized implementations of:
- Core fractional derivatives: Caputo, Riemann-Liouville, Grünwald-Letnikov
- Advanced methods: Weyl, Marchaud, Hadamard, Reiz-Feller derivatives
- Special methods: Fractional Laplacian, Fractional Fourier Transform, Fractional Z-Transform
- GPU acceleration via JAX
- Parallel computing via NUMBA
"""

__version__ = "0.1.0"
__author__ = "David"
__email__ = "dave2k77@gmail.com"

# Import core optimized methods for easy access
try:
    from .algorithms.optimized_methods import (
        OptimizedRiemannLiouville,
        OptimizedCaputo,
        OptimizedGrunwaldLetnikov,
        optimized_riemann_liouville,
        optimized_caputo,
        optimized_grunwald_letnikov,
    )
except ImportError:
    pass

# Import advanced methods
try:
    from .algorithms.advanced_methods import (
        WeylDerivative,
        MarchaudDerivative,
        HadamardDerivative,
        ReizFellerDerivative,
        AdomianDecomposition,
    )
except ImportError:
    pass

# Import optimized advanced methods
try:
    from .algorithms.advanced_optimized_methods import (
        OptimizedWeylDerivative,
        OptimizedMarchaudDerivative,
        OptimizedHadamardDerivative,
        OptimizedReizFellerDerivative,
        OptimizedAdomianDecomposition,
        optimized_weyl_derivative,
        optimized_marchaud_derivative,
        optimized_hadamard_derivative,
        optimized_reiz_feller_derivative,
        optimized_adomian_decomposition,
    )
except ImportError:
    pass

# Import special methods
try:
    from .algorithms.special_methods import (
        FractionalLaplacian,
        FractionalFourierTransform,
        FractionalZTransform,
        fractional_laplacian,
        fractional_fourier_transform,
        fractional_z_transform,
    )
except ImportError:
    pass

# Import special optimized methods
try:
    from .algorithms.special_optimized_methods import (
        SpecialOptimizedWeylDerivative,
        SpecialOptimizedMarchaudDerivative,
        SpecialOptimizedReizFellerDerivative,
        UnifiedSpecialMethods,
        special_optimized_weyl_derivative,
        special_optimized_marchaud_derivative,
        special_optimized_reiz_feller_derivative,
        unified_special_derivative,
    )
except ImportError:
    pass

# Import core definitions
try:
    from .core.definitions import FractionalOrder
except ImportError:
    pass

# Convenience imports for common use cases
__all__ = [
    # Core optimized methods
    "OptimizedRiemannLiouville",
    "OptimizedCaputo", 
    "OptimizedGrunwaldLetnikov",
    "optimized_riemann_liouville",
    "optimized_caputo",
    "optimized_grunwald_letnikov",
    
    # Advanced methods
    "WeylDerivative",
    "MarchaudDerivative",
    "HadamardDerivative",
    "ReizFellerDerivative",
    "AdomianDecomposition",
    
    # Optimized advanced methods
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
    
    # Special methods
    "FractionalLaplacian",
    "FractionalFourierTransform",
    "FractionalZTransform",
    "fractional_laplacian",
    "fractional_fourier_transform",
    "fractional_z_transform",
    
    # Special optimized methods
    "SpecialOptimizedWeylDerivative",
    "SpecialOptimizedMarchaudDerivative",
    "SpecialOptimizedReizFellerDerivative",
    "UnifiedSpecialMethods",
    "special_optimized_weyl_derivative",
    "special_optimized_marchaud_derivative",
    "special_optimized_reiz_feller_derivative",
    "unified_special_derivative",
    
    # Core definitions
    "FractionalOrder",
]
