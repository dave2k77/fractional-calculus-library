"""
High-Performance Fractional Calculus Library (hpfracc)

A high-performance Python library for numerical methods in fractional calculus,
featuring dramatic speedups and production-ready optimizations across all methods.

This library provides optimized implementations of:
- Core fractional derivatives: Caputo, Riemann-Liouville, Grünwald-Letnikov
- Advanced methods: Weyl, Marchaud, Hadamard, Reiz-Feller derivatives
- Special methods: Fractional Laplacian, Fractional Fourier Transform, Fractional Z-Transform, Fractional Mellin Transform
- GPU acceleration via JAX
- Parallel computing via NUMBA
"""

__version__ = "0.2.0"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__affiliation__ = "Department of Biomedical Engineering, University of Reading"

# Import core optimized methods for easy access
try:
    from hpfracc.algorithms.optimized_methods import (
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
    from hpfracc.algorithms.advanced_methods import (
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
    from hpfracc.algorithms.advanced_optimized_methods import (
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
    from hpfracc.algorithms.special_methods import (
        FractionalLaplacian,
        FractionalFourierTransform,
        FractionalZTransform,
        FractionalMellinTransform,
        fractional_laplacian,
        fractional_fourier_transform,
        fractional_z_transform,
        fractional_mellin_transform,
    )
except ImportError:
    pass

# Import fractional integrals
try:
    from hpfracc.algorithms.integral_methods import (
        RiemannLiouvilleIntegral,
        CaputoIntegral,
        riemann_liouville_integral,
        caputo_integral,
        optimized_riemann_liouville_integral,
        optimized_caputo_integral,
    )
except ImportError:
    pass

# Import novel fractional derivatives
try:
    from hpfracc.algorithms.novel_derivatives import (
        CaputoFabrizioDerivative,
        AtanganaBaleanuDerivative,
        caputo_fabrizio_derivative,
        atangana_baleanu_derivative,
        optimized_caputo_fabrizio_derivative,
        optimized_atangana_baleanu_derivative,
    )
except ImportError:
    pass

# Import special optimized methods
try:
    from hpfracc.algorithms.special_optimized_methods import (
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
    from hpfracc.core.definitions import FractionalOrder
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
    "FractionalMellinTransform",
    "fractional_laplacian",
    "fractional_fourier_transform",
    "fractional_z_transform",
    "fractional_mellin_transform",
    
    # Fractional integrals
    "RiemannLiouvilleIntegral",
    "CaputoIntegral",
    "riemann_liouville_integral",
    "caputo_integral",
    "optimized_riemann_liouville_integral",
    "optimized_caputo_integral",
    
    # Novel fractional derivatives
    "CaputoFabrizioDerivative",
    "AtanganaBaleanuDerivative",
    "caputo_fabrizio_derivative",
    "atangana_baleanu_derivative",
    "optimized_caputo_fabrizio_derivative",
    "optimized_atangana_baleanu_derivative",
    
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
