"""
High-Performance Fractional Calculus Library (hpfracc)

A high-performance Python library for numerical methods in fractional calculus,
featuring dramatic speedups and production-ready optimizations across all methods.

This library provides optimized implementations of:
- Core fractional derivatives: Caputo, Riemann-Liouville, Grünwald-Letnikov
- Advanced methods: Weyl, Marchaud, Hadamard, Reiz-Feller derivatives
- Special methods: Fractional Laplacian, Fractional Fourier Transform, Fractional Z-Transform, Fractional Mellin Transform
- GPU acceleration via JAX, PyTorch, and CuPy
- Parallel computing via NUMBA
"""

__version__ = "3.0.0"
__author__ = "Davian R. Chin"
__email__ = "d.r.chin@pgr.reading.ac.uk"
__affiliation__ = "Department of Biomedical Engineering, University of Reading"

# Keep the top-level package import extremely lightweight to avoid importing
# optional heavy dependencies (e.g., torch, jax, numba) during package import.
# Users should import symbols from submodules explicitly, e.g.:
#   from hpfracc.algorithms.optimized_methods import OptimizedCaputo

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__affiliation__",
    "OptimizedRiemannLiouville",
    "OptimizedCaputo",
    "OptimizedGrunwaldLetnikov",
    "FractionalOrder",
    "WeylDerivative",
    "MarchaudDerivative",
    "FractionalLaplacian",
    "FractionalFourierTransform",
    "RiemannLiouvilleIntegral",
    "CaputoIntegral",
    "CaputoFabrizioDerivative",
    "AtanganaBaleanuDerivative",
    "Caputo",
    "HadamardDerivative",
    "ReizFellerDerivative",
    "FractionalZTransform",
    "FractionalMellinTransform",
]
