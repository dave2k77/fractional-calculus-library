"""
Algorithms Module

This module provides numerical algorithms for computing fractional derivatives
and integrals, including various definitions and optimization methods.
"""

# Import main algorithm classes
from .caputo import (
    CaputoDerivative,
    JAXCaputoDerivative,
    caputo_derivative,
    caputo_derivative_jax,
    caputo_derivative_numba,
    caputo_l1_numba,
    caputo_direct_numba
)

from .riemann_liouville import (
    RiemannLiouvilleDerivative,
    JAXRiemannLiouvilleDerivative,
    riemann_liouville_derivative,
    riemann_liouville_derivative_jax,
    riemann_liouville_derivative_numba,
    riemann_liouville_grunwald_numba,
    riemann_liouville_direct_numba
)

from .grunwald_letnikov import (
    GrunwaldLetnikovDerivative,
    JAXGrunwaldLetnikovDerivative,
    grunwald_letnikov_derivative,
    grunwald_letnikov_derivative_jax,
    grunwald_letnikov_derivative_numba,
    grunwald_letnikov_direct_numba,
    grunwald_letnikov_short_memory_numba,
    grunwald_coefficient_numba
)

from .fft_methods import (
    FFTFractionalMethods,
    JAXFFTFractionalMethods,
    fft_fractional_derivative,
    fft_fractional_integral,
    fft_fractional_derivative_jax,
    fft_fractional_derivative_numba,
    fft_convolution_derivative_numba,
    fft_spectral_derivative_numba
)

from .L1_L2_schemes import (
    L1L2Schemes,
    JAXL1L2Schemes,
    solve_time_fractional_pde,
    solve_time_fractional_pde_numba,
    l1_scheme_numba,
    l2_scheme_numba
)

# Define what gets imported with "from algorithms import *"
__all__ = [
    # Caputo derivative algorithms
    'CaputoDerivative',
    'JAXCaputoDerivative',
    'caputo_derivative',
    'caputo_derivative_jax',
    'caputo_derivative_numba',
    'caputo_l1_numba',
    'caputo_direct_numba',
    
    # Riemann-Liouville derivative algorithms
    'RiemannLiouvilleDerivative',
    'JAXRiemannLiouvilleDerivative',
    'riemann_liouville_derivative',
    'riemann_liouville_derivative_jax',
    'riemann_liouville_derivative_numba',
    'riemann_liouville_grunwald_numba',
    'riemann_liouville_direct_numba',
    
    # Grünwald-Letnikov derivative algorithms
    'GrunwaldLetnikovDerivative',
    'JAXGrunwaldLetnikovDerivative',
    'grunwald_letnikov_derivative',
    'grunwald_letnikov_derivative_jax',
    'grunwald_letnikov_derivative_numba',
    'grunwald_letnikov_direct_numba',
    'grunwald_letnikov_short_memory_numba',
    'grunwald_coefficient_numba',
    
    # FFT-based methods
    'FFTFractionalMethods',
    'JAXFFTFractionalMethods',
    'fft_fractional_derivative',
    'fft_fractional_integral',
    'fft_fractional_derivative_jax',
    'fft_fractional_derivative_numba',
    'fft_convolution_derivative_numba',
    'fft_spectral_derivative_numba',
    
    # L1/L2 schemes for time-fractional PDEs
    'L1L2Schemes',
    'JAXL1L2Schemes',
    'solve_time_fractional_pde',
    'solve_time_fractional_pde_numba',
    'l1_scheme_numba',
    'l2_scheme_numba'
]
