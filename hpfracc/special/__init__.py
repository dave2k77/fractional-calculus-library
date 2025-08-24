"""
Special functions for fractional calculus.

This module provides optimized implementations of special functions
that are fundamental to fractional calculus, including:
- Gamma and Beta functions
- Mittag-Leffler function
- Binomial coefficients
"""

from .gamma_beta import GammaFunction, BetaFunction, gamma, beta, log_gamma, log_beta

from .mittag_leffler import (
    MittagLefflerFunction,
    MittagLefflerMatrix,
    mittag_leffler,
    mittag_leffler_derivative,
    mittag_leffler_matrix,
    exponential,
    cosine_fractional,
    sinc_fractional,
)

from .binomial_coeffs import (
    BinomialCoefficients,
    GrunwaldLetnikovCoefficients,
    binomial,
    binomial_fractional,
    grunwald_letnikov_coefficients,
    grunwald_letnikov_weighted_coefficients,
    pascal_triangle,
    fractional_pascal_triangle,
)

__all__ = [
    # Gamma and Beta functions
    "GammaFunction",
    "BetaFunction",
    "gamma",
    "beta",
    "log_gamma",
    "log_beta",
    # Mittag-Leffler function
    "MittagLefflerFunction",
    "MittagLefflerMatrix",
    "mittag_leffler",
    "mittag_leffler_derivative",
    "mittag_leffler_matrix",
    "exponential",
    "cosine_fractional",
    "sinc_fractional",
    # Binomial coefficients
    "BinomialCoefficients",
    "GrunwaldLetnikovCoefficients",
    "binomial",
    "binomial_fractional",
    "grunwald_letnikov_coefficients",
    "grunwald_letnikov_weighted_coefficients",
    "pascal_triangle",
    "fractional_pascal_triangle",
]
