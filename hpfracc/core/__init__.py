"""
Core Module

This module provides the fundamental components of the HPFRACC library:
- Fractional calculus definitions and types
- Fractional derivatives and integrals
- Core utilities and helper functions
"""

from .definitions import (
    FractionalOrder,
    FractionalType,
    FractionalMethod,
    FractionalBackend,
    validate_fractional_order
)

from .derivatives import (
    create_fractional_derivative,
    riemann_liouville_derivative,
    caputo_derivative,
    grunwald_letnikov_derivative,
    weyl_derivative,
    marchaud_derivative,
    hadamard_derivative,
    fractional_derivative_properties,
    validate_fractional_derivative
)

from .integrals import (
    create_fractional_integral,
    RiemannLiouvilleIntegral,
    CaputoIntegral,
    WeylIntegral,
    HadamardIntegral,
    analytical_fractional_integral,
    trapezoidal_fractional_integral,
    simpson_fractional_integral,
    fractional_integral_properties,
    validate_fractional_integral
)

from .utilities import (
    # Mathematical utilities
    factorial_fractional,
    binomial_coefficient,
    pochhammer_symbol,
    hypergeometric_series,
    bessel_function_first_kind,
    modified_bessel_function_first_kind,
    
    # Type checking and validation
    validate_fractional_order,
    validate_function,
    validate_tensor_input,
    
    # Performance monitoring
    timing_decorator,
    memory_usage_decorator,
    PerformanceMonitor,
    
    # Error handling
    FractionalCalculusError,
    ConvergenceError,
    ValidationError,
    safe_divide,
    check_numerical_stability,
    
    # Common mathematical operations
    vectorize_function,
    normalize_array,
    smooth_function,
    fractional_power,
    fractional_exponential,
    
    # Configuration utilities
    get_default_precision,
    set_default_precision,
    get_available_methods,
    get_method_properties,
    
    # Logging utilities
    setup_logging,
    get_logger
)

__all__ = [
    # Definitions
    'FractionalOrder',
    'FractionalType',
    'FractionalMethod',
    'FractionalBackend',
    'validate_fractional_order',
    
    # Derivatives
    'create_fractional_derivative',
    'riemann_liouville_derivative',
    'caputo_derivative',
    'grunwald_letnikov_derivative',
    'weyl_derivative',
    'marchaud_derivative',
    'hadamard_derivative',
    'fractional_derivative_properties',
    'validate_fractional_derivative',
    
    # Integrals
    'create_fractional_integral',
    'RiemannLiouvilleIntegral',
    'CaputoIntegral',
    'WeylIntegral',
    'HadamardIntegral',
    'analytical_fractional_integral',
    'trapezoidal_fractional_integral',
    'simpson_fractional_integral',
    'fractional_integral_properties',
    'validate_fractional_integral',
    
    # Mathematical utilities
    'factorial_fractional',
    'binomial_coefficient',
    'pochhammer_symbol',
    'hypergeometric_series',
    'bessel_function_first_kind',
    'modified_bessel_function_first_kind',
    
    # Type checking and validation
    'validate_function',
    'validate_tensor_input',
    
    # Performance monitoring
    'timing_decorator',
    'memory_usage_decorator',
    'PerformanceMonitor',
    
    # Error handling
    'FractionalCalculusError',
    'ConvergenceError',
    'ValidationError',
    'safe_divide',
    'check_numerical_stability',
    
    # Common mathematical operations
    'vectorize_function',
    'normalize_array',
    'smooth_function',
    'fractional_power',
    'fractional_exponential',
    
    # Configuration utilities
    'get_default_precision',
    'set_default_precision',
    'get_available_methods',
    'get_method_properties',
    
    # Logging utilities
    'setup_logging',
    'get_logger'
]
