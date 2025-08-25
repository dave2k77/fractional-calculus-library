"""
Core Utilities Module

This module provides common utility functions used throughout the HPFRACC library:
- Mathematical utilities and helper functions
- Type checking and validation utilities
- Performance monitoring utilities
- Error handling and debugging utilities
- Common mathematical operations
"""

import numpy as np
import torch
import warnings
from typing import Union, Callable, Optional, Tuple, List, Dict, Any
from functools import wraps
import time
import logging
from scipy.special import gamma, factorial
import math

from .definitions import FractionalOrder, DefinitionType


# Mathematical utilities
def factorial_fractional(n: Union[int, float]) -> float:
    """
    Compute factorial for integer and fractional values.
    
    Args:
        n: Number to compute factorial for
        
    Returns:
        Factorial value
    """
    if isinstance(n, int) and n >= 0:
        return float(factorial(n))
    elif isinstance(n, float) and n > -1:
        return gamma(n + 1)
    else:
        raise ValueError(f"Factorial not defined for {n}")


def binomial_coefficient(n: Union[int, float], k: Union[int, float]) -> float:
    """
    Compute binomial coefficient for real numbers.
    
    Args:
        n: Upper parameter
        k: Lower parameter
        
    Returns:
        Binomial coefficient value
    """
    if k < 0:
        return 0.0
    elif k == 0:
        return 1.0
    else:
        return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))


def pochhammer_symbol(x: float, n: int) -> float:
    """
    Compute Pochhammer symbol (x)_n = x(x+1)...(x+n-1).
    
    Args:
        x: Base value
        n: Number of factors
        
    Returns:
        Pochhammer symbol value
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return gamma(x + n) / gamma(x)


def hypergeometric_series(a: List[float], b: List[float], z: float, max_terms: int = 100) -> float:
    """
    Compute hypergeometric series pFq(a; b; z).
    
    Args:
        a: List of upper parameters
        b: List of lower parameters
        z: Variable
        max_terms: Maximum number of terms to compute
        
    Returns:
        Hypergeometric series value
    """
    result = 1.0
    term = 1.0
    
    for n in range(1, max_terms + 1):
        # Compute numerator and denominator
        numerator = 1.0
        denominator = 1.0
        
        for ai in a:
            numerator *= pochhammer_symbol(ai, n)
        
        for bi in b:
            denominator *= pochhammer_symbol(bi, n)
        
        term *= (numerator / denominator) * (z ** n) / factorial(n)
        result += term
        
        # Check convergence
        if abs(term) < 1e-12:
            break
    
    return result


def bessel_function_first_kind(nu: float, x: float) -> float:
    """
    Compute Bessel function of the first kind J_ν(x).
    
    Args:
        nu: Order of Bessel function
        x: Argument
        
    Returns:
        Bessel function value
    """
    # Use hypergeometric series representation
    if x == 0:
        return 1.0 if nu == 0 else 0.0
    
    z = -(x ** 2) / 4
    return (x / 2) ** nu * hypergeometric_series([], [nu + 1], z) / gamma(nu + 1)


def modified_bessel_function_first_kind(nu: float, x: float) -> float:
    """
    Compute modified Bessel function of the first kind I_ν(x).
    
    Args:
        nu: Order of Bessel function
        x: Argument
        
    Returns:
        Modified Bessel function value
    """
    # Use hypergeometric series representation
    if x == 0:
        return 1.0 if nu == 0 else 0.0
    
    z = (x ** 2) / 4
    return (x / 2) ** nu * hypergeometric_series([], [nu + 1], z) / gamma(nu + 1)


# Type checking and validation utilities
def validate_fractional_order(alpha: Union[float, FractionalOrder], min_val: float = 0.0, max_val: float = 2.0) -> FractionalOrder:
    """
    Validate and convert fractional order.
    
    Args:
        alpha: Fractional order value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated FractionalOrder object
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = float(alpha)
    
    if not (min_val <= alpha_val <= max_val):
        raise ValueError(f"Fractional order must be in [{min_val}, {max_val}], got {alpha_val}")
    
    return FractionalOrder(alpha_val)


def validate_function(f: Callable, domain: Tuple[float, float] = (0.0, 1.0), n_points: int = 100) -> bool:
    """
    Validate that a function is callable and well-behaved on a domain.
    
    Args:
        f: Function to validate
        domain: Domain to test on (min, max)
        n_points: Number of test points
        
    Returns:
        True if function is valid
    """
    if not callable(f):
        raise ValueError("Function must be callable")
    
    try:
        x_test = np.linspace(domain[0], domain[1], n_points)
        y_test = f(x_test)
        
        # Check for finite values
        if not np.all(np.isfinite(y_test)):
            return False
        
        return True
    except Exception:
        return False


def validate_tensor_input(x: Union[np.ndarray, torch.Tensor], expected_shape: Optional[Tuple] = None) -> bool:
    """
    Validate tensor input for fractional calculus operations.
    
    Args:
        x: Input tensor
        expected_shape: Expected shape (optional)
        
    Returns:
        True if input is valid
    """
    if isinstance(x, np.ndarray):
        if not np.isfinite(x).all():
            return False
    elif isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            return False
    else:
        return False
    
    if expected_shape is not None:
        if x.shape != expected_shape:
            return False
    
    return True


# Performance monitoring utilities
def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Log timing information
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} executed in {execution_time:.6f} seconds")
        
        return result
    
    return wrapper


def memory_usage_decorator(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with memory monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process()
        
        # Memory before execution
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        # Memory after execution
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        logging.info(f"{func.__name__} used {memory_used:.2f} MB of memory")
        
        return result
    
    return wrapper


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking execution times and memory usage.
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        self.timings[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing an operation and return duration."""
        if name not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.time() - self.timings[name]
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        
        logging.info(f"{name} took {duration:.6f} seconds")
        return duration
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "call_counts": self.call_counts.copy(),
            "timings": self.timings.copy(),
            "memory_usage": self.memory_usage.copy()
        }


# Error handling and debugging utilities
class FractionalCalculusError(Exception):
    """Base exception for fractional calculus operations."""
    pass


class ConvergenceError(FractionalCalculusError):
    """Exception raised when numerical methods fail to converge."""
    pass


class ValidationError(FractionalCalculusError):
    """Exception raised when input validation fails."""
    pass


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    if abs(denominator) < 1e-12:
        warnings.warn(f"Division by zero detected, using default value {default}")
        return default
    return numerator / denominator


def check_numerical_stability(values: Union[np.ndarray, torch.Tensor], tolerance: float = 1e-10) -> bool:
    """
    Check if numerical values are stable.
    
    Args:
        values: Array of values to check
        tolerance: Tolerance for stability check
        
    Returns:
        True if values are stable
    """
    if isinstance(values, np.ndarray):
        return np.all(np.isfinite(values)) and np.all(np.abs(values) < 1/tolerance)
    elif isinstance(values, torch.Tensor):
        return torch.all(torch.isfinite(values)) and torch.all(torch.abs(values) < 1/tolerance)
    else:
        return False


# Common mathematical operations
def vectorize_function(func: Callable, vectorize: bool = True) -> Callable:
    """
    Vectorize a scalar function for array inputs.
    
    Args:
        func: Scalar function to vectorize
        vectorize: Whether to use numpy vectorize
        
    Returns:
        Vectorized function
    """
    if vectorize:
        return np.vectorize(func)
    else:
        def vectorized_func(x):
            if isinstance(x, (list, tuple)):
                return [func(xi) for xi in x]
            elif isinstance(x, np.ndarray):
                return np.array([func(xi) for xi in x])
            elif isinstance(x, torch.Tensor):
                return torch.tensor([func(float(xi)) for xi in x])
            else:
                return func(x)
        return vectorized_func


def normalize_array(arr: Union[np.ndarray, torch.Tensor], norm_type: str = "l2") -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize an array using different norm types.
    
    Args:
        arr: Array to normalize
        norm_type: Type of normalization ("l1", "l2", "max", "minmax")
        
    Returns:
        Normalized array
    """
    if isinstance(arr, np.ndarray):
        if norm_type == "l1":
            norm = np.sum(np.abs(arr))
        elif norm_type == "l2":
            norm = np.sqrt(np.sum(arr ** 2))
        elif norm_type == "max":
            norm = np.max(np.abs(arr))
        elif norm_type == "minmax":
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        return arr / norm if norm > 0 else arr
    
    elif isinstance(arr, torch.Tensor):
        if norm_type == "l1":
            norm = torch.sum(torch.abs(arr))
        elif norm_type == "l2":
            norm = torch.sqrt(torch.sum(arr ** 2))
        elif norm_type == "max":
            norm = torch.max(torch.abs(arr))
        elif norm_type == "minmax":
            return (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr))
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        return arr / norm if norm > 0 else arr
    
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


def smooth_function(func: Callable, smoothing_factor: float = 0.1) -> Callable:
    """
    Create a smoothed version of a function using convolution.
    
    Args:
        func: Original function
        smoothing_factor: Smoothing factor (0-1)
        
    Returns:
        Smoothed function
    """
    def smoothed_func(x):
        if isinstance(x, (int, float)):
            return func(x)
        
        # Apply simple moving average smoothing
        if isinstance(x, np.ndarray):
            window_size = max(1, int(len(x) * smoothing_factor))
            kernel = np.ones(window_size) / window_size
            return np.convolve(func(x), kernel, mode='same')
        else:
            return func(x)
    
    return smoothed_func


# Utility functions for fractional calculus
def fractional_power(x: Union[float, np.ndarray, torch.Tensor], alpha: float) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute fractional power with proper handling of negative values.
    
    Args:
        x: Base value(s)
        alpha: Fractional exponent
        
    Returns:
        Fractional power result
    """
    if isinstance(x, (int, float)):
        if x >= 0:
            return x ** alpha
        else:
            # For negative values, use complex power
            return (-x) ** alpha * np.exp(1j * np.pi * alpha)
    
    elif isinstance(x, np.ndarray):
        result = np.zeros_like(x, dtype=complex)
        positive_mask = x >= 0
        negative_mask = x < 0
        
        result[positive_mask] = x[positive_mask] ** alpha
        result[negative_mask] = (-x[negative_mask]) ** alpha * np.exp(1j * np.pi * alpha)
        
        return result
    
    elif isinstance(x, torch.Tensor):
        result = torch.zeros_like(x, dtype=torch.complex64)
        positive_mask = x >= 0
        negative_mask = x < 0
        
        result[positive_mask] = x[positive_mask] ** alpha
        result[negative_mask] = (-x[negative_mask]) ** alpha * torch.exp(1j * torch.pi * alpha)
        
        return result
    
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def fractional_exponential(x: Union[float, np.ndarray, torch.Tensor], alpha: float) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute fractional exponential function.
    
    Args:
        x: Input value(s)
        alpha: Fractional order
        
    Returns:
        Fractional exponential result
    """
    # Use Mittag-Leffler function approximation
    if isinstance(x, (int, float)):
        return hypergeometric_series([1], [1], x ** alpha)
    
    elif isinstance(x, np.ndarray):
        return np.array([hypergeometric_series([1], [1], xi ** alpha) for xi in x])
    
    elif isinstance(x, torch.Tensor):
        return torch.tensor([hypergeometric_series([1], [1], float(xi) ** alpha) for xi in x])
    
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


# Configuration utilities
def get_default_precision() -> int:
    """Get default numerical precision for the library."""
    return 64


def set_default_precision(precision: int):
    """Set default numerical precision for the library."""
    if precision not in [32, 64, 128]:
        raise ValueError("Precision must be 32, 64, or 128")
    
    # This would typically set global precision settings
    warnings.warn("Precision setting not fully implemented")


def get_available_methods() -> List[str]:
    """Get list of available fractional calculus methods."""
    return ["RL", "Caputo", "GL", "Weyl", "Marchaud", "Hadamard"]


def get_method_properties(method: str) -> Dict[str, Any]:
    """
    Get properties of a specific fractional calculus method.
    
    Args:
        method: Method name
        
    Returns:
        Dictionary of method properties
    """
    properties = {
        "RL": {
            "full_name": "Riemann-Liouville",
            "order_range": (0, 2),
            "memory_effect": True,
            "initial_conditions": "Complex",
            "numerical_stability": "Good"
        },
        "Caputo": {
            "full_name": "Caputo",
            "order_range": (0, 1),
            "memory_effect": True,
            "initial_conditions": "Simple",
            "numerical_stability": "Excellent"
        },
        "GL": {
            "full_name": "Grünwald-Letnikov",
            "order_range": (0, 2),
            "memory_effect": True,
            "initial_conditions": "Discrete",
            "numerical_stability": "Good"
        },
        "Weyl": {
            "full_name": "Weyl",
            "order_range": (0, 2),
            "memory_effect": False,
            "initial_conditions": "Periodic",
            "numerical_stability": "Good"
        }
    }
    
    return properties.get(method, {})


# Logging utilities
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging for the HPFRACC library.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file
    )


def get_logger(name: str = "hpfracc") -> logging.Logger:
    """
    Get a logger for the HPFRACC library.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
