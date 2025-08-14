"""
Advanced JAX Implementations for Fractional Calculus

This module provides advanced JAX optimizations including GPU acceleration,
automatic differentiation, vectorization, and performance tuning for
fractional calculus operations.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, jacrev, hessian
from jax import lax, random, tree_util
from jax.scipy import special, linalg
from jax.numpy.fft import fft, ifft, fftfreq
from typing import Union, Optional, Tuple, Callable, Dict, Any
import time

from src.core.definitions import FractionalOrder
from src.special import gamma, mittag_leffler


class JAXOptimizer:
    """
    Advanced JAX optimizer for fractional calculus operations.
    
    Provides GPU acceleration, automatic differentiation, and vectorization
    for high-performance fractional calculus computations.
    """
    
    def __init__(self, device: str = "auto", precision: str = "float32"):
        """
        Initialize JAX optimizer.
        
        Args:
            device: Target device ("cpu", "gpu", "tpu", "auto")
            precision: Numerical precision ("float32", "float64")
        """
        self.device = device
        self.precision = precision
        
        # Set JAX configuration
        if precision == "float64":
            jax.config.update("jax_enable_x64", True)
        
        # Set device
        if device == "gpu":
            jax.config.update("jax_platform_name", "gpu")
        elif device == "tpu":
            jax.config.update("jax_platform_name", "tpu")
    
    def optimize_fractional_derivative(self, 
                                     derivative_func: Callable,
                                     **kwargs) -> Callable:
        """
        Optimize a fractional derivative function with JAX.
        
        Args:
            derivative_func: Function to optimize
            **kwargs: Optimization parameters
            
        Returns:
            Optimized function
        """
        # Apply JIT compilation
        optimized_func = jit(derivative_func)
        
        # Apply vectorization if needed
        if kwargs.get('vectorize', True):
            optimized_func = vmap(optimized_func)
        
        return optimized_func
    
    def create_gpu_kernel(self, 
                         kernel_func: Callable,
                         input_shapes: Tuple[Tuple[int, ...], ...],
                         **kwargs) -> Callable:
        """
        Create an optimized GPU kernel for fractional calculus.
        
        Args:
            kernel_func: Kernel function to optimize
            input_shapes: Expected input shapes
            **kwargs: Kernel parameters
            
        Returns:
            Optimized GPU kernel
        """
        # Compile for GPU
        gpu_kernel = jit(kernel_func, device=jax.devices("gpu")[0])
        
        # Pre-compile with concrete shapes if provided
        if input_shapes:
            gpu_kernel = jax.jit(kernel_func, static_argnums=kwargs.get('static_argnums', ()))
        
        return gpu_kernel


class JAXFractionalDerivatives:
    """
    Advanced JAX implementations of fractional derivatives.
    
    Provides GPU-accelerated, automatically differentiable fractional
    derivative computations.
    """
    
    def __init__(self, optimizer: Optional[JAXOptimizer] = None):
        """
        Initialize JAX fractional derivatives.
        
        Args:
            optimizer: JAX optimizer instance
        """
        self.optimizer = optimizer or JAXOptimizer()
    
    @staticmethod
    @jit
    def caputo_derivative_gpu(f_values: jnp.ndarray, 
                             t_values: jnp.ndarray,
                             alpha: float,
                             h: float) -> jnp.ndarray:
        """
        GPU-accelerated Caputo derivative computation.
        
        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size
            
        Returns:
            Caputo derivative values
        """
        n = jnp.ceil(alpha).astype(int)
        
        # Create convolution kernel
        kernel = (t_values ** (n - alpha - 1)) / gamma(n - alpha)
        
        # Pad for convolution
        N = len(f_values)
        f_padded = jnp.pad(f_values, (0, N), mode='constant')
        kernel_padded = jnp.pad(kernel, (0, N), mode='constant')
        
        # FFT convolution on GPU
        f_fft = fft(f_padded)
        kernel_fft = fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(ifft(conv_fft))
        
        return conv[:N] * h
    
    @staticmethod
    @jit
    def riemann_liouville_derivative_gpu(f_values: jnp.ndarray,
                                       t_values: jnp.ndarray,
                                       alpha: float,
                                       h: float) -> jnp.ndarray:
        """
        GPU-accelerated Riemann-Liouville derivative computation.
        
        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size
            
        Returns:
            Riemann-Liouville derivative values
        """
        n = jnp.ceil(alpha).astype(int)
        
        # Create convolution kernel
        kernel = (t_values ** (n - alpha - 1)) / gamma(n - alpha)
        
        # Pad for convolution
        N = len(f_values)
        f_padded = jnp.pad(f_values, (0, N), mode='constant')
        kernel_padded = jnp.pad(kernel, (0, N), mode='constant')
        
        # FFT convolution on GPU
        f_fft = fft(f_padded)
        kernel_fft = fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(ifft(conv_fft))
        
        # Apply nth derivative using finite differences
        result = jnp.zeros(N)
        
        # First n points are zero
        result = result.at[:n].set(0.0)
        
        # Apply finite differences for remaining points
        for i in range(n, N):
            if n == 1:
                if i < N - 1:
                    result = result.at[i].set((conv[i+1] - conv[i-1]) / (2 * h))
                else:
                    result = result.at[i].set((conv[i] - conv[i-1]) / h)
            else:
                # Simplified higher derivative computation
                if i < N - n:
                    result = result.at[i].set(
                        (conv[i+1] - 2*conv[i] + conv[i-1]) / (h**2)
                    )
                else:
                    result = result.at[i].set((conv[i] - conv[i-1]) / h)
        
        return result * h
    
    @staticmethod
    @jit
    def grunwald_letnikov_derivative_gpu(f_values: jnp.ndarray,
                                        t_values: jnp.ndarray,
                                        alpha: float,
                                        h: float) -> jnp.ndarray:
        """
        GPU-accelerated Grünwald-Letnikov derivative computation.
        
        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size
            
        Returns:
            Grünwald-Letnikov derivative values
        """
        N = len(f_values)
        
        # Create Grünwald-Letnikov coefficient kernel
        kernel = jnp.zeros(N)
        
        # Compute coefficients using vectorized operations
        j_indices = jnp.arange(N)
        kernel = jnp.where(
            j_indices == 0,
            1.0,
            (-1) ** j_indices * gamma(alpha + 1) / (gamma(j_indices + 1) * gamma(alpha - j_indices + 1))
        )
        
        # Pad for convolution
        f_padded = jnp.pad(f_values, (0, N), mode='constant')
        kernel_padded = jnp.pad(kernel, (0, N), mode='constant')
        
        # FFT convolution on GPU
        f_fft = fft(f_padded)
        kernel_fft = fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(ifft(conv_fft))
        
        return conv[:N] * (h ** (-alpha))
    
    @staticmethod
    @jit
    def fft_fractional_derivative_gpu(f_values: jnp.ndarray,
                                     t_values: jnp.ndarray,
                                     alpha: float,
                                     h: float,
                                     method: str = "spectral") -> jnp.ndarray:
        """
        GPU-accelerated FFT-based fractional derivative.
        
        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size
            method: FFT method ("spectral" or "convolution")
            
        Returns:
            FFT-based fractional derivative values
        """
        N = len(f_values)
        
        if method == "spectral":
            # Compute FFT
            f_fft = fft(f_values)
            
            # Create frequency array
            freqs = fftfreq(N, h)
            
            # Spectral derivative operator
            spectral_op = (1j * 2 * jnp.pi * freqs) ** alpha
            
            # Apply spectral operator
            result_fft = f_fft * spectral_op
            
            # Inverse FFT
            result = jnp.real(ifft(result_fft))
            
            return result
        
        elif method == "convolution":
            # Convolution-based approach
            kernel = jnp.where(
                jnp.arange(N) == 0,
                0.0,
                (t_values ** (-alpha - 1)) / gamma(-alpha)
            )
            
            # Pad for convolution
            f_padded = jnp.pad(f_values, (0, N), mode='constant')
            kernel_padded = jnp.pad(kernel, (0, N), mode='constant')
            
            # FFT convolution
            f_fft = fft(f_padded)
            kernel_fft = fft(kernel_padded)
            conv_fft = f_fft * kernel_fft
            conv = jnp.real(ifft(conv_fft))
            
            return conv[:N] * h
        
        else:
            raise ValueError("Method must be 'spectral' or 'convolution'")


class JAXAutomaticDifferentiation:
    """
    Automatic differentiation for fractional calculus using JAX.
    
    Provides gradients, Jacobians, and Hessians for fractional
    derivative computations.
    """
    
    @staticmethod
    def gradient_wrt_alpha(derivative_func: Callable,
                          f_values: jnp.ndarray,
                          t_values: jnp.ndarray,
                          alpha: float,
                          h: float) -> jnp.ndarray:
        """
        Compute gradient of fractional derivative with respect to alpha.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values: Time points
            alpha: Fractional order
            h: Step size
            
        Returns:
            Gradient with respect to alpha
        """
        grad_func = grad(derivative_func, argnums=2)  # Differentiate w.r.t. alpha
        return grad_func(f_values, t_values, alpha, h)
    
    @staticmethod
    def jacobian_wrt_function(derivative_func: Callable,
                             f_values: jnp.ndarray,
                             t_values: jnp.ndarray,
                             alpha: float,
                             h: float) -> jnp.ndarray:
        """
        Compute Jacobian of fractional derivative with respect to function values.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values: Time points
            alpha: Fractional order
            h: Step size
            
        Returns:
            Jacobian with respect to function values
        """
        jacobian_func = jacfwd(derivative_func, argnums=0)  # Differentiate w.r.t. f_values
        return jacobian_func(f_values, t_values, alpha, h)
    
    @staticmethod
    def hessian_wrt_alpha(derivative_func: Callable,
                         f_values: jnp.ndarray,
                         t_values: jnp.ndarray,
                         alpha: float,
                         h: float) -> jnp.ndarray:
        """
        Compute Hessian of fractional derivative with respect to alpha.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values: Time points
            alpha: Fractional order
            h: Step size
            
        Returns:
            Hessian with respect to alpha
        """
        hessian_func = hessian(derivative_func, argnums=2)  # Second derivative w.r.t. alpha
        return hessian_func(f_values, t_values, alpha, h)


class JAXVectorization:
    """
    Vectorized operations for fractional calculus using JAX.
    
    Provides efficient vectorized computations for multiple
    fractional orders, functions, or parameters.
    """
    
    @staticmethod
    def vectorize_over_alpha(derivative_func: Callable,
                           f_values: jnp.ndarray,
                           t_values: jnp.ndarray,
                           alphas: jnp.ndarray,
                           h: float) -> jnp.ndarray:
        """
        Vectorize fractional derivative computation over multiple alpha values.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values: Time points
            alphas: Array of fractional orders
            h: Step size
            
        Returns:
            Array of derivatives for each alpha
        """
        vectorized_func = vmap(lambda alpha: derivative_func(f_values, t_values, alpha, h))
        return vectorized_func(alphas)
    
    @staticmethod
    def vectorize_over_functions(derivative_func: Callable,
                               f_values_array: jnp.ndarray,
                               t_values: jnp.ndarray,
                               alpha: float,
                               h: float) -> jnp.ndarray:
        """
        Vectorize fractional derivative computation over multiple functions.
        
        Args:
            derivative_func: Fractional derivative function
            f_values_array: Array of function values (batch dimension first)
            t_values: Time points
            alpha: Fractional order
            h: Step size
            
        Returns:
            Array of derivatives for each function
        """
        vectorized_func = vmap(lambda f: derivative_func(f, t_values, alpha, h))
        return vectorized_func(f_values_array)
    
    @staticmethod
    def vectorize_over_time(derivative_func: Callable,
                          f_values: jnp.ndarray,
                          t_values_array: jnp.ndarray,
                          alpha: float,
                          h: float) -> jnp.ndarray:
        """
        Vectorize fractional derivative computation over multiple time grids.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values_array: Array of time grids
            alpha: Fractional order
            h: Step size
            
        Returns:
            Array of derivatives for each time grid
        """
        vectorized_func = vmap(lambda t: derivative_func(f_values, t, alpha, h))
        return vectorized_func(t_values_array)


class JAXPerformanceMonitor:
    """
    Performance monitoring and optimization for JAX computations.
    
    Provides tools for measuring and optimizing performance of
    fractional calculus operations.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.timings = {}
        self.memory_usage = {}
    
    def time_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Time a JAX function execution.
        
        Args:
            func: Function to time
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with timing information
        """
        # Warm-up run
        _ = func(*args, **kwargs)
        
        # Compile if needed
        compiled_func = jit(func)
        
        # Time the execution
        start_time = time.time()
        result = compiled_func(*args, **kwargs)
        end_time = time.time()
        
        # Block until computation is complete
        result.block_until_ready()
        
        execution_time = end_time - start_time
        
        return {
            'execution_time': execution_time,
            'result_shape': jnp.shape(result),
            'result_dtype': result.dtype
        }
    
    def benchmark_derivatives(self, 
                            f_values: jnp.ndarray,
                            t_values: jnp.ndarray,
                            alpha: float,
                            h: float) -> Dict[str, Any]:
        """
        Benchmark different fractional derivative implementations.
        
        Args:
            f_values: Function values
            t_values: Time points
            alpha: Fractional order
            h: Step size
            
        Returns:
            Dictionary with benchmark results
        """
        derivatives = {
            'caputo': JAXFractionalDerivatives.caputo_derivative_gpu,
            'riemann_liouville': JAXFractionalDerivatives.riemann_liouville_derivative_gpu,
            'grunwald_letnikov': JAXFractionalDerivatives.grunwald_letnikov_derivative_gpu,
            'fft_spectral': lambda f, t, a, h: JAXFractionalDerivatives.fft_fractional_derivative_gpu(f, t, a, h, "spectral"),
            'fft_convolution': lambda f, t, a, h: JAXFractionalDerivatives.fft_fractional_derivative_gpu(f, t, a, h, "convolution")
        }
        
        results = {}
        
        for name, func in derivatives.items():
            try:
                timing = self.time_function(func, f_values, t_values, alpha, h)
                results[name] = timing
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def optimize_batch_size(self, 
                          derivative_func: Callable,
                          f_values: jnp.ndarray,
                          t_values: jnp.ndarray,
                          alpha: float,
                          h: float,
                          max_batch_size: int = 1000) -> Dict[str, Any]:
        """
        Find optimal batch size for vectorized operations.
        
        Args:
            derivative_func: Fractional derivative function
            f_values: Function values
            t_values: Time points
            alpha: Fractional order
            h: Step size
            max_batch_size: Maximum batch size to test
            
        Returns:
            Dictionary with optimization results
        """
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
        
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create batch of functions
                f_batch = jnp.tile(f_values, (batch_size, 1))
                
                # Time vectorized computation
                vectorized_func = vmap(lambda f: derivative_func(f, t_values, alpha, h))
                timing = self.time_function(vectorized_func, f_batch)
                
                results[batch_size] = {
                    'execution_time': timing['execution_time'],
                    'throughput': batch_size / timing['execution_time']  # functions per second
                }
                
            except Exception as e:
                results[batch_size] = {'error': str(e)}
        
        return results


# Convenience functions
def optimize_fractional_derivative_jax(derivative_func: Callable,
                                      device: str = "auto",
                                      precision: str = "float32",
                                      **kwargs) -> Callable:
    """
    Optimize a fractional derivative function with JAX.
    
    Args:
        derivative_func: Function to optimize
        device: Target device
        precision: Numerical precision
        **kwargs: Additional optimization parameters
        
    Returns:
        Optimized function
    """
    optimizer = JAXOptimizer(device, precision)
    return optimizer.optimize_fractional_derivative(derivative_func, **kwargs)


def compute_fractional_derivative_gpu(f_values: jnp.ndarray,
                                    t_values: jnp.ndarray,
                                    alpha: float,
                                    h: float,
                                    method: str = "caputo") -> jnp.ndarray:
    """
    Compute fractional derivative using GPU acceleration.
    
    Args:
        f_values: Function values array
        t_values: Time points array
        alpha: Fractional order
        h: Step size
        method: Derivative method
        
    Returns:
        Fractional derivative values
    """
    if method == "caputo":
        return JAXFractionalDerivatives.caputo_derivative_gpu(f_values, t_values, alpha, h)
    elif method == "riemann_liouville":
        return JAXFractionalDerivatives.riemann_liouville_derivative_gpu(f_values, t_values, alpha, h)
    elif method == "grunwald_letnikov":
        return JAXFractionalDerivatives.grunwald_letnikov_derivative_gpu(f_values, t_values, alpha, h)
    elif method == "fft_spectral":
        return JAXFractionalDerivatives.fft_fractional_derivative_gpu(f_values, t_values, alpha, h, "spectral")
    elif method == "fft_convolution":
        return JAXFractionalDerivatives.fft_fractional_derivative_gpu(f_values, t_values, alpha, h, "convolution")
    else:
        raise ValueError(f"Unknown method: {method}")


def vectorize_fractional_derivatives(f_values: jnp.ndarray,
                                   t_values: jnp.ndarray,
                                   alphas: jnp.ndarray,
                                   h: float,
                                   method: str = "caputo") -> jnp.ndarray:
    """
    Vectorize fractional derivative computation over multiple alpha values.
    
    Args:
        f_values: Function values
        t_values: Time points
        alphas: Array of fractional orders
        h: Step size
        method: Derivative method
        
    Returns:
        Array of derivatives for each alpha
    """
    if method == "caputo":
        derivative_func = JAXFractionalDerivatives.caputo_derivative_gpu
    elif method == "riemann_liouville":
        derivative_func = JAXFractionalDerivatives.riemann_liouville_derivative_gpu
    elif method == "grunwald_letnikov":
        derivative_func = JAXFractionalDerivatives.grunwald_letnikov_derivative_gpu
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return JAXVectorization.vectorize_over_alpha(derivative_func, f_values, t_values, alphas, h)
