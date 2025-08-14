"""
FFT-based Methods for Fractional Calculus

This module implements various FFT-based algorithms for computing fractional
derivatives and integrals, including convolution methods, spectral approaches,
and optimized implementations using JAX and NUMBA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit, prange
from typing import Union, Optional, Tuple, Callable
from scipy.fft import fft, ifft, fftfreq
import numpy.fft as np_fft

from src.core.definitions import FractionalOrder
from src.special import gamma
from src.optimisation.numba_kernels import gamma_approx


class FFTFractionalMethods:
    """
    FFT-based methods for fractional calculus operations.

    This class provides various FFT-based algorithms for computing fractional
    derivatives and integrals using convolution and spectral methods.
    """

    def __init__(self, method: str = "convolution"):
        """
        Initialize FFT-based fractional calculus calculator.

        Args:
            method: FFT method ("convolution", "spectral", "fractional_fourier")
        """
        self.method = method.lower()

        # Validate method
        valid_methods = ["convolution", "spectral", "fractional_fourier", "wavelet"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def compute_derivative(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Compute fractional derivative using FFT-based method.

        Args:
            f: Function values array
            t: Time points array
            alpha: Fractional order
            **kwargs: Additional method-specific parameters

        Returns:
            Fractional derivative values
        """
        if self.method == "convolution":
            return self._convolution_derivative(f, t, alpha, **kwargs)
        elif self.method == "spectral":
            return self._spectral_derivative(f, t, alpha, **kwargs)
        elif self.method == "fractional_fourier":
            return self._fractional_fourier_derivative(f, t, alpha, **kwargs)
        elif self.method == "wavelet":
            return self._wavelet_derivative(f, t, alpha, **kwargs)

    def compute_integral(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Compute fractional integral using FFT-based method.

        Args:
            f: Function values array
            t: Time points array
            alpha: Fractional order (positive)
            **kwargs: Additional method-specific parameters

        Returns:
            Fractional integral values
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Fractional integral is derivative of negative order
        return self.compute_derivative(f, t, -alpha_val, **kwargs)

    def _convolution_derivative(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Convolution-based fractional derivative using FFT.

        Uses the fact that fractional derivative can be written as a convolution
        with a power-law kernel.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        N = len(f)
        h = t[1] - t[0] if len(t) > 1 else 1.0

        # Create power-law kernel
        kernel = np.zeros(N)
        for i in range(N):
            if i == 0:
                kernel[i] = 0
            else:
                kernel[i] = (t[i] ** (-alpha_val - 1)) / gamma_approx(-alpha_val)

        # Pad arrays for circular convolution
        f_padded = np.pad(f, (0, N), mode="constant")
        kernel_padded = np.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = fft(f_padded)
        kernel_fft = fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = np.real(ifft(conv_fft))

        return conv[:N] * h

    def _spectral_derivative(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Spectral fractional derivative using FFT.

        Uses the spectral representation of fractional derivative in Fourier space.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        N = len(f)
        h = t[1] - t[0] if len(t) > 1 else 1.0

        # Compute FFT of function
        f_fft = fft(f)

        # Create frequency array
        freqs = fftfreq(N, h)

        # Spectral derivative operator
        # For fractional derivative: (i*2*pi*freq)^alpha
        spectral_op = (1j * 2 * np.pi * freqs) ** alpha_val

        # Apply spectral operator
        result_fft = f_fft * spectral_op

        # Inverse FFT
        result = np.real(ifft(result_fft))

        return result

    def _fractional_fourier_derivative(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Fractional Fourier transform based derivative.

        Uses the relationship between fractional Fourier transform and
        fractional derivatives.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        N = len(f)
        h = t[1] - t[0] if len(t) > 1 else 1.0

        # Fractional Fourier transform parameter
        phi = np.pi * alpha_val / 2

        # Compute fractional Fourier transform
        f_frft = self._fractional_fourier_transform(f, phi)

        # Apply derivative in fractional Fourier domain
        freqs = fftfreq(N, h)
        derivative_op = (1j * 2 * np.pi * freqs) ** alpha_val
        result_frft = f_frft * derivative_op

        # Inverse fractional Fourier transform
        result = self._inverse_fractional_fourier_transform(result_frft, phi)

        return np.real(result)

    def _fractional_fourier_transform(self, f: np.ndarray, phi: float) -> np.ndarray:
        """
        Compute fractional Fourier transform.

        Args:
            f: Input array
            phi: Transform angle

        Returns:
            Fractional Fourier transform
        """
        N = len(f)

        # Create kernel matrix
        kernel = np.zeros((N, N), dtype=complex)
        for m in range(N):
            for n in range(N):
                if phi != 0:
                    kernel[m, n] = (
                        np.exp(1j * phi)
                        * np.sqrt(1j / (2 * np.pi * np.sin(phi)))
                        * np.exp(
                            1j
                            * ((m**2 + n**2) * np.cos(phi) - 2 * m * n)
                            / (2 * np.sin(phi))
                        )
                    )
                else:
                    kernel[m, n] = 1 if m == n else 0

        # Apply transform
        return kernel @ f

    def _inverse_fractional_fourier_transform(
        self, f: np.ndarray, phi: float
    ) -> np.ndarray:
        """
        Compute inverse fractional Fourier transform.

        Args:
            f: Input array
            phi: Transform angle

        Returns:
            Inverse fractional Fourier transform
        """
        return self._fractional_fourier_transform(f, -phi)

    def _wavelet_derivative(
        self,
        f: np.ndarray,
        t: np.ndarray,
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> np.ndarray:
        """
        Wavelet-based fractional derivative.

        Uses wavelet transform for computing fractional derivatives.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # This is a simplified wavelet implementation
        # In practice, you would use a proper wavelet library like PyWavelets

        N = len(f)
        result = np.zeros(N)

        # Simple wavelet-like approach using FFT
        f_fft = fft(f)
        freqs = fftfreq(N)

        # Wavelet-like spectral operator
        wavelet_op = (1j * 2 * np.pi * freqs) ** alpha_val * np.exp(-(freqs**2))

        result_fft = f_fft * wavelet_op
        result = np.real(ifft(result_fft))

        return result


# JAX-optimized implementations
class JAXFFTFractionalMethods:
    """JAX-optimized FFT-based fractional calculus methods."""

    @staticmethod
    @jax.jit
    def convolution_derivative_jax(
        f_values: jnp.ndarray, t_values: jnp.ndarray, alpha: float, h: float
    ) -> jnp.ndarray:
        """
        JAX-compiled convolution-based fractional derivative.

        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size

        Returns:
            Fractional derivative values
        """
        N = len(f_values)

        # Create power-law kernel
        kernel = jnp.where(
            jnp.arange(N) == 0, 0.0, (t_values ** (-alpha - 1)) / gamma_approx(-alpha)
        )

        # Pad for convolution
        f_padded = jnp.pad(f_values, (0, N), mode="constant")
        kernel_padded = jnp.pad(kernel, (0, N), mode="constant")

        # FFT convolution
        f_fft = jnp.fft.fft(f_padded)
        kernel_fft = jnp.fft.fft(kernel_padded)
        conv_fft = f_fft * kernel_fft
        conv = jnp.real(jnp.fft.ifft(conv_fft))

        return conv[:N] * h

    @staticmethod
    @jax.jit
    def spectral_derivative_jax(
        f_values: jnp.ndarray, t_values: jnp.ndarray, alpha: float, h: float
    ) -> jnp.ndarray:
        """
        JAX-compiled spectral fractional derivative.

        Args:
            f_values: Function values array
            t_values: Time points array
            alpha: Fractional order
            h: Step size

        Returns:
            Fractional derivative values
        """
        N = len(f_values)

        # Compute FFT
        f_fft = jnp.fft.fft(f_values)

        # Create frequency array
        freqs = jnp.fft.fftfreq(N, h)

        # Spectral derivative operator
        spectral_op = (1j * 2 * jnp.pi * freqs) ** alpha

        # Apply spectral operator
        result_fft = f_fft * spectral_op

        # Inverse FFT
        result = jnp.real(jnp.fft.ifft(result_fft))

        return result


# NUMBA-optimized implementations
@jit(nopython=True, parallel=True)
def fft_convolution_derivative_numba(
    f: np.ndarray, t: np.ndarray, alpha: float, h: float
) -> np.ndarray:
    """
    NUMBA-optimized FFT convolution-based fractional derivative.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        h: Step size

    Returns:
        Fractional derivative values
    """
    N = len(f)

    # Create power-law kernel
    kernel = np.zeros(N)
    for i in range(1, N):
        kernel[i] = (t[i] ** (-alpha - 1)) / gamma_approx(-alpha)

    # Pad arrays for circular convolution
    f_padded = np.zeros(2 * N)
    kernel_padded = np.zeros(2 * N)

    f_padded[:N] = f
    kernel_padded[:N] = kernel

    # FFT convolution
    f_fft = np_fft.fft(f_padded)
    kernel_fft = np_fft.fft(kernel_padded)
    conv_fft = f_fft * kernel_fft
    conv = np.real(np_fft.ifft(conv_fft))

    return conv[:N] * h


@jit(nopython=True, parallel=True)
def fft_spectral_derivative_numba(
    f: np.ndarray, t: np.ndarray, alpha: float, h: float
) -> np.ndarray:
    """
    NUMBA-optimized FFT spectral fractional derivative.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        h: Step size

    Returns:
        Fractional derivative values
    """
    N = len(f)

    # Compute FFT
    f_fft = np_fft.fft(f)

    # Create frequency array
    freqs = np_fft.fftfreq(N, h)

    # Spectral derivative operator
    spectral_op = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        if freqs[i] != 0:
            spectral_op[i] = (1j * 2 * np.pi * freqs[i]) ** alpha
        else:
            spectral_op[i] = 0.0

    # Apply spectral operator
    result_fft = f_fft * spectral_op

    # Inverse FFT
    result = np.real(np_fft.ifft(result_fft))

    return result


# Convenience functions
def fft_fractional_derivative(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "convolution",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for computing FFT-based fractional derivative.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        method: FFT method
        **kwargs: Additional parameters

    Returns:
        Fractional derivative values
    """
    calculator = FFTFractionalMethods(method)
    return calculator.compute_derivative(f, t, alpha, **kwargs)


def fft_fractional_integral(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "convolution",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for computing FFT-based fractional integral.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order (positive)
        method: FFT method
        **kwargs: Additional parameters

    Returns:
        Fractional integral values
    """
    calculator = FFTFractionalMethods(method)
    return calculator.compute_integral(f, t, alpha, **kwargs)


def fft_fractional_derivative_jax(
    f_values: np.ndarray,
    t_values: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: float,
    method: str = "convolution",
) -> np.ndarray:
    """
    JAX-optimized FFT-based fractional derivative computation.

    Args:
        f_values: Function values array
        t_values: Time points array
        alpha: Fractional order
        h: Step size
        method: Method ("convolution" or "spectral")

    Returns:
        Fractional derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    if method == "convolution":
        return JAXFFTFractionalMethods.convolution_derivative_jax(
            jnp.array(f_values), jnp.array(t_values), alpha_val, h
        )
    elif method == "spectral":
        return JAXFFTFractionalMethods.spectral_derivative_jax(
            jnp.array(f_values), jnp.array(t_values), alpha_val, h
        )
    else:
        raise ValueError(
            "Method must be 'convolution' or 'spectral' for JAX implementation"
        )


def fft_fractional_derivative_numba(
    f: np.ndarray,
    t: np.ndarray,
    alpha: Union[float, FractionalOrder],
    method: str = "convolution",
) -> np.ndarray:
    """
    NUMBA-optimized FFT-based fractional derivative computation.

    Args:
        f: Function values array
        t: Time points array
        alpha: Fractional order
        method: Method ("convolution" or "spectral")

    Returns:
        Fractional derivative values
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha

    h = t[1] - t[0] if len(t) > 1 else 1.0

    if method == "convolution":
        return fft_convolution_derivative_numba(f, t, alpha_val, h)
    elif method == "spectral":
        return fft_spectral_derivative_numba(f, t, alpha_val, h)
    else:
        raise ValueError(
            "Method must be 'convolution' or 'spectral' for NUMBA implementation"
        )
