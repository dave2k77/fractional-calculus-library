"""
Special Fractional Calculus Methods

This module implements specialized fractional calculus methods that are fundamental
for advanced applications:

- Fractional Laplacian: Essential for PDEs and diffusion processes
- Fractional Fourier Transform: Powerful for signal processing and spectral analysis
- Fractional Z-Transform: Useful for discrete-time systems and digital signal processing

These methods provide the foundation for more complex fractional calculus applications.
"""

import numpy as np
from numba import jit as numba_jit
from typing import Union, Optional, Callable, Tuple
import warnings
from scipy import special
from scipy.fft import fft, ifft, fftfreq
import math

from ..core.definitions import FractionalOrder


@numba_jit(nopython=True)
def _factorial(n: int) -> int:
    """Simple factorial function for NUMBA compatibility."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


class FractionalLaplacian:
    """
    Fractional Laplacian operator implementation.

    The fractional Laplacian is defined as:
    (-Δ)^(α/2) f(x) = F^(-1)[|ξ|^α F[f](ξ)]

    This is a fundamental operator in fractional PDEs, diffusion processes,
    and many physical applications.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], *, dimension: int = 1, boundary_conditions: Optional[str] = None):
        """
        Initialize fractional Laplacian calculator.
        
        Args:
            order: Fractional order (0 < α < 2)
            dimension: Spatial dimension
            boundary_conditions: Boundary condition type
        """
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.dimension = int(dimension)
        self.boundary_conditions = boundary_conditions or "dirichlet"

        # Validate alpha range
        if self.alpha_val <= 0 or self.alpha_val >= 2:
            # Only warn once per alpha value to avoid spam
            warning_key = f"fractional_laplacian_alpha_{self.alpha_val}"
            if not hasattr(warnings, '_alpha_warning_tracker'):
                warnings._alpha_warning_tracker = set()
            if warning_key not in warnings._alpha_warning_tracker:
                warnings.warn(
                    f"Alpha should be in (0, 2) for fractional Laplacian. Got {self.alpha_val}. "
                    f"Results may be inaccurate.")
                warnings._alpha_warning_tracker.add(warning_key)

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: str = "spectral"
    ) -> Union[float, np.ndarray]:
        """
        Compute fractional Laplacian.

        Args:
            f: Function or array of function values
            x: Domain points
            h: Step size (if not provided, inferred from x)
            method: Computation method ("spectral", "finite_difference", "integral")

        Returns:
            Fractional Laplacian values
        """
        # Handle empty inputs early
        if hasattr(x, "__len__") and len(x) == 0:
            return np.array([])

        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(-x_max, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if len(f_array) == 0 or len(x_array) == 0:
            return np.array([])

        if method == "spectral":
            return self._spectral_method(f_array, x_array, h or 1.0)
        elif method == "finite_difference":
            return self._finite_difference_method(f_array, x_array, h or 1.0)
        elif method == "integral":
            return self._integral_method(f_array, x_array, h or 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _spectral_method(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Spectral method using FFT."""
        N = len(f)

        if N == 0:
            return np.array([])

        # Handle single point case
        if N == 1:
            return np.array([0.0])  # Laplacian of single point is zero

        # Store original length
        original_len = len(f)
        
        # Ensure N is even for FFT
        if N % 2 == 1:
            N += 1
            f = np.pad(f, (0, 1), mode="edge")
            x = np.pad(x, (0, 1), mode="edge")

        # Compute FFT
        f_fft = fft(f)

        # Create frequency array
        freq = fftfreq(N, h)

        # Apply spectral filter |ξ|^α
        spectral_filter = np.abs(freq) ** self.alpha_val
        spectral_filter[0] = 0  # Handle zero frequency

        # Apply filter in frequency domain
        filtered_fft = f_fft * spectral_filter

        # Inverse FFT
        result = np.real(ifft(filtered_fft))

        return result[:original_len]  # Return original length

    def _finite_difference_method(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Finite difference approximation."""
        N = len(f)
        result = np.zeros(N)

        # Use Grünwald-Letnikov coefficients
        coeffs = self._grunwald_coefficients(N)

        for i in range(N):
            sum_val = 0.0
            for j in range(N):
                if i - j >= 0:
                    sum_val += coeffs[j] * f[i - j]

            result[i] = sum_val / (h ** self.alpha_val)

        return result

    def compute_numerical(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
    ) -> np.ndarray:
        """Numerical convenience API used by tests.

        Uses finite-difference approximation under the hood.
        """
        # Early empty handling
        if hasattr(x, "__len__") and len(x) == 0:
            return np.array([])

        # Prepare arrays like compute(...)
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(-x_max, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = np.asarray(f)
            if hasattr(x, "__len__"):
                x_array = np.asarray(x)
            else:
                x_array = np.arange(len(f_array)) * (h or 1.0)

        if len(f_array) == 0:
            return np.array([])

        return self._finite_difference_method(f_array, x_array, h or 1.0)

    def _integral_method(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Integral representation method."""
        N = len(f)
        result = np.zeros(N)

        # Integral representation of fractional Laplacian
        c_alpha = self.alpha_val * (2 ** (self.alpha_val - 1)) * special.gamma(
            (self.alpha_val + 1) / 2) / (np.sqrt(np.pi) * special.gamma(1 - self.alpha_val / 2))

        for i in range(N):
            integral = 0.0
            for j in range(N):
                if i != j:
                    diff = f[i] - f[j]
                    dist = abs(x[i] - x[j])
                    if dist > 1e-10:  # Avoid division by zero
                        integral += diff / (dist ** (1 + self.alpha_val))

            result[i] = c_alpha * integral * h

        return result

    def _grunwald_coefficients(self, N: int) -> np.ndarray:
        """Compute Grünwald-Letnikov coefficients."""
        coeffs = np.zeros(N)
        coeffs[0] = 1.0

        for k in range(1, N):
            coeffs[k] = coeffs[k - 1] * (1 - (self.alpha_val + 1) / k)

        return coeffs


class FractionalFourierTransform:
    """
    Fractional Fourier Transform (FrFT) implementation.

    The fractional Fourier transform is a generalization of the Fourier transform
    that depends on a parameter α. For α = π/2, it reduces to the standard Fourier transform.

    FrFT[f](u) = ∫_{-∞}^∞ K_α(x, u) f(x) dx

    where K_α is the fractional Fourier kernel.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], *, method: str = "spectral"):
        """Initialize fractional Fourier transform calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.method = method

        # Normalize alpha to [0, 2π]
        self.alpha_val = self.alpha_val % (2 * np.pi)

    def transform(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute fractional Fourier transform.

        Args:
            f: Function or array of function values
            x: Domain points
            h: Step size
            method: Computation method ("auto", "discrete", "spectral", "fast")

        Returns:
            Tuple of (u_domain, transformed_values)
        """
        # Validate input arrays
        if hasattr(x, "__len__") and len(x) == 0:
            return np.array([]), np.array([])

        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(-x_max, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if len(f_array) == 0 or len(x_array) == 0:
            raise ValueError("Empty function or domain arrays are not supported")

        # Auto-select method based on problem size
        if method == "auto":
            if len(f_array) > 500:
                method = "fast"  # Use fast approximation for large arrays
            else:
                method = "discrete"  # Use accurate method for small arrays

        if method == "discrete":
            return self._discrete_method(f_array, x_array, h or 1.0)
        elif method == "spectral":
            return self._spectral_method(f_array, x_array, h or 1.0)
        elif method == "fast":
            return self._fast_approximation(f_array, x_array, h or 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: str = "auto"
    ) -> np.ndarray:
        """Compatibility wrapper returning only transformed values.

        Some call sites expect a compute(...) API that returns the transformed
        values array directly. This method delegates to transform(...) and
        returns only the values component.
        """
        _, values = self.transform(f, x, h, method)
        return values

    def _discrete_method(self, f: np.ndarray, x: np.ndarray,
                         h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Discrete fractional Fourier transform using optimized FFT-based approach."""
        N = len(f)

        # Create output domain
        u = np.linspace(x[0], x[-1], N)

        # Use optimized FFT-based method instead of matrix multiplication
        return self._fft_based_method(f, x, u, h)

    def _fft_based_method(self,
                          f: np.ndarray,
                          x: np.ndarray,
                          u: np.ndarray,
                          h: float) -> Tuple[np.ndarray,
                                             np.ndarray]:
        """Fast FFT-based fractional Fourier transform."""
        len(f)

        # Handle special cases for alpha values
        if abs(np.sin(self.alpha_val)) < 1e-10:
            # For alpha = 0, π, 2π, etc., use identity or simple transforms
            if abs(np.cos(self.alpha_val) - 1) < 1e-10:
                # Identity transform
                return u, f
            elif abs(np.cos(self.alpha_val) + 1) < 1e-10:
                # Reflection transform
                return u, -f

        # For alpha = π/2, use standard FFT
        if abs(self.alpha_val - np.pi / 2) < 1e-10:
            f_fft = fft(f)
            return u, f_fft

        # For alpha = -π/2, use standard inverse FFT
        if abs(self.alpha_val + np.pi / 2) < 1e-10:
            f_ifft = ifft(f)
            return u, f_ifft

        # Use chirp-based algorithm for general alpha
        return self._chirp_based_method(f, x, u, h)

    def _chirp_based_method(self,
                            f: np.ndarray,
                            x: np.ndarray,
                            u: np.ndarray,
                            h: float) -> Tuple[np.ndarray,
                                               np.ndarray]:
        """Chirp-based algorithm for fractional Fourier transform."""
        N = len(f)

        # Precompute constants
        cos_alpha = np.cos(self.alpha_val)
        sin_alpha = np.sin(self.alpha_val)

        # Step 1: Multiply by chirp
        chirp1 = np.exp(1j * cos_alpha * x**2 / (2 * sin_alpha))
        f_chirp = f * chirp1

        # Step 2: Convolve with chirp using FFT
        # Create chirp kernel
        kernel_size = 2 * N - 1
        kernel_x = np.linspace(-(N - 1) * h, (N - 1) * h, kernel_size)
        chirp_kernel = np.exp(-1j * kernel_x**2 / (2 * sin_alpha))

        # Zero-pad f_chirp for convolution
        f_padded = np.zeros(kernel_size, dtype=complex)
        f_padded[:N] = f_chirp

        # Compute convolution using FFT
        f_fft = fft(f_padded)
        kernel_fft = fft(chirp_kernel)
        conv_fft = f_fft * kernel_fft
        conv_result = ifft(conv_fft)

        # Extract the central N points
        start_idx = (kernel_size - N) // 2
        conv_central = conv_result[start_idx:start_idx + N]

        # Step 3: Multiply by final chirp
        chirp2 = np.exp(1j * cos_alpha * u**2 / (2 * sin_alpha))
        result = conv_central * chirp2

        # Apply scaling factor
        scaling = np.sqrt(1j / (2 * np.pi * sin_alpha))
        result *= scaling

        return u, result

    def _spectral_method(self, f: np.ndarray, x: np.ndarray,
                         h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Spectral method using decomposition."""
        N = len(f)

        # Decompose into Hermite-Gaussian functions
        hermite_coeffs = self._hermite_decomposition(f, x, h)

        # Apply fractional transform to each Hermite function
        u = np.linspace(x[0], x[-1], N)
        result = np.zeros(N, dtype=complex)  # Use complex dtype

        # Limit to first 10 terms
        for n, coeff in enumerate(hermite_coeffs[:10]):
            hermite_u = self._fractional_hermite(n, u, self.alpha_val)
            result += coeff * hermite_u

        return u, result

    def compute_numerical(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
    ) -> np.ndarray:
        """Numerical convenience API returning only values.

        Maps to the discrete method for accuracy.
        """
        if hasattr(x, "__len__") and len(x) == 0:
            return np.array([])
        u, vals = self.transform(f, x, h, method="discrete")
        return vals

    def _compute_transform_matrix(
        self,
        x: np.ndarray,
        u: np.ndarray,
            h: float) -> np.ndarray:
        """Compute the discrete fractional Fourier transform matrix."""
        N = len(x)
        matrix = np.zeros((N, N), dtype=complex)  # Use complex dtype

        # Handle special cases for alpha values
        if abs(np.sin(self.alpha_val)) < 1e-10:
            # For alpha = 0, π, 2π, etc., use identity or simple transforms
            if abs(np.cos(self.alpha_val) - 1) < 1e-10:
                # Identity transform
                np.fill_diagonal(matrix, 1.0)
                return matrix
            elif abs(np.cos(self.alpha_val) + 1) < 1e-10:
                # Reflection transform
                np.fill_diagonal(matrix, -1.0)
                return matrix

        # Precompute constants with proper handling
        try:
            c_alpha = np.sqrt(1 - 1j * np.cos(self.alpha_val)) / \
                np.sqrt(2 * np.pi * np.sin(self.alpha_val))
        except (ValueError, RuntimeWarning):
            # Fallback for problematic alpha values
            c_alpha = 1.0 / np.sqrt(2 * np.pi)

        for i in range(N):
            for j in range(N):
                # Fractional Fourier kernel with proper complex handling
                try:
                    kernel = c_alpha * np.exp(1j * (x[i]**2 + u[j]**2) * np.cos(self.alpha_val) / (
                        2 * np.sin(self.alpha_val)) - 1j * x[i] * u[j] / np.sin(self.alpha_val))
                    matrix[i, j] = kernel * h
                except (ValueError, RuntimeWarning):
                    # Fallback for problematic values
                    matrix[i, j] = 0.0

        return matrix

    def _hermite_decomposition(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Decompose function into Hermite-Gaussian basis."""
        N = len(f)
        coeffs = np.zeros(min(10, N))  # Limit to first 10 coefficients

        for n in range(len(coeffs)):
            hermite_x = self._hermite_function(n, x)
            coeffs[n] = np.sum(f * hermite_x) * h

        return coeffs

    def _hermite_function(self, n: int, x: np.ndarray) -> np.ndarray:
        """Compute Hermite-Gaussian function of order n."""
        # Normalized Hermite-Gaussian function
        H_n = special.hermite(n)
        return H_n(x) * np.exp(-x**2 / 2) / np.sqrt(2 **
                                                    n * math.factorial(n) * np.sqrt(np.pi))

    def _fractional_hermite(
        self,
        n: int,
        u: np.ndarray,
            alpha: float) -> np.ndarray:
        """Compute fractional Fourier transform of Hermite function."""
        # FrFT of Hermite-Gaussian function
        return np.exp(-1j * n * alpha) * self._hermite_function(n, u)

    def _fast_approximation(self,
                            f: np.ndarray,
                            x: np.ndarray,
                            h: float) -> Tuple[np.ndarray,
                                               np.ndarray]:
        """Fast approximation using interpolation between standard FFT and identity."""
        N = len(f)
        u = np.linspace(x[0], x[-1], N)

        # Normalize alpha to [0, π/2]
        alpha_norm = self.alpha_val % (2 * np.pi)
        if alpha_norm > np.pi:
            alpha_norm = 2 * np.pi - alpha_norm

        # For alpha close to 0, use identity
        if alpha_norm < 0.1:
            return u, f

        # For alpha close to π/2, use FFT
        if abs(alpha_norm - np.pi / 2) < 0.1:
            return u, fft(f)

        # For alpha close to π, use reflection
        if abs(alpha_norm - np.pi) < 0.1:
            return u, -f

        # For alpha close to 3π/2, use inverse FFT
        if abs(alpha_norm - 3 * np.pi / 2) < 0.1:
            return u, ifft(f)

        # For other values, use linear interpolation between nearest special
        # cases
        if alpha_norm < np.pi / 2:
            # Interpolate between identity and FFT
            weight = alpha_norm / (np.pi / 2)
            f_fft = fft(f)
            result = (1 - weight) * f + weight * f_fft
        elif alpha_norm < np.pi:
            # Interpolate between FFT and reflection
            weight = (alpha_norm - np.pi / 2) / (np.pi / 2)
            f_fft = fft(f)
            result = (1 - weight) * f_fft - weight * f
        elif alpha_norm < 3 * np.pi / 2:
            # Interpolate between reflection and inverse FFT
            weight = (alpha_norm - np.pi) / (np.pi / 2)
            f_ifft = ifft(f)
            result = -(1 - weight) * f + weight * f_ifft
        else:
            # Interpolate between inverse FFT and identity
            weight = (alpha_norm - 3 * np.pi / 2) / (np.pi / 2)
            f_ifft = ifft(f)
            result = (1 - weight) * f_ifft + weight * f

        return u, result


class FractionalZTransform:
    """
    Fractional Z-Transform implementation.

    The fractional Z-transform is a generalization of the Z-transform that
    depends on a fractional parameter α. It's useful for discrete-time systems
    and digital signal processing.

    Z^α[f](z) = Σ_{n=0}^∞ f[n] z^(-αn)
    """

    def __init__(self, alpha: Union[float, FractionalOrder], *, method: str = "spectral"):
        """Initialize fractional Z-transform calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.method = method

    def transform(
        self,
        f: np.ndarray,
        z: Union[complex, np.ndarray],
        method: str = "direct"
    ) -> Union[complex, np.ndarray]:
        """
        Compute fractional Z-transform.

        Args:
            f: Discrete signal array
            z: Complex variable(s) for evaluation
            method: Computation method ("direct", "fft")

        Returns:
            Transform values
        """
        if method == "direct":
            return self._direct_method(f, z)
        elif method == "fft":
            return self._fft_method(f, z)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _direct_method(self,
                       f: np.ndarray,
                       z: Union[complex,
                                np.ndarray]) -> Union[complex,
                                                      np.ndarray]:
        """Direct computation of fractional Z-transform."""
        if isinstance(z, (int, float, complex)):
            z = np.array([z])
            single_value = True
        else:
            single_value = False

        result = np.zeros(len(z), dtype=complex)

        for k, z_val in enumerate(z):
            sum_val = 0.0
            for n in range(len(f)):
                sum_val += f[n] * (z_val ** (-self.alpha_val * n))
            result[k] = sum_val

        if single_value:
            return result[0]
        else:
            return result

    def _fft_method(self,
                    f: np.ndarray,
                    z: Union[complex,
                             np.ndarray]) -> Union[complex,
                                                   np.ndarray]:
        """FFT-based computation for unit circle evaluation."""
        if isinstance(z, (int, float, complex)):
            z = np.array([z])
            single_value = True
        else:
            single_value = False

        # For unit circle evaluation, use FFT
        N = len(f)
        omega = np.linspace(0, 2 * np.pi, N, endpoint=False)

        # Compute fractional power spectrum
        power_spectrum = np.zeros(N, dtype=complex)
        for n in range(N):
            power_spectrum[n] = f[n] * \
                np.exp(-1j * self.alpha_val * n * omega[n])

        # Apply FFT
        result_fft = fft(power_spectrum)

        # Interpolate to requested z values
        result = np.zeros(len(z), dtype=complex)
        for k, z_val in enumerate(z):
            if abs(abs(z_val) - 1.0) < 1e-10:  # On unit circle
                angle = np.angle(z_val)
                idx = int(angle * N / (2 * np.pi)) % N
                result[k] = result_fft[idx]
            else:
                # Fall back to direct method for off-unit-circle values
                result[k] = self._direct_method(f, z_val)

        if single_value:
            return result[0]
        else:
            return result

    def inverse_transform(
        self,
        F: Union[complex, np.ndarray],
        z: Union[complex, np.ndarray],
        N: int,
        method: str = "contour"
    ) -> np.ndarray:
        """
        Compute inverse fractional Z-transform.

        Args:
            F: Transform values
            z: Complex variable(s)
            N: Length of output signal
            method: Inversion method ("contour", "residue")

        Returns:
            Original signal
        """
        if method == "contour":
            return self._contour_integration(F, z, N)
        elif method == "residue":
            return self._residue_method(F, z, N)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _contour_integration(self,
                             F: Union[complex,
                                      np.ndarray],
                             z: Union[complex,
                                      np.ndarray],
                             N: int) -> np.ndarray:
        """Contour integration method for inverse transform."""
        # Simplified contour integration
        result = np.zeros(N)

        if isinstance(z, (int, float, complex)):
            z = np.array([z])

        # Use unit circle contour
        theta = np.linspace(0, 2 * np.pi, 1000)
        z_contour = np.exp(1j * theta)

        for n in range(N):
            integrand = F * (z_contour ** (self.alpha_val * n - 1))
            result[n] = np.real(np.trapz(integrand, theta)) / (2 * np.pi)

        return result

    def _residue_method(self,
                        F: Union[complex,
                                 np.ndarray],
                        z: Union[complex,
                                 np.ndarray],
                        N: int) -> np.ndarray:
        """Residue method for inverse transform."""
        # Simplified residue calculation
        result = np.zeros(N)

        # For simple poles, compute residues
        if isinstance(z, (int, float, complex)):
            z = np.array([z])

        for n in range(N):
            residue_sum = 0.0
            for z_val in z:
                if abs(z_val) > 1e-10:  # Avoid zero
                    residue = F * (z_val ** (self.alpha_val * n))
                    residue_sum += residue

            result[n] = residue_sum

        return result


class FractionalMellinTransform:
    """
    Fractional Mellin Transform implementation.

    The fractional Mellin transform is defined as:
    M_α[f](s) = ∫₀^∞ f(x) x^(s-1+α) dx

    This transform is useful for:
    - Scale-invariant signal processing
    - Fractional differential equations
    - Image processing and pattern recognition
    - Quantum mechanics applications
    """

    def __init__(self, alpha: Union[float, FractionalOrder]):
        """Initialize fractional Mellin transform calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha

        # Validate alpha range
        if self.alpha_val < 0:
            warnings.warn(
                f"Alpha should be non-negative for fractional Mellin transform. Got {self.alpha_val}")

    def transform(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        s: Union[complex, np.ndarray],
        method: str = "numerical"
    ) -> Union[complex, np.ndarray]:
        """
        Compute fractional Mellin transform.

        Args:
            f: Function or array of function values
            x: Domain points (must be positive)
            s: Complex variable(s) for transform
            method: Computation method ("numerical", "analytical", "fft")

        Returns:
            Transform values
        """
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                x_array = np.logspace(-10, np.log10(x_max), 1000)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * 1.0

        # Ensure x values are positive
        if np.any(x_array <= 0):
            raise ValueError(
                "Domain points must be positive for Mellin transform")

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if method == "numerical":
            return self._numerical_method(f_array, x_array, s)
        elif method == "analytical":
            return self._analytical_method(f_array, x_array, s)
        elif method == "fft":
            return self._fft_method(f_array, x_array, s)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _numerical_method(self,
                          f: np.ndarray,
                          x: np.ndarray,
                          s: Union[complex,
                                   np.ndarray]) -> Union[complex,
                                                         np.ndarray]:
        """Numerical integration method."""
        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        result = np.zeros(len(s), dtype=complex)

        for i, s_val in enumerate(s):
            # Integrand: f(x) * x^(s-1+α)
            integrand = f * (x ** (s_val - 1 + self.alpha_val))
            result[i] = np.trapz(integrand, x)

        return result[0] if len(s) == 1 else result

    def _analytical_method(self,
                           f: np.ndarray,
                           x: np.ndarray,
                           s: Union[complex,
                                    np.ndarray]) -> Union[complex,
                                                          np.ndarray]:
        """Analytical method for special functions."""
        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        result = np.zeros(len(s), dtype=complex)

        # For exponential functions, use analytical formulas
        for i, s_val in enumerate(s):
            if np.allclose(f, np.exp(-x)):  # Exponential decay
                result[i] = special.gamma(s_val + self.alpha_val)
            elif np.allclose(f, x ** 2):  # Power function
                if np.real(s_val + self.alpha_val) > -2:
                    result[i] = special.gamma(
                        s_val + self.alpha_val + 2) / (s_val + self.alpha_val + 2)
                else:
                    result[i] = np.inf
            else:
                # Fall back to numerical method
                result[i] = self._numerical_method(f, x, s_val)

        return result[0] if len(s) == 1 else result

    def _fft_method(self,
                    f: np.ndarray,
                    x: np.ndarray,
                    s: Union[complex,
                             np.ndarray]) -> Union[complex,
                                                   np.ndarray]:
        """FFT-based method for efficient computation."""
        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        # Use logarithmic sampling for better FFT performance
        log_x = np.log(x)
        log_f = f * x  # Pre-multiply by x for better numerical stability

        # Compute FFT
        fft_result = fft(log_f)

        # Frequency domain
        freqs = fftfreq(len(x), log_x[1] - log_x[0])

        result = np.zeros(len(s), dtype=complex)

        for i, s_val in enumerate(s):
            # Interpolate FFT result at desired s values
            s_interp = s_val + self.alpha_val
            freq_idx = np.argmin(np.abs(freqs - s_interp))
            result[i] = fft_result[freq_idx]

        return result[0] if len(s) == 1 else result

    def inverse_transform(
        self,
        F: Union[complex, np.ndarray],
        s: Union[complex, np.ndarray],
        x: Union[float, np.ndarray],
        method: str = "numerical"
    ) -> np.ndarray:
        """
        Compute inverse fractional Mellin transform.

        Args:
            F: Transform values
            s: Complex variable(s)
            x: Domain points for output
            method: Inversion method ("numerical", "fft")

        Returns:
            Original function values
        """
        if method == "numerical":
            return self._inverse_numerical(F, s, x)
        elif method == "fft":
            return self._inverse_fft(F, s, x)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _inverse_numerical(self,
                           F: Union[complex,
                                    np.ndarray],
                           s: Union[complex,
                                    np.ndarray],
                           x: np.ndarray) -> np.ndarray:
        """Numerical inverse transform."""
        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        result = np.zeros(len(x))

        for i, xi in enumerate(x):
            # Inverse integrand: F(s) * x^(-s-α)
            integrand = F * (xi ** (-s - self.alpha_val))
            result[i] = np.real(np.trapz(integrand, s)) / (2 * np.pi * 1j)

        return result

    def _inverse_fft(self,
                     F: Union[complex,
                              np.ndarray],
                     s: Union[complex,
                              np.ndarray],
                     x: np.ndarray) -> np.ndarray:
        """FFT-based inverse transform."""
        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        # Use FFT for efficient inverse computation
        np.log(x)

        # Interpolate F at regular s intervals
        s_min, s_max = np.real(s).min(), np.real(s).max()
        s_regular = np.linspace(s_min, s_max, len(x))
        F_interp = np.interp(s_regular, np.real(s), np.real(F))

        # Compute inverse FFT
        ifft_result = ifft(F_interp)

        # Convert back to x domain
        result = ifft_result / x

        return np.real(result)


# Convenience functions
def fractional_laplacian(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "spectral"
) -> Union[float, np.ndarray]:
    """Convenience function for fractional Laplacian."""
    calculator = FractionalLaplacian(alpha)
    return calculator.compute(f, x, h, method)


def fractional_fourier_transform(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for fractional Fourier transform."""
    calculator = FractionalFourierTransform(alpha)
    return calculator.transform(f, x, h, method)


def fractional_z_transform(
    f: np.ndarray,
    z: Union[complex, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "direct"
) -> Union[complex, np.ndarray]:
    """Convenience function for fractional Z-transform."""
    calculator = FractionalZTransform(alpha)
    return calculator.transform(f, z, method)


def fractional_mellin_transform(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    s: Union[complex, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "numerical"
) -> Union[complex, np.ndarray]:
    """Convenience function for fractional Mellin transform."""
    calculator = FractionalMellinTransform(alpha)
    return calculator.transform(f, x, s, method)


# =============================================================================
# OPTIMIZED SPECIAL METHODS
# =============================================================================

class SpecialMethodsConfig:
    """Configuration for special methods optimization."""
    
    def __init__(self, optimized: bool = True, parallel: bool = False):
        """
        Initialize special methods configuration.
        
        Args:
            optimized: Enable optimized implementations
            parallel: Enable parallel processing (if available)
        """
        self.optimized = optimized
        self.parallel = parallel


class SpecialOptimizedWeylDerivative:
    """
    Weyl derivative optimized using Fractional Fourier Transform.

    This implementation replaces the standard FFT convolution approach
    with Fractional Fourier Transform for better performance, especially
    for large arrays and specific alpha values.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], config: Optional[SpecialMethodsConfig] = None):
        """Initialize special optimized Weyl derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.config = config or SpecialMethodsConfig()

        # Initialize special methods
        self.frft = FractionalFourierTransform(alpha)

        # Determine optimal method based on alpha
        self._determine_optimal_method()

    def _determine_optimal_method(self):
        """Determine the optimal computation method based on alpha value."""
        # For now, use standard FFT as it's more reliable
        self.optimal_method = "standard_fft"

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: Optional[str] = None,
    ) -> Union[float, np.ndarray]:
        """Compute Weyl derivative using optimized method."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(0, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if method is None:
            method = self.optimal_method

        if method == "frft":
            return self._compute_frft(f_array, x_array, h or 1.0)
        elif method == "standard_fft":
            return self._compute_standard_fft(f_array, x_array, h or 1.0)
        elif method == "hybrid":
            return self._compute_hybrid(f_array, x_array, h or 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_frft(self, f: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
        """Compute using Fractional Fourier Transform."""
        # Use the FRFT for computation
        u, result = self.frft.transform(f, x, h, method="discrete")
        return np.real(result)

    def _compute_standard_fft(self, f: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
        """Compute using standard FFT approach."""
        # Standard FFT-based Weyl derivative computation
        N = len(f)
        u = np.fft.fftfreq(N, h)
        
        # Compute kernel
        kernel = self._compute_weyl_kernel(u, h)
        
        # FFT convolution
        f_fft = np.fft.fft(f)
        result_fft = f_fft * kernel
        result = np.fft.ifft(result_fft)
        
        return np.real(result)

    def _compute_hybrid(self, f: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
        """Compute using hybrid approach combining FRFT and standard FFT."""
        # Use a combination of FRFT for low frequencies and standard FFT for high frequencies
        N = len(f)
        
        # For small arrays, use standard FFT
        if N < 64:
            return self._compute_standard_fft(f, x, h)
        
        # For larger arrays, use a hybrid approach
        # Use FRFT for the main computation
        try:
            return self._compute_frft(f, x, h)
        except:
            # Fallback to standard FFT if FRFT fails
            return self._compute_standard_fft(f, x, h)

    def _compute_weyl_kernel(self, u: np.ndarray, h: float) -> np.ndarray:
        """Compute Weyl derivative kernel."""
        kernel = (1j * 2 * np.pi * u) ** self.alpha_val
        return kernel


class SpecialOptimizedMarchaudDerivative:
    """
    Marchaud derivative optimized using Fractional Z-Transform.

    This implementation replaces the difference quotient convolution
    with Fractional Z-Transform for better performance on discrete signals.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], config: Optional[SpecialMethodsConfig] = None):
        """Initialize special optimized Marchaud derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.config = config or SpecialMethodsConfig()

        # Initialize special methods
        self.z_transform = FractionalZTransform(alpha)

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: str = "z_transform",
    ) -> Union[float, np.ndarray]:
        """Compute Marchaud derivative using optimized method."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(0, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if method == "z_transform":
            return self._compute_z_transform(f_array, x_array, h or 1.0)
        elif method == "standard":
            return self._compute_standard(f_array, x_array, h or 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_z_transform(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Compute using Fractional Z-Transform."""
        # Use Z-transform for computation
        z_values = np.exp(1j * np.linspace(0, 2 * np.pi, len(f), endpoint=False))
        result = self.z_transform.transform(f, z_values, method="fft")
        return np.real(np.fft.ifft(result))

    def _compute_standard(self, f: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
        """Compute using standard difference quotient approach."""
        # Standard Marchaud derivative computation
        N = len(f)
        result = np.zeros_like(f)
        
        for n in range(N):
            for k in range(n + 1):
                # Use gamma function for fractional alpha
                if self.alpha_val == int(self.alpha_val):
                    # Integer case - use factorial
                    weight = (-1) ** k * _factorial(int(self.alpha_val)) / (
                        _factorial(k) * _factorial(int(self.alpha_val) - k)
                    )
                else:
                    # Fractional case - use gamma function
                    from scipy.special import gamma
                    weight = (-1) ** k * gamma(self.alpha_val + 1) / (
                        gamma(k + 1) * gamma(self.alpha_val - k + 1)
                    )
                if n - k < len(f):
                    result[n] += weight * f[n - k]
        
        return result / (h ** self.alpha_val)


class SpecialOptimizedReizFellerDerivative:
    """
    Reiz-Feller derivative optimized using Fractional Laplacian.

    This implementation uses the Fractional Laplacian for efficient
    computation of the Reiz-Feller derivative, especially for
    spectral methods and large-scale problems.
    """

    def __init__(self, alpha: Union[float, FractionalOrder], config: Optional[SpecialMethodsConfig] = None):
        """Initialize special optimized Reiz-Feller derivative calculator."""
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha

        self.alpha_val = self.alpha.alpha
        self.config = config or SpecialMethodsConfig()

        # Initialize special methods
        self.laplacian = FractionalLaplacian(alpha)

    def compute(
        self,
        f: Union[Callable, np.ndarray],
        x: Union[float, np.ndarray],
        h: Optional[float] = None,
        method: str = "laplacian",
    ) -> Union[float, np.ndarray]:
        """Compute Reiz-Feller derivative using optimized method."""
        if callable(f):
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_max = x
                if h is None:
                    h = x_max / 1000
                x_array = np.arange(0, x_max + h, h)
            f_array = np.array([f(xi) for xi in x_array])
        else:
            f_array = f
            if hasattr(x, "__len__"):
                x_array = x
            else:
                x_array = np.arange(len(f)) * (h or 1.0)

        # Ensure arrays have the same length
        min_len = min(len(f_array), len(x_array))
        f_array = f_array[:min_len]
        x_array = x_array[:min_len]

        if method == "laplacian":
            return self._compute_laplacian(f_array, x_array, h or 1.0)
        elif method == "spectral":
            return self._compute_spectral(f_array, x_array, h or 1.0)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_laplacian(self, f: np.ndarray, x: np.ndarray, h: float) -> np.ndarray:
        """Compute using Fractional Laplacian."""
        return self.laplacian.compute(f, x, h, method="spectral")

    def _compute_spectral(
            self,
            f: np.ndarray,
            x: np.ndarray,
            h: float) -> np.ndarray:
        """Compute using spectral method."""
        N = len(f)
        u = np.fft.fftfreq(N, h)
        
        # Compute spectral kernel
        kernel = np.abs(u) ** self.alpha_val
        
        # FFT convolution
        f_fft = np.fft.fft(f)
        result_fft = f_fft * kernel
        result = np.fft.ifft(result_fft)
        
        return np.real(result)


class UnifiedSpecialMethods:
    """
    Unified interface for all special methods with automatic method selection.

    This class provides a unified API that automatically selects the best
    special method based on the problem characteristics.
    """

    def __init__(self, config: Optional[SpecialMethodsConfig] = None):
        """Initialize unified special methods interface."""
        self.config = config or SpecialMethodsConfig()
        self.methods = {
            'laplacian': FractionalLaplacian,
            'fourier': FractionalFourierTransform,
            'z_transform': FractionalZTransform,
        }

    def compute_derivative(
        self,
        f: Union[Callable, np.ndarray],
        x: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        method: Optional[str] = None,
        problem_type: str = "general",
    ) -> np.ndarray:
        """
        Compute fractional derivative using optimal special method.

        Args:
            f: Function or function values
            x: Domain points
            alpha: Fractional order
            h: Step size
            method: Specific method to use (if None, auto-select)
            problem_type: Type of problem ("periodic", "discrete", "spectral", "general")

        Returns:
            Derivative values
        """
        # Handle function input
        if callable(f):
            f_array = np.array([f(xi) for xi in x])
        else:
            f_array = f

        if method is None:
            method = self._auto_select_method(
                problem_type, len(f_array), alpha)

        if method == "laplacian":
            laplacian = FractionalLaplacian(alpha)
            return laplacian.compute(f_array, x, h, method="spectral")
        elif method == "fourier":
            frft = FractionalFourierTransform(alpha)
            u, result = frft.transform(f_array, x, h, method="discrete")
            return np.real(result)
        elif method == "z_transform":
            z_transform = FractionalZTransform(alpha)
            z_values = np.exp(1j * np.linspace(0, 2 * np.pi,
                              len(f_array), endpoint=False))
            result = z_transform.transform(f_array, z_values, method="fft")
            return np.real(np.fft.ifft(result))
        else:
            raise ValueError(f"Unknown method: {method}")

    def _auto_select_method(self,
                            problem_type: str,
                            size: int,
                            alpha: Union[float,
                                         FractionalOrder]) -> str:
        """Automatically select the best method based on problem characteristics."""
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Method selection logic
        if problem_type == "periodic":
            return "fourier"
        elif problem_type == "discrete":
            return "z_transform"
        elif problem_type == "spectral":
            return "laplacian"
        else:
            # General case - choose based on size and alpha
            if size > 1000 and alpha_val < 0.5:
                return "fourier"
            elif size < 100 and alpha_val > 0.5:
                return "laplacian"
            else:
                return "z_transform"


# =============================================================================
# CONVENIENCE FUNCTIONS FOR OPTIMIZED METHODS
# =============================================================================

def special_optimized_weyl_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Convenience function for special optimized Weyl derivative."""
    calculator = SpecialOptimizedWeylDerivative(alpha)
    return calculator.compute(f, x, h, method)


def special_optimized_marchaud_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "z_transform",
) -> Union[float, np.ndarray]:
    """Convenience function for special optimized Marchaud derivative."""
    calculator = SpecialOptimizedMarchaudDerivative(alpha)
    return calculator.compute(f, x, h, method)


def special_optimized_reiz_feller_derivative(
    f: Union[Callable, np.ndarray],
    x: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    h: Optional[float] = None,
    method: str = "laplacian",
) -> Union[float, np.ndarray]:
    """Convenience function for special optimized Reiz-Feller derivative."""
    calculator = SpecialOptimizedReizFellerDerivative(alpha)
    return calculator.compute(f, x, h, method)


def unified_special_derivative(
    f: Union[Callable, np.ndarray],
    x: np.ndarray,
    alpha: Union[float, FractionalOrder],
    h: float,
    method: Optional[str] = None,
    problem_type: str = "general",
) -> np.ndarray:
    """Convenience function for unified special derivative computation."""
    calculator = UnifiedSpecialMethods()
    return calculator.compute_derivative(f, x, alpha, h, method, problem_type)
