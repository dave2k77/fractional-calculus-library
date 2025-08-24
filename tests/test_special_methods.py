"""
Tests for Special Fractional Calculus Methods

This module tests the special fractional calculus methods:
- Fractional Laplacian
- Fractional Fourier Transform  
- Fractional Z-Transform

These methods are fundamental for advanced fractional calculus applications.
"""

import pytest
import numpy as np
import time
from typing import Callable

from hpfracc.algorithms.special_methods import (
    FractionalLaplacian,
    FractionalFourierTransform,
    FractionalZTransform,
    fractional_laplacian,
    fractional_fourier_transform,
    fractional_z_transform,
)

from hpfracc.core.definitions import FractionalOrder


class TestFractionalLaplacian:
    """Test Fractional Laplacian implementation."""

    def test_fractional_laplacian_basic(self):
        """Test basic fractional Laplacian computation."""
        alpha = 1.0
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-(x**2))

        # Test spectral method
        laplacian = FractionalLaplacian(alpha)
        result = laplacian.compute(f, x, method="spectral")

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_fractional_laplacian_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        x = np.linspace(-3, 3, 50)

        def f(x):
            return np.sin(x)

        laplacian = FractionalLaplacian(alpha)

        # Test all methods
        result_spectral = laplacian.compute(f, x, method="spectral")
        result_finite = laplacian.compute(f, x, method="finite_difference")
        result_integral = laplacian.compute(f, x, method="integral")

        # All methods should produce results
        assert len(result_spectral) == len(x)
        assert len(result_finite) == len(x)
        assert len(result_integral) == len(x)

        # Check for valid results
        assert not np.any(np.isnan(result_spectral))
        assert not np.any(np.isnan(result_finite))
        assert not np.any(np.isnan(result_integral))

    def test_fractional_laplacian_edge_cases(self):
        """Test edge cases for fractional Laplacian."""
        alpha = 1.5
        x = np.linspace(-2, 2, 20)

        def f(x):
            return x**2

        laplacian = FractionalLaplacian(alpha)

        # Test with different alpha values
        for alpha_val in [0.1, 0.5, 1.0, 1.5, 1.9]:
            laplacian.alpha_val = alpha_val
            result = laplacian.compute(f, x, method="spectral")
            assert len(result) == len(x)
            assert not np.any(np.isnan(result))

    def test_fractional_laplacian_convenience(self):
        """Test convenience function."""
        alpha = 0.8
        x = np.linspace(-1, 1, 30)
        f = np.sin(x)

        result = fractional_laplacian(f, x, alpha, method="spectral")

        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestFractionalFourierTransform:
    """Test Fractional Fourier Transform implementation."""

    def test_fractional_fourier_transform_basic(self):
        """Test basic fractional Fourier transform."""
        alpha = np.pi / 2  # Standard Fourier transform
        x = np.linspace(-5, 5, 100)

        def f(x):
            return np.exp(-(x**2))

        # Test discrete method
        frft = FractionalFourierTransform(alpha)
        u, result = frft.transform(f, x, method="discrete")

        assert len(u) == len(x)
        assert len(result) == len(x)
        assert not np.any(np.isnan(result))

    def test_fractional_fourier_transform_methods(self):
        """Test different computation methods."""
        alpha = 0.5
        x = np.linspace(-5, 5, 100)
        f = np.exp(-x**2)

        # Test different methods
        u_discrete, result_discrete = fractional_fourier_transform(f, x, alpha, method="discrete")
        u_spectral, result_spectral = fractional_fourier_transform(f, x, alpha, method="spectral")
        u_fast, result_fast = fractional_fourier_transform(f, x, alpha, method="fast")
        u_auto, result_auto = fractional_fourier_transform(f, x, alpha, method="auto")

        # All methods should produce results
        assert len(result_discrete) == len(x)
        assert len(result_spectral) == len(x)
        assert len(result_fast) == len(x)
        assert len(result_auto) == len(x)

        # Check for valid results
        assert not np.any(np.isnan(result_discrete))
        assert not np.any(np.isnan(result_spectral))
        assert not np.any(np.isnan(result_fast))
        assert not np.any(np.isnan(result_auto))

    def test_fractional_fourier_transform_alpha_values(self):
        """Test with different alpha values."""
        x = np.linspace(-2, 2, 30)

        def f(x):
            return np.exp(-x**2)

        frft = FractionalFourierTransform(0.5)

        # Test various alpha values
        for alpha in [0.1, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
            frft.alpha_val = alpha
            u, result = frft.transform(f, x, method="discrete")
            assert len(result) == len(x)
            assert not np.any(np.isnan(result))

    def test_fractional_fourier_transform_convenience(self):
        """Test convenience function."""
        alpha = np.pi / 3
        x = np.linspace(-1, 1, 20)
        f = np.sin(x)

        u, result = fractional_fourier_transform(f, x, alpha, method="discrete")

        assert len(u) == len(x)
        assert len(result) == len(x)
        assert not np.any(np.isnan(result))


class TestFractionalZTransform:
    """Test Fractional Z-Transform implementation."""

    def test_fractional_z_transform_basic(self):
        """Test basic fractional Z-transform."""
        alpha = 0.5
        f = np.array([1, 2, 3, 4, 5])
        z = 0.5 + 0.5j

        # Test direct method
        z_transform = FractionalZTransform(alpha)
        result = z_transform.transform(f, z, method="direct")

        assert isinstance(result, complex)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_fractional_z_transform_methods(self):
        """Test different computation methods."""
        alpha = 0.3
        f = np.array([1, -1, 1, -1, 1])
        z_values = np.array([0.5, 1.0, 1.5])

        z_transform = FractionalZTransform(alpha)

        # Test both methods
        result_direct = z_transform.transform(f, z_values, method="direct")
        result_fft = z_transform.transform(f, z_values, method="fft")

        # Both methods should produce results
        assert len(result_direct) == len(z_values)
        assert len(result_fft) == len(z_values)

        # Check for valid results
        assert not np.any(np.isnan(result_direct))
        assert not np.any(np.isnan(result_fft))

    def test_fractional_z_transform_unit_circle(self):
        """Test Z-transform on unit circle."""
        alpha = 0.7
        f = np.array([1, 2, 3, 4, 5])
        
        # Points on unit circle
        theta = np.linspace(0, 2*np.pi, 10, endpoint=False)
        z_unit = np.exp(1j * theta)

        z_transform = FractionalZTransform(alpha)
        result = z_transform.transform(f, z_unit, method="fft")

        assert len(result) == len(z_unit)
        assert not np.any(np.isnan(result))

    def test_fractional_z_transform_inverse(self):
        """Test inverse Z-transform."""
        alpha = 0.5
        f_original = np.array([1, 2, 3, 4, 5])
        z = 0.5

        z_transform = FractionalZTransform(alpha)
        
        # Forward transform
        F = z_transform.transform(f_original, z, method="direct")
        
        # Inverse transform
        f_reconstructed = z_transform.inverse_transform(F, z, len(f_original), method="contour")

        assert len(f_reconstructed) == len(f_original)
        assert not np.any(np.isnan(f_reconstructed))

    def test_fractional_z_transform_convenience(self):
        """Test convenience function."""
        alpha = 0.6
        f = np.array([1, -1, 1, -1])
        z = 0.8

        result = fractional_z_transform(f, z, alpha, method="direct")

        assert isinstance(result, complex)
        assert not np.isnan(result)


class TestSpecialMethodsIntegration:
    """Test integration between special methods."""

    def test_laplacian_fourier_connection(self):
        """Test connection between Laplacian and Fourier transform."""
        alpha = 1.0
        x = np.linspace(-3, 3, 50)

        def f(x):
            return np.exp(-(x**2))

        # Compute fractional Laplacian using spectral method
        laplacian = FractionalLaplacian(alpha)
        result_laplacian = laplacian.compute(f, x, method="spectral")

        # Compute fractional Fourier transform
        frft = FractionalFourierTransform(alpha)
        u, result_frft = frft.transform(f, x, method="discrete")

        # Both should produce valid results
        assert len(result_laplacian) == len(x)
        assert len(result_frft) == len(x)
        assert not np.any(np.isnan(result_laplacian))
        assert not np.any(np.isnan(result_frft))

    def test_z_transform_discrete_connection(self):
        """Test connection between Z-transform and discrete methods."""
        alpha = 0.5
        f = np.array([1, 2, 3, 4, 5])
        
        # Z-transform on unit circle
        z_unit = np.exp(1j * np.linspace(0, 2*np.pi, len(f), endpoint=False))
        
        z_transform = FractionalZTransform(alpha)
        result_z = z_transform.transform(f, z_unit, method="fft")

        # Should produce valid results
        assert len(result_z) == len(z_unit)
        assert not np.any(np.isnan(result_z))


class TestSpecialMethodsPerformance:
    """Test performance of special methods."""

    def test_laplacian_performance(self):
        """Test performance of fractional Laplacian."""
        alpha = 1.0
        x = np.linspace(-5, 5, 200)

        def f(x):
            return np.sin(x)

        laplacian = FractionalLaplacian(alpha)

        # Time spectral method
        start_time = time.time()
        result = laplacian.compute(f, x, method="spectral")
        spectral_time = time.time() - start_time

        # Time finite difference method
        start_time = time.time()
        result = laplacian.compute(f, x, method="finite_difference")
        finite_time = time.time() - start_time

        # Both should complete in reasonable time
        assert spectral_time < 10.0  # Less than 10 seconds
        assert finite_time < 10.0

    def test_fourier_transform_performance(self):
        """Test performance of fractional Fourier transform."""
        alpha = np.pi / 4  # Use a non-special case
        x = np.linspace(-3, 3, 100)

        def f(x):
            return np.cos(x)

        frft = FractionalFourierTransform(alpha)

        # Time discrete method
        start_time = time.time()
        u, result = frft.transform(f, x, method="discrete")
        discrete_time = time.time() - start_time

        # Time spectral method
        start_time = time.time()
        u, result = frft.transform(f, x, method="spectral")
        spectral_time = time.time() - start_time

        # Time fast method
        start_time = time.time()
        u, result = frft.transform(f, x, method="fast")
        fast_time = time.time() - start_time

        # Time auto method
        start_time = time.time()
        u, result = frft.transform(f, x, method="auto")
        auto_time = time.time() - start_time

        print(f"\nFractional Fourier Transform Performance (size=100):")
        print(f"  Discrete method: {discrete_time:.4f}s")
        print(f"  Spectral method: {spectral_time:.4f}s")
        print(f"  Fast method: {fast_time:.4f}s")
        print(f"  Auto method: {auto_time:.4f}s")

        # Test with larger array
        x_large = np.linspace(-5, 5, 1000)
        f_large = np.cos(x_large)

        # Time auto method for large array
        start_time = time.time()
        u, result = frft.transform(f_large, x_large, method="auto")
        auto_large_time = time.time() - start_time

        print(f"\nLarge array performance (size=1000):")
        print(f"  Auto method: {auto_large_time:.4f}s")

        # All methods should complete in reasonable time
        assert discrete_time < 10.0
        assert spectral_time < 10.0
        assert fast_time < 1.0  # Fast method should be very fast
        assert auto_time < 10.0
        assert auto_large_time < 10.0

    def test_z_transform_performance(self):
        """Test performance of fractional Z-transform."""
        alpha = 0.5
        f = np.random.random(50)
        z = np.exp(1j * np.linspace(0, 2*np.pi, 20))

        z_transform = FractionalZTransform(alpha)

        # Time direct method
        start_time = time.time()
        result = z_transform.transform(f, z, method="direct")
        direct_time = time.time() - start_time

        # Time FFT method
        start_time = time.time()
        result = z_transform.transform(f, z, method="fft")
        fft_time = time.time() - start_time

        # Both should complete in reasonable time
        assert direct_time < 5.0
        assert fft_time < 5.0


class TestSpecialMethodsEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_alpha_values(self):
        """Test behavior with invalid alpha values."""
        x = np.linspace(-1, 1, 10)

        # Test fractional Laplacian with invalid alpha
        with pytest.warns(UserWarning):
            laplacian = FractionalLaplacian(2.5)  # Should warn
            result = laplacian.compute(np.sin(x), x, method="spectral")
            assert len(result) == len(x)

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        alpha = 0.5
        x = np.array([])
        f = np.array([])

        # Test Z-transform with empty array
        z_transform = FractionalZTransform(alpha)
        result = z_transform.transform(f, 0.5, method="direct")
        assert isinstance(result, complex)

    def test_single_point(self):
        """Test behavior with single point."""
        alpha = 0.5
        x = np.array([1.0])
        f = np.array([1.0])

        # Test fractional Laplacian with single point
        laplacian = FractionalLaplacian(alpha)
        result = laplacian.compute(f, x, method="spectral")
        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_invalid_methods(self):
        """Test error handling for invalid methods."""
        alpha = 0.5
        x = np.linspace(-1, 1, 10)

        # Test invalid method for Laplacian
        laplacian = FractionalLaplacian(alpha)
        with pytest.raises(ValueError):
            laplacian.compute(np.sin(x), x, method="invalid_method")

        # Test invalid method for Fourier transform
        frft = FractionalFourierTransform(alpha)
        with pytest.raises(ValueError):
            frft.transform(np.sin(x), x, method="invalid_method")

        # Test invalid method for Z-transform
        z_transform = FractionalZTransform(alpha)
        with pytest.raises(ValueError):
            z_transform.transform(np.array([1, 2, 3]), 0.5, method="invalid_method")


if __name__ == "__main__":
    # Run performance tests
    print("Running special methods performance tests...")

    # Test fractional Laplacian performance
    alpha = 1.0
    x = np.linspace(-5, 5, 500)

    def f(x):
        return np.exp(-(x**2))

    laplacian = FractionalLaplacian(alpha)

    start_time = time.time()
    result = laplacian.compute(f, x, method="spectral")
    spectral_time = time.time() - start_time

    start_time = time.time()
    result = laplacian.compute(f, x, method="finite_difference")
    finite_time = time.time() - start_time

    print(f"Fractional Laplacian Performance:")
    print(f"  Spectral method: {spectral_time:.4f}s")
    print(f"  Finite difference: {finite_time:.4f}s")

    # Test fractional Fourier transform performance
    frft = FractionalFourierTransform(np.pi/2)

    start_time = time.time()
    u, result = frft.transform(f, x, method="discrete")
    discrete_time = time.time() - start_time

    start_time = time.time()
    u, result = frft.transform(f, x, method="spectral")
    spectral_time = time.time() - start_time

    start_time = time.time()
    u, result = frft.transform(f, x, method="fast")
    fast_time = time.time() - start_time

    start_time = time.time()
    u, result = frft.transform(f, x, method="auto")
    auto_time = time.time() - start_time

    print(f"\nFractional Fourier Transform Performance:")
    print(f"  Discrete method: {discrete_time:.4f}s")
    print(f"  Spectral method: {spectral_time:.4f}s")
    print(f"  Fast method: {fast_time:.4f}s")
    print(f"  Auto method: {auto_time:.4f}s")

    # Test fractional Z-transform performance
    f_discrete = np.random.random(100)
    z_values = np.exp(1j * np.linspace(0, 2*np.pi, 50))

    z_transform = FractionalZTransform(0.5)

    start_time = time.time()
    result = z_transform.transform(f_discrete, z_values, method="direct")
    direct_time = time.time() - start_time

    start_time = time.time()
    result = z_transform.transform(f_discrete, z_values, method="fft")
    fft_time = time.time() - start_time

    print(f"\nFractional Z-Transform Performance:")
    print(f"  Direct method: {direct_time:.4f}s")
    print(f"  FFT method: {fft_time:.4f}s")

    print("\nAll special methods tests completed successfully!")
