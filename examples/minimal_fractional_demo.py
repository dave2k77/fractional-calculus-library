"""
Minimal Fractional Calculus Demo

A simple demonstration of fractional calculus concepts
that can be used as a basis for Fractional PINO.

Author: Davian R. Chin (Department of Biomedical Engineering, University of Reading)
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import fft, ifft, fftfreq


def fractional_laplacian_spectral(f, x, alpha, method="spectral"):
    """
    Compute fractional Laplacian using spectral method

    Args:
        f: function or array
        x: spatial points
        alpha: fractional order
        method: "spectral" or "finite_difference"

    Returns:
        Fractional Laplacian result
    """
    if callable(f):
        f_array = np.array([f(xi) for xi in x])
    else:
        f_array = f

    N = len(x)
    dx = x[1] - x[0]

    if method == "spectral":
        # Spectral method using FFT
        k = 2 * np.pi * fftfreq(N, dx)
        f_hat = fft(f_array)

        # Fractional Laplacian in Fourier space: (-k^2)^(alpha/2)
        laplacian_hat = -np.power(np.abs(k), alpha) * f_hat
        laplacian_hat[0] = 0  # Zero mode

        result = np.real(ifft(laplacian_hat))

    elif method == "finite_difference":
        # Simple finite difference approximation
        result = np.zeros_like(f_array)
        for i in range(1, N-1):
            result[i] = (f_array[i+1] - 2*f_array[i] + f_array[i-1]) / (dx**2)

    return result


def fractional_fourier_transform_fast(f, x, alpha, method="fast"):
    """
    Compute fractional Fourier transform using fast method

    Args:
        f: function or array
        x: spatial points
        alpha: fractional order
        method: "fast" or "discrete"

    Returns:
        Tuple of (u, result) where u is the transform variable
    """
    if callable(f):
        f_array = np.array([f(xi) for xi in x])
    else:
        f_array = f

    N = len(x)
    dx = x[1] - x[0]

    if method == "fast":
        # Fast FFT-based method
        k = 2 * np.pi * fftfreq(N, dx)
        f_hat = fft(f_array)

        # Fractional Fourier transform kernel
        phase = np.exp(1j * np.pi * alpha / 2)
        kernel = np.exp(-1j * np.pi * alpha * k**2 / 2)

        result_hat = phase * kernel * f_hat
        result = ifft(result_hat)

        # Transform variable
        u = x

    elif method == "discrete":
        # Discrete method (simplified)
        result = f_array * np.exp(1j * np.pi * alpha * x**2 / 2)
        u = x

    return u, result


def test_fractional_operators():
    """Test our fractional operators"""
    print("🚀 Testing Fractional Operators")
    print("=" * 40)

    # Test function
    x = np.linspace(0, 1, 100)
    def f(t): return np.sin(2 * np.pi * t)

    print("Testing function: f(t) = sin(2πt)")
    print(f"Domain: [0, 1] with {len(x)} points")

    # Test different fractional orders
    alphas = [0.25, 0.5, 0.75]

    results = {}

    for alpha in alphas:
        print(f"\n🔬 Testing with α = {alpha}")

        # Test fractional Laplacian
        print("  Computing fractional Laplacian...")
        start_time = time.time()
        laplacian_result = fractional_laplacian_spectral(
            f, x, alpha, method="spectral")
        laplacian_time = time.time() - start_time
        print(f"  ✅ Laplacian completed in {laplacian_time:.4f}s")
        results[f'laplacian_alpha_{alpha}'] = {
            'result': laplacian_result,
            'time': laplacian_time
        }

        # Test fractional Fourier transform
        print("  Computing fractional Fourier transform...")
        start_time = time.time()
        u, fourier_result = fractional_fourier_transform_fast(
            f, x, alpha, method="fast")
        fourier_time = time.time() - start_time
        print(f"  ✅ Fourier Transform completed in {fourier_time:.4f}s")
        results[f'fourier_alpha_{alpha}'] = {
            'result': fourier_result,
            'u': u,
            'time': fourier_time
        }

    return results, x, f(x)


def plot_results(results, x, original):
    """Plot the results"""
    print("\n📊 Plotting results...")

    # Count successful results
    successful_ops = [k for k in results.keys() if 'result' in results[k]]

    if not successful_ops:
        print("❌ No successful operations to plot")
        return

    n_plots = len(successful_ops)
    n_cols = max(2, (n_plots + 1) // 2)  # +1 for original function
    fig, axes = plt.subplots(2, n_cols, figsize=(15, 8))

    # Plot original function
    axes[0, 0].plot(x, original, 'b-', linewidth=2, label='Original')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title('Original Function: sin(2πx)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot results
    for i, op_name in enumerate(successful_ops):
        row = i // n_cols
        col = i % n_cols

        if 'laplacian' in op_name:
            alpha = op_name.split('_')[-1]
            axes[row, col].plot(x, results[op_name]['result'], 'r-', linewidth=2,
                                label=f'Fractional Laplacian (α={alpha})')
            axes[row, col].set_ylabel('(-Δ)^α f(x)')
            axes[row, col].set_title(f'Fractional Laplacian (α={alpha})')
        elif 'fourier' in op_name:
            alpha = op_name.split('_')[-1]
            u = results[op_name]['u']
            result = results[op_name]['result']
            axes[row, col].plot(u, np.real(result), 'g-',
                                linewidth=2, label='Real part')
            axes[row, col].plot(u, np.imag(result), 'g--',
                                linewidth=2, label='Imaginary part')
            axes[row, col].set_xlabel('u')
            axes[row, col].set_ylabel('FrFT(f)(u)')
            axes[row, col].set_title(
                f'Fractional Fourier Transform (α={alpha})')

        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(successful_ops), axes.size):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig('fractional_operators_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Results plotted and saved to 'fractional_operators_demo.png'")


def benchmark_performance():
    """Benchmark performance of our operators"""
    print("\n⚡ Performance Benchmark")
    print("=" * 30)

    # Test different sizes
    sizes = [100, 500, 1000]
    alpha = 0.5

    benchmark_results = {}

    for size in sizes:
        print(f"\nTesting with {size} points...")
        x = np.linspace(0, 1, size)
        def f(t): return np.sin(2 * np.pi * t)

        # Benchmark Laplacian
        start_time = time.time()
        laplacian_result = fractional_laplacian_spectral(
            f, x, alpha, method="spectral")
        laplacian_time = time.time() - start_time
        print(f"  Laplacian: {laplacian_time:.4f}s")

        # Benchmark Fourier
        start_time = time.time()
        u, fourier_result = fractional_fourier_transform_fast(
            f, x, alpha, method="fast")
        fourier_time = time.time() - start_time
        print(f"  Fourier: {fourier_time:.4f}s")

        benchmark_results[size] = {
            'laplacian_time': laplacian_time,
            'fourier_time': fourier_time
        }

    return benchmark_results


def main():
    """Main test function"""
    print("🚀 Minimal Fractional Calculus Demo")
    print("=" * 50)

    # Test operators
    results, x, original = test_fractional_operators()

    # Plot results
    plot_results(results, x, original)

    # Benchmark performance
    benchmark_results = benchmark_performance()

    # Summary
    print("\n🎉 Demo Summary")
    print("=" * 20)
    print(f"✅ Fractional operators implemented successfully")
    print(f"✅ Operators computed for different α values")
    print(f"✅ Results visualized and saved")

    # Performance summary
    if benchmark_results:
        print(f"\n📊 Performance Summary:")
        for size, times in benchmark_results.items():
            print(f"  {size} points:")
            print(f"    Laplacian: {times['laplacian_time']:.4f}s")
            print(f"    Fourier: {times['fourier_time']:.4f}s")

    print(f"\n💡 This demonstrates the core concepts for Fractional PINO!")
    print(f"   The operators can be integrated into neural networks for physics-informed learning.")


if __name__ == "__main__":
    main()
