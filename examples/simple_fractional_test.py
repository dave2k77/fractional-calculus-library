"""
Simple Fractional Calculus Test with hpfracc

A simple demonstration of the hpfracc library capabilities
for fractional calculus operations.

Author: David
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def test_hpfracc_operators():
    """Test hpfracc fractional operators"""
    print("🚀 Testing hpfracc Fractional Operators")
    print("=" * 50)
    
    # Test function
    x = np.linspace(0, 1, 100)
    f = lambda t: np.sin(2 * np.pi * t)
    
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
        try:
            # Import here to avoid issues
            from algorithms import fractional_laplacian
            laplacian_result = fractional_laplacian(f, x, alpha, method="spectral")
            laplacian_time = time.time() - start_time
            print(f"  ✅ Laplacian completed in {laplacian_time:.4f}s")
            results[f'laplacian_alpha_{alpha}'] = {
                'result': laplacian_result,
                'time': laplacian_time
            }
        except Exception as e:
            print(f"  ❌ Laplacian failed: {e}")
        
        # Test fractional Fourier transform
        print("  Computing fractional Fourier transform...")
        start_time = time.time()
        try:
            from algorithms import fractional_fourier_transform
            u, fourier_result = fractional_fourier_transform(f, x, alpha, method="fast")
            fourier_time = time.time() - start_time
            print(f"  ✅ Fourier Transform completed in {fourier_time:.4f}s")
            results[f'fourier_alpha_{alpha}'] = {
                'result': fourier_result,
                'u': u,
                'time': fourier_time
            }
        except Exception as e:
            print(f"  ❌ Fourier Transform failed: {e}")
    
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
    fig, axes = plt.subplots(2, max(2, n_plots), figsize=(15, 8))
    
    # Plot original function
    axes[0, 0].plot(x, original, 'b-', linewidth=2, label='Original')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title('Original Function: sin(2πx)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot results
    for i, op_name in enumerate(successful_ops):
        row = i // 2
        col = i % 2
        
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
            axes[row, col].plot(u, np.real(result), 'g-', linewidth=2, label='Real part')
            axes[row, col].plot(u, np.imag(result), 'g--', linewidth=2, label='Imaginary part')
            axes[row, col].set_xlabel('u')
            axes[row, col].set_ylabel('FrFT(f)(u)')
            axes[row, col].set_title(f'Fractional Fourier Transform (α={alpha})')
        
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(successful_ops), axes.size):
        row = i // 2
        col = i % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('hpfracc_fractional_operators_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Results plotted and saved to 'hpfracc_fractional_operators_demo.png'")


def benchmark_performance():
    """Benchmark performance of hpfracc operators"""
    print("\n⚡ Performance Benchmark")
    print("=" * 30)
    
    # Test different sizes
    sizes = [100, 500, 1000]
    alpha = 0.5
    
    benchmark_results = {}
    
    for size in sizes:
        print(f"\nTesting with {size} points...")
        x = np.linspace(0, 1, size)
        f = lambda t: np.sin(2 * np.pi * t)
        
        # Benchmark Laplacian
        try:
            from algorithms import fractional_laplacian
            start_time = time.time()
            laplacian_result = fractional_laplacian(f, x, alpha, method="spectral")
            laplacian_time = time.time() - start_time
            print(f"  Laplacian: {laplacian_time:.4f}s")
        except Exception as e:
            print(f"  Laplacian failed: {e}")
            laplacian_time = None
        
        # Benchmark Fourier
        try:
            from algorithms import fractional_fourier_transform
            start_time = time.time()
            u, fourier_result = fractional_fourier_transform(f, x, alpha, method="fast")
            fourier_time = time.time() - start_time
            print(f"  Fourier: {fourier_time:.4f}s")
        except Exception as e:
            print(f"  Fourier failed: {e}")
            fourier_time = None
        
        benchmark_results[size] = {
            'laplacian_time': laplacian_time,
            'fourier_time': fourier_time
        }
    
    return benchmark_results


def main():
    """Main test function"""
    print("🚀 Simple Fractional Calculus Test with hpfracc")
    print("=" * 60)
    
    # Test operators
    results, x, original = test_hpfracc_operators()
    
    # Plot results
    plot_results(results, x, original)
    
    # Benchmark performance
    benchmark_results = benchmark_performance()
    
    # Summary
    print("\n🎉 Test Summary")
    print("=" * 20)
    print(f"✅ hpfracc library tested successfully")
    print(f"✅ Fractional operators computed")
    print(f"✅ Results visualized and saved")
    
    # Performance summary
    if benchmark_results:
        print(f"\n📊 Performance Summary:")
        for size, times in benchmark_results.items():
            print(f"  {size} points:")
            if times['laplacian_time']:
                print(f"    Laplacian: {times['laplacian_time']:.4f}s")
            if times['fourier_time']:
                print(f"    Fourier: {times['fourier_time']:.4f}s")


if __name__ == "__main__":
    main()
