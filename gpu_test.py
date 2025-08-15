#!/usr/bin/env python3
"""
GPU Access Test Script
Tests GPU capabilities for the fractional calculus library
"""

import numpy as np
import sys


def test_basic_gpu_access():
    """Test basic GPU access through the library."""
    print("🔍 Testing GPU Access for NVIDIA RTX 3050")
    print("=" * 50)

    # Test 1: Check if we can import GPU modules
    print("\n1. Testing GPU module imports...")
    try:
        from src.optimisation.gpu_optimization import create_gpu_optimizer, GPUBackend

        print("✅ GPU optimization module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GPU module: {e}")
        return False

    # Test 2: Try to create GPU optimizer
    print("\n2. Testing GPU optimizer creation...")
    try:
        optimizer = create_gpu_optimizer(backend=GPUBackend.CUPY)  # Force CuPy backend
        print("✅ GPU optimizer created successfully")
        print(f"   Backend: {optimizer.backend}")
        print(f"   Device count: {optimizer.device_count}")
        print(f"   Default device: {optimizer.default_device}")
    except Exception as e:
        print(f"❌ Failed to create GPU optimizer: {e}")
        return False

    # Test 3: Test GPU computation
    print("\n3. Testing GPU computation...")
    try:
        # Create test data
        x = np.random.random(1000)
        alpha = 0.5

        # Run GPU computation
        result = optimizer.optimize_fractional_derivative(x, alpha, method="caputo")

        print("✅ GPU computation successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {result.shape}")
        print(f"   Result type: {type(result)}")

        # Check if result is reasonable
        if np.any(np.isnan(result)):
            print("⚠️  Warning: Result contains NaN values")
        else:
            print("✅ Result validation passed")

    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

    # Test 4: Performance comparison
    print("\n4. Testing performance comparison...")
    try:
        import time

        # CPU computation
        from src.algorithms.optimized_methods import optimized_caputo

        t = np.linspace(0, 1, 1000)

        start_time = time.time()
        cpu_result = optimized_caputo(x, t, alpha)
        cpu_time = time.time() - start_time

        # GPU computation
        start_time = time.time()
        gpu_result = optimizer.optimize_fractional_derivative(x, alpha, method="caputo")
        gpu_time = time.time() - start_time

        print(f"   CPU time: {cpu_time:.4f}s")
        print(f"   GPU time: {gpu_time:.4f}s")
        print(f"   Speedup: {cpu_time/gpu_time:.2f}x")

        # Check accuracy
        if np.allclose(cpu_result, gpu_result, rtol=1e-3):
            print("✅ CPU/GPU results match within tolerance")
        else:
            print("⚠️  Warning: CPU/GPU results differ significantly")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")

    return True


def test_gpu_libraries():
    """Test available GPU libraries."""
    print("\n📚 Testing Available GPU Libraries")
    print("=" * 40)

    libraries = {
        "JAX": "jax",
        "CuPy": "cupy",
        "PyTorch": "torch",
        "TensorFlow": "tensorflow",
    }

    for name, module in libraries.items():
        try:
            __import__(module)
            print(f"✅ {name} is available")

            # Test JAX specifically
            if name == "JAX":
                import jax

                print(f"   JAX version: {jax.__version__}")
                print(f"   JAX devices: {jax.devices()}")
                try:
                    gpu_devices = jax.devices("gpu")
                    if gpu_devices:
                        print(f"   GPU devices: {gpu_devices}")
                    else:
                        print("   No GPU devices found")
                except RuntimeError as e:
                    print(f"   GPU detection failed: {e}")
                    print("   Only CPU devices available")

        except ImportError:
            print(f"❌ {name} is not available")


def main():
    """Main test function."""
    print("🚀 GPU Access Test for Fractional Calculus Library")
    print("=" * 60)

    # Test available libraries
    test_gpu_libraries()

    # Test GPU access
    success = test_basic_gpu_access()

    print("\n" + "=" * 60)
    if success:
        print("🎉 GPU access test completed successfully!")
        print("Your NVIDIA RTX 3050 should be accessible for GPU acceleration.")
    else:
        print("❌ GPU access test failed.")
        print("You may need to install GPU libraries or drivers.")

    print("\n💡 Recommendations:")
    print("- Install JAX with GPU support: pip install jax[cuda]")
    print("- Install CuPy: pip install cupy-cuda12x")
    print("- Ensure NVIDIA drivers are up to date")


if __name__ == "__main__":
    main()
