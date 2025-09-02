#!/usr/bin/env python3
"""
Simple test script to debug backend issues
"""

import numpy as np
from hpfracc.ml import BackendType, switch_backend

# Import FractionalOrder for proper typing
from hpfracc.core.definitions import FractionalOrder


def test_backend_switching():
    """Test basic backend switching"""
    print("Testing backend switching...")

    # Test PyTorch
    try:
        switch_backend(BackendType.TORCH)
        print("✅ PyTorch backend activated")
    except Exception as e:
        print(f"❌ PyTorch backend failed: {e}")

    # Test JAX
    try:
        switch_backend(BackendType.JAX)
        print("✅ JAX backend activated")
    except Exception as e:
        print(f"❌ JAX backend failed: {e}")

    # Test NUMBA
    try:
        switch_backend(BackendType.NUMBA)
        print("✅ NUMBA backend activated")
    except Exception as e:
        print(f"❌ NUMBA backend failed: {e}")


def test_tensor_creation():
    """Test tensor creation in each backend"""
    print("\nTesting tensor creation...")

    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            switch_backend(backend)
            print(f"\n🔧 Testing {backend.value} backend:")

            # Test basic tensor creation
            from hpfracc.ml import get_tensor_ops
            tensor_ops = get_tensor_ops(backend)

            # Create simple tensors
            zeros = tensor_ops.zeros((2, 3))
            ones = tensor_ops.ones((2, 3))
            rand_tensor = tensor_ops.create_tensor(np.random.randn(2, 3))

            print(f"  ✅ Zeros shape: {zeros.shape}")
            print(f"  ✅ Ones shape: {ones.shape}")
            print(f"  ✅ Random tensor shape: {rand_tensor.shape}")

            # Test basic operations
            result = tensor_ops.matmul(rand_tensor, rand_tensor.T)
            print(f"  ✅ Matrix multiplication result shape: {result.shape}")

        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")


def test_fractional_network():
    """Test fractional neural network creation"""
    print("\nTesting fractional neural network...")

    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            switch_backend(backend)
            print(f"\n🔧 Testing {backend.value} backend:")

            from hpfracc.ml import FractionalNeuralNetwork

            # Create a simple network
            network = FractionalNeuralNetwork(
                input_size=4,
                hidden_sizes=[8, 6],
                output_size=2,
                fractional_order=FractionalOrder(0.5),
                backend=backend
            )

            print(f"  ✅ Network created successfully")
            print(f"  ✅ Weights: {len(network.weights)}")
            print(f"  ✅ Biases: {len(network.biases)}")

            # Test forward pass with simple data
            if backend == BackendType.TORCH:
                import torch
                x = torch.randn(1, 4, dtype=torch.float32)
            elif backend == BackendType.JAX:
                import jax.random as random
                x = random.normal(random.PRNGKey(0), (1, 4))
            else:  # NUMBA
                x = np.random.randn(1, 4).astype(np.float32)

            try:
                output = network(x)
                print(
                    f"  ✅ Forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"  ❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")
            if backend == BackendType.JAX:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    print("🚀 HPFRACC Simple Backend Test")
    print("=" * 40)

    test_backend_switching()
    test_tensor_creation()
    test_fractional_network()

    print("\n🎉 Simple backend test completed!")
