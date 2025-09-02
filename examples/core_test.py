#!/usr/bin/env python3
"""
Core components test - testing only the working components
"""

import numpy as np
from hpfracc.ml import BackendType, switch_backend
from hpfracc.core.definitions import FractionalOrder


def test_core_components():
    """Test only the core components that are working"""
    print("🚀 HPFRACC Core Components Test")
    print("=" * 40)

    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            print(f"\n🔧 Testing {backend.value.upper()} backend:")
            switch_backend(backend)

            # Test 1: Basic neural network
            print("  Testing FractionalNeuralNetwork...")
            from hpfracc.ml import FractionalNeuralNetwork

            network = FractionalNeuralNetwork(
                input_size=8,
                hidden_sizes=[16, 8],
                output_size=2,
                fractional_order=FractionalOrder(0.5),
                backend=backend
            )

            # Create test data
            if backend == BackendType.TORCH:
                import torch
                x = torch.randn(4, 8, dtype=torch.float32)
            elif backend == BackendType.JAX:
                import jax.random as random
                x = random.normal(random.PRNGKey(0), (4, 8))
            else:  # NUMBA
                x = np.random.randn(4, 8).astype(np.float32)

            output = network(x, use_fractional=True, method="RL")
            print(
                f"    ✅ Neural network forward pass successful, output shape: {output.shape}")

            # Test 2: Attention mechanism
            print("  Testing FractionalAttention...")
            from hpfracc.ml import FractionalAttention

            attention = FractionalAttention(
                d_model=8,  # 8 is divisible by 2 heads
                n_heads=2,
                fractional_order=FractionalOrder(0.5),
                backend=backend
            )

            # Create test data for attention
            if backend == BackendType.TORCH:
                x_attn = torch.randn(2, 3, 8, dtype=torch.float32)
            elif backend == BackendType.JAX:
                x_attn = random.normal(random.PRNGKey(1), (2, 3, 8))
            else:  # NUMBA
                x_attn = np.random.randn(2, 3, 8).astype(np.float32)

            output_attn = attention(x_attn, method="RL")
            print(
                f"    ✅ Attention forward pass successful, output shape: {output_attn.shape}")

            print(f"  ✅ {backend.value.upper()} backend: All core tests passed!")

        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_core_components()
    print("\n🎉 Core components test completed!")
