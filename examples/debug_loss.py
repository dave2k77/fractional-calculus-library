#!/usr/bin/env python3
"""
Debug script to test loss functions and identify tensor shape issues
"""
import numpy as np
from hpfracc.ml import BackendType, switch_backend, FractionalMSELoss
from hpfracc.core.definitions import FractionalOrder


def debug_loss_functions():
    """Debug loss functions to identify shape issues"""
    print("🔍 Debugging Loss Functions")
    print("=" * 40)

    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            print(f"\n🔧 Testing {backend.value.upper()} backend:")
            switch_backend(backend)

            # Create simple test data
            predictions = np.random.randn(10, 16)  # 10 samples, 16 features
            targets = np.random.randn(10, 16)      # Same shape

            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Predictions type: {type(predictions)}")
            print(f"  Targets type: {type(targets)}")

            # Test loss function
            loss_fn = FractionalMSELoss(
                fractional_order=FractionalOrder(0.5), backend=backend)
            print(f"  Loss function created successfully")
            print(f"  Loss function backend: {loss_fn.backend}")

            # Test forward pass
            # Disable fractional for now
            loss = loss_fn(predictions, targets, use_fractional=False)
            print(f"  Loss computed successfully: {loss}")
            print(f"  Loss type: {type(loss)}")

            print(f"  ✅ {backend.value.upper()} backend: Loss function working!")

        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    debug_loss_functions()
    print("\n🎉 Loss function debugging completed!")
