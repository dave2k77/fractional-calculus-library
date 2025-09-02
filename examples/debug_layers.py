#!/usr/bin/env python3
"""
Debug script to test individual layers and identify tensor shape issues
"""
import numpy as np
from hpfracc.ml import BackendType, switch_backend
from hpfracc.ml import FractionalConv1D, FractionalConv2D, FractionalLSTM, FractionalTransformer


def debug_layers():
    """Debug individual layers to identify shape issues"""
    print("🔍 Debugging Individual Layers")
    print("=" * 40)

    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            print(f"\n🔧 Testing {backend.value.upper()} backend:")
            switch_backend(backend)

            # Test Conv1D
            print("  Testing FractionalConv1D...")
            conv1d = FractionalConv1D(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                backend=backend
            )

            # Create 1D input (batch_size, channels, seq_len)
            X_1d = np.random.randn(32, 16, 64)
            print(f"    Input shape: {X_1d.shape}")
            print(f"    Expected in_channels: 16, out_channels: 32")

            try:
                output = conv1d.forward(X_1d)
                print(
                    f"    ✅ Conv1D forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"    ❌ Conv1D failed: {e}")

            # Test Conv2D
            print("  Testing FractionalConv2D...")
            conv2d = FractionalConv2D(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                backend=backend
            )

            # Create 2D input (batch_size, channels, height, width)
            X_2d = np.random.randn(16, 16, 32, 32)
            print(f"    Input shape: {X_2d.shape}")
            print(f"    Expected in_channels: 16, out_channels: 32")

            try:
                output = conv2d.forward(X_2d)
                print(
                    f"    ✅ Conv2D forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"    ❌ Conv2D failed: {e}")

            # Test LSTM
            print("  Testing FractionalLSTM...")
            lstm = FractionalLSTM(
                input_size=16,
                hidden_size=32,
                backend=backend
            )

            # Create sequence input (batch_size, seq_len, input_size)
            X_seq = np.random.randn(16, 20, 16)
            print(f"    Input shape: {X_seq.shape}")
            print(f"    Expected input_size: 16, hidden_size: 32")

            try:
                output, (h, c) = lstm.forward(X_seq)
                print(
                    f"    ✅ LSTM forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"    ❌ LSTM failed: {e}")

            # Test Transformer
            print("  Testing FractionalTransformer...")
            transformer = FractionalTransformer(
                d_model=16,
                n_heads=4,
                backend=backend
            )

            # Create transformer input (batch_size, seq_len, d_model)
            X_trans = np.random.randn(16, 20, 16)
            print(f"    Input shape: {X_trans.shape}")
            print(f"    Expected d_model: 16, n_heads: 4")

            try:
                output = transformer.forward(X_trans)
                print(
                    f"    ✅ Transformer forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"    ❌ Transformer failed: {e}")

            print(
                f"  ✅ {backend.value.upper()} backend: Layer testing completed!")

        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    debug_layers()
    print("\n🎉 Layer debugging completed!")
