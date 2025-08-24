#!/usr/bin/env python3
"""
Test script specifically for the attention mechanism
"""

import numpy as np
from hpfracc.ml import BackendType, get_backend_manager, switch_backend, FractionalAttention

# Import FractionalOrder for proper typing
from hpfracc.core.definitions import FractionalOrder

def test_attention():
    """Test the attention mechanism across all backends"""
    print("🚀 Testing FractionalAttention Mechanism")
    print("=" * 50)
    
    for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
        try:
            print(f"\n🔧 Testing {backend.value.upper()} backend:")
            switch_backend(backend)
            
            # Create attention mechanism
            attention = FractionalAttention(
                d_model=16,  # 16 is divisible by 4 heads
                n_heads=4,
                fractional_order=FractionalOrder(0.5),
                backend=backend
            )
            
            print(f"  ✅ Attention created with d_model={attention.d_model}, d_k={attention.d_k}")
            
            # Create test input (batch_size=2, seq_len=3, d_model=16)
            if backend == BackendType.TORCH:
                import torch
                x = torch.randn(2, 3, attention.d_model, dtype=torch.float32)
            elif backend == BackendType.JAX:
                import jax.random as random
                x = random.normal(random.PRNGKey(0), (2, 3, attention.d_model))
            else:  # NUMBA
                x = np.random.randn(2, 3, attention.d_model).astype(np.float32)
            
            print(f"  ✅ Input tensor created with shape: {x.shape}")
            
            # Test forward pass
            try:
                output = attention(x, method="RL")
                print(f"  ✅ Forward pass successful, output shape: {output.shape}")
                
                # Test with different sequence lengths
                if backend == BackendType.TORCH:
                    x2 = torch.randn(1, 1, attention.d_model, dtype=torch.float32)
                elif backend == BackendType.JAX:
                    x2 = random.normal(random.PRNGKey(1), (1, 1, attention.d_model))
                else:  # NUMBA
                    x2 = np.random.randn(1, 1, attention.d_model).astype(np.float32)
                
                output2 = attention(x2, method="RL")
                print(f"  ✅ Single sequence test successful, output shape: {output2.shape}")
                
            except Exception as e:
                print(f"  ❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"  ❌ {backend.value} failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_attention()
    print("\n🎉 Attention test completed!")
