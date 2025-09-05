"""
Simple test for probabilistic fractional orders.
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

def simple_test():
    """Simple test of probabilistic fractional orders."""
    print("Simple probabilistic fractional orders test...")
    
    # Create test input
    x = torch.linspace(0, 2*np.pi, 50, requires_grad=True)
    
    # Create probabilistic layer
    layer = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=False)
    
    try:
        # Debug: check layer structure
        print(f"  Layer alpha_dist: {layer.alpha_dist}")
        print(f"  Layer alpha_dist.learnable: {layer.alpha_dist.learnable}")
        print(f"  Layer alpha_dist._parameters: {layer.alpha_dist._parameters}")
        
        # Test forward pass
        frac_deriv = layer(x)
        print(f"  ✓ Forward pass successful: {frac_deriv.item():.4f}")
        
        # Test gradient computation
        print(f"  frac_deriv.requires_grad: {frac_deriv.requires_grad}")
        print(f"  frac_deriv.grad_fn: {frac_deriv.grad_fn}")
        
        if frac_deriv.requires_grad:
            try:
                loss = frac_deriv.sum()
                loss.backward()
                print(f"  ✓ Backward pass successful")
            except Exception as e:
                print(f"  ⚠ Backward pass failed: {e}")
        else:
            print(f"  ⚠ No gradients to compute")
        
        # Test multiple samples
        samples = []
        for _ in range(10):
            sample = layer(x)
            samples.append(sample.item())
        
        mean_sample = sum(samples) / len(samples)
        std_sample = (sum((s - mean_sample)**2 for s in samples) / len(samples))**0.5
        
        print(f"  ✓ Multiple samples: mean={mean_sample:.4f}, std={std_sample:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎉 Probabilistic fractional orders working!")
    else:
        print("\n⚠️  Probabilistic fractional orders need fixes.")
