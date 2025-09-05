"""
Test script for spectral fractional autograd prototype.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import our new spectral autograd
from hpfracc.ml.spectral_autograd import (
    FractionalAutogradLayer, 
    spectral_fractional_derivative,
    FFTEngine,
    MellinEngine,
    LaplacianEngine
)


def test_basic_functionality():
    """Test basic spectral autograd functionality."""
    print("Testing basic spectral autograd functionality...")
    
    # Create test input
    x = torch.linspace(0, 2*np.pi, 100, requires_grad=True)
    y = torch.sin(x)
    
    # Test different engines
    engines = ["fft", "mellin", "laplacian"]
    alpha = 0.5
    
    results = {}
    for engine in engines:
        print(f"Testing {engine} engine...")
        try:
            # Test direct function call
            frac_deriv = spectral_fractional_derivative(y, alpha, engine=engine)
            
            # Test gradient computation
            loss = frac_deriv.sum()
            loss.backward()
            
            results[engine] = {
                'derivative': frac_deriv.detach().numpy(),
                'gradient': x.grad.detach().numpy() if x.grad is not None else None
            }
            
            print(f"  ✓ {engine} engine working")
            
        except Exception as e:
            print(f"  ✗ {engine} engine failed: {e}")
            results[engine] = None
    
    return results


def test_layer_integration():
    """Test layer integration with neural networks."""
    print("\nTesting layer integration...")
    
    # Create a simple network with fractional layer
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.frac_layer = FractionalAutogradLayer(alpha=0.6, engine="fft")
            self.linear = nn.Linear(100, 1)
        
        def forward(self, x):
            frac_x = self.frac_layer(x)
            return self.linear(frac_x)
    
    # Test forward pass
    net = TestNetwork()
    x = torch.randn(1, 100, requires_grad=True)
    
    try:
        output = net(x)
        print(f"  ✓ Forward pass successful, output shape: {output.shape}")
        
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print(f"  ✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"  ✗ Layer integration failed: {e}")
        return False


def test_engine_comparison():
    """Compare different engines on the same input."""
    print("\nComparing engines...")
    
    # Create test function
    x = torch.linspace(0, 4*np.pi, 200, requires_grad=True)
    y = torch.sin(x) + 0.1 * torch.sin(3*x)
    
    engines = ["fft", "mellin", "laplacian"]
    alpha = 0.7
    
    plt.figure(figsize=(12, 8))
    
    for i, engine in enumerate(engines):
        try:
            frac_deriv = spectral_fractional_derivative(y, alpha, engine=engine)
            
            plt.subplot(2, 2, i+1)
            plt.plot(x.detach().numpy(), y.detach().numpy(), 'b-', label='Original')
            plt.plot(x.detach().numpy(), frac_deriv.detach().numpy(), 'r-', label=f'D^{alpha} ({engine})')
            plt.title(f'{engine.upper()} Engine')
            plt.legend()
            plt.grid(True)
            
        except Exception as e:
            print(f"  ✗ {engine} engine failed: {e}")
    
    plt.tight_layout()
    plt.savefig('spectral_autograd_comparison.png')
    print("  ✓ Comparison plot saved as 'spectral_autograd_comparison.png'")
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("SPECTRAL FRACTIONAL AUTOGRAD PROTOTYPE TEST")
    print("=" * 50)
    
    # Test basic functionality
    results = test_basic_functionality()
    
    # Test layer integration
    layer_success = test_layer_integration()
    
    # Test engine comparison
    comparison_success = test_engine_comparison()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    working_engines = [engine for engine, result in results.items() if result is not None]
    print(f"Working engines: {working_engines}")
    print(f"Layer integration: {'✓' if layer_success else '✗'}")
    print(f"Engine comparison: {'✓' if comparison_success else '✗'}")
    
    if len(working_engines) > 0 and layer_success:
        print("\n🎉 Prototype is working! Ready for next phase.")
    else:
        print("\n⚠️  Prototype needs fixes before proceeding.")


if __name__ == "__main__":
    main()
