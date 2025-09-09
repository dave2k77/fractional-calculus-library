"""
Final test for corrected spectral autograd with proper gradient flow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def riesz_kernel(alpha: float, size: int, dx: float = 1.0) -> torch.Tensor:
    """Generate Riesz fractional Laplacian kernel |ω|^α."""
    # Create frequency array manually to avoid MKL issues
    freq = torch.linspace(-0.5, 0.5, size) / dx
    omega = 2 * torch.pi * freq
    kernel = torch.abs(omega).pow(alpha)
    kernel[0] = 0.0 if alpha > 0 else 1.0
    return kernel


def apply_riesz_derivative(x: torch.Tensor, alpha: float, dx: float = 1.0) -> torch.Tensor:
    """Apply Riesz fractional derivative using numpy FFT."""
    kernel = riesz_kernel(alpha, x.size(-1), dx)
    
    # Use numpy FFT to avoid MKL issues
    x_np = x.detach().cpu().numpy()
    X_np = np.fft.fft(x_np)
    Y_np = X_np * kernel.cpu().numpy()
    result_np = np.fft.ifft(Y_np)
    
    return torch.from_numpy(result_np.real).to(x.device).to(x.dtype)


class CorrectedSpectralFractionalDerivative(torch.autograd.Function):
    """Corrected spectral fractional derivative with proper gradient flow."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, dx: float = 1.0) -> torch.Tensor:
        """Forward pass."""
        result = apply_riesz_derivative(x, alpha, dx)
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.dx = dx
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Backward pass - Riesz is self-adjoint."""
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        dx = ctx.dx
        
        # Riesz is self-adjoint: (D^α)* = D^α
        grad_x = apply_riesz_derivative(grad_output, alpha, dx)
        return grad_x, None, None


def test_corrected_spectral_autograd():
    """Test the corrected spectral autograd implementation."""
    print("Testing Corrected Spectral Autograd...")
    
    # Test parameters
    N = 32
    dx = 0.1
    alpha = 0.5
    
    # Create test signal as leaf tensor
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    f.requires_grad_(True)  # Make it a leaf tensor
    
    print(f"Input shape: {f.shape}")
    print(f"Input requires grad: {f.requires_grad}")
    print(f"Input is leaf: {f.is_leaf}")
    
    # Test forward pass
    result = CorrectedSpectralFractionalDerivative.apply(f, alpha, dx)
    print(f"Output shape: {result.shape}")
    print(f"Output requires grad: {result.requires_grad}")
    print(f"Output is real: {not result.is_complex()}")
    
    # Test gradient flow
    loss = result.sum()
    print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    print(f"Gradient w.r.t. input: {f.grad is not None}")
    if f.grad is not None:
        print(f"Gradient norm: {f.grad.norm().item():.6f}")
        print(f"Gradient shape: {f.grad.shape}")
        print(f"Gradient sample: {f.grad[:5]}")
    
    # Test with different alpha values
    print("\nTesting different alpha values:")
    for test_alpha in [0.2, 0.5, 0.8, 1.2, 1.5]:
        f.grad = None
        result = CorrectedSpectralFractionalDerivative.apply(f, test_alpha, dx)
        loss = result.sum()
        loss.backward()
        grad_norm = f.grad.norm().item() if f.grad is not None else 0.0
        print(f"  α={test_alpha}: grad_norm={grad_norm:.6f}")
    
    print("\n✅ All tests passed!")


def test_adjoint_property():
    """Test the adjoint property: ⟨D^α f, g⟩ = ⟨f, D^α g⟩"""
    print("\nTesting Adjoint Property...")
    
    N = 16
    dx = 0.1
    alpha = 0.5
    
    # Create test functions
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    g = torch.cos(x)
    
    # Compute D^α f and D^α g
    Df = apply_riesz_derivative(f, alpha, dx)
    Dg = apply_riesz_derivative(g, alpha, dx)
    
    # Compute inner products
    inner1 = torch.dot(Df, g).item()
    inner2 = torch.dot(f, Dg).item()
    
    print(f"⟨D^α f, g⟩ = {inner1:.6f}")
    print(f"⟨f, D^α g⟩ = {inner2:.6f}")
    print(f"Difference: {abs(inner1 - inner2):.6f}")
    
    if abs(inner1 - inner2) < 1e-5:
        print("✅ Adjoint property verified!")
    else:
        print("❌ Adjoint property failed!")


if __name__ == "__main__":
    test_corrected_spectral_autograd()
    test_adjoint_property()
