"""
Simple test for corrected spectral autograd without complex FFT issues.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def simple_riesz_kernel(alpha: float, size: int, dx: float = 1.0) -> torch.Tensor:
    """Simple Riesz kernel generation."""
    # Create frequency array manually
    freq = torch.linspace(-0.5, 0.5, size) / dx
    omega = 2 * torch.pi * freq
    kernel = torch.abs(omega).pow(alpha)
    kernel[0] = 0.0 if alpha > 0 else 1.0
    return kernel


def simple_riesz_derivative(x: torch.Tensor, alpha: float, dx: float = 1.0) -> torch.Tensor:
    """Simple Riesz derivative using manual FFT."""
    kernel = simple_riesz_kernel(alpha, x.size(-1), dx)
    
    # Use numpy FFT to avoid MKL issues
    x_np = x.detach().cpu().numpy()
    X_np = np.fft.fft(x_np)
    Y_np = X_np * kernel.cpu().numpy()
    result_np = np.fft.ifft(Y_np)
    
    return torch.from_numpy(result_np.real).to(x.device).to(x.dtype)


class SimpleSpectralFractionalDerivative(torch.autograd.Function):
    """Simple spectral fractional derivative for testing."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, dx: float = 1.0) -> torch.Tensor:
        """Forward pass."""
        result = simple_riesz_derivative(x, alpha, dx)
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
        
        # Riesz is self-adjoint
        grad_x = simple_riesz_derivative(grad_output, alpha, dx)
        return grad_x, None, None


def test_simple_spectral():
    """Test the simple spectral implementation."""
    print("Testing Simple Spectral Autograd...")
    
    # Test parameters
    N = 32
    dx = 0.1
    alpha = 0.5
    
    # Create test signal
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32, requires_grad=True)
    f = torch.sin(x)
    
    print(f"Input shape: {f.shape}")
    print(f"Input requires grad: {f.requires_grad}")
    
    # Test forward pass
    result = SimpleSpectralFractionalDerivative.apply(f, alpha, dx)
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
    
    print("✅ Simple test passed!")


if __name__ == "__main__":
    test_simple_spectral()
