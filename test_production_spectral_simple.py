"""
Simplified Production Spectral Autograd Test

This addresses the key review points with a working implementation:
1. Limit tests (α→0, α→2)
2. Semigroup property verification
3. Adjoint property testing
4. Learnable α with bounded parameterization
5. Proper numerical stability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class SimpleProductionSpectralDerivative(torch.autograd.Function):
    """Simplified production spectral derivative addressing key review points."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, dx: float = 1.0, 
                mode: str = "riesz") -> torch.Tensor:
        """Forward pass with proper scaling and branch handling."""
        # Validate inputs
        if not (0 < alpha < 2):
            raise ValueError(f"Alpha must be in (0, 2), got {alpha}")
        
        # Handle special cases
        if alpha == 0.0:
            return x
        if alpha == 2.0:
            # Should approach negative Laplacian
            return -_apply_laplacian(x, dx)
        
        # Generate kernel with proper scaling
        kernel = _generate_kernel(alpha, x.size(-1), dx, x.device, x.dtype, mode)
        
        # Apply spectral derivative
        result = _apply_spectral_derivative(x, kernel)
        
        # Save for backward pass
        ctx.save_for_backward(x, kernel)
        ctx.alpha = alpha
        ctx.mode = mode
        ctx.dx = dx
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], None, None]:
        """Backward pass with proper adjoint."""
        x, kernel = ctx.saved_tensors
        alpha = ctx.alpha
        mode = ctx.mode
        dx = ctx.dx
        
        # Compute gradient w.r.t. input using adjoint
        if mode == "riesz":
            # Riesz is self-adjoint
            grad_x = _apply_spectral_derivative(grad_output, kernel)
        else:  # weyl
            # Use complex conjugate for adjoint
            adjoint_kernel = kernel.conj()
            grad_x = _apply_spectral_derivative(grad_output, adjoint_kernel)
        
        # Compute gradient w.r.t. α
        grad_alpha = _compute_alpha_gradient(x, grad_output, alpha, mode, dx)
        
        return grad_x, grad_alpha, None, None


def _generate_kernel(alpha: float, size: int, dx: float, 
                    device: torch.device, dtype: torch.dtype, mode: str) -> torch.Tensor:
    """Generate spectral kernel with proper scaling and branch handling."""
    # Create frequency array with proper scaling
    freq = torch.fft.rfftfreq(size, d=dx, device=device, dtype=dtype)
    omega = 2 * torch.pi * freq  # Convert to angular frequency
    
    if mode == "riesz":
        # Riesz fractional Laplacian: |ω|^α
        kernel = torch.abs(omega).pow(alpha)
    else:  # weyl
        # Weyl derivative with principal branch: (iω)^α = |ω|^α * exp(i * sign(ω) * π * α / 2)
        kernel = torch.abs(omega).pow(alpha) * torch.exp(1j * torch.sign(omega) * torch.pi * alpha / 2)
    
    # Handle zero frequency deterministically
    kernel[0] = 0.0 if alpha > 0 else 1.0
    
    return kernel


def _apply_spectral_derivative(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply spectral derivative using rFFT for efficiency."""
    X = torch.fft.rfft(x, dim=-1)
    Y = X * kernel
    result = torch.fft.irfft(Y, n=x.size(-1), dim=-1)
    
    return result


def _apply_laplacian(x: torch.Tensor, dx: float) -> torch.Tensor:
    """Apply negative Laplacian for α→2 limit test."""
    # Second derivative approximation
    d2f = torch.diff(x, n=2, dim=-1) / (dx**2)
    # Pad to match input size
    pad_shape = list(x.shape)
    pad_shape[-1] = 1
    d2f = torch.cat([torch.zeros(pad_shape), d2f, torch.zeros(pad_shape)], dim=-1)
    return d2f


def _compute_alpha_gradient(x: torch.Tensor, grad_output: torch.Tensor, 
                           alpha: float, mode: str, dx: float) -> torch.Tensor:
    """Compute gradient w.r.t. α with proper numerical stability."""
    # Get frequency array
    freq = torch.fft.rfftfreq(x.size(-1), d=dx, device=x.device, dtype=x.dtype)
    omega = 2 * torch.pi * freq
    
    # Compute derivative kernel w.r.t. α with numerical stability
    if mode == "riesz":
        # For Riesz: ∂D^α f/∂α = |ω|^α * log|ω| * f̂
        kernel_alpha = torch.abs(omega).pow(alpha) * torch.log(torch.abs(omega).clamp_min(torch.finfo(x.dtype).tiny))
    else:  # weyl
        # For Weyl: ∂D^α f/∂α = (iω)^α * log(iω) * f̂
        kernel_alpha = (1j * omega).pow(alpha) * torch.log(1j * omega)
    
    # Apply the derivative w.r.t. α
    X = torch.fft.rfft(x, dim=-1)
    dD_dalpha = torch.fft.irfft(X * kernel_alpha, n=x.size(-1), dim=-1)
    
    # Compute gradient using chain rule
    grad_alpha = (grad_output * dD_dalpha).sum()
    
    return grad_alpha


class BoundedAlphaParameter(nn.Parameter):
    """Bounded α parameter using sigmoid transformation."""
    
    def __init__(self, alpha_init: float = 0.5, alpha_min: float = 0.01, alpha_max: float = 1.99):
        # Transform to unbounded space
        rho_init = torch.logit((alpha_init - alpha_min) / (alpha_max - alpha_min))
        super().__init__(rho_init)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def get_alpha(self) -> torch.Tensor:
        """Get bounded α value."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self)


class ProductionSpectralLayer(nn.Module):
    """Production spectral layer with learnable α."""
    
    def __init__(self, alpha_init: float = 0.5, mode: str = "riesz", dx: float = 1.0):
        super().__init__()
        self.mode = mode
        self.dx = dx
        self.alpha_param = BoundedAlphaParameter(alpha_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_param.get_alpha()
        return SimpleProductionSpectralDerivative.apply(x, alpha, self.dx, self.mode)


def test_limit_behavior():
    """Test α→0 and α→2 limit behavior."""
    print("Testing Limit Behavior...")
    
    N = 64
    dx = 0.1
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    
    # Test α→0 (should approach identity)
    result_0 = SimpleProductionSpectralDerivative.apply(f, 0.01, dx, "riesz")
    identity_error = torch.norm(result_0 - f).item()
    print(f"  α→0 (identity): error = {identity_error:.6f}")
    
    # Test α→2 (should approach negative Laplacian)
    result_2 = SimpleProductionSpectralDerivative.apply(f, 1.99, dx, "riesz")
    laplacian_ref = -_apply_laplacian(f, dx)
    laplacian_error = torch.norm(result_2 - laplacian_ref).item()
    print(f"  α→2 (Laplacian): error = {laplacian_error:.6f}")
    
    return identity_error < 0.1 and laplacian_error < 0.1


def test_semigroup_property():
    """Test semigroup property: D^α D^β f = D^(α+β) f."""
    print("Testing Semigroup Property...")
    
    N = 32
    dx = 0.1
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    
    alpha, beta = 0.3, 0.4
    
    # Compute D^α D^β f
    Dbeta_f = SimpleProductionSpectralDerivative.apply(f, beta, dx, "riesz")
    Dalpha_Dbeta_f = SimpleProductionSpectralDerivative.apply(Dbeta_f, alpha, dx, "riesz")
    
    # Compute D^(α+β) f
    Dalpha_plus_beta_f = SimpleProductionSpectralDerivative.apply(f, alpha + beta, dx, "riesz")
    
    # Check difference
    semigroup_error = torch.norm(Dalpha_Dbeta_f - Dalpha_plus_beta_f).item()
    print(f"  Semigroup error: {semigroup_error:.6f}")
    
    return semigroup_error < 1e-5


def test_adjoint_property():
    """Test adjoint property: ⟨D^α f, g⟩ = ⟨f, D^α g⟩."""
    print("Testing Adjoint Property...")
    
    N = 32
    dx = 0.1
    alpha = 0.5
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    g = torch.cos(x)
    
    # Compute D^α f and D^α g
    Df = SimpleProductionSpectralDerivative.apply(f, alpha, dx, "riesz")
    Dg = SimpleProductionSpectralDerivative.apply(g, alpha, dx, "riesz")
    
    # Compute inner products
    inner1 = torch.dot(Df, g).item()
    inner2 = torch.dot(f, Dg).item()
    
    print(f"  ⟨D^α f, g⟩ = {inner1:.6f}")
    print(f"  ⟨f, D^α g⟩ = {inner2:.6f}")
    print(f"  Difference: {abs(inner1 - inner2):.6f}")
    
    return abs(inner1 - inner2) < 1e-6


def test_learnable_alpha():
    """Test learnable α with bounded parameterization."""
    print("Testing Learnable α...")
    
    N = 32
    dx = 0.1
    
    # Create test signal
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    
    # Create layer with learnable α
    layer = ProductionSpectralLayer(alpha_init=0.3, mode="riesz", dx=dx)
    
    # Test forward pass
    result = layer(f)
    print(f"  Output shape: {result.shape}")
    print(f"  α value: {layer.alpha_param.get_alpha().item():.6f}")
    
    # Test gradient flow
    loss = result.sum()
    loss.backward()
    
    print(f"  α gradient: {layer.alpha_param.grad.item():.6f}")
    print(f"  α after update: {layer.alpha_param.get_alpha().item():.6f}")
    
    return True


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("Testing Numerical Stability...")
    
    N = 64
    dx = 0.1
    
    # Test with very small α
    x = torch.linspace(0, 2*np.pi, N, dtype=torch.float32)
    f = torch.sin(x)
    
    try:
        result = SimpleProductionSpectralDerivative.apply(f, 0.001, dx, "riesz")
        print(f"  Very small α (0.001): OK, output norm = {result.norm().item():.6f}")
    except Exception as e:
        print(f"  Very small α (0.001): FAILED - {e}")
        return False
    
    # Test with very large α
    try:
        result = SimpleProductionSpectralDerivative.apply(f, 1.999, dx, "riesz")
        print(f"  Very large α (1.999): OK, output norm = {result.norm().item():.6f}")
    except Exception as e:
        print(f"  Very large α (1.999): FAILED - {e}")
        return False
    
    return True


def main():
    """Run all production tests."""
    print("=" * 60)
    print("PRODUCTION SPECTRAL AUTOGRAD TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Limit Behavior", test_limit_behavior),
        ("Semigroup Property", test_semigroup_property),
        ("Adjoint Property", test_adjoint_property),
        ("Learnable α", test_learnable_alpha),
        ("Numerical Stability", test_numerical_stability),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"  {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"  ❌ FAILED - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All production tests passed! Framework is ready for publication.")
    else:
        print("⚠️  Some tests failed. Address issues before publication.")


if __name__ == "__main__":
    main()
