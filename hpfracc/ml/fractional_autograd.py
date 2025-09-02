"""
PyTorch Autograd Implementation of Fractional Derivatives

This module provides fractional derivative implementations that preserve
the PyTorch computation graph, enabling proper gradient flow during training.
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..core.definitions import FractionalOrder


class FractionalDerivativeFunction(torch.autograd.Function):
    """
    Custom autograd function for fractional derivatives implemented via
    Grünwald–Letnikov (GL) convolution with differentiable PyTorch ops.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, method: str = "RL") -> torch.Tensor:
        # Save context
        ctx.alpha = float(alpha)
        ctx.method = method

        # Compute method-specific kernel, then convolve along last dim
        y, kernel = _fractional_convolution_forward(x, ctx.alpha, method)
        ctx.save_for_backward(kernel)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (kernel,) = ctx.saved_tensors
        # Gradient wrt input is convolution with flipped kernel
        grad_input = _conv_last_dim(grad_output, kernel.flip(-1))
        return grad_input, None, None


def _gl_weights(alpha: float, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute GL binomial weights w_k = (-1)^k * C(alpha, k) for k=0..K-1."""
    w = torch.empty(K, device=device, dtype=dtype)
    w[0] = 1.0
    for k in range(1, K):
        w[k] = w[k - 1] * (alpha - (k - 1)) / k * (-1.0)
    return w


def _exp_kernel(alpha: float, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Caputo–Fabrizio-like exponential kernel (normalized)."""
    # Rate parameter: tie smoothly to alpha in (0,1]; avoid zero
    lam = max(1e-6, float(alpha))
    k = torch.arange(K, device=device, dtype=dtype)
    w = torch.exp(-lam * k)
    # Normalize for stability
    w = w / (w.abs().sum() + 1e-12)
    return w


def _ab_kernel(alpha: float, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Atangana–Baleanu-like kernel: blend GL weights with an exponential tail."""
    gl = _gl_weights(alpha, K, device, dtype)
    expw = _exp_kernel(alpha, K, device, dtype)
    # Blend keeps sign structure from GL while damping long tail
    w = 0.7 * gl + 0.3 * expw
    return w


def _conv_last_dim(x: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """Apply 1D convolution along the last dimension with given 1D kernel.
    Uses padding to keep the same output length as input.
    """
    K = int(kernel_1d.numel())
    pad = K - 1

    orig_shape = x.shape
    L = orig_shape[-1]
    # Collapse all leading dims into batch, use single channel
    x_ = x.reshape(-1, 1, L)
    weight = kernel_1d.view(1, 1, K)
    y_ = torch.nn.functional.conv1d(x_, weight, padding=pad)
    # Trim to original length (causal GL form)
    y_ = y_[:, :, :L]
    return y_.reshape(orig_shape)


def _fractional_convolution_forward(x: torch.Tensor, alpha: float, method: str = "RL") -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable fractional derivative via 1D convolution along last dim.
    method selects kernel: 'RL'/'GL' -> GL binomials, 'Caputo' -> GL, 'CF' -> exponential,
    'AB' -> blended GL/exponential.
    """
    if alpha == 0.0:
        kernel = torch.zeros(1, device=x.device, dtype=x.dtype)
        kernel[0] = 1.0
        return x, kernel

    if alpha == 1.0:
        # First difference kernel [1, -1]
        kernel = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)
        y = _conv_last_dim(x, kernel)
        return y, kernel

    L = x.shape[-1]
    K = int(min(max(8, L), 128))
    m = (method or "RL").upper()
    if m in ("RL", "GL", "CAPUTO"):
        kernel = _gl_weights(alpha, K, x.device, x.dtype)
    elif m in ("CF", "CAPUTO-FABRIZIO", "CAPUTO_FABRIZIO"):
        kernel = _exp_kernel(alpha, K, x.device, x.dtype)
    elif m in ("AB", "ATANGANA-BALEANU", "ATANGANA_BALEANU"):
        kernel = _ab_kernel(alpha, K, x.device, x.dtype)
    else:
        # Default to GL for unknown methods to remain robust
        kernel = _gl_weights(alpha, K, x.device, x.dtype)
    y = _conv_last_dim(x, kernel)
    return y, kernel


def _caputo_forward(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Use same GL convolution path for differentiability
    y, _ = _fractional_convolution_forward(x, alpha)
    return y


def _grunwald_letnikov_forward(x: torch.Tensor, alpha: float) -> torch.Tensor:
    y, _ = _fractional_convolution_forward(x, alpha)
    return y


def _riemann_liouville_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        alpha: float) -> torch.Tensor:
    # Use convolutional adjoint for consistency
    _, kernel = _fractional_convolution_forward(x, alpha)
    return _conv_last_dim(grad_output, kernel.flip(-1))


def _caputo_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        alpha: float) -> torch.Tensor:
    """Backward pass for Caputo fractional derivative"""
    # Similar to Riemann-Liouville for this simplified implementation
    return _riemann_liouville_backward(grad_output, x, alpha)


def _grunwald_letnikov_backward(
        grad_output: torch.Tensor,
        x: torch.Tensor,
        alpha: float) -> torch.Tensor:
    """Backward pass for Grünwald-Letnikov fractional derivative"""
    # Similar to Riemann-Liouville for this simplified implementation
    return _riemann_liouville_backward(grad_output, x, alpha)


def fractional_derivative(
        x: torch.Tensor,
        alpha: float,
        method: str = "RL") -> torch.Tensor:
    """
    Compute fractional derivative using PyTorch autograd

    Args:
        x: Input tensor
        alpha: Fractional order
        method: Derivative method ("RL", "Caputo", "GL")

    Returns:
        Fractional derivative tensor with preserved computation graph
    """
    return FractionalDerivativeFunction.apply(x, alpha, method)


class FractionalDerivativeLayer(nn.Module):
    """
    PyTorch module for fractional derivatives

    This layer can be integrated into neural networks and preserves
    the computation graph for proper gradient flow.
    """

    def __init__(self, alpha: float, method: str = "RL"):
        super().__init__()
        self.alpha = FractionalOrder(alpha)
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return fractional_derivative(x, self.alpha.alpha, self.method)

    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, method={self.method}'


# Convenience functions for common fractional derivatives
def rl_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Riemann-Liouville fractional derivative"""
    return fractional_derivative(x, alpha, "RL")


def caputo_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Caputo fractional derivative"""
    return fractional_derivative(x, alpha, "Caputo")


def gl_derivative(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Grünwald-Letnikov fractional derivative"""
    return fractional_derivative(x, alpha, "GL")
