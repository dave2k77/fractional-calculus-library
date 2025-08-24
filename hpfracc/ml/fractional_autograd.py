"""
PyTorch Autograd Implementation of Fractional Derivatives

This module provides fractional derivative implementations that preserve
the PyTorch computation graph, enabling proper gradient flow during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from ..core.definitions import FractionalOrder


class FractionalDerivativeFunction(torch.autograd.Function):
    """
    Custom autograd function for fractional derivatives
    
    This implements the forward and backward passes for fractional derivatives
    using PyTorch's autograd system, ensuring gradients flow properly.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, method: str = "RL") -> torch.Tensor:
        """
        Forward pass for fractional derivative computation
        
        Args:
            x: Input tensor
            alpha: Fractional order
            method: Derivative method ("RL", "Caputo", "GL")
            
        Returns:
            Fractional derivative tensor
        """
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.method = method
        
        # For now, use a simplified implementation that preserves gradients
        # This is a placeholder - we'll implement proper fractional derivatives
        if method == "RL":
            # Riemann-Liouville approximation using finite differences
            return _riemann_liouville_forward(x, alpha)
        elif method == "Caputo":
            # Caputo approximation using finite differences
            return _caputo_forward(x, alpha)
        elif method == "GL":
            # Grünwald-Letnikov approximation using finite differences
            return _grunwald_letnikov_forward(x, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass for gradient computation
        
        Args:
            grad_output: Gradient of the output
            
        Returns:
            Gradient with respect to input, None for alpha, None for method
        """
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        method = ctx.method
        
        # Compute gradient with respect to input
        if method == "RL":
            grad_input = _riemann_liouville_backward(grad_output, x, alpha)
        elif method == "Caputo":
            grad_input = _caputo_backward(grad_output, x, alpha)
        elif method == "GL":
            grad_input = _grunwald_letnikov_backward(grad_output, x, alpha)
        else:
            grad_input = grad_output
        
        return grad_input, None, None


def _riemann_liouville_forward(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Forward pass for Riemann-Liouville fractional derivative
    
    This is a simplified implementation using finite differences
    """
    if alpha == 0:
        return x
    
    if alpha == 1:
        return torch.gradient(x, dim=-1)[0]
    
    # For non-integer alpha, use a simplified approximation
    # This is a placeholder - in practice, you'd want a more sophisticated method
    result = x.clone()
    
    # Apply a simplified fractional derivative approximation
    # This preserves the computation graph while providing a reasonable approximation
    if alpha > 0 and alpha < 1:
        # Use a weighted combination of the original signal and its gradient
        gradient = torch.gradient(x, dim=-1)[0]
        result = (1 - alpha) * x + alpha * gradient
    elif alpha > 1:
        # For alpha > 1, apply multiple derivatives
        n = int(alpha)
        fractional_part = alpha - n
        result = x
        for _ in range(n):
            result = torch.gradient(result, dim=-1)[0]
        if fractional_part > 0:
            gradient = torch.gradient(result, dim=-1)[0]
            result = (1 - fractional_part) * result + fractional_part * gradient
    
    return result


def _caputo_forward(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Forward pass for Caputo fractional derivative
    
    This is a simplified implementation using finite differences
    """
    if alpha == 0:
        return x
    
    if alpha == 1:
        return torch.gradient(x, dim=-1)[0]
    
    # For non-integer alpha, use a simplified approximation
    result = x.clone()
    
    if alpha > 0 and alpha < 1:
        # Use a weighted combination of the original signal and its gradient
        gradient = torch.gradient(x, dim=-1)[0]
        result = (1 - alpha) * x + alpha * gradient
    elif alpha > 1:
        # For alpha > 1, apply multiple derivatives
        n = int(alpha)
        fractional_part = alpha - n
        result = x
        for _ in range(n):
            result = torch.gradient(result, dim=-1)[0]
        if fractional_part > 0:
            gradient = torch.gradient(result, dim=-1)[0]
            result = (1 - fractional_part) * result + fractional_part * gradient
    
    return result


def _grunwald_letnikov_forward(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Forward pass for Grünwald-Letnikov fractional derivative
    
    This is a simplified implementation using finite differences
    """
    if alpha == 0:
        return x
    
    if alpha == 1:
        return torch.gradient(x, dim=-1)[0]
    
    # For non-integer alpha, use a simplified approximation
    result = x.clone()
    
    if alpha > 0 and alpha < 1:
        # Use a weighted combination of the original signal and its gradient
        gradient = torch.gradient(x, dim=-1)[0]
        result = (1 - alpha) * x + alpha * gradient
    elif alpha > 1:
        # For alpha > 1, apply multiple derivatives
        n = int(alpha)
        fractional_part = alpha - n
        result = x
        for _ in range(n):
            result = torch.gradient(result, dim=-1)[0]
        if fractional_part > 0:
            gradient = torch.gradient(result, dim=-1)[0]
            result = (1 - fractional_part) * result + fractional_part * gradient
    
    return result


def _riemann_liouville_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Backward pass for Riemann-Liouville fractional derivative"""
    if alpha == 0:
        return grad_output
    
    if alpha == 1:
        # For first derivative, the adjoint is -gradient
        return -torch.gradient(grad_output, dim=-1)[0]
    
    # Simplified backward pass
    if alpha > 0 and alpha < 1:
        gradient_grad = torch.gradient(grad_output, dim=-1)[0]
        return (1 - alpha) * grad_output - alpha * gradient_grad
    elif alpha > 1:
        n = int(alpha)
        fractional_part = alpha - n
        result = grad_output
        for _ in range(n):
            result = -torch.gradient(result, dim=-1)[0]
        if fractional_part > 0:
            gradient_grad = torch.gradient(result, dim=-1)[0]
            result = (1 - fractional_part) * result - fractional_part * gradient_grad
        return result
    
    return grad_output


def _caputo_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Backward pass for Caputo fractional derivative"""
    # Similar to Riemann-Liouville for this simplified implementation
    return _riemann_liouville_backward(grad_output, x, alpha)


def _grunwald_letnikov_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Backward pass for Grünwald-Letnikov fractional derivative"""
    # Similar to Riemann-Liouville for this simplified implementation
    return _riemann_liouville_backward(grad_output, x, alpha)


def fractional_derivative(x: torch.Tensor, alpha: float, method: str = "RL") -> torch.Tensor:
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
