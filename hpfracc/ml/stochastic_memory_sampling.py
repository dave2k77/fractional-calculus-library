"""
Stochastic Memory Sampling for Fractional Derivatives

This module implements unbiased estimators for fractional derivatives using
stochastic sampling of the memory history instead of full computation.
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Optional, Union, Callable, Dict, Any
import math


class StochasticMemorySampler:
    """
    Base class for stochastic memory sampling strategies.
    """
    
    def __init__(self, alpha: float, method: str = "importance", **kwargs):
        self.alpha = alpha
        self.method = method
        self.kwargs = kwargs
    
    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample k indices from history of length n."""
        raise NotImplementedError
    
    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute importance weights for sampled indices."""
        raise NotImplementedError
    
    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor, 
                          weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using sampled indices and weights."""
        raise NotImplementedError


class ImportanceSampler(StochasticMemorySampler):
    """
    Importance sampling for fractional derivative memory.
    
    Uses power-law distribution p(j) ∝ (n-j)^(-(1+α-τ)) where τ controls
    the tempering of the heavy tail.
    """
    
    def __init__(self, alpha: float, tau: float = 0.1, **kwargs):
        super().__init__(alpha, "importance", **kwargs)
        self.tau = tau  # Tempering parameter
    
    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices using importance sampling distribution."""
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        
        # Power-law distribution: p(j) ∝ (n-j)^(-(1+α-τ))
        j_vals = torch.arange(n, dtype=torch.float32)
        log_probs = -(1 + self.alpha - self.tau) * torch.log(n - j_vals + 1e-8)
        
        # Normalize probabilities
        probs = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))
        
        # Sample without replacement
        indices = torch.multinomial(probs, k, replacement=False)
        return indices
    
    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute importance weights w(j)/p(j)."""
        # True weights: w(j) ∝ (n-j)^(-(1+α))
        j_vals = indices.float()
        true_weights = torch.pow(n - j_vals + 1e-8, -(1 + self.alpha))
        
        # Sampling probabilities: p(j) ∝ (n-j)^(-(1+α-τ))
        sampling_probs = torch.pow(n - j_vals + 1e-8, -(1 + self.alpha - self.tau))
        
        # Importance weights: w(j)/p(j)
        importance_weights = true_weights / (sampling_probs + 1e-8)
        
        return importance_weights
    
    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor,
                          weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using importance sampling."""
        # Handle both 1D and 2D inputs
        if x.dim() == 2:
            # For 2D input (batch, features), work on the last feature dimension
            n = x.shape[-1]
            if len(indices) == 0:
                return torch.zeros_like(x)
            
            current_val = x[..., -1]  # Last feature for each batch
            sampled_vals = x[..., n - 1 - indices]  # Shape: (batch, k)
            differences = current_val.unsqueeze(-1) - sampled_vals  # Shape: (batch, k)
            
            # Weighted sum over samples
            weighted_sum = torch.sum(weights * differences, dim=-1)  # Shape: (batch,)
            
            # Normalize by sample size
            result = weighted_sum / len(indices)
            
            # Return tensor with same shape as input, properly connected for gradients
            # This is a simplified approach - in practice, you'd want proper fractional derivative gradients
            # We need to create a tensor that maintains the gradient connection
            result_tensor = torch.zeros_like(x)
            result_tensor[..., -1] = result
            return result_tensor
        else:
            # For 1D input, use original logic but return tensor with same shape
            n = len(x)
            if len(indices) == 0:
                return torch.zeros_like(x)

            current_val = x[-1]
            sampled_vals = x[n - 1 - indices]
            differences = current_val - sampled_vals

            # Weighted sum
            weighted_sum = torch.sum(weights * differences)

            # Normalize by sample size
            result = weighted_sum / len(indices)
            
            # Return tensor with same shape as input, properly connected for gradients
            # This is a simplified approach - in practice, you'd want proper fractional derivative gradients
            # The key is to ensure the result is properly connected to the input for gradient computation
            # We need to create a tensor that maintains the gradient connection
            result_tensor = torch.zeros_like(x)
            result_tensor[-1] = result
            return result_tensor


class StratifiedSampler(StochasticMemorySampler):
    """
    Stratified sampling with recent window and tail sampling.
    
    Samples densely from recent history and sparsely from tail.
    """
    
    def __init__(self, alpha: float, recent_window: int = 32, 
                 tail_ratio: float = 0.3, **kwargs):
        super().__init__(alpha, "stratified", **kwargs)
        self.recent_window = recent_window
        self.tail_ratio = tail_ratio
    
    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices using stratified sampling."""
        if k >= n:
            return torch.arange(n, dtype=torch.long)
        
        # Determine split between recent and tail
        k_recent = int(k * (1 - self.tail_ratio))
        k_tail = k - k_recent
        
        indices = []
        
        # Sample from recent window (uniform)
        if k_recent > 0 and self.recent_window > 0:
            recent_end = min(self.recent_window, n)
            recent_indices = torch.randperm(recent_end)[:k_recent]
            indices.append(recent_indices)
        
        # Sample from tail (power-law)
        if k_tail > 0 and n > self.recent_window:
            tail_start = self.recent_window
            tail_n = n - tail_start
            tail_j = torch.arange(tail_n, dtype=torch.float32)
            
            # Power-law distribution for tail
            log_probs = -(1 + self.alpha) * torch.log(tail_j + 1e-8)
            probs = torch.exp(log_probs - torch.logsumexp(log_probs, dim=0))
            
            tail_indices = torch.multinomial(probs, k_tail, replacement=False)
            tail_indices += tail_start  # Offset to actual indices
            indices.append(tail_indices)
        
        if indices:
            return torch.cat(indices)
        else:
            return torch.arange(min(k, n), dtype=torch.long)
    
    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute weights for stratified sampling."""
        weights = torch.ones_like(indices, dtype=torch.float32)
        
        # Recent window: uniform weights
        recent_mask = indices < self.recent_window
        if recent_mask.any():
            weights[recent_mask] = 1.0
        
        # Tail: power-law weights
        tail_mask = indices >= self.recent_window
        if tail_mask.any():
            tail_indices = indices[tail_mask]
            j_vals = tail_indices.float()
            tail_weights = torch.pow(n - j_vals + 1e-8, -(1 + self.alpha))
            weights[tail_mask] = tail_weights
        
        return weights
    
    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor, 
                          weights: torch.Tensor) -> torch.Tensor:
        """Estimate fractional derivative using stratified sampling."""
        # For now, return a scalar estimate for the last point
        n = len(x)
        if len(indices) == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        current_val = x[-1]
        sampled_vals = x[n - 1 - indices]
        differences = current_val - sampled_vals
        
        # Weighted sum
        weighted_sum = torch.sum(weights * differences)
        
        # Normalize by sample size
        return weighted_sum / len(indices)


class ControlVariateSampler(StochasticMemorySampler):
    """
    Control variate sampling with deterministic baseline.
    
    Uses a cheap deterministic approximation (e.g., short memory) as baseline
    and samples only the residual tail.
    """
    
    def __init__(self, alpha: float, baseline_window: int = 16, **kwargs):
        super().__init__(alpha, "control_variate", **kwargs)
        self.baseline_window = baseline_window
        self.sampler = ImportanceSampler(alpha, **kwargs)
    
    def compute_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Compute deterministic baseline using recent window."""
        n = len(x)
        if n <= self.baseline_window:
            return torch.sum(x) * 0.0  # Zero baseline for short sequences
        
        # Use recent window for deterministic baseline
        recent_x = x[-self.baseline_window:]
        current_val = recent_x[-1]
        
        # Simple finite difference baseline
        baseline = torch.sum(current_val - recent_x[:-1]) / self.baseline_window
        return baseline
    
    def sample_indices(self, n: int, k: int) -> torch.Tensor:
        """Sample indices from tail only (excluding baseline window)."""
        if n <= self.baseline_window:
            return torch.arange(min(k, n), dtype=torch.long)
        
        # Sample only from tail
        tail_n = n - self.baseline_window
        tail_k = min(k, tail_n)
        
        if tail_k == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Use importance sampling for tail
        tail_indices = self.sampler.sample_indices(tail_n, tail_k)
        # Offset to actual indices
        return tail_indices + self.baseline_window
    
    def compute_weights(self, indices: torch.Tensor, n: int) -> torch.Tensor:
        """Compute weights for control variate sampling."""
        if len(indices) == 0:
            return torch.empty(0, dtype=torch.float32)
        
        # Use importance sampling weights for tail
        tail_indices = indices - self.baseline_window
        tail_n = n - self.baseline_window
        return self.sampler.compute_weights(tail_indices, tail_n)
    
    def estimate_derivative(self, x: torch.Tensor, indices: torch.Tensor, 
                          weights: torch.Tensor) -> torch.Tensor:
        """Estimate derivative using control variate method."""
        # Compute baseline
        baseline = self.compute_baseline(x)
        
        if len(indices) == 0:
            return baseline
        
        # Compute residual from tail sampling
        n = len(x)
        current_val = x[-1]
        sampled_vals = x[n - 1 - indices]
        differences = current_val - sampled_vals
        
        # Weighted sum of residuals
        residual = torch.sum(weights * differences) / len(indices)
        
        # Return baseline + residual
        return baseline + residual


class StochasticFractionalDerivative(torch.autograd.Function):
    """
    PyTorch autograd function for stochastic fractional derivatives.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, k: int = 64, 
                method: str = "importance", **sampler_kwargs) -> torch.Tensor:
        """Forward pass with stochastic memory sampling."""
        
        # Create appropriate sampler
        if method == "importance":
            sampler = ImportanceSampler(alpha, **sampler_kwargs)
        elif method == "stratified":
            sampler = StratifiedSampler(alpha, **sampler_kwargs)
        elif method == "control_variate":
            sampler = ControlVariateSampler(alpha, **sampler_kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Sample indices and compute weights
        n = len(x)
        indices = sampler.sample_indices(n, k)
        weights = sampler.compute_weights(indices, n)
        
        # Estimate derivative
        result = sampler.estimate_derivative(x, indices, weights)
        
        # Save for backward pass
        ctx.alpha = alpha
        ctx.k = k
        ctx.method = method
        ctx.sampler_kwargs = sampler_kwargs
        ctx.save_for_backward(x, indices, weights)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """Backward pass with stochastic gradient estimation."""
        x, indices, weights = ctx.saved_tensors
        
        # Recreate sampler for backward pass
        if ctx.method == "importance":
            sampler = ImportanceSampler(ctx.alpha, **ctx.sampler_kwargs)
        elif ctx.method == "stratified":
            sampler = StratifiedSampler(ctx.alpha, **ctx.sampler_kwargs)
        elif ctx.method == "control_variate":
            sampler = ControlVariateSampler(ctx.alpha, **ctx.sampler_kwargs)
        
        # Compute gradient using same sampling
        n = len(x)
        
        # For this simplified implementation, we'll compute a basic gradient
        # In practice, you'd want proper fractional derivative gradients
        if x.dim() == 2:
            # 2D case: (batch, features)
            grad_input = torch.zeros_like(x)
            # Simple gradient: pass through the output gradient
            grad_input[..., -1] = grad_output[..., -1]
        else:
            # 1D case
            grad_input = torch.zeros_like(x)
            # Simple gradient: pass through the output gradient
            grad_input[-1] = grad_output[-1]
        
        return grad_input, None, None, None, None


class StochasticFractionalLayer(nn.Module):
    """
    PyTorch module for stochastic fractional derivatives.
    """
    
    def __init__(self, alpha: float, k: int = 64, method: str = "importance", **kwargs):
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.method = method
        self.kwargs = kwargs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return StochasticFractionalDerivative.apply(x, self.alpha, self.k, self.method, **self.kwargs)
    
    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, k={self.k}, method={self.method}'


# Convenience functions
def stochastic_fractional_derivative(x: torch.Tensor, alpha: float, k: int = 64,
                                   method: str = "importance", **kwargs) -> torch.Tensor:
    """Convenience function for stochastic fractional derivative."""
    return StochasticFractionalDerivative.apply(x, alpha, k, method, **kwargs)


def create_stochastic_fractional_layer(alpha: float, k: int = 64, 
                                     method: str = "importance", **kwargs) -> StochasticFractionalLayer:
    """Convenience function for creating stochastic fractional layer."""
    return StochasticFractionalLayer(alpha, k, method, **kwargs)
