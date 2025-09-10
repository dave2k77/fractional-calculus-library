"""
Probabilistic Fractional Orders Implementation

This module implements probabilistic fractional orders where the fractional order
itself becomes a random variable, enabling uncertainty quantification and robust optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Tuple, Optional, Union, Callable, Dict, Any
import math


class ProbabilisticFractionalOrder(nn.Module):
    """
    Base class for probabilistic fractional orders.
    """
    
    def __init__(self, distribution: dist.Distribution, learnable: bool = True):
        super().__init__()
        self.distribution = distribution
        self.learnable = learnable
        self._parameters = {}
        
        if learnable:
            self._setup_learnable_parameters()
    
    def _setup_learnable_parameters(self):
        """Setup learnable parameters for the distribution."""
        if isinstance(self.distribution, dist.Normal):
            self._parameters['loc'] = nn.Parameter(self.distribution.loc.clone().requires_grad_(True))
            self._parameters['scale'] = nn.Parameter(self.distribution.scale.clone().requires_grad_(True))
            # Register parameters with PyTorch
            self.register_parameter('loc', self._parameters['loc'])
            self.register_parameter('scale', self._parameters['scale'])
        elif isinstance(self.distribution, dist.Uniform):
            self._parameters['low'] = nn.Parameter(self.distribution.low.clone().requires_grad_(True))
            self._parameters['high'] = nn.Parameter(self.distribution.high.clone().requires_grad_(True))
            # Register parameters with PyTorch
            self.register_parameter('low', self._parameters['low'])
            self.register_parameter('high', self._parameters['high'])
        elif isinstance(self.distribution, dist.Beta):
            self._parameters['concentration1'] = nn.Parameter(self.distribution.concentration1.clone().requires_grad_(True))
            self._parameters['concentration0'] = nn.Parameter(self.distribution.concentration0.clone().requires_grad_(True))
            # Register parameters with PyTorch
            self.register_parameter('concentration1', self._parameters['concentration1'])
            self.register_parameter('concentration0', self._parameters['concentration0'])
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample fractional order from distribution."""
        if self.learnable and self._parameters:
            # Create distribution with current parameters
            if isinstance(self.distribution, dist.Normal):
                current_dist = dist.Normal(self._parameters['loc'], 
                                         F.softplus(self._parameters['scale']))
            elif isinstance(self.distribution, dist.Uniform):
                current_dist = dist.Uniform(self._parameters['low'], 
                                          self._parameters['high'])
            elif isinstance(self.distribution, dist.Beta):
                current_dist = dist.Beta(F.softplus(self._parameters['concentration1']),
                                       F.softplus(self._parameters['concentration0']))
            else:
                current_dist = self.distribution
        else:
            current_dist = self.distribution
        
        return current_dist.sample(sample_shape)
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability of fractional order."""
        if self.learnable and self._parameters:
            if isinstance(self.distribution, dist.Normal):
                current_dist = dist.Normal(self._parameters['loc'], 
                                         F.softplus(self._parameters['scale']))
            elif isinstance(self.distribution, dist.Uniform):
                current_dist = dist.Uniform(self._parameters['low'], 
                                          self._parameters['high'])
            elif isinstance(self.distribution, dist.Beta):
                current_dist = dist.Beta(F.softplus(self._parameters['concentration1']),
                                       F.softplus(self._parameters['concentration0']))
            else:
                current_dist = self.distribution
        else:
            current_dist = self.distribution
        
        return current_dist.log_prob(value)
    
    def parameters(self):
        """Return learnable parameters."""
        if self.learnable:
            return self._parameters.values()
        else:
            return []


class ReparameterizedFractionalDerivative(torch.autograd.Function):
    """
    Reparameterized fractional derivative using reparameterization trick.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha_dist: ProbabilisticFractionalOrder,
                epsilon: torch.Tensor, method: str, k: int) -> torch.Tensor:
        """Forward pass with reparameterized fractional order."""
        
        # Sample alpha using reparameterization trick
        if alpha_dist.learnable and alpha_dist._parameters:
            if isinstance(alpha_dist.distribution, dist.Normal):
                # Normal: alpha = mu + sigma * epsilon
                alpha = alpha_dist._parameters['loc'] + F.softplus(alpha_dist._parameters['scale']) * epsilon
            elif isinstance(alpha_dist.distribution, dist.Uniform):
                # Uniform: alpha = low + (high - low) * epsilon
                alpha = alpha_dist._parameters['low'] + (alpha_dist._parameters['high'] - alpha_dist._parameters['low']) * epsilon
            elif isinstance(alpha_dist.distribution, dist.Beta):
                # Beta: use inverse CDF sampling (approximate)
                # For simplicity, use normal approximation
                mean = alpha_dist._parameters['concentration1'] / (alpha_dist._parameters['concentration1'] + alpha_dist._parameters['concentration0'])
                var = (alpha_dist._parameters['concentration1'] * alpha_dist._parameters['concentration0']) / \
                      ((alpha_dist._parameters['concentration1'] + alpha_dist._parameters['concentration0'])**2 * 
                       (alpha_dist._parameters['concentration1'] + alpha_dist._parameters['concentration0'] + 1))
                alpha = mean + torch.sqrt(var) * epsilon
            else:
                # Fallback to direct sampling
                alpha = alpha_dist.sample()
        else:
            # Fallback to direct sampling
            alpha = alpha_dist.sample()
        
        # Compute fractional derivative with sampled alpha
        from .stochastic_memory_sampling import stochastic_fractional_derivative
        result = stochastic_fractional_derivative(x, alpha, method=method, k=k)
        
        # Save for backward pass
        ctx.alpha_dist = alpha_dist
        ctx.epsilon = epsilon
        ctx.method = method
        ctx.k = k
        ctx.save_for_backward(x, alpha)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """Backward pass with reparameterized gradients."""
        x, alpha = ctx.saved_tensors
        
        # Compute gradient with respect to input x
        # For fractional derivatives, we need to compute the gradient through the stochastic function
        # This is a simplified implementation - in practice, you'd need proper fractional derivative gradients
        # Ensure grad_x has the same shape as x
        grad_x = torch.zeros_like(x)
        
        # Gradient with respect to distribution parameters (only if learnable)
        if ctx.alpha_dist.learnable and ctx.alpha_dist._parameters:
            # Compute gradients for distribution parameters using reparameterization trick
            if isinstance(ctx.alpha_dist.distribution, dist.Normal):
                # For Normal distribution: alpha = mu + sigma * epsilon
                # dL/dmu = dL/dalpha * dalpha/dmu = dL/dalpha * 1
                # dL/dsigma = dL/dalpha * dalpha/dsigma = dL/dalpha * epsilon
                if 'loc' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['loc'].requires_grad:
                    if ctx.alpha_dist._parameters['loc'].grad is None:
                        ctx.alpha_dist._parameters['loc'].grad = torch.zeros_like(ctx.alpha_dist._parameters['loc'])
                    ctx.alpha_dist._parameters['loc'].grad += grad_output.sum()
                
                if 'scale' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['scale'].requires_grad:
                    if ctx.alpha_dist._parameters['scale'].grad is None:
                        ctx.alpha_dist._parameters['scale'].grad = torch.zeros_like(ctx.alpha_dist._parameters['scale'])
                    ctx.alpha_dist._parameters['scale'].grad += (grad_output.sum() * ctx.epsilon)
            
            elif isinstance(ctx.alpha_dist.distribution, dist.Uniform):
                # For Uniform distribution: alpha = low + (high - low) * epsilon
                # dL/dlow = dL/dalpha * dalpha/dlow = dL/dalpha * (1 - epsilon)
                # dL/dhigh = dL/dalpha * dalpha/dhigh = dL/dalpha * epsilon
                if 'low' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['low'].requires_grad:
                    if ctx.alpha_dist._parameters['low'].grad is None:
                        ctx.alpha_dist._parameters['low'].grad = torch.zeros_like(ctx.alpha_dist._parameters['low'])
                    ctx.alpha_dist._parameters['low'].grad += (grad_output.sum() * (1 - ctx.epsilon))
                
                if 'high' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['high'].requires_grad:
                    if ctx.alpha_dist._parameters['high'].grad is None:
                        ctx.alpha_dist._parameters['high'].grad = torch.zeros_like(ctx.alpha_dist._parameters['high'])
                    ctx.alpha_dist._parameters['high'].grad += (grad_output.sum() * ctx.epsilon)
            
            elif isinstance(ctx.alpha_dist.distribution, dist.Beta):
                # For Beta distribution: alpha = mean + sqrt(var) * epsilon
                # where mean = c1/(c1+c0) and var = c1*c0/((c1+c0)^2*(c1+c0+1))
                # This is a simplified gradient computation
                if 'concentration1' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['concentration1'].requires_grad:
                    if ctx.alpha_dist._parameters['concentration1'].grad is None:
                        ctx.alpha_dist._parameters['concentration1'].grad = torch.zeros_like(ctx.alpha_dist._parameters['concentration1'])
                    ctx.alpha_dist._parameters['concentration1'].grad += grad_output.sum()
                
                if 'concentration0' in ctx.alpha_dist._parameters and ctx.alpha_dist._parameters['concentration0'].requires_grad:
                    if ctx.alpha_dist._parameters['concentration0'].grad is None:
                        ctx.alpha_dist._parameters['concentration0'].grad = torch.zeros_like(ctx.alpha_dist._parameters['concentration0'])
                    ctx.alpha_dist._parameters['concentration0'].grad += grad_output.sum()
        
        return grad_x, None, None, None, None


class ScoreFunctionFractionalDerivative(torch.autograd.Function):
    """
    Score function fractional derivative using REINFORCE estimator.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha_dist: ProbabilisticFractionalOrder,
                method: str, k: int) -> torch.Tensor:
        """Forward pass with score function estimator."""
        
        # Sample alpha from distribution
        alpha = alpha_dist.sample()
        
        # Compute fractional derivative
        from .stochastic_memory_sampling import stochastic_fractional_derivative
        result = stochastic_fractional_derivative(x, alpha, method=method, k=k)
        
        # Save for backward pass
        ctx.alpha_dist = alpha_dist
        ctx.alpha = alpha
        ctx.method = method
        ctx.k = k
        ctx.save_for_backward(x)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """Backward pass with score function estimator."""
        x = ctx.saved_tensors[0]
        
        # For now, return zero gradients for input x
        # TODO: Implement proper score function gradient computation
        grad_x = torch.zeros_like(x)
        
        # Gradient with respect to distribution parameters (only if learnable)
        if ctx.alpha_dist.learnable and ctx.alpha_dist._parameters:
            # Compute gradients for distribution parameters
            # This is a simplified implementation - in practice, you'd need
            # to properly implement the score function gradient
            for name, param in ctx.alpha_dist._parameters.items():
                if param.requires_grad:
                    # Placeholder gradient computation
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
        
        return grad_x, None, None, None


class ProbabilisticFractionalLayer(nn.Module):
    """
    PyTorch module for probabilistic fractional derivatives.
    """
    
    def __init__(self, alpha_dist: Union[dist.Distribution, ProbabilisticFractionalOrder],
                 method: str = "reparameterized", learnable: bool = True, **kwargs):
        super().__init__()
        
        if isinstance(alpha_dist, dist.Distribution):
            self.alpha_dist = ProbabilisticFractionalOrder(alpha_dist, learnable)
        else:
            self.alpha_dist = alpha_dist
        
        self.method = method
        self.kwargs = kwargs
        
        # Add parameters to module
        if learnable:
            for name, param in self.alpha_dist._parameters.items():
                self.register_parameter(f'alpha_{name}', param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        method = self.kwargs.get('method', 'importance')
        k = self.kwargs.get('k', 32)
        
        if self.method == "reparameterized":
            # Generate epsilon for reparameterization with correct shape
            if self.alpha_dist.learnable and self.alpha_dist._parameters:
                if 'loc' in self.alpha_dist._parameters:
                    epsilon = torch.randn_like(self.alpha_dist._parameters['loc'])
                elif 'low' in self.alpha_dist._parameters:
                    epsilon = torch.randn_like(self.alpha_dist._parameters['low'])
                elif 'concentration1' in self.alpha_dist._parameters:
                    epsilon = torch.randn_like(self.alpha_dist._parameters['concentration1'])
                else:
                    epsilon = torch.randn_like(torch.tensor(0.0))
            else:
                epsilon = torch.randn_like(torch.tensor(0.0))
            return ReparameterizedFractionalDerivative.apply(x, self.alpha_dist, epsilon, method, k)
        elif self.method == "score_function":
            return ScoreFunctionFractionalDerivative.apply(x, self.alpha_dist, method, k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def sample_alpha(self, n_samples: int = 1) -> torch.Tensor:
        """Sample fractional orders from the distribution."""
        return self.alpha_dist.sample(torch.Size([n_samples]))
    
    def get_alpha_statistics(self) -> Dict[str, torch.Tensor]:
        """Get statistics of the fractional order distribution."""
        stats = {}
        
        if isinstance(self.alpha_dist.distribution, dist.Normal):
            stats['mean'] = self.alpha_dist._parameters['loc']
            stats['std'] = F.softplus(self.alpha_dist._parameters['scale'])
        elif isinstance(self.alpha_dist.distribution, dist.Uniform):
            stats['low'] = self.alpha_dist._parameters['low']
            stats['high'] = self.alpha_dist._parameters['high']
            stats['mean'] = (stats['low'] + stats['high']) / 2
        elif isinstance(self.alpha_dist.distribution, dist.Beta):
            conc1 = F.softplus(self.alpha_dist._parameters['concentration1'])
            conc0 = F.softplus(self.alpha_dist._parameters['concentration0'])
            stats['mean'] = conc1 / (conc1 + conc0)
            stats['var'] = (conc1 * conc0) / ((conc1 + conc0)**2 * (conc1 + conc0 + 1))
        
        return stats
    
    def extra_repr(self) -> str:
        return f'method={self.method}, learnable={self.alpha_dist.learnable}'


class BayesianFractionalOptimizer:
    """
    Bayesian optimizer for probabilistic fractional orders.
    """
    
    def __init__(self, model: nn.Module, alpha_layers: list, 
                 prior_weight: float = 0.01, lr: float = 0.001):
        self.model = model
        self.alpha_layers = alpha_layers
        self.prior_weight = prior_weight
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Separate optimizer for alpha parameters
        alpha_params = []
        for layer in alpha_layers:
            alpha_params.extend(layer.alpha_dist.parameters())
        self.alpha_optimizer = torch.optim.Adam(alpha_params, lr=lr)
    
    def step(self, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Optimization step with Bayesian updates."""
        
        # Forward pass
        y_pred = self.model(x)
        
        # Data loss
        data_loss = loss_fn(y_pred, y)
        
        # Prior loss (KL divergence from prior)
        prior_loss = 0.0
        for layer in self.alpha_layers:
            if layer.alpha_dist.learnable:
                # Sample from current distribution
                alpha = layer.alpha_dist.sample()
                log_prob = layer.alpha_dist.log_prob(alpha)
                
                # Prior (e.g., uniform on [0, 1])
                prior = dist.Uniform(0.0, 1.0)
                prior_log_prob = prior.log_prob(alpha)
                
                # KL divergence approximation
                kl_div = log_prob - prior_log_prob
                prior_loss += kl_div
        
        # Total loss
        total_loss = data_loss + self.prior_weight * prior_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        total_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        self.alpha_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'prior_loss': prior_loss.item()
        }


# Convenience functions
def create_probabilistic_fractional_layer(alpha_dist: dist.Distribution,
                                        method: str = "reparameterized",
                                        learnable: bool = True,
                                        **kwargs) -> ProbabilisticFractionalLayer:
    """Create a probabilistic fractional layer."""
    return ProbabilisticFractionalLayer(alpha_dist, method, learnable, **kwargs)


def create_normal_alpha_layer(mean: float = 0.5, std: float = 0.1,
                            method: str = "reparameterized",
                            learnable: bool = True,
                            **kwargs) -> ProbabilisticFractionalLayer:
    """Create probabilistic fractional layer with normal distribution."""
    alpha_dist = dist.Normal(mean, std)
    return ProbabilisticFractionalLayer(alpha_dist, method, learnable, **kwargs)


def create_uniform_alpha_layer(low: float = 0.1, high: float = 0.9,
                             method: str = "reparameterized",
                             learnable: bool = True,
                             **kwargs) -> ProbabilisticFractionalLayer:
    """Create probabilistic fractional layer with uniform distribution."""
    alpha_dist = dist.Uniform(low, high)
    return ProbabilisticFractionalLayer(alpha_dist, method, learnable, **kwargs)


def create_beta_alpha_layer(concentration1: float = 2.0, concentration0: float = 2.0,
                          method: str = "reparameterized",
                          learnable: bool = True,
                          **kwargs) -> ProbabilisticFractionalLayer:
    """Create probabilistic fractional layer with beta distribution."""
    alpha_dist = dist.Beta(concentration1, concentration0)
    return ProbabilisticFractionalLayer(alpha_dist, method, learnable, **kwargs)
