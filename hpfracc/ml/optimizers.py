"""
Optimizers with Fractional Calculus Integration

This module provides optimizers that incorporate fractional derivatives,
enabling enhanced optimization dynamics and potentially better convergence.
"""

import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative


class FractionalOptimizer(ABC):
    """
    Base class for optimizers with fractional calculus integration
    
    This class provides a framework for optimizers that can apply
    fractional derivatives to gradients before updating parameters.
    """
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", use_fractional: bool = True):
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.use_fractional = use_fractional
    
    def fractional_update(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Apply fractional derivative to gradients
        
        Args:
            gradients: Input gradients
            
        Returns:
            Gradients with fractional derivative applied
        """
        if not self.use_fractional:
            return gradients
        
        # Create a copy to avoid modifying the original tensor
        gradients_copy = gradients.clone()
        
        # Store original gradient magnitude for scaling
        original_norm = gradients_copy.norm()
        
        # Apply fractional derivative
        updated_gradients = fractional_derivative(gradients_copy, self.fractional_order.alpha, self.method)
        
        # Scale to preserve gradient magnitude (important for optimization)
        if original_norm > 0:
            updated_norm = updated_gradients.norm()
            if updated_norm > 0:
                # Scale to maintain similar magnitude
                scale_factor = original_norm / updated_norm
                updated_gradients = updated_gradients * scale_factor
        
        return updated_gradients
    
    @abstractmethod
    def step(self, closure: Optional[callable] = None):
        """Perform a single optimization step"""
        pass
    
    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        pass


class FractionalAdam(FractionalOptimizer, optim.Adam):
    """
    Adam optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        fractional_order: float = 0.5,
        method: str = "RL",
        use_fractional: bool = True
    ):
        FractionalOptimizer.__init__(self, fractional_order, method, use_fractional)
        optim.Adam.__init__(self, params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply fractional derivative to gradients (create new tensor)

                
                p.grad.data = self.fractional_update(p.grad.data.clone())
        
        # Call parent step method - explicitly call Adam\'s step method

        
        return optim.Adam.step(self, closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        return super().zero_grad(set_to_none)


class FractionalSGD(FractionalOptimizer, optim.SGD):
    """
    SGD optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        fractional_order: float = 0.5,
        method: str = "RL",
        use_fractional: bool = True
    ):
        FractionalOptimizer.__init__(self, fractional_order, method, use_fractional)
        optim.SGD.__init__(self, params, lr, momentum, dampening, weight_decay, nesterov)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply fractional derivative to gradients (create new tensor)

                
                p.grad.data = self.fractional_update(p.grad.data.clone())
        
        # Call parent step method - explicitly call Adam\'s step method

        
        return optim.Adam.step(self, closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        return super().zero_grad(set_to_none)


class FractionalRMSprop(FractionalOptimizer, optim.RMSprop):
    """
    RMSprop optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        fractional_order: float = 0.5,
        method: str = "RL",
        use_fractional: bool = True
    ):
        FractionalOptimizer.__init__(self, fractional_order, method, use_fractional)
        optim.RMSprop.__init__(self, params, lr, alpha, eps, weight_decay, momentum, centered)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply fractional derivative to gradients (create new tensor)

                
                p.grad.data = self.fractional_update(p.grad.data.clone())
        
        # Call parent step method - explicitly call Adam\'s step method

        
        return optim.Adam.step(self, closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        return super().zero_grad(set_to_none)


class FractionalAdagrad(FractionalOptimizer, optim.Adagrad):
    """
    Adagrad optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        fractional_order: float = 0.5,
        method: str = "RL",
        use_fractional: bool = True
    ):
        FractionalOptimizer.__init__(self, fractional_order, method, use_fractional)
        optim.Adagrad.__init__(self, params, lr, lr_decay, weight_decay, initial_accumulator_value, eps)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply fractional derivative to gradients (create new tensor)

                
                p.grad.data = self.fractional_update(p.grad.data.clone())
        
        # Call parent step method - explicitly call Adam\'s step method

        
        return optim.Adam.step(self, closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        return super().zero_grad(set_to_none)


class FractionalAdamW(FractionalOptimizer, optim.AdamW):
    """
    AdamW optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        fractional_order: float = 0.5,
        method: str = "RL",
        use_fractional: bool = True
    ):
        FractionalOptimizer.__init__(self, fractional_order, method, use_fractional)
        optim.AdamW.__init__(self, params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply fractional derivative to gradients (create new tensor)

                
                p.grad.data = self.fractional_update(p.grad.data.clone())
        
        # Call parent step method - explicitly call Adam\'s step method

        
        return optim.Adam.step(self, closure)
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        return super().zero_grad(set_to_none)


# RAdam optimizer not available in current PyTorch version
# class FractionalRAdam(optim.RAdam):
#     """RAdam optimizer with fractional calculus integration"""
#     pass

# Lion optimizer not available in current PyTorch version
# class FractionalLion(optim.Lion):
#     """Lion optimizer with fractional calculus integration"""
#     pass
