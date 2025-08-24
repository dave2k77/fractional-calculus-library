"""
Optimizers with Fractional Calculus Integration

This module provides optimizers that incorporate fractional derivatives,
enabling enhanced optimization dynamics and potentially better convergence.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class FractionalOptimizer(ABC):
    """
    Base class for optimizers with fractional calculus integration
    
    This class provides a framework for optimizers that can apply
    fractional derivatives to gradients before updating parameters.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", use_fractional: bool = True, backend: Optional[BackendType] = None):
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.use_fractional = use_fractional
        
        # Set backend
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
    
    def fractional_update(self, gradients: Any) -> Any:
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
        if self.backend == BackendType.TORCH:
            gradients_copy = gradients.clone()
        else:
            gradients_copy = self.tensor_ops.create_tensor(gradients)
        
        # Store original gradient magnitude for scaling
        original_norm = self.tensor_ops.norm(gradients_copy)
        
        # Apply fractional derivative
        updated_gradients = fractional_derivative(gradients_copy, self.fractional_order.alpha, self.method)
        
        # Scale to preserve gradient magnitude (important for optimization)
        if original_norm > 0:
            updated_norm = self.tensor_ops.norm(updated_gradients)
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


class FractionalAdam(FractionalOptimizer):
    """
    Adam optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        use_fractional: bool = True,
        backend: Optional[BackendType] = None
    ):
        super().__init__(fractional_order, method, use_fractional, backend)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize optimizer state
        self._initialize_optimizer_state(params)
    
    def _initialize_optimizer_state(self, params):
        """Initialize optimizer state based on backend"""
        self.state = {}
        self.param_groups = [{'params': params, 'lr': self.lr, 'betas': self.betas, 'eps': self.eps, 'weight_decay': self.weight_decay}]
        
        # Initialize momentum and variance buffers
        for group in self.param_groups:
            for p in group['params']:
                if self.backend == BackendType.TORCH:
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                else:
                    # JAX/NUMBA state management
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['exp_avg'] = self.tensor_ops.zeros_like(p)
                    state['exp_avg_sq'] = self.tensor_ops.zeros_like(p)
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = self.tensor_ops.zeros_like(p)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply fractional derivative to gradients
                grad = self.fractional_update(grad)
                
                # Get state
                state = self.state[p]
                
                # Update step
                state['step'] += 1
                
                # Get parameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update momentum and variance
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Update max variance for AMSGrad
                if self.amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    exp_avg_sq = max_exp_avg_sq
                
                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update parameters
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(eps), value=-step_size / bias_correction2_sqrt)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()


class FractionalSGD(FractionalOptimizer):
    """
    SGD optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        use_fractional: bool = True,
        backend: Optional[BackendType] = None
    ):
        super().__init__(fractional_order, method, use_fractional, backend)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize optimizer state
        self._initialize_optimizer_state(params)
    
    def _initialize_optimizer_state(self, params):
        """Initialize optimizer state based on backend"""
        self.state = {}
        self.param_groups = [{'params': params, 'lr': self.lr, 'momentum': self.momentum, 'dampening': self.dampening, 'weight_decay': self.weight_decay, 'nesterov': self.nesterov}]
        
        # Initialize momentum buffers
        for group in self.param_groups:
            for p in group['params']:
                if self.backend == BackendType.TORCH:
                    state = self.state[p] = {}
                    if group['momentum'] != 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    # JAX/NUMBA state management
                    state = self.state[p] = {}
                    if group['momentum'] != 0:
                        state['momentum_buffer'] = self.tensor_ops.zeros_like(p)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply fractional derivative to gradients
                grad = self.fractional_update(grad)
                
                # Get parameters
                lr = group['lr']
                momentum = group['momentum']
                dampening = group['dampening']
                weight_decay = group['weight_decay']
                nesterov = group['nesterov']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        if self.backend == BackendType.TORCH:
                            state['momentum_buffer'] = torch.zeros_like(p)
                        else:
                            state['momentum_buffer'] = self.tensor_ops.zeros_like(p)
                    
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(momentum_buffer, alpha=momentum)
                    else:
                        grad = momentum_buffer
                
                # Update parameters
                p.data.add_(grad, alpha=-lr)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()


class FractionalRMSprop(FractionalOptimizer):
    """
    RMSprop optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        use_fractional: bool = True,
        backend: Optional[BackendType] = None
    ):
        super().__init__(fractional_order, method, use_fractional, backend)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        
        # Initialize optimizer state
        self._initialize_optimizer_state(params)
    
    def _initialize_optimizer_state(self, params):
        """Initialize optimizer state based on backend"""
        self.state = {}
        self.param_groups = [{'params': params, 'lr': self.lr, 'alpha': self.alpha, 'eps': self.eps, 'weight_decay': self.weight_decay, 'momentum': self.momentum, 'centered': self.centered}]
        
        # Initialize state buffers
        for group in self.param_groups:
            for p in group['params']:
                if self.backend == BackendType.TORCH:
                    state = self.state[p] = {}
                    state['square_avg'] = torch.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    # JAX/NUMBA state management
                    state = self.state[p] = {}
                    state['square_avg'] = self.tensor_ops.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = self.tensor_ops.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = self.tensor_ops.zeros_like(p)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply fractional derivative to gradients
                grad = self.fractional_update(grad)
                
                # Get parameters
                lr = group['lr']
                alpha = group['alpha']
                eps = group['eps']
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                centered = group['centered']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Get state
                state = self.state[p]
                square_avg = state['square_avg']
                
                # Update square average
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                # Update centered average if needed
                if centered:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul_(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
                else:
                    avg = square_avg.sqrt().add_(eps)
                
                # Apply momentum if needed
                if momentum > 0:
                    momentum_buffer = state['momentum_buffer']
                    momentum_buffer.mul_(momentum).addcdiv_(grad, avg, value=lr)
                    p.data.add_(momentum_buffer, alpha=-1)
                else:
                    p.data.addcdiv_(grad, avg, value=-lr)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()


class FractionalAdagrad(FractionalOptimizer):
    """
    Adagrad optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        use_fractional: bool = True,
        backend: Optional[BackendType] = None
    ):
        super().__init__(fractional_order, method, use_fractional, backend)
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        
        # Initialize optimizer state
        self._initialize_optimizer_state(params)
    
    def _initialize_optimizer_state(self, params):
        """Initialize optimizer state based on backend"""
        self.state = {}
        self.param_groups = [{'params': params, 'lr': self.lr, 'lr_decay': self.lr_decay, 'weight_decay': self.weight_decay, 'initial_accumulator_value': self.initial_accumulator_value, 'eps': self.eps}]
        
        # Initialize state buffers
        for group in self.param_groups:
            for p in group['params']:
                if self.backend == BackendType.TORCH:
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['sum'] = torch.full_like(p, group['initial_accumulator_value'])
                else:
                    # JAX/NUMBA state management
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['sum'] = self.tensor_ops.full_like(p, group['initial_accumulator_value'])
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply fractional derivative to gradients
                grad = self.fractional_update(grad)
                
                # Get parameters
                lr = group['lr']
                lr_decay = group['lr_decay']
                weight_decay = group['weight_decay']
                eps = group['eps']
                
                # Get state
                state = self.state[p]
                state['step'] += 1
                sum_ = state['sum']
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update sum
                sum_.addcmul_(grad, grad, value=1)
                
                # Compute learning rate
                clr = lr / (1 + (state['step'] - 1) * lr_decay)
                
                # Update parameters
                p.data.addcdiv_(grad, sum_.sqrt().add_(eps), value=-clr)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()


class FractionalAdamW(FractionalOptimizer):
    """
    AdamW optimizer with fractional calculus integration
    
    This optimizer applies fractional derivatives to gradients before
    updating parameters, enabling enhanced optimization dynamics.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        use_fractional: bool = True,
        backend: Optional[BackendType] = None
    ):
        super().__init__(fractional_order, method, use_fractional, backend)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # Initialize optimizer state
        self._initialize_optimizer_state(params)
    
    def _initialize_optimizer_state(self, params):
        """Initialize optimizer state based on backend"""
        self.state = {}
        self.param_groups = [{'params': params, 'lr': self.lr, 'betas': self.betas, 'eps': self.eps, 'weight_decay': self.weight_decay, 'amsgrad': self.amsgrad}]
        
        # Initialize momentum and variance buffers
        for group in self.param_groups:
            for p in group['params']:
                if self.backend == BackendType.TORCH:
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                else:
                    # JAX/NUMBA state management
                    state = self.state[p] = {}
                    state['step'] = 0
                    state['exp_avg'] = self.tensor_ops.zeros_like(p)
                    state['exp_avg_sq'] = self.tensor_ops.zeros_like(p)
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = self.tensor_ops.zeros_like(p)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step with fractional gradients"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply fractional derivative to gradients
                grad = self.fractional_update(grad)
                
                # Get state
                state = self.state[p]
                
                # Update step
                state['step'] += 1
                
                # Get parameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # Update momentum and variance
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Update max variance for AMSGrad
                if self.amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = torch.maximum(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    exp_avg_sq = max_exp_avg_sq
                
                # Compute bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Update parameters (AdamW style with weight decay)
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5
                
                # Apply weight decay separately (AdamW style)
                p.data.add_(p.data, alpha=-lr * weight_decay)
                
                # Update with gradients
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(eps), value=-step_size / bias_correction2_sqrt)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of all optimized tensors"""
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()


# RAdam optimizer not available in current PyTorch version
# class FractionalRAdam(optim.RAdam):
#     """RAdam optimizer with fractional calculus integration"""
#     pass

# Lion optimizer not available in current PyTorch version
# class FractionalLion(optim.Lion):
#     """Lion optimizer with fractional calculus integration"""
#     pass
