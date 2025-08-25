"""
Simplified Optimizers with Fractional Calculus Integration

This module provides clean, backend-agnostic optimizers that incorporate 
fractional derivatives, designed to work seamlessly across PyTorch, JAX, and NUMBA.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import warnings

from ..core.definitions import FractionalOrder
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class SimpleFractionalOptimizer(ABC):
    """
    Simplified base class for optimizers with fractional calculus integration
    
    This class provides a clean framework that avoids complex state management
    and PyTorch-specific dependencies. Supports multiple backends: PyTorch, JAX, and NUMBA.
    """
    
    def __init__(self, 
                 lr: float = 0.001,
                 fractional_order: float = 0.5, 
                 method: str = "RL", 
                 use_fractional: bool = True, 
                 backend: Optional[BackendType] = None):
        self.lr = lr
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.use_fractional = use_fractional
        
        # Set backend
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        
        # Simple state storage using indices instead of objects as keys
        self.state = {}
        self.param_count = 0
        self._param_id_map = {}  # Map from object id to parameter index
    
    def _get_param_index(self, param) -> int:
        """Get or create a unique index for a parameter"""
        # Use object id as a hashable identifier instead of setting attributes
        param_id = id(param)
        if param_id not in self._param_id_map:
            self._param_id_map[param_id] = self.param_count
            self.param_count += 1
        return self._param_id_map[param_id]
    
    def _get_state(self, param) -> Dict[str, Any]:
        """Get state for a parameter using its index"""
        param_idx = self._get_param_index(param)
        if param_idx not in self.state:
            self.state[param_idx] = self._initialize_param_state(param)
        return self.state[param_idx]
    
    def _initialize_param_state(self, param) -> Dict[str, Any]:
        """Initialize state for a single parameter"""
        # This will be overridden by subclasses
        return {}
    
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
        
        # For now, only PyTorch backend supports fractional derivatives
        # JAX and NUMBA backends return gradients unchanged
        if self.backend == BackendType.TORCH:
            try:
                import torch
                from .fractional_autograd import fractional_derivative
                
                # Create a copy to avoid modifying the original tensor
                if hasattr(gradients, 'clone'):
                    gradients_copy = gradients.clone()
                else:
                    gradients_copy = gradients
                
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
            except Exception as e:
                warnings.warn(f"Fractional derivative failed: {e}. Using original gradients.")
                return gradients
        else:
            # For JAX/NUMBA, return gradients unchanged (no fractional derivative support yet)
            return gradients
    
    @abstractmethod
    def step(self, params: List[Any], gradients: List[Any]) -> None:
        """Perform a single optimization step"""
        pass
    
    def zero_grad(self, params: Optional[List[Any]] = None) -> None:
        """Zero the gradients of all optimized tensors (if supported by backend)"""
        # This is a simplified version - in practice, gradients are managed by the training loop
        # Accept optional params to match torch-like API
        pass


class SimpleFractionalSGD(SimpleFractionalOptimizer):
    """
    Simplified SGD optimizer with fractional calculus integration
    """
    
    def __init__(self, 
                 lr: float = 0.001,
                 momentum: float = 0.0,
                 fractional_order: float = 0.5, 
                 method: str = "RL", 
                 use_fractional: bool = True, 
                 backend: Optional[BackendType] = None):
        super().__init__(lr, fractional_order, method, use_fractional, backend)
        self.momentum = momentum
    
    def _initialize_param_state(self, param) -> Dict[str, Any]:
        """Initialize state for a single parameter"""
        state = {}
        if self.momentum > 0:
            # Create momentum buffer with same shape as parameter
            if hasattr(param, 'shape'):
                state['momentum_buffer'] = self.tensor_ops.zeros_like(param)
            else:
                # Fallback for parameters without shape attribute
                state['momentum_buffer'] = self.tensor_ops.create_tensor(0.0)
        return state
    
    def step(self, params: List[Any], gradients: List[Any]) -> None:
        """Perform a single optimization step with fractional gradients"""
        if len(params) != len(gradients):
            raise ValueError("Number of parameters must match number of gradients")
        
        for param, grad in zip(params, gradients):
            if grad is None:
                continue
            
            # Apply fractional derivative to gradients
            grad = self.fractional_update(grad)
            
            # Get state for this parameter
            state = self._get_state(param)
            
            # Apply momentum if enabled
            if self.momentum > 0:
                if 'momentum_buffer' in state:
                    momentum_buffer = state['momentum_buffer']
                    # Update momentum buffer: v = momentum * v + grad
                    momentum_buffer = self.momentum * momentum_buffer + grad
                    state['momentum_buffer'] = momentum_buffer
                    grad = momentum_buffer
            
            # Update parameter: param = param - lr * grad
            try:
                # Try PyTorch-style parameter update
                if hasattr(param, 'data') and hasattr(param.data, '__setitem__'):
                    param.data = param.data - self.lr * grad
                else:
                    # For other backends, we need to handle the update differently
                    # Since we can't modify the original array, we'll return the new value
                    # In practice, this would be handled by the training loop
                    new_param = param - self.lr * grad
                    # For now, just store the update in state for demonstration
                    state['last_update'] = new_param
            except (AttributeError, TypeError):
                # Fallback for non-writable parameters
                new_param = param - self.lr * grad
                state['last_update'] = new_param


class SimpleFractionalAdam(SimpleFractionalOptimizer):
    """
    Simplified Adam optimizer with fractional calculus integration
    """
    
    def __init__(self, 
                 params=None,
                 lr: float = 0.001,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 fractional_order: float = 0.5, 
                 method: str = "RL", 
                 use_fractional: bool = True, 
                 backend: Optional[BackendType] = None, *args, **kwargs):
        # Accept params as first argument to match torch-like signature in tests
        super().__init__(lr, fractional_order, method, use_fractional, backend)
        self.betas = betas
        self.eps = eps
        self.params = params  # Store parameters for step() method
    
    def _initialize_param_state(self, param) -> Dict[str, Any]:
        """Initialize state for a single parameter"""
        state = {}
        if hasattr(param, 'shape'):
            state['exp_avg'] = self.tensor_ops.zeros_like(param)
            state['exp_avg_sq'] = self.tensor_ops.zeros_like(param)
        else:
            # Fallback for parameters without shape attribute
            state['exp_avg'] = self.tensor_ops.create_tensor(0.0)
            state['exp_avg_sq'] = self.tensor_ops.create_tensor(0.0)
        state['step'] = 0
        return state
    
    def step(self, params: Optional[List[Any]] = None, gradients: Optional[List[Any]] = None) -> None:
        """Perform a single optimization step with fractional gradients"""
        # Use stored params if none provided
        if params is None:
            params = self.params
        if params is None:
            raise ValueError("No parameters provided for optimization step")
        
        # For torch-like API, gradients are computed via autograd and stored in param.grad
        if gradients is None:
            gradients = []
            for param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    gradients.append(param.grad)
                else:
                    gradients.append(None)
        
        if len(params) != len(gradients):
            raise ValueError("Number of parameters must match number of gradients")
        
        for param, grad in zip(params, gradients):
            if grad is None:
                continue
            
            # Apply fractional derivative to gradients
            grad = self.fractional_update(grad)
            
            # Get state for this parameter
            state = self._get_state(param)
            
            # Update step count
            state['step'] += 1
            
            # Get momentum and variance parameters
            beta1, beta2 = self.betas
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            
            # Update momentum and variance
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
            
            # Store updated state
            state['exp_avg'] = exp_avg
            state['exp_avg_sq'] = exp_avg_sq
            
            # Compute bias correction
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            
            # Update parameter
            step_size = self.lr / bias_correction1
            bias_correction2_sqrt = bias_correction2 ** 0.5
            
            # param = param - step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
            # Use tensor operations for sqrt to ensure backend compatibility
            sqrt_exp_avg_sq = self.tensor_ops.sqrt(exp_avg_sq)
            
            try:
                # Try PyTorch-style parameter update
                if hasattr(param, 'data') and hasattr(param.data, '__setitem__'):
                    param.data = param.data - step_size * exp_avg / (sqrt_exp_avg_sq + self.eps)
                else:
                    # For other backends, we need to handle the update differently
                    # Since we can't modify the original array, we'll return the new value
                    # In practice, this would be handled by the training loop
                    new_param = param - step_size * exp_avg / (sqrt_exp_avg_sq + self.eps)
                    # For now, just store the update in state for demonstration
                    state['last_update'] = new_param
            except (AttributeError, TypeError):
                # Fallback for non-writable parameters
                new_param = param - step_size * exp_avg / (sqrt_exp_avg_sq + self.eps)
                state['last_update'] = new_param


class SimpleFractionalRMSprop(SimpleFractionalOptimizer):
    """
    Simplified RMSprop optimizer with fractional calculus integration
    """
    
    def __init__(self, 
                 lr: float = 0.001,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 fractional_order: float = 0.5, 
                 method: str = "RL", 
                 use_fractional: bool = True, 
                 backend: Optional[BackendType] = None):
        super().__init__(lr, fractional_order, method, use_fractional, backend)
        self.alpha = alpha
        self.eps = eps
    
    def _initialize_param_state(self, param) -> Dict[str, Any]:
        """Initialize state for a single parameter"""
        state = {}
        if hasattr(param, 'shape'):
            state['square_avg'] = self.tensor_ops.zeros_like(param)
        else:
            # Fallback for parameters without shape attribute
            state['square_avg'] = self.tensor_ops.create_tensor(0.0)
        return state
    
    def step(self, params: List[Any], gradients: List[Any]) -> None:
        """Perform a single optimization step with fractional gradients"""
        if len(params) != len(gradients):
            raise ValueError("Number of parameters must match number of gradients")
        
        for param, grad in zip(params, gradients):
            if grad is None:
                continue
            
            # Apply fractional derivative to gradients
            grad = self.fractional_update(grad)
            
            # Get state for this parameter
            state = self._get_state(param)
            square_avg = state['square_avg']
            
            # Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
            square_avg = self.alpha * square_avg + (1 - self.alpha) * (grad * grad)
            state['square_avg'] = square_avg
            
            # Update parameter: param = param - lr * grad / (sqrt(square_avg) + eps)
            # Use tensor operations for sqrt to ensure backend compatibility
            sqrt_square_avg = self.tensor_ops.sqrt(square_avg)
            
            try:
                # Try PyTorch-style parameter update
                if hasattr(param, 'data') and hasattr(param.data, '__setitem__'):
                    param.data = param.data - self.lr * grad / (sqrt_square_avg + self.eps)
                else:
                    # For other backends, we need to handle the update differently
                    # Since we can't modify the original array, we'll return the new value
                    # In practice, this would be handled by the training loop
                    new_param = param - self.lr * grad / (sqrt_square_avg + self.eps)
                    # For now, just store the update in state for demonstration
                    state['last_update'] = new_param
            except (AttributeError, TypeError):
                # Fallback for non-writable parameters
                new_param = param - self.lr * grad / (sqrt_square_avg + self.eps)
                state['last_update'] = new_param


# Convenience aliases for backward compatibility
FractionalOptimizer = SimpleFractionalOptimizer
FractionalSGD = SimpleFractionalSGD
FractionalAdam = SimpleFractionalAdam
FractionalRMSprop = SimpleFractionalRMSprop

# Additional optimizers can be added here following the same pattern
