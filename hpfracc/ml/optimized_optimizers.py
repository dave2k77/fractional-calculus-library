"""
Optimized Fractional Calculus Optimizers

This module provides highly optimized fractional calculus optimizers with:
- Unified base classes for common functionality
- Efficient parameter state management
- Fractional derivative caching and optimization
- Optimized tensor operations and backend handling
- Reduced code duplication with shared implementations

Author: Davian R. Chin, Department of Biomedical Engineering, University of Reading
Optimized Optimizers Implementation: September 2025
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
import warnings
import threading

from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops

# Global cache for fractional derivatives
_fractional_cache = {}
_cache_lock = threading.Lock()


@dataclass
class OptimizerConfig:
    """Configuration for optimized optimizers"""
    lr: float = 0.001
    fractional_order: float = 0.5
    method: str = "RL"
    use_fractional: bool = True
    backend: Optional[BackendType] = None
    cache_fractional: bool = True
    memory_efficient: bool = True
    use_jit: bool = True


class OptimizedParameterState:
    """Efficient parameter state management"""

    def __init__(self, param_shape: Tuple[int, ...], backend: BackendType, tensor_ops):
        self.param_shape = param_shape
        self.backend = backend
        self.tensor_ops = tensor_ops
        self._state = {}
        self._step_count = 0

    def get_state(self, key: str, default_factory: Callable = None) -> Any:
        """Get state value with lazy initialization"""
        if key not in self._state and default_factory is not None:
            self._state[key] = default_factory()
        return self._state.get(key)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value"""
        self._state[key] = value

    def increment_step(self) -> int:
        """Increment and return step count"""
        self._step_count += 1
        return self._step_count

    @property
    def step_count(self) -> int:
        return self._step_count


class OptimizedFractionalDerivative:
    """Optimized fractional derivative computation with caching"""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.cache = {} if config.cache_fractional else None
        self.tensor_ops = get_tensor_ops(
            config.backend or get_backend_manager().active_backend)

    def compute_fractional_derivative(self, gradients: Any, alpha: float, method: str) -> Any:
        """Compute fractional derivative with caching and optimization"""

        if not self.config.use_fractional:
            return gradients

        # Create cache key
        if self.cache is not None:
            cache_key = self._create_cache_key(gradients, alpha, method)
            if cache_key in self.cache:
                return self.cache[cache_key]

        try:
            # Use optimized fractional derivative computation
            if self.config.backend == BackendType.TORCH:
                result = self._compute_torch_fractional_derivative(
                    gradients, alpha, method)
            else:
                result = self._compute_fallback_fractional_derivative(
                    gradients, alpha, method)

            # Cache result if enabled
            if self.cache is not None:
                self.cache[cache_key] = result
                # Limit cache size
                if len(self.cache) > 1000:
                    # Remove oldest entries
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]

            return result

        except Exception as e:
            warnings.warn(
                f"Fractional derivative failed: {e}. Using original gradients.")
            return gradients

    def _create_cache_key(self, gradients: Any, alpha: float, method: str) -> str:
        """Create cache key for fractional derivative"""
        # Use tensor hash for caching
        if hasattr(gradients, 'data_ptr'):
            return f"{gradients.data_ptr()}_{alpha}_{method}"
        else:
            return f"{id(gradients)}_{alpha}_{method}"

    def _compute_torch_fractional_derivative(self, gradients: Any, alpha: float, method: str) -> Any:
        """Compute fractional derivative using PyTorch backend"""
        try:
            from .fractional_autograd import fractional_derivative

            # Create a copy to avoid modifying the original tensor
            if hasattr(gradients, 'clone'):
                gradients_copy = gradients.clone()
            else:
                gradients_copy = gradients

            # Store original gradient magnitude for scaling
            original_norm = self.tensor_ops.norm(gradients_copy)

            # Apply fractional derivative
            updated_gradients = fractional_derivative(
                gradients_copy, alpha, method)

            # Scale to preserve gradient magnitude
            if original_norm > 0:
                updated_norm = self.tensor_ops.norm(updated_gradients)
                if updated_norm > 0:
                    scale_factor = original_norm / updated_norm
                    updated_gradients = updated_gradients * scale_factor

            return updated_gradients

        except Exception as e:
            warnings.warn(f"Torch fractional derivative failed: {e}")
            return gradients

    def _compute_fallback_fractional_derivative(self, gradients: Any, alpha: float, method: str) -> Any:
        """Fallback fractional derivative computation"""
        # Simplified fractional derivative approximation
        return gradients * (alpha ** 0.5)


class OptimizedBaseOptimizer(ABC):
    """Optimized base class for fractional calculus optimizers"""

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.backend = config.backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.fractional_derivative = OptimizedFractionalDerivative(config)

        # Efficient parameter state management
        self._param_states = {}
        self._param_count = 0
        self._param_id_map = {}

        # Performance monitoring
        self._step_times = []
        self._fractional_times = []

    def _get_param_state(self, param: Any) -> OptimizedParameterState:
        """Get or create parameter state"""
        param_id = id(param)
        if param_id not in self._param_id_map:
            self._param_id_map[param_id] = self._param_count
            self._param_count += 1

        param_idx = self._param_id_map[param_id]
        if param_idx not in self._param_states:
            # Get parameter shape efficiently
            if hasattr(param, 'shape'):
                param_shape = param.shape
            else:
                param_shape = (1,)  # Fallback

            self._param_states[param_idx] = OptimizedParameterState(
                param_shape, self.backend, self.tensor_ops
            )

        return self._param_states[param_idx]

    def _apply_fractional_derivative(self, gradients: Any) -> Any:
        """Apply fractional derivative with optimization"""
        if not self.config.use_fractional:
            return gradients

        start_time = time.time()
        result = self.fractional_derivative.compute_fractional_derivative(
            gradients, self.config.fractional_order, self.config.method
        )
        self._fractional_times.append(time.time() - start_time)

        return result

    def _update_parameter(self, param: Any, update: Any) -> None:
        """Efficient parameter update"""
        try:
            # Try PyTorch-style parameter update
            if hasattr(param, 'data') and hasattr(param.data, '__setitem__'):
                param.data = param.data - update
            else:
                # For other backends, store update in state
                state = self._get_param_state(param)
                state.set_state('last_update', param - update)
        except (AttributeError, TypeError):
            # Fallback for non-writable parameters
            state = self._get_param_state(param)
            state.set_state('last_update', param - update)

    def zero_grad(self, params: List[Any]) -> None:
        """Zero gradients efficiently"""
        for param in params:
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad.zero_()

    @abstractmethod
    def step(self, params: List[Any], gradients: Optional[List[Any]] = None) -> None:
        """Perform optimization step"""
        pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'avg_step_time': np.mean(self._step_times) if self._step_times else 0,
            'avg_fractional_time': np.mean(self._fractional_times) if self._fractional_times else 0,
            'total_steps': len(self._step_times),
            'fractional_ratio': len(self._fractional_times) / max(len(self._step_times), 1)
        }


class OptimizedFractionalSGD(OptimizedBaseOptimizer):
    """Optimized SGD with fractional calculus integration"""

    def __init__(self,
                 lr: float = 0.001,
                 momentum: float = 0.0,
                 fractional_order: float = 0.5,
                 method: str = "RL",
                 use_fractional: bool = True,
                 backend: Optional[BackendType] = None,
                 **kwargs):

        config = OptimizerConfig(
            lr=lr,
            fractional_order=fractional_order,
            method=method,
            use_fractional=use_fractional,
            backend=backend,
            **kwargs
        )
        super().__init__(config)
        self.momentum = momentum

    def step(self, params: List[Any], gradients: Optional[List[Any]] = None) -> None:
        """Optimized SGD step"""
        start_time = time.time()

        # Get gradients
        if gradients is None:
            gradients = []
            for param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    gradients.append(param.grad)
                else:
                    gradients.append(None)

        if len(params) != len(gradients):
            raise ValueError(
                "Number of parameters must match number of gradients")

        for param, grad in zip(params, gradients):
            if grad is None:
                continue

            # Apply fractional derivative
            if self.config.use_fractional:
                grad = self._apply_fractional_derivative(grad)

            # Get parameter state
            state = self._get_param_state(param)

            # Apply momentum if enabled
            if self.momentum > 0:
                momentum_buffer = state.get_state(
                    'momentum_buffer',
                    lambda: self.tensor_ops.zeros_like(param)
                )

                # Update momentum buffer
                momentum_buffer = self.momentum * momentum_buffer + grad
                state.set_state('momentum_buffer', momentum_buffer)
                grad = momentum_buffer

            # Update parameter
            update = self.config.lr * grad
            self._update_parameter(param, update)

        self._step_times.append(time.time() - start_time)


class OptimizedFractionalAdam(OptimizedBaseOptimizer):
    """Optimized Adam with fractional calculus integration"""

    def __init__(self,
                 lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 fractional_order: float = 0.5,
                 method: str = "RL",
                 use_fractional: bool = True,
                 backend: Optional[BackendType] = None,
                 **kwargs):

        config = OptimizerConfig(
            lr=lr,
            fractional_order=fractional_order,
            method=method,
            use_fractional=use_fractional,
            backend=backend,
            **kwargs
        )
        super().__init__(config)
        self.betas = betas
        self.eps = eps

    def step(self, params: List[Any], gradients: Optional[List[Any]] = None) -> None:
        """Optimized Adam step"""
        start_time = time.time()

        # Get gradients
        if gradients is None:
            gradients = []
            for param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    gradients.append(param.grad)
                else:
                    gradients.append(None)

        if len(params) != len(gradients):
            raise ValueError(
                "Number of parameters must match number of gradients")

        for param, grad in zip(params, gradients):
            if grad is None:
                continue

            # Apply fractional derivative
            if self.config.use_fractional:
                grad = self._apply_fractional_derivative(grad)

            # Get parameter state
            state = self._get_param_state(param)
            step_count = state.increment_step()

            # Get momentum and variance parameters
            beta1, beta2 = self.betas
            exp_avg = state.get_state(
                'exp_avg',
                lambda: self.tensor_ops.zeros_like(param)
            )
            exp_avg_sq = state.get_state(
                'exp_avg_sq',
                lambda: self.tensor_ops.zeros_like(param)
            )

            # Update momentum and variance
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

            # Store updated state
            state.set_state('exp_avg', exp_avg)
            state.set_state('exp_avg_sq', exp_avg_sq)

            # Compute bias correction
            bias_correction1 = 1 - beta1 ** step_count
            bias_correction2 = 1 - beta2 ** step_count

            # Update parameter
            step_size = self.config.lr / bias_correction1
            sqrt_exp_avg_sq = self.tensor_ops.sqrt(exp_avg_sq)
            update = step_size * exp_avg / (sqrt_exp_avg_sq + self.eps)

            self._update_parameter(param, update)

        self._step_times.append(time.time() - start_time)


class OptimizedFractionalRMSprop(OptimizedBaseOptimizer):
    """Optimized RMSprop with fractional calculus integration"""

    def __init__(self,
                 lr: float = 0.001,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 fractional_order: float = 0.5,
                 method: str = "RL",
                 use_fractional: bool = True,
                 backend: Optional[BackendType] = None,
                 **kwargs):

        config = OptimizerConfig(
            lr=lr,
            fractional_order=fractional_order,
            method=method,
            use_fractional=use_fractional,
            backend=backend,
            **kwargs
        )
        super().__init__(config)
        self.alpha = alpha
        self.eps = eps

    def step(self, params: List[Any], gradients: Optional[List[Any]] = None) -> None:
        """Optimized RMSprop step"""
        start_time = time.time()

        # Get gradients
        if gradients is None:
            gradients = []
            for param in params:
                if hasattr(param, 'grad') and param.grad is not None:
                    gradients.append(param.grad)
                else:
                    gradients.append(None)

        if len(params) != len(gradients):
            raise ValueError(
                "Number of parameters must match number of gradients")

        for param, grad in zip(params, gradients):
            if grad is None:
                continue

            # Apply fractional derivative
            if self.config.use_fractional:
                grad = self._apply_fractional_derivative(grad)

            # Get parameter state
            state = self._get_param_state(param)
            square_avg = state.get_state(
                'square_avg',
                lambda: self.tensor_ops.zeros_like(param)
            )

            # Update square average
            square_avg = self.alpha * square_avg + \
                (1 - self.alpha) * (grad * grad)
            state.set_state('square_avg', square_avg)

            # Update parameter
            sqrt_square_avg = self.tensor_ops.sqrt(square_avg)
            update = self.config.lr * grad / (sqrt_square_avg + self.eps)

            self._update_parameter(param, update)

        self._step_times.append(time.time() - start_time)


# Convenience aliases for backward compatibility
OptimizedFractionalOptimizer = OptimizedBaseOptimizer

# Factory functions for easy creation


def create_optimized_sgd(lr: float = 0.001, momentum: float = 0.0, **kwargs) -> OptimizedFractionalSGD:
    """Create optimized fractional SGD optimizer"""
    return OptimizedFractionalSGD(lr=lr, momentum=momentum, **kwargs)


def create_optimized_adam(lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), **kwargs) -> OptimizedFractionalAdam:
    """Create optimized fractional Adam optimizer"""
    return OptimizedFractionalAdam(lr=lr, betas=betas, **kwargs)


def create_optimized_rmsprop(lr: float = 0.001, alpha: float = 0.99, **kwargs) -> OptimizedFractionalRMSprop:
    """Create optimized fractional RMSprop optimizer"""
    return OptimizedFractionalRMSprop(lr=lr, alpha=alpha, **kwargs)

# Performance comparison utilities


def compare_optimizer_performance(optimizers: List[OptimizedBaseOptimizer]) -> Dict[str, Any]:
    """Compare performance of multiple optimizers"""
    results = {}
    for opt in optimizers:
        stats = opt.get_performance_stats()
        results[opt.__class__.__name__] = stats
    return results


def benchmark_optimizer_performance(optimizer_class, model, criterion, data_loader, num_epochs: int = 10) -> Dict[str, Any]:
    """Benchmark optimizer performance"""
    optimizer = optimizer_class()
    params = list(model.parameters())

    times = []
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            start_time = time.time()
            optimizer.step(params)
            times.append(time.time() - start_time)

            optimizer.zero_grad(params)

            if batch_idx >= 5:  # Limit batches for benchmarking
                break

    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times
    }
