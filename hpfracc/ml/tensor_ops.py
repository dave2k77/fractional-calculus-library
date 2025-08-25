"""
Unified Tensor Operations for Multi-Backend Support

This module provides consistent tensor operations across PyTorch, JAX, and NUMBA,
enabling seamless switching between frameworks while maintaining the same API.
"""

from typing import Optional, Union, Any, List, Tuple, Callable
import warnings

from .backends import get_backend_manager, BackendType


class TensorOps:
    """
    Unified tensor operations across different backends
    
    This class provides a consistent interface for common tensor operations
    regardless of the underlying backend (PyTorch, JAX, or NUMBA).
    """
    
    def __init__(self, backend: Optional[BackendType] = None):
        self.backend_manager = get_backend_manager()
        self.backend = backend or self.backend_manager.active_backend
        self.tensor_lib = self.backend_manager.get_tensor_lib()
    
    def create_tensor(self, data: Any, **kwargs) -> Any:
        """Create a tensor in the current backend"""
        # Filter out backend-specific arguments
        if self.backend == BackendType.TORCH:
            # PyTorch supports requires_grad
            return self.backend_manager.create_tensor(data, **kwargs)
        elif self.backend == BackendType.JAX:
            # JAX doesn't support requires_grad, filter it out
            jax_kwargs = {k: v for k, v in kwargs.items() if k != 'requires_grad'}
            return self.backend_manager.create_tensor(data, **jax_kwargs)
        elif self.backend == BackendType.NUMBA:
            # NUMBA doesn't support requires_grad, filter it out
            numba_kwargs = {k: v for k, v in kwargs.items() if k != 'requires_grad'}
            return self.backend_manager.create_tensor(data, **numba_kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def zeros(self, shape: Tuple[int, ...], **kwargs) -> Any:
        """Create a tensor of zeros"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.zeros(shape, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.zeros(shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.zeros(shape, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def ones(self, shape: Tuple[int, ...], **kwargs) -> Any:
        """Create a tensor of ones"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.ones(shape, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.ones(shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.ones(shape, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def eye(self, n: int, **kwargs) -> Any:
        """Create an identity matrix"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.eye(n, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.eye(n, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.eye(n, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def arange(self, start: int, end: int, step: int = 1, **kwargs) -> Any:
        """Create a range of values"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.arange(start, end, step, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def linspace(self, start: float, end: float, num: int, **kwargs) -> Any:
        """Create linearly spaced values"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.linspace(start, end, num, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def zeros_like(self, tensor: Any, **kwargs) -> Any:
        """Create a tensor of zeros with the same shape as the input tensor"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.zeros_like(tensor, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.zeros_like(tensor, **kwargs)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            if hasattr(tensor, 'shape'):
                return np.zeros_like(tensor, **kwargs)
            else:
                # Fallback for scalars or objects without shape
                return np.zeros(1, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def sqrt(self, tensor: Any) -> Any:
        """Compute the square root of a tensor"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.sqrt(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.sqrt(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.sqrt(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        """Stack tensors along a dimension"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.stack(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.stack(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.stack(tensors, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        """Concatenate tensors along a dimension"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.cat(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.concatenate(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.concatenate(tensors, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        """Reshape a tensor"""
        if self.backend == BackendType.TORCH:
            return tensor.reshape(shape)
        elif self.backend == BackendType.JAX:
            return tensor.reshape(shape)
        elif self.backend == BackendType.NUMBA:
            return tensor.reshape(shape)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def repeat(self, tensor: Any, repeats: Union[int, Tuple[int, ...]], dim: int = 0) -> Any:
        """Repeat a tensor along specified dimensions"""
        if self.backend == BackendType.TORCH:
            return tensor.repeat(repeats, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.repeat(tensor, repeats, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.repeat(tensor, repeats, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def clip(self, tensor: Any, min_val: float, max_val: float) -> Any:
        """Clip tensor values to a specified range"""
        if self.backend == BackendType.TORCH:
            return tensor.clamp(min_val, max_val)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.clip(tensor, min_val, max_val)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.clip(tensor, min_val, max_val)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def unsqueeze(self, tensor: Any, dim: int) -> Any:
        """Add a dimension to a tensor at the specified position"""
        if self.backend == BackendType.TORCH:
            return tensor.unsqueeze(dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.expand_dims(tensor, dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.expand_dims(tensor, dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def expand(self, tensor: Any, *sizes: int) -> Any:
        """Expand a tensor to new dimensions"""
        if self.backend == BackendType.TORCH:
            return tensor.expand(*sizes)
        elif self.backend == BackendType.JAX:
            # JAX doesn't have expand, use broadcast_to instead
            return self.tensor_lib.broadcast_to(tensor, sizes)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            # NUMBA doesn't have expand, use broadcast_to instead
            return np.broadcast_to(tensor, sizes)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def gather(self, tensor: Any, dim: int, index: Any) -> Any:
        """Gather values from a tensor using indices"""
        if self.backend == BackendType.TORCH:
            return tensor.gather(dim, index)
        elif self.backend == BackendType.JAX:
            # JAX doesn't have gather, use advanced indexing instead
            return tensor[index]
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            # NUMBA doesn't have gather, use advanced indexing instead
            return tensor[index]
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def squeeze(self, tensor: Any, dim: Optional[int] = None) -> Any:
        """Remove dimensions of size 1 from a tensor"""
        if self.backend == BackendType.TORCH:
            return tensor.squeeze(dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.squeeze(tensor, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.squeeze(tensor, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def transpose(self, tensor: Any, dims: Tuple[int, ...]) -> Any:
        """Transpose a tensor"""
        if self.backend == BackendType.TORCH:
            return tensor.permute(dims)
        elif self.backend == BackendType.JAX:
            return tensor.transpose(dims)
        elif self.backend == BackendType.NUMBA:
            return tensor.transpose(dims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.matmul(a, b)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.matmul(a, b)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.matmul(a, b)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def einsum(self, equation: str, *operands) -> Any:
        """Einstein summation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.einsum(equation, *operands)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.einsum(equation, *operands)
        elif self.backend == BackendType.NUMBA:
            # NUMBA doesn't have einsum, fall back to basic operations
            warnings.warn("NUMBA backend doesn't support einsum, using basic operations")
            return self._numba_einsum_fallback(equation, *operands)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def _numba_einsum_fallback(self, equation: str, *operands) -> Any:
        """Fallback einsum implementation for NUMBA"""
        # This is a simplified fallback - in practice, you might want to
        # implement specific einsum patterns or use a different approach
        import numpy as np
        if equation == "ij,jk->ik":
            return self.matmul(operands[0], operands[1])
        elif equation == "i,i->":
            return np.sum(operands[0] * operands[1])
        else:
            raise NotImplementedError(f"NUMBA backend doesn't support einsum pattern: {equation}")
    
    def sum(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        """Sum tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.sum(dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.sum(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.sum(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def mean(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        """Mean of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.mean(dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.mean(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.mean(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def max(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        """Maximum of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.max(dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.max(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.max(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def min(self, tensor: Any, dim: Optional[int] = None, keepdims: bool = False) -> Any:
        """Minimum of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.min(dim=dim, keepdim=keepdims)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.min(tensor, axis=dim, keepdims=keepdims)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.min(tensor, axis=dim, keepdims=keepdims)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def norm(self, tensor: Any, p: float = 2, dim: Optional[int] = None) -> Any:
        """Compute norm of tensor"""
        if self.backend == BackendType.TORCH:
            return tensor.norm(p=p, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.linalg.norm(tensor, ord=p, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.linalg.norm(tensor, ord=p, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def softmax(self, tensor: Any, dim: int = -1) -> Any:
        """Apply softmax activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.softmax(tensor, dim=dim)
        elif self.backend == BackendType.JAX:
            import jax.nn
            return jax.nn.softmax(tensor, axis=dim)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.exp(tensor) / np.sum(np.exp(tensor), axis=dim, keepdims=True)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def relu(self, tensor: Any) -> Any:
        """Apply ReLU activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.relu(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.maximum(tensor, 0)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.maximum(tensor, 0)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def sigmoid(self, tensor: Any) -> Any:
        """Apply sigmoid activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.sigmoid(tensor)
        elif self.backend == BackendType.JAX:
            return 1 / (1 + self.tensor_lib.exp(-tensor))
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return 1 / (1 + np.exp(-tensor))
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def tanh(self, tensor: Any) -> Any:
        """Apply tanh activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.tanh(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.tanh(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.tanh(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def sqrt(self, tensor: Any) -> Any:
        """Compute square root"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.sqrt(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.sqrt(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.sqrt(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def log(self, tensor: Any) -> Any:
        """Compute natural logarithm"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.log(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.log(tensor)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            return np.log(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def dropout(self, tensor: Any, p: float = 0.5, training: bool = True) -> Any:
        """Apply dropout"""
        if not training or p == 0:
            return tensor
        
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.dropout(tensor, p=p, train=training)
        elif self.backend == BackendType.JAX:
            # JAX doesn't have built-in dropout, implement manually
            import jax.random as random
            key = random.PRNGKey(0)  # You might want to pass a proper key
            mask = random.bernoulli(key, 1 - p, tensor.shape)
            return tensor * mask / (1 - p)
        elif self.backend == BackendType.NUMBA:
            # NUMBA doesn't have built-in dropout, implement manually
            import numpy as np
            mask = np.random.random(tensor.shape) > p
            return tensor * mask / (1 - p)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")


# Global tensor operations instance
_tensor_ops: Optional[TensorOps] = None


def get_tensor_ops(backend: Optional[BackendType] = None) -> TensorOps:
    """Get the global tensor operations instance"""
    global _tensor_ops
    if _tensor_ops is None or (backend is not None and _tensor_ops.backend != backend):
        _tensor_ops = TensorOps(backend)
    return _tensor_ops


def create_tensor(data: Any, **kwargs) -> Any:
    """Create a tensor using the current backend"""
    return get_tensor_ops().create_tensor(data, **kwargs)


def switch_backend(backend: BackendType) -> None:
    """Switch to a different backend and update tensor operations"""
    from .backends import switch_backend as switch_backend_manager
    if switch_backend_manager(backend):
        global _tensor_ops
        _tensor_ops = None  # Reset tensor ops for new backend
