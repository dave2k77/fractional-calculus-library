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
            return self.tensor_lib.zeros(shape, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def ones(self, shape: Tuple[int, ...], **kwargs) -> Any:
        """Create a tensor of ones"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.ones(shape, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.ones(shape, **kwargs)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.ones(shape, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def eye(self, n: int, **kwargs) -> Any:
        """Create an identity matrix"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.eye(n, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.eye(n, **kwargs)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.eye(n, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def arange(self, start: int, end: int, step: int = 1, **kwargs) -> Any:
        """Create a range of values"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.arange(start, end, step, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def linspace(self, start: float, end: float, num: int, **kwargs) -> Any:
        """Create linearly spaced values"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.linspace(start, end, num, **kwargs)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        """Stack tensors along a dimension"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.stack(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.stack(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.stack(tensors, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        """Concatenate tensors along a dimension"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.cat(tensors, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.concatenate(tensors, axis=dim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.concatenate(tensors, axis=dim)
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
            return self.tensor_lib.matmul(a, b)
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
        if equation == "ij,jk->ik":
            return self.matmul(operands[0], operands[1])
        elif equation == "i,i->":
            return self.tensor_lib.sum(operands[0] * operands[1])
        else:
            raise NotImplementedError(f"NUMBA backend doesn't support einsum pattern: {equation}")
    
    def sum(self, tensor: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """Sum tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.sum(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.sum(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.sum(tensor, axis=dim, keepdims=keepdim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def mean(self, tensor: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """Mean of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.mean(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.mean(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.mean(tensor, axis=dim, keepdims=keepdim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def max(self, tensor: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """Maximum of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.max(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.max(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.max(tensor, axis=dim, keepdims=keepdim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def min(self, tensor: Any, dim: Optional[int] = None, keepdim: bool = False) -> Any:
        """Minimum of tensor elements"""
        if self.backend == BackendType.TORCH:
            return tensor.min(dim=dim, keepdim=keepdim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.min(tensor, axis=dim, keepdims=keepdim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.min(tensor, axis=dim, keepdims=keepdim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def norm(self, tensor: Any, p: float = 2, dim: Optional[int] = None) -> Any:
        """Compute norm of tensor"""
        if self.backend == BackendType.TORCH:
            return tensor.norm(p=p, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.linalg.norm(tensor, ord=p, axis=dim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.linalg.norm(tensor, ord=p, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def softmax(self, tensor: Any, dim: int = -1) -> Any:
        """Apply softmax activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.softmax(tensor, dim=dim)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.softmax(tensor, axis=dim)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.softmax(tensor, axis=dim)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def relu(self, tensor: Any) -> Any:
        """Apply ReLU activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.relu(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.maximum(tensor, 0)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.maximum(tensor, 0)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def sigmoid(self, tensor: Any) -> Any:
        """Apply sigmoid activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.sigmoid(tensor)
        elif self.backend == BackendType.JAX:
            return 1 / (1 + self.tensor_lib.exp(-tensor))
        elif self.backend == BackendType.NUMBA:
            return 1 / (1 + self.tensor_lib.exp(-tensor))
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def tanh(self, tensor: Any) -> Any:
        """Apply tanh activation"""
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.tanh(tensor)
        elif self.backend == BackendType.JAX:
            return self.tensor_lib.tanh(tensor)
        elif self.backend == BackendType.NUMBA:
            return self.tensor_lib.tanh(tensor)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")
    
    def dropout(self, tensor: Any, p: float = 0.5, training: bool = True) -> Any:
        """Apply dropout"""
        if not training or p == 0:
            return tensor
        
        if self.backend == BackendType.TORCH:
            return self.tensor_lib.dropout(tensor, p=p, training=training)
        elif self.backend == BackendType.JAX:
            # JAX doesn't have built-in dropout, implement manually
            key = self.tensor_lib.random.PRNGKey(0)  # You might want to pass a proper key
            mask = self.tensor_lib.random.bernoulli(key, 1 - p, tensor.shape)
            return tensor * mask / (1 - p)
        elif self.backend == BackendType.NUMBA:
            # NUMBA doesn't have built-in dropout, implement manually
            mask = self.tensor_lib.random.random(tensor.shape) > p
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
