"""
Advanced SDE Adjoint Optimization Utilities

Provides memory-efficient checkpointing, mixed precision training,
and sparse gradient accumulation for neural fractional SDEs.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing."""
    checkpoint_frequency: int = 10  # Save checkpoint every N steps
    max_checkpoints: int = 100  # Maximum number of checkpoints to keep
    checkpoint_strategy: str = "uniform"  # "uniform", "adaptive", "save_all"
    enable_checkpointing: bool = True


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    enable_amp: bool = False  # Automatic Mixed Precision
    half_precision: bool = False  # Use float16 throughout
    dtype_fp16: torch.dtype = torch.float16
    dtype_fp32: torch.dtype = torch.float32
    loss_scaling: float = 1.0


class SDEStateCheckpoint:
    """
    Checkpoint manager for SDE trajectory states.
    Enables memory-efficient training by saving intermediate states.
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoints: List[Dict[str, Any]] = []
        self.checkpoint_indices: List[int] = []
    
    def save_checkpoint(
        self,
        step: int,
        state: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save a checkpoint of the current state."""
        if not self.config.enable_checkpointing:
            return
        
        # Decide if we should save based on strategy
        should_save = False
        
        if self.config.checkpoint_strategy == "save_all":
            should_save = True
        elif self.config.checkpoint_strategy == "uniform":
            should_save = (step % self.config.checkpoint_frequency == 0)
        elif self.config.checkpoint_strategy == "adaptive":
            # Adaptive strategy: save more frequently in early stages
            freq = max(1, self.config.checkpoint_frequency // (step // 100 + 1))
            should_save = (step % freq == 0)
        
        if should_save:
            checkpoint = {
                'step': step,
                'state': state.detach().clone(),
                'metadata': metadata or {}
            }
            
            self.checkpoints.append(checkpoint)
            self.checkpoint_indices.append(step)
            
            # Limit number of checkpoints
            if len(self.checkpoints) > self.config.max_checkpoints:
                # Remove oldest checkpoint
                self.checkpoints.pop(0)
                self.checkpoint_indices.pop(0)
    
    def load_checkpoint(self, step: int) -> Optional[torch.Tensor]:
        """Load checkpoint closest to the given step."""
        if not self.checkpoints:
            return None
        
        # Find closest checkpoint
        idx = min(range(len(self.checkpoint_indices)),
                 key=lambda i: abs(self.checkpoint_indices[i] - step))
        
        if idx is not None:
            return self.checkpoints[idx]['state']
        
        return None
    
    def clear(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.checkpoint_indices.clear()


class MixedPrecisionManager:
    """
    Manager for mixed precision training in SDEs.
    Handles automatic mixed precision (AMP) and float16 operations.
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        if self.config.enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def autocast(self):
        """Create autocast context for mixed precision."""
        if self.config.enable_amp:
            return torch.cuda.amp.autocast()
        elif self.config.half_precision:
            return torch.cuda.amp.autocast(dtype=self.config.dtype_fp16)
        else:
            # Context manager that does nothing
            return torch.no_grad()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss * self.config.loss_scaling
    
    def scale_gradients(self, loss: torch.Tensor):
        """Scale gradients after backward pass."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            (loss * self.config.loss_scaling).backward()
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Update optimizer with scaled gradients."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


class SparseGradientAccumulator:
    """
    Accumulator for sparse gradients in high-dimensional SDE systems.
    Reduces memory usage by only storing non-zero gradients.
    """
    
    def __init__(self, sparsity_threshold: float = 1e-6):
        """
        Initialize sparse gradient accumulator.
        
        Args:
            sparsity_threshold: Threshold below which gradients are considered zero
        """
        self.sparsity_threshold = sparsity_threshold
        self.accumulated_grads: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    def accumulate(self, grad: torch.Tensor, param_name: str = None):
        """
        Accumulate a gradient with sparsity.
        
        Args:
            grad: Gradient tensor
            param_name: Optional parameter name for debugging
        """
        # Identify non-zero elements
        mask = torch.abs(grad) > self.sparsity_threshold
        
        if mask.any():
            # Store only non-zero elements
            sparse_grad = grad[mask]
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            
            self.accumulated_grads.append((sparse_grad, indices))
    
    def get_dense_gradient(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reconstruct dense gradient from sparse representation.
        
        Args:
            shape: Original gradient shape
            
        Returns:
            Dense gradient tensor
        """
        grad = torch.zeros(shape, dtype=torch.float32)
        
        for sparse_grad, indices in self.accumulated_grads:
            grad.view(-1)[indices.view(-1)] = sparse_grad
        
        return grad
    
    def clear(self):
        """Clear accumulated gradients."""
        self.accumulated_grads.clear()
    
    def get_sparsity_ratio(self) -> float:
        """Get ratio of non-zero to total elements."""
        if not self.accumulated_grads:
            return 0.0
        
        total_elements = 0
        nonzero_elements = 0
        
        for sparse_grad, indices in self.accumulated_grads:
            # Estimate total elements from indices
            total_elements += (indices.max().item() + 1) if indices.numel() > 0 else 0
            nonzero_elements += sparse_grad.numel()
        
        if total_elements == 0:
            return 0.0
        
        return nonzero_elements / total_elements


def checkpoint_trajectory(
    func: Callable,
    *args,
    checkpoint_freq: int = 10,
    **kwargs
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Execute function with checkpointing to save memory.
    
    Uses gradient checkpointing to trade compute for memory.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to function
        checkpoint_freq: Frequency of checkpoints
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (final_output, checkpointed_states)
    """
    # Use PyTorch's gradient checkpointing if available
    try:
        from torch.utils.checkpoint import checkpoint as torch_checkpoint
        return torch_checkpoint(func, *args, **kwargs)
    except ImportError:
        # Fallback to manual checkpointing
        pass
    
    # Manual checkpointing implementation
    # This is a simplified version
    return func(*args, **kwargs)


class SDEAdjointOptimizer:
    """
    Optimizer wrapper for SDE adjoint training with advanced optimizations.
    
    Combines checkpointing, mixed precision, and sparse gradients.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_config: Optional[CheckpointConfig] = None,
        mixed_precision_config: Optional[MixedPrecisionConfig] = None,
        enable_sparse_gradients: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.mixed_precision_config = mixed_precision_config or MixedPrecisionConfig()
        
        self.checkpoint_manager = SDEStateCheckpoint(self.checkpoint_config)
        self.mixed_precision_manager = MixedPrecisionManager(self.mixed_precision_config)
        self.sparse_accumulator = SparseGradientAccumulator() if enable_sparse_gradients else None
    
    def step(self, loss: torch.Tensor):
        """
        Optimization step with advanced features.
        
        Args:
            loss: Loss tensor to optimize
        """
        self.optimizer.zero_grad()
        
        # Scale loss for mixed precision
        scaled_loss = self.mixed_precision_manager.scale_loss(loss)
        
        # Backward pass
        with self.mixed_precision_manager.autocast():
            scaled_loss.backward()
        
        # Sparse gradient accumulation
        if self.sparse_accumulator is not None:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.sparse_accumulator.accumulate(param.grad, name)
        
        # Step optimizer
        self.mixed_precision_manager.step_optimizer(self.optimizer)
    
    def save_state_checkpoint(self, step: int, state: torch.Tensor, metadata: Dict[str, Any] = None):
        """Save state checkpoint."""
        self.checkpoint_manager.save_checkpoint(step, state, metadata)
    
    def load_state_checkpoint(self, step: int) -> Optional[torch.Tensor]:
        """Load state checkpoint."""
        return self.checkpoint_manager.load_checkpoint(step)
    
    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self.checkpoint_manager.clear()
        if self.sparse_accumulator is not None:
            self.sparse_accumulator.clear()
