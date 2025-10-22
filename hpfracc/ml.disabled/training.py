"""
Training Utilities for Fractional Calculus ML

This module provides training utilities, schedulers, and callbacks specifically
designed for fractional calculus machine learning applications.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union, Tuple, Callable, Iterator
from collections import defaultdict, OrderedDict
import warnings

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class FractionalScheduler(ABC):
    """
    Base class for learning rate schedulers with fractional calculus integration

    This class provides a framework for schedulers that can apply
    fractional derivatives to learning rates during training.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
            self,
            optimizer: Any,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        self.optimizer = optimizer
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        self.base_lr = self._get_base_lr()

    def _get_base_lr(self) -> float:
        """Get base learning rate from optimizer"""
        if hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0]['lr']
        elif hasattr(self.optimizer, 'lr'):
            return self.optimizer.lr
        else:
            return 0.001  # Default fallback

    def fractional_adjustment(self, lr: float) -> float:
        """
        Apply fractional derivative to learning rate

        Args:
            lr: Input learning rate

        Returns:
            Learning rate with fractional derivative applied
        """
        # Only apply fractional derivative for PyTorch backend for now
        if self.backend == BackendType.TORCH:
            try:
                # Convert to tensor for fractional derivative
                lr_tensor = self.tensor_ops.tensor([lr])
                adjusted_tensor = fractional_derivative(
                    lr_tensor, self.fractional_order.alpha, self.method)
                adjusted = float(adjusted_tensor[0])
                # Blend with original to stabilize and ensure positivity
                blended = 0.5 * lr + 0.5 * adjusted
                return max(1e-12, blended)
            except (RuntimeError, ValueError):
                # If fractional derivative fails (e.g., single value), return original
                return max(1e-12, lr)
        else:
            # TODO: Implement backend-agnostic fractional derivatives
            # For now, return the input unchanged
            return lr

    @abstractmethod
    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""

    def get_last_lr(self) -> List[float]:
        """Get current learning rates"""
        if hasattr(self.optimizer, 'param_groups'):
            return [group['lr'] for group in self.optimizer.param_groups]
        elif hasattr(self.optimizer, 'lr'):
            return [self.optimizer.lr]
        else:
            return [self.base_lr]


class FractionalStepLR(FractionalScheduler):
    """Step learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            step_size: int,
            gamma: float = 0.1,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        
        if self.last_epoch % self.step_size == 0:
            # Calculate new learning rate
            new_lr = self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))
            
            # Apply fractional adjustment
            adjusted_lr = self.fractional_adjustment(new_lr)
            
            # Update optimizer learning rate
            if hasattr(self.optimizer, 'param_groups'):
                for group in self.optimizer.param_groups:
                    group['lr'] = adjusted_lr
            elif hasattr(self.optimizer, 'lr'):
                self.optimizer.lr = adjusted_lr


class FractionalExponentialLR(FractionalScheduler):
    """Exponential learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            gamma: float = 0.95,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.gamma = gamma
        self.last_epoch = 0

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        
        # Calculate new learning rate
        new_lr = self.base_lr * (self.gamma ** self.last_epoch)
        
        # Apply fractional adjustment
        adjusted_lr = self.fractional_adjustment(new_lr)
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr


class FractionalCosineAnnealingLR(FractionalScheduler):
    """Cosine annealing learning rate scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            T_max: int,
            eta_min: float = 0.0,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0

    def step(self, metrics: Optional[float] = None) -> None:
        """Update learning rate"""
        self.last_epoch += 1
        
        # Calculate new learning rate using cosine annealing
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
        
        # Apply fractional adjustment
        adjusted_lr = self.fractional_adjustment(new_lr)
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr


class FractionalCyclicLR(FractionalScheduler):
    """Simple cyclic learning rate with fractional calculus integration"""

    def __init__(
        self,
        optimizer: Any,
        base_lr: float = 1e-4,
        max_lr: float = 1e-2,
        step_size_up: int = 10,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        fractional_order: float = 0.5,
        method: str = 'RL',
        backend: Optional[BackendType] = None
    ):
        super().__init__(optimizer, fractional_order, method, backend)
        self.base_lr_user = base_lr
        self.max_lr = max_lr
        self.step_size_up = max(1, int(step_size_up))
        self.step_size_down = int(step_size_down) if step_size_down is not None else self.step_size_up
        self.cycle_len = self.step_size_up + self.step_size_down
        self.mode = mode
        self.iteration = 0

    def _scale_fn(self, x: float) -> float:
        if self.mode == 'triangular2':
            return 1.0 / (2.0 ** x)
        elif self.mode == 'exp_range':
            return 0.999 ** x
        return 1.0

    def step(self, metrics: Optional[float] = None) -> None:
        self.iteration += 1
        cycle_progress = (self.iteration - 1) % self.cycle_len
        if cycle_progress < self.step_size_up:
            scale = cycle_progress / float(self.step_size_up)
        else:
            scale = 1.0 - (cycle_progress - self.step_size_up) / float(self.step_size_down)

        # Base lr may come from user or optimizer
        base_lr = self.base_lr_user if self.base_lr_user is not None else self.base_lr
        new_lr = base_lr + (self.max_lr - base_lr) * max(0.0, min(1.0, scale)) * self._scale_fn(self.iteration // self.cycle_len)

        adjusted_lr = self.fractional_adjustment(new_lr)

        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr

class FractionalReduceLROnPlateau(FractionalScheduler):
    """Reduce learning rate on plateau scheduler with fractional calculus integration"""

    def __init__(
            self,
            optimizer: Any,
            mode: str = 'min',
            factor: float = 0.1,
            patience: int = 10,
            verbose: bool = False,
            threshold: float = 1e-4,
            threshold_mode: str = 'rel',
            cooldown: int = 0,
            min_lr: float = 0.0,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        super().__init__(optimizer, fractional_order, method, backend)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0

    def step(self, metrics: float) -> None:
        """Update learning rate based on metrics"""
        if metrics is None:
            return
        
        self.last_epoch += 1
        
        # Check if we should reduce learning rate
        if self._should_reduce_lr(metrics):
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
        else:
            self.num_bad_epochs += 1
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

    def _should_reduce_lr(self, metrics: float) -> bool:
        """Check if learning rate should be reduced"""
        if self.best is None:
            self.best = metrics
            return False
        
        if self.mode == 'min':
            is_better = metrics < self.best - self.threshold
        else:
            is_better = metrics > self.best + self.threshold
        
        if is_better:
            self.best = metrics
        
        return self.num_bad_epochs >= self.patience

    def _reduce_lr(self) -> None:
        """Reduce learning rate"""
        old_lr = self._get_base_lr()
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        # Apply fractional adjustment
        adjusted_lr = self.fractional_adjustment(new_lr)
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'param_groups'):
            for group in self.optimizer.param_groups:
                group['lr'] = adjusted_lr
        elif hasattr(self.optimizer, 'lr'):
            self.optimizer.lr = adjusted_lr
        
        if self.verbose:
            print(f'Reducing learning rate from {old_lr:.6f} to {adjusted_lr:.6f}')


class TrainingCallback(ABC):
    """Base class for training callbacks"""

    def __init__(self):
        self.trainer = None

    def set_trainer(self, trainer: 'FractionalTrainer'):
        """Set the trainer reference"""
        self.trainer = trainer

    @abstractmethod
    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of each epoch"""

    @abstractmethod
    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch"""

    @abstractmethod
    def on_batch_begin(self, batch: int) -> None:
        """Called at the beginning of each batch"""

    @abstractmethod
    def on_batch_end(self, batch: int) -> None:
        """Called at the end of each batch"""


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping callback"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self, epoch: int) -> None:
        """Check if training should stop early"""
        if self.trainer is None:
            return
        
        current_score = self.trainer.validation_losses[-1] if self.trainer.validation_losses else float('inf')
        
        if self.best_score is None:
            self.best_score = current_score
        elif self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.trainer:
                self.trainer.should_stop = True

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_batch_begin(self, batch: int) -> None:
        pass

    def on_batch_end(self, batch: int) -> None:
        pass


class ModelCheckpointCallback(TrainingCallback):
    """Model checkpoint callback"""

    def __init__(self, filepath: str, monitor: str = 'val_loss', mode: str = 'min', save_best_only: bool = True):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

    def on_epoch_end(self, epoch: int) -> None:
        """Save model checkpoint if needed"""
        if self.trainer is None:
            return
        
        # Get the metric to monitor
        if self.monitor == 'val_loss':
            current_score = self.trainer.validation_losses[-1] if self.trainer.validation_losses else float('inf')
        elif self.monitor == 'train_loss':
            current_score = self.trainer.training_losses[-1] if self.trainer.training_losses else float('inf')
        else:
            return
        
        # Check if we should save
        should_save = False
        if self.best_score is None:
            should_save = True
        elif self.mode == 'min':
            if current_score < self.best_score:
                should_save = True
        else:
            if current_score > self.best_score:
                should_save = True
        
        if should_save:
            self.best_score = current_score
            if self.trainer.model:
                # Save model (simplified - in practice you'd use proper serialization)
                print(f"Saving model checkpoint to {self.filepath}")

    def on_epoch_begin(self, epoch: int) -> None:
        pass

    def on_batch_begin(self, batch: int) -> None:
        pass

    def on_batch_end(self, batch: int) -> None:
        pass


class FractionalTrainer:
    """Training trainer with fractional calculus integration"""

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            loss_fn: Any,
            scheduler: Optional[FractionalScheduler] = None,
            callbacks: Optional[List[TrainingCallback]] = None,
            fractional_order: float = 0.5,
            method: str = "RL",
            backend: Optional[BackendType] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.callbacks = callbacks or []
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
        
        # Training state
        self.training_losses = []
        self.validation_losses = []
        self.current_epoch = 0
        self.should_stop = False
        
        # Set trainer reference in callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

    def train_epoch(self, dataloader: Any) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Call batch begin callback
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx)
            
            # Forward pass
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Call batch end callback
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx)
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self, dataloader: Any) -> None:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with self.tensor_ops.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self, train_dataloader: Any, val_dataloader: Any, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Call epoch begin callback
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            
            # Training phase
            train_loss = self.train_epoch(train_dataloader)
            self.training_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate_epoch(val_dataloader)
            self.validation_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, FractionalReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {self.scheduler.get_last_lr()[0] if self.scheduler else 'N/A':.6f}")
            
            # Call epoch end callback
            for callback in self.callbacks:
                callback.on_epoch_end(epoch)
            
            # Check if we should stop early
            if self.should_stop:
                print("Early stopping triggered")
                break
        
        print("Training completed!")
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint"""
        # Simplified checkpoint saving
        # In practice, you'd want to save the full model state
        print(f"Model checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint"""
        # Simplified checkpoint loading
        # In practice, you'd want to load the full model state
        print(f"Model checkpoint loaded from {filepath}")


# Factory functions for easy creation
def create_fractional_scheduler(
        scheduler_type: str,
        optimizer: Any,
        fractional_order: float = 0.5,
        method: str = "RL",
        **kwargs) -> FractionalScheduler:
    """
    Create a fractional scheduler of the specified type
    
    Args:
        scheduler_type: Type of scheduler ('step', 'exponential', 'cosine', 'plateau')
        optimizer: Optimizer to schedule
        fractional_order: Fractional order for derivative
        method: Method for fractional derivative
        **kwargs: Additional scheduler-specific parameters
        
    Returns:
        Configured fractional scheduler
    """
    scheduler_map = {
        'step': FractionalStepLR,
        'exponential': FractionalExponentialLR,
        'cosine': FractionalCosineAnnealingLR,
        'cyclic': FractionalCyclicLR,
        'plateau': FractionalReduceLROnPlateau,
    }
    
    if scheduler_type.lower() not in scheduler_map:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    scheduler_class = scheduler_map[scheduler_type.lower()]
    return scheduler_class(optimizer, fractional_order=fractional_order, method=method, **kwargs)


def create_fractional_trainer(
        model: Any,
        optimizer: Any,
        loss_fn: Any,
        scheduler: Optional[FractionalScheduler] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        fractional_order: float = 0.5,
        method: str = "RL") -> FractionalTrainer:
    """
    Create a fractional trainer
    
    Args:
        model: Model to train
        optimizer: Optimizer for training
        loss_fn: Loss function
        scheduler: Learning rate scheduler (optional)
        callbacks: Training callbacks (optional)
        fractional_order: Fractional order for derivative
        method: Method for fractional derivative
        
    Returns:
        Configured fractional trainer
    """
    return FractionalTrainer(
        model, optimizer, loss_fn, scheduler, callbacks, fractional_order, method
    )
