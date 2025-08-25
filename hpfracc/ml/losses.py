"""
Loss Functions with Fractional Calculus Integration

This module provides loss functions that incorporate fractional derivatives,
enabling enhanced training dynamics and potentially better convergence.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


class FractionalLossFunction(ABC):
    """
    Base class for loss functions with fractional calculus integration
    
    This class provides a framework for loss functions that can apply
    fractional derivatives to predictions before computing the loss.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", backend: Optional[BackendType] = None):
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
        
        # Set backend
        self.backend = backend or get_backend_manager().active_backend
        self.tensor_ops = get_tensor_ops(self.backend)
    
    def fractional_forward(self, x: Any) -> Any:
        """
        Apply fractional derivative to input tensor
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with fractional derivative applied
        """
        # Only apply fractional derivative for PyTorch backend for now
        if self.backend == BackendType.TORCH:
            return fractional_derivative(x, self.fractional_order.alpha, self.method)
        else:
            # TODO: Implement backend-agnostic fractional derivatives
            # For now, return the input unchanged
            return x
    
    @abstractmethod
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        """
        Compute the base loss function
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Loss value
        """
        pass
    
    def forward(
        self,
        predictions: Any,
        targets: Any,
        use_fractional: bool = True
    ) -> Any:
        """
        Forward pass for loss computation
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            use_fractional: Whether to apply fractional derivatives
            
        Returns:
            Loss value
        """
        if use_fractional:
            # Apply fractional derivative to predictions
            predictions = self.fractional_forward(predictions)
        
        # Compute the base loss
        return self.compute_loss(predictions, targets)
    
    def __call__(self, predictions: Any, targets: Any, use_fractional: bool = True) -> Any:
        """Make the loss function callable"""
        return self.forward(predictions, targets, use_fractional)


class FractionalMSELoss(FractionalLossFunction):
    """Mean Squared Error loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch
            import torch.nn.functional as F
            # Ensure inputs are PyTorch tensors
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions, dtype=torch.float32)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, dtype=torch.float32)
            return F.mse_loss(predictions, targets, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            squared_diff = (predictions - targets) ** 2
            if self.reduction == "mean":
                return self.tensor_ops.mean(squared_diff)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(squared_diff)
            else:  # none
                return squared_diff


class FractionalCrossEntropyLoss(FractionalLossFunction):
    """Cross Entropy loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.cross_entropy(predictions, targets, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Apply softmax to predictions
            softmax_pred = self.tensor_ops.softmax(predictions, dim=-1)
            
            # Compute cross-entropy
            log_softmax = self.tensor_ops.log(softmax_pred + 1e-8)  # Add small epsilon for numerical stability
            
            # For one-hot targets
            if len(targets.shape) == 2:
                loss = -self.tensor_ops.sum(targets * log_softmax, dim=-1)
            else:
                # For class indices
                batch_size = predictions.shape[0]
                loss = -log_softmax[np.arange(batch_size), targets]
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalHuberLoss(FractionalLossFunction):
    """Huber loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", delta: float = 1.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.delta = delta
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.huber_loss(predictions, targets, delta=self.delta, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            diff = predictions - targets
            abs_diff = self.tensor_ops.abs(diff)
            
            # Huber loss computation
            quadratic = self.tensor_ops.minimum(abs_diff, self.delta)
            linear = abs_diff - quadratic
            loss = 0.5 * quadratic ** 2 + self.delta * linear
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalSmoothL1Loss(FractionalLossFunction):
    """Smooth L1 loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", beta: float = 1.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.beta = beta
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            diff = predictions - targets
            abs_diff = self.tensor_ops.abs(diff)
            
            # Smooth L1 loss computation
            quadratic = 0.5 * (abs_diff ** 2) / self.beta
            linear = abs_diff - 0.5 * self.beta
            loss = self.tensor_ops.where(abs_diff < self.beta, quadratic, linear)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalKLDivLoss(FractionalLossFunction):
    """KL Divergence loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.kl_div(predictions, targets, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # KL divergence: KL(targets || predictions)
            # targets should be log-probabilities, predictions should be probabilities
            loss = targets * (targets - predictions)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalBCELoss(FractionalLossFunction):
    """Binary Cross Entropy loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.binary_cross_entropy(predictions, targets, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Binary cross-entropy: -[targets * log(predictions) + (1 - targets) * log(1 - predictions)]
            epsilon = 1e-8  # Small epsilon for numerical stability
            predictions = self.tensor_ops.clip(predictions, epsilon, 1 - epsilon)
            
            loss = -targets * self.tensor_ops.log(predictions) - (1 - targets) * self.tensor_ops.log(1 - predictions)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalNLLLoss(FractionalLossFunction):
    """Negative Log Likelihood loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.nll_loss(predictions, targets, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # NLL loss: -log(predictions[targets])
            batch_size = predictions.shape[0]
            loss = -predictions[np.arange(batch_size), targets]
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalPoissonNLLLoss(FractionalLossFunction):
    """Poisson Negative Log Likelihood loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", log_input: bool = True, full: bool = False, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.log_input = log_input
        self.full = full
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.poisson_nll_loss(predictions, targets, log_input=self.log_input, full=self.full, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            if self.log_input:
                # predictions are log(input)
                loss = self.tensor_ops.exp(predictions) - targets * predictions
            else:
                # predictions are input
                loss = predictions - targets * self.tensor_ops.log(predictions + 1e-8)
            
            if self.full:
                # Add Stirling approximation term
                loss += targets * self.tensor_ops.log(targets + 1e-8) - targets + 0.5 * self.tensor_ops.log(2 * np.pi * targets + 1e-8)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalCosineEmbeddingLoss(FractionalLossFunction):
    """Cosine Embedding loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", margin: float = 0.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.margin = margin
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.cosine_embedding_loss(predictions, targets, margin=self.margin, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Cosine embedding loss
            cos_sim = self.tensor_ops.sum(predictions * targets, dim=-1) / (
                self.tensor_ops.norm(predictions, dim=-1) * self.tensor_ops.norm(targets, dim=-1) + 1e-8
            )
            
            loss = self.tensor_ops.where(
                targets == 1,
                1 - cos_sim,
                self.tensor_ops.maximum(0, cos_sim - self.margin)
            )
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalMarginRankingLoss(FractionalLossFunction):
    """Margin Ranking loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", margin: float = 0.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.margin = margin
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.margin_ranking_loss(predictions[0], predictions[1], targets, margin=self.margin, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Margin ranking loss: max(0, -targets * (predictions[0] - predictions[1]) + margin)
            x1, x2 = predictions[0], predictions[1]
            loss = self.tensor_ops.maximum(0, -targets * (x1 - x2) + self.margin)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalMultiMarginLoss(FractionalLossFunction):
    """Multi Margin loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", p: int = 1, margin: float = 1.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.p = p
        self.margin = margin
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.multi_margin_loss(predictions, targets, p=self.p, margin=self.margin, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Multi margin loss
            batch_size = predictions.shape[0]
            num_classes = predictions.shape[1]
            
            loss = self.tensor_ops.zeros(batch_size)
            
            for i in range(batch_size):
                target = targets[i]
                pred = predictions[i]
                
                # Compute margin loss for each sample
                target_pred = pred[target]
                margin_loss = self.tensor_ops.maximum(0, self.margin - target_pred + pred)
                margin_loss = self.tensor_ops.where(
                    np.arange(num_classes) == target,
                    0,
                    margin_loss
                )
                
                loss = loss.at[i].set(self.tensor_ops.sum(margin_loss ** self.p) ** (1 / self.p))
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalTripletMarginLoss(FractionalLossFunction):
    """Triplet Margin loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", margin: float = 1.0, p: float = 2.0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.margin = margin
        self.p = p
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.triplet_margin_loss(predictions[0], predictions[1], predictions[2], margin=self.margin, p=self.p, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Triplet margin loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)
            anchor, positive, negative = predictions[0], predictions[1], predictions[2]
            
            # Compute distances
            pos_dist = self.tensor_ops.norm(anchor - positive, p=self.p, dim=-1)
            neg_dist = self.tensor_ops.norm(anchor - negative, p=self.p, dim=-1)
            
            loss = self.tensor_ops.maximum(0, pos_dist - neg_dist + self.margin)
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalCTCLoss(FractionalLossFunction):
    """Connectionist Temporal Classification loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", blank: int = 0, reduction: str = "mean", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.blank = blank
        self.reduction = reduction
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.ctc_loss(predictions, targets, blank=self.blank, reduction=self.reduction)
        else:
            # JAX/NUMBA implementation
            # Simplified CTC loss (in practice, you'd want a more sophisticated implementation)
            # This is a placeholder implementation
            batch_size = predictions.shape[0]
            loss = self.tensor_ops.zeros(batch_size)
            
            # Simplified loss computation
            for i in range(batch_size):
                pred = predictions[i]
                target = targets[i]
                
                # Basic alignment-based loss (simplified)
                pred_probs = self.tensor_ops.softmax(pred, dim=-1)
                target_probs = self.tensor_ops.zeros_like(pred_probs)
                target_probs = target_probs.at[np.arange(len(target)), target].set(1.0)
                
                loss = loss.at[i].set(-self.tensor_ops.sum(target_probs * self.tensor_ops.log(pred_probs + 1e-8)))
            
            if self.reduction == "mean":
                return self.tensor_ops.mean(loss)
            elif self.reduction == "sum":
                return self.tensor_ops.sum(loss)
            else:  # none
                return loss


class FractionalCustomLoss(FractionalLossFunction):
    """Custom loss function with fractional calculus integration"""
    
    def __init__(self, loss_fn, fractional_order: float = 0.5, method: str = "RL", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.loss_fn = loss_fn
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        return self.loss_fn(predictions, targets)


class FractionalCombinedLoss(FractionalLossFunction):
    """Combined loss function with fractional calculus integration"""
    
    def __init__(self, loss_functions: list, weights: list = None, fractional_order: float = 0.5, method: str = "RL", backend: Optional[BackendType] = None):
        super().__init__(fractional_order, method, backend)
        self.loss_functions = loss_functions
        self.weights = weights or [1.0] * len(loss_functions)
        
        if len(self.weights) != len(self.loss_functions):
            raise ValueError("Number of weights must match number of loss functions")
    
    def compute_loss(self, predictions: Any, targets: Any) -> Any:
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            if isinstance(loss_fn, FractionalLossFunction):
                # Use fractional loss function
                loss = loss_fn(predictions, targets, use_fractional=False)  # Don't apply fractional twice
            else:
                # Use regular loss function
                loss = loss_fn(predictions, targets)
            
            total_loss += weight * loss
        
        return total_loss
