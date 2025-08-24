"""
Loss Functions with Fractional Calculus Integration

This module provides loss functions that incorporate fractional derivatives,
enabling enhanced training dynamics and potentially better convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative


class FractionalLossFunction(nn.Module, ABC):
    """
    Base class for loss functions with fractional calculus integration
    
    This class provides a framework for loss functions that can apply
    fractional derivatives to predictions before computing the loss.
    """
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL"):
        super().__init__()
        self.fractional_order = FractionalOrder(fractional_order)
        self.method = method
    
    def fractional_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fractional derivative to input tensor
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with fractional derivative applied
        """
        return fractional_derivative(x, self.fractional_order.alpha, self.method)
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        predictions: torch.Tensor,
        targets: torch.Tensor,
        use_fractional: bool = True
    ) -> torch.Tensor:
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


class FractionalMSELoss(FractionalLossFunction):
    """Mean Squared Error loss with fractional calculus integration"""
    
    def __init__(self, fractional_order: float = 0.5, method: str = "RL", reduction: str = "mean"):
        super().__init__(fractional_order, method)
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets, reduction=self.reduction)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        use_fractional: bool = True
    ) -> torch.Tensor:
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


class FractionalCrossEntropyLoss(FractionalLossFunction):
    """Cross Entropy loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        super().__init__(fractional_order, method)
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            predictions, targets, weight=self.weight, ignore_index=self.ignore_index,
            reduction=self.reduction, label_smoothing=self.label_smoothing
        )


class FractionalHuberLoss(FractionalLossFunction):
    """Huber loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        delta: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.delta = delta
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(predictions, targets, delta=self.delta, reduction=self.reduction)


class FractionalSmoothL1Loss(FractionalLossFunction):
    """Smooth L1 loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        beta: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.beta = beta
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction=self.reduction)


class FractionalKLDivLoss(FractionalLossFunction):
    """KL Divergence loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        reduction: str = "mean",
        log_target: bool = False
    ):
        super().__init__(fractional_order, method)
        self.reduction = reduction
        self.log_target = log_target
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.kl_div(predictions, targets, reduction=self.reduction, log_target=self.log_target)


class FractionalBCELoss(FractionalLossFunction):
    """Binary Cross Entropy loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.weight = weight
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(
            predictions, targets, weight=self.weight, reduction=self.reduction
        )


class FractionalNLLLoss(FractionalLossFunction):
    """Negative Log Likelihood loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(
            predictions, targets, weight=self.weight, ignore_index=self.ignore_index,
            reduction=self.reduction
        )


class FractionalPoissonNLLLoss(FractionalLossFunction):
    """Poisson Negative Log Likelihood loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        log_input: bool = True,
        full: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.poisson_nll_loss(
            predictions, targets, log_input=self.log_input, full=self.full,
            eps=self.eps, reduction=self.reduction
        )


class FractionalCosineEmbeddingLoss(FractionalLossFunction):
    """Cosine Embedding loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        margin: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.margin = margin
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cosine_embedding_loss(
            predictions, targets, margin=self.margin, reduction=self.reduction
        )


class FractionalMarginRankingLoss(FractionalLossFunction):
    """Margin Ranking loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        margin: float = 1.0,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.margin = margin
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.margin_ranking_loss(
            predictions, targets, margin=self.margin, reduction=self.reduction
        )


class FractionalMultiMarginLoss(FractionalLossFunction):
    """Multi Margin loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        p: int = 1,
        margin: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.multi_margin_loss(
            predictions, targets, p=self.p, margin=self.margin,
            weight=self.weight, reduction=self.reduction
        )


class FractionalTripletMarginLoss(FractionalLossFunction):
    """Triplet Margin loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        margin: float = 1.0,
        p: float = 2.0,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: str = "mean"
    ):
        super().__init__(fractional_order, method)
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.triplet_margin_loss(
            predictions, targets, margin=self.margin, p=self.p, eps=self.eps,
            swap=self.swap, reduction=self.reduction
        )


class FractionalCTCLoss(FractionalLossFunction):
    """Connectionist Temporal Classification loss with fractional calculus integration"""
    
    def __init__(
        self,
        fractional_order: float = 0.5,
        method: str = "RL",
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False
    ):
        super().__init__(fractional_order, method)
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.ctc_loss(
            predictions, targets, blank=self.blank, reduction=self.reduction,
            zero_infinity=self.zero_infinity
        )


class FractionalCustomLoss(FractionalLossFunction):
    """Custom loss function with fractional calculus integration"""
    
    def __init__(
        self,
        loss_function: callable,
        fractional_order: float = 0.5,
        method: str = "RL",
        **kwargs
    ):
        super().__init__(fractional_order, method)
        self.loss_function = loss_function
        self.kwargs = kwargs
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_function(predictions, targets, **self.kwargs)


class FractionalCombinedLoss(FractionalLossFunction):
    """Combined loss function with fractional calculus integration"""
    
    def __init__(
        self,
        loss_functions: list,
        weights: Optional[list] = None,
        fractional_order: float = 0.5,
        method: str = "RL"
    ):
        super().__init__(fractional_order, method)
        self.loss_functions = loss_functions
        self.weights = weights or [1.0] * len(loss_functions)
        
        if len(self.weights) != len(self.loss_functions):
            raise ValueError("Number of weights must match number of loss functions")
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.loss_functions, self.weights):
            if isinstance(loss_fn, FractionalLossFunction):
                # Use fractional loss without applying fractional derivatives again
                loss = loss_fn.compute_loss(predictions, targets)
            else:
                # Use standard loss function
                loss = loss_fn(predictions, targets)
            
            total_loss += weight * loss
        
        return total_loss
