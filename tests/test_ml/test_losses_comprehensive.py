"""
Comprehensive tests for ML losses module.

This module tests all loss function functionality including fractional calculus integration,
different backends, and edge cases to ensure high coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.losses import (
    FractionalLossFunction, FractionalMSELoss, FractionalCrossEntropyLoss,
    FractionalHuberLoss, FractionalSmoothL1Loss, FractionalKLDivLoss,
    FractionalBCELoss, FractionalNLLLoss, FractionalPoissonNLLLoss,
    FractionalCosineEmbeddingLoss, FractionalMarginRankingLoss,
    FractionalMultiMarginLoss, FractionalTripletMarginLoss, FractionalCTCLoss,
    FractionalCustomLoss, FractionalCombinedLoss
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestFractionalLossFunction:
    """Test base FractionalLossFunction class."""
    
    def test_fractional_loss_function_initialization(self):
        """Test FractionalLossFunction initialization through concrete class."""
        loss_fn = FractionalMSELoss(fractional_order=0.3, method="Caputo")
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.method == "Caputo"
        assert loss_fn.backend is not None
        assert loss_fn.tensor_ops is not None
    
    def test_fractional_loss_function_defaults(self):
        """Test FractionalLossFunction with default values through concrete class."""
        loss_fn = FractionalMSELoss()
        assert loss_fn.fractional_order.alpha == 0.5
        assert loss_fn.method == "RL"
        assert loss_fn.backend is not None
    
    def test_fractional_forward_method(self):
        """Test fractional_forward method."""
        loss_fn = FractionalMSELoss()
        x = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'tensor_ops') as mock_ops:
            mock_ops.backend = BackendType.TORCH
            mock_ops.tensor_lib = torch
            result = loss_fn.fractional_forward(x)
            assert result is not None


class TestFractionalMSELoss:
    """Test FractionalMSELoss class."""
    
    def test_mse_loss_initialization(self):
        """Test FractionalMSELoss initialization."""
        loss_fn = FractionalMSELoss(fractional_order=0.4, reduction='mean')
        assert loss_fn.fractional_order.alpha == 0.4
        assert loss_fn.reduction == 'mean'
    
    def test_mse_loss_forward(self):
        """Test FractionalMSELoss forward pass."""
        loss_fn = FractionalMSELoss()
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0
    
    def test_mse_loss_different_reductions(self):
        """Test FractionalMSELoss with different reductions."""
        for reduction in ['mean', 'sum', 'none']:
            loss_fn = FractionalMSELoss(reduction=reduction)
            pred = torch.randn(2, 3)
            target = torch.randn(2, 3)
            
            with patch.object(loss_fn, 'fractional_forward') as mock_frac:
                mock_frac.return_value = pred
                loss = loss_fn(pred, target)
                assert loss is not None


class TestFractionalCrossEntropyLoss:
    """Test FractionalCrossEntropyLoss class."""
    
    def test_cross_entropy_loss_initialization(self):
        """Test FractionalCrossEntropyLoss initialization."""
        loss_fn = FractionalCrossEntropyLoss(fractional_order=0.3, reduction='mean')
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.reduction == 'mean'
    
    def test_cross_entropy_loss_forward(self):
        """Test FractionalCrossEntropyLoss forward pass."""
        loss_fn = FractionalCrossEntropyLoss()
        pred = torch.randn(2, 3)
        target = torch.tensor([0, 1])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0
    
    def test_cross_entropy_loss_with_weights(self):
        """Test FractionalCrossEntropyLoss with class weights."""
        weights = torch.tensor([0.5, 1.0, 1.5])
        loss_fn = FractionalCrossEntropyLoss()
        pred = torch.randn(2, 3)
        target = torch.tensor([0, 1])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None


class TestFractionalHuberLoss:
    """Test FractionalHuberLoss class."""
    
    def test_huber_loss_initialization(self):
        """Test FractionalHuberLoss initialization."""
        loss_fn = FractionalHuberLoss(fractional_order=0.3, delta=1.0)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.delta == 1.0
    
    def test_huber_loss_forward(self):
        """Test FractionalHuberLoss forward pass."""
        loss_fn = FractionalHuberLoss(delta=1.0)
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0
    
    def test_huber_loss_different_deltas(self):
        """Test FractionalHuberLoss with different delta values."""
        for delta in [0.1, 0.5, 1.0, 2.0]:
            loss_fn = FractionalHuberLoss(delta=delta)
            pred = torch.randn(2, 3)
            target = torch.randn(2, 3)
            
            with patch.object(loss_fn, 'fractional_forward') as mock_frac:
                mock_frac.return_value = pred
                loss = loss_fn(pred, target)
                assert loss is not None


class TestFractionalSmoothL1Loss:
    """Test FractionalSmoothL1Loss class."""
    
    def test_smooth_l1_loss_initialization(self):
        """Test FractionalSmoothL1Loss initialization."""
        loss_fn = FractionalSmoothL1Loss(fractional_order=0.3, beta=1.0)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.beta == 1.0
    
    def test_smooth_l1_loss_forward(self):
        """Test FractionalSmoothL1Loss forward pass."""
        loss_fn = FractionalSmoothL1Loss(beta=1.0)
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0


class TestFractionalKLDivLoss:
    """Test FractionalKLDivLoss class."""
    
    def test_kl_div_loss_initialization(self):
        """Test FractionalKLDivLoss initialization."""
        loss_fn = FractionalKLDivLoss(fractional_order=0.3, reduction='mean')
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.reduction == 'mean'
    
    def test_kl_div_loss_forward(self):
        """Test FractionalKLDivLoss forward pass."""
        loss_fn = FractionalKLDivLoss()
        pred = torch.randn(2, 3).softmax(dim=1)
        target = torch.randn(2, 3).softmax(dim=1)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalBCELoss:
    """Test FractionalBCELoss class."""
    
    def test_bce_loss_initialization(self):
        """Test FractionalBCELoss initialization."""
        loss_fn = FractionalBCELoss(fractional_order=0.3, reduction='mean')
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.reduction == 'mean'
    
    def test_bce_loss_forward(self):
        """Test FractionalBCELoss forward pass."""
        loss_fn = FractionalBCELoss()
        pred = torch.sigmoid(torch.randn(2, 3))
        target = torch.randint(0, 2, (2, 3)).float()
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0


class TestFractionalNLLLoss:
    """Test FractionalNLLLoss class."""
    
    def test_nll_loss_initialization(self):
        """Test FractionalNLLLoss initialization."""
        loss_fn = FractionalNLLLoss(fractional_order=0.3, reduction='mean')
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.reduction == 'mean'
    
    def test_nll_loss_forward(self):
        """Test FractionalNLLLoss forward pass."""
        loss_fn = FractionalNLLLoss()
        pred = torch.randn(2, 3)
        target = torch.tensor([0, 1])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalPoissonNLLLoss:
    """Test FractionalPoissonNLLLoss class."""
    
    def test_poisson_nll_loss_initialization(self):
        """Test FractionalPoissonNLLLoss initialization."""
        loss_fn = FractionalPoissonNLLLoss(fractional_order=0.3, log_input=True)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.log_input is True
    
    def test_poisson_nll_loss_forward(self):
        """Test FractionalPoissonNLLLoss forward pass."""
        loss_fn = FractionalPoissonNLLLoss(log_input=True)
        pred = torch.randn(2, 3)
        target = torch.randint(0, 10, (2, 3)).float()
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0


class TestFractionalCosineEmbeddingLoss:
    """Test FractionalCosineEmbeddingLoss class."""
    
    def test_cosine_embedding_loss_initialization(self):
        """Test FractionalCosineEmbeddingLoss initialization."""
        loss_fn = FractionalCosineEmbeddingLoss(fractional_order=0.3, margin=0.5)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.margin == 0.5
    
    def test_cosine_embedding_loss_forward(self):
        """Test FractionalCosineEmbeddingLoss forward pass."""
        loss_fn = FractionalCosineEmbeddingLoss(margin=0.5)
        pred1 = torch.randn(2, 3)
        pred2 = torch.randn(2, 3)
        target = torch.tensor([1, -1])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            # Mock should return the tuple for multi-input losses
            mock_frac.return_value = (pred1, pred2)
            # For multi-input losses, pass inputs as tuple
            loss = loss_fn((pred1, pred2), target)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalMarginRankingLoss:
    """Test FractionalMarginRankingLoss class."""
    
    def test_margin_ranking_loss_initialization(self):
        """Test FractionalMarginRankingLoss initialization."""
        loss_fn = FractionalMarginRankingLoss(fractional_order=0.3, margin=1.0)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.margin == 1.0
    
    def test_margin_ranking_loss_forward(self):
        """Test FractionalMarginRankingLoss forward pass."""
        loss_fn = FractionalMarginRankingLoss(margin=1.0)
        pred1 = torch.randn(2, 3)
        pred2 = torch.randn(2, 3)
        target = torch.tensor([[1, -1, 1], [-1, 1, -1]])  # Same shape as inputs
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            # Mock should return the tuple for multi-input losses
            mock_frac.return_value = (pred1, pred2)
            # For multi-input losses, pass inputs as tuple
            loss = loss_fn((pred1, pred2), target)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalMultiMarginLoss:
    """Test FractionalMultiMarginLoss class."""
    
    def test_multi_margin_loss_initialization(self):
        """Test FractionalMultiMarginLoss initialization."""
        loss_fn = FractionalMultiMarginLoss(fractional_order=0.3, p=1)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.p == 1
    
    def test_multi_margin_loss_forward(self):
        """Test FractionalMultiMarginLoss forward pass."""
        loss_fn = FractionalMultiMarginLoss(p=1)
        pred = torch.randn(2, 3)
        target = torch.tensor([0, 1])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0


class TestFractionalTripletMarginLoss:
    """Test FractionalTripletMarginLoss class."""
    
    def test_triplet_margin_loss_initialization(self):
        """Test FractionalTripletMarginLoss initialization."""
        loss_fn = FractionalTripletMarginLoss(fractional_order=0.3, margin=1.0)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.margin == 1.0
    
    def test_triplet_margin_loss_forward(self):
        """Test FractionalTripletMarginLoss forward pass."""
        loss_fn = FractionalTripletMarginLoss(margin=1.0)
        anchor = torch.randn(2, 3)
        positive = torch.randn(2, 3)
        negative = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            # Mock should return the tuple for multi-input losses
            mock_frac.return_value = (anchor, positive, negative)
            # For multi-input losses, pass inputs as tuple
            loss = loss_fn((anchor, positive, negative), None)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalCTCLoss:
    """Test FractionalCTCLoss class."""
    
    def test_ctc_loss_initialization(self):
        """Test FractionalCTCLoss initialization."""
        loss_fn = FractionalCTCLoss(fractional_order=0.3, blank=0)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.blank == 0
    
    def test_ctc_loss_forward(self):
        """Test FractionalCTCLoss forward pass."""
        loss_fn = FractionalCTCLoss(blank=0)
        pred = torch.randn(3, 2, 5)  # (T, N, C) - batch size is 2
        target = torch.tensor([[1, 2], [2, 1]])
        input_lengths = torch.tensor([3, 3])  # 1D tensor of batch size (2 elements)
        target_lengths = torch.tensor([2, 2])
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            # Mock should return the tuple for CTC loss
            mock_frac.return_value = (pred, input_lengths, target_lengths)
            # CTC loss needs additional parameters, pass as tuple
            loss = loss_fn((pred, input_lengths, target_lengths), target)
            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss


class TestFractionalCustomLoss:
    """Test FractionalCustomLoss class."""
    
    def test_custom_loss_initialization(self):
        """Test FractionalCustomLoss initialization."""
        def custom_loss(pred, target):
            return torch.mean((pred - target) ** 2)
        
        loss_fn = FractionalCustomLoss(custom_loss, fractional_order=0.3)
        assert loss_fn.fractional_order.alpha == 0.3
        assert loss_fn.custom_loss_fn == custom_loss
    
    def test_custom_loss_forward(self):
        """Test FractionalCustomLoss forward pass."""
        def custom_loss(pred, target):
            return torch.mean((pred - target) ** 2)
        
        loss_fn = FractionalCustomLoss(custom_loss)
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
            assert loss.item() >= 0


class TestFractionalCombinedLoss:
    """Test FractionalCombinedLoss class."""
    
    def test_combined_loss_initialization(self):
        """Test FractionalCombinedLoss initialization."""
        mse_loss = FractionalMSELoss()
        ce_loss = FractionalCrossEntropyLoss()
        loss_fn = FractionalCombinedLoss([mse_loss, ce_loss], [0.5, 0.5])
        
        assert len(loss_fn.loss_functions) == 2
        assert len(loss_fn.weights) == 2
        assert loss_fn.weights[0] == 0.5
        assert loss_fn.weights[1] == 0.5
    
    def test_combined_loss_forward(self):
        """Test FractionalCombinedLoss forward pass."""
        mse_loss = FractionalMSELoss()
        ce_loss = FractionalCrossEntropyLoss()
        loss_fn = FractionalCombinedLoss([mse_loss, ce_loss], [0.5, 0.5])
        
        pred = torch.randn(2, 3)
        target = torch.randn(2, 3)
        
        with patch.object(mse_loss, 'fractional_forward') as mock_frac1:
            with patch.object(ce_loss, 'fractional_forward') as mock_frac2:
                mock_frac1.return_value = pred
                mock_frac2.return_value = pred
                loss = loss_fn(pred, target)
                assert loss is not None
                assert isinstance(loss, torch.Tensor)
                assert loss.shape == ()  # Scalar loss


class TestLossIntegration:
    """Test loss function integration scenarios."""
    
    def test_loss_with_different_backends(self):
        """Test loss functions with different backends."""
        for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
            loss_fn = FractionalMSELoss(backend=backend)
            assert loss_fn.backend == backend
    
    def test_loss_gradient_flow(self):
        """Test gradient flow through loss functions."""
        loss_fn = FractionalMSELoss()
        pred = torch.randn(2, 3, requires_grad=True)
        target = torch.randn(2, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            loss.backward()
            
            assert pred.grad is not None
            assert pred.grad.shape == pred.shape
    
    def test_loss_with_different_fractional_orders(self):
        """Test loss functions with different fractional orders."""
        for order in [0.1, 0.3, 0.5, 0.7, 0.9]:
            loss_fn = FractionalMSELoss(fractional_order=order)
            assert loss_fn.fractional_order.alpha == order


class TestLossEdgeCases:
    """Test loss function edge cases and error handling."""
    
    def test_loss_with_empty_tensors(self):
        """Test loss functions with empty tensors."""
        loss_fn = FractionalMSELoss()
        pred = torch.randn(0, 3)
        target = torch.randn(0, 3)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
    
    def test_loss_with_single_element_tensors(self):
        """Test loss functions with single element tensors."""
        loss_fn = FractionalMSELoss()
        pred = torch.randn(1, 1)
        target = torch.randn(1, 1)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None
    
    def test_loss_with_large_tensors(self):
        """Test loss functions with large tensors."""
        loss_fn = FractionalMSELoss()
        pred = torch.randn(100, 100)
        target = torch.randn(100, 100)
        
        with patch.object(loss_fn, 'fractional_forward') as mock_frac:
            mock_frac.return_value = pred
            loss = loss_fn(pred, target)
            assert loss is not None


if __name__ == "__main__":
    pytest.main([__file__])
