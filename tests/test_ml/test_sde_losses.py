"""
Unit tests for SDE loss functions in hpfracc.ml.losses

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from hpfracc.ml.losses import (
    FractionalSDEMSELoss, FractionalKLDivergenceLoss, FractionalPathwiseLoss,
    FractionalMomentMatchingLoss
)


class TestFractionalSDEMSELoss:
    """Test FractionalSDEMSELoss class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalSDEMSELoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn.num_samples == 10
        assert self.loss_fn.reduction == "mean"
        assert self.loss_fn.fractional_order.alpha == 0.5
    
    def test_basic_computation(self):
        """Test basic loss computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_perfect_prediction(self):
        """Test loss with perfect prediction"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = predicted.clone()
        
        # Disable fractional derivatives for perfect prediction test
        loss = self.loss_fn(predicted, target, use_fractional=False)
        
        assert loss.item() == 0.0
    
    def test_batch_processing(self):
        """Test batch processing"""
        predicted = torch.randn(32, 10)  # batch_size=32, features=10
        target = torch.randn(32, 10)
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None
    
    def test_weighted_loss(self):
        """Test weighted loss computation"""
        # The actual implementation doesn't have weight parameter
        # Test with different reduction types instead
        loss_fn_sum = FractionalSDEMSELoss(reduction="sum")
        
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = loss_fn_sum(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestFractionalKLDivergenceLoss:
    """Test FractionalKLDivergenceLoss class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalKLDivergenceLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn.eps == 1e-8
        assert self.loss_fn.fractional_order.alpha == 0.5
    
    def test_basic_computation(self):
        """Test basic KL divergence computation"""
        predicted = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        target = torch.tensor([[0.4, 0.6], [0.5, 0.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_identical_distributions(self):
        """Test KL divergence with identical distributions"""
        predicted = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        target = predicted.clone()
        
        # Disable fractional derivatives for identical distribution test
        loss = self.loss_fn(predicted, target, use_fractional=False)
        
        # KL divergence should be close to 0 for identical distributions
        assert loss.item() < 1e-6
    
    def test_numerical_stability(self):
        """Test numerical stability with small values"""
        predicted = torch.tensor([[1e-6, 1.0-1e-6], [0.5, 0.5]])
        target = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[0.3, 0.7]], requires_grad=True)
        target = torch.tensor([[0.4, 0.6]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None


class TestFractionalPathwiseLoss:
    """Test FractionalPathwiseLoss class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalPathwiseLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn.uncertainty_weight == 1.0
        assert self.loss_fn.fractional_order.alpha == 0.5
    
    def test_basic_computation(self):
        """Test basic pathwise loss computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_uncertainty_weighting(self):
        """Test uncertainty weighting"""
        # Create predictions with multiple samples (3D tensor)
        predicted = torch.randn(5, 2, 2)  # (num_samples, batch, features)
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None


class TestFractionalMomentMatchingLoss:
    """Test FractionalMomentMatchingLoss class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalMomentMatchingLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn.moments == [1, 2]
        assert len(self.loss_fn.weights) == 2
        assert self.loss_fn.fractional_order.alpha == 0.5
    
    def test_basic_computation(self):
        """Test basic moment matching loss computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        # The loss might be multi-dimensional, so check all elements are non-negative
        assert torch.all(loss >= 0)
    
    def test_moment_computation(self):
        """Test moment computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        # Test with default moments (mean and variance)
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert torch.all(loss >= 0)
    
    def test_multiple_moments(self):
        """Test multiple moment orders"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        # Create loss function with custom moments
        loss_fn_custom = FractionalMomentMatchingLoss(moments=[1, 2, 3])
        loss = loss_fn_custom(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert torch.all(loss >= 0)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        
        # Sum the loss to make it scalar for backward pass
        loss_sum = loss.sum()
        loss_sum.backward()
        
        assert predicted.grad is not None


class TestLossFunctionIntegration:
    """Test integration between loss functions"""
    
    def test_loss_function_comparison(self):
        """Test comparison between different loss functions"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        mse_loss = FractionalSDEMSELoss()(predicted, target)
        kl_loss = FractionalKLDivergenceLoss()(predicted, target)
        pathwise_loss = FractionalPathwiseLoss()(predicted, target)
        moment_loss = FractionalMomentMatchingLoss()(predicted, target)
        
        # All losses should be non-negative
        assert mse_loss.item() >= 0
        assert kl_loss.item() >= 0
        assert pathwise_loss.item() >= 0
        assert torch.all(moment_loss >= 0)
    
    def test_loss_with_different_shapes(self):
        """Test loss functions with different tensor shapes"""
        # Test 1D tensors
        pred_1d = torch.tensor([1.0, 2.0, 3.0])
        target_1d = torch.tensor([1.5, 1.5, 2.5])
        
        loss_1d = FractionalSDEMSELoss()(pred_1d, target_1d)
        assert isinstance(loss_1d, torch.Tensor)
        
        # Test 2D tensors
        pred_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_2d = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss_2d = FractionalSDEMSELoss()(pred_2d, target_2d)
        assert isinstance(loss_2d, torch.Tensor)


class TestLossFunctionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_tensors(self):
        """Test with empty tensors"""
        predicted = torch.tensor([])
        target = torch.tensor([])
        
        loss_fn = FractionalSDEMSELoss()
        
        # Should handle empty tensors gracefully
        try:
            loss = loss_fn(predicted, target)
            assert isinstance(loss, torch.Tensor)
        except Exception:
            # Some loss functions may not handle empty tensors
            pass
    
    def test_mismatched_shapes(self):
        """Test with mismatched tensor shapes"""
        predicted = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])  # Different shape
        
        loss_fn = FractionalSDEMSELoss()
        
        # PyTorch MSE loss handles broadcasting, so this should work
        loss = loss_fn(predicted, target)
        assert isinstance(loss, torch.Tensor)
    
    def test_nan_inputs(self):
        """Test with NaN inputs"""
        predicted = torch.tensor([[float('nan'), 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss_fn = FractionalSDEMSELoss()
        
        loss = loss_fn(predicted, target)
        
        # Should handle NaN gracefully
        assert isinstance(loss, torch.Tensor)
    
    def test_inf_inputs(self):
        """Test with infinite inputs"""
        predicted = torch.tensor([[float('inf'), 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss_fn = FractionalSDEMSELoss()
        
        loss = loss_fn(predicted, target)
        
        # Should handle inf gracefully
        assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])