"""
Unit tests for SDE-specific loss functions in hpfracc.ml.losses

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from hpfracc.ml.losses import (
    FractionalSDEMSELoss, FractionalKLDivergenceLoss,
    FractionalPathwiseLoss, FractionalMomentMatchingLoss
)


class TestFractionalSDEMSELoss:
    """Test FractionalSDEMSELoss"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalSDEMSELoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn is not None
    
    def test_basic_computation(self):
        """Test basic MSE computation"""
        # Simple case
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_perfect_prediction(self):
        """Test loss with perfect prediction"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = predicted.clone()
        
        loss = self.loss_fn(predicted, target)
        
        assert loss.item() == 0.0
    
    def test_batch_processing(self):
        """Test batch processing"""
        batch_size = 5
        seq_len = 10
        state_dim = 3
        
        predicted = torch.randn(batch_size, seq_len, state_dim)
        target = torch.randn(batch_size, seq_len, state_dim)
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None
        assert not torch.any(torch.isnan(predicted.grad))
    
    def test_weighted_loss(self):
        """Test weighted loss computation"""
        loss_fn = FractionalSDEMSELoss(weight=2.0)
        
        predicted = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.5, 1.5]])
        
        loss = loss_fn(predicted, target)
        
        # Should be scaled by weight
        assert loss.item() > 0


class TestFractionalKLDivergenceLoss:
    """Test FractionalKLDivergenceLoss"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalKLDivergenceLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn is not None
    
    def test_basic_computation(self):
        """Test basic KL divergence computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # KL divergence is non-negative
        assert not torch.isnan(loss)
    
    def test_identical_distributions(self):
        """Test KL divergence with identical distributions"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = predicted.clone()
        
        loss = self.loss_fn(predicted, target)
        
        # KL divergence should be close to 0 for identical distributions
        assert loss.item() < 1e-6
    
    def test_numerical_stability(self):
        """Test numerical stability"""
        # Test with very small values
        predicted = torch.tensor([[1e-8, 1e-8], [1e-8, 1e-8]])
        target = torch.tensor([[1e-7, 1e-7], [1e-7, 1e-7]])
        
        loss = self.loss_fn(predicted, target)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None
        assert not torch.any(torch.isnan(predicted.grad))


class TestFractionalPathwiseLoss:
    """Test FractionalPathwiseLoss"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalPathwiseLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn is not None
    
    def test_basic_computation(self):
        """Test basic pathwise loss computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_uncertainty_weighting(self):
        """Test uncertainty weighting"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        uncertainty = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        
        loss = self.loss_fn(predicted, target, uncertainty=uncertainty)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None
        assert not torch.any(torch.isnan(predicted.grad))


class TestFractionalMomentMatchingLoss:
    """Test FractionalMomentMatchingLoss"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loss_fn = FractionalMomentMatchingLoss()
    
    def test_initialization(self):
        """Test loss function initialization"""
        assert self.loss_fn is not None
    
    def test_basic_computation(self):
        """Test basic moment matching loss computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_moment_computation(self):
        """Test moment computation"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        # Test first moment (mean)
        loss_mean = self.loss_fn(predicted, target, moment_order=1)
        
        # Test second moment (variance)
        loss_var = self.loss_fn(predicted, target, moment_order=2)
        
        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_var, torch.Tensor)
        assert not torch.isnan(loss_mean)
        assert not torch.isnan(loss_var)
    
    def test_multiple_moments(self):
        """Test multiple moment orders"""
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        loss = self.loss_fn(predicted, target, moment_orders=[1, 2, 3])
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        predicted = torch.tensor([[1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[1.5, 1.5]])
        
        loss = self.loss_fn(predicted, target)
        loss.backward()
        
        assert predicted.grad is not None
        assert not torch.any(torch.isnan(predicted.grad))


class TestLossFunctionIntegration:
    """Test integration between different loss functions"""
    
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
        assert moment_loss.item() >= 0
        
        # All losses should be finite
        assert torch.isfinite(mse_loss)
        assert torch.isfinite(kl_loss)
        assert torch.isfinite(pathwise_loss)
        assert torch.isfinite(moment_loss)
    
    def test_loss_with_different_shapes(self):
        """Test loss functions with different input shapes"""
        # 1D case
        pred_1d = torch.tensor([1.0, 2.0])
        target_1d = torch.tensor([1.5, 1.5])
        
        # 2D case
        pred_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_2d = torch.tensor([[1.5, 1.5], [2.5, 3.5]])
        
        # 3D case (batch, sequence, features)
        pred_3d = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target_3d = torch.tensor([[[1.5, 1.5], [2.5, 3.5]]])
        
        loss_fn = FractionalSDEMSELoss()
        
        loss_1d = loss_fn(pred_1d, target_1d)
        loss_2d = loss_fn(pred_2d, target_2d)
        loss_3d = loss_fn(pred_3d, target_3d)
        
        assert isinstance(loss_1d, torch.Tensor)
        assert isinstance(loss_2d, torch.Tensor)
        assert isinstance(loss_3d, torch.Tensor)
        
        assert not torch.isnan(loss_1d)
        assert not torch.isnan(loss_2d)
        assert not torch.isnan(loss_3d)


class TestLossFunctionEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_tensors(self):
        """Test with empty tensors"""
        predicted = torch.tensor([])
        target = torch.tensor([])
        
        loss_fn = FractionalSDEMSELoss()
        
        with pytest.raises(RuntimeError):
            loss_fn(predicted, target)
    
    def test_mismatched_shapes(self):
        """Test with mismatched tensor shapes"""
        predicted = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.5, 1.5], [2.5, 3.5]])  # Different shape
        
        loss_fn = FractionalSDEMSELoss()
        
        with pytest.raises(RuntimeError):
            loss_fn(predicted, target)
    
    def test_nan_inputs(self):
        """Test with NaN inputs"""
        predicted = torch.tensor([[float('nan'), 2.0]])
        target = torch.tensor([[1.5, 1.5]])
        
        loss_fn = FractionalSDEMSELoss()
        
        loss = loss_fn(predicted, target)
        
        # Loss should handle NaN gracefully
        assert torch.isnan(loss)
    
    def test_inf_inputs(self):
        """Test with infinite inputs"""
        predicted = torch.tensor([[float('inf'), 2.0]])
        target = torch.tensor([[1.5, 1.5]])
        
        loss_fn = FractionalSDEMSELoss()
        
        loss = loss_fn(predicted, target)
        
        # Loss should handle inf gracefully
        assert torch.isinf(loss) or torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
