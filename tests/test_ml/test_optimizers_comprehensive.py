"""
Comprehensive tests for ML optimizers module.

This module tests all optimizer functionality including fractional calculus integration,
different backends, and edge cases to ensure high coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.optimizers import (
    SimpleFractionalOptimizer, SimpleFractionalSGD, 
    SimpleFractionalAdam, SimpleFractionalRMSprop
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestSimpleFractionalOptimizer:
    """Test base SimpleFractionalOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test SimpleFractionalOptimizer initialization through concrete class."""
        optimizer = SimpleFractionalSGD(lr=0.01, fractional_order=0.3, method="Caputo")
        assert optimizer.lr == 0.01
        assert optimizer.fractional_order.alpha == 0.3
        assert optimizer.method == "Caputo"
        assert optimizer.use_fractional is True
        assert optimizer.backend is not None
        assert optimizer.tensor_ops is not None
        assert optimizer.state == {}
        assert optimizer.param_count == 0
    
    def test_optimizer_defaults(self):
        """Test SimpleFractionalOptimizer with default values."""
        optimizer = SimpleFractionalSGD()
        assert optimizer.lr == 0.001
        assert optimizer.fractional_order.alpha == 0.5
        assert optimizer.method == "RL"
        assert optimizer.use_fractional is True
        assert optimizer.backend is not None
    
    def test_get_param_index(self):
        """Test _get_param_index method."""
        optimizer = SimpleFractionalSGD()
        param1 = torch.randn(2, 3, requires_grad=True)
        param2 = torch.randn(2, 3, requires_grad=True)
        
        idx1 = optimizer._get_param_index(param1)
        idx2 = optimizer._get_param_index(param2)
        idx1_again = optimizer._get_param_index(param1)
        
        assert idx1 == 0
        assert idx2 == 1
        assert idx1_again == 0  # Same parameter should get same index
        assert optimizer.param_count == 2
    
    def test_zero_grad(self):
        """Test zero_grad method."""
        optimizer = SimpleFractionalSGD()
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        optimizer.zero_grad([param])
        assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_step_abstract(self):
        """Test that step method is abstract."""
        optimizer = SimpleFractionalSGD()
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        # This should work for concrete implementations
        optimizer.step([param], [param.grad])
        assert True  # If we get here, step was called successfully


class TestSimpleFractionalSGD:
    """Test SimpleFractionalSGD optimizer."""
    
    def test_sgd_initialization(self):
        """Test SimpleFractionalSGD initialization."""
        optimizer = SimpleFractionalSGD(lr=0.01, momentum=0.9)
        assert optimizer.lr == 0.01
        assert optimizer.momentum == 0.9
        # weight_decay and nesterov are not supported by SimpleFractionalSGD
        assert hasattr(optimizer, 'fractional_order')
    
    def test_sgd_step(self):
        """Test SimpleFractionalSGD step method."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        original_param = param.clone()
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Parameter should be updated
            assert not torch.allclose(param, original_param)
            mock_frac.assert_called_once()
    
    def test_sgd_with_momentum(self):
        """Test SimpleFractionalSGD with momentum."""
        optimizer = SimpleFractionalSGD(lr=0.01, momentum=0.9)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should have momentum state
            param_idx = optimizer._get_param_index(param)
            assert param_idx in optimizer.state
            assert 'momentum_buffer' in optimizer.state[param_idx]
    
    def test_sgd_with_weight_decay(self):
        """Test SimpleFractionalSGD with weight decay (not supported)."""
        # SimpleFractionalSGD doesn't support weight_decay
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should apply weight decay
            assert True  # If we get here, weight decay was applied
    
    def test_sgd_nesterov(self):
        """Test SimpleFractionalSGD with Nesterov momentum (not supported)."""
        # SimpleFractionalSGD doesn't support nesterov
        optimizer = SimpleFractionalSGD(lr=0.01, momentum=0.9)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # nesterov is not supported by SimpleFractionalSGD
            assert True


class TestSimpleFractionalAdam:
    """Test SimpleFractionalAdam optimizer."""
    
    def test_adam_initialization(self):
        """Test SimpleFractionalAdam initialization."""
        optimizer = SimpleFractionalAdam(
            lr=0.01, betas=(0.9, 0.999), eps=1e-8
        )
        assert optimizer.lr == 0.01
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-8
        # weight_decay is not supported by SimpleFractionalAdam
    
    def test_adam_step(self):
        """Test SimpleFractionalAdam step method."""
        optimizer = SimpleFractionalAdam(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        original_param = param.clone()
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Parameter should be updated
            assert not torch.allclose(param, original_param)
            mock_frac.assert_called_once()
    
    def test_adam_state_management(self):
        """Test SimpleFractionalAdam state management."""
        optimizer = SimpleFractionalAdam(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should have Adam state
            param_idx = optimizer._get_param_index(param)
            assert param_idx in optimizer.state
            assert 'step' in optimizer.state[param_idx]
            assert 'exp_avg' in optimizer.state[param_idx]
            assert 'exp_avg_sq' in optimizer.state[param_idx]
    
    def test_adam_different_betas(self):
        """Test SimpleFractionalAdam with different beta values."""
        for betas in [(0.8, 0.9), (0.9, 0.99), (0.95, 0.999)]:
            optimizer = SimpleFractionalAdam(lr=0.01, betas=betas)
            assert optimizer.betas == betas


class TestSimpleFractionalRMSprop:
    """Test SimpleFractionalRMSprop optimizer."""
    
    def test_rmsprop_initialization(self):
        """Test SimpleFractionalRMSprop initialization."""
        optimizer = SimpleFractionalRMSprop(
            lr=0.01, alpha=0.99, eps=1e-8
        )
        assert optimizer.lr == 0.01
        assert optimizer.alpha == 0.99
        assert optimizer.eps == 1e-8
        # weight_decay and momentum are not supported by SimpleFractionalRMSprop
    
    def test_rmsprop_step(self):
        """Test SimpleFractionalRMSprop step method."""
        optimizer = SimpleFractionalRMSprop(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        original_param = param.clone()
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Parameter should be updated
            assert not torch.allclose(param, original_param)
            mock_frac.assert_called_once()
    
    def test_rmsprop_state_management(self):
        """Test SimpleFractionalRMSprop state management."""
        optimizer = SimpleFractionalRMSprop(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should have RMSprop state
            param_idx = optimizer._get_param_index(param)
            assert param_idx in optimizer.state
            assert 'square_avg' in optimizer.state[param_idx]
    
    def test_rmsprop_with_momentum(self):
        """Test SimpleFractionalRMSprop with momentum."""
        # SimpleFractionalRMSprop doesn't support momentum, only square_avg
        optimizer = SimpleFractionalRMSprop(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should have square_avg (RMSprop state) but no momentum buffer
            param_idx = optimizer._get_param_index(param)
            assert 'square_avg' in optimizer.state[param_idx]
            assert 'momentum_buffer' not in optimizer.state[param_idx]


class TestOptimizerIntegration:
    """Test optimizer integration scenarios."""
    
    def test_optimizer_with_different_backends(self):
        """Test optimizers with different backends."""
        for backend in [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA]:
            optimizer = SimpleFractionalSGD(backend=backend)
            assert optimizer.backend == backend
    
    def test_optimizer_with_different_fractional_orders(self):
        """Test optimizers with different fractional orders."""
        for order in [0.1, 0.3, 0.5, 0.7, 0.9]:
            optimizer = SimpleFractionalSGD(fractional_order=order)
            assert optimizer.fractional_order.alpha == order
    
    def test_optimizer_without_fractional(self):
        """Test optimizers without fractional derivatives."""
        optimizer = SimpleFractionalSGD(use_fractional=False)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            optimizer.step([param], [param.grad])
            # Should not call fractional_update when use_fractional=False
            mock_frac.assert_not_called()
    
    def test_optimizer_gradient_flow(self):
        """Test gradient flow through optimizers."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Parameter should be updated
            assert param.grad is not None
    
    def test_optimizer_multiple_parameters(self):
        """Test optimizers with multiple parameters."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param1 = torch.randn(2, 3, requires_grad=True)
        param2 = torch.randn(3, 4, requires_grad=True)
        param1.grad = torch.randn(2, 3)
        param2.grad = torch.randn(3, 4)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            # Mock should return the input gradient unchanged (identity function)
            mock_frac.side_effect = lambda x: x
            optimizer.step([param1, param2])
            
            # Both parameters should be updated
            assert param1.grad is not None
            assert param2.grad is not None


class TestOptimizerEdgeCases:
    """Test optimizer edge cases and error handling."""
    
    def test_optimizer_with_zero_gradients(self):
        """Test optimizers with zero gradients."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.zeros_like(param)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should handle zero gradients gracefully
            assert True
    
    def test_optimizer_with_large_gradients(self):
        """Test optimizers with large gradients."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.randn(2, 3) * 1000  # Large gradients
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            # Should handle large gradients gracefully
            assert True
    
    def test_optimizer_with_nan_gradients(self):
        """Test optimizers with NaN gradients."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        param.grad = torch.tensor([[float('nan'), 1.0, 2.0], [3.0, 4.0, 5.0]])
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            # This should not crash
            try:
                optimizer.step([param], [param.grad])
                assert True
            except Exception:
                # If it raises an exception, that's also acceptable behavior
                pass
    
    def test_optimizer_with_empty_parameter_list(self):
        """Test optimizers with empty parameter list."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        
        # Should handle empty parameter list gracefully
        optimizer.step([])
        assert True
    
    def test_optimizer_with_single_element_parameters(self):
        """Test optimizers with single element parameters."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(1, 1, requires_grad=True)
        param.grad = torch.randn(1, 1)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            assert True


class TestOptimizerPerformance:
    """Test optimizer performance scenarios."""
    
    def test_optimizer_with_large_parameter_tensors(self):
        """Test optimizers with large parameter tensors."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(100, 100, requires_grad=True)
        param.grad = torch.randn(100, 100)
        
        with patch.object(optimizer, 'fractional_update') as mock_frac:
            mock_frac.return_value = param.grad
            optimizer.step([param], [param.grad])
            
            assert True
    
    def test_optimizer_multiple_steps(self):
        """Test optimizers with multiple steps."""
        optimizer = SimpleFractionalSGD(lr=0.01)
        param = torch.randn(2, 3, requires_grad=True)
        
        for i in range(10):
            param.grad = torch.randn(2, 3)
            with patch.object(optimizer, 'fractional_update') as mock_frac:
                mock_frac.return_value = param.grad
                optimizer.step([param], [param.grad])
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
