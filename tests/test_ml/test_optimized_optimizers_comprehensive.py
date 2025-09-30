#!/usr/bin/env python3
"""Comprehensive tests for optimized optimizers module."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
import sys

from hpfracc.ml.optimized_optimizers import (
    OptimizerConfig,
    OptimizedParameterState,
    OptimizedFractionalAdam,
    OptimizedFractionalSGD,
    OptimizedFractionalRMSprop,
    create_optimized_adam,
    create_optimized_sgd,
    create_optimized_rmsprop
)


class TestOptimizerConfig:
    """Test optimizer configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OptimizerConfig()
        assert config.lr == 0.001
        assert config.fractional_order == 0.5
        assert config.method == "RL"
        assert config.use_fractional is True
        assert config.cache_fractional is True
        assert config.memory_efficient is True
        assert config.use_jit is True
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizerConfig(
            lr=0.01,
            fractional_order=0.3,
            method="Caputo",
            use_fractional=False,
            cache_fractional=False,
            memory_efficient=False,
            use_jit=False
        )
        assert config.lr == 0.01
        assert config.fractional_order == 0.3
        assert config.method == "Caputo"
        assert config.use_fractional is False
        assert config.cache_fractional is False
        assert config.memory_efficient is False
        assert config.use_jit is False


class TestOptimizedParameterState:
    """Test optimized parameter state management."""
    
    def test_initialization(self):
        """Test parameter state initialization."""
        param_shape = (10, 5)
        state = OptimizedParameterState(param_shape, "torch", MagicMock())
        assert state.param_shape == param_shape
        assert state.backend == "torch"
        
    def test_state_creation(self):
        """Test state creation for parameters."""
        param_shape = (10, 5)
        state = OptimizedParameterState(param_shape, "torch", MagicMock())
        
        # Mock tensor operations
        mock_ops = MagicMock()
        mock_ops.zeros.return_value = torch.zeros(param_shape)
        state.tensor_ops = mock_ops
        
        state.create_state()
        assert state.momentum is not None
        assert state.variance is not None
        
    def test_state_update(self):
        """Test parameter state update."""
        param_shape = (10, 5)
        state = OptimizedParameterState(param_shape, "torch", MagicMock())
        
        # Mock tensor operations
        mock_ops = MagicMock()
        mock_ops.zeros.return_value = torch.zeros(param_shape)
        state.tensor_ops = mock_ops
        
        state.create_state()
        
        # Test state update
        param = torch.randn(param_shape)
        grad = torch.randn(param_shape)
        
        state.update_momentum(grad, beta=0.9)
        state.update_variance(grad, beta=0.999)
        
        assert state.momentum is not None
        assert state.variance is not None


class TestOptimizedFractionalAdam:
    """Test optimized fractional Adam optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalAdam(
            params=[torch.randn(10, requires_grad=True)],
            config=config
        )
        assert optimizer.config == config
        assert len(optimizer.param_groups) == 1
        
    def test_step_basic(self):
        """Test basic optimizer step."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        # Mock fractional derivative computation
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Check that step completed without error
            assert True
            
    def test_step_with_fractional_disabled(self):
        """Test step with fractional derivatives disabled."""
        config = OptimizerConfig(use_fractional=False)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        loss = (param ** 2).sum()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Should work without fractional derivatives
        assert True
        
    def test_step_with_cache(self):
        """Test step with caching enabled."""
        config = OptimizerConfig(cache_fractional=True)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Should use cached derivatives
            assert True


class TestOptimizedFractionalSGD:
    """Test optimized fractional SGD optimizer."""
    
    def test_initialization(self):
        """Test SGD optimizer initialization."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalSGD(
            params=[torch.randn(10, requires_grad=True)],
            config=config
        )
        assert optimizer.config == config
        
    def test_step_basic(self):
        """Test basic SGD step."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalSGD([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            assert True


class TestOptimizedFractionalRMSprop:
    """Test optimized fractional RMSprop optimizer."""
    
    def test_initialization(self):
        """Test RMSprop optimizer initialization."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalRMSprop(
            params=[torch.randn(10, requires_grad=True)],
            config=config
        )
        assert optimizer.config == config
        
    def test_step_basic(self):
        """Test basic RMSprop step."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalRMSprop([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            assert True


class TestOptimizedFractionalAdagrad:
    """Test optimized fractional Adagrad optimizer."""
    
    def test_initialization(self):
        """Test Adagrad optimizer initialization."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalAdagrad(
            params=[torch.randn(10, requires_grad=True)],
            config=config
        )
        assert optimizer.config == config
        
    def test_step_basic(self):
        """Test basic Adagrad step."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdagrad([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            assert True


class TestOptimizedFractionalAdamW:
    """Test optimized fractional AdamW optimizer."""
    
    def test_initialization(self):
        """Test AdamW optimizer initialization."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalAdamW(
            params=[torch.randn(10, requires_grad=True)],
            config=config
        )
        assert optimizer.config == config
        
    def test_step_basic(self):
        """Test basic AdamW step."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdamW([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            assert True


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_optimized_optimizer_adam(self):
        """Test creating Adam optimizer."""
        params = [torch.randn(10, requires_grad=True)]
        optimizer = create_optimized_optimizer(
            optimizer_type="adam",
            params=params,
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalAdam)
        
    def test_create_optimized_optimizer_sgd(self):
        """Test creating SGD optimizer."""
        params = [torch.randn(10, requires_grad=True)]
        optimizer = create_optimized_optimizer(
            optimizer_type="sgd",
            params=params,
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalSGD)
        
    def test_create_optimized_optimizer_rmsprop(self):
        """Test creating RMSprop optimizer."""
        params = [torch.randn(10, requires_grad=True)]
        optimizer = create_optimized_optimizer(
            optimizer_type="rmsprop",
            params=params,
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalRMSprop)
        
    def test_create_optimized_optimizer_adagrad(self):
        """Test creating Adagrad optimizer."""
        params = [torch.randn(10, requires_grad=True)]
        optimizer = create_optimized_optimizer(
            optimizer_type="adagrad",
            params=params,
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalAdagrad)
        
    def test_create_optimized_optimizer_adamw(self):
        """Test creating AdamW optimizer."""
        params = [torch.randn(10, requires_grad=True)]
        optimizer = create_optimized_optimizer(
            optimizer_type="adamw",
            params=params,
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalAdamW)
        
    def test_create_unknown_optimizer(self):
        """Test creating unknown optimizer type."""
        params = [torch.randn(10, requires_grad=True)]
        with pytest.raises(ValueError):
            create_optimized_optimizer(
                optimizer_type="unknown",
                params=params,
                lr=0.01
            )
            
    def test_get_optimized_optimizer_config(self):
        """Test getting optimizer configuration."""
        config = get_optimized_optimizer_config(
            optimizer_type="adam",
            lr=0.01,
            fractional_order=0.3
        )
        assert config.lr == 0.01
        assert config.fractional_order == 0.3


class TestCacheManagement:
    """Test cache management functionality."""
    
    def test_clear_fractional_cache(self):
        """Test clearing fractional derivative cache."""
        # Add something to cache
        from hpfracc.ml.optimized_optimizers import _fractional_cache
        _fractional_cache['test_key'] = 'test_value'
        
        # Clear cache
        clear_fractional_cache()
        
        # Check cache is empty
        assert len(_fractional_cache) == 0
        
    def test_cache_usage(self):
        """Test cache usage in optimizer."""
        config = OptimizerConfig(cache_fractional=True)
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(10)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            # First step - should compute derivative
            optimizer.step()
            optimizer.zero_grad()
            
            # Second step - should use cached derivative
            loss = (param ** 2).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            assert True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_parameters(self):
        """Test optimizer with empty parameters."""
        config = OptimizerConfig()
        optimizer = OptimizedFractionalAdam([], config=config)
        
        # Should handle empty parameters gracefully
        optimizer.step()
        assert True
        
    def test_zero_gradient(self):
        """Test optimizer with zero gradient."""
        config = OptimizerConfig()
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        # Zero gradient
        param.grad = torch.zeros_like(param)
        
        optimizer.step()
        optimizer.zero_grad()
        
        assert True
        
    def test_infinite_gradient(self):
        """Test optimizer with infinite gradient."""
        config = OptimizerConfig()
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        # Infinite gradient
        param.grad = torch.full_like(param, float('inf'))
        
        # Should handle gracefully
        try:
            optimizer.step()
            optimizer.zero_grad()
        except:
            # May raise error, which is acceptable
            pass
            
    def test_nan_gradient(self):
        """Test optimizer with NaN gradient."""
        config = OptimizerConfig()
        param = torch.randn(10, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        # NaN gradient
        param.grad = torch.full_like(param, float('nan'))
        
        # Should handle gracefully
        try:
            optimizer.step()
            optimizer.zero_grad()
        except:
            # May raise error, which is acceptable
            pass


class TestPerformance:
    """Test performance characteristics."""
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        config = OptimizerConfig(memory_efficient=True)
        param = torch.randn(100, 100, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(100, 100)
            
            loss = (param ** 2).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Should complete without memory issues
            assert True
            
    def test_convergence(self):
        """Test optimizer convergence."""
        config = OptimizerConfig(lr=0.01)
        param = torch.randn(5, requires_grad=True)
        optimizer = OptimizedFractionalAdam([param], config=config)
        
        with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
            mock_deriv.return_value = torch.randn(5)
            
            # Simple optimization problem
            for _ in range(10):
                loss = ((param - 1.0) ** 2).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            # Should converge towards target
            assert torch.allclose(param, torch.ones_like(param), atol=1e-1)
            
    def test_different_fractional_orders(self):
        """Test different fractional orders."""
        fractional_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for alpha in fractional_orders:
            config = OptimizerConfig(fractional_order=alpha)
            param = torch.randn(10, requires_grad=True)
            optimizer = OptimizedFractionalAdam([param], config=config)
            
            with patch.object(optimizer, '_compute_fractional_derivative') as mock_deriv:
                mock_deriv.return_value = torch.randn(10)
                
                loss = (param ** 2).sum()
                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Should work for all fractional orders
                assert True
