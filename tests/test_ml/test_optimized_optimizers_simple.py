#!/usr/bin/env python3
"""Simple tests for optimized optimizers module."""

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
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizerConfig(
            lr=0.01,
            fractional_order=0.3,
            method="Caputo",
            use_fractional=False
        )
        assert config.lr == 0.01
        assert config.fractional_order == 0.3
        assert config.method == "Caputo"
        assert config.use_fractional is False


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


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_optimized_adam(self):
        """Test creating Adam optimizer."""
        optimizer = create_optimized_adam(
            params=[torch.randn(10, requires_grad=True)],
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalAdam)
        
    def test_create_optimized_sgd(self):
        """Test creating SGD optimizer."""
        optimizer = create_optimized_sgd(
            params=[torch.randn(10, requires_grad=True)],
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalSGD)
        
    def test_create_optimized_rmsprop(self):
        """Test creating RMSprop optimizer."""
        optimizer = create_optimized_rmsprop(
            params=[torch.randn(10, requires_grad=True)],
            lr=0.01
        )
        assert isinstance(optimizer, OptimizedFractionalRMSprop)


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
