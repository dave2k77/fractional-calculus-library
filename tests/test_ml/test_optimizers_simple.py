"""
Simple Tests for Fractional Calculus Optimizers

This module tests the enhanced optimizers with fractional calculus integration.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import directly to avoid __init__.py issues
from hpfracc.ml.optimizers import (
    SimpleFractionalOptimizer, SimpleFractionalSGD, SimpleFractionalAdam, SimpleFractionalRMSprop
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class MockParameter:
    """Mock parameter for testing"""
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

    def zero_grad(self):
        self.grad = None


class TestSimpleFractionalSGD:
    """Test SimpleFractionalSGD optimizer"""

    def test_simple_fractional_sgd_creation(self):
        """Test SimpleFractionalSGD creation"""
        sgd = SimpleFractionalSGD(lr=0.001, fractional_order=0.5, method="RL")
        assert sgd.fractional_order.alpha == 0.5
        assert sgd.method == "RL"
        assert sgd.lr == 0.001
        assert sgd.momentum == 0.0

    def test_simple_fractional_sgd_with_momentum(self):
        """Test SimpleFractionalSGD with momentum"""
        sgd = SimpleFractionalSGD(lr=0.001, momentum=0.9, fractional_order=0.5, method="RL")
        assert sgd.momentum == 0.9

    def test_simple_fractional_sgd_step(self):
        """Test SimpleFractionalSGD step"""
        sgd = SimpleFractionalSGD(lr=0.001, fractional_order=0.5, method="RL")
        params = [MockParameter(torch.randn(2, 3))]
        gradients = [torch.ones(2, 3)]
        
        # Mock the fractional_update method
        with patch.object(sgd, 'fractional_update', return_value=torch.ones(2, 3)):
            sgd.step(params, gradients)
            # Check that step was called without error


class TestSimpleFractionalAdam:
    """Test SimpleFractionalAdam optimizer"""

    def test_simple_fractional_adam_creation(self):
        """Test SimpleFractionalAdam creation"""
        adam = SimpleFractionalAdam(lr=0.001, fractional_order=0.5, method="RL")
        assert adam.fractional_order.alpha == 0.5
        assert adam.method == "RL"
        assert adam.betas == (0.9, 0.999)
        assert adam.eps == 1e-8

    def test_simple_fractional_adam_step(self):
        """Test SimpleFractionalAdam step"""
        adam = SimpleFractionalAdam(lr=0.001, fractional_order=0.5, method="RL")
        params = [MockParameter(torch.randn(2, 3))]
        gradients = [torch.ones(2, 3)]
        
        # Mock the fractional_update method
        with patch.object(adam, 'fractional_update', return_value=torch.ones(2, 3)):
            adam.step(params, gradients)
            # Check that step was called without error


class TestSimpleFractionalRMSprop:
    """Test SimpleFractionalRMSprop optimizer"""

    def test_simple_fractional_rmsprop_creation(self):
        """Test SimpleFractionalRMSprop creation"""
        rmsprop = SimpleFractionalRMSprop(lr=0.001, fractional_order=0.5, method="RL")
        assert rmsprop.fractional_order.alpha == 0.5
        assert rmsprop.method == "RL"
        assert rmsprop.alpha == 0.99
        assert rmsprop.eps == 1e-8

    def test_simple_fractional_rmsprop_step(self):
        """Test SimpleFractionalRMSprop step"""
        rmsprop = SimpleFractionalRMSprop(lr=0.001, fractional_order=0.5, method="RL")
        params = [MockParameter(torch.randn(2, 3))]
        gradients = [torch.ones(2, 3)]
        
        # Mock the fractional_update method
        with patch.object(rmsprop, 'fractional_update', return_value=torch.ones(2, 3)):
            rmsprop.step(params, gradients)
            # Check that step was called without error


class TestOptimizerIntegration:
    """Test optimizer integration and edge cases"""

    def test_optimizer_with_different_fractional_orders(self):
        """Test optimizers with different fractional orders"""
        orders = [0.1, 0.5, 0.9]
        for order in orders:
            sgd = SimpleFractionalSGD(lr=0.001, fractional_order=order, method="RL")
            assert sgd.fractional_order.alpha == order

    def test_optimizer_with_different_methods(self):
        """Test optimizers with different fractional methods"""
        methods = ["RL", "Caputo", "Grunwald"]
        for method in methods:
            sgd = SimpleFractionalSGD(lr=0.001, fractional_order=0.5, method=method)
            assert sgd.method == method

    def test_optimizer_state_management(self):
        """Test optimizer state management"""
        sgd = SimpleFractionalSGD(lr=0.001, fractional_order=0.5, method="RL")
        
        # Check that state is initialized
        assert len(sgd.state) == 0
        
        # Create a mock parameter
        param = MockParameter(torch.randn(2, 3))
        
        # Get state for the parameter
        state = sgd._get_state(param)
        assert state is not None
        
        # Check that state is stored
        assert len(sgd.state) > 0
