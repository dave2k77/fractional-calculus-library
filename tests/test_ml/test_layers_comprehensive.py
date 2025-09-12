"""
Comprehensive tests for ML layers module.

This module tests all layer functionality including fractional calculus integration,
different backends, and edge cases to ensure high coverage.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.layers import (
    LayerConfig, FractionalConv1D, FractionalConv2D, FractionalLSTM,
    FractionalTransformer, FractionalPooling, FractionalBatchNorm1d,
    FractionalDropout, FractionalLayerNorm
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestLayerConfig:
    """Test LayerConfig dataclass."""
    
    def test_layer_config_defaults(self):
        """Test LayerConfig with default values."""
        config = LayerConfig()
        assert config.fractional_order is not None
        assert config.method == "RL"
        assert config.use_fractional is True
        assert config.activation == "relu"
        assert config.dropout == 0.1
        assert config.backend == BackendType.AUTO
    
    def test_layer_config_custom(self):
        """Test LayerConfig with custom values."""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.3),
            method="Caputo",
            use_fractional=False,
            activation="sigmoid",
            dropout=0.2,
            backend=BackendType.TORCH
        )
        assert config.fractional_order.alpha == 0.3
        assert config.method == "Caputo"
        assert config.use_fractional is False
        assert config.activation == "sigmoid"
        assert config.dropout == 0.2
        assert config.backend == BackendType.TORCH


class TestFractionalConv1D:
    """Test FractionalConv1D layer."""
    
    def test_conv1d_initialization(self):
        """Test FractionalConv1D initialization."""
        layer = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.padding == 1
        assert layer.dilation == 1
        assert layer.config.use_fractional is True
        assert layer.config.fractional_order is not None
    
    def test_conv1d_forward_pass(self):
        """Test FractionalConv1D forward pass."""
        layer = FractionalConv1D(3, 16, 3, padding=1)
        x = torch.randn(2, 3, 10)  # batch_size=2, channels=3, length=10
        
        with patch.object(layer, '_apply_fractional_derivative') as mock_frac:
            with patch.object(layer, '_apply_convolution') as mock_conv:
                mock_conv.return_value = torch.randn(2, 16, 10)
                output = layer(x)
                
                assert output.shape == (2, 16, 10)
                mock_frac.assert_called_once()
                mock_conv.assert_called_once()
    
    def test_conv1d_without_fractional(self):
        """Test FractionalConv1D without fractional derivatives."""
        config = LayerConfig(use_fractional=False)
        layer = FractionalConv1D(3, 16, 3, config=config)
        x = torch.randn(2, 3, 10)
        
        with patch.object(layer, '_apply_convolution') as mock_conv:
            mock_conv.return_value = torch.randn(2, 16, 10)
            output = layer(x)
            
            assert output.shape == (2, 16, 10)
            mock_conv.assert_called_once()
    
    def test_conv1d_different_kernel_sizes(self):
        """Test FractionalConv1D with different kernel sizes."""
        for kernel_size in [1, 3, 5, 7]:
            layer = FractionalConv1D(3, 16, kernel_size)
            assert layer.kernel_size == kernel_size
    
    def test_conv1d_edge_cases(self):
        """Test FractionalConv1D edge cases."""
        # Single channel
        layer = FractionalConv1D(1, 1, 1)
        x = torch.randn(1, 1, 1)
        
        with patch.object(layer, '_apply_convolution') as mock_conv:
            mock_conv.return_value = torch.randn(1, 1, 1)
            output = layer(x)
            assert output.shape == (1, 1, 1)


if __name__ == "__main__":
    pytest.main([__file__])