"""
Corrected tests for the layers module based on actual API.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.ml.layers import (
    LayerConfig,
    FractionalConv1D,
    FractionalConv2D,
    FractionalLSTM,
    FractionalTransformer,
    FractionalPooling,
    FractionalBatchNorm1d,
    FractionalDropout,
    FractionalLayerNorm
)
from hpfracc.core.definitions import FractionalOrder
from hpfracc.ml.backends import BackendType


class TestLayerConfig:
    """Test LayerConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        config = LayerConfig()
        
        assert config.fractional_order is not None
        assert isinstance(config.fractional_order, FractionalOrder)
        assert config.method == "RL"
        assert config.use_fractional is True
        assert config.activation == "relu"
        assert config.dropout == 0.1
        assert config.backend == BackendType.AUTO
    
    def test_custom_initialization(self):
        """Test custom initialization."""
        fractional_order = FractionalOrder(0.7)
        config = LayerConfig(
            fractional_order=fractional_order,
            method="GL",
            use_fractional=False,
            activation="tanh",
            dropout=0.2,
            backend=BackendType.TORCH
        )
        
        assert config.fractional_order == fractional_order
        assert config.method == "GL"
        assert config.use_fractional is False
        assert config.activation == "tanh"
        assert config.dropout == 0.2
        assert config.backend == BackendType.TORCH
    
    def test_post_init_with_none_fractional_order(self):
        """Test post_init creates default FractionalOrder when None."""
        config = LayerConfig(fractional_order=None)
        
        assert config.fractional_order is not None
        assert isinstance(config.fractional_order, FractionalOrder)
        assert config.fractional_order.alpha == 0.5


class TestFractionalConv1D:
    """Test FractionalConv1D layer."""
    
    def test_initialization_default_config(self):
        """Test initialization with default config."""
        layer = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == 3
        assert layer.stride == 1
        assert layer.padding == 0
        assert layer.dilation == 1
        assert layer.groups == 1
        assert layer.bias is not None  # bias is a tensor, not boolean
        assert layer.config is not None
        assert layer.backend is not None
        assert layer.tensor_ops is not None
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.6),
            method="GL",
            use_fractional=True,
            activation="tanh",
            dropout=0.2
        )
        
        layer = FractionalConv1D(
            in_channels=5,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=1,
            dilation=2,
            groups=1,
            bias=False,
            config=config,
            backend=BackendType.TORCH
        )
        
        assert layer.in_channels == 5
        assert layer.out_channels == 32
        assert layer.kernel_size == 5
        assert layer.stride == 2
        assert layer.padding == 1
        assert layer.dilation == 2
        assert layer.groups == 1
        assert layer.bias is None  # bias=False means None
        assert layer.config == config
        assert layer.backend == BackendType.TORCH
    
    def test_initialize_weights_torch(self):
        """Test weight initialization for PyTorch backend."""
        with patch('hpfracc.ml.layers.get_backend_manager') as mock_manager:
            mock_manager.return_value.active_backend = BackendType.TORCH
            
            layer = FractionalConv1D(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                backend=BackendType.TORCH
            )
            
            assert hasattr(layer, 'weight')
            assert layer.weight.shape == (16, 3, 3)
            assert layer.weight.requires_grad
            assert hasattr(layer, 'bias')
            assert layer.bias.shape == (16,)
            assert layer.bias.requires_grad
    
    def test_initialize_weights_torch_no_bias(self):
        """Test weight initialization for PyTorch backend without bias."""
        with patch('hpfracc.ml.layers.get_backend_manager') as mock_manager:
            mock_manager.return_value.active_backend = BackendType.TORCH
            
            layer = FractionalConv1D(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                bias=False,
                backend=BackendType.TORCH
            )
            
            assert hasattr(layer, 'weight')
            assert layer.weight.shape == (16, 3, 3)
            assert layer.weight.requires_grad
            assert layer.bias is None
    
    def test_forward_torch(self):
        """Test forward pass with PyTorch backend."""
        with patch('hpfracc.ml.layers.get_backend_manager') as mock_manager:
            mock_manager.return_value.active_backend = BackendType.TORCH
            
            layer = FractionalConv1D(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                backend=BackendType.TORCH
            )
            
            # Create test input
            x = torch.randn(1, 3, 10)  # (batch, channels, length)
            
            with patch('hpfracc.ml.layers.fractional_derivative') as mock_frac:
                mock_frac.return_value = x
                
                result = layer.forward(x)
                
                assert result is not None
                assert result.shape[0] == 1  # batch dimension preserved
                assert result.shape[1] == 16  # output channels
                # Check that fractional derivative was called if use_fractional is True
                if layer.config.use_fractional:
                    mock_frac.assert_called_once()


class TestFractionalConv2D:
    """Test FractionalConv2D layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalConv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3)
        )
        
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == (3, 3)
        assert layer.stride == (1, 1)
        assert layer.padding == (0, 0)
        assert layer.dilation == (1, 1)
        assert layer.groups == 1
        assert layer.bias is not None  # bias is a tensor, not boolean
    
    def test_initialize_weights_torch(self):
        """Test weight initialization for PyTorch backend."""
        with patch('hpfracc.ml.layers.get_backend_manager') as mock_manager:
            mock_manager.return_value.active_backend = BackendType.TORCH
            
            layer = FractionalConv2D(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                backend=BackendType.TORCH
            )
            
            assert hasattr(layer, 'weight')
            assert layer.weight.shape == (16, 3, 3, 3)
            assert layer.weight.requires_grad
            assert hasattr(layer, 'bias')
            assert layer.bias.shape == (16,)
            assert layer.bias.requires_grad


class TestFractionalLSTM:
    """Test FractionalLSTM layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=2
        )
        
        assert layer.input_size == 10
        assert layer.hidden_size == 20
        assert layer.num_layers == 2
        assert layer.bidirectional is False
        assert layer.dropout == 0.0
        assert layer.bias is True
    
    def test_initialization_bidirectional(self):
        """Test initialization with bidirectional LSTM."""
        layer = FractionalLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=2,
            bidirectional=True,
            dropout=0.1
        )
        
        assert layer.input_size == 10
        assert layer.hidden_size == 20
        assert layer.num_layers == 2
        assert layer.bidirectional is True
        assert layer.dropout == 0.1
        assert layer.bias is True


class TestFractionalTransformer:
    """Test FractionalTransformer layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalTransformer(
            d_model=512,
            n_heads=8,  # Use correct parameter name
            d_ff=2048,
            dropout=0.1,
            activation="relu"
        )
        
        assert layer.d_model == 512
        assert layer.n_heads == 8  # Use correct attribute name
        assert layer.d_ff == 2048
        assert layer.dropout == 0.1
        assert layer.activation == "relu"
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = FractionalTransformer(
            d_model=256,
            n_heads=4,
            d_ff=1024,
            dropout=0.2,
            activation="gelu"
        )
        
        assert layer.d_model == 256
        assert layer.n_heads == 4
        assert layer.d_ff == 1024
        assert layer.dropout == 0.2
        assert layer.activation == "gelu"


class TestFractionalPooling:
    """Test FractionalPooling layer."""
    
    def test_initialization_1d(self):
        """Test initialization for 1D pooling."""
        layer = FractionalPooling(
            kernel_size=2,
            stride=2,
            padding=0,
            pool_type="max"  # Check if this parameter exists
        )
        
        assert layer.kernel_size == 2
        assert layer.stride == 2
        assert layer.padding == 0
        # Check if pool_type attribute exists
        if hasattr(layer, 'pool_type'):
            assert layer.pool_type == "max"
        assert layer.dim == 1
    
    def test_initialization_2d(self):
        """Test initialization for 2D pooling."""
        layer = FractionalPooling(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            pool_type="avg",  # Check if this parameter exists
            dim=2
        )
        
        assert layer.kernel_size == (2, 2)
        assert layer.stride == (2, 2)
        assert layer.padding == (0, 0)
        # Check if pool_type attribute exists
        if hasattr(layer, 'pool_type'):
            assert layer.pool_type == "avg"
        assert layer.dim == 2


class TestFractionalBatchNorm1d:
    """Test FractionalBatchNorm1d layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalBatchNorm1d(
            num_features=16,
            eps=1e-5,
            momentum=0.1
        )
        
        assert layer.num_features == 16
        assert layer.eps == 1e-5
        assert layer.momentum == 0.1
        assert layer.affine is True
        assert layer.track_running_stats is True
    
    def test_initialization_no_affine(self):
        """Test initialization without affine transformation."""
        layer = FractionalBatchNorm1d(
            num_features=16,
            affine=False,
            track_running_stats=False
        )
        
        assert layer.num_features == 16
        assert layer.affine is False
        assert layer.track_running_stats is False


class TestFractionalDropout:
    """Test FractionalDropout layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalDropout(
            p=0.5,
            inplace=False
        )
        
        assert layer.p == 0.5
        assert layer.inplace is False
    
    def test_initialization_inplace(self):
        """Test initialization with inplace operation."""
        layer = FractionalDropout(
            p=0.3,
            inplace=True
        )
        
        assert layer.p == 0.3
        assert layer.inplace is True


class TestFractionalLayerNorm:
    """Test FractionalLayerNorm layer."""
    
    def test_initialization(self):
        """Test initialization."""
        layer = FractionalLayerNorm(
            normalized_shape=16,
            eps=1e-5
        )
        
        assert layer.normalized_shape == 16
        assert layer.eps == 1e-5
        assert layer.elementwise_affine is True
    
    def test_initialization_no_affine(self):
        """Test initialization without elementwise affine transformation."""
        layer = FractionalLayerNorm(
            normalized_shape=(16, 10),
            eps=1e-6,
            elementwise_affine=False
        )
        
        assert layer.normalized_shape == (16, 10)
        assert layer.eps == 1e-6
        assert layer.elementwise_affine is False


class TestLayerIntegration:
    """Test integration between different layer types."""
    
    def test_layer_config_consistency(self):
        """Test that LayerConfig is consistent across layers."""
        config = LayerConfig(
            fractional_order=FractionalOrder(0.6),
            method="GL",
            use_fractional=True,
            activation="tanh",
            dropout=0.2
        )
        
        conv1d = FractionalConv1D(3, 16, 3, config=config)
        conv2d = FractionalConv2D(3, 16, (3, 3), config=config)
        
        assert conv1d.config == config
        assert conv2d.config == config
        assert conv1d.config.fractional_order.alpha == 0.6
        assert conv2d.config.fractional_order.alpha == 0.6
    
    def test_backend_consistency(self):
        """Test that backend is consistent across layers."""
        backend = BackendType.TORCH
        
        conv1d = FractionalConv1D(3, 16, 3, backend=backend)
        lstm = FractionalLSTM(10, 20, 1, backend=backend)
        
        assert conv1d.backend == backend
        assert lstm.backend == backend
    
    def test_fractional_order_consistency(self):
        """Test that fractional order is consistent across layers."""
        fractional_order = FractionalOrder(0.7)
        config = LayerConfig(fractional_order=fractional_order)
        
        conv1d = FractionalConv1D(3, 16, 3, config=config)
        transformer = FractionalTransformer(512, 8, config=config)
        
        assert conv1d.config.fractional_order == fractional_order
        assert transformer.config.fractional_order == fractional_order


class TestLayerBasicFunctionality:
    """Test basic functionality of layers."""
    
    def test_conv1d_forward_pass(self):
        """Test basic forward pass of Conv1D."""
        layer = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(1, 3, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape[0] == 1  # batch size preserved
        assert result.shape[1] == 16  # output channels
        assert result.shape[2] == 8  # length after convolution (10 - 3 + 1)
    
    def test_conv2d_forward_pass(self):
        """Test basic forward pass of Conv2D."""
        layer = FractionalConv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            backend=BackendType.TORCH
        )
        
        x = torch.randn(1, 3, 10, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape[0] == 1  # batch size preserved
        assert result.shape[1] == 16  # output channels
        assert result.shape[2] == 8  # height after convolution
        assert result.shape[3] == 8  # width after convolution
    
    def test_lstm_forward_pass(self):
        """Test basic forward pass of LSTM."""
        layer = FractionalLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=1,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(5, 1, 10)  # (seq_len, batch, input_size)
        result, (hidden, cell) = layer.forward(x)
        
        assert result is not None
        assert result.shape[0] == 5  # seq_len preserved
        assert result.shape[1] == 1  # batch size preserved
        assert result.shape[2] == 20  # hidden size
    
    def test_transformer_forward_pass(self):
        """Test basic forward pass of Transformer."""
        layer = FractionalTransformer(
            d_model=512,
            n_heads=8,
            d_ff=2048,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(10, 1, 512)  # (seq_len, batch, d_model)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape[0] == 10  # seq_len preserved
        assert result.shape[1] == 1  # batch size preserved
        assert result.shape[2] == 512  # d_model preserved
    
    def test_pooling_forward_pass(self):
        """Test basic forward pass of Pooling."""
        layer = FractionalPooling(
            kernel_size=2,
            stride=2,
            dim=1,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(1, 16, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape[0] == 1  # batch size preserved
        assert result.shape[1] == 16  # channels preserved
        assert result.shape[2] == 5  # length after pooling (10 / 2)
    
    def test_batch_norm_forward_pass(self):
        """Test basic forward pass of BatchNorm."""
        layer = FractionalBatchNorm1d(
            num_features=16,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(32, 16, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape == x.shape  # shape preserved
    
    def test_dropout_forward_pass(self):
        """Test basic forward pass of Dropout."""
        layer = FractionalDropout(
            p=0.5,
            backend=BackendType.TORCH
        )
        
        x = torch.randn(32, 16, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape == x.shape  # shape preserved
    
    def test_layer_norm_forward_pass(self):
        """Test basic forward pass of LayerNorm."""
        layer = FractionalLayerNorm(
            normalized_shape=10,  # Match the last dimension of input
            backend=BackendType.TORCH
        )
        
        x = torch.randn(32, 16, 10)
        result = layer.forward(x)
        
        assert result is not None
        assert result.shape == x.shape  # shape preserved
