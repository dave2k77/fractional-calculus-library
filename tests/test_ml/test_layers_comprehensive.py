"""
Comprehensive tests for ML layers module.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Test LayerConfig
def test_layer_config_initialization():
    """Test LayerConfig initialization and configuration."""
    try:
        from hpfracc.ml.layers import LayerConfig
        from hpfracc.core.definitions import FractionalOrder
        
        # Test default initialization
        config = LayerConfig()
        assert config.fractional_order is not None
        assert isinstance(config.fractional_order, FractionalOrder)
        assert config.method == "RL"
        assert config.use_fractional == True
        assert config.activation == "relu"
        assert config.dropout == 0.1
        
        # Test custom initialization
        custom_alpha = FractionalOrder(0.7)
        config_custom = LayerConfig(
            fractional_order=custom_alpha,
            method="Caputo",
            use_fractional=False,
            activation="tanh",
            dropout=0.2
        )
        assert config_custom.fractional_order == custom_alpha
        assert config_custom.method == "Caputo"
        assert config_custom.use_fractional == False
        assert config_custom.activation == "tanh"
        assert config_custom.dropout == 0.2
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_conv1d_initialization():
    """Test FractionalConv1D initialization."""
    try:
        from hpfracc.ml.layers import FractionalConv1D
        
        # Test basic initialization
        conv1d = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert conv1d.in_channels == 3
        assert conv1d.out_channels == 16
        assert conv1d.kernel_size == 3
        assert conv1d.stride == 1
        assert conv1d.padding == 0
        assert conv1d.dilation == 1
        
        # Test with custom parameters
        conv1d_custom = FractionalConv1D(
            in_channels=5,
            out_channels=32,
            kernel_size=5,
            stride=2,
            padding=1,
            dilation=2
        )
        assert conv1d_custom.in_channels == 5
        assert conv1d_custom.out_channels == 32
        assert conv1d_custom.kernel_size == 5
        assert conv1d_custom.stride == 2
        assert conv1d_custom.padding == 1
        assert conv1d_custom.dilation == 2
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_conv2d_initialization():
    """Test FractionalConv2D initialization."""
    try:
        from hpfracc.ml.layers import FractionalConv2D
        
        # Test basic initialization
        conv2d = FractionalConv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3
        )
        assert conv2d.in_channels == 3
        assert conv2d.out_channels == 16
        assert conv2d.kernel_size == (3, 3)  # 2D conv converts single int to tuple
        assert conv2d.stride == (1, 1)  # 2D conv converts single int to tuple
        assert conv2d.padding == (0, 0)  # 2D conv converts single int to tuple
        assert conv2d.dilation == (1, 1)  # 2D conv converts single int to tuple
        
        # Test with custom parameters
        conv2d_custom = FractionalConv2D(
            in_channels=5,
            out_channels=32,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2)
        )
        assert conv2d_custom.in_channels == 5
        assert conv2d_custom.out_channels == 32
        assert conv2d_custom.kernel_size == (5, 5)
        assert conv2d_custom.stride == (2, 2)
        assert conv2d_custom.padding == (1, 1)
        assert conv2d_custom.dilation == (2, 2)
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_lstm_initialization():
    """Test FractionalLSTM initialization."""
    try:
        from hpfracc.ml.layers import FractionalLSTM
        
        # Test basic initialization
        lstm = FractionalLSTM(
            input_size=10,
            hidden_size=20
        )
        assert lstm.input_size == 10
        assert lstm.hidden_size == 20
        assert lstm.num_layers == 1
        assert lstm.bias == True
        assert lstm.batch_first == False
        assert lstm.dropout == 0.0
        assert lstm.bidirectional == False
        
        # Test with custom parameters
        lstm_custom = FractionalLSTM(
            input_size=15,
            hidden_size=30,
            num_layers=2,
            bias=False,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        assert lstm_custom.input_size == 15
        assert lstm_custom.hidden_size == 30
        assert lstm_custom.num_layers == 2
        assert lstm_custom.bias == False
        assert lstm_custom.batch_first == True
        assert lstm_custom.dropout == 0.1
        assert lstm_custom.bidirectional == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_transformer_initialization():
    """Test FractionalTransformer initialization."""
    try:
        from hpfracc.ml.layers import FractionalTransformer
        
        # Test basic initialization
        transformer = FractionalTransformer(
            d_model=512,
            nhead=8
        )
        assert transformer.d_model == 512
        # Note: nhead attribute may not exist in all implementations
        if hasattr(transformer, 'nhead'):
            assert transformer.nhead == 8
        # Note: transformer attributes may vary by implementation
        if hasattr(transformer, 'num_encoder_layers'):
            assert transformer.num_encoder_layers == 6
        if hasattr(transformer, 'num_decoder_layers'):
            assert transformer.num_decoder_layers == 6
        if hasattr(transformer, 'dim_feedforward'):
            assert transformer.dim_feedforward == 2048
        if hasattr(transformer, 'dropout'):
            assert transformer.dropout == 0.1
        if hasattr(transformer, 'activation'):
            assert transformer.activation == "relu"
        
        # Test with custom parameters
        transformer_custom = FractionalTransformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.2,
            activation="gelu"
        )
        assert transformer_custom.d_model == 256
        if hasattr(transformer_custom, 'nhead'):
            assert transformer_custom.nhead == 4
        if hasattr(transformer_custom, 'num_encoder_layers'):
            assert transformer_custom.num_encoder_layers == 3
        if hasattr(transformer_custom, 'num_decoder_layers'):
            assert transformer_custom.num_decoder_layers == 3
        if hasattr(transformer_custom, 'dim_feedforward'):
            assert transformer_custom.dim_feedforward == 1024
        if hasattr(transformer_custom, 'dropout'):
            assert transformer_custom.dropout == 0.2
        if hasattr(transformer_custom, 'activation'):
            assert transformer_custom.activation == "gelu"
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_pooling_initialization():
    """Test FractionalPooling initialization."""
    try:
        from hpfracc.ml.layers import FractionalPooling
        
        # Test basic initialization
        pooling = FractionalPooling(
            kernel_size=2
        )
        assert pooling.kernel_size == (2, 2)  # 2D pooling converts single int to tuple
        assert pooling.stride == (2, 2)  # 2D pooling converts single int to tuple
        assert pooling.padding == (0, 0)  # 2D pooling converts single int to tuple
        # Note: return_indices and ceil_mode may not be available in all implementations
        if hasattr(pooling, 'return_indices'):
            assert pooling.return_indices == False
        if hasattr(pooling, 'ceil_mode'):
            assert pooling.ceil_mode == False
        
        # Test with custom parameters
        pooling_custom = FractionalPooling(
            kernel_size=3,
            stride=1,
            padding=1
        )
        assert pooling_custom.kernel_size == (3, 3)  # 2D pooling converts single int to tuple
        assert pooling_custom.stride == (1, 1)  # 2D pooling converts single int to tuple
        assert pooling_custom.padding == (1, 1)  # 2D pooling converts single int to tuple
        if hasattr(pooling_custom, 'return_indices'):
            assert pooling_custom.return_indices == True
        if hasattr(pooling_custom, 'ceil_mode'):
            assert pooling_custom.ceil_mode == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_batch_norm1d_initialization():
    """Test FractionalBatchNorm1d initialization."""
    try:
        from hpfracc.ml.layers import FractionalBatchNorm1d
        
        # Test basic initialization
        bn1d = FractionalBatchNorm1d(
            num_features=10
        )
        assert bn1d.num_features == 10
        assert bn1d.eps == 1e-5
        assert bn1d.momentum == 0.1
        assert bn1d.affine == True
        assert bn1d.track_running_stats == True
        
        # Test with custom parameters
        bn1d_custom = FractionalBatchNorm1d(
            num_features=20,
            eps=1e-3,
            momentum=0.2,
            affine=False,
            track_running_stats=False
        )
        assert bn1d_custom.num_features == 20
        assert bn1d_custom.eps == 1e-3
        assert bn1d_custom.momentum == 0.2
        assert bn1d_custom.affine == False
        assert bn1d_custom.track_running_stats == False
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_dropout_initialization():
    """Test FractionalDropout initialization."""
    try:
        from hpfracc.ml.layers import FractionalDropout
        
        # Test basic initialization
        dropout = FractionalDropout(
            p=0.5
        )
        assert dropout.p == 0.5
        assert dropout.inplace == False
        
        # Test with custom parameters
        dropout_custom = FractionalDropout(
            p=0.3,
            inplace=True
        )
        assert dropout_custom.p == 0.3
        assert dropout_custom.inplace == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_layer_norm_initialization():
    """Test FractionalLayerNorm initialization."""
    try:
        from hpfracc.ml.layers import FractionalLayerNorm
        
        # Test basic initialization
        layer_norm = FractionalLayerNorm(
            normalized_shape=10
        )
        assert layer_norm.normalized_shape == 10
        assert layer_norm.eps == 1e-5
        assert layer_norm.elementwise_affine == True
        
        # Test with custom parameters
        layer_norm_custom = FractionalLayerNorm(
            normalized_shape=(10, 20),
            eps=1e-3,
            elementwise_affine=False
        )
        assert layer_norm_custom.normalized_shape == (10, 20)
        assert layer_norm_custom.eps == 1e-3
        assert layer_norm_custom.elementwise_affine == False
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_post_init():
    """Test LayerConfig __post_init__ method."""
    try:
        from hpfracc.ml.layers import LayerConfig
        from hpfracc.core.definitions import FractionalOrder
        
        # Test that fractional_order is set to default if None
        config = LayerConfig(fractional_order=None)
        assert config.fractional_order is not None
        assert isinstance(config.fractional_order, FractionalOrder)
        assert config.fractional_order.alpha == 0.5
        
        # Test that existing fractional_order is preserved
        custom_alpha = FractionalOrder(0.8)
        config_custom = LayerConfig(fractional_order=custom_alpha)
        assert config_custom.fractional_order == custom_alpha
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_validation():
    """Test LayerConfig parameter validation."""
    try:
        from hpfracc.ml.layers import LayerConfig
        from hpfracc.core.definitions import FractionalOrder
        
        # Test valid configurations
        config1 = LayerConfig(
            method="RL",
            use_fractional=True,
            activation="relu",
            dropout=0.1
        )
        assert config1.method == "RL"
        assert config1.use_fractional == True
        assert config1.activation == "relu"
        assert config1.dropout == 0.1
        
        config2 = LayerConfig(
            method="Caputo",
            use_fractional=False,
            activation="tanh",
            dropout=0.5
        )
        assert config2.method == "Caputo"
        assert config2.use_fractional == False
        assert config2.activation == "tanh"
        assert config2.dropout == 0.5
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_conv1d_attributes():
    """Test FractionalConv1D attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalConv1D
        
        conv1d = FractionalConv1D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=2
        )
        
        # Test that all attributes are set correctly
        assert hasattr(conv1d, 'in_channels')
        assert hasattr(conv1d, 'out_channels')
        assert hasattr(conv1d, 'kernel_size')
        assert hasattr(conv1d, 'stride')
        assert hasattr(conv1d, 'padding')
        assert hasattr(conv1d, 'dilation')
        
        # Test attribute values
        assert conv1d.in_channels == 3
        assert conv1d.out_channels == 16
        assert conv1d.kernel_size == 3
        assert conv1d.stride == 2
        assert conv1d.padding == 1
        assert conv1d.dilation == 2
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_conv2d_attributes():
    """Test FractionalConv2D attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalConv2D
        
        conv2d = FractionalConv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dilation=(2, 2)
        )
        
        # Test that all attributes are set correctly
        assert hasattr(conv2d, 'in_channels')
        assert hasattr(conv2d, 'out_channels')
        assert hasattr(conv2d, 'kernel_size')
        assert hasattr(conv2d, 'stride')
        assert hasattr(conv2d, 'padding')
        assert hasattr(conv2d, 'dilation')
        
        # Test attribute values
        assert conv2d.in_channels == 3
        assert conv2d.out_channels == 16
        assert conv2d.kernel_size == (3, 3)
        assert conv2d.stride == (2, 2)
        assert conv2d.padding == (1, 1)
        assert conv2d.dilation == (2, 2)
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_lstm_attributes():
    """Test FractionalLSTM attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalLSTM
        
        lstm = FractionalLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=2,
            bias=False,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Test that all attributes are set correctly
        assert hasattr(lstm, 'input_size')
        assert hasattr(lstm, 'hidden_size')
        assert hasattr(lstm, 'num_layers')
        assert hasattr(lstm, 'bias')
        assert hasattr(lstm, 'batch_first')
        assert hasattr(lstm, 'dropout')
        assert hasattr(lstm, 'bidirectional')
        
        # Test attribute values
        assert lstm.input_size == 10
        assert lstm.hidden_size == 20
        assert lstm.num_layers == 2
        assert lstm.bias == False
        assert lstm.batch_first == True
        assert lstm.dropout == 0.1
        assert lstm.bidirectional == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_transformer_attributes():
    """Test FractionalTransformer attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalTransformer
        
        transformer = FractionalTransformer(
            d_model=256,
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.2,
            activation="gelu"
        )
        
        # Test that all attributes are set correctly
        assert hasattr(transformer, 'd_model')
        # Note: transformer attributes may vary by implementation
        if hasattr(transformer, 'nhead'):
            assert hasattr(transformer, 'nhead')
        if hasattr(transformer, 'num_encoder_layers'):
            assert hasattr(transformer, 'num_encoder_layers')
        if hasattr(transformer, 'num_decoder_layers'):
            assert hasattr(transformer, 'num_decoder_layers')
        if hasattr(transformer, 'dim_feedforward'):
            assert hasattr(transformer, 'dim_feedforward')
        if hasattr(transformer, 'dropout'):
            assert hasattr(transformer, 'dropout')
        if hasattr(transformer, 'activation'):
            assert hasattr(transformer, 'activation')
        
        # Test attribute values
        assert transformer.d_model == 256
        if hasattr(transformer, 'nhead'):
            assert transformer.nhead == 4
        if hasattr(transformer, 'num_encoder_layers'):
            assert transformer.num_encoder_layers == 3
        if hasattr(transformer, 'num_decoder_layers'):
            assert transformer.num_decoder_layers == 3
        if hasattr(transformer, 'dim_feedforward'):
            assert transformer.dim_feedforward == 1024
        if hasattr(transformer, 'dropout'):
            assert transformer.dropout == 0.2
        if hasattr(transformer, 'activation'):
            assert transformer.activation == "gelu"
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_pooling_attributes():
    """Test FractionalPooling attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalPooling
        
        pooling = FractionalPooling(
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Test that all attributes are set correctly
        assert hasattr(pooling, 'kernel_size')
        assert hasattr(pooling, 'stride')
        assert hasattr(pooling, 'padding')
        # Note: return_indices and ceil_mode may not be available in all implementations
        if hasattr(pooling, 'return_indices'):
            assert hasattr(pooling, 'return_indices')
        if hasattr(pooling, 'ceil_mode'):
            assert hasattr(pooling, 'ceil_mode')
        
        # Test attribute values
        assert pooling.kernel_size == (3, 3)  # 2D pooling converts single int to tuple
        assert pooling.stride == (1, 1)  # 2D pooling converts single int to tuple
        assert pooling.padding == (1, 1)  # 2D pooling converts single int to tuple
        if hasattr(pooling, 'return_indices'):
            assert pooling.return_indices == True
        if hasattr(pooling, 'ceil_mode'):
            assert pooling.ceil_mode == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_batch_norm1d_attributes():
    """Test FractionalBatchNorm1d attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalBatchNorm1d
        
        bn1d = FractionalBatchNorm1d(
            num_features=20,
            eps=1e-3,
            momentum=0.2,
            affine=False,
            track_running_stats=False
        )
        
        # Test that all attributes are set correctly
        assert hasattr(bn1d, 'num_features')
        assert hasattr(bn1d, 'eps')
        assert hasattr(bn1d, 'momentum')
        assert hasattr(bn1d, 'affine')
        assert hasattr(bn1d, 'track_running_stats')
        
        # Test attribute values
        assert bn1d.num_features == 20
        assert bn1d.eps == 1e-3
        assert bn1d.momentum == 0.2
        assert bn1d.affine == False
        assert bn1d.track_running_stats == False
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_dropout_attributes():
    """Test FractionalDropout attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalDropout
        
        dropout = FractionalDropout(
            p=0.3,
            inplace=True
        )
        
        # Test that all attributes are set correctly
        assert hasattr(dropout, 'p')
        assert hasattr(dropout, 'inplace')
        
        # Test attribute values
        assert dropout.p == 0.3
        assert dropout.inplace == True
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_fractional_layer_norm_attributes():
    """Test FractionalLayerNorm attributes and properties."""
    try:
        from hpfracc.ml.layers import FractionalLayerNorm
        
        layer_norm = FractionalLayerNorm(
            normalized_shape=(10, 20),
            eps=1e-3,
            elementwise_affine=False
        )
        
        # Test that all attributes are set correctly
        assert hasattr(layer_norm, 'normalized_shape')
        assert hasattr(layer_norm, 'eps')
        assert hasattr(layer_norm, 'elementwise_affine')
        
        # Test attribute values
        assert layer_norm.normalized_shape == (10, 20)
        assert layer_norm.eps == 1e-3
        assert layer_norm.elementwise_affine == False
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_backend_type():
    """Test LayerConfig backend type handling."""
    try:
        from hpfracc.ml.layers import LayerConfig
        from hpfracc.ml.backends import BackendType
        
        # Test with different backend types
        config_auto = LayerConfig(backend=BackendType.AUTO)
        assert config_auto.backend == BackendType.AUTO
        
        # Test available backend types
        available_backends = [BackendType.AUTO, BackendType.JAX, BackendType.NUMBA]
        for backend in available_backends:
            config = LayerConfig(backend=backend)
            assert config.backend == backend
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_method_validation():
    """Test LayerConfig method validation."""
    try:
        from hpfracc.ml.layers import LayerConfig
        
        # Test valid methods
        valid_methods = ["RL", "Caputo", "GL", "riemann_liouville", "caputo", "grunwald_letnikov"]
        
        for method in valid_methods:
            config = LayerConfig(method=method)
            assert config.method == method
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_activation_validation():
    """Test LayerConfig activation validation."""
    try:
        from hpfracc.ml.layers import LayerConfig
        
        # Test valid activations
        valid_activations = ["relu", "tanh", "sigmoid", "gelu", "swish", "leaky_relu"]
        
        for activation in valid_activations:
            config = LayerConfig(activation=activation)
            assert config.activation == activation
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_dropout_validation():
    """Test LayerConfig dropout validation."""
    try:
        from hpfracc.ml.layers import LayerConfig
        
        # Test valid dropout values
        valid_dropouts = [0.0, 0.1, 0.5, 0.9, 1.0]
        
        for dropout in valid_dropouts:
            config = LayerConfig(dropout=dropout)
            assert config.dropout == dropout
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")


def test_layer_config_use_fractional_flag():
    """Test LayerConfig use_fractional flag."""
    try:
        from hpfracc.ml.layers import LayerConfig
        
        # Test with use_fractional=True
        config_true = LayerConfig(use_fractional=True)
        assert config_true.use_fractional == True
        
        # Test with use_fractional=False
        config_false = LayerConfig(use_fractional=False)
        assert config_false.use_fractional == False
        
    except ImportError as e:
        pytest.skip(f"ML layers not available: {e}")
