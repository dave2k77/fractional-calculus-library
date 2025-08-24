"""
Neural Network Layers with Fractional Calculus Integration

This module provides PyTorch layers that incorporate fractional derivatives,
enabling enhanced neural network architectures with fractional calculus.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative


@dataclass
class LayerConfig:
    """Configuration for fractional layers"""
    fractional_order: FractionalOrder = None
    method: str = "RL"
    use_fractional: bool = True
    activation: str = "relu"
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.fractional_order is None:
            self.fractional_order = FractionalOrder(0.5)


class FractionalConv1D(nn.Module):
    """
    1D Convolutional layer with fractional calculus integration
    
    This layer applies fractional derivatives to the input before
    performing standard 1D convolution operations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fractional derivative"""
        if self.config.use_fractional:
            x = fractional_derivative(x, self.config.fractional_order.alpha, self.config.method)
        
        return self.conv(x)


class FractionalConv2D(nn.Module):
    """
    2D Convolutional layer with fractional calculus integration
    
    This layer applies fractional derivatives to the input before
    performing standard 2D convolution operations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fractional derivative"""
        if self.config.use_fractional:
            x = fractional_derivative(x, self.config.fractional_order.alpha, self.config.method)
        
        return self.conv(x)


class FractionalLSTM(nn.Module):
    """
    LSTM layer with fractional calculus integration
    
    This layer applies fractional derivatives to the input and hidden states
    before performing standard LSTM operations.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional fractional derivative"""
        if self.config.use_fractional:
            x = fractional_derivative(x, self.config.fractional_order.alpha, self.config.method)
            
            if hx is not None:
                h, c = hx
                h = fractional_derivative(h, self.config.fractional_order.alpha, self.config.method)
                c = fractional_derivative(c, self.config.fractional_order.alpha, self.config.method)
                hx = (h, c)
        
        return self.lstm(x, hx)


class FractionalTransformer(nn.Module):
    """
    Transformer layer with fractional calculus integration
    
    This layer applies fractional derivatives to the input before
    performing standard transformer operations.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        custom_encoder: Optional[nn.Module] = None,
        custom_decoder: Optional[nn.Module] = None,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.d_model = d_model
        self.nhead = nhead
        
        # Create encoder-only transformer for single input processing
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        
        # Keep full transformer for encoder-decoder tasks
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout, activation, custom_encoder, custom_decoder
        )
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional fractional derivative
        
        Args:
            src: Source tensor (required)
            tgt: Target tensor (optional, if None uses encoder-only mode)
            ...: Other transformer arguments
        """
        if self.config.use_fractional:
            src = fractional_derivative(src, self.config.fractional_order.alpha, self.config.method)
            if tgt is not None:
                tgt = fractional_derivative(tgt, self.config.fractional_order.alpha, self.config.method)
        
        # If no target provided, use encoder-only mode
        if tgt is None:
            # Ensure input has correct shape (batch_size, seq_len, d_model) for batch_first=True
            if src.dim() == 2:
                src = src.unsqueeze(0)  # Add batch dimension if missing
            elif src.dim() == 3 and src.size(-1) != self.d_model:
                # If last dimension is not d_model, assume it's (batch, seq, features)
                # and project to d_model
                src = F.linear(src, torch.randn(self.d_model, src.size(-1)).to(src.device))
            
            return self.encoder(src, src_mask, src_key_padding_mask)
        
        # Full transformer mode - PyTorch expects (seq_len, batch_size, d_model) by default
        # Convert from (batch_size, seq_len, d_model) to (seq_len, batch_size, d_model)
        src_transposed = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        tgt_transposed = tgt.transpose(0, 1)  # (seq_len, batch_size, d_model)
        
        return self.transformer(
            src_transposed, tgt_transposed, src_mask, tgt_mask, memory_mask,
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask
        )


class FractionalPooling(nn.Module):
    """
    Pooling layer with fractional calculus integration
    
    This layer applies fractional derivatives to the input before
    performing standard pooling operations.
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional fractional derivative"""
        if self.config.use_fractional:
            x = fractional_derivative(x, self.config.fractional_order.alpha, self.config.method)
        
        return F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding,
            self.dilation, self.return_indices, self.ceil_mode
        )


class FractionalBatchNorm1d(nn.Module):
    """
    1D Batch Normalization with fractional calculus integration
    
    This layer applies fractional derivatives to the input before
    performing standard batch normalization operations.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        config: LayerConfig = None
    ):
        super().__init__()
        self.config = config or LayerConfig()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fractional derivative"""
        if self.config.use_fractional:
            x = fractional_derivative(x, self.config.fractional_order.alpha, self.config.method)
        
        return self.bn(x)
