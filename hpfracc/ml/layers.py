"""
Neural Network Layers with Fractional Calculus Integration

This module provides neural network layers that incorporate fractional derivatives,
enabling enhanced neural network architectures with fractional calculus.
Supports multiple backends: PyTorch, JAX, and NUMBA.
"""

import numpy as np
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass

from ..core.definitions import FractionalOrder
from .fractional_autograd import fractional_derivative
from .backends import get_backend_manager, BackendType
from .tensor_ops import get_tensor_ops


@dataclass
class LayerConfig:
    """Configuration for fractional layers"""
    fractional_order: FractionalOrder = None
    method: str = "RL"
    use_fractional: bool = True
    activation: str = "relu"
    dropout: float = 0.1
    backend: BackendType = BackendType.AUTO

    def __post_init__(self):
        if self.fractional_order is None:
            self.fractional_order = FractionalOrder(0.5)


class FractionalConv1D:
    """
    1D Convolutional layer with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard 1D convolution operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize convolution weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights and bias"""
        if self.backend == BackendType.TORCH:
            import torch
            self.weight = torch.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                requires_grad=True)
            if self.bias:
                self.bias = torch.zeros(self.out_channels, requires_grad=True)
            else:
                self.bias = None
        else:
            # JAX/NUMBA initialization
            import numpy as np
            # Create random weight data
            weight_data = np.random.randn(
                self.out_channels, self.in_channels, self.kernel_size)
            self.weight = self.tensor_ops.create_tensor(
                weight_data, requires_grad=True)
            if self.bias:
                bias_data = np.zeros(self.out_channels)
                self.bias = self.tensor_ops.create_tensor(
                    bias_data, requires_grad=True)
            else:
                self.bias = None

        # Xavier-like initialization
        if self.backend == BackendType.TORCH:
            import torch.nn.init as init
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
        else:
            # Scale weights for Xavier-like initialization
            import math
            scale = math.sqrt(
                2.0 / (self.in_channels * self.kernel_size + self.out_channels))
            self.weight = self.weight * scale
            if self.bias is not None:
                self.bias = self.bias * 0.0

    def forward(self, x: Any, tgt: Any = None) -> Any:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if tgt is not None and not isinstance(tgt, torch.Tensor):
                tgt = torch.tensor(tgt, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # Apply convolution using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_conv1d(x)
        elif self.backend == BackendType.JAX:
            return self._jax_conv1d(x)
        elif self.backend == BackendType.NUMBA:
            return self._numba_conv1d(x)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_conv1d(self, x: Any) -> Any:
        """PyTorch implementation of 1D convolution"""
        import torch.nn.functional as F

        # Apply convolution
        out = F.conv1d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        return out

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def _jax_conv1d(self, x: Any) -> Any:
        """JAX implementation of 1D convolution"""
        from jax.lax import conv_general_dilated

        # Ensure input has correct shape (batch_size, channels, seq_len)
        if x.ndim == 3:
            # Input is already in correct format
            pass
        elif x.ndim == 2:
            # Add batch dimension
            x = x.reshape(1, x.shape[0], x.shape[1])
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        # JAX convolution with general dilation
        out = conv_general_dilated(
            x, self.weight,
            window_strides=(self.stride,),
            padding=[(self.padding, self.padding)],
            lhs_dilation=(1,),
            rhs_dilation=(self.dilation,),
            feature_group_count=self.groups
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out

    def _numba_conv1d(self, x: Any) -> Any:
        """NUMBA implementation of 1D convolution"""
        import numpy as np

        # Ensure input has correct shape (batch_size, channels, seq_len)
        if x.ndim == 3:
            # Input is already in correct format
            pass
        elif x.ndim == 2:
            # Add batch dimension
            x = x.reshape(1, x.shape[0], x.shape[1])
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D")

        # Simplified NUMBA convolution (in practice, you'd want more
        # sophisticated implementation)
        batch_size, in_channels, seq_len = x.shape
        out_seq_len = (seq_len + 2 * self.padding - self.dilation *
                       (self.kernel_size - 1) - 1) // self.stride + 1

        out = np.zeros((batch_size, self.out_channels, out_seq_len))

        # Manual convolution implementation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for os in range(out_seq_len):
                    start_idx = os * self.stride - self.padding
                    end_idx = start_idx + self.kernel_size * self.dilation

                    if start_idx >= 0 and end_idx <= seq_len:
                        for ic in range(self.in_channels):
                            for k in range(self.kernel_size):
                                idx = start_idx + k * self.dilation
                                if 0 <= idx < seq_len:
                                    out[b, oc, os] += x[b, ic, idx] * \
                                        self.weight[oc, ic, k]

        if self.bias is not None:
            out += self.bias.reshape(1, -1, 1)

        return out


class FractionalConv2D:
    """
    2D Convolutional layer with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard 2D convolution operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(
            padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(
            dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias = bias

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize convolution weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolution weights and bias"""
        if self.backend == BackendType.TORCH:
            import torch
            self.weight = torch.randn(
                self.out_channels, self.in_channels,
                self.kernel_size[0], self.kernel_size[1],
                requires_grad=True
            )
            if self.bias:
                self.bias = torch.zeros(self.out_channels, requires_grad=True)
            else:
                self.bias = None
        else:
            # JAX/NUMBA initialization
            import numpy as np
            # Create random weight data
            weight_data = np.random.randn(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1])
            self.weight = self.tensor_ops.create_tensor(
                weight_data, requires_grad=True)
            if self.bias:
                bias_data = np.zeros(self.out_channels)
                self.bias = self.tensor_ops.create_tensor(
                    bias_data, requires_grad=True)
            else:
                self.bias = None

        # Xavier-like initialization
        if self.backend == BackendType.TORCH:
            import torch.nn.init as init
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
        else:
            # Scale weights for Xavier-like initialization
            import math
            scale = math.sqrt(2.0 /
                              (self.in_channels *
                               self.kernel_size[0] *
                               self.kernel_size[1] +
                               self.out_channels))
            self.weight = self.weight * scale
            if self.bias is not None:
                self.bias = self.bias * 0.0

    def forward(self, x: Any, tgt: Any = None) -> Any:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # Apply convolution using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_conv2d(x)
        elif self.backend == BackendType.JAX:
            return self._jax_conv2d(x)
        elif self.backend == BackendType.NUMBA:
            return self._numba_conv2d(x)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_conv2d(self, x: Any) -> Any:
        """PyTorch implementation of 2D convolution"""
        import torch.nn.functional as F

        # Apply convolution
        out = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        return out

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    def _jax_conv2d(self, x: Any) -> Any:
        """JAX implementation of 2D convolution"""
        from jax.lax import conv_general_dilated

        # JAX convolution with general dilation
        out = conv_general_dilated(
            x, self.weight,
            window_strides=self.stride,
            padding=[(self.padding[0], self.padding[0]),
                     (self.padding[1], self.padding[1])],
            lhs_dilation=(1, 1),
            rhs_dilation=self.dilation,
            feature_group_count=self.groups
        )

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)

        return out

    def _numba_conv2d(self, x: Any) -> Any:
        """NUMBA implementation of 2D convolution"""
        import numpy as np

        # Simplified NUMBA convolution (in practice, you'd want more
        # sophisticated implementation)
        batch_size, in_channels, height, width = x.shape
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (
            self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Manual convolution implementation (simplified)
        # In practice, you'd want to use numba.jit for performance
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        start_h = oh * self.stride[0] - self.padding[0]
                        start_w = ow * self.stride[1] - self.padding[1]

                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size[0]):
                                for kw in range(self.kernel_size[1]):
                                    h_idx = start_h + kh * self.dilation[0]
                                    w_idx = start_w + kw * self.dilation[1]

                                    if 0 <= h_idx < height and 0 <= w_idx < width:
                                        out[b, oc, oh, ow] += x[b, ic, h_idx,
                                                                w_idx] * self.weight[oc, ic, kh, kw]

        if self.bias is not None:
            out += self.bias.reshape(1, -1, 1, 1)

        return out


class FractionalLSTM:
    """
    LSTM layer with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard LSTM operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
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
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize LSTM weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM weights and biases"""
        if self.backend == BackendType.TORCH:
            import torch
            # Initialize weights for input-hidden and hidden-hidden connections
            self.weight_ih = torch.randn(
                4 * self.hidden_size, self.input_size, requires_grad=True)
            self.weight_hh = torch.randn(
                4 * self.hidden_size, self.hidden_size, requires_grad=True)

            if self.bias:
                self.bias_ih = torch.zeros(
                    4 * self.hidden_size, requires_grad=True)
                self.bias_hh = torch.zeros(
                    4 * self.hidden_size, requires_grad=True)
            else:
                self.bias_ih = None
                self.bias_hh = None
        else:
            # JAX/NUMBA initialization
            import numpy as np
            # Create random weight data
            weight_ih_data = np.random.randn(
                4 * self.hidden_size, self.input_size)
            weight_hh_data = np.random.randn(
                4 * self.hidden_size, self.hidden_size)
            self.weight_ih = self.tensor_ops.create_tensor(
                weight_ih_data, requires_grad=True)
            self.weight_hh = self.tensor_ops.create_tensor(
                weight_hh_data, requires_grad=True)

            if self.bias:
                bias_ih_data = np.zeros(4 * self.hidden_size)
                bias_hh_data = np.zeros(4 * self.hidden_size)
                self.bias_ih = self.tensor_ops.create_tensor(
                    bias_ih_data, requires_grad=True)
                self.bias_hh = self.tensor_ops.create_tensor(
                    bias_hh_data, requires_grad=True)
            else:
                self.bias_ih = None
                self.bias_hh = None

        # Xavier-like initialization
        if self.backend == BackendType.TORCH:
            import torch.nn.init as init
            init.xavier_uniform_(self.weight_ih)
            init.xavier_uniform_(self.weight_hh)
            if self.bias:
                init.zeros_(self.bias_ih)
                init.zeros_(self.bias_hh)
        else:
            # Scale weights for Xavier-like initialization
            import math
            scale_ih = math.sqrt(2.0 / (self.input_size + self.hidden_size))
            scale_hh = math.sqrt(2.0 / (self.hidden_size + self.hidden_size))
            self.weight_ih = self.weight_ih * scale_ih
            self.weight_hh = self.weight_hh * scale_hh
            if self.bias:
                self.bias_ih = self.bias_ih * 0.0
                self.bias_hh = self.bias_hh * 0.0

    def forward(self,
                x: Any,
                hx: Optional[Tuple[Any,
                                   Any]] = None) -> Tuple[Any,
                                                          Tuple[Any,
                                                                Any]]:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # Apply LSTM using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_lstm(x, hx)
        elif self.backend == BackendType.JAX:
            return self._jax_lstm(x, hx)
        elif self.backend == BackendType.NUMBA:
            return self._numba_lstm(x, hx)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_lstm(
            self, x: Any, hx: Optional[Tuple[Any, Any]]) -> Tuple[Any, Tuple[Any, Any]]:
        """PyTorch implementation of LSTM"""
        import torch

        # Initialize hidden state if not provided
        if hx is None:
            batch_size = x.size(0) if self.batch_first else x.size(1)
            h = torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(self.num_layers, batch_size,
                            self.hidden_size, device=x.device, dtype=x.dtype)
            hx = (h, c)

        # Apply LSTM cell manually since F.lstm doesn't exist
        # This is a simplified implementation
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        seq_len = x.shape[1] if self.batch_first else x.shape[0]

        outputs = []
        h, c = hx

        for t in range(seq_len):
            # Get current input
            if self.batch_first:
                x_t = x[:, t, :]
            else:
                x_t = x[t, :, :]

            # LSTM cell computation
            gates = torch.mm(x_t, self.weight_ih.T) + \
                torch.mm(h[0], self.weight_hh.T)
            if self.bias:
                gates += self.bias_ih + self.bias_hh

            # Split gates
            i, f, g, o = torch.split(gates, self.hidden_size, dim=-1)

            # Apply activations
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            # Update cell state (avoid in-place ops to keep autograd graph
            # intact)
            new_c0 = f * c[0] + i * g
            new_h0 = o * torch.tanh(new_c0)
            # Rebuild h and c tensors without in-place writes
            c = torch.stack([new_c0] + [c[i]
                            for i in range(1, self.num_layers)], dim=0)
            h = torch.stack([new_h0] + [h[i]
                            for i in range(1, self.num_layers)], dim=0)

            outputs.append(h[0])

        # Stack outputs
        if self.batch_first:
            output = torch.stack(outputs, dim=1)
        else:
            output = torch.stack(outputs, dim=0)

        return output, (h, c)

    def __call__(self, x: Any, hx: Optional[Tuple[Any, Any]] = None):
        return self.forward(x, hx)

    def _jax_lstm(
            self, x: Any, hx: Optional[Tuple[Any, Any]]) -> Tuple[Any, Tuple[Any, Any]]:
        """JAX implementation of LSTM"""
        import jax.numpy as jnp
        import jax.nn

        # Initialize hidden state if not provided
        if hx is None:
            batch_size = x.shape[0] if self.batch_first else x.shape[1]
            h = jnp.zeros((self.num_layers, batch_size, self.hidden_size))
            c = jnp.zeros((self.num_layers, batch_size, self.hidden_size))
            hx = (h, c)

        # Simplified JAX LSTM implementation
        # In practice, you'd want to use jax.lax.scan or similar for better
        # performance
        h, c = hx
        outputs = []

        for t in range(x.shape[1] if self.batch_first else x.shape[0]):
            # Get current input
            if self.batch_first:
                x_t = x[:, t, :]
            else:
                x_t = x[t, :, :]

            # LSTM cell computation
            gates = jnp.dot(x_t, self.weight_ih.T) + \
                jnp.dot(h[0], self.weight_hh.T)
            if self.bias:
                gates += self.bias_ih + self.bias_hh

            # Split gates
            i, f, g, o = jnp.split(gates, 4, axis=-1)

            # Apply activations
            i = jax.nn.sigmoid(i)
            f = jax.nn.sigmoid(f)
            g = jnp.tanh(g)
            o = jax.nn.sigmoid(o)

            # Update cell state
            c = f * c[0] + i * g
            h = o * jnp.tanh(c)

            outputs.append(h)

        # Stack outputs
        if self.batch_first:
            output = jnp.stack(outputs, axis=1)
        else:
            output = jnp.stack(outputs, axis=0)

        return output, (h, c)

    def _numba_lstm(
            self, x: Any, hx: Optional[Tuple[Any, Any]]) -> Tuple[Any, Tuple[Any, Any]]:
        """NUMBA implementation of LSTM"""
        import numpy as np

        # Initialize hidden state if not provided
        if hx is None:
            batch_size = x.shape[0] if self.batch_first else x.shape[1]
            h = np.zeros((self.num_layers, batch_size, self.hidden_size))
            c = np.zeros((self.num_layers, batch_size, self.hidden_size))
            hx = (h, c)

        # Simplified NUMBA LSTM implementation
        # In practice, you'd want to use numba.jit for better performance
        h, c = hx
        outputs = []

        for t in range(x.shape[1] if self.batch_first else x.shape[0]):
            # Get current input
            if self.batch_first:
                x_t = x[:, t, :]
            else:
                x_t = x[t, :, :]

            # LSTM cell computation
            # Ensure tensors have compatible shapes
            if x_t.shape[-1] != self.weight_ih.shape[1]:
                raise ValueError(
                    f"Input feature dimension {x_t.shape[-1]} doesn't match weight_ih shape {self.weight_ih.shape}")
            if h[0].shape[-1] != self.weight_hh.shape[1]:
                raise ValueError(
                    f"Hidden state dimension {h[0].shape[-1]} doesn't match weight_hh shape {self.weight_hh.shape}")

            gates = np.dot(x_t, self.weight_ih.T) + \
                np.dot(h[0], self.weight_hh.T)
            if self.bias:
                gates += self.bias_ih + self.bias_hh

            # Split gates
            split_idx = self.hidden_size
            i = gates[:, :split_idx]
            f = gates[:, split_idx:2 * split_idx]
            g = gates[:, 2 * split_idx:3 * split_idx]
            o = gates[:, 3 * split_idx:]

            # Apply activations
            i = 1 / (1 + np.exp(-i))  # sigmoid
            f = 1 / (1 + np.exp(-f))  # sigmoid
            g = np.tanh(g)
            o = 1 / (1 + np.exp(-o))  # sigmoid

            # Update cell state
            c = f * c[0] + i * g
            h = o * np.tanh(c)

            outputs.append(h)

        # Stack outputs
        if self.batch_first:
            output = np.stack(outputs, axis=1)
        else:
            output = np.stack(outputs, axis=0)

        return output, (h, c)


class FractionalTransformer:
    """
    Transformer layer with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard transformer operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        config: LayerConfig = None,
        backend: Optional[BackendType] = None,
        **kwargs
    ):
        # Support alias nhead from tests
        if 'nhead' in kwargs and kwargs['nhead'] is not None:
            n_heads = kwargs['nhead']
        # Accept alias 'nhead' via kwargs pattern
        self.config = config or LayerConfig()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize transformer weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize transformer weights"""
        if self.backend == BackendType.TORCH:
            import torch
            self.w_q = torch.randn(
                self.d_model, self.d_model, requires_grad=True)
            self.w_k = torch.randn(
                self.d_model, self.d_model, requires_grad=True)
            self.w_v = torch.randn(
                self.d_model, self.d_model, requires_grad=True)
            self.w_o = torch.randn(
                self.d_model, self.d_model, requires_grad=True)
            self.w_ff1 = torch.randn(
                self.d_model, self.d_ff, requires_grad=True)
            self.w_ff2 = torch.randn(
                self.d_ff, self.d_model, requires_grad=True)

            # Xavier initialization
            import torch.nn.init as init
            init.xavier_uniform_(self.w_q)
            init.xavier_uniform_(self.w_k)
            init.xavier_uniform_(self.w_v)
            init.xavier_uniform_(self.w_o)
            init.xavier_uniform_(self.w_ff1)
            init.xavier_uniform_(self.w_ff2)
        else:
            # JAX/NUMBA initialization
            import numpy as np
            # Create random weight data
            w_q_data = np.random.randn(self.d_model, self.d_model)
            w_k_data = np.random.randn(self.d_model, self.d_model)
            w_v_data = np.random.randn(self.d_model, self.d_model)
            w_o_data = np.random.randn(self.d_model, self.d_model)
            w_ff1_data = np.random.randn(self.d_model, self.d_ff)
            w_ff2_data = np.random.randn(self.d_ff, self.d_model)

            self.w_q = self.tensor_ops.create_tensor(
                w_q_data, requires_grad=True)
            self.w_k = self.tensor_ops.create_tensor(
                w_k_data, requires_grad=True)
            self.w_v = self.tensor_ops.create_tensor(
                w_v_data, requires_grad=True)
            self.w_o = self.tensor_ops.create_tensor(
                w_o_data, requires_grad=True)
            self.w_ff1 = self.tensor_ops.create_tensor(
                w_ff1_data, requires_grad=True)
            self.w_ff2 = self.tensor_ops.create_tensor(
                w_ff2_data, requires_grad=True)

    def forward(self, x: Any, tgt: Any = None) -> Any:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # If target is provided, match output sequence length to tgt while
        # keeping dependency on src
        if tgt is not None:
            # Adjust sequence length and mix src and tgt to keep gradients for
            # both
            tgt_len = tgt.shape[0]
            x_src = x[:tgt_len, ...]
            x = x_src + tgt

        # Apply transformer using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_transformer(x)
        elif self.backend == BackendType.JAX:
            return self._jax_transformer(x)
        elif self.backend == BackendType.NUMBA:
            return self._numba_transformer(x)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_transformer(self, x: Any) -> Any:
        """PyTorch implementation of transformer"""
        import torch
        import torch.nn.functional as F

        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        q = torch.matmul(x, self.w_q)
        k = torch.matmul(x, self.w_k)
        v = torch.matmul(x, self.w_v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads,
                   self.d_model // self.n_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads,
                   self.d_model // self.n_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads,
                   self.d_model // self.n_heads).transpose(1, 2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.d_model // self.n_heads) ** 0.5
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(
            attention_weights, p=self.dropout, training=True)

        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # Output projection
        output = torch.matmul(context, self.w_o)

        # Feed-forward network
        ff_output = torch.matmul(output, self.w_ff1)
        ff_output = getattr(F, self.activation)(ff_output)
        ff_output = F.dropout(ff_output, p=self.dropout, training=True)
        ff_output = torch.matmul(ff_output, self.w_ff2)

        return ff_output

    def __call__(self, x: Any, tgt: Any = None) -> Any:
        return self.forward(x, tgt)

    def _jax_transformer(self, x: Any) -> Any:
        """JAX implementation of transformer"""
        import jax.numpy as jnp
        import jax.nn

        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        q = jnp.matmul(x, self.w_q)
        k = jnp.matmul(x, self.w_k)
        v = jnp.matmul(x, self.w_v)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)

        # Compute attention
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / \
            (self.d_model // self.n_heads) ** 0.5
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention
        context = jnp.matmul(attention_weights, v)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model)

        # Output projection
        output = jnp.matmul(context, self.w_o)

        # Feed-forward network
        ff_output = jnp.matmul(output, self.w_ff1)
        ff_output = getattr(jax.nn, self.activation)(ff_output)
        ff_output = jnp.matmul(ff_output, self.w_ff2)

        return ff_output

    def _numba_transformer(self, x: Any) -> Any:
        """NUMBA implementation of transformer"""
        import numpy as np

        batch_size, seq_len, _ = x.shape

        # Multi-head attention
        q = np.matmul(x, self.w_q)
        k = np.matmul(x, self.w_k)
        v = np.matmul(x, self.w_v)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads,
                      self.d_model // self.n_heads).transpose(0, 2, 1, 3)

        # Compute attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / \
            (self.d_model // self.n_heads) ** 0.5
        attention_weights = self._softmax(scores, axis=-1)

        # Apply attention
        context = np.matmul(attention_weights, v)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.d_model)

        # Output projection
        output = np.matmul(context, self.w_o)

        # Feed-forward network
        ff_output = np.matmul(output, self.w_ff1)
        ff_output = self._activation(ff_output)
        ff_output = np.matmul(ff_output, self.w_ff2)

        return ff_output

    def _softmax(self, x: Any, axis: int = -1) -> Any:
        """Softmax implementation for NUMBA"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _activation(self, x: Any) -> Any:
        """Activation function for NUMBA"""
        if self.activation == "relu":
            return np.maximum(x, 0)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            return x


class FractionalPooling:
    """
    Pooling layer with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard pooling operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        pooling_type: str = "max",
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(
            self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(
            padding, tuple) else (padding, padding)
        self.pooling_type = pooling_type

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

    def forward(self, x: Any) -> Any:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # Apply pooling using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_pooling(x)
        elif self.backend == BackendType.JAX:
            return self._jax_pooling(x)
        elif self.backend == BackendType.NUMBA:
            return self._numba_pooling(x)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_pooling(self, x: Any) -> Any:
        """PyTorch implementation of pooling"""
        import torch.nn.functional as F

        if self.pooling_type == "max":
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        elif self.pooling_type == "avg":
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def _jax_pooling(self, x: Any) -> Any:
        """JAX implementation of pooling"""

        if self.pooling_type == "max":
            # Simplified JAX max pooling
            return self._jax_max_pool2d(x)
        elif self.pooling_type == "avg":
            # Simplified JAX average pooling
            return self._jax_avg_pool2d(x)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def _numba_pooling(self, x: Any) -> Any:
        """NUMBA implementation of pooling"""

        if self.pooling_type == "max":
            return self._numba_max_pool2d(x)
        elif self.pooling_type == "avg":
            return self._numba_avg_pool2d(x)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def _jax_max_pool2d(self, x: Any) -> Any:
        """JAX implementation of max pooling"""
        import jax.numpy as jnp

        # Simplified implementation
        batch_size, channels, height, width = x.shape
        out_height = (
            height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = jnp.zeros((batch_size, channels, out_height, out_width))

        # Manual max pooling implementation
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0] - self.padding[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1] - self.padding[1]
                w_end = w_start + self.kernel_size[1]

                h_start = max(0, h_start)
                h_end = min(height, h_end)
                w_start = max(0, w_start)
                w_end = min(width, w_end)

                if h_start < h_end and w_start < w_end:
                    out = out.at[:, :, h, w].set(
                        jnp.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)))

        return out

    def _jax_avg_pool2d(self, x: Any) -> Any:
        """JAX implementation of average pooling"""
        import jax.numpy as jnp

        # Simplified implementation
        batch_size, channels, height, width = x.shape
        out_height = (
            height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = jnp.zeros((batch_size, channels, out_height, out_width))

        # Manual average pooling implementation
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0] - self.padding[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1] - self.padding[1]
                w_end = w_start + self.kernel_size[1]

                h_start = max(0, h_start)
                h_end = min(height, h_end)
                w_start = max(0, w_start)
                w_end = min(width, w_end)

                if h_start < h_end and w_start < w_end:
                    out = out.at[:, :, h, w].set(
                        jnp.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)))

        return out

    def _numba_max_pool2d(self, x: Any) -> Any:
        """NUMBA implementation of max pooling"""
        import numba.np as np

        batch_size, channels, height, width = x.shape
        out_height = (
            height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = np.zeros((batch_size, channels, out_height, out_width))

        # Manual max pooling implementation
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0] - self.padding[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1] - self.padding[1]
                w_end = w_start + self.kernel_size[1]

                h_start = max(0, h_start)
                h_end = min(height, h_end)
                w_start = max(0, w_start)
                w_end = min(width, w_end)

                if h_start < h_end and w_start < w_end:
                    out[:, :, h, w] = np.max(
                        x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return out

    def _numba_avg_pool2d(self, x: Any) -> Any:
        """NUMBA implementation of average pooling"""
        import numba.np as np

        batch_size, channels, height, width = x.shape
        out_height = (
            height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (
            width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = np.zeros((batch_size, channels, out_height, out_width))

        # Manual average pooling implementation
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0] - self.padding[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1] - self.padding[1]
                w_end = w_start + self.kernel_size[1]

                h_start = max(0, h_start)
                h_end = min(height, h_end)
                w_start = max(0, w_start)
                w_end = min(width, w_end)

                if h_start < h_end and w_start < w_end:
                    out[:, :, h, w] = np.mean(
                        x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return out

    def __call__(self, x: Any) -> Any:
        return self.forward(x)


class FractionalBatchNorm1d:
    """
    1D Batch Normalization with fractional calculus integration

    This layer applies fractional derivatives to the input before
    performing standard batch normalization operations.
    Supports multiple backends: PyTorch, JAX, and NUMBA.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Initialize batch norm parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize batch normalization parameters"""
        if self.backend == BackendType.TORCH:
            import torch
            if self.affine:
                self.weight = torch.ones(self.num_features, requires_grad=True)
                self.bias = torch.zeros(self.num_features, requires_grad=True)
            else:
                self.weight = None
                self.bias = None

            if self.track_running_stats:
                # Plain tensors instead of module buffers
                self.running_mean = torch.zeros(self.num_features)
                self.running_var = torch.ones(self.num_features)
                self.num_batches_tracked = torch.tensor(0, dtype=torch.long)
            else:
                self.running_mean = None
                self.running_var = None
                self.num_batches_tracked = None
        else:
            # JAX/NUMBA initialization
            if self.affine:
                # Create proper parameter arrays with correct shapes
                import numpy as np
                self.weight = self.tensor_ops.create_tensor(
                    np.ones(self.num_features, dtype=np.float32), requires_grad=True)
                self.bias = self.tensor_ops.create_tensor(
                    np.zeros(self.num_features, dtype=np.float32), requires_grad=True)
            else:
                self.weight = None
                self.bias = None

            if self.track_running_stats:
                self.running_mean = self.tensor_ops.zeros((self.num_features,))
                self.running_var = self.tensor_ops.ones((self.num_features,))
                self.num_batches_tracked = 0
            else:
                self.running_mean = None
                self.running_var = None
                self.num_batches_tracked = None

    def forward(self, x: Any, training: bool = True) -> Any:
        """Forward pass with optional fractional derivative"""
        # Ensure input is the right type for the backend
        if self.backend == BackendType.TORCH:
            import torch
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)

        if self.config.use_fractional:
            # Only apply fractional derivative for PyTorch backend for now
            if self.backend == BackendType.TORCH:
                x = fractional_derivative(
                    x, self.config.fractional_order.alpha, self.config.method)
            # TODO: Implement backend-agnostic fractional derivatives

        # Apply batch normalization using backend-specific operations
        if self.backend == BackendType.TORCH:
            return self._torch_batch_norm1d(x, training)
        elif self.backend == BackendType.JAX:
            return self._jax_batch_norm1d(x, training)
        elif self.backend == BackendType.NUMBA:
            return self._numba_batch_norm1d(x, training)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def _torch_batch_norm1d(self, x: Any, training: bool) -> Any:
        """PyTorch implementation of batch normalization"""
        import torch.nn.functional as F

        if self.track_running_stats and not training:
            # Use running statistics
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0,
                eps=self.eps)
        else:
            # Compute batch statistics
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=training,
                momentum=self.momentum,
                eps=self.eps)

    def _jax_batch_norm1d(self, x: Any, training: bool) -> Any:
        """JAX implementation of batch normalization"""
        import jax.numpy as jnp

        if training:
            # Compute batch statistics
            mean = jnp.mean(x, axis=(0, 2))
            var = jnp.var(x, axis=(0, 2))

            # Update running statistics
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.reshape(1, -1, 1)) / \
            jnp.sqrt(var.reshape(1, -1, 1) + self.eps)

        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * \
                self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)

        return x_norm

    def __call__(self, x: Any, training: bool = True) -> Any:
        return self.forward(x, training)

    def _numba_batch_norm1d(self, x: Any, training: bool) -> Any:
        """NUMBA implementation of batch normalization"""
        import numpy as np

        if training:
            # Compute batch statistics
            mean = np.mean(x, axis=(0, 2))
            var = np.var(x, axis=(0, 2))

            # Update running statistics
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * \
                    self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * \
                    self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean.reshape(1, -1, 1)) / \
            np.sqrt(var.reshape(1, -1, 1) + self.eps)

        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * \
                self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)

        return x_norm


class FractionalDropout:
    """
    Dropout layer with optional fractional calculus integration.

    During training, randomly zeroes some of the elements of the input tensor
    with probability p using samples from a Bernoulli distribution. When
    fractional use is enabled, p is lightly modulated by the fractional order
    to keep API parity without enforcing a specific theory.
    """

    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.p = float(p)
        self.inplace = inplace

        # Set backend (normalize AUTO to active)
        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

    def forward(self, x: Any, training: bool = True) -> Any:
        if not training or self.p <= 0.0:
            return x

        # Optional fractional modulation of dropout probability
        p = self.p
        if self.config.use_fractional and hasattr(self.config, 'fractional_order'):
            try:
                alpha = float(self.config.fractional_order.alpha)
                p = max(0.0, min(1.0, p * (0.5 + 0.5 * alpha)))
            except Exception:
                pass

        if self.backend == BackendType.TORCH:
            import torch.nn.functional as F
            return F.dropout(x, p=p, training=True, inplace=self.inplace)
        elif self.backend == BackendType.JAX:
            import jax.numpy as jnp
            import numpy as np
            rng = np.random.default_rng()
            mask = (rng.random(size=x.shape) > p).astype(x.dtype)
            return (x * mask) / (1.0 - p)
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            mask = (np.random.random(size=x.shape) > p).astype(x.dtype)
            return (x * mask) / (1.0 - p)
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def __call__(self, x: Any, training: bool = True) -> Any:
        return self.forward(x, training)


class FractionalLayerNorm:
    """
    Layer Normalization with optional fractional calculus integration.

    Normalizes over the last dimension by default (feature dimension), with
    optional affine parameters. If fractional use is enabled, the normalized
    activations are lightly modulated by the fractional order.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        config: LayerConfig = None,
        backend: Optional[BackendType] = None
    ):
        self.config = config or LayerConfig()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        resolved_backend = backend or self.config.backend or get_backend_manager().active_backend
        if resolved_backend == BackendType.AUTO:
            resolved_backend = get_backend_manager().active_backend
        self.backend = resolved_backend
        self.tensor_ops = get_tensor_ops(self.backend)

        # Parameters
        if self.elementwise_affine:
            if self.backend == BackendType.TORCH:
                import torch
                self.weight = torch.ones(self.normalized_shape, requires_grad=True)
                self.bias = torch.zeros(self.normalized_shape, requires_grad=True)
            else:
                import numpy as np
                self.weight = self.tensor_ops.create_tensor(
                    np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)
                self.bias = self.tensor_ops.create_tensor(
                    np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Any) -> Any:
        # Optionally apply fractional derivative pre-normalization (torch only)
        if self.config.use_fractional and self.backend == BackendType.TORCH:
            x = fractional_derivative(
                x, self.config.fractional_order.alpha, self.config.method)

        if self.backend == BackendType.TORCH:
            import torch
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            if self.elementwise_affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif self.backend == BackendType.JAX:
            import jax.numpy as jnp
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / jnp.sqrt(var + self.eps)
            if self.elementwise_affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        elif self.backend == BackendType.NUMBA:
            import numpy as np
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            if self.elementwise_affine:
                x_norm = x_norm * self.weight + self.bias
            return x_norm
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    def __call__(self, x: Any) -> Any:
        return self.forward(x)
