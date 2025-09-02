import pytest

from hpfracc.ml.layers import FractionalLayerNorm, LayerConfig
from hpfracc.core.definitions import FractionalOrder
from hpfracc.ml.backends import BackendType


def test_fractional_layernorm_torch_shapes_and_grad():
    import torch
    x = torch.randn(4, 8, requires_grad=True)
    cfg = LayerConfig(fractional_order=FractionalOrder(0.5), use_fractional=True)
    ln = FractionalLayerNorm(normalized_shape=8, config=cfg, backend=BackendType.TORCH)
    y = ln(x)
    assert y.shape == x.shape
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_fractional_layernorm_affine_off():
    import torch
    x = torch.randn(2, 6)
    ln = FractionalLayerNorm(normalized_shape=6, elementwise_affine=False, backend=BackendType.TORCH)
    y = ln(x)
    assert y.shape == x.shape


def test_fractional_layernorm_stats_unit_variance_mean_zero():
    import torch
    torch.manual_seed(0)
    x = torch.randn(16, 10)
    ln = FractionalLayerNorm(normalized_shape=10, backend=BackendType.TORCH)
    y = ln(x)
    m = y.mean(dim=-1)
    v = y.var(dim=-1, unbiased=False)
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-5)
    assert torch.all(v > 0.9) and torch.all(v < 1.1)
