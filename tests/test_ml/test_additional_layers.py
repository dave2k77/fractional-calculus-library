import pytest

from hpfracc.ml.layers import FractionalDropout, LayerConfig
from hpfracc.ml.backends import BackendType, get_backend_manager


class TestFractionalDropout:
    def test_fractional_dropout_noop_in_eval(self):
        import torch
        x = torch.ones(2, 3)
        layer = FractionalDropout(p=0.5, config=LayerConfig(), backend=BackendType.TORCH)
        out = layer.forward(x, training=False)
        assert out is x or out.equal(x)

    def test_fractional_dropout_reduces_magnitude_on_average(self):
        import torch
        torch.manual_seed(0)
        x = torch.ones(1000, 10)
        layer = FractionalDropout(p=0.2, config=LayerConfig(), backend=BackendType.TORCH)
        out = layer.forward(x, training=True)
        # After dropout with p=0.2 and scaling by 1/(1-p), mean should be close to 1.0
        assert abs(out.mean().item() - 1.0) < 0.1

