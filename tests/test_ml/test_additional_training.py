import pytest

from hpfracc.ml.training import FractionalCyclicLR
from hpfracc.ml.backends import BackendType


class MockOptimizer:
    def __init__(self, lr=1e-3):
        self.param_groups = [{'lr': lr}]
    def zero_grad(self):
        pass
    def step(self):
        pass


class TestFractionalCyclicLR:
    def test_cyclic_lr_increases_then_decreases(self):
        opt = MockOptimizer(lr=1e-4)
        sched = FractionalCyclicLR(opt, base_lr=1e-4, max_lr=1e-3, step_size_up=2, step_size_down=2)

        lrs = []
        for _ in range(6):
            sched.step()
            lrs.append(opt.param_groups[0]['lr'])

        # Should move up from ~1e-4 towards ~1e-3 and then back down
        assert lrs[1] >= lrs[0]
        assert lrs[2] >= lrs[1]
        assert lrs[3] <= lrs[2]

