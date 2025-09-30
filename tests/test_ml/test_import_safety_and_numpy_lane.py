#!/usr/bin/env python3
"""Import-safety and NumPy-lane basic tests for Phase 2 architecture."""

import os
import sys
import importlib
import types


def test_package_import_is_lightweight(monkeypatch):
    """Importing hpfracc should not import torch or jax implicitly."""
    # Ensure fresh import state
    for mod in [m for m in list(sys.modules.keys()) if m.startswith('hpfracc') or m in ('torch','jax')]:
        sys.modules.pop(mod, None)

    importlib.invalidate_caches()

    import hpfracc  # noqa: F401

    assert 'torch' not in sys.modules, "hpfracc import should not load torch"
    assert 'jax' not in sys.modules, "hpfracc import should not load jax"


def test_ml_backends_import_is_lazy(monkeypatch):
    """Importing backends submodule should not import torch eagerly."""
    for mod in [m for m in list(sys.modules.keys()) if m.startswith('hpfracc') or m in ('torch','jax')]:
        sys.modules.pop(mod, None)

    importlib.invalidate_caches()

    importlib.import_module('hpfracc.ml.backends')  # noqa: F401

    assert 'torch' not in sys.modules, "backends import should be lazy for torch"


def test_tensor_ops_numpy_lane(monkeypatch):
    """With Torch/JAX disabled via env, TensorOps should fall back to NumPy lane."""
    monkeypatch.setenv('HPFRACC_DISABLE_TORCH', '1')
    monkeypatch.setenv('HPFRACC_DISABLE_JAX', '1')

    for mod in [m for m in list(sys.modules.keys()) if m.startswith('hpfracc') or m in ('torch','jax')]:
        sys.modules.pop(mod, None)

    importlib.invalidate_caches()

    from hpfracc.ml.tensor_ops import TensorOps, BackendType
    ops = TensorOps()

    assert ops.backend == BackendType.NUMBA

    x = ops.ones((2, 3))
    y = ops.zeros((2, 3))
    s = ops.sum(ops.add(x, y))
    assert float(s) == 6.0


