#!/usr/bin/env python3
"""Adapter capability/contract tests (using NumPy lane only)."""

import importlib

import numpy as np

from hpfracc.ml.adapters import get_optimal_adapter, HighPerformanceAdapter
from hpfracc.ml.backends import BackendType


def test_numpy_adapter_capabilities():
    adapter = HighPerformanceAdapter(BackendType.NUMBA)
    lib = adapter.get_lib()
    assert lib is importlib.import_module('numpy')

    caps = adapter.get_capabilities()
    assert caps.device_kind == 'cpu'
    assert caps.has_fft is True
    assert caps.has_autograd is False
    assert caps.supports_amp is False
    assert caps.supports_jit is False


def test_detect_capabilities_numpy():
    adapter = HighPerformanceAdapter(BackendType.NUMBA)
    caps = adapter.get_capabilities()
    assert caps.device_kind == 'cpu'


def test_numpy_adapter_tensor_ops():
    adapter = HighPerformanceAdapter(BackendType.NUMBA)
    lib = adapter.get_lib()
    arr = lib.ones((2, 3))
    assert lib.sum(arr) == 6
    assert isinstance(arr, np.ndarray)

