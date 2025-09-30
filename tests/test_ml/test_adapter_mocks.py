#!/usr/bin/env python3
"""Adapter tests with mocked torch/jax imports."""

import sys
import types

import pytest

import importlib.util

from hpfracc.ml import adapters
from hpfracc.ml.backends import BackendType


@pytest.fixture
def fake_torch(monkeypatch):
    fake = types.SimpleNamespace(
        softmax=lambda x, dim=-1: ('torch_softmax', dim),
        relu=lambda x: ('torch_relu', x),
        sigmoid=lambda x: ('torch_sigmoid', x),
        tanh=lambda x: ('torch_tanh', x),
        log=lambda x: ('torch_log', x),
        matmul=lambda a, b: ('torch_matmul', a, b),
        einsum=lambda eq, *ops: ('torch_einsum', eq, ops),
        exp=lambda x: ('torch_exp', x),
        fft=types.SimpleNamespace(fft=lambda x: ('torch_fft', x), ifft=lambda x: ('torch_ifft', x)),
        random=types.SimpleNamespace(randn=lambda *shape: ('torch_randn', shape)),
    )
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    fake.float32 = 'torch_float32'
    fake.__dict__['compile'] = None

    module = types.ModuleType('torch')
    for k, v in fake.__dict__.items():
        setattr(module, k, v)

    monkeypatch.setitem(sys.modules, 'torch', module)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: module if name == 'torch' else original_find_spec(name))
    return module


@pytest.fixture
def fake_jax(monkeypatch):
    jnp = types.SimpleNamespace(
        sum=lambda x, axis=None, keepdims=False: ('jax_sum', axis, keepdims),
        mean=lambda x, axis=None, keepdims=False: ('jax_mean', axis, keepdims),
        std=lambda x, axis=None, keepdims=False: ('jax_std', axis, keepdims),
        max=lambda x, axis=None, keepdims=False: ('jax_max', axis, keepdims),
        min=lambda x, axis=None, keepdims=False: ('jax_min', axis, keepdims),
        log=lambda x: ('jax_log', x),
        tanh=lambda x: ('jax_tanh', x),
        sigmoid=lambda x: ('jax_sigmoid', x),
        exp=lambda x: ('jax_exp', x),
        linalg=types.SimpleNamespace(norm=lambda x, ord=2, axis=None: ('jax_norm', ord, axis)),
        fft=types.SimpleNamespace(fft=lambda x: ('jax_fft', x), ifft=lambda x: ('jax_ifft', x)),
        random=types.SimpleNamespace(normal=lambda key, shape, **kwargs: ('jax_randn', shape)),
        maximum=lambda x, y: ('jax_maximum', x, y),
    )

    module = types.ModuleType('jax')
    module.devices = lambda: []

    monkeypatch.setitem(sys.modules, 'jax', module)
    monkeypatch.setitem(sys.modules, 'jax.numpy', jnp)
    monkeypatch.setitem(sys.modules, 'jax.numpy.fft', jnp.fft)
    monkeypatch.setitem(sys.modules, 'jax.nn', types.SimpleNamespace(softmax=lambda x, axis=-1: ('jax_softmax', axis)))
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: module if name in ('jax', 'jax.numpy') else original_find_spec(name))
    return module


def test_torch_adapter_capabilities(monkeypatch, fake_torch):
    adapter = adapters.get_adapter(BackendType.TORCH)
    lib = adapter.get_lib()

    assert lib.softmax('x', dim=1) == ('torch_softmax', 1)
    caps = adapter.capabilities
    assert caps.device_kind == 'gpu'
    assert caps.has_fft


def test_jax_adapter_capabilities(monkeypatch, fake_jax):
    adapter = adapters.get_adapter(BackendType.JAX)
    adapter.get_lib()

    caps = adapter.capabilities
    assert caps.device_kind == 'cpu'
    assert caps.supports_jit


def test_jax_adapter_sum(monkeypatch, fake_jax):
    adapter = adapters.get_adapter(BackendType.JAX)
    lib = adapter.get_lib()

    assert lib.sum('x', axis=0, keepdims=True) == ('jax_sum', 0, True)
