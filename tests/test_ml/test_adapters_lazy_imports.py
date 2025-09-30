import importlib
import sys
import types

import pytest

from hpfracc.ml import adapters, backends
from hpfracc.ml.backends import BackendType


@pytest.fixture(autouse=True)
def reset_backend_manager():
    backends._backend_manager = None
    yield
    backends._backend_manager = None


def _set_availability(monkeypatch, torch=True, jax=True, numba=True, numpy=True):
    monkeypatch.setattr(backends, "TORCH_AVAILABLE", torch)
    monkeypatch.setattr(backends, "JAX_AVAILABLE", jax)
    monkeypatch.setattr(backends, "NUMBA_AVAILABLE", numba)
    monkeypatch.setattr(backends, "NUMPY_AVAILABLE", numpy)


def _mock_imports(monkeypatch, mapping):
    original_backend_import = backends.importlib.import_module
    original_adapter_import = adapters.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name in mapping:
            return mapping[name]
        return original_backend_import(name, *args, **kwargs)

    def fake_import_adapters(name, *args, **kwargs):
        if name in mapping:
            return mapping[name]
        return original_adapter_import(name, *args, **kwargs)

    monkeypatch.setattr(backends.importlib, "import_module", fake_import)
    monkeypatch.setattr(adapters.importlib, "import_module", fake_import_adapters)


def _mock_spec(monkeypatch, available_names):
    monkeypatch.setattr(adapters, "_spec_available", lambda name: name in available_names)


def test_backends_torch_lazy_detection(monkeypatch):
    fake_torch = types.SimpleNamespace(
        __version__="1.0",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        fft=True,
        autograd=True,
        compile=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _set_availability(monkeypatch, torch=True, jax=False, numba=False, numpy=True)
    _mock_imports(monkeypatch, {"torch": fake_torch})
    _mock_spec(monkeypatch, {"torch"})

    manager = backends.BackendManager()
    assert manager.available_backends == [BackendType.TORCH]
    assert manager.active_backend == BackendType.TORCH

    adapter = adapters.get_adapter(BackendType.TORCH)
    assert adapter.get_lib() is fake_torch
    assert adapter.capabilities.device_kind == "gpu"


def test_backends_torch_disabled(monkeypatch):
    _set_availability(monkeypatch, torch=False, jax=False, numba=True, numpy=True)
    _mock_spec(monkeypatch, set())

    manager = backends.BackendManager()
    assert BackendType.TORCH not in manager.available_backends
    with pytest.raises(ImportError):
        adapters.get_adapter(BackendType.TORCH)


def test_backends_jax_detection(monkeypatch):
    fake_jax = types.SimpleNamespace(devices=lambda: [])
    fake_jax_numpy = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jax_numpy)
    monkeypatch.setitem(sys.modules, "jax.numpy.fft", types.SimpleNamespace())

    _set_availability(monkeypatch, torch=False, jax=True, numba=False, numpy=True)
    _mock_imports(monkeypatch, {"jax": fake_jax, "jax.numpy": fake_jax_numpy, "jax.numpy.fft": types.SimpleNamespace()})
    _mock_spec(monkeypatch, {"jax", "jax.numpy"})

    manager = backends.BackendManager()
    assert manager.available_backends == [BackendType.JAX]

    adapter = adapters.get_adapter(BackendType.JAX)
    assert adapter.get_lib() is fake_jax_numpy
    assert adapter.capabilities.has_fft


def test_backends_jax_disabled(monkeypatch):
    _set_availability(monkeypatch, torch=False, jax=False, numba=True, numpy=True)
    _mock_spec(monkeypatch, set())

    manager = backends.BackendManager()
    assert BackendType.JAX not in manager.available_backends
    with pytest.raises(ImportError):
        adapters.get_adapter(BackendType.JAX)


def test_backends_numpy_lane(monkeypatch):
    fake_numpy = types.SimpleNamespace(__version__="1.0", fft=True)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)

    _set_availability(monkeypatch, torch=False, jax=False, numba=True, numpy=True)
    _mock_imports(monkeypatch, {"numpy": fake_numpy})
    _mock_spec(monkeypatch, {"numpy"})

    manager = backends.BackendManager()
    assert manager.available_backends == [BackendType.NUMBA]

    adapter = adapters.get_adapter(BackendType.NUMBA)
    assert adapter.get_lib() is fake_numpy
    assert adapter.capabilities.has_fft
    assert not adapter.capabilities.has_autograd


def test_detect_capabilities_fallback(monkeypatch):
    def raise_import(_backend):
        raise ImportError

    monkeypatch.setattr(adapters, "get_adapter", raise_import)
    caps = adapters.detect_capabilities(BackendType.TORCH)
    assert caps.device_kind == "cpu"
    assert not caps.has_fft

