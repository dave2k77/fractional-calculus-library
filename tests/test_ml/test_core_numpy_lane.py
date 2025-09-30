import json
import types
from pathlib import Path

import numpy as np
import pytest

from hpfracc.ml.backends import BackendType, get_backend_manager
from hpfracc.ml.core import (
    FractionalNeuralNetwork,
    FractionalAttention,
    FractionalMSELoss,
    FractionalCrossEntropyLoss,
    FractionalAutoML,
)


def _identity_compute(values, *_args, **_kwargs):
    return values


def test_fractional_neural_network_forward_numpy(monkeypatch):
    manager = get_backend_manager()
    manager.active_backend = BackendType.NUMPY
    network = FractionalNeuralNetwork(
        input_size=3,
        hidden_sizes=[4],
        output_size=2,
        fractional_order=0.6,
        dropout=0.0,
    )

    monkeypatch.setattr(network.rl_calculator, "compute", _identity_compute)

    batch_input = np.ones((2, 3), dtype=np.float32)
    output = network.forward(batch_input, use_fractional=True, method="RL")

    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 2)


def test_fractional_neural_network_fractional_forward_1d(monkeypatch):
    manager = get_backend_manager()
    manager.active_backend = BackendType.NUMPY
    network = FractionalNeuralNetwork(
        input_size=4,
        hidden_sizes=[5],
        output_size=1,
        fractional_order=0.7,
        dropout=0.0,
    )

    monkeypatch.setattr(network.rl_calculator, "compute", _identity_compute)

    vector_input = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    result = network.fractional_forward(vector_input, method="RL")

    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)


def test_fractional_neural_network_save_load_numpy(tmp_path):
    manager = get_backend_manager()
    manager.active_backend = BackendType.NUMPY
    network = FractionalNeuralNetwork(
        input_size=2,
        hidden_sizes=[3],
        output_size=1,
    )

    model_path = tmp_path / "model.pth"
    network.save_model(str(model_path))

    loaded = FractionalNeuralNetwork.load_model(str(model_path))

    assert loaded.backend == manager.active_backend
    assert len(loaded.weights) == len(network.weights)
    assert all(w.shape == lw.shape for w, lw in zip(network.weights, loaded.weights))


def test_fractional_attention_forward_numpy(monkeypatch):
    manager = get_backend_manager()
    manager.active_backend = BackendType.NUMPY
    attention = FractionalAttention(
        d_model=4,
        n_heads=2,
        fractional_order=0.5,
        dropout=0.0,
    )

    monkeypatch.setattr(attention.rl_calculator, "compute", _identity_compute)
    monkeypatch.setattr(attention.caputo_calculator, "compute", _identity_compute)

    batch = np.random.randn(2, 3, attention.d_model).astype(np.float32)
    output = attention.forward(batch, method="Caputo")

    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 3, attention.d_model)


def test_fractional_losses_fractional_and_plain(monkeypatch):
    manager = get_backend_manager()
    manager.active_backend = BackendType.NUMPY
    predictions = np.random.randn(3, 2).astype(np.float32)
    targets = np.zeros((3, 2), dtype=np.float32)

    mse = FractionalMSELoss()
    monkeypatch.setattr(mse.rl_calculator, "compute", _identity_compute)

    frac_loss = mse.forward(predictions, targets, use_fractional=True)
    base_loss = mse.forward(predictions, targets, use_fractional=False)
    assert np.isclose(frac_loss, base_loss)

    ce = FractionalCrossEntropyLoss()
    monkeypatch.setattr(ce.rl_calculator, "compute", _identity_compute)
    frac_ce = ce.forward(predictions, targets, use_fractional=True)
    base_ce = ce.forward(predictions, targets, use_fractional=False)
    assert np.isfinite(frac_ce)
    assert np.isfinite(base_ce)


def test_fractional_automl_get_best_model_requires_optimization():
    automl = FractionalAutoML()
    with pytest.raises(ValueError):
        automl.get_best_model(lambda **kwargs: kwargs)


def test_fractional_automl_optimize_and_get_best_model(monkeypatch):
    class DummyModel:
        def __init__(self, fractional_order: float, learning_rate: float, backend: str):
            self.fractional_order = fractional_order
            self.learning_rate = learning_rate
            self.backend = backend

        def __call__(self, *_args, **_kwargs):
            return None

    class DummyTrial:
        def __init__(self):
            self.params = {}

        def suggest_int(self, name, low, high):
            value = low
            self.params[name] = value
            return value

        def suggest_float(self, name, low, high):
            value = (low + high) / 2
            self.params[name] = value
            return value

        def suggest_categorical(self, name, choices):
            value = choices[0]
            self.params[name] = value
            return value

    class DummyStudy:
        def __init__(self):
            self.best_params = {}
            self.best_value = None
            self.trials = []

        def optimize(self, objective, n_trials):
            trial = DummyTrial()
            value = objective(trial)
            self.best_params = dict(trial.params)
            self.best_value = value
            self.trials.append({"value": value, "params": dict(trial.params)})

    dummy_optuna = types.SimpleNamespace(create_study=lambda direction: DummyStudy())

    from hpfracc import ml as ml_module
    monkeypatch.setattr(ml_module.core, "optuna", dummy_optuna)

    automl = FractionalAutoML()

    param_ranges = {
        "fractional_order": [0.4, 0.8],
        "learning_rate": [0.001, 0.01],
        "backend": ["numpy", "torch"],
    }

    train_X = np.zeros((4, 2), dtype=np.float32)
    train_y = np.zeros((4, 1), dtype=np.float32)
    val_X = np.zeros((2, 2), dtype=np.float32)
    val_y = np.zeros((2, 1), dtype=np.float32)

    results = automl.optimize_fractional_order(
        DummyModel,
        train_data=(train_X, train_y),
        val_data=(val_X, val_y),
        param_ranges=param_ranges,
        n_trials=1,
        metric="accuracy",
    )

    assert "best_params" in results
    assert automl.best_params == results["best_params"]

    best_model = automl.get_best_model(DummyModel)
    assert isinstance(best_model, DummyModel)
    assert best_model.backend in ("numpy", "torch")

