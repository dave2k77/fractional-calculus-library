#!/usr/bin/env python3
"""
Unified Spectral Fractional FNN - Minimal Example

This example demonstrates how to use the unified-by-default SpectralFractionalNetwork
for a simple regression task on synthetic data. It runs on CPU and finishes quickly.
"""

from __future__ import annotations

import os
import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_regression_dataset(
    num_samples: int = 2048,
    input_dim: int = 32,
    noise_std: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(num_samples, input_dim)
    # Ground truth: nonlinear target with memory-like mixing across features
    weights = torch.linspace(0.2, 1.0, input_dim)
    y_clean = (X.sin() * weights).sum(dim=1, keepdim=True)
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return X, y


def main() -> None:
    set_seed(7)

    # Config
    input_dim = 32
    hidden_dims = [64, 64]
    output_dim = 1
    alpha = 0.6
    lr = 2e-3
    epochs = 5  # keep short for a quick demo
    batch_size = 128

    # Data
    X, y = make_synthetic_regression_dataset(num_samples=2048, input_dim=input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model: unified mode is default; specify dims
    model = SpectralFractionalNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        alpha=alpha,
        mode="unified",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.size(0)
        epoch_loss = running / len(dataset)
        print(f"epoch={epoch:02d} mse={epoch_loss:.6f}")

    # Quick evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X)
        mse = loss_fn(preds, y).item()
    print(f"final_mse={mse:.6f}")


if __name__ == "__main__":
    # Allow shorter torch thread usage for deterministic quick runs in CI
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


