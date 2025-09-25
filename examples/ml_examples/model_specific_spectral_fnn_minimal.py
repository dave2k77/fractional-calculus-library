#!/usr/bin/env python3
"""
Model-Specific (Legacy) Spectral Fractional FNN - Minimal Example

This example opts into the legacy/model-specific mode of SpectralFractionalNetwork
using the coverage-style constructor arguments.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_toy_classification(
    num_samples: int = 2000,
    input_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(num_samples, input_size)
    # Two-class nonlinear separation via fractional-friendly transform
    logits = (X.cos() * torch.linspace(0.3, 1.2, input_size)).sum(dim=1)
    y = (logits > 0).long()
    return X, y


def main() -> None:
    set_seed(123)

    input_size = 16
    hidden_sizes = [32, 32]
    output_size = 2
    alpha = 0.5
    lr = 1e-3
    epochs = 5
    batch_size = 128

    X, y = make_toy_classification(num_samples=2000, input_size=input_size)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Legacy/model-specific mode via legacy args and mode="model"
    model = SpectralFractionalNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        alpha=alpha,
        mode="model",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * xb.size(0)
        epoch_loss = running / len(dataset)
        print(f"epoch={epoch:02d} loss={epoch_loss:.6f}")

    # Quick accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    print(f"final_acc={acc:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


