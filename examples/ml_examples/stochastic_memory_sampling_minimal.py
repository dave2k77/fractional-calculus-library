#!/usr/bin/env python3
"""
Stochastic Memory Sampling - Minimal Example

Demonstrates StochasticFractionalLayer with importance sampling on a toy task.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer


def set_seed(seed: int = 11) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n: int = 1024, d: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, d)
    # Memory-like nonlinear map produces regression target
    coeff = torch.linspace(1.0, 0.2, d)
    y = (X.relu() * coeff).sum(dim=1, keepdim=True) + 0.05 * torch.randn(n, 1)
    return X, y


class TinyModel(nn.Module):
    def __init__(self, d: int = 64, alpha: float = 0.5, k: int = 32):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.frac = StochasticFractionalLayer(alpha=alpha, k=k, method="importance")
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.proj(x))
        z = self.frac(z)
        # If stochastic layer returns vector, use it directly
        if z.dim() == 2:
            z = z
        return self.head(z)


def main() -> None:
    set_seed(11)
    d = 64
    X, y = make_data(1024, d)
    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model = TinyModel(d=d, alpha=0.5, k=32)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, 4):
        total = 0.0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.size(0)
        print(f"epoch={epoch:02d} mse={total/len(ds):.6f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


