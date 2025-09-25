#!/usr/bin/env python3
"""
Probabilistic Fractional Orders - Minimal Example

Shows a layer with a learnable normal distribution over alpha.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer


def set_seed(seed: int = 5) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n: int = 1500, d: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, d)
    y = (X.tanh()[:, :8].sum(dim=1, keepdim=True) + 0.1 * torch.randn(n, 1))
    return X, y


class ProbFracNet(nn.Module):
    def __init__(self, d: int = 24):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        # Probabilistic alpha layer (reparameterized when possible)
        self.prob_alpha = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        pa = self.prob_alpha(z)
        if pa.dim() == 2:
            pa = pa.mean(dim=1, keepdim=True)
        return self.head(z) + 0.1 * pa


def main() -> None:
    set_seed(5)
    d = 24
    X, y = make_data(1500, d)
    ds = torch.utils.data.TensorDataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model = ProbFracNet(d=d)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, 6):
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


