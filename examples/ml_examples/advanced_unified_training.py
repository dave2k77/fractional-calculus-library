#!/usr/bin/env python3
"""
Advanced Unified Spectral FNN Training with Validation

Demonstrates train/val split, simple evaluation loop, and saving best model.
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork


def set_seed(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_regression(n: int = 6000, d: int = 64, noise: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, d)
    w = torch.linspace(0.1, 1.0, d)
    y = (x.sin() * w).sum(dim=1, keepdim=True)
    y = y + noise * torch.randn_like(y)
    return x, y


@dataclass
class TrainConfig:
    input_dim: int = 64
    hidden_dims: Tuple[int, int] = (128, 128)
    output_dim: int = 1
    alpha: float = 0.6
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 256
    val_frac: float = 0.2
    model_out: str = "./advanced_unified_best.pt"


def split_dataset(x: torch.Tensor, y: torch.Tensor, val_frac: float) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    n = x.size(0)
    n_val = int(n * val_frac)
    idx = torch.randperm(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_va, y_va = x[val_idx], y[val_idx]
    return torch.utils.data.TensorDataset(x_tr, y_tr), torch.utils.data.TensorDataset(x_va, y_va)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total += float(loss.detach()) * xb.size(0)
            count += xb.size(0)
    return total / max(count, 1)


def main() -> None:
    set_seed(2024)
    cfg = TrainConfig()

    x, y = make_regression(n=6000, d=cfg.input_dim)
    ds_tr, ds_va = split_dataset(x, y, cfg.val_frac)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False)

    model = SpectralFractionalNetwork(
        input_dim=cfg.input_dim,
        hidden_dims=list(cfg.hidden_dims),
        output_dim=cfg.output_dim,
        alpha=cfg.alpha,
        mode="unified",
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)
        train_mse = running / max(seen, 1)
        val_mse = evaluate(model, dl_va, loss_fn)
        print(f"epoch={epoch:02d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            try:
                torch.save(model.state_dict(), cfg.model_out)
            except Exception:
                pass

    print(f"best_val_mse={best_val:.6f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


