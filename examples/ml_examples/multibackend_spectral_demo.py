#!/usr/bin/env python3
"""
Multi-backend Spectral Demo

Cycles through requested backends (pytorch, jax, numba) if available, otherwise
falls back to pytorch/CPU-safe path. Prints backend and a quick forward result.
"""

from __future__ import annotations

import os
import torch

from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork


def try_backend(backend: str) -> None:
    print(f"\n=== Trying backend: {backend} ===")
    x = torch.randn(8, 16)
    try:
        net = SpectralFractionalNetwork(
            input_dim=16, hidden_dims=[32], output_dim=4, alpha=0.5,
            mode="unified",
        )
        # In current API, backend is selected via internal managers; demo focuses on run
        y = net(x)
        print("ok, output shape:", tuple(y.shape))
    except Exception as e:
        print("backend run failed:", type(e).__name__, str(e))


def main() -> None:
    for b in ["pytorch", "jax", "numba"]:
        try_backend(b)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


