#!/usr/bin/env python3
"""
GPU-Optimized Spectral Example (with CPU fallbacks)

Uses GPUConfig and GPUOptimizedRiemannLiouville paths where available.
Falls back to CPU-safe computation gracefully when CUDA/CuPy are unavailable.
"""

from __future__ import annotations

import os
import numpy as np

import torch

from hpfracc.algorithms.gpu_optimized_methods import (
    GPUConfig,
    GPUOptimizedRiemannLiouville,
)


def main() -> None:
    # Synthetic function: f(t) = sin(t)
    t = np.linspace(0.0, 10.0, 2048)
    f = np.sin(t)

    # GPU config tries CUDA; code falls back automatically if not available
    config = GPUConfig(backend="auto", memory_limit=0.8, device_id=0)
    rl = GPUOptimizedRiemannLiouville(alpha=0.6, gpu_config=config)

    # Compute derivative (array interface)
    d = rl.compute(f, t, h=(t[1] - t[0]))
    print("result_shape:", d.shape)
    print("result_sample:", d[:5])


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


