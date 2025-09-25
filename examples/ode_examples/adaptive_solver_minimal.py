#!/usr/bin/env python3
"""
Adaptive Fractional ODE Solver - Minimal Example

Demonstrates aliases (fractional_order, max_step/min_step, rtol/atol) usage.
"""

from __future__ import annotations

import os
import numpy as np

from hpfracc.solvers.ode_solvers import AdaptiveFractionalODESolver


def rhs(t: float, y: float) -> float:
    # Simple RHS: dy/dt = -y + sin(t)
    return -y + np.sin(t)


def main() -> None:
    solver = AdaptiveFractionalODESolver(
        derivative_type="caputo",
        method="predictor_corrector",
        tol=1e-5,
        max_iter=2000,
        rtol=1e-5,
        atol=1e-6,
        max_step=0.05,
        min_step=1e-5,
        fractional_order=0.6,
    )

    t0, t1 = 0.0, 5.0
    y0 = 0.0
    t = np.linspace(t0, t1, 200)
    # Solve is a stub in current implementation; this example focuses on API shape
    print("solver configured:", solver.derivative_type, solver.method)
    print("rtol, atol:", getattr(solver, "rtol", None), getattr(solver, "atol", None))
    print("min_h, max_h:", solver.min_h, solver.max_h)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


