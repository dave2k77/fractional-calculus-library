#!/usr/bin/env python3
"""
Fixed-Step Fractional ODE Solver - Minimal Example

Demonstrates fixed-step fractional ODE solver usage.
Note: Adaptive solver is currently disabled due to implementation issues.
"""

from __future__ import annotations

import os
import numpy as np

from hpfracc.solvers.ode_solvers import FixedStepODESolver


def rhs(t: float, y: float) -> float:
    # Simple RHS: dy/dt = -y + sin(t)
    return -y + np.sin(t)


def main() -> None:
    solver = FixedStepODESolver(
        derivative_type="caputo",
        method="predictor_corrector",
        adaptive=False
    )

    t0, t1 = 0.0, 5.0
    y0 = 0.0
    alpha = 0.6
    h = 0.01
    
    # Solve the fractional ODE
    t, y = solver.solve(rhs, (t0, t1), y0, alpha, h)
    
    print("solver configured:", solver.derivative_type, solver.method)
    print("fractional order:", alpha)
    print("step size:", h)
    print("solution shape:", y.shape)
    print("time points:", t.shape)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


