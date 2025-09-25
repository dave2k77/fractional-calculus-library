#!/usr/bin/env python3
"""
Fractional PDE Solver - Minimal Example

Demonstrates initialization with boundary_conditions and fractional_order aliases.
"""

from __future__ import annotations

import os

from hpfracc.solvers.pde_solvers import FractionalPDESolver


def main() -> None:
    solver = FractionalPDESolver(
        pde_type="diffusion",
        method="finite_difference",
        spatial_order=2,
        temporal_order=1,
        adaptive=False,
        fractional_order=0.7,
        boundary_conditions="dirichlet",
    )
    print("pde solver configured:", solver.pde_type, solver.method)
    print("fractional_order:", solver.fractional_order)
    print("boundary_conditions:", solver.boundary_conditions)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()


