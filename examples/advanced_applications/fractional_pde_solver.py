#!/usr/bin/env python3
"""
Fractional PDE Solver Example

This example demonstrates solving fractional partial differential equations
using the library's advanced solver capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.solvers.pde_solvers import FractionalDiffusionSolver
from src.algorithms.L1_L2_schemes import L1L2Schemes
from src.solvers.predictor_corrector import PredictorCorrectorSolver
from src.special.gamma_beta import gamma


def ensure_output_dir():
    """Ensure the output directory exists."""
    output_dir = os.path.join("examples", "advanced_applications")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def fractional_diffusion_equation():
    """Solve the fractional diffusion equation."""
    print("🌊 Fractional Diffusion Equation Solver")
    print("=" * 50)

    # Problem parameters
    L = 1.0  # Domain length
    T = 1.0  # Final time
    Nx = 50  # Spatial grid points
    Nt = 100  # Time grid points
    alpha = 0.5  # Fractional order

    # Create solver
    solver = FractionalDiffusionSolver()

    # Define initial condition: u(x,0) = sin(πx)
    def initial_condition(x):
        return np.sin(np.pi * x)

    # Define boundary conditions: u(0,t) = u(1,t) = 0
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    print(f"Solving fractional diffusion equation:")
    print(f"  ∂ᵅu/∂tᵅ = ∂²u/∂x²")
    print(f"  u(x,0) = sin(πx)")
    print(f"  u(0,t) = u(1,t) = 0")
    print(f"  α = {alpha}, L = {L}, T = {T}")
    print(f"  Grid: {Nx} × {Nt} points")

    # Solve the equation
    x, t, u = solver.solve(
        x_span=(0, L),
        t_span=(0, T),
        initial_condition=initial_condition,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        beta=2.0,  # Standard second-order spatial derivative
        nx=Nx,
        nt=Nt,
    )

    # Plot results
    plt.figure(figsize=(15, 10))

    # 3D surface plot
    ax1 = plt.subplot(2, 2, 1, projection="3d")
    X, T_mesh = np.meshgrid(x, t)
    surf = ax1.plot_surface(X, T_mesh, u.T, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Time t")
    ax1.set_zlabel("Solution u(x,t)")
    ax1.set_title("Fractional Diffusion: 3D Surface")
    plt.colorbar(surf, ax=ax1)

    # Time evolution at different positions
    ax2 = plt.subplot(2, 2, 2)
    positions = [0.25, 0.5, 0.75]
    for pos in positions:
        idx = np.argmin(np.abs(x - pos))
        ax2.plot(t, u[idx, :], linewidth=2, label=f"x = {pos}")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Solution u(x,t)")
    ax2.set_title("Time Evolution at Different Positions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Spatial profiles at different times
    ax3 = plt.subplot(2, 2, 3)
    times = [0.1, 0.3, 0.5, 0.7, 0.9]
    for time_val in times:
        idx = np.argmin(np.abs(t - time_val))
        ax3.plot(x, u[:, idx], linewidth=2, label=f"t = {time_val}")
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Solution u(x,t)")
    ax3.set_title("Spatial Profiles at Different Times")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Contour plot
    ax4 = plt.subplot(2, 2, 4)
    contour = ax4.contourf(X, T_mesh, u.T, levels=20, cmap="viridis")
    ax4.set_xlabel("Position x")
    ax4.set_ylabel("Time t")
    ax4.set_title("Fractional Diffusion: Contour Plot")
    plt.colorbar(contour, ax=ax4)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "fractional_diffusion.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Fractional diffusion equation solved!")


def fractional_wave_equation():
    """Solve the fractional wave equation using diffusion solver."""
    print("\n🌊 Fractional Wave Equation Solver (using diffusion solver)")
    print("=" * 50)

    # Problem parameters
    L = 2.0  # Domain length
    T = 2.0  # Final time
    Nx = 50  # Spatial grid points
    Nt = 100  # Time grid points
    alpha = 1.5  # Fractional order (wave-like)

    # Create solver (using diffusion solver for now)
    solver = FractionalDiffusionSolver()

    # Define initial condition: u(x,0) = exp(-(x-1)²)
    def initial_condition(x):
        return np.exp(-((x - 1) ** 2))

    # Define boundary conditions: u(0,t) = u(2,t) = 0
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    print(f"Solving fractional diffusion equation (simulating wave behavior):")
    print(f"  ∂ᵅu/∂tᵅ = ∂²u/∂x²")
    print(f"  u(x,0) = exp(-(x-1)²)")
    print(f"  u(0,t) = u(2,t) = 0")
    print(f"  α = {alpha}, L = {L}, T = {T}")
    print(f"  Grid: {Nx} × {Nt} points")

    # Solve the equation
    x, t, u = solver.solve(
        x_span=(0, L),
        t_span=(0, T),
        initial_condition=initial_condition,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        beta=2.0,  # Standard second-order spatial derivative
        nx=Nx,
        nt=Nt,
    )

    # Plot results
    plt.figure(figsize=(15, 10))

    # 3D surface plot
    ax1 = plt.subplot(2, 2, 1, projection="3d")
    X, T_mesh = np.meshgrid(x, t)
    surf = ax1.plot_surface(X, T_mesh, u.T, cmap="coolwarm", alpha=0.8)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Time t")
    ax1.set_zlabel("Solution u(x,t)")
    ax1.set_title("Fractional Diffusion: 3D Surface")
    plt.colorbar(surf, ax=ax1)

    # Time evolution at different positions
    ax2 = plt.subplot(2, 2, 2)
    positions = [0.5, 1.0, 1.5]
    for pos in positions:
        idx = np.argmin(np.abs(x - pos))
        ax2.plot(t, u[idx, :], linewidth=2, label=f"x = {pos}")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Solution u(x,t)")
    ax2.set_title("Time Evolution at Different Positions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Spatial profiles at different times
    ax3 = plt.subplot(2, 2, 3)
    times = [0.2, 0.5, 1.0, 1.5, 2.0]
    for time_val in times:
        idx = np.argmin(np.abs(t - time_val))
        ax3.plot(x, u[:, idx], linewidth=2, label=f"t = {time_val}")
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Solution u(x,t)")
    ax3.set_title("Spatial Profiles at Different Times")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Contour plot
    ax4 = plt.subplot(2, 2, 4)
    contour = ax4.contourf(X, T_mesh, u.T, levels=20, cmap="coolwarm")
    ax4.set_xlabel("Position x")
    ax4.set_ylabel("Time t")
    ax4.set_title("Fractional Diffusion: Contour Plot")
    plt.colorbar(contour, ax=ax4)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "fractional_wave.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Fractional diffusion equation solved!")


def L1_L2_schemes_comparison():
    """Compare L1 and L2 schemes for time-fractional PDEs."""
    print("\n📊 L1/L2 Schemes Comparison")
    print("=" * 50)

    # Problem parameters
    L = 1.0
    T = 1.0
    Nx = 40
    Nt = 80
    alpha = 0.5

    # Create schemes
    l1_scheme = L1L2Schemes(scheme="l1")
    l2_scheme = L1L2Schemes(scheme="l2")

    # Define initial condition
    def initial_condition(x):
        return np.sin(np.pi * x)

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    print(f"Comparing L1 and L2 schemes for fractional diffusion:")
    print(f"  ∂ᵅu/∂tᵅ = ∂²u/∂x²")
    print(f"  α = {alpha}, L = {L}, T = {T}")
    print(f"  Grid: {Nx} × {Nt} points")

    # Create spatial grid
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]
    dt = T / Nt

    # Create initial condition array
    u0 = np.array([initial_condition(xi) for xi in x])

    # Solve with L1 scheme
    print("\n🧪 Solving with L1 scheme...")
    t_l1, x_l1, u_l1 = l1_scheme.solve_time_fractional_pde(
        initial_condition=u0,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        t_final=T,
        dt=dt,
        dx=dx,
    )

    # Solve with L2 scheme
    print("🧪 Solving with L2 scheme...")
    t_l2, x_l2, u_l2 = l2_scheme.solve_time_fractional_pde(
        initial_condition=u0,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        t_final=T,
        dt=dt,
        dx=dx,
    )

    # Plot comparison
    plt.figure(figsize=(15, 10))

    # 3D comparison
    ax1 = plt.subplot(2, 3, 1, projection="3d")
    X, T_mesh = np.meshgrid(x_l1, t_l1)
    surf1 = ax1.plot_surface(X, T_mesh, u_l1, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Time t")
    ax1.set_zlabel("Solution u(x,t)")
    ax1.set_title("L1 Scheme: 3D Surface")
    plt.colorbar(surf1, ax=ax1)

    ax2 = plt.subplot(2, 3, 2, projection="3d")
    X2, T_mesh2 = np.meshgrid(x_l2, t_l2)
    surf2 = ax2.plot_surface(X2, T_mesh2, u_l2, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Time t")
    ax2.set_zlabel("Solution u(x,t)")
    ax2.set_title("L2 Scheme: 3D Surface")
    plt.colorbar(surf2, ax=ax2)

    # Difference (use L1 grid for comparison)
    ax3 = plt.subplot(2, 3, 3, projection="3d")
    # Interpolate L2 solution to L1 grid for comparison
    from scipy.interpolate import RegularGridInterpolator

    if len(t_l2) > 1 and len(x_l2) > 1:
        f_interp = RegularGridInterpolator((t_l2, x_l2), u_l2, method="linear")
        T_mesh_flat, X_flat = np.meshgrid(t_l1, x_l1, indexing="ij")
        points = np.column_stack((T_mesh_flat.ravel(), X_flat.ravel()))
        u_l2_interp = f_interp(points).reshape(T_mesh_flat.shape)
        diff = u_l2_interp - u_l1
    else:
        diff = u_l2 - u_l1
    surf3 = ax3.plot_surface(X, T_mesh, diff, cmap="RdBu", alpha=0.8)
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Time t")
    ax3.set_zlabel("Difference")
    ax3.set_title("L2 - L1: Difference")
    plt.colorbar(surf3, ax=ax3)

    # Time evolution comparison
    ax4 = plt.subplot(2, 3, 4)
    pos = 0.5
    idx = np.argmin(np.abs(x_l1 - pos))
    ax4.plot(t_l1, u_l1[:, idx], "b-", linewidth=2, label="L1 Scheme")
    # Interpolate L2 to L1 time grid
    if len(t_l2) > 1:
        from scipy.interpolate import interp1d

        f_interp = interp1d(
            t_l2,
            u_l2[:, idx],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        u_l2_interp = f_interp(t_l1)
        ax4.plot(t_l1, u_l2_interp, "r--", linewidth=2, label="L2 Scheme")
    ax4.set_xlabel("Time t")
    ax4.set_ylabel("Solution u(x,t)")
    ax4.set_title(f"Time Evolution at x = {pos}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Spatial profiles comparison
    ax5 = plt.subplot(2, 3, 5)
    time_val = 0.5
    idx = np.argmin(np.abs(t_l1 - time_val))
    ax5.plot(x_l1, u_l1[idx, :], "b-", linewidth=2, label="L1 Scheme")
    # Interpolate L2 to L1 spatial grid
    if len(x_l2) > 1:
        from scipy.interpolate import interp1d

        f_interp = interp1d(
            x_l2,
            u_l2[idx, :],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        u_l2_interp = f_interp(x_l1)
        ax5.plot(x_l1, u_l2_interp, "r--", linewidth=2, label="L2 Scheme")
    ax5.set_xlabel("Position x")
    ax5.set_ylabel("Solution u(x,t)")
    ax5.set_title(f"Spatial Profile at t = {time_val}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Error analysis
    ax6 = plt.subplot(2, 3, 6)
    error = np.abs(diff)
    max_error = np.max(error, axis=1)  # Max over spatial dimension
    ax6.semilogy(t_l1, max_error, "g-", linewidth=2)
    ax6.set_xlabel("Time t")
    ax6.set_ylabel("Max Absolute Error")
    ax6.set_title("Error Evolution")
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "L1_L2_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ L1/L2 schemes comparison completed!")


def predictor_corrector_demo():
    """Demonstrate predictor-corrector methods."""
    print("\n🎯 Predictor-Corrector Methods Demo")
    print("=" * 50)

    # Problem parameters
    L = 1.0
    T = 1.0
    Nx = 30
    Nt = 60
    alpha = 0.7

    # Create predictor-corrector solver
    pc_solver = PredictorCorrectorSolver()

    # Define initial condition
    def initial_condition(x):
        return np.sin(2 * np.pi * x)

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    print(f"Solving with predictor-corrector method:")
    print(f"  ∂ᵅu/∂tᵅ = ∂²u/∂x²")
    print(f"  u(x,0) = sin(2πx)")
    print(f"  α = {alpha}, L = {L}, T = {T}")
    print(f"  Grid: {Nx} × {Nt} points")

    # For predictor-corrector, we'll solve a simple ODE instead of PDE
    # since the predictor-corrector solver is designed for ODEs
    def f(t, y):
        return -y  # Simple decay equation

    # Solve with predictor-corrector
    t_pc, y_pc = pc_solver.solve(f=f, t_span=(0, T), y0=1.0, alpha=alpha, h0=0.01)

    # Compare with standard diffusion solver
    solver = FractionalDiffusionSolver()
    x, t, u_std = solver.solve(
        x_span=(0, L),
        t_span=(0, T),
        initial_condition=initial_condition,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        beta=2.0,  # Standard second-order spatial derivative
        nx=Nx,
        nt=Nt,
    )

    # Plot results
    plt.figure(figsize=(15, 10))

    # 2D comparison (since predictor-corrector solves ODE, not PDE)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t_pc, y_pc, "b-", linewidth=2, label="Predictor-Corrector")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Solution y(t)")
    ax1.set_title("Predictor-Corrector: ODE Solution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2, projection="3d")
    X, T_mesh = np.meshgrid(x, t)
    surf2 = ax2.plot_surface(X, T_mesh, u_std.T, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Time t")
    ax2.set_zlabel("Solution u(x,t)")
    ax2.set_title("Standard Method: PDE Solution")
    plt.colorbar(surf2, ax=ax2)

    # Compare ODE solution with PDE solution at a specific position
    ax3 = plt.subplot(2, 3, 3)
    pos = 0.5
    idx = np.argmin(np.abs(x - pos))
    ax3.plot(t, u_std[idx, :], "r--", linewidth=2, label=f"PDE at x={pos}")
    # Interpolate ODE solution to PDE time grid
    from scipy.interpolate import interp1d

    if len(t_pc) > 1 and len(y_pc) > 1:
        # Handle case where y_pc might be a list of arrays
        if isinstance(y_pc, list):
            y_pc_array = np.array(y_pc).flatten()
        else:
            y_pc_array = y_pc.flatten()

        # Ensure arrays have the same length
        min_len = min(len(t_pc), len(y_pc_array))
        t_pc_trim = t_pc[:min_len]
        y_pc_trim = y_pc_array[:min_len]

        if min_len > 1:
            f_interp = interp1d(
                t_pc_trim,
                y_pc_trim,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            y_pc_interp = f_interp(t)
            ax3.plot(t, y_pc_interp, "b-", linewidth=2, label="ODE Solution")
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("Solution")
    ax3.set_title("ODE vs PDE Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Time evolution comparison
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t_pc, y_pc, "b-", linewidth=2, label="Predictor-Corrector (ODE)")
    ax4.set_xlabel("Time t")
    ax4.set_ylabel("Solution y(t)")
    ax4.set_title("Predictor-Corrector Time Evolution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Spatial profiles comparison
    ax5 = plt.subplot(2, 3, 5)
    time_val = 0.5
    idx = np.argmin(np.abs(t - time_val))
    ax5.plot(x, u_std[:, idx], "r--", linewidth=2, label="Standard Method")
    ax5.set_xlabel("Position x")
    ax5.set_ylabel("Solution u(x,t)")
    ax5.set_title(f"Spatial Profile at t = {time_val}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Convergence analysis
    ax6 = plt.subplot(2, 3, 6)
    # Compare ODE and PDE solutions at a specific position
    if len(t_pc) > 1 and len(t) > 1 and len(y_pc) > 1:
        # Handle case where y_pc might be a list of arrays
        if isinstance(y_pc, list):
            y_pc_array = np.array(y_pc).flatten()
        else:
            y_pc_array = y_pc.flatten()

        # Ensure arrays have the same length
        min_len = min(len(t_pc), len(y_pc_array))
        t_pc_trim = t_pc[:min_len]
        y_pc_trim = y_pc_array[:min_len]

        if min_len > 1:
            f_interp = interp1d(
                t_pc_trim,
                y_pc_trim,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            y_pc_interp = f_interp(t)
            error = np.abs(y_pc_interp - u_std[idx, :])
            ax6.semilogy(t, error, "g-", linewidth=2)
            ax6.set_xlabel("Time t")
            ax6.set_ylabel("Absolute Error")
            ax6.set_title("ODE vs PDE Error")
            ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "predictor_corrector.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Predictor-corrector demo completed!")


def anomalous_transport_demo():
    """Demonstrate anomalous transport: superdiffusion and subdiffusion."""
    print("\n🚀 Anomalous Transport Demo")
    print("=" * 50)

    # Problem parameters
    L = 2.0
    T = 2.0
    Nx = 100
    Nt = 200

    # Create solver
    solver = FractionalDiffusionSolver()

    # Define initial condition: Gaussian pulse
    def initial_condition(x):
        return np.exp(-(((x - L / 2) / 0.1) ** 2))

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    # Compare different fractional orders
    alphas = [0.3, 0.5, 1.0, 1.5, 2.0]  # Subdiffusion to superdiffusion
    solutions = []

    print(f"Comparing anomalous transport regimes:")
    print(f"  α < 1: Subdiffusion (slower than normal)")
    print(f"  α = 1: Normal diffusion")
    print(f"  α > 1: Superdiffusion (faster than normal)")

    for alpha in alphas:
        print(f"\n🧪 Solving with α = {alpha}...")
        x, t, u = solver.solve(
            x_span=(0, L),
            t_span=(0, T),
            initial_condition=initial_condition,
            boundary_conditions=(boundary_condition_left, boundary_condition_right),
            alpha=alpha,
            beta=2.0,
            nx=Nx,
            nt=Nt,
        )
        solutions.append((alpha, x, t, u))

    # Plot comparison
    plt.figure(figsize=(20, 15))

    # 3D surfaces for different alphas
    for i, (alpha, x, t, u) in enumerate(solutions):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        X, T_mesh = np.meshgrid(x, t)
        surf = ax.plot_surface(X, T_mesh, u.T, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Time t")
        ax.set_zlabel("u(x,t)")
        ax.set_title(f"α = {alpha}")
        plt.colorbar(surf, ax=ax)

    # Time evolution comparison at center
    ax_center = plt.subplot(3, 3, 6)
    center_idx = len(x) // 2
    for alpha, x, t, u in solutions:
        ax_center.plot(t, u[center_idx, :], linewidth=2, label=f"α = {alpha}")
    ax_center.set_xlabel("Time t")
    ax_center.set_ylabel("u(x,t) at x = L/2")
    ax_center.set_title("Time Evolution at Center")
    ax_center.legend()
    ax_center.grid(True, alpha=0.3)

    # Spatial profiles at final time
    ax_final = plt.subplot(3, 3, 7)
    for alpha, x, t, u in solutions:
        ax_final.plot(x, u[:, -1], linewidth=2, label=f"α = {alpha}")
    ax_final.set_xlabel("Position x")
    ax_final.set_ylabel("u(x,T)")
    ax_final.set_title("Final Spatial Profiles")
    ax_final.legend()
    ax_final.grid(True, alpha=0.3)

    # MSD (Mean Square Displacement) analysis
    ax_msd = plt.subplot(3, 3, 8)
    for alpha, x, t, u in solutions:
        # Calculate MSD
        msd = np.zeros(len(t))
        for i, ti in enumerate(t):
            # Calculate variance of the distribution
            mean_pos = np.sum(x * u[:, i]) / np.sum(u[:, i])
            msd[i] = np.sum((x - mean_pos) ** 2 * u[:, i]) / np.sum(u[:, i])

        # Fit power law: MSD ~ t^γ
        if len(t) > 10:
            log_t = np.log(t[1:])  # Skip t=0
            log_msd = np.log(msd[1:])
            coeffs = np.polyfit(log_t, log_msd, 1)
            gamma = coeffs[0]
            ax_msd.loglog(
                t, msd, "o-", linewidth=2, label=f"α = {alpha}, γ = {gamma:.2f}"
            )
        else:
            ax_msd.loglog(t, msd, "o-", linewidth=2, label=f"α = {alpha}")

    ax_msd.set_xlabel("Time t")
    ax_msd.set_ylabel("MSD")
    ax_msd.set_title("Mean Square Displacement")
    ax_msd.legend()
    ax_msd.grid(True, alpha=0.3)

    # Contour comparison
    ax_contour = plt.subplot(3, 3, 9)
    # Show subdiffusion vs superdiffusion
    sub_idx = 0  # α = 0.3
    super_idx = 3  # α = 1.5

    X, T_mesh = np.meshgrid(x, t)
    contour1 = ax_contour.contour(
        X, T_mesh, solutions[sub_idx][3].T, levels=10, colors="blue", alpha=0.7
    )
    contour2 = ax_contour.contour(
        X, T_mesh, solutions[super_idx][3].T, levels=10, colors="red", alpha=0.7
    )
    ax_contour.set_xlabel("Position x")
    ax_contour.set_ylabel("Time t")
    ax_contour.set_title("Subdiffusion vs Superdiffusion")
    ax_contour.clabel(contour1, inline=True, fontsize=8)
    ax_contour.clabel(contour2, inline=True, fontsize=8)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "anomalous_transport.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Anomalous transport demo completed!")


def memory_effects_demo():
    """Demonstrate memory effects in fractional dynamics."""
    print("\n🧠 Memory Effects Demo")
    print("=" * 50)

    # Problem parameters
    L = 1.0
    T = 3.0
    Nx = 50
    Nt = 300
    alpha = 0.7  # Fractional order for memory effects

    # Create solver
    solver = FractionalDiffusionSolver()

    # Define initial condition: localized pulse
    def initial_condition(x):
        return np.exp(-(((x - 0.3) / 0.05) ** 2))

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    print(f"Demonstrating memory effects with α = {alpha}")
    print(f"Memory effects are stronger for smaller α values")

    # Solve with fractional dynamics (has memory)
    x, t, u_fractional = solver.solve(
        x_span=(0, L),
        t_span=(0, T),
        initial_condition=initial_condition,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=alpha,
        beta=2.0,
        nx=Nx,
        nt=Nt,
    )

    # Solve with normal diffusion (no memory) for comparison
    x_normal, t_normal, u_normal = solver.solve(
        x_span=(0, L),
        t_span=(0, T),
        initial_condition=initial_condition,
        boundary_conditions=(boundary_condition_left, boundary_condition_right),
        alpha=1.0,  # Normal diffusion
        beta=2.0,
        nx=Nx,
        nt=Nt,
    )

    # Plot results
    plt.figure(figsize=(20, 12))

    # 3D comparison
    ax1 = plt.subplot(2, 4, 1, projection="3d")
    X, T_mesh = np.meshgrid(x, t)
    surf1 = ax1.plot_surface(X, T_mesh, u_fractional.T, cmap="viridis", alpha=0.8)
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Time t")
    ax1.set_zlabel("u(x,t)")
    ax1.set_title("Fractional (with memory)")
    plt.colorbar(surf1, ax=ax1)

    ax2 = plt.subplot(2, 4, 2, projection="3d")
    surf2 = ax2.plot_surface(X, T_mesh, u_normal.T, cmap="plasma", alpha=0.8)
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Time t")
    ax2.set_zlabel("u(x,t)")
    ax2.set_title("Normal (no memory)")
    plt.colorbar(surf2, ax=ax2)

    # Time evolution at different positions
    positions = [0.3, 0.5, 0.7]
    ax3 = plt.subplot(2, 4, 3)
    for pos in positions:
        idx = np.argmin(np.abs(x - pos))
        ax3.plot(t, u_fractional[idx, :], linewidth=2, label=f"Fractional x={pos}")
        ax3.plot(t, u_normal[idx, :], "--", linewidth=2, label=f"Normal x={pos}")
    ax3.set_xlabel("Time t")
    ax3.set_ylabel("u(x,t)")
    ax3.set_title("Time Evolution Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Spatial profiles at different times
    times = [0.5, 1.0, 1.5, 2.0]
    ax4 = plt.subplot(2, 4, 4)
    for time_val in times:
        idx = np.argmin(np.abs(t - time_val))
        ax4.plot(x, u_fractional[:, idx], linewidth=2, label=f"Fractional t={time_val}")
        ax4.plot(x, u_normal[:, idx], "--", linewidth=2, label=f"Normal t={time_val}")
    ax4.set_xlabel("Position x")
    ax4.set_ylabel("u(x,t)")
    ax4.set_title("Spatial Profiles Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Memory kernel visualization
    ax5 = plt.subplot(2, 4, 5)
    t_kernel = np.linspace(0.01, 2, 100)  # Avoid t=0 to prevent division by zero
    # Fractional memory kernel: t^(α-1)
    memory_kernel = t_kernel ** (alpha - 1) / gamma(alpha)
    ax5.plot(t_kernel, memory_kernel, "b-", linewidth=2, label=f"Fractional α={alpha}")
    # Normal memory kernel (delta function)
    ax5.axhline(y=0, color="r", linestyle="--", linewidth=2, label="Normal (no memory)")
    ax5.set_xlabel("Time t")
    ax5.set_ylabel("Memory Kernel")
    ax5.set_title("Memory Kernel Comparison")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Contour comparison
    ax6 = plt.subplot(2, 4, 6)
    contour1 = ax6.contourf(X, T_mesh, u_fractional.T, levels=20, cmap="viridis")
    ax6.set_xlabel("Position x")
    ax6.set_ylabel("Time t")
    ax6.set_title("Fractional: Contour")
    plt.colorbar(contour1, ax=ax6)

    ax7 = plt.subplot(2, 4, 7)
    contour2 = ax7.contourf(X, T_mesh, u_normal.T, levels=20, cmap="plasma")
    ax7.set_xlabel("Position x")
    ax7.set_ylabel("Time t")
    ax7.set_title("Normal: Contour")
    plt.colorbar(contour2, ax=ax7)

    # Difference plot
    ax8 = plt.subplot(2, 4, 8)
    diff = u_fractional - u_normal
    contour3 = ax8.contourf(X, T_mesh, diff.T, levels=20, cmap="RdBu")
    ax8.set_xlabel("Position x")
    ax8.set_ylabel("Time t")
    ax8.set_title("Difference (Fractional - Normal)")
    plt.colorbar(contour3, ax=ax8)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "memory_effects.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Memory effects demo completed!")


def levy_flights_demo():
    """Demonstrate Lévy flights and non-local spatial coupling."""
    print("\n🦅 Lévy Flights Demo")
    print("=" * 50)

    # Problem parameters
    L = 4.0
    T = 2.0
    Nx = 200
    Nt = 100

    # Create solver
    solver = FractionalDiffusionSolver()

    # Define initial condition: localized pulse
    def initial_condition(x):
        return np.exp(-(((x - L / 2) / 0.1) ** 2))

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    # Compare different spatial fractional orders (Lévy flights)
    betas = [1.5, 1.8, 2.0, 2.2, 2.5]  # Spatial fractional orders
    solutions = []

    print(f"Comparing Lévy flights with different spatial orders:")
    print(f"  β < 2: Lévy flights (long-range jumps)")
    print(f"  β = 2: Normal diffusion")
    print(f"  β > 2: Subdiffusion in space")

    for beta in betas:
        print(f"\n🧪 Solving with β = {beta}...")
        x, t, u = solver.solve(
            x_span=(0, L),
            t_span=(0, T),
            initial_condition=initial_condition,
            boundary_conditions=(boundary_condition_left, boundary_condition_right),
            alpha=1.0,  # Normal time evolution
            beta=beta,  # Spatial fractional order
            nx=Nx,
            nt=Nt,
        )
        solutions.append((beta, x, t, u))

    # Plot results
    plt.figure(figsize=(20, 15))

    # 3D surfaces for different betas
    for i, (beta, x, t, u) in enumerate(solutions):
        ax = plt.subplot(3, 3, i + 1, projection="3d")
        X, T_mesh = np.meshgrid(x, t)
        surf = ax.plot_surface(X, T_mesh, u.T, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Time t")
        ax.set_zlabel("u(x,t)")
        ax.set_title(f"β = {beta}")
        plt.colorbar(surf, ax=ax)

    # Time evolution at center
    ax_center = plt.subplot(3, 3, 6)
    center_idx = len(x) // 2
    for beta, x, t, u in solutions:
        ax_center.plot(t, u[center_idx, :], linewidth=2, label=f"β = {beta}")
    ax_center.set_xlabel("Time t")
    ax_center.set_ylabel("u(x,t) at x = L/2")
    ax_center.set_title("Time Evolution at Center")
    ax_center.legend()
    ax_center.grid(True, alpha=0.3)

    # Final spatial profiles (log scale for tails)
    ax_final = plt.subplot(3, 3, 7)
    for beta, x, t, u in solutions:
        profile = u[:, -1]
        # Remove zeros for log plot
        mask = profile > 1e-10
        ax_final.semilogy(x[mask], profile[mask], linewidth=2, label=f"β = {beta}")
    ax_final.set_xlabel("Position x")
    ax_final.set_ylabel("u(x,T) (log scale)")
    ax_final.set_title("Final Profiles (Log Scale)")
    ax_final.legend()
    ax_final.grid(True, alpha=0.3)

    # Tail analysis
    ax_tail = plt.subplot(3, 3, 8)
    for beta, x, t, u in solutions:
        profile = u[:, -1]
        # Calculate tail exponent
        center_idx = len(x) // 2
        right_tail = profile[center_idx:]
        x_right = x[center_idx:]

        # Fit power law to tail
        if len(right_tail) > 10:
            # Find where tail starts (after peak)
            peak_idx = np.argmax(right_tail)
            if peak_idx < len(right_tail) - 5:
                tail_x = x_right[peak_idx:]
                tail_y = right_tail[peak_idx:]
                # Remove zeros
                mask = tail_y > 1e-10
                if np.sum(mask) > 5:
                    log_x = np.log(tail_x[mask])
                    log_y = np.log(tail_y[mask])
                    coeffs = np.polyfit(log_x, log_y, 1)
                    exponent = -coeffs[
                        0
                    ]  # Negative because we're fitting u ~ x^(-exponent)
                    ax_tail.loglog(
                        tail_x[mask],
                        tail_y[mask],
                        "o-",
                        linewidth=2,
                        label=f"β = {beta}, exponent = {exponent:.2f}",
                    )

    ax_tail.set_xlabel("Position x")
    ax_tail.set_ylabel("u(x,T)")
    ax_tail.set_title("Tail Analysis")
    ax_tail.legend()
    ax_tail.grid(True, alpha=0.3)

    # Contour comparison
    ax_contour = plt.subplot(3, 3, 9)
    # Show Lévy flight vs normal diffusion
    levy_idx = 0  # β = 1.5
    normal_idx = 2  # β = 2.0

    X, T_mesh = np.meshgrid(x, t)
    contour1 = ax_contour.contour(
        X, T_mesh, solutions[levy_idx][3].T, levels=10, colors="blue", alpha=0.7
    )
    contour2 = ax_contour.contour(
        X, T_mesh, solutions[normal_idx][3].T, levels=10, colors="red", alpha=0.7
    )
    ax_contour.set_xlabel("Position x")
    ax_contour.set_ylabel("Time t")
    ax_contour.set_title("Lévy Flight vs Normal Diffusion")
    ax_contour.clabel(contour1, inline=True, fontsize=8)
    ax_contour.clabel(contour2, inline=True, fontsize=8)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "levy_flights.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Lévy flights demo completed!")


def heavy_tailed_demo():
    """Demonstrate heavy-tailed distributions in waiting times and jump lengths."""
    print("\n📊 Heavy-Tailed Distributions Demo")
    print("=" * 50)

    # Problem parameters
    L = 2.0
    T = 3.0
    Nx = 100
    Nt = 200

    # Create solver
    solver = FractionalDiffusionSolver()

    # Define initial condition: multiple pulses
    def initial_condition(x):
        return np.exp(-(((x - 0.3) / 0.05) ** 2)) + 0.5 * np.exp(
            -(((x - 1.7) / 0.05) ** 2)
        )

    # Define boundary conditions
    def boundary_condition_left(t):
        return 0.0

    def boundary_condition_right(t):
        return 0.0

    # Compare different scenarios
    scenarios = [
        (0.3, 2.0, "Heavy-tailed waiting times"),
        (1.0, 1.5, "Heavy-tailed jump lengths"),
        (0.5, 1.8, "Both heavy-tailed"),
        (1.0, 2.0, "Normal diffusion"),
    ]

    solutions = []

    print(f"Comparing different heavy-tailed scenarios:")
    for alpha, beta, description in scenarios:
        print(f"\n🧪 {description} (α = {alpha}, β = {beta})...")
        x, t, u = solver.solve(
            x_span=(0, L),
            t_span=(0, T),
            initial_condition=initial_condition,
            boundary_conditions=(boundary_condition_left, boundary_condition_right),
            alpha=alpha,
            beta=beta,
            nx=Nx,
            nt=Nt,
        )
        solutions.append((alpha, beta, description, x, t, u))

    # Plot results
    plt.figure(figsize=(20, 15))

    # 3D surfaces
    for i, (alpha, beta, description, x, t, u) in enumerate(solutions):
        ax = plt.subplot(3, 4, i + 1, projection="3d")
        X, T_mesh = np.meshgrid(x, t)
        surf = ax.plot_surface(X, T_mesh, u.T, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Time t")
        ax.set_zlabel("u(x,t)")
        ax.set_title(f"{description}\nα = {alpha}, β = {beta}")
        plt.colorbar(surf, ax=ax)

    # Time evolution comparison
    ax_time = plt.subplot(3, 4, 5)
    center_idx = len(x) // 2
    for alpha, beta, description, x, t, u in solutions:
        ax_time.plot(t, u[center_idx, :], linewidth=2, label=f"{description}")
    ax_time.set_xlabel("Time t")
    ax_time.set_ylabel("u(x,t) at x = L/2")
    ax_time.set_title("Time Evolution Comparison")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    # Spatial profiles at different times
    times = [0.5, 1.0, 1.5, 2.0]
    ax_spatial = plt.subplot(3, 4, 6)
    for time_val in times:
        idx = np.argmin(np.abs(t - time_val))
        ax_spatial.plot(x, u[:, idx], linewidth=2, label=f"t = {time_val}")
    ax_spatial.set_xlabel("Position x")
    ax_spatial.set_ylabel("u(x,t)")
    ax_spatial.set_title("Spatial Profiles (Heavy-tailed)")
    ax_spatial.legend()
    ax_spatial.grid(True, alpha=0.3)

    # Waiting time distribution comparison
    ax_waiting = plt.subplot(3, 4, 7)
    t_waiting = np.linspace(0.1, 5, 100)
    for alpha, beta, description, x, t, u in solutions:
        if alpha < 1:  # Heavy-tailed waiting times
            # Power law: P(t) ~ t^(-α-1)
            try:
                waiting_dist = t_waiting ** (-alpha - 1) / gamma(1 - alpha)
                ax_waiting.loglog(
                    t_waiting, waiting_dist, linewidth=2, label=f"α = {alpha}"
                )
            except:
                # Fallback for problematic values
                waiting_dist = t_waiting ** (-alpha - 1)
                ax_waiting.loglog(
                    t_waiting, waiting_dist, linewidth=2, label=f"α = {alpha}"
                )
    ax_waiting.set_xlabel("Waiting Time t")
    ax_waiting.set_ylabel("P(t)")
    ax_waiting.set_title("Waiting Time Distribution")
    ax_waiting.legend()
    ax_waiting.grid(True, alpha=0.3)

    # Jump length distribution comparison
    ax_jump = plt.subplot(3, 4, 8)
    x_jump = np.linspace(0.1, 2, 100)
    for alpha, beta, description, x, t, u in solutions:
        if beta < 2:  # Heavy-tailed jump lengths
            # Power law: P(x) ~ x^(-β-1)
            try:
                jump_dist = x_jump ** (-beta - 1) / gamma(1 - beta)
                ax_jump.loglog(x_jump, jump_dist, linewidth=2, label=f"β = {beta}")
            except:
                # Fallback for problematic values
                jump_dist = x_jump ** (-beta - 1)
                ax_jump.loglog(x_jump, jump_dist, linewidth=2, label=f"β = {beta}")
    ax_jump.set_xlabel("Jump Length x")
    ax_jump.set_ylabel("P(x)")
    ax_jump.set_title("Jump Length Distribution")
    ax_jump.legend()
    ax_jump.grid(True, alpha=0.3)

    # Contour plots
    for i, (alpha, beta, description, x, t, u) in enumerate(solutions):
        ax = plt.subplot(3, 4, 9 + i)
        X, T_mesh = np.meshgrid(x, t)
        contour = ax.contourf(X, T_mesh, u.T, levels=20, cmap="viridis")
        ax.set_xlabel("Position x")
        ax.set_ylabel("Time t")
        ax.set_title(f"{description}\nContour")
        plt.colorbar(contour, ax=ax)

    plt.tight_layout()
    output_dir = ensure_output_dir()
    plt.savefig(
        os.path.join(output_dir, "heavy_tailed.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Heavy-tailed distributions demo completed!")


def main():
    """Run all advanced PDE solver examples."""
    print("🚀 Advanced Fractional PDE Solver Examples")
    print("=" * 60)

    # Run examples
    fractional_diffusion_equation()
    fractional_wave_equation()
    L1_L2_schemes_comparison()
    predictor_corrector_demo()
    anomalous_transport_demo()
    memory_effects_demo()
    levy_flights_demo()
    heavy_tailed_demo()

    print("\n🎉 All advanced PDE solver examples completed!")
    print("\n📁 Generated plots saved in 'examples/advanced_applications/' directory")


if __name__ == "__main__":
    main()
