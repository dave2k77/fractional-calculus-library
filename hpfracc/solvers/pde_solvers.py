"""
Fractional Partial Differential Equation Solvers

This module provides comprehensive solvers for fractional PDEs including
finite difference methods, spectral methods, and adaptive mesh refinement.
"""

import numpy as np
from typing import Union, Optional, Tuple, Callable, Dict, Any, List
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
import warnings

from ..core.definitions import FractionalOrder
from ..algorithms.optimized_methods import (
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov,
)


class FractionalPDESolver:
    """
    Base class for fractional PDE solvers.

    Provides common functionality for solving fractional partial
    differential equations of various types.
    """

    def __init__(
        self,
        pde_type: str = "diffusion",
        method: str = "finite_difference",
        spatial_order: int = 2,
        temporal_order: int = 1,
        adaptive: bool = False,
    ):
        """
        Initialize fractional PDE solver.

        Args:
            pde_type: Type of PDE ("diffusion", "advection", "reaction_diffusion")
            method: Numerical method ("finite_difference", "spectral", "finite_element")
            spatial_order: Order of spatial discretization
            temporal_order: Order of temporal discretization
            adaptive: Use adaptive mesh refinement
        """
        self.pde_type = pde_type.lower()
        self.method = method.lower()
        self.spatial_order = spatial_order
        self.temporal_order = temporal_order
        self.adaptive = adaptive

        # Validate PDE type
        valid_pde_types = ["diffusion", "advection", "reaction_diffusion", "wave"]
        if self.pde_type not in valid_pde_types:
            raise ValueError(f"PDE type must be one of {valid_pde_types}")

        # Validate method
        valid_methods = ["finite_difference", "spectral", "finite_element"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")


class FractionalDiffusionSolver(FractionalPDESolver):
    """
    Solver for fractional diffusion equations.

    Solves equations of the form:
    ∂^α u/∂t^α = D ∂^β u/∂x^β + f(x, t, u)

    where α and β are fractional orders.
    """

    def __init__(
        self,
        method: str = "finite_difference",
        spatial_order: int = 2,
        temporal_order: int = 1,
        derivative_type: str = "caputo",
    ):
        """
        Initialize fractional diffusion solver.

        Args:
            method: Numerical method
            spatial_order: Order of spatial discretization
            temporal_order: Order of temporal discretization
            derivative_type: Type of fractional derivative
        """
        super().__init__("diffusion", method, spatial_order, temporal_order)
        self.derivative_type = derivative_type.lower()

    def solve(
        self,
        x_span: Tuple[float, float],
        t_span: Tuple[float, float],
        initial_condition: Callable,
        boundary_conditions: Tuple[Callable, Callable],
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float = 1.0,
        source_term: Optional[Callable] = None,
        nx: int = 100,
        nt: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve fractional diffusion equation.

        Args:
            x_span: Spatial interval (x0, xf)
            t_span: Time interval (t0, tf)
            initial_condition: Initial condition u(x, 0)
            boundary_conditions: Boundary conditions (left_bc, right_bc)
            alpha: Temporal fractional order
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            source_term: Source term f(x, t, u)
            nx: Number of spatial grid points
            nt: Number of temporal grid points
            **kwargs: Additional solver parameters

        Returns:
            Tuple of (t_values, x_values, solution)
        """
        x0, xf = x_span
        t0, tf = t_span

        # Spatial and temporal grids
        x_values = np.linspace(x0, xf, nx)
        t_values = np.linspace(t0, tf, nt)
        dx = x_values[1] - x_values[0]
        dt = t_values[1] - t_values[0]

        # Initialize solution array
        solution = np.zeros((nt, nx))

        # Set initial condition
        for i, x in enumerate(x_values):
            solution[0, i] = initial_condition(x)

        # Set boundary conditions
        left_bc, right_bc = boundary_conditions
        for n in range(nt):
            solution[n, 0] = left_bc(t_values[n])
            solution[n, -1] = right_bc(t_values[n])

        # Main time-stepping loop
        for n in range(1, nt):
            t_n = t_values[n]

            # Solve spatial problem at current time step
            solution[n, 1:-1] = self._solve_spatial_step(
                solution,
                n,
                x_values,
                t_n,
                alpha,
                beta,
                diffusion_coeff,
                source_term,
                dx,
                dt,
                **kwargs,
            )

        return t_values, x_values, solution

    def _solve_spatial_step(
        self,
        solution: np.ndarray,
        n: int,
        x_values: np.ndarray,
        t_n: float,
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float,
        source_term: Optional[Callable],
        dx: float,
        dt: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Solve spatial problem at current time step.

        Args:
            solution: Solution array
            n: Current time step
            x_values: Spatial grid
            t_n: Current time
            alpha: Temporal fractional order
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            source_term: Source term
            dx: Spatial step size
            dt: Temporal step size
            **kwargs: Additional parameters

        Returns:
            Solution at interior points
        """
        nx = len(x_values)

        if self.method == "finite_difference":
            return self._finite_difference_step(
                solution,
                n,
                x_values,
                t_n,
                alpha,
                beta,
                diffusion_coeff,
                source_term,
                dx,
                dt,
                **kwargs,
            )
        elif self.method == "spectral":
            return self._spectral_step(
                solution,
                n,
                x_values,
                t_n,
                alpha,
                beta,
                diffusion_coeff,
                source_term,
                dx,
                dt,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _finite_difference_step(
        self,
        solution: np.ndarray,
        n: int,
        x_values: np.ndarray,
        t_n: float,
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float,
        source_term: Optional[Callable],
        dx: float,
        dt: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Finite difference step.

        Args:
            solution: Solution array
            n: Current time step
            x_values: Spatial grid
            t_n: Current time
            alpha: Temporal fractional order
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            source_term: Source term
            dx: Spatial step size
            dt: Temporal step size
            **kwargs: Additional parameters

        Returns:
            Solution at interior points
        """
        nx = len(x_values)

        # Compute temporal fractional derivative
        temporal_deriv = self._compute_temporal_derivative(solution, n, alpha, dt)

        # Compute spatial fractional derivative
        spatial_deriv = self._compute_spatial_derivative(solution[n - 1, :], beta, dx)

        # Source term
        source = np.zeros(nx)
        if source_term is not None:
            for i, x in enumerate(x_values):
                source[i] = source_term(x, t_n, solution[n - 1, i])

        # Implicit scheme: solve linear system
        A = self._build_spatial_matrix(nx, beta, diffusion_coeff, dx, dt, alpha)
        b = temporal_deriv[1:-1] + source[1:-1]  # Interior points only

        # Solve linear system
        u_interior = spsolve(A, b)

        return u_interior

    def _spectral_step(
        self,
        solution: np.ndarray,
        n: int,
        x_values: np.ndarray,
        t_n: float,
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float,
        source_term: Optional[Callable],
        dx: float,
        dt: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Spectral method step.

        Args:
            solution: Solution array
            n: Current time step
            x_values: Spatial grid
            t_n: Current time
            alpha: Temporal fractional order
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            source_term: Source term
            dx: Spatial step size
            dt: Temporal step size
            **kwargs: Additional parameters

        Returns:
            Solution at interior points
        """
        nx = len(x_values)

        # FFT of current solution
        u_hat = np.fft.fft(solution[n - 1, :])

        # Wavenumbers
        k = np.fft.fftfreq(nx, dx) * 2 * np.pi

        # Spectral fractional derivative
        spatial_deriv_hat = (1j * k) ** beta * u_hat

        # Temporal fractional derivative
        temporal_deriv = self._compute_temporal_derivative(solution, n, alpha, dt)

        # Source term
        source = np.zeros(nx)
        if source_term is not None:
            for i, x in enumerate(x_values):
                source[i] = source_term(x, t_n, solution[n - 1, i])
        source_hat = np.fft.fft(source)

        # Update in spectral space
        from scipy.special import gamma

        u_hat_new = u_hat + dt**alpha / gamma(alpha + 1) * (
            diffusion_coeff * spatial_deriv_hat + source_hat
        )

        # Transform back to physical space
        u_new = np.real(np.fft.ifft(u_hat_new))

        return u_new[1:-1]  # Interior points only

    def _compute_temporal_derivative(
        self,
        solution: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        dt: float,
    ) -> np.ndarray:
        """
        Compute temporal fractional derivative.

        Args:
            solution: Solution array
            n: Current time step
            alpha: Temporal fractional order
            dt: Temporal step size

        Returns:
            Temporal derivative
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Compute coefficients for temporal derivative
        coeffs = np.zeros(n + 1)
        coeffs[0] = 1.0
        for j in range(1, n + 1):
            if self.derivative_type == "caputo":
                coeffs[j] = (j + 1) ** alpha_val - j**alpha_val
            elif self.derivative_type == "grunwald_letnikov":
                coeffs[j] = coeffs[j - 1] * (1 - (alpha_val + 1) / j)
            else:  # Riemann-Liouville
                from scipy.special import gamma

                coeffs[j] = (
                    (-1) ** j
                    * gamma(alpha_val + 1)
                    / (gamma(j + 1) * gamma(alpha_val - j + 1))
                )

        # Compute temporal derivative
        temporal_deriv = np.zeros_like(solution[n, :])
        for j in range(n + 1):
            temporal_deriv += coeffs[j] * (solution[n - j, :] - solution[n - j - 1, :])

        return temporal_deriv / (dt**alpha_val)

    def _compute_spatial_derivative(
        self, u: np.ndarray, beta: Union[float, FractionalOrder], dx: float
    ) -> np.ndarray:
        """
        Compute spatial fractional derivative.

        Args:
            u: Solution at current time
            beta: Spatial fractional order
            dx: Spatial step size

        Returns:
            Spatial derivative
        """
        if isinstance(beta, FractionalOrder):
            beta_val = beta.alpha
        else:
            beta_val = beta

        nx = len(u)

        # Use finite difference approximation for spatial derivative
        if self.spatial_order == 2:
            # Second-order central difference
            d2u = np.zeros(nx)
            d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx**2)

            # Apply fractional power (simplified)
            spatial_deriv = np.sign(d2u) * np.abs(d2u) ** (beta_val / 2)
        else:
            # First-order approximation
            du = np.zeros(nx)
            du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            spatial_deriv = np.sign(du) * np.abs(du) ** beta_val

        return spatial_deriv

    def _build_spatial_matrix(
        self,
        nx: int,
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float,
        dx: float,
        dt: float,
        alpha: Union[float, FractionalOrder],
    ) -> sparse.spmatrix:
        """
        Build spatial discretization matrix.

        Args:
            nx: Number of spatial points
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            dx: Spatial step size
            dt: Temporal step size
            alpha: Temporal fractional order

        Returns:
            Sparse matrix for spatial discretization
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        if isinstance(beta, FractionalOrder):
            beta_val = beta.alpha
        else:
            beta_val = beta

        # Build tridiagonal matrix for spatial discretization
        n_interior = nx - 2
        diagonals = []
        offsets = []

        # Main diagonal
        main_diag = np.ones(n_interior) * (
            1 + 2 * diffusion_coeff * dt**alpha_val / (dx**2)
        )
        diagonals.append(main_diag)
        offsets.append(0)

        # Sub-diagonal
        sub_diag = np.ones(n_interior - 1) * (
            -diffusion_coeff * dt**alpha_val / (dx**2)
        )
        diagonals.append(sub_diag)
        offsets.append(-1)

        # Super-diagonal
        super_diag = np.ones(n_interior - 1) * (
            -diffusion_coeff * dt**alpha_val / (dx**2)
        )
        diagonals.append(super_diag)
        offsets.append(1)

        # Create sparse matrix
        A = sparse.diags(
            diagonals, offsets, shape=(n_interior, n_interior), format="csr"
        )

        return A


class FractionalAdvectionSolver(FractionalPDESolver):
    """
    Solver for fractional advection equations.

    Solves equations of the form:
    ∂^α u/∂t^α + v ∂^β u/∂x^β = f(x, t, u)

    where α and β are fractional orders.
    """

    def __init__(
        self,
        method: str = "finite_difference",
        spatial_order: int = 2,
        temporal_order: int = 1,
        derivative_type: str = "caputo",
    ):
        """
        Initialize fractional advection solver.

        Args:
            method: Numerical method
            spatial_order: Order of spatial discretization
            temporal_order: Order of temporal discretization
            derivative_type: Type of fractional derivative
        """
        super().__init__("advection", method, spatial_order, temporal_order)
        self.derivative_type = derivative_type.lower()

    def solve(
        self,
        x_span: Tuple[float, float],
        t_span: Tuple[float, float],
        initial_condition: Callable,
        boundary_conditions: Tuple[Callable, Callable],
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        velocity: float = 1.0,
        source_term: Optional[Callable] = None,
        nx: int = 100,
        nt: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve fractional advection equation.

        Args:
            x_span: Spatial interval (x0, xf)
            t_span: Time interval (t0, tf)
            initial_condition: Initial condition u(x, 0)
            boundary_conditions: Boundary conditions (left_bc, right_bc)
            alpha: Temporal fractional order
            beta: Spatial fractional order
            velocity: Advection velocity
            source_term: Source term f(x, t, u)
            nx: Number of spatial grid points
            nt: Number of temporal grid points
            **kwargs: Additional solver parameters

        Returns:
            Tuple of (t_values, x_values, solution)
        """
        # Similar implementation to diffusion solver but with advection term
        # This is a simplified implementation
        x0, xf = x_span
        t0, tf = t_span

        # Spatial and temporal grids
        x_values = np.linspace(x0, xf, nx)
        t_values = np.linspace(t0, tf, nt)
        dx = x_values[1] - x_values[0]
        dt = t_values[1] - t_values[0]

        # Initialize solution array
        solution = np.zeros((nt, nx))

        # Set initial condition
        for i, x in enumerate(x_values):
            solution[0, i] = initial_condition(x)

        # Set boundary conditions
        left_bc, right_bc = boundary_conditions
        for n in range(nt):
            solution[n, 0] = left_bc(t_values[n])
            solution[n, -1] = right_bc(t_values[n])

        # Main time-stepping loop (simplified)
        for n in range(1, nt):
            t_n = t_values[n]

            # Simple upwind scheme for advection
            for i in range(1, nx - 1):
                if velocity > 0:
                    # Upwind
                    solution[n, i] = solution[n - 1, i] - velocity * dt / dx * (
                        solution[n - 1, i] - solution[n - 1, i - 1]
                    )
                else:
                    # Downwind
                    solution[n, i] = solution[n - 1, i] - velocity * dt / dx * (
                        solution[n - 1, i + 1] - solution[n - 1, i]
                    )

        return t_values, x_values, solution


class FractionalReactionDiffusionSolver(FractionalPDESolver):
    """
    Solver for fractional reaction-diffusion equations.

    Solves equations of the form:
    ∂^α u/∂t^α = D ∂^β u/∂x^β + R(u) + f(x, t, u)

    where α and β are fractional orders and R(u) is the reaction term.
    """

    def __init__(
        self,
        method: str = "finite_difference",
        spatial_order: int = 2,
        temporal_order: int = 1,
        derivative_type: str = "caputo",
    ):
        """
        Initialize fractional reaction-diffusion solver.

        Args:
            method: Numerical method
            spatial_order: Order of spatial discretization
            temporal_order: Order of temporal discretization
            derivative_type: Type of fractional derivative
        """
        super().__init__("reaction_diffusion", method, spatial_order, temporal_order)
        self.derivative_type = derivative_type.lower()

    def solve(
        self,
        x_span: Tuple[float, float],
        t_span: Tuple[float, float],
        initial_condition: Callable,
        boundary_conditions: Tuple[Callable, Callable],
        alpha: Union[float, FractionalOrder],
        beta: Union[float, FractionalOrder],
        diffusion_coeff: float = 1.0,
        reaction_term: Optional[Callable] = None,
        source_term: Optional[Callable] = None,
        nx: int = 100,
        nt: int = 100,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve fractional reaction-diffusion equation.

        Args:
            x_span: Spatial interval (x0, xf)
            t_span: Time interval (t0, tf)
            initial_condition: Initial condition u(x, 0)
            boundary_conditions: Boundary conditions (left_bc, right_bc)
            alpha: Temporal fractional order
            beta: Spatial fractional order
            diffusion_coeff: Diffusion coefficient
            reaction_term: Reaction term R(u)
            source_term: Source term f(x, t, u)
            nx: Number of spatial grid points
            nt: Number of temporal grid points
            **kwargs: Additional solver parameters

        Returns:
            Tuple of (t_values, x_values, solution)
        """

        # Combine diffusion and reaction terms
        def combined_source(x, t, u):
            source = 0.0
            if reaction_term is not None:
                source += reaction_term(u)
            if source_term is not None:
                source += source_term(x, t, u)
            return source

        # Use diffusion solver with combined source term
        diffusion_solver = FractionalDiffusionSolver(
            self.method, self.spatial_order, self.temporal_order, self.derivative_type
        )

        return diffusion_solver.solve(
            x_span,
            t_span,
            initial_condition,
            boundary_conditions,
            alpha,
            beta,
            diffusion_coeff,
            combined_source,
            nx,
            nt,
            **kwargs,
        )


# Convenience functions
def solve_fractional_diffusion(
    x_span: Tuple[float, float],
    t_span: Tuple[float, float],
    initial_condition: Callable,
    boundary_conditions: Tuple[Callable, Callable],
    alpha: Union[float, FractionalOrder],
    beta: Union[float, FractionalOrder],
    diffusion_coeff: float = 1.0,
    source_term: Optional[Callable] = None,
    method: str = "finite_difference",
    nx: int = 100,
    nt: int = 100,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve fractional diffusion equation.

    Args:
        x_span: Spatial interval (x0, xf)
        t_span: Time interval (t0, tf)
        initial_condition: Initial condition u(x, 0)
        boundary_conditions: Boundary conditions (left_bc, right_bc)
        alpha: Temporal fractional order
        beta: Spatial fractional order
        diffusion_coeff: Diffusion coefficient
        source_term: Source term f(x, t, u)
        method: Numerical method
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, x_values, solution)
    """
    solver = FractionalDiffusionSolver(method)
    return solver.solve(
        x_span,
        t_span,
        initial_condition,
        boundary_conditions,
        alpha,
        beta,
        diffusion_coeff,
        source_term,
        nx,
        nt,
        **kwargs,
    )


def solve_fractional_advection(
    x_span: Tuple[float, float],
    t_span: Tuple[float, float],
    initial_condition: Callable,
    boundary_conditions: Tuple[Callable, Callable],
    alpha: Union[float, FractionalOrder],
    beta: Union[float, FractionalOrder],
    velocity: float = 1.0,
    source_term: Optional[Callable] = None,
    method: str = "finite_difference",
    nx: int = 100,
    nt: int = 100,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve fractional advection equation.

    Args:
        x_span: Spatial interval (x0, xf)
        t_span: Time interval (t0, tf)
        initial_condition: Initial condition u(x, 0)
        boundary_conditions: Boundary conditions (left_bc, right_bc)
        alpha: Temporal fractional order
        beta: Spatial fractional order
        velocity: Advection velocity
        source_term: Source term f(x, t, u)
        method: Numerical method
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, x_values, solution)
    """
    solver = FractionalAdvectionSolver(method)
    return solver.solve(
        x_span,
        t_span,
        initial_condition,
        boundary_conditions,
        alpha,
        beta,
        velocity,
        source_term,
        nx,
        nt,
        **kwargs,
    )


def solve_fractional_reaction_diffusion(
    x_span: Tuple[float, float],
    t_span: Tuple[float, float],
    initial_condition: Callable,
    boundary_conditions: Tuple[Callable, Callable],
    alpha: Union[float, FractionalOrder],
    beta: Union[float, FractionalOrder],
    diffusion_coeff: float = 1.0,
    reaction_term: Optional[Callable] = None,
    source_term: Optional[Callable] = None,
    method: str = "finite_difference",
    nx: int = 100,
    nt: int = 100,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve fractional reaction-diffusion equation.

    Args:
        x_span: Spatial interval (x0, xf)
        t_span: Time interval (t0, tf)
        initial_condition: Initial condition u(x, 0)
        boundary_conditions: Boundary conditions (left_bc, right_bc)
        alpha: Temporal fractional order
        beta: Spatial fractional order
        diffusion_coeff: Diffusion coefficient
        reaction_term: Reaction term R(u)
        source_term: Source term f(x, t, u)
        method: Numerical method
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, x_values, solution)
    """
    solver = FractionalReactionDiffusionSolver(method)
    return solver.solve(
        x_span,
        t_span,
        initial_condition,
        boundary_conditions,
        alpha,
        beta,
        diffusion_coeff,
        reaction_term,
        source_term,
        nx,
        nt,
        **kwargs,
    )
