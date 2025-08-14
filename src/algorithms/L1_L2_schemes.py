"""
L1 and L2 Schemes for Time-Fractional PDEs

This module implements L1 and L2 numerical schemes for solving time-fractional
partial differential equations, including finite difference methods, stability
analysis, and optimized implementations using JAX and NUMBA.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numba import jit, prange
from typing import Union, Optional, Tuple, Callable
from scipy import linalg

from src.core.definitions import FractionalOrder
from src.special import gamma


class L1L2Schemes:
    """
    L1 and L2 schemes for time-fractional PDEs.
    
    This class provides numerical schemes for solving time-fractional partial
    differential equations using L1 and L2 finite difference methods.
    """
    
    def __init__(self, scheme: str = "l1"):
        """
        Initialize L1/L2 scheme solver.
        
        Args:
            scheme: Numerical scheme ("l1", "l2", "l2_1_sigma")
        """
        self.scheme = scheme.lower()
        
        # Validate scheme
        valid_schemes = ["l1", "l2", "l2_1_sigma", "l2_1_theta"]
        if self.scheme not in valid_schemes:
            raise ValueError(f"Scheme must be one of {valid_schemes}")
    
    def solve_time_fractional_pde(self, 
                                 initial_condition: np.ndarray,
                                 boundary_conditions: Tuple[Callable, Callable],
                                 alpha: Union[float, FractionalOrder],
                                 t_final: float,
                                 dt: float,
                                 dx: float,
                                 diffusion_coeff: float = 1.0,
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve time-fractional diffusion equation using L1/L2 scheme.
        
        Solves: ∂^α u/∂t^α = D ∂²u/∂x²
        
        Args:
            initial_condition: Initial condition u(x, 0)
            boundary_conditions: Tuple of (left_bc, right_bc) functions
            alpha: Fractional order (0 < α < 1)
            t_final: Final time
            dt: Time step size
            dx: Spatial step size
            diffusion_coeff: Diffusion coefficient D
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (time_points, spatial_points, solution_matrix)
        """
        if self.scheme == "l1":
            return self._solve_l1_scheme(initial_condition, boundary_conditions,
                                       alpha, t_final, dt, dx, diffusion_coeff, **kwargs)
        elif self.scheme == "l2":
            return self._solve_l2_scheme(initial_condition, boundary_conditions,
                                       alpha, t_final, dt, dx, diffusion_coeff, **kwargs)
        elif self.scheme == "l2_1_sigma":
            return self._solve_l2_1_sigma_scheme(initial_condition, boundary_conditions,
                                               alpha, t_final, dt, dx, diffusion_coeff, **kwargs)
        elif self.scheme == "l2_1_theta":
            return self._solve_l2_1_theta_scheme(initial_condition, boundary_conditions,
                                               alpha, t_final, dt, dx, diffusion_coeff, **kwargs)
    
    def _solve_l1_scheme(self, 
                        initial_condition: np.ndarray,
                        boundary_conditions: Tuple[Callable, Callable],
                        alpha: Union[float, FractionalOrder],
                        t_final: float,
                        dt: float,
                        dx: float,
                        diffusion_coeff: float,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve using L1 scheme (first-order accurate).
        
        Suitable for 0 < α < 1.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha
        
        if alpha_val >= 1:
            raise ValueError("L1 scheme requires 0 < α < 1")
        
        # Setup grid
        N_t = int(t_final / dt) + 1
        N_x = len(initial_condition)
        
        t_points = np.linspace(0, t_final, N_t)
        x_points = np.linspace(0, (N_x - 1) * dx, N_x)
        
        # Initialize solution matrix
        u = np.zeros((N_t, N_x))
        u[0, :] = initial_condition
        
        # L1 coefficients
        coeffs = self._compute_l1_coefficients(alpha_val, N_t)
        
        # Spatial discretization matrix (central difference for second derivative)
        A = self._build_spatial_matrix(N_x, dx, diffusion_coeff)
        
        # Time stepping
        for n in range(1, N_t):
            # Right-hand side
            rhs = np.zeros(N_x)
            
            # History term
            for j in range(1, n + 1):
                rhs += coeffs[j] * (u[n-j+1, :] - u[n-j, :])
            
            # Apply boundary conditions
            left_bc, right_bc = boundary_conditions
            rhs[0] = left_bc(t_points[n])
            rhs[-1] = right_bc(t_points[n])
            
            # Solve linear system
            u[n, :] = linalg.solve(A, rhs)
        
        return t_points, x_points, u
    
    def _solve_l2_scheme(self, 
                        initial_condition: np.ndarray,
                        boundary_conditions: Tuple[Callable, Callable],
                        alpha: Union[float, FractionalOrder],
                        t_final: float,
                        dt: float,
                        dx: float,
                        diffusion_coeff: float,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve using L2 scheme (second-order accurate).
        
        Suitable for 0 < α < 1.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha
        
        if alpha_val >= 1:
            raise ValueError("L2 scheme requires 0 < α < 1")
        
        # Setup grid
        N_t = int(t_final / dt) + 1
        N_x = len(initial_condition)
        
        t_points = np.linspace(0, t_final, N_t)
        x_points = np.linspace(0, (N_x - 1) * dx, N_x)
        
        # Initialize solution matrix
        u = np.zeros((N_t, N_x))
        u[0, :] = initial_condition
        
        # L2 coefficients
        coeffs = self._compute_l2_coefficients(alpha_val, N_t)
        
        # Spatial discretization matrix
        A = self._build_spatial_matrix(N_x, dx, diffusion_coeff)
        
        # Time stepping
        for n in range(2, N_t):
            # Right-hand side
            rhs = np.zeros(N_x)
            
            # History term
            for j in range(1, n + 1):
                rhs += coeffs[j] * u[n-j, :]
            
            # Apply boundary conditions
            left_bc, right_bc = boundary_conditions
            rhs[0] = left_bc(t_points[n])
            rhs[-1] = right_bc(t_points[n])
            
            # Solve linear system
            u[n, :] = linalg.solve(A, rhs)
        
        # Use L1 for first step
        if N_t > 1:
            u[1, :] = self._solve_l1_scheme(initial_condition, boundary_conditions,
                                          alpha, dt, dt, dx, diffusion_coeff)[2][1, :]
        
        return t_points, x_points, u
    
    def _solve_l2_1_sigma_scheme(self, 
                                initial_condition: np.ndarray,
                                boundary_conditions: Tuple[Callable, Callable],
                                alpha: Union[float, FractionalOrder],
                                t_final: float,
                                dt: float,
                                dx: float,
                                diffusion_coeff: float,
                                **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve using L2-1_σ scheme (second-order accurate).
        
        Suitable for 0 < α < 1.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha
        
        if alpha_val >= 1:
            raise ValueError("L2-1_σ scheme requires 0 < α < 1")
        
        # Setup grid
        N_t = int(t_final / dt) + 1
        N_x = len(initial_condition)
        
        t_points = np.linspace(0, t_final, N_t)
        x_points = np.linspace(0, (N_x - 1) * dx, N_x)
        
        # Initialize solution matrix
        u = np.zeros((N_t, N_x))
        u[0, :] = initial_condition
        
        # L2-1_σ coefficients
        sigma = 1 - alpha_val / 2
        coeffs = self._compute_l2_1_sigma_coefficients(alpha_val, sigma, N_t)
        
        # Spatial discretization matrix
        A = self._build_spatial_matrix(N_x, dx, diffusion_coeff)
        
        # Time stepping
        for n in range(1, N_t):
            # Right-hand side
            rhs = np.zeros(N_x)
            
            # History term
            for j in range(1, n + 1):
                rhs += coeffs[j] * u[n-j, :]
            
            # Apply boundary conditions
            left_bc, right_bc = boundary_conditions
            rhs[0] = left_bc(t_points[n])
            rhs[-1] = right_bc(t_points[n])
            
            # Solve linear system
            u[n, :] = linalg.solve(A, rhs)
        
        return t_points, x_points, u
    
    def _solve_l2_1_theta_scheme(self, 
                                initial_condition: np.ndarray,
                                boundary_conditions: Tuple[Callable, Callable],
                                alpha: Union[float, FractionalOrder],
                                t_final: float,
                                dt: float,
                                dx: float,
                                diffusion_coeff: float,
                                **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve using L2-1_θ scheme (second-order accurate).
        
        Suitable for 0 < α < 1.
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha
        
        if alpha_val >= 1:
            raise ValueError("L2-1_θ scheme requires 0 < α < 1")
        
        # Setup grid
        N_t = int(t_final / dt) + 1
        N_x = len(initial_condition)
        
        t_points = np.linspace(0, t_final, N_t)
        x_points = np.linspace(0, (N_x - 1) * dx, N_x)
        
        # Initialize solution matrix
        u = np.zeros((N_t, N_x))
        u[0, :] = initial_condition
        
        # L2-1_θ coefficients
        theta = kwargs.get('theta', 0.5)
        coeffs = self._compute_l2_1_theta_coefficients(alpha_val, theta, N_t)
        
        # Spatial discretization matrix
        A = self._build_spatial_matrix(N_x, dx, diffusion_coeff)
        
        # Time stepping
        for n in range(1, N_t):
            # Right-hand side
            rhs = np.zeros(N_x)
            
            # History term
            for j in range(1, n + 1):
                rhs += coeffs[j] * u[n-j, :]
            
            # Apply boundary conditions
            left_bc, right_bc = boundary_conditions
            rhs[0] = left_bc(t_points[n])
            rhs[-1] = right_bc(t_points[n])
            
            # Solve linear system
            u[n, :] = linalg.solve(A, rhs)
        
        return t_points, x_points, u
    
    def _compute_l1_coefficients(self, alpha: float, N: int) -> np.ndarray:
        """Compute L1 scheme coefficients."""
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - j ** alpha
        return coeffs
    
    def _compute_l2_coefficients(self, alpha: float, N: int) -> np.ndarray:
        """Compute L2 scheme coefficients."""
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + 1) ** alpha - 2 * j ** alpha + (j - 1) ** alpha
        return coeffs
    
    def _compute_l2_1_sigma_coefficients(self, alpha: float, sigma: float, N: int) -> np.ndarray:
        """Compute L2-1_σ scheme coefficients."""
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + sigma) ** alpha - 2 * (j + sigma - 1) ** alpha + (j + sigma - 2) ** alpha
        return coeffs
    
    def _compute_l2_1_theta_coefficients(self, alpha: float, theta: float, N: int) -> np.ndarray:
        """Compute L2-1_θ scheme coefficients."""
        coeffs = np.zeros(N)
        coeffs[0] = 1.0
        for j in range(1, N):
            coeffs[j] = (j + theta) ** alpha - 2 * (j + theta - 1) ** alpha + (j + theta - 2) ** alpha
        return coeffs
    
    def _build_spatial_matrix(self, N_x: int, dx: float, diffusion_coeff: float) -> np.ndarray:
        """Build spatial discretization matrix for ∂²u/∂x²."""
        A = np.zeros((N_x, N_x))
        
        # Central difference for interior points
        for i in range(1, N_x - 1):
            A[i, i-1] = diffusion_coeff / (dx**2)
            A[i, i] = -2 * diffusion_coeff / (dx**2)
            A[i, i+1] = diffusion_coeff / (dx**2)
        
        # Boundary conditions (Dirichlet)
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        
        return A
    
    def stability_analysis(self, alpha: Union[float, FractionalOrder],
                          dt: float, dx: float, diffusion_coeff: float) -> dict:
        """
        Perform stability analysis for the scheme.
        
        Args:
            alpha: Fractional order
            dt: Time step size
            dx: Spatial step size
            diffusion_coeff: Diffusion coefficient
            
        Returns:
            Dictionary with stability information
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha
        
        # Stability parameter
        r = diffusion_coeff * dt**alpha_val / dx**2
        
        # Stability conditions
        if self.scheme == "l1":
            # L1 scheme is unconditionally stable
            is_stable = True
            stability_condition = "Unconditionally stable"
        elif self.scheme == "l2":
            # L2 scheme stability condition
            is_stable = r <= 1.0
            stability_condition = f"r ≤ 1.0 (r = {r:.4f})"
        else:
            # L2-1 schemes are generally more stable
            is_stable = r <= 2.0
            stability_condition = f"r ≤ 2.0 (r = {r:.4f})"
        
        return {
            "scheme": self.scheme,
            "alpha": alpha_val,
            "stability_parameter": r,
            "is_stable": is_stable,
            "stability_condition": stability_condition,
            "dt": dt,
            "dx": dx,
            "diffusion_coeff": diffusion_coeff
        }


# JAX-optimized implementations
class JAXL1L2Schemes:
    """JAX-optimized L1/L2 schemes for time-fractional PDEs."""
    
    @staticmethod
    @jax.jit
    def l1_step_jax(u_prev: jnp.ndarray, coeffs: jnp.ndarray, 
                    A: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        """
        JAX-compiled L1 scheme time step.
        
        Args:
            u_prev: Previous time step solution
            coeffs: L1 coefficients
            A: Spatial discretization matrix
            rhs: Right-hand side
            
        Returns:
            Next time step solution
        """
        # Solve linear system
        return jnp.linalg.solve(A, rhs)
    
    @staticmethod
    @jax.jit
    def compute_l1_coefficients_jax(alpha: float, N: int) -> jnp.ndarray:
        """JAX-compiled L1 coefficient computation."""
        j = jnp.arange(1, N)
        coeffs = jnp.concatenate([jnp.array([1.0]), (j + 1) ** alpha - j ** alpha])
        return coeffs


# NUMBA-optimized implementations
@jit(nopython=True, parallel=True)
def l1_scheme_numba(initial_condition: np.ndarray, alpha: float, dt: float, 
                   dx: float, diffusion_coeff: float, N_t: int) -> np.ndarray:
    """
    NUMBA-optimized L1 scheme implementation.
    
    Args:
        initial_condition: Initial condition
        alpha: Fractional order
        dt: Time step size
        dx: Spatial step size
        diffusion_coeff: Diffusion coefficient
        N_t: Number of time steps
        
    Returns:
        Solution matrix
    """
    N_x = len(initial_condition)
    u = np.zeros((N_t, N_x))
    u[0, :] = initial_condition
    
    # L1 coefficients
    coeffs = np.zeros(N_t)
    coeffs[0] = 1.0
    for j in range(1, N_t):
        coeffs[j] = (j + 1) ** alpha - j ** alpha
    
    # Spatial matrix
    A = np.zeros((N_x, N_x))
    for i in range(1, N_x - 1):
        A[i, i-1] = diffusion_coeff / (dx**2)
        A[i, i] = -2 * diffusion_coeff / (dx**2)
        A[i, i+1] = diffusion_coeff / (dx**2)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    
    # Time stepping
    for n in prange(1, N_t):
        rhs = np.zeros(N_x)
        
        # History term
        for j in range(1, n + 1):
            rhs += coeffs[j] * (u[n-j+1, :] - u[n-j, :])
        
        # Solve linear system (simplified)
        u[n, :] = np.linalg.solve(A, rhs)
    
    return u


@jit(nopython=True, parallel=True)
def l2_scheme_numba(initial_condition: np.ndarray, alpha: float, dt: float,
                   dx: float, diffusion_coeff: float, N_t: int) -> np.ndarray:
    """
    NUMBA-optimized L2 scheme implementation.
    
    Args:
        initial_condition: Initial condition
        alpha: Fractional order
        dt: Time step size
        dx: Spatial step size
        diffusion_coeff: Diffusion coefficient
        N_t: Number of time steps
        
    Returns:
        Solution matrix
    """
    N_x = len(initial_condition)
    u = np.zeros((N_t, N_x))
    u[0, :] = initial_condition
    
    # L2 coefficients
    coeffs = np.zeros(N_t)
    coeffs[0] = 1.0
    for j in range(1, N_t):
        coeffs[j] = (j + 1) ** alpha - 2 * j ** alpha + (j - 1) ** alpha
    
    # Spatial matrix
    A = np.zeros((N_x, N_x))
    for i in range(1, N_x - 1):
        A[i, i-1] = diffusion_coeff / (dx**2)
        A[i, i] = -2 * diffusion_coeff / (dx**2)
        A[i, i+1] = diffusion_coeff / (dx**2)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    
    # Time stepping
    for n in prange(2, N_t):
        rhs = np.zeros(N_x)
        
        # History term
        for j in range(1, n + 1):
            rhs += coeffs[j] * u[n-j, :]
        
        # Solve linear system
        u[n, :] = np.linalg.solve(A, rhs)
    
    # Use L1 for first step
    if N_t > 1:
        u[1, :] = l1_scheme_numba(initial_condition, alpha, dt, dx, diffusion_coeff, 2)[1, :]
    
    return u


# Convenience functions
def solve_time_fractional_pde(initial_condition: np.ndarray,
                             boundary_conditions: Tuple[Callable, Callable],
                             alpha: Union[float, FractionalOrder],
                             t_final: float,
                             dt: float,
                             dx: float,
                             diffusion_coeff: float = 1.0,
                             scheme: str = "l1",
                             **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for solving time-fractional PDE.
    
    Args:
        initial_condition: Initial condition u(x, 0)
        boundary_conditions: Tuple of (left_bc, right_bc) functions
        alpha: Fractional order (0 < α < 1)
        t_final: Final time
        dt: Time step size
        dx: Spatial step size
        diffusion_coeff: Diffusion coefficient D
        scheme: Numerical scheme
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (time_points, spatial_points, solution_matrix)
    """
    solver = L1L2Schemes(scheme)
    return solver.solve_time_fractional_pde(initial_condition, boundary_conditions,
                                          alpha, t_final, dt, dx, diffusion_coeff, **kwargs)


def solve_time_fractional_pde_numba(initial_condition: np.ndarray,
                                   alpha: Union[float, FractionalOrder],
                                   t_final: float,
                                   dt: float,
                                   dx: float,
                                   diffusion_coeff: float = 1.0,
                                   scheme: str = "l1") -> np.ndarray:
    """
    NUMBA-optimized time-fractional PDE solver.
    
    Args:
        initial_condition: Initial condition u(x, 0)
        alpha: Fractional order (0 < α < 1)
        t_final: Final time
        dt: Time step size
        dx: Spatial step size
        diffusion_coeff: Diffusion coefficient D
        scheme: Numerical scheme ("l1" or "l2")
        
    Returns:
        Solution matrix
    """
    if isinstance(alpha, FractionalOrder):
        alpha_val = alpha.alpha
    else:
        alpha_val = alpha
    
    N_t = int(t_final / dt) + 1
    
    if scheme == "l1":
        return l1_scheme_numba(initial_condition, alpha_val, dt, dx, diffusion_coeff, N_t)
    elif scheme == "l2":
        return l2_scheme_numba(initial_condition, alpha_val, dt, dx, diffusion_coeff, N_t)
    else:
        raise ValueError("Scheme must be 'l1' or 'l2' for NUMBA implementation")
