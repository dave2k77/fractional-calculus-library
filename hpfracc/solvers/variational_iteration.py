"""
Variational Iteration Method (VIM) for Fractional Differential Equations

This module implements the Variational Iteration Method for solving
fractional differential equations. VIM is an analytical method that
uses Lagrange multipliers and correction functionals.
"""

import numpy as np
import torch
from typing import Union, Callable, Optional, Tuple, List, Dict, Any
from scipy.special import gamma
import warnings

from ..core.definitions import FractionalOrder
from ..core.derivatives import create_fractional_derivative
from ..core.integrals import create_fractional_integral
from ..core.utilities import validate_fractional_order, validate_function


class VariationalIterationMethod:
    """
    Variational Iteration Method solver for fractional differential equations.
    
    VIM constructs a correction functional using Lagrange multipliers
    and iteratively improves the solution approximation.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], max_iterations: int = 20, 
                 tolerance: float = 1e-6, lagrange_multiplier: Optional[float] = None):
        """
        Initialize VIM solver.
        
        Args:
            alpha: Fractional order
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            lagrange_multiplier: Lagrange multiplier (if None, computed automatically)
        """
        self.alpha = validate_fractional_order(alpha)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.lagrange_multiplier = lagrange_multiplier
        
        # Initialize fractional derivative and integral operators
        # For now, we'll use placeholder functions that can be implemented later
        self.D_alpha = None  # Will be implemented when needed
        self.I_alpha = None  # Will be implemented when needed
    
    def solve(self, equation: Callable, initial_condition: Callable, 
              boundary_conditions: Optional[List[Callable]] = None,
              domain: Tuple[float, float] = (0.0, 1.0), 
              n_points: int = 100) -> Dict[str, Any]:
        """
        Solve fractional differential equation using VIM.
        
        Args:
            equation: The differential equation L(u) + N(u) = f(x)
            initial_condition: Initial condition u(0) = g(x)
            boundary_conditions: List of boundary conditions
            domain: Solution domain (x_min, x_max)
            n_points: Number of points for solution evaluation
            
        Returns:
            Dictionary containing solution and metadata
        """
        # Validate inputs
        if not callable(equation):
            raise ValueError("Equation must be callable")
        if not callable(initial_condition):
            raise ValueError("Initial condition must be callable")
        
        # Create solution grid
        x = np.linspace(domain[0], domain[1], n_points)
        
        # Initial approximation
        u0 = initial_condition(x)
        solution = u0.copy()
        
        # Compute Lagrange multiplier if not provided
        if self.lagrange_multiplier is None:
            self.lagrange_multiplier = self._compute_lagrange_multiplier()
        
        # VIM iteration
        for n in range(self.max_iterations):
            # Compute correction term
            correction = self._compute_correction(equation, x, solution)
            
            # Update solution
            solution_new = solution + self.lagrange_multiplier * correction
            
            # Check convergence
            error = np.max(np.abs(solution_new - solution))
            solution = solution_new
            
            if error < self.tolerance:
                break
        
        return {
            "solution": solution,
            "x": x,
            "iterations": n + 1,
            "converged": n + 1 < self.max_iterations,
            "final_error": error,
            "lagrange_multiplier": self.lagrange_multiplier
        }
    
    def _compute_lagrange_multiplier(self) -> float:
        """
        Compute optimal Lagrange multiplier.
        
        Returns:
            Lagrange multiplier value
        """
        # For fractional differential equations, the Lagrange multiplier
        # is typically related to the fractional order
        if self.alpha.alpha == 1:
            return -1.0  # Standard case
        else:
            # Fractional case - simplified approximation
            return -1.0 / gamma(self.alpha.alpha)
    
    def _compute_correction(self, equation: Callable, x: np.ndarray, 
                          current_solution: np.ndarray) -> np.ndarray:
        """
        Compute correction term for VIM iteration.
        
        Args:
            equation: The differential equation
            x: Spatial coordinates
            current_solution: Current solution approximation
            
        Returns:
            Correction term
        """
        # Compute residual
        residual = equation(x, current_solution)
        
        # Apply fractional integral operator
        correction = self.I_alpha(lambda t: residual, x)
        
        return correction
    
    def solve_linear_fde(self, L: Callable, N: Callable, f: Callable, 
                        initial_condition: Callable, domain: Tuple[float, float] = (0.0, 1.0),
                        n_points: int = 100) -> Dict[str, Any]:
        """
        Solve linear fractional differential equation using VIM.
        
        The equation is: L(u) + N(u) = f(x)
        where L is the linear operator and N is the nonlinear operator.
        
        Args:
            L: Linear operator
            N: Nonlinear operator
            f: Source term
            initial_condition: Initial condition
            domain: Solution domain
            n_points: Number of evaluation points
            
        Returns:
            Solution dictionary
        """
        x = np.linspace(domain[0], domain[1], n_points)
        
        # Initial approximation
        u0 = initial_condition(x)
        solution = u0.copy()
        
        # Compute Lagrange multiplier
        if self.lagrange_multiplier is None:
            self.lagrange_multiplier = self._compute_lagrange_multiplier()
        
        # VIM iteration
        for n in range(self.max_iterations):
            # Compute linear and nonlinear terms
            linear_term = L(x, solution)
            nonlinear_term = N(x, solution)
            
            # Compute residual
            residual = f(x) - linear_term - nonlinear_term
            
            # Apply correction
            correction = self.I_alpha(lambda t: residual, x)
            solution_new = solution + self.lagrange_multiplier * correction
            
            # Check convergence
            error = np.max(np.abs(solution_new - solution))
            solution = solution_new
            
            if error < self.tolerance:
                break
        
        return {
            "solution": solution,
            "x": x,
            "iterations": n + 1,
            "converged": n + 1 < self.max_iterations,
            "final_error": error,
            "lagrange_multiplier": self.lagrange_multiplier
        }


class VIMFractionalDiffusion:
    """
    Specialized VIM solver for fractional diffusion equations.
    
    Solves: ∂^α u/∂t^α = D ∂²u/∂x² + f(x,t)
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], D: float = 1.0):
        """
        Initialize fractional diffusion VIM solver.
        
        Args:
            alpha: Fractional order
            D: Diffusion coefficient
        """
        self.alpha = validate_fractional_order(alpha)
        self.D = D
        
        # Initialize operators
        # For now, we'll use placeholder functions that can be implemented later
        self.D_alpha = None  # Will be implemented when needed
        self.I_alpha = None  # Will be implemented when needed
        
        # Lagrange multiplier for diffusion equation
        self.lagrange_multiplier = -1.0 / gamma(self.alpha.alpha)
    
    def solve(self, f: Callable, initial_condition: Callable, 
              boundary_conditions: List[Callable], domain: Tuple[float, float] = (0.0, 1.0),
              time_domain: Tuple[float, float] = (0.0, 1.0), 
              n_x: int = 50, n_t: int = 50) -> Dict[str, Any]:
        """
        Solve fractional diffusion equation using VIM.
        
        Args:
            f: Source term f(x,t)
            initial_condition: Initial condition u(x,0) = g(x)
            boundary_conditions: [u(0,t) = h1(t), u(L,t) = h2(t)]
            domain: Spatial domain (x_min, x_max)
            time_domain: Time domain (t_min, t_max)
            n_x: Number of spatial points
            n_t: Number of time points
            
        Returns:
            Solution dictionary
        """
        x = np.linspace(domain[0], domain[1], n_x)
        t = np.linspace(time_domain[0], time_domain[1], n_t)
        X, T = np.meshgrid(x, t)
        
        # Initialize solution
        solution = np.zeros((n_t, n_x))
        
        # Set initial condition
        solution[0, :] = initial_condition(x)
        
        # Set boundary conditions
        solution[:, 0] = boundary_conditions[0](t)
        solution[:, -1] = boundary_conditions[1](t)
        
        # VIM iteration for each time step
        for i in range(1, n_t):
            ti = t[i]
            u_current = self._solve_time_step(x, ti, solution[i-1, :], f, boundary_conditions)
            solution[i, :] = u_current
        
        return {
            "solution": solution,
            "x": x,
            "t": t,
            "X": X,
            "T": T,
            "alpha": self.alpha.alpha,
            "D": self.D
        }
    
    def _solve_time_step(self, x: np.ndarray, t: float, u_prev: np.ndarray, 
                        f: Callable, boundary_conditions: List[Callable]) -> np.ndarray:
        """
        Solve for a single time step using VIM.
        
        Args:
            x: Spatial coordinates
            t: Current time
            u_prev: Solution at previous time step
            f: Source term
            boundary_conditions: Boundary conditions
            
        Returns:
            Solution at current time step
        """
        # Initial approximation
        u0 = u_prev.copy()
        solution = u0.copy()
        
        # VIM iteration
        for n in range(10):  # Use 10 iterations for time step
            # Compute spatial derivatives
            dx = x[1] - x[0]
            d2u_dx2 = np.zeros_like(solution)
            
            for i in range(1, len(x) - 1):
                d2u_dx2[i] = (solution[i+1] - 2*solution[i] + solution[i-1]) / (dx**2)
            
            d2u_dx2[0] = d2u_dx2[1]
            d2u_dx2[-1] = d2u_dx2[-2]
            
            # Compute residual
            diffusion_term = self.D * d2u_dx2
            source_term = f(x, t)
            residual = source_term - diffusion_term
            
            # Apply correction
            correction = self.I_alpha(lambda tau: residual, t)
            solution_new = solution + self.lagrange_multiplier * correction
            
            # Check convergence
            error = np.max(np.abs(solution_new - solution))
            solution = solution_new
            
            if error < 1e-6:
                break
        
        # Apply boundary conditions
        solution[0] = boundary_conditions[0](t)
        solution[-1] = boundary_conditions[1](t)
        
        return solution


class VIMFractionalWave:
    """
    Specialized VIM solver for fractional wave equations.
    
    Solves: ∂^α u/∂t^α = c² ∂²u/∂x² + f(x,t)
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], c: float = 1.0):
        """
        Initialize fractional wave VIM solver.
        
        Args:
            alpha: Fractional order (1 < α < 2)
            c: Wave speed
        """
        if not (1 < alpha < 2):
            raise ValueError("Fractional wave equation requires 1 < α < 2")
        
        self.alpha = validate_fractional_order(alpha)
        self.c = c
        
        # Initialize operators
        # For now, we'll use placeholder functions that can be implemented later
        self.D_alpha = None  # Will be implemented when needed
        self.I_alpha = None  # Will be implemented when needed
        
        # Lagrange multiplier for wave equation
        self.lagrange_multiplier = -1.0 / gamma(self.alpha.alpha)
    
    def solve(self, f: Callable, initial_condition: Callable, 
              initial_velocity: Callable, boundary_conditions: List[Callable],
              domain: Tuple[float, float] = (0.0, 1.0),
              time_domain: Tuple[float, float] = (0.0, 1.0),
              n_x: int = 50, n_t: int = 50) -> Dict[str, Any]:
        """
        Solve fractional wave equation using VIM.
        
        Args:
            f: Source term f(x,t)
            initial_condition: Initial condition u(x,0) = g(x)
            initial_velocity: Initial velocity ∂u/∂t(x,0) = h(x)
            boundary_conditions: [u(0,t) = h1(t), u(L,t) = h2(t)]
            domain: Spatial domain
            time_domain: Time domain
            n_x: Number of spatial points
            n_t: Number of time points
            
        Returns:
            Solution dictionary
        """
        x = np.linspace(domain[0], domain[1], n_x)
        t = np.linspace(time_domain[0], time_domain[1], n_t)
        X, T = np.meshgrid(x, t)
        
        # Initialize solution
        solution = np.zeros((n_t, n_x))
        
        # Set initial conditions
        solution[0, :] = initial_condition(x)
        
        # Set boundary conditions
        solution[:, 0] = boundary_conditions[0](t)
        solution[:, -1] = boundary_conditions[1](t)
        
        # VIM iteration for each time step
        for i in range(1, n_t):
            ti = t[i]
            u_current = self._solve_wave_time_step(x, ti, solution[i-1, :], initial_velocity, f, boundary_conditions)
            solution[i, :] = u_current
        
        return {
            "solution": solution,
            "x": x,
            "t": t,
            "X": X,
            "T": T,
            "alpha": self.alpha.alpha,
            "c": self.c
        }
    
    def _solve_wave_time_step(self, x: np.ndarray, t: float, u_prev: np.ndarray,
                             initial_velocity: Callable, f: Callable, 
                             boundary_conditions: List[Callable]) -> np.ndarray:
        """
        Solve for a single time step in wave equation.
        
        Args:
            x: Spatial coordinates
            t: Current time
            u_prev: Solution at previous time step
            initial_velocity: Initial velocity
            f: Source term
            boundary_conditions: Boundary conditions
            
        Returns:
            Solution at current time step
        """
        # Initial approximation
        u0 = u_prev.copy()
        velocity_contribution = initial_velocity(x) * t
        solution = u0 + velocity_contribution
        
        # VIM iteration
        for n in range(10):
            # Compute spatial derivatives
            dx = x[1] - x[0]
            d2u_dx2 = np.zeros_like(solution)
            
            for i in range(1, len(x) - 1):
                d2u_dx2[i] = (solution[i+1] - 2*solution[i] + solution[i-1]) / (dx**2)
            
            d2u_dx2[0] = d2u_dx2[1]
            d2u_dx2[-1] = d2u_dx2[-2]
            
            # Compute residual
            wave_term = (self.c ** 2) * d2u_dx2
            source_term = f(x, t)
            residual = source_term - wave_term
            
            # Apply correction
            correction = self.I_alpha(lambda tau: residual, t)
            solution_new = solution + self.lagrange_multiplier * correction
            
            # Check convergence
            error = np.max(np.abs(solution_new - solution))
            solution = solution_new
            
            if error < 1e-6:
                break
        
        # Apply boundary conditions
        solution[0] = boundary_conditions[0](t)
        solution[-1] = boundary_conditions[1](t)
        
        return solution


class VIMFractionalAdvection:
    """
    Specialized VIM solver for fractional advection equations.
    
    Solves: ∂^α u/∂t^α + v ∂u/∂x = f(x,t)
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], v: float = 1.0):
        """
        Initialize fractional advection VIM solver.
        
        Args:
            alpha: Fractional order (0 < α < 1)
            v: Advection velocity
        """
        if not (0 < alpha < 1):
            raise ValueError("Fractional advection equation requires 0 < α < 1")
        
        self.alpha = validate_fractional_order(alpha)
        self.v = v
        
        # Initialize operators
        # For now, we'll use placeholder functions that can be implemented later
        self.D_alpha = None  # Will be implemented when needed
        self.I_alpha = None  # Will be implemented when needed
        
        # Lagrange multiplier for advection equation
        self.lagrange_multiplier = -1.0 / gamma(self.alpha.alpha)
    
    def solve(self, f: Callable, initial_condition: Callable, 
              boundary_conditions: List[Callable], domain: Tuple[float, float] = (0.0, 1.0),
              time_domain: Tuple[float, float] = (0.0, 1.0),
              n_x: int = 50, n_t: int = 50) -> Dict[str, Any]:
        """
        Solve fractional advection equation using VIM.
        
        Args:
            f: Source term f(x,t)
            initial_condition: Initial condition u(x,0) = g(x)
            boundary_conditions: [u(0,t) = h1(t), u(L,t) = h2(t)]
            domain: Spatial domain
            time_domain: Time domain
            n_x: Number of spatial points
            n_t: Number of time points
            
        Returns:
            Solution dictionary
        """
        x = np.linspace(domain[0], domain[1], n_x)
        t = np.linspace(time_domain[0], time_domain[1], n_t)
        X, T = np.meshgrid(x, t)
        
        # Initialize solution
        solution = np.zeros((n_t, n_x))
        
        # Set initial condition
        solution[0, :] = initial_condition(x)
        
        # Set boundary conditions
        solution[:, 0] = boundary_conditions[0](t)
        solution[:, -1] = boundary_conditions[1](t)
        
        # VIM iteration for each time step
        for i in range(1, n_t):
            ti = t[i]
            u_current = self._solve_advection_time_step(x, ti, solution[i-1, :], f, boundary_conditions)
            solution[i, :] = u_current
        
        return {
            "solution": solution,
            "x": x,
            "t": t,
            "X": X,
            "T": T,
            "alpha": self.alpha.alpha,
            "v": self.v
        }
    
    def _solve_advection_time_step(self, x: np.ndarray, t: float, u_prev: np.ndarray,
                                  f: Callable, boundary_conditions: List[Callable]) -> np.ndarray:
        """
        Solve for a single time step in advection equation.
        
        Args:
            x: Spatial coordinates
            t: Current time
            u_prev: Solution at previous time step
            f: Source term
            boundary_conditions: Boundary conditions
            
        Returns:
            Solution at current time step
        """
        # Initial approximation
        u0 = u_prev.copy()
        solution = u0.copy()
        
        # VIM iteration
        for n in range(10):
            # Compute spatial derivatives
            dx = x[1] - x[0]
            du_dx = np.zeros_like(solution)
            
            for i in range(1, len(x) - 1):
                du_dx[i] = (solution[i+1] - solution[i-1]) / (2 * dx)
            
            du_dx[0] = du_dx[1]
            du_dx[-1] = du_dx[-2]
            
            # Compute residual
            advection_term = self.v * du_dx
            source_term = f(x, t)
            residual = source_term - advection_term
            
            # Apply correction
            correction = self.I_alpha(lambda tau: residual, t)
            solution_new = solution + self.lagrange_multiplier * correction
            
            # Check convergence
            error = np.max(np.abs(solution_new - solution))
            solution = solution_new
            
            if error < 1e-6:
                break
        
        # Apply boundary conditions
        solution[0] = boundary_conditions[0](t)
        solution[-1] = boundary_conditions[1](t)
        
        return solution


# Utility functions
def validate_vim_solution(solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate VIM solution.
    
    Args:
        solution: Solution dictionary from VIM solver
        
    Returns:
        Validation results
    """
    validation = {
        "converged": solution.get("converged", False),
        "iterations": solution.get("iterations", 0),
        "final_error": solution.get("final_error", np.inf),
        "solution_norm": np.linalg.norm(solution.get("solution", [])),
        "is_finite": np.all(np.isfinite(solution.get("solution", []))),
        "lagrange_multiplier": solution.get("lagrange_multiplier", None)
    }
    
    return validation


def vim_convergence_analysis(equation: Callable, initial_condition: Callable,
                           domain: Tuple[float, float] = (0.0, 1.0),
                           max_iterations_list: List[int] = [5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Analyze convergence of VIM method.
    
    Args:
        equation: Differential equation
        initial_condition: Initial condition
        domain: Solution domain
        max_iterations_list: List of maximum iterations to test
        
    Returns:
        Convergence analysis results
    """
    results = {}
    
    for max_iterations in max_iterations_list:
        solver = VariationalIterationMethod(alpha=0.5, max_iterations=max_iterations)
        solution = solver.solve(equation, initial_condition, domain=domain)
        
        results[f"iterations_{max_iterations}"] = {
            "converged": solution["converged"],
            "final_error": solution["final_error"],
            "solution_norm": np.linalg.norm(solution["solution"]),
            "lagrange_multiplier": solution["lagrange_multiplier"]
        }
    
    return results


def compare_hpm_vim(equation: Callable, initial_condition: Callable,
                   domain: Tuple[float, float] = (0.0, 1.0)) -> Dict[str, Any]:
    """
    Compare HPM and VIM methods.
    
    Args:
        equation: Differential equation
        initial_condition: Initial condition
        domain: Solution domain
        
    Returns:
        Comparison results
    """
    from .homotopy_perturbation import HomotopyPerturbationMethod
    
    # Solve using HPM
    hpm_solver = HomotopyPerturbationMethod(alpha=0.5)
    hpm_solution = hpm_solver.solve(equation, initial_condition, domain=domain)
    
    # Solve using VIM
    vim_solver = VariationalIterationMethod(alpha=0.5)
    vim_solution = vim_solver.solve(equation, initial_condition, domain=domain)
    
    # Compare solutions
    if len(hpm_solution["solution"]) == len(vim_solution["solution"]):
        solution_difference = np.max(np.abs(hpm_solution["solution"] - vim_solution["solution"]))
    else:
        solution_difference = np.inf
    
    return {
        "hpm": {
            "converged": hpm_solution["converged"],
            "terms_computed": hpm_solution["terms_computed"],
            "final_error": hpm_solution["final_error"]
        },
        "vim": {
            "converged": vim_solution["converged"],
            "iterations": vim_solution["iterations"],
            "final_error": vim_solution["final_error"]
        },
        "comparison": {
            "solution_difference": solution_difference,
            "hpm_faster": hpm_solution["terms_computed"] < vim_solution["iterations"],
            "vim_more_accurate": vim_solution["final_error"] < hpm_solution["final_error"]
        }
    }


def create_vim_solver(solver_type: str = "general", **kwargs) -> Union[VariationalIterationMethod, VIMFractionalDiffusion, VIMFractionalWave, VIMFractionalAdvection]:
    """
    Factory function to create VIM solver instances.
    
    Args:
        solver_type: Type of solver ("general", "diffusion", "wave", "advection")
        **kwargs: Additional arguments for solver initialization
        
    Returns:
        VIM solver instance
        
    Raises:
        ValueError: If solver_type is not recognized
    """
    if solver_type == "general":
        return VariationalIterationMethod(**kwargs)
    elif solver_type == "diffusion":
        return VIMFractionalDiffusion(**kwargs)
    elif solver_type == "wave":
        return VIMFractionalWave(**kwargs)
    elif solver_type == "advection":
        return VIMFractionalAdvection(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Must be one of: general, diffusion, wave, advection")


def get_vim_properties(solver: Union[VariationalIterationMethod, VIMFractionalDiffusion, VIMFractionalWave, VIMFractionalAdvection]) -> Dict[str, Any]:
    """
    Get properties of a VIM solver instance.
    
    Args:
        solver: VIM solver instance
        
    Returns:
        Dictionary containing solver properties
    """
    properties = {
        "alpha": solver.alpha,
        "max_iterations": getattr(solver, 'max_iterations', None),
        "tolerance": getattr(solver, 'tolerance', None),
        "lagrange_multiplier": getattr(solver, 'lagrange_multiplier', None),
        "solver_type": solver.__class__.__name__
    }
    
    # Add specialized properties for specialized solvers
    if hasattr(solver, 'diffusion_coefficient'):
        properties["diffusion_coefficient"] = solver.diffusion_coefficient
    if hasattr(solver, 'wave_speed'):
        properties["wave_speed"] = solver.wave_speed
    if hasattr(solver, 'advection_velocity'):
        properties["advection_velocity"] = solver.advection_velocity
    
    return properties


def validate_vim_parameters(alpha: Union[float, FractionalOrder], max_iterations: int = 20, 
                           tolerance: float = 1e-6, lagrange_multiplier: Optional[float] = None) -> Dict[str, Any]:
    """
    Validate VIM solver parameters.
    
    Args:
        alpha: Fractional order
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        lagrange_multiplier: Lagrange multiplier (optional)
        
    Returns:
        Validation results dictionary
        
    Raises:
        ValueError: If parameters are invalid
    """
    validation = {"valid": True, "errors": []}
    
    # Validate alpha
    try:
        validated_alpha = validate_fractional_order(alpha)
        validation["alpha"] = validated_alpha
    except ValueError as e:
        validation["valid"] = False
        validation["errors"].append(f"Invalid alpha: {e}")
    
    # Validate max_iterations
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        validation["valid"] = False
        validation["errors"].append("max_iterations must be a positive integer")
    else:
        validation["max_iterations"] = max_iterations
    
    # Validate tolerance
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        validation["valid"] = False
        validation["errors"].append("tolerance must be a positive number")
    else:
        validation["tolerance"] = tolerance
    
    # Validate lagrange_multiplier (optional)
    if lagrange_multiplier is not None:
        if not isinstance(lagrange_multiplier, (int, float)):
            validation["valid"] = False
            validation["errors"].append("lagrange_multiplier must be a number")
        else:
            validation["lagrange_multiplier"] = lagrange_multiplier
    
    if not validation["valid"]:
        raise ValueError(f"Invalid VIM parameters: {'; '.join(validation['errors'])}")
    
    return validation
