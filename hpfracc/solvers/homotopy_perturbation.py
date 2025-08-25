"""
Homotopy Perturbation Method (HPM) for Fractional Differential Equations

This module implements the Homotopy Perturbation Method for solving
fractional differential equations. HPM is an analytical method that
combines homotopy theory with perturbation techniques.
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


class HomotopyPerturbationMethod:
    """
    Homotopy Perturbation Method solver for fractional differential equations.
    
    The HPM constructs a homotopy that continuously deforms a simple problem
    into the original problem, then uses perturbation theory to find the solution.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], max_terms: int = 10, 
                 tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize HPM solver.
        
        Args:
            alpha: Fractional order
            max_terms: Maximum number of terms in the series solution
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
        """
        self.alpha = validate_fractional_order(alpha)
        self.max_terms = max_terms
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        
        # Initialize fractional derivative and integral operators
        self.D_alpha = create_fractional_derivative(self.alpha, method="RL")
        self.I_alpha = create_fractional_integral(self.alpha, method="RL")
    
    def solve(self, equation: Callable, initial_condition: Callable, 
              boundary_conditions: Optional[List[Callable]] = None,
              domain: Tuple[float, float] = (0.0, 1.0), 
              n_points: int = 100) -> Dict[str, Any]:
        """
        Solve fractional differential equation using HPM.
        
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
        
        # Initialize solution with initial condition
        u0 = initial_condition(x)
        solution = u0.copy()
        
        # HPM iteration
        for n in range(1, self.max_terms + 1):
            # Compute next term in the series
            un = self._compute_term_n(equation, x, solution, n)
            
            # Add to solution
            solution += un
            
            # Check convergence
            if np.max(np.abs(un)) < self.tolerance:
                break
        
        return {
            "solution": solution,
            "x": x,
            "terms_computed": n,
            "converged": n < self.max_terms,
            "final_error": np.max(np.abs(un)) if n < self.max_terms else np.inf
        }
    
    def _compute_term_n(self, equation: Callable, x: np.ndarray, 
                       current_solution: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the n-th term in the HPM series.
        
        Args:
            equation: The differential equation
            x: Spatial coordinates
            current_solution: Current solution approximation
            n: Term index
            
        Returns:
            n-th term in the series
        """
        # This is a simplified implementation
        # In practice, this would involve more complex analytical computations
        
        # For demonstration, we use a simple perturbation approach
        h = 0.1  # Perturbation parameter
        
        # Compute perturbation
        perturbation = h * equation(x, current_solution)
        
        # Apply fractional integral operator
        un = self.I_alpha(lambda t: perturbation, x)
        
        return un
    
    def solve_linear_fde(self, L: Callable, N: Callable, f: Callable, 
                        initial_condition: Callable, domain: Tuple[float, float] = (0.0, 1.0),
                        n_points: int = 100) -> Dict[str, Any]:
        """
        Solve linear fractional differential equation using HPM.
        
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
        
        # Initial approximation (solution of linear part)
        u0 = self._solve_linear_part(L, f, initial_condition, x)
        solution = u0.copy()
        
        # HPM series solution
        for n in range(1, self.max_terms + 1):
            # Compute nonlinear correction
            correction = self._compute_nonlinear_correction(N, x, solution, n)
            
            # Add correction to solution
            solution += correction
            
            # Check convergence
            if np.max(np.abs(correction)) < self.tolerance:
                break
        
        return {
            "solution": solution,
            "x": x,
            "terms_computed": n,
            "converged": n < self.max_terms,
            "final_error": np.max(np.abs(correction)) if n < self.max_terms else np.inf
        }
    
    def _solve_linear_part(self, L: Callable, f: Callable, 
                          initial_condition: Callable, x: np.ndarray) -> np.ndarray:
        """
        Solve the linear part of the equation.
        
        Args:
            L: Linear operator
            f: Source term
            initial_condition: Initial condition
            x: Spatial coordinates
            
        Returns:
            Solution of linear part
        """
        # For simplicity, assume L is invertible
        # In practice, this would involve solving the linear equation
        
        # Use initial condition as first approximation
        u0 = initial_condition(x)
        
        # Apply linear operator
        Lu0 = L(x, u0)
        
        # Compute residual
        residual = f(x) - Lu0
        
        # Apply inverse of linear operator (simplified)
        correction = self.I_alpha(lambda t: residual, x)
        
        return u0 + correction
    
    def _compute_nonlinear_correction(self, N: Callable, x: np.ndarray, 
                                    current_solution: np.ndarray, n: int) -> np.ndarray:
        """
        Compute nonlinear correction term.
        
        Args:
            N: Nonlinear operator
            x: Spatial coordinates
            current_solution: Current solution
            n: Term index
            
        Returns:
            Nonlinear correction
        """
        # Compute nonlinear term
        nonlinear_term = N(x, current_solution)
        
        # Apply fractional integral operator
        correction = self.I_alpha(lambda t: nonlinear_term, x)
        
        # Scale by perturbation parameter
        h = 0.1
        return h * correction


class HPMFractionalDiffusion:
    """
    Specialized HPM solver for fractional diffusion equations.
    
    Solves: ∂^α u/∂t^α = D ∂²u/∂x² + f(x,t)
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], D: float = 1.0):
        """
        Initialize fractional diffusion HPM solver.
        
        Args:
            alpha: Fractional order
            D: Diffusion coefficient
        """
        self.alpha = validate_fractional_order(alpha)
        self.D = D
        
        # Initialize operators
        self.D_alpha = create_fractional_derivative(self.alpha, method="RL")
        self.I_alpha = create_fractional_integral(self.alpha, method="RL")
    
    def solve(self, f: Callable, initial_condition: Callable, 
              boundary_conditions: List[Callable], domain: Tuple[float, float] = (0.0, 1.0),
              time_domain: Tuple[float, float] = (0.0, 1.0), 
              n_x: int = 50, n_t: int = 50) -> Dict[str, Any]:
        """
        Solve fractional diffusion equation using HPM.
        
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
        
        # HPM iteration for each time step
        for i in range(1, n_t):
            # Current time
            ti = t[i]
            
            # Solve for current time step using HPM
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
        Solve for a single time step using HPM.
        
        Args:
            x: Spatial coordinates
            t: Current time
            u_prev: Solution at previous time step
            f: Source term
            boundary_conditions: Boundary conditions
            
        Returns:
            Solution at current time step
        """
        # Initial approximation (previous time step)
        u0 = u_prev.copy()
        
        # HPM series solution
        solution = u0.copy()
        
        for n in range(1, 6):  # Use 5 terms for time step
            # Compute correction term
            correction = self._compute_diffusion_correction(x, t, solution, f, n)
            
            # Add correction
            solution += correction
            
            # Check convergence
            if np.max(np.abs(correction)) < 1e-6:
                break
        
        # Apply boundary conditions
        solution[0] = boundary_conditions[0](t)
        solution[-1] = boundary_conditions[1](t)
        
        return solution
    
    def _compute_diffusion_correction(self, x: np.ndarray, t: float, 
                                    current_solution: np.ndarray, f: Callable, n: int) -> np.ndarray:
        """
        Compute diffusion correction term.
        
        Args:
            x: Spatial coordinates
            t: Current time
            current_solution: Current solution
            f: Source term
            n: Term index
            
        Returns:
            Correction term
        """
        # Compute spatial derivatives (simplified)
        dx = x[1] - x[0]
        d2u_dx2 = np.zeros_like(current_solution)
        
        # Second derivative using finite differences
        for i in range(1, len(x) - 1):
            d2u_dx2[i] = (current_solution[i+1] - 2*current_solution[i] + current_solution[i-1]) / (dx**2)
        
        # Apply boundary conditions for derivatives
        d2u_dx2[0] = d2u_dx2[1]
        d2u_dx2[-1] = d2u_dx2[-2]
        
        # Diffusion term
        diffusion_term = self.D * d2u_dx2
        
        # Source term
        source_term = f(x, t)
        
        # Total right-hand side
        rhs = diffusion_term + source_term
        
        # Apply fractional integral operator
        correction = self.I_alpha(lambda tau: rhs, t)
        
        # Scale by perturbation parameter
        h = 0.1
        return h * correction


class HPMFractionalWave:
    """
    Specialized HPM solver for fractional wave equations.
    
    Solves: ∂^α u/∂t^α = c² ∂²u/∂x² + f(x,t)
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], c: float = 1.0):
        """
        Initialize fractional wave HPM solver.
        
        Args:
            alpha: Fractional order (1 < α < 2)
            c: Wave speed
        """
        if not (1 < alpha < 2):
            raise ValueError("Fractional wave equation requires 1 < α < 2")
        
        self.alpha = validate_fractional_order(alpha)
        self.c = c
        
        # Initialize operators
        self.D_alpha = create_fractional_derivative(self.alpha, method="RL")
        self.I_alpha = create_fractional_integral(self.alpha, method="RL")
    
    def solve(self, f: Callable, initial_condition: Callable, 
              initial_velocity: Callable, boundary_conditions: List[Callable],
              domain: Tuple[float, float] = (0.0, 1.0),
              time_domain: Tuple[float, float] = (0.0, 1.0),
              n_x: int = 50, n_t: int = 50) -> Dict[str, Any]:
        """
        Solve fractional wave equation using HPM.
        
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
        
        # HPM iteration for each time step
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
        
        # Add velocity contribution
        velocity_contribution = initial_velocity(x) * t
        
        solution = u0 + velocity_contribution
        
        # HPM series solution
        for n in range(1, 6):
            correction = self._compute_wave_correction(x, t, solution, f, n)
            solution += correction
            
            if np.max(np.abs(correction)) < 1e-6:
                break
        
        # Apply boundary conditions
        solution[0] = boundary_conditions[0](t)
        solution[-1] = boundary_conditions[1](t)
        
        return solution
    
    def _compute_wave_correction(self, x: np.ndarray, t: float, 
                               current_solution: np.ndarray, f: Callable, n: int) -> np.ndarray:
        """
        Compute wave equation correction term.
        
        Args:
            x: Spatial coordinates
            t: Current time
            current_solution: Current solution
            f: Source term
            n: Term index
            
        Returns:
            Correction term
        """
        # Compute spatial derivatives
        dx = x[1] - x[0]
        d2u_dx2 = np.zeros_like(current_solution)
        
        for i in range(1, len(x) - 1):
            d2u_dx2[i] = (current_solution[i+1] - 2*current_solution[i] + current_solution[i-1]) / (dx**2)
        
        d2u_dx2[0] = d2u_dx2[1]
        d2u_dx2[-1] = d2u_dx2[-2]
        
        # Wave term
        wave_term = (self.c ** 2) * d2u_dx2
        
        # Source term
        source_term = f(x, t)
        
        # Total right-hand side
        rhs = wave_term + source_term
        
        # Apply fractional integral operator
        correction = self.I_alpha(lambda tau: rhs, t)
        
        # Scale by perturbation parameter
        h = 0.1
        return h * correction


# Utility functions
def validate_hpm_solution(solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate HPM solution.
    
    Args:
        solution: Solution dictionary from HPM solver
        
    Returns:
        Validation results
    """
    validation = {
        "converged": solution.get("converged", False),
        "terms_computed": solution.get("terms_computed", 0),
        "final_error": solution.get("final_error", np.inf),
        "solution_norm": np.linalg.norm(solution.get("solution", [])),
        "is_finite": np.all(np.isfinite(solution.get("solution", [])))
    }
    
    return validation


def hpm_convergence_analysis(equation: Callable, initial_condition: Callable,
                           domain: Tuple[float, float] = (0.0, 1.0),
                           max_terms_list: List[int] = [5, 10, 15, 20]) -> Dict[str, Any]:
    """
    Analyze convergence of HPM method.
    
    Args:
        equation: Differential equation
        initial_condition: Initial condition
        domain: Solution domain
        max_terms_list: List of maximum terms to test
        
    Returns:
        Convergence analysis results
    """
    results = {}
    
    for max_terms in max_terms_list:
        solver = HomotopyPerturbationMethod(alpha=0.5, max_terms=max_terms)
        solution = solver.solve(equation, initial_condition, domain=domain)
        
        results[f"terms_{max_terms}"] = {
            "converged": solution["converged"],
            "final_error": solution["final_error"],
            "solution_norm": np.linalg.norm(solution["solution"])
        }
    
    return results
