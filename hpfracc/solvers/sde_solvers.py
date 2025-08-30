"""
Stochastic Differential Equation (SDE) Solvers

This module implements various numerical methods for solving stochastic
differential equations, including Euler-Maruyama, Milstein, and Heun methods.
"""

import numpy as np
import torch
from typing import Union, Callable, Optional, Tuple, List, Dict, Any
from scipy.stats import norm
import warnings

from ..core.definitions import FractionalOrder
from ..core.utilities import validate_fractional_order


class BaseSDESolver:
    """
    Base class for SDE solvers.
    
    This class provides common functionality for all SDE solvers
    including noise generation, error estimation, and stability analysis.
    """
    
    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        """
        Initialize base SDE solver.
        
        Args:
            dt: Time step size
            seed: Random seed for reproducibility
        """
        self.dt = dt
        if seed is not None:
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.manual_seed(seed)
    
    def generate_wiener_process(self, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        Generate Wiener process (Brownian motion) increments.
        
        Args:
            n_steps: Number of time steps
            n_paths: Number of sample paths
            
        Returns:
            Array of Wiener process increments with shape (n_paths, n_steps)
        """
        # Generate standard normal random variables
        increments = np.random.normal(0, np.sqrt(self.dt), (n_paths, n_steps))
        return increments
    
    def estimate_error(self, solution: np.ndarray, analytical_solution: Optional[Callable] = None) -> Dict[str, float]:
        """
        Estimate numerical error of the solution.
        
        Args:
            solution: Numerical solution
            analytical_solution: Analytical solution function (if available)
            
        Returns:
            Dictionary containing error estimates
        """
        if analytical_solution is not None:
            # Compute analytical solution at the same time points
            t = np.arange(0, solution.shape[-1]) * self.dt
            analytical = analytical_solution(t)
            
            # Compute error
            error = np.abs(solution - analytical)
            mean_error = np.mean(error)
            max_error = np.max(error)
            rmse = np.sqrt(np.mean(error**2))
        else:
            # Estimate error based on solution properties
            mean_error = np.mean(np.abs(np.diff(solution)))
            max_error = np.max(np.abs(np.diff(solution)))
            rmse = np.sqrt(np.mean(np.diff(solution)**2))
        
        return {
            "mean_error": mean_error,
            "max_error": max_error,
            "rmse": rmse,
            "dt": self.dt
        }
    
    def check_stability(self, solution: np.ndarray, threshold: float = 1e6) -> Dict[str, Any]:
        """
        Check numerical stability of the solution.
        
        Args:
            solution: Numerical solution
            threshold: Threshold for detecting instability
            
        Returns:
            Dictionary containing stability analysis
        """
        is_finite = np.all(np.isfinite(solution))
        is_stable = np.all(np.abs(solution) < threshold)
        
        if is_finite and is_stable:
            max_magnitude = np.max(np.abs(solution))
            mean_magnitude = np.mean(np.abs(solution))
        else:
            max_magnitude = np.inf
            mean_magnitude = np.inf
        
        return {
            "is_finite": is_finite,
            "is_stable": is_stable,
            "max_magnitude": max_magnitude,
            "mean_magnitude": mean_magnitude,
            "threshold": threshold
        }


class EulerMaruyama(BaseSDESolver):
    """
    Euler-Maruyama method for solving SDEs.
    
    This is a first-order method (convergence order 0.5 in the strong sense)
    that extends the Euler method to stochastic differential equations.
    """
    
    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        """
        Initialize Euler-Maruyama solver.
        
        Args:
            dt: Time step size
            seed: Random seed for reproducibility
        """
        super().__init__(dt, seed)
    
    def solve(self, drift: Callable, diffusion: Callable, x0: Union[float, np.ndarray],
              t_span: Tuple[float, float], n_steps: Optional[int] = None,
              n_paths: int = 1) -> Dict[str, Any]:
        """
        Solve SDE using Euler-Maruyama method.
        
        Args:
            drift: Drift function a(x, t)
            diffusion: Diffusion function b(x, t)
            x0: Initial condition
            t_span: Time interval (t0, tf)
            n_steps: Number of time steps (if None, computed from dt)
            n_paths: Number of sample paths
            
        Returns:
            Dictionary containing solution and metadata
        """
        t0, tf = t_span
        
        if n_steps is None:
            n_steps = int((tf - t0) / self.dt)
        
        # Time grid
        t = np.linspace(t0, tf, n_steps + 1)
        dt = (tf - t0) / n_steps
        
        # Initialize solution
        if isinstance(x0, (int, float)):
            x0 = np.array([x0])
        
        solution = np.zeros((n_paths, n_steps + 1, len(x0)))
        solution[:, 0, :] = x0
        
        # Generate Wiener process increments
        dW = self.generate_wiener_process(n_steps, n_paths)
        
        # Euler-Maruyama iteration
        for i in range(n_steps):
            ti = t[i]
            xi = solution[:, i, :]
            
            # Drift and diffusion terms
            a = drift(xi, ti)
            b = diffusion(xi, ti)
            
            # Update solution
            solution[:, i + 1, :] = xi + a * dt + b * dW[:, i:i+1]
        
        # Error and stability analysis
        error_estimate = self.estimate_error(solution)
        stability = self.check_stability(solution)
        
        return {
            "solution": solution,
            "t": t,
            "dt": dt,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "error_estimate": error_estimate,
            "stability": stability,
            "method": "Euler-Maruyama"
        }


class Milstein(BaseSDESolver):
    """
    Milstein method for solving SDEs.
    
    This is a first-order method (convergence order 1.0 in the strong sense)
    that improves upon Euler-Maruyama by including higher-order terms.
    """
    
    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        """
        Initialize Milstein solver.
        
        Args:
            dt: Time step size
            seed: Random seed for reproducibility
        """
        super().__init__(dt, seed)
    
    def solve(self, drift: Callable, diffusion: Callable, diffusion_derivative: Callable,
              x0: Union[float, np.ndarray], t_span: Tuple[float, float],
              n_steps: Optional[int] = None, n_paths: int = 1) -> Dict[str, Any]:
        """
        Solve SDE using Milstein method.
        
        Args:
            drift: Drift function a(x, t)
            diffusion: Diffusion function b(x, t)
            diffusion_derivative: Derivative of diffusion function b'(x, t)
            x0: Initial condition
            t_span: Time interval (t0, tf)
            n_steps: Number of time steps (if None, computed from dt)
            n_paths: Number of sample paths
            
        Returns:
            Dictionary containing solution and metadata
        """
        t0, tf = t_span
        
        if n_steps is None:
            n_steps = int((tf - t0) / self.dt)
        
        # Time grid
        t = np.linspace(t0, tf, n_steps + 1)
        dt = (tf - t0) / n_steps
        
        # Initialize solution
        if isinstance(x0, (int, float)):
            x0 = np.array([x0])
        
        solution = np.zeros((n_paths, n_steps + 1, len(x0)))
        solution[:, 0, :] = x0
        
        # Generate Wiener process increments
        dW = self.generate_wiener_process(n_steps, n_paths)
        
        # Milstein iteration
        for i in range(n_steps):
            ti = t[i]
            xi = solution[:, i, :]
            
            # Drift and diffusion terms
            a = drift(xi, ti)
            b = diffusion(xi, ti)
            b_prime = diffusion_derivative(xi, ti)
            
            # Milstein correction term
            correction = 0.5 * b * b_prime * (dW[:, i:i+1]**2 - dt)
            
            # Update solution
            solution[:, i + 1, :] = xi + a * dt + b * dW[:, i:i+1] + correction
        
        # Error and stability analysis
        error_estimate = self.estimate_error(solution)
        stability = self.check_stability(solution)
        
        return {
            "solution": solution,
            "t": t,
            "dt": dt,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "error_estimate": error_estimate,
            "stability": stability,
            "method": "Milstein"
        }


class Heun(BaseSDESolver):
    """
    Heun method for solving SDEs.
    
    This is a predictor-corrector method that improves stability
    compared to explicit methods like Euler-Maruyama.
    """
    
    def __init__(self, dt: float = 0.01, seed: Optional[int] = None):
        """
        Initialize Heun solver.
        
        Args:
            dt: Time step size
            seed: Random seed for reproducibility
        """
        super().__init__(dt, seed)
    
    def solve(self, drift: Callable, diffusion: Callable, x0: Union[float, np.ndarray],
              t_span: Tuple[float, float], n_steps: Optional[int] = None,
              n_paths: int = 1) -> Dict[str, Any]:
        """
        Solve SDE using Heun method.
        
        Args:
            drift: Drift function a(x, t)
            diffusion: Diffusion function b(x, t)
            x0: Initial condition
            t_span: Time interval (t0, tf)
            n_steps: Number of time steps (if None, computed from dt)
            n_paths: Number of sample paths
            
        Returns:
            Dictionary containing solution and metadata
        """
        t0, tf = t_span
        
        if n_steps is None:
            n_steps = int((tf - t0) / self.dt)
        
        # Time grid
        t = np.linspace(t0, tf, n_steps + 1)
        dt = (tf - t0) / n_steps
        
        # Initialize solution
        if isinstance(x0, (int, float)):
            x0 = np.array([x0])
        
        solution = np.zeros((n_paths, n_steps + 1, len(x0)))
        solution[:, 0, :] = x0
        
        # Generate Wiener process increments
        dW = self.generate_wiener_process(n_steps, n_paths)
        
        # Heun iteration
        for i in range(n_steps):
            ti = t[i]
            xi = solution[:, i, :]
            
            # Predictor step (Euler-Maruyama)
            a_pred = drift(xi, ti)
            b_pred = diffusion(xi, ti)
            x_pred = xi + a_pred * dt + b_pred * dW[:, i:i+1]
            
            # Corrector step
            a_corr = drift(x_pred, t[i + 1])
            b_corr = diffusion(x_pred, t[i + 1])
            
            # Heun update
            solution[:, i + 1, :] = xi + 0.5 * (a_pred + a_corr) * dt + \
                                   0.5 * (b_pred + b_corr) * dW[:, i:i+1]
        
        # Error and stability analysis
        error_estimate = self.estimate_error(solution)
        stability = self.check_stability(solution)
        
        return {
            "solution": solution,
            "t": t,
            "dt": dt,
            "n_steps": n_steps,
            "n_paths": n_paths,
            "error_estimate": error_estimate,
            "stability": stability,
            "method": "Heun"
        }


# Factory functions for creating SDE solvers
def create_sde_solver(solver_type: str = "euler_maruyama", **kwargs) -> BaseSDESolver:
    """
    Factory function to create SDE solver instances.
    
    Args:
        solver_type: Type of solver ("euler_maruyama", "milstein", "heun")
        **kwargs: Additional arguments for solver initialization
        
    Returns:
        SDE solver instance
        
    Raises:
        ValueError: If solver_type is not recognized
    """
    if solver_type == "euler_maruyama":
        return EulerMaruyama(**kwargs)
    elif solver_type == "milstein":
        return Milstein(**kwargs)
    elif solver_type == "heun":
        return Heun(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Must be one of: euler_maruyama, milstein, heun")


def get_sde_solver_properties(solver: BaseSDESolver) -> Dict[str, Any]:
    """
    Get properties of an SDE solver instance.
    
    Args:
        solver: SDE solver instance
        
    Returns:
        Dictionary containing solver properties
    """
    properties = {
        "dt": solver.dt,
        "solver_type": solver.__class__.__name__,
        "method": getattr(solver, 'method', 'Unknown')
    }
    
    return properties


def validate_sde_parameters(dt: float, n_steps: Optional[int] = None,
                           n_paths: int = 1) -> Dict[str, Any]:
    """
    Validate SDE solver parameters.
    
    Args:
        dt: Time step size
        n_steps: Number of time steps
        n_paths: Number of sample paths
        
    Returns:
        Validation results dictionary
        
    Raises:
        ValueError: If parameters are invalid
    """
    validation = {"valid": True, "errors": []}
    
    # Validate dt
    if not isinstance(dt, (int, float)) or dt <= 0:
        validation["valid"] = False
        validation["errors"].append("dt must be a positive number")
    else:
        validation["dt"] = dt
    
    # Validate n_steps
    if n_steps is not None:
        if not isinstance(n_steps, int) or n_steps <= 0:
            validation["valid"] = False
            validation["errors"].append("n_steps must be a positive integer")
        else:
            validation["n_steps"] = n_steps
    
    # Validate n_paths
    if not isinstance(n_paths, int) or n_paths <= 0:
        validation["valid"] = False
        validation["errors"].append("n_paths must be a positive integer")
    else:
        validation["n_paths"] = n_paths
    
    if not validation["valid"]:
        raise ValueError(f"Invalid SDE parameters: {'; '.join(validation['errors'])}")
    
    return validation
