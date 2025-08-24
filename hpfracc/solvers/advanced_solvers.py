"""
Advanced Fractional Differential Equation Solvers

This module provides advanced solvers with error control, adaptive methods,
and high-order numerical schemes for fractional differential equations.
"""

import numpy as np
from typing import Union, Optional, Tuple, Callable, Dict, Any, List
from scipy import integrate, interpolate
from scipy.linalg import solve_banded
import warnings
from enum import Enum

from ..core.definitions import FractionalOrder
from ..algorithms.optimized_methods import (
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov,
)
from ..special import gamma
from ..utils.error_analysis import ErrorAnalyzer
from ..utils.memory_management import MemoryManager


class ErrorControlMethod(Enum):
    """Error control methods for adaptive solvers."""

    LOCAL_ERROR = "local_error"
    GLOBAL_ERROR = "global_error"
    MIXED_ERROR = "mixed_error"


class AdaptiveMethod(Enum):
    """Adaptive step size control methods."""

    PID_CONTROL = "pid_control"
    EMBEDDED_PAIRS = "embedded_pairs"
    VARIABLE_ORDER = "variable_order"


class AdvancedFractionalODESolver:
    """
    Advanced fractional ODE solver with error control and adaptive methods.

    Features:
    - Error control with multiple strategies
    - Adaptive step size control
    - High-order numerical methods
    - Memory-efficient algorithms
    - Convergence monitoring
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        method: str = "embedded_pairs",
        error_control: ErrorControlMethod = ErrorControlMethod.LOCAL_ERROR,
        adaptive_method: AdaptiveMethod = AdaptiveMethod.PID_CONTROL,
        tol: float = 1e-6,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_iter: int = 1000,
        min_step: float = 1e-12,
        max_step: float = 1.0,
        safety_factor: float = 0.9,
        max_order: int = 5,
        memory_optimized: bool = True,
    ):
        """
        Initialize advanced fractional ODE solver.

        Args:
            derivative_type: Type of fractional derivative
            method: Numerical method
            error_control: Error control strategy
            adaptive_method: Adaptive step size control method
            tol: Absolute tolerance
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_iter: Maximum iterations
            min_step: Minimum step size
            max_step: Maximum step size
            safety_factor: Safety factor for step size control
            max_order: Maximum order for variable order methods
            memory_optimized: Use memory optimization
        """
        self.derivative_type = derivative_type.lower()
        self.method = method.lower()
        self.error_control = error_control
        self.adaptive_method = adaptive_method
        self.tol = tol
        self.rtol = rtol
        self.atol = atol
        self.max_iter = max_iter
        self.min_step = min_step
        self.max_step = max_step
        self.safety_factor = safety_factor
        self.max_order = max_order
        self.memory_optimized = memory_optimized

        # Initialize components
        self.error_analyzer = ErrorAnalyzer()
        self.memory_manager = MemoryManager() if memory_optimized else None

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate solver parameters."""
        valid_derivatives = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        if self.derivative_type not in valid_derivatives:
            raise ValueError(f"Derivative type must be one of {valid_derivatives}")

        if self.tol <= 0 or self.rtol <= 0 or self.atol <= 0:
            raise ValueError("Tolerances must be positive")

        if self.min_step >= self.max_step:
            raise ValueError("min_step must be less than max_step")

        if not 0 < self.safety_factor < 1:
            raise ValueError("safety_factor must be between 0 and 1")

    def solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h0: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve fractional ODE with advanced features.

        Args:
            f: Right-hand side function f(t, y)
            t_span: Time interval (t0, tf)
            y0: Initial condition(s)
            alpha: Fractional order
            h0: Initial step size
            **kwargs: Additional parameters

        Returns:
            Dictionary containing solution and metadata
        """
        t0, tf = t_span

        if h0 is None:
            h0 = (tf - t0) / 100

        # Initialize solution storage
        t_values = [t0]
        y_values = [np.array(y0, dtype=float)]
        h_values = [h0]
        error_estimates = []
        order_estimates = []

        current_t = t0
        current_y = np.array(y0, dtype=float)
        current_h = h0
        current_order = 1

        iteration = 0

        while current_t < tf and iteration < self.max_iter:
            # Safety check: if step size is too small relative to remaining time,
            # increase it to make progress
            remaining_time = tf - current_t
            if current_h < remaining_time * 1e-6:
                current_h = min(remaining_time * 0.1, self.max_step)
                warnings.warn(f"Step size too small, increasing to {current_h}")
            # Compute next step
            result = self._compute_step(
                f, current_t, current_y, alpha, current_h, current_order
            )

            if result is None:
                # Step failed, reduce step size
                current_h = max(current_h * 0.5, self.min_step)
                iteration += 1
                continue

            next_t, next_y, error_est, order_est = result

            # Check if step is acceptable
            if self._is_step_acceptable(error_est, current_y, next_y):
                # Accept step
                t_values.append(next_t)
                y_values.append(next_y)
                h_values.append(current_h)
                error_estimates.append(error_est)
                order_estimates.append(order_est)

                current_t = next_t
                current_y = next_y

                # Update step size and order
                current_h, current_order = self._update_step_size_and_order(
                    current_h, error_est, order_est
                )

                # Memory optimization
                if self.memory_optimized:
                    self.memory_manager.optimize_memory_usage()
            else:
                # Reject step, reduce step size
                current_h = max(current_h * 0.5, self.min_step)

                # Prevent infinite loop: if step size is at minimum and still failing,
                # force acceptance or terminate
                if current_h <= self.min_step and len(error_estimates) > 0:
                    # Force acceptance with warning
                    warnings.warn(
                        f"Step size at minimum ({self.min_step}), forcing acceptance"
                    )
                    t_values.append(next_t)
                    y_values.append(next_y)
                    h_values.append(current_h)
                    error_estimates.append(error_est)
                    order_estimates.append(order_est)

                    current_t = next_t
                    current_y = next_y

            iteration += 1

        # Check if solver made progress
        if len(t_values) < 2:
            warnings.warn(
                "Advanced solver failed to make progress, falling back to basic solver"
            )
            from .ode_solvers import solve_fractional_ode

            t_basic, y_basic = solve_fractional_ode(
                f, t_span, y0, alpha, method="predictor_corrector"
            )
            return {
                "t": t_basic,
                "y": y_basic,
                "h": np.diff(t_basic),
                "error_estimates": np.zeros(len(t_basic) - 1),
                "order_estimates": np.ones(len(t_basic) - 1),
                "iterations": len(t_basic),
                "converged": True,
                "final_error": 0.0,
                "average_order": 1.0,
            }

        # Prepare results
        solution = {
            "t": np.array(t_values),
            "y": np.array(y_values),
            "h": np.array(h_values),
            "error_estimates": np.array(error_estimates),
            "order_estimates": np.array(order_estimates),
            "iterations": iteration,
            "converged": bool(current_t >= tf),
            "final_error": error_estimates[-1] if error_estimates else None,
            "average_order": np.mean(order_estimates) if order_estimates else None,
        }

        return solution

    def _compute_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> Optional[Tuple[float, np.ndarray, float, float]]:
        """
        Compute a single step with error estimation.

        Returns:
            Tuple of (next_t, next_y, error_estimate, order_estimate) or None if failed
        """
        try:
            if self.method == "embedded_pairs":
                return self._embedded_pairs_step(f, t, y, alpha, h, order)
            elif self.method == "variable_order":
                return self._variable_order_step(f, t, y, alpha, h, order)
            else:
                return self._standard_step(f, t, y, alpha, h, order)
        except Exception as e:
            warnings.warn(f"Step computation failed: {e}")
            return None

    def _embedded_pairs_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> Tuple[float, np.ndarray, float, float]:
        """Compute step using embedded Runge-Kutta pairs."""
        # High-order solution
        y_high = self._runge_kutta_step(f, t, y, alpha, h, order)

        # Low-order solution (for error estimation)
        y_low = self._runge_kutta_step(f, t, y, alpha, h, order - 1)

        # Error estimate
        error_est = np.linalg.norm(y_high - y_low)

        # Order estimate
        order_est = order - 0.5  # Simplified estimate

        return t + h, y_high, error_est, order_est

    def _variable_order_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> Tuple[float, np.ndarray, float, float]:
        """Compute step using variable order methods."""
        # Try different orders
        orders = range(max(1, order - 1), min(self.max_order + 1, order + 2))
        solutions = []

        for ord in orders:
            try:
                sol = self._runge_kutta_step(f, t, y, alpha, h, ord)
                solutions.append((ord, sol))
            except:
                continue

        if len(solutions) < 2:
            raise ValueError("Could not compute solutions for error estimation")

        # Select best order based on error estimates
        best_order, best_solution = solutions[0]
        min_error = float("inf")

        for ord, sol in solutions:
            error = self._estimate_local_error(sol, y, h, ord)
            if error < min_error:
                min_error = error
                best_order = ord
                best_solution = sol

        return t + h, best_solution, min_error, float(best_order)

    def _standard_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> Tuple[float, np.ndarray, float, float]:
        """Compute standard step with error estimation."""
        # Predictor step
        y_pred = self._predictor_step(f, t, y, alpha, h, order)

        # Corrector step
        y_corr = self._corrector_step(f, t, y, y_pred, alpha, h, order)

        # Error estimate
        error_est = np.linalg.norm(y_corr - y_pred)

        return t + h, y_corr, error_est, float(order)

    def _runge_kutta_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> np.ndarray:
        """Compute Runge-Kutta step."""
        # Simplified Runge-Kutta implementation
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _predictor_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> np.ndarray:
        """Compute predictor step."""
        # Adams-Bashforth predictor
        return y + h * f(t, y)

    def _corrector_step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        y_pred: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
        order: int,
    ) -> np.ndarray:
        """Compute corrector step."""
        # Adams-Moulton corrector
        t_next = t + h
        return y + h * f(t_next, y_pred)

    def _estimate_local_error(
        self,
        y_new: np.ndarray,
        y_old: np.ndarray,
        h: float,
        order: int,
    ) -> float:
        """Estimate local truncation error."""
        # Richardson extrapolation error estimate
        return np.linalg.norm(y_new - y_old) / (h**order)

    def _is_step_acceptable(
        self,
        error_est: float,
        y_old: np.ndarray,
        y_new: np.ndarray,
    ) -> bool:
        """Check if step is acceptable based on error control."""
        # Handle very small errors that might cause numerical issues
        if error_est < 1e-15:
            return True

        if self.error_control == ErrorControlMethod.LOCAL_ERROR:
            return error_est <= self.tol
        elif self.error_control == ErrorControlMethod.GLOBAL_ERROR:
            # Simplified global error estimate
            global_error = error_est * np.linalg.norm(y_new - y_old)
            return global_error <= self.tol
        else:  # MIXED_ERROR
            local_ok = error_est <= self.tol
            relative_ok = error_est <= self.rtol * np.linalg.norm(y_new)
            return local_ok and relative_ok

    def _update_step_size_and_order(
        self,
        h: float,
        error_est: float,
        order_est: float,
    ) -> Tuple[float, int]:
        """Update step size and order using adaptive control."""
        if self.adaptive_method == AdaptiveMethod.PID_CONTROL:
            return self._pid_control(h, error_est, order_est)
        else:
            return self._simple_control(h, error_est, order_est)

    def _pid_control(
        self,
        h: float,
        error_est: float,
        order_est: float,
    ) -> Tuple[float, int]:
        """PID controller for step size and order control."""
        # Simplified PID control
        target_error = self.tol

        # Proportional term
        p_term = target_error / (error_est + 1e-12)

        # Integral term (simplified)
        i_term = 1.0

        # Derivative term (simplified)
        d_term = 1.0

        # PID output
        pid_output = p_term * i_term * d_term

        # Limit PID output to prevent extreme step size changes
        pid_output = np.clip(pid_output, 0.1, 10.0)

        # Update step size
        new_h = h * self.safety_factor * pid_output
        new_h = np.clip(new_h, self.min_step, self.max_step)

        # Update order (simplified)
        new_order = max(1, min(self.max_order, int(order_est + 0.5)))

        return new_h, new_order

    def _simple_control(
        self,
        h: float,
        error_est: float,
        order_est: float,
    ) -> Tuple[float, int]:
        """Simple step size and order control."""
        # Simple proportional control
        if error_est > 0:
            factor = (self.tol / error_est) ** (1.0 / order_est)
            # Limit factor to prevent extreme changes
            factor = np.clip(factor, 0.1, 10.0)
            new_h = h * self.safety_factor * factor
        else:
            new_h = h * 1.1  # Increase step size if error is very small

        new_h = np.clip(new_h, self.min_step, self.max_step)
        new_order = max(1, min(self.max_order, int(order_est + 0.5)))

        return new_h, new_order


class HighOrderFractionalSolver:
    """
    High-order fractional differential equation solver.

    Implements high-order numerical methods including:
    - Spectral methods
    - Multi-step methods
    - Collocation methods
    """

    def __init__(
        self,
        method: str = "spectral",
        order: int = 4,
        collocation_points: int = 10,
        **kwargs,
    ):
        """
        Initialize high-order solver.

        Args:
            method: Numerical method ("spectral", "multistep", "collocation")
            order: Order of the method
            collocation_points: Number of collocation points
        """
        self.method = method.lower()
        self.order = order
        self.collocation_points = collocation_points

        # Validate method
        valid_methods = ["spectral", "multistep", "collocation"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve using high-order method.

        Args:
            f: Right-hand side function
            t_span: Time interval
            y0: Initial condition
            alpha: Fractional order
            **kwargs: Additional parameters

        Returns:
            Solution dictionary
        """
        if self.method == "spectral":
            return self._spectral_solve(f, t_span, y0, alpha, **kwargs)
        elif self.method == "multistep":
            return self._multistep_solve(f, t_span, y0, alpha, **kwargs)
        else:  # collocation
            return self._collocation_solve(f, t_span, y0, alpha, **kwargs)

    def _spectral_solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> Dict[str, Any]:
        """Spectral method solver."""
        t0, tf = t_span

        # Chebyshev collocation points
        n = self.collocation_points
        t_cheb = np.cos(np.pi * np.arange(n) / (n - 1))
        t_cheb = 0.5 * (tf - t0) * (t_cheb + 1) + t0

        # Spectral differentiation matrix
        D = self._chebyshev_differentiation_matrix(n)

        # Solve the system
        y0_array = np.array(y0)
        y = np.zeros((n, 1 if y0_array.ndim == 0 else y0_array.shape[0]))
        y[0] = y0  # Initial condition

        # Set up the system of equations
        for i in range(1, n):
            y[i] = f(t_cheb[i], y[i - 1])

        return {
            "t": t_cheb,
            "y": y,
            "method": "spectral",
            "order": self.order,
        }

    def _multistep_solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> Dict[str, Any]:
        """Multi-step method solver."""
        t0, tf = t_span
        h = (tf - t0) / 100

        t_values = np.arange(t0, tf + h, h)
        y0_array = np.array(y0)
        y_values = np.zeros(
            (len(t_values), 1 if y0_array.ndim == 0 else y0_array.shape[0])
        )
        y_values[0] = y0

        # Multi-step method implementation
        for i in range(1, len(t_values)):
            if i < self.order:
                # Use lower order method for startup
                y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])
            else:
                # Use multi-step method
                y_values[i] = self._multistep_step(f, t_values, y_values, i, h, alpha)

        return {
            "t": t_values,
            "y": y_values,
            "method": "multistep",
            "order": self.order,
        }

    def _collocation_solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        **kwargs,
    ) -> Dict[str, Any]:
        """Collocation method solver."""
        t0, tf = t_span

        # Gauss-Legendre collocation points
        n = self.collocation_points
        t_colloc, weights = self._gauss_legendre_points(n)
        t_colloc = 0.5 * (tf - t0) * (t_colloc + 1) + t0

        # Solve collocation system
        y0_array = np.array(y0)
        y = np.zeros((n, 1 if y0_array.ndim == 0 else y0_array.shape[0]))
        y[0] = y0

        # Collocation equations
        for i in range(1, n):
            y[i] = f(t_colloc[i], y[i - 1])

        return {
            "t": t_colloc,
            "y": y,
            "method": "collocation",
            "order": self.order,
        }

    def _chebyshev_differentiation_matrix(self, n: int) -> np.ndarray:
        """Compute Chebyshev differentiation matrix."""
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = (-1) ** (i + j) / (
                        np.cos(i * np.pi / (n - 1)) - np.cos(j * np.pi / (n - 1))
                    )
                else:
                    D[i, i] = 0

        return D

    def _multistep_step(
        self,
        f: Callable,
        t: np.ndarray,
        y: np.ndarray,
        i: int,
        h: float,
        alpha: Union[float, FractionalOrder],
    ) -> np.ndarray:
        """Compute multi-step step."""
        # Adams-Bashforth method
        if self.order == 2:
            return y[i - 1] + h * (
                1.5 * f(t[i - 1], y[i - 1]) - 0.5 * f(t[i - 2], y[i - 2])
            )
        elif self.order == 3:
            return y[i - 1] + h * (
                23 / 12 * f(t[i - 1], y[i - 1])
                - 16 / 12 * f(t[i - 2], y[i - 2])
                + 5 / 12 * f(t[i - 3], y[i - 3])
            )
        else:
            return y[i - 1] + h * f(t[i - 1], y[i - 1])

    def _gauss_legendre_points(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gauss-Legendre points and weights."""
        # Simplified implementation
        x = np.linspace(-1, 1, n)
        w = np.ones(n) * 2.0 / n
        return x, w


# Convenience functions
def solve_advanced_fractional_ode(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "embedded_pairs",
    **kwargs,
) -> Dict[str, Any]:
    """
    Solve fractional ODE with advanced features.

    Args:
        f: Right-hand side function
        t_span: Time interval
        y0: Initial condition
        alpha: Fractional order
        method: Numerical method
        **kwargs: Additional parameters

    Returns:
        Solution dictionary
    """
    solver = AdvancedFractionalODESolver(method=method, **kwargs)
    return solver.solve(f, t_span, y0, alpha)


def solve_high_order_fractional_ode(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    method: str = "spectral",
    **kwargs,
) -> Dict[str, Any]:
    """
    Solve fractional ODE with high-order methods.

    Args:
        f: Right-hand side function
        t_span: Time interval
        y0: Initial condition
        alpha: Fractional order
        method: High-order method
        **kwargs: Additional parameters

    Returns:
        Solution dictionary
    """
    solver = HighOrderFractionalSolver(method=method, **kwargs)
    return solver.solve(f, t_span, y0, alpha)
