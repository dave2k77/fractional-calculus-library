"""
Advanced Predictor-Corrector Methods for Fractional Differential Equations

This module provides advanced predictor-corrector methods including
Adams-Bashforth-Moulton schemes, variable step size control, and error estimation.
"""

import numpy as np
from typing import Union, Optional, Tuple, Callable, Dict, Any, List
from scipy import linalg
import warnings

from hpfracc.core.definitions import FractionalOrder
from hpfracc.special import gamma


class PredictorCorrectorSolver:
    """
    Advanced predictor-corrector solver for fractional differential equations.

    Implements Adams-Bashforth-Moulton type schemes with variable step size
    control and error estimation.
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        order: int = 1,
        adaptive: bool = True,
        tol: float = 1e-6,
        max_iter: int = 10,
        min_h: float = 1e-8,
        max_h: float = 1e-2,
    ):
        """
        Initialize predictor-corrector solver.

        Args:
            derivative_type: Type of fractional derivative
            order: Order of the method
            adaptive: Use adaptive step size control
            tol: Tolerance for convergence
            max_iter: Maximum number of corrector iterations
            min_h: Minimum step size
            max_h: Maximum step size
        """
        self.derivative_type = derivative_type.lower()
        self.order = order
        self.adaptive = adaptive
        self.tol = tol
        self.max_iter = max_iter
        self.min_h = min_h
        self.max_h = max_h

        # Validate derivative type
        valid_derivatives = ["caputo", "riemann_liouville", "grunwald_letnikov"]
        if self.derivative_type not in valid_derivatives:
            raise ValueError(f"Derivative type must be one of {valid_derivatives}")

    def solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h0: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve fractional differential equation using predictor-corrector method.

        Args:
            f: Right-hand side function f(t, y)
            t_span: Time interval (t0, tf)
            y0: Initial condition(s)
            alpha: Fractional order
            h: Step size (None for adaptive)
            **kwargs: Additional solver parameters

        Returns:
            Tuple of (t_values, y_values)
        """
        t0, tf = t_span

        if h0 is None:
            h0 = (tf - t0) / 100

        if self.adaptive:
            return self._solve_adaptive(f, t0, tf, y0, alpha, h0, **kwargs)
        else:
            return self._solve_fixed_step(f, t0, tf, y0, alpha, h0, **kwargs)

    def _solve_fixed_step(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h0: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with fixed step size.

        Args:
            f: Right-hand side function
            t0: Initial time
            tf: Final time
            y0: Initial condition
            alpha: Fractional order
            h: Step size
            **kwargs: Additional parameters

        Returns:
            Tuple of (t_values, y_values)
        """
        # Convert to arrays if needed
        if np.isscalar(y0):
            y0 = np.array([y0])

        # Time grid
        t_values = np.arange(t0, tf + h0, h0)
        N = len(t_values)

        # Solution array
        y_values = np.zeros((N, len(y0)))
        y_values[0] = y0

        # Compute fractional derivative coefficients
        coeffs = self._compute_fractional_coefficients(alpha, N)

        # Main iteration loop
        for n in range(1, N):
            t_n = t_values[n]

            # Predictor step
            y_pred = self._predictor_step(f, t_values, y_values, n, alpha, coeffs, h0)

            # Corrector step with iteration
            y_corr = self._corrector_step(
                f, t_values, y_values, y_pred, n, alpha, coeffs, h0
            )

            # Iterative correction
            for _ in range(self.max_iter):
                y_old = y_corr.copy()
                y_corr = self._corrector_step(
                    f, t_values, y_values, y_pred, n, alpha, coeffs, h0
                )

                if np.allclose(y_corr, y_old, rtol=self.tol):
                    break

            y_values[n] = y_corr

        return t_values, y_values

    def _solve_adaptive(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h0: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve with adaptive step size control.

        Args:
            f: Right-hand side function
            t0: Initial time
            tf: Final time
            y0: Initial condition
            alpha: Fractional order
            h0: Initial step size
            **kwargs: Additional parameters

        Returns:
            Tuple of (t_values, y_values)
        """
        # Convert to arrays if needed
        if np.isscalar(y0):
            y0 = np.array([y0])

        # Initialize solution arrays
        t_values = [t0]
        y_values = [y0.copy()]

        t_current = t0
        y_current = y0.copy()
        h_current = h0

        iteration_count = 0
        max_iterations = int((tf - t0) / self.min_h) + 1000  # Safety limit

        while t_current < tf and iteration_count < max_iterations:
            iteration_count += 1

            # Compute solution with current step size
            t_next = min(t_current + h_current, tf)
            h_actual = t_next - t_current

            # Predictor step
            y_pred = self._adaptive_predictor_step(
                f, t_current, t_next, y_current, alpha, h_actual
            )

            # Corrector step
            y_corr = self._adaptive_corrector_step(
                f, t_current, t_next, y_current, y_pred, alpha, h_actual
            )

            # Estimate error
            error = self._estimate_error(y_pred, y_corr)

            # Check if error is acceptable
            if error <= self.tol:
                # Accept step
                t_values.append(t_next)
                y_values.append(y_corr)
                t_current = t_next
                y_current = y_corr

                # Adjust step size for next step
                h_current = min(self.max_h, h_current * (self.tol / error) ** 0.5)
            else:
                # Reject step and reduce step size
                h_current = max(self.min_h, h_current * (self.tol / error) ** 0.25)

                # Safety check: if step size is at minimum and still not converging, force progress
                if h_current <= self.min_h and error > self.tol:
                    # Force acceptance with warning
                    import warnings

                    warnings.warn(
                        f"Step size at minimum ({self.min_h}), forcing acceptance of step with error {error:.2e}"
                    )
                    t_values.append(t_next)
                    y_values.append(y_corr)
                    t_current = t_next
                    y_current = y_corr
                    h_current = self.min_h * 2  # Try to increase step size

        return np.array(t_values), np.array(y_values)

    def _compute_fractional_coefficients(
        self, alpha: Union[float, FractionalOrder], N: int
    ) -> np.ndarray:
        """
        Compute fractional derivative coefficients.

        Args:
            alpha: Fractional order
            N: Number of time steps

        Returns:
            Array of coefficients
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        coeffs = np.zeros(N)
        coeffs[0] = 1.0

        for j in range(1, N):
            if self.derivative_type == "caputo":
                coeffs[j] = (j + 1) ** alpha_val - j**alpha_val
            elif self.derivative_type == "grunwald_letnikov":
                coeffs[j] = coeffs[j - 1] * (1 - (alpha_val + 1) / j)
            else:  # Riemann-Liouville
                coeffs[j] = (
                    (-1) ** j
                    * gamma(alpha_val + 1)
                    / (gamma(j + 1) * gamma(alpha_val - j + 1))
                )

        return coeffs

    def _predictor_step(
        self,
        f: Callable,
        t_values: np.ndarray,
        y_values: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        coeffs: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Predictor step (Adams-Bashforth type).

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Predicted solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Compute fractional derivative term
        frac_term = 0.0
        for j in range(n):
            frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])

        # Predictor formula
        t_n = t_values[n]
        y_pred = y_values[n - 1] + (h**alpha_val / gamma(alpha_val + 1)) * (
            f(t_n, y_values[n - 1]) - frac_term
        )

        return y_pred

    def _corrector_step(
        self,
        f: Callable,
        t_values: np.ndarray,
        y_values: np.ndarray,
        y_pred: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        coeffs: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Corrector step (Adams-Moulton type).

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            y_pred: Predicted solution
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Corrected solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Compute fractional derivative term
        frac_term = 0.0
        for j in range(n):
            frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])

        # Corrector formula
        t_n = t_values[n]
        y_corr = y_values[n - 1] + (h**alpha_val / gamma(alpha_val + 1)) * (
            0.5 * (f(t_n, y_pred) + f(t_values[n - 1], y_values[n - 1])) - frac_term
        )

        return y_corr

    def _adaptive_predictor_step(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Adaptive predictor step.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            alpha: Fractional order
            h: Step size

        Returns:
            Predicted solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Simple predictor
        y_pred = y_current + (h**alpha_val / gamma(alpha_val + 1)) * f(
            t_current, y_current
        )

        return y_pred

    def _adaptive_corrector_step(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        y_pred: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Adaptive corrector step.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            y_pred: Predicted solution
            alpha: Fractional order
            h: Step size

        Returns:
            Corrected solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Corrector with iteration
        y_corr = y_pred
        for _ in range(self.max_iter):
            y_old = y_corr.copy()
            y_corr = y_current + (h**alpha_val / gamma(alpha_val + 1)) * (
                0.5 * (f(t_next, y_pred) + f(t_current, y_current))
            )

            if np.allclose(y_corr, y_old, rtol=self.tol):
                break

        return y_corr

    def _estimate_error(self, y_pred: np.ndarray, y_corr: np.ndarray) -> float:
        """
        Estimate local truncation error.

        Args:
            y_pred: Predicted solution
            y_corr: Corrected solution

        Returns:
            Estimated error
        """
        return np.linalg.norm(y_corr - y_pred)


class AdamsBashforthMoultonSolver(PredictorCorrectorSolver):
    """
    Adams-Bashforth-Moulton solver for fractional differential equations.

    Implements the classical Adams-Bashforth-Moulton scheme adapted for
    fractional differential equations.
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        order: int = 1,
        adaptive: bool = True,
        tol: float = 1e-6,
        max_iter: int = 10,
    ):
        """
        Initialize Adams-Bashforth-Moulton solver.

        Args:
            derivative_type: Type of fractional derivative
            order: Order of the method
            adaptive: Use adaptive step size control
            tol: Tolerance for convergence
            max_iter: Maximum number of corrector iterations
        """
        super().__init__(derivative_type, order, adaptive, tol, max_iter)

    def _predictor_step(
        self,
        f: Callable,
        t_values: np.ndarray,
        y_values: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        coeffs: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Adams-Bashforth predictor step.

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Predicted solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Adams-Bashforth weights
        weights = self._compute_adams_bashforth_weights(n, alpha_val)

        # Predictor formula
        t_n = t_values[n]
        y_pred = y_values[n - 1]

        for j in range(n):
            t_j = t_values[j]
            y_pred += (
                h**alpha_val / gamma(alpha_val + 1) * weights[j] * f(t_j, y_values[j])
            )

        return y_pred

    def _corrector_step(
        self,
        f: Callable,
        t_values: np.ndarray,
        y_values: np.ndarray,
        y_pred: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        coeffs: np.ndarray,
        h: float,
    ) -> np.ndarray:
        """
        Adams-Moulton corrector step.

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            y_pred: Predicted solution
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Corrected solution
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Adams-Moulton weights
        weights = self._compute_adams_moulton_weights(n, alpha_val)

        # Corrector formula
        t_n = t_values[n]
        y_corr = y_values[n - 1]

        for j in range(n):
            t_j = t_values[j]
            if j == n - 1:
                y_corr += (
                    h**alpha_val / gamma(alpha_val + 1) * weights[j] * f(t_j, y_pred)
                )
            else:
                y_corr += (
                    h**alpha_val
                    / gamma(alpha_val + 1)
                    * weights[j]
                    * f(t_j, y_values[j])
                )

        return y_corr

    def _compute_adams_bashforth_weights(self, n: int, alpha: float) -> np.ndarray:
        """
        Compute Adams-Bashforth weights.

        Args:
            n: Current time step
            alpha: Fractional order

        Returns:
            Array of weights
        """
        weights = np.zeros(n)

        # Simplified Adams-Bashforth weights
        for j in range(n):
            weights[j] = (n - j) ** alpha - (n - j - 1) ** alpha

        return weights

    def _compute_adams_moulton_weights(self, n: int, alpha: float) -> np.ndarray:
        """
        Compute Adams-Moulton weights.

        Args:
            n: Current time step
            alpha: Fractional order

        Returns:
            Array of weights
        """
        weights = np.zeros(n)

        # Simplified Adams-Moulton weights
        for j in range(n):
            if j == n - 1:
                weights[j] = 1.0
            else:
                weights[j] = (n - j) ** alpha - (n - j - 1) ** alpha

        return weights


class VariableStepPredictorCorrector(PredictorCorrectorSolver):
    """
    Variable step size predictor-corrector solver.

    Implements predictor-corrector methods with sophisticated step size
    control and error estimation.
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        order: int = 1,
        tol: float = 1e-6,
        max_iter: int = 10,
        min_h: float = 1e-8,
        max_h: float = 1e-2,
        safety_factor: float = 0.9,
    ):
        """
        Initialize variable step predictor-corrector solver.

        Args:
            derivative_type: Type of fractional derivative
            order: Order of the method
            tol: Tolerance for error control
            max_iter: Maximum number of corrector iterations
            min_h: Minimum step size
            max_h: Maximum step size
            safety_factor: Safety factor for step size control
        """
        super().__init__(derivative_type, order, True, tol, max_iter, min_h, max_h)
        self.safety_factor = safety_factor

    def _estimate_error(self, y_pred: np.ndarray, y_corr: np.ndarray) -> float:
        """
        Enhanced error estimation.

        Args:
            y_pred: Predicted solution
            y_corr: Corrected solution

        Returns:
            Estimated error
        """
        # Use relative error for better control
        error = np.linalg.norm(y_corr - y_pred) / (np.linalg.norm(y_corr) + 1e-12)
        return error

    def _adaptive_step_size_control(
        self, error: float, h_current: float, order: int
    ) -> float:
        """
        Adaptive step size control.

        Args:
            error: Estimated error
            h_current: Current step size
            order: Method order
            alpha: Fractional order

        Returns:
            New step size
        """
        if error <= self.tol:
            # Accept step and increase step size
            factor = min(
                2.0, self.safety_factor * (self.tol / error) ** (1.0 / (order + 1))
            )
            h_new = min(self.max_h, h_current * factor)
        else:
            # Reject step and decrease step size
            factor = max(
                0.1, self.safety_factor * (self.tol / error) ** (1.0 / (order + 1))
            )
            h_new = max(self.min_h, h_current * factor)

        return h_new


# Convenience functions
def solve_predictor_corrector(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    derivative_type: str = "caputo",
    method: str = "standard",
    adaptive: bool = True,
    h: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve using predictor-corrector method.

    Args:
        f: Right-hand side function f(t, y)
        t_span: Time interval (t0, tf)
        y0: Initial condition(s)
        alpha: Fractional order
        derivative_type: Type of fractional derivative
        method: Method type ("standard", "adams_bashforth_moulton", "variable_step")
        adaptive: Use adaptive step size control
        h: Step size (None for adaptive)
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, y_values)
    """
    if method == "standard":
        solver = PredictorCorrectorSolver(derivative_type, adaptive=adaptive)
    elif method == "adams_bashforth_moulton":
        solver = AdamsBashforthMoultonSolver(derivative_type, adaptive=adaptive)
    elif method == "variable_step":
        solver = VariableStepPredictorCorrector(derivative_type)
    else:
        raise ValueError(f"Unknown method: {method}")

    return solver.solve(f, t_span, y0, alpha, h, **kwargs)


def solve_adams_bashforth_moulton(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    derivative_type: str = "caputo",
    adaptive: bool = True,
    h: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve using Adams-Bashforth-Moulton method.

    Args:
        f: Right-hand side function f(t, y)
        t_span: Time interval (t0, tf)
        y0: Initial condition(s)
        alpha: Fractional order
        derivative_type: Type of fractional derivative
        adaptive: Use adaptive step size control
        h: Step size (None for adaptive)
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, y_values)
    """
    solver = AdamsBashforthMoultonSolver(derivative_type, adaptive=adaptive)
    return solver.solve(f, t_span, y0, alpha, h, **kwargs)


def solve_variable_step_predictor_corrector(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    derivative_type: str = "caputo",
    h0: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve using variable step size predictor-corrector method.

    Args:
        f: Right-hand side function f(t, y)
        t_span: Time interval (t0, tf)
        y0: Initial condition(s)
        alpha: Fractional order
        derivative_type: Type of fractional derivative
        h0: Initial step size
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, y_values)
    """
    solver = VariableStepPredictorCorrector(derivative_type)
    return solver.solve(f, t_span, y0, alpha, h0, **kwargs)
