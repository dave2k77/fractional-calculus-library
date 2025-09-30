"""
Fractional Ordinary Differential Equation Solvers

This module provides comprehensive solvers for fractional ODEs including
various numerical methods, adaptive step size control, and error estimation.
"""

import numpy as np
from typing import Union, Optional, Tuple, Callable

from ..core.definitions import FractionalOrder

# Use adapter system for gamma function instead of direct imports
def _get_gamma_function():
    """Get gamma function through adapter system."""
    try:
        from ..special.gamma_beta import gamma_function as gamma
        return gamma
    except Exception:
        # Fallback to scipy
        from scipy.special import gamma
        return gamma

gamma = _get_gamma_function()


class FractionalODESolver:
    """
    Base class for fractional ODE solvers.

    Provides common functionality for solving fractional ordinary
    differential equations of the form:

    D^α y(t) = f(t, y(t))

    where D^α is a fractional derivative operator.
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        method: str = "predictor_corrector",
        adaptive: bool = True,
        tol: float = 1e-6,
        max_iter: int = 1000,
        *,
        fractional_order: Optional[Union[float, FractionalOrder]] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        """
        Initialize fractional ODE solver.

        Args:
            derivative_type: Type of fractional derivative ("caputo", "riemann_liouville", "grunwald_letnikov")
            method: Numerical method ("predictor_corrector", "adams_bashforth", "runge_kutta")
            adaptive: Use adaptive step size control
            tol: Tolerance for convergence
            max_iter: Maximum number of iterations
        """
        self.derivative_type = derivative_type.lower()
        self.method = method.lower()
        self.adaptive = adaptive
        self.tol = tol
        self.max_iter = max_iter
        # Accept optional fractional_order for compatibility; stored as attribute only
        self.fractional_order = fractional_order
        # Accept rtol/atol but map to tol if provided (basic solver uses single tol)
        if rtol is not None:
            self.tol = min(self.tol, float(rtol))
        if atol is not None:
            self.tol = min(self.tol, float(atol))

        # Validate derivative type
        valid_derivatives = [
            "caputo", "riemann_liouville", "grunwald_letnikov"]
        if self.derivative_type not in valid_derivatives:
            raise ValueError(
                f"Derivative type must be one of {valid_derivatives}")

        # Validate method
        valid_methods = [
            "predictor_corrector",
            "adams_bashforth",
            "runge_kutta",
            "euler",
        ]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve fractional ODE.

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

        if h is None:
            h = (tf - t0) / 100  # Default step size

        if self.method == "predictor_corrector":
            return self._solve_predictor_corrector(
                f, t0, tf, y0, alpha, h, **kwargs)
        elif self.method == "adams_bashforth":
            return self._solve_adams_bashforth(
                f, t0, tf, y0, alpha, h, **kwargs)
        elif self.method == "runge_kutta":
            return self._solve_runge_kutta(f, t0, tf, y0, alpha, h, **kwargs)
        elif self.method == "euler":
            return self._solve_euler(f, t0, tf, y0, alpha, h, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _solve_predictor_corrector(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using predictor-corrector method.

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
        t_values = np.arange(t0, tf + h, h)
        N = len(t_values)

        # Solution array
        y_values = np.zeros((N, len(y0)))
        y_values[0] = y0

        # Compute fractional derivative coefficients
        coeffs = self._compute_fractional_coefficients(alpha, N)

        # Main iteration loop
        for n in range(1, N):
            t_values[n]

            # Predictor step (Adams-Bashforth type)
            y_pred = self._predictor_step(
                f, t_values, y_values, n, alpha, coeffs, h)

            # Corrector step (Adams-Moulton type)
            y_corr = self._corrector_step(
                f, t_values, y_values, y_pred, n, alpha, coeffs, h
            )

            # Iterative correction
            for _ in range(self.max_iter):
                y_old = y_corr.copy()
                y_corr = self._corrector_step(
                    f, t_values, y_values, y_pred, n, alpha, coeffs, h
                )

                if np.allclose(y_corr, y_old, rtol=self.tol):
                    break

            y_values[n] = y_corr

        return t_values, y_values

    def _solve_adams_bashforth(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using Adams-Bashforth method.

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
        t_values = np.arange(t0, tf + h, h)
        N = len(t_values)

        # Solution array
        y_values = np.zeros((N, len(y0)))
        y_values[0] = y0

        # Compute fractional derivative coefficients
        coeffs = self._compute_fractional_coefficients(alpha, N)

        # Main iteration loop
        for n in range(1, N):
            t_values[n]

            # Adams-Bashforth step
            y_values[n] = self._adams_bashforth_step(
                f, t_values, y_values, n, alpha, coeffs, h
            )

        return t_values, y_values

    def _solve_runge_kutta(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using fractional Runge-Kutta method.

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
        t_values = np.arange(t0, tf + h, h)
        N = len(t_values)

        # Solution array
        y_values = np.zeros((N, len(y0)))
        y_values[0] = y0

        # Main iteration loop
        for n in range(1, N):
            t_values[n]

            # Fractional Runge-Kutta step
            y_values[n] = self._runge_kutta_step(
                f, t_values, y_values, n, alpha, h)

        return t_values, y_values

    def _solve_euler(
        self,
        f: Callable,
        t0: float,
        tf: float,
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h: float,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using fractional Euler method.

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
        t_values = np.arange(t0, tf + h, h)
        N = len(t_values)

        # Solution array
        y_values = np.zeros((N, len(y0)))
        y_values[0] = y0

        # Compute fractional derivative coefficients
        coeffs = self._compute_fractional_coefficients(alpha, N)

        # Main iteration loop
        for n in range(1, N):
            t_values[n]

            # Fractional Euler step
            y_values[n] = self._euler_step(
                f, t_values, y_values, n, alpha, coeffs, h)

        return t_values, y_values

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
        Predictor step for Adams-Bashforth type method.

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
        Corrector step for Adams-Moulton type method.

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
            0.5 * (f(t_n, y_pred) +
                   f(t_values[n - 1], y_values[n - 1])) - frac_term
        )

        return y_corr

    def _adams_bashforth_step(
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
        Adams-Bashforth step.

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Solution at next time step
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Compute fractional derivative term
        frac_term = 0.0
        for j in range(n):
            frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])

        # Adams-Bashforth formula
        t_n = t_values[n]
        y_next = y_values[n - 1] + (h**alpha_val / gamma(alpha_val + 1)) * (
            f(t_n, y_values[n - 1]) - frac_term
        )

        return y_next

    def _runge_kutta_step(
        self,
        f: Callable,
        t_values: np.ndarray,
        y_values: np.ndarray,
        n: int,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Fractional Runge-Kutta step.

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            n: Current time step
            alpha: Fractional order
            h: Step size

        Returns:
            Solution at next time step
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        t_n = t_values[n]
        y_n = y_values[n - 1]

        # Runge-Kutta coefficients
        k1 = f(t_n, y_n)
        k2 = f(t_n + h / 2, y_n + h / 2 * k1)
        k3 = f(t_n + h / 2, y_n + h / 2 * k2)
        k4 = f(t_n + h, y_n + h * k3)

        # Runge-Kutta formula
        y_next = (
            y_n
            + (h**alpha_val / gamma(alpha_val + 1)) *
            (k1 + 2 * k2 + 2 * k3 + k4) / 6
        )

        return y_next

    def _euler_step(
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
        Fractional Euler step.

        Args:
            f: Right-hand side function
            t_values: Time points
            y_values: Solution values
            n: Current time step
            alpha: Fractional order
            coeffs: Fractional coefficients
            h: Step size

        Returns:
            Solution at next time step
        """
        if isinstance(alpha, FractionalOrder):
            alpha_val = alpha.alpha
        else:
            alpha_val = alpha

        # Compute fractional derivative term
        frac_term = 0.0
        for j in range(n):
            frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])

        # Euler formula
        t_n = t_values[n]
        y_next = y_values[n - 1] + (h**alpha_val / gamma(alpha_val + 1)) * (
            f(t_n, y_values[n - 1]) - frac_term
        )

        return y_next


class AdaptiveFractionalODESolver(FractionalODESolver):
    """
    Adaptive fractional ODE solver with error estimation and step size control.
    """

    def __init__(
        self,
        derivative_type: str = "caputo",
        method: str = "predictor_corrector",
        tol: float = 1e-6,
        max_iter: int = 1000,
        min_h: float = 1e-8,
        max_h: float = 1e-2,
        *,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_step: Optional[float] = None,
        min_step: Optional[float] = None,
        fractional_order: Optional[Union[float, FractionalOrder]] = None,
    ):
        """
        Initialize adaptive fractional ODE solver.

        Args:
            derivative_type: Type of fractional derivative
            method: Numerical method
            tol: Tolerance for error control
            max_iter: Maximum number of iterations
            min_h: Minimum step size
            max_h: Maximum step size
        """
        super().__init__(derivative_type, method, True, tol, max_iter)
        # Map aliases for step sizes if provided
        if max_step is not None:
            max_h = float(max_step)
        if min_step is not None:
            min_h = float(min_step)
        self.min_h = min_h
        self.max_h = max_h
        # Accept rtol/atol for compatibility; use the most stringent among tol/rtol/atol
        if rtol is not None:
            self.tol = min(self.tol, float(rtol))
        if atol is not None:
            self.tol = min(self.tol, float(atol))
        # Preserve fractional_order argument for compatibility
        self.fractional_order = fractional_order

    def solve(
        self,
        f: Callable,
        t_span: Tuple[float, float],
        y0: Union[float, np.ndarray],
        alpha: Union[float, FractionalOrder],
        h0: Optional[float] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:

        t0, tf = t_span
        if h0 is None:
            h0 = (tf - t0) / 100.0

        # ensure array
        y0 = np.atleast_1d(np.array(y0, dtype=float))

        t_values = [t0]
        y_values = [y0.copy()]

        t = float(t0)
        y = y0.copy()
        h = float(h0)

        # Controller parameters (robust defaults)
        safety = 0.9          # safety factor on step adaptation
        max_growth = 5.0      # limit how fast h can grow
        min_shrink = 0.2      # limit how fast h can shrink
        p_acc = 0.5           # exponent when accepting (typical for embedded order diff)
        p_rej = 0.25          # exponent when rejecting
        tiny = 10*np.finfo(float).eps

        # Global iteration cap as a last resort
        total_iters = 0
        max_total_iters = getattr(self, "max_total_iters", 10_0000)

        while t < tf - tiny:
            total_iters += 1
            if total_iters > max_total_iters:
                raise RuntimeError("Adaptive solver exceeded maximum total iterations; likely stalled.")

            # Enforce a floor on the step that depends on current magnitude of t
            h_floor = max(getattr(self, "min_h", 1e-12),
                        10*np.finfo(float).eps * max(1.0, abs(t)))

            # avoid negative or zero step
            h = max(h, h_floor)

            # Do not step beyond tf
            if t + h > tf:
                h = tf - t

            # If FP says no progress, force a slightly bigger step
            if t + h <= t:
                h = max(h_floor, (tf - t))  # snap to remaining interval if needed

            # Inner retry loop for the same target (PI controller pattern)
            retries = 0
            max_retries = getattr(self, "max_retries", 20)

            while True:
                t_next = t + h
                # Protect again against degeneracy
                if t_next <= t + tiny:
                    # If even advancing by tiny doesn’t move, we’re at FP limit; break
                    t_next = np.nextafter(t, tf)
                    h = t_next - t

                # Take a trial step
                y_trial = self._adaptive_step(f, t, t_next, y, alpha, h)

                # Compute a scalar error norm robustly
                err_val = self._estimate_error(f, t, t_next, y, y_trial, alpha, h)

                # Convert to finite scalar
                if isinstance(err_val, np.ndarray):
                    err = float(np.linalg.norm(err_val, ord=np.inf))
                else:
                    err = float(err_val)

                if not np.isfinite(err):
                    # Non-finite error: shrink hard and retry
                    h_new = max(min_shrink * h, h_floor)
                    if h_new == h or retries >= max_retries:
                        # Give up on error estimate; accept to avoid infinite loop
                        break
                    h = h_new
                    retries += 1
                    continue

                # Bound away from zero to avoid division overflow,
                # but treat exact zero as "perfect" and grow to the cap.
                if err == 0.0:
                    factor = max_growth
                else:
                    factor = safety * (self.tol / err) ** (p_acc if err <= self.tol else p_rej)
                    # clamp
                    factor = float(np.clip(factor, min_shrink, max_growth))

                if err <= self.tol:
                    # Accept the step
                    t = t_next
                    y = y_trial
                    t_values.append(t)
                    y_values.append(y.copy())
                    # Propose next h
                    h = max(h_floor, min(getattr(self, "max_h", np.inf), h * factor))
                    break
                else:
                    # Reject: shrink and retry (do not advance t)
                    h_new = max(h_floor, h * factor)
                    # If we can’t shrink further or we’ve retried too much, force accept to avoid stall
                    if (h_new >= h * 0.9999 and h <= h_floor * 1.0001) or retries >= max_retries:
                        # Accept with warning behaviour: take the step but increase a bit to escape
                        t = t_next
                        y = y_trial
                        t_values.append(t)
                        y_values.append(y.copy())
                        h = max(h_floor, min(getattr(self, "max_h", np.inf), h * 1.25))
                        break
                    h = h_new
                    retries += 1

        return np.array(t_values, dtype=float), np.vstack(y_values)


    def _adaptive_step(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Compute adaptive step.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            alpha: Fractional order
            h: Step size

        Returns:
            Solution at next time
        """
        if self.method == "predictor_corrector":
            return self._adaptive_predictor_corrector(
                f, t_current, t_next, y_current, alpha, h
            )
        elif self.method == "runge_kutta":
            return self._adaptive_runge_kutta(
                f, t_current, t_next, y_current, alpha, h)
        else:
            # Fall back to non-adaptive method
            return self._euler_step(
                f,
                np.array([t_current, t_next]),
                np.array([y_current, y_current]),
                1,
                alpha,
                self._compute_fractional_coefficients(alpha, 2),
                h,
            )

    def _adaptive_predictor_corrector(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Adaptive predictor-corrector step.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            alpha: Fractional order
            h: Step size

        Returns:
            Solution at next time
        """
        # Predictor step
        y_pred = y_current + (h**alpha / gamma(alpha + 1)
                              ) * f(t_current, y_current)

        # Corrector step with iteration
        y_corr = y_pred
        for _ in range(self.max_iter):
            y_old = y_corr.copy()
            y_corr = y_current + (h**alpha / gamma(alpha + 1)) * (
                0.5 * (f(t_next, y_pred) + f(t_current, y_current))
            )

            if np.allclose(y_corr, y_old, rtol=self.tol):
                break

        return y_corr

    def _adaptive_runge_kutta(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> np.ndarray:
        """
        Adaptive Runge-Kutta step.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            alpha: Fractional order
            h: Step size

        Returns:
            Solution at next time
        """
        # Runge-Kutta coefficients
        k1 = f(t_current, y_current)
        k2 = f(t_current + h / 2, y_current + h / 2 * k1)
        k3 = f(t_current + h / 2, y_current + h / 2 * k2)
        k4 = f(t_next, y_current + h * k3)

        # Runge-Kutta formula
        y_next = (
            y_current + (h**alpha / gamma(alpha + 1)) *
            (k1 + 2 * k2 + 2 * k3 + k4) / 6
        )

        return y_next

    def _estimate_error(
        self,
        f: Callable,
        t_current: float,
        t_next: float,
        y_current: np.ndarray,
        y_next: np.ndarray,
        alpha: Union[float, FractionalOrder],
        h: float,
    ) -> float:
        """
        Estimate local truncation error.

        Args:
            f: Right-hand side function
            t_current: Current time
            t_next: Next time
            y_current: Current solution
            y_next: Next solution
            alpha: Fractional order
            h: Step size

        Returns:
            Estimated error
        """
        # Use difference between predictor and corrector as error estimate
        y_pred = y_current + (h**alpha / gamma(alpha + 1)
                              ) * f(t_current, y_current)
        error = np.linalg.norm(y_next - y_pred)

        return error


# Convenience functions
def solve_fractional_ode(
    f: Callable,
    t_span: Tuple[float, float],
    y0: Union[float, np.ndarray],
    alpha: Union[float, FractionalOrder],
    derivative_type: str = "caputo",
    method: str = "predictor_corrector",
    adaptive: bool = True,
    h: Optional[float] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve fractional ODE.

    Args:
        f: Right-hand side function f(t, y)
        t_span: Time interval (t0, tf)
        y0: Initial condition(s)
        alpha: Fractional order
        derivative_type: Type of fractional derivative
        method: Numerical method
        adaptive: Use adaptive step size control
        h: Step size (None for adaptive)
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, y_values)
    """
    if adaptive:
        solver = AdaptiveFractionalODESolver(derivative_type, method)
    else:
        solver = FractionalODESolver(derivative_type, method, adaptive)

    return solver.solve(f, t_span, y0, alpha, h, **kwargs)


def solve_fractional_system(
    f: Callable,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    alpha: Union[float, np.ndarray],
    derivative_type: str = "caputo",
    method: str = "predictor_corrector",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve system of fractional ODEs.

    Args:
        f: Right-hand side function f(t, y)
        t_span: Time interval (t0, tf)
        y0: Initial conditions
        alpha: Fractional orders (scalar or array)
        derivative_type: Type of fractional derivative
        method: Numerical method
        **kwargs: Additional solver parameters

    Returns:
        Tuple of (t_values, y_values)
    """
    # For now, use the same solver for systems
    # In practice, you might want specialized system solvers
    return solve_fractional_ode(
        f,
        t_span,
        y0,
        alpha,
        derivative_type,
        method,
        **kwargs)
