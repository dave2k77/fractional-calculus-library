"""
Convergence tests for fractional calculus numerical methods.

This module provides tools for analyzing the convergence rates
of numerical methods for fractional derivatives.
"""

import numpy as np
from typing import Callable, Dict, List
import warnings
from enum import Enum


class OrderOfAccuracy(Enum):
    """Enumeration for expected orders of accuracy."""

    FIRST_ORDER = 1.0
    SECOND_ORDER = 2.0
    THIRD_ORDER = 3.0
    FOURTH_ORDER = 4.0


class ConvergenceTester:
    """Tester for analyzing convergence of numerical methods."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the convergence tester.

        Args:
            tolerance: Numerical tolerance for convergence calculations
        """
        self.tolerance = tolerance

    def test_convergence(
        self,
        method_func: Callable,
        analytical_func: Callable,
        grid_sizes: List[int],
        test_params: Dict,
        error_norm: str = "l2",
    ) -> Dict:
        """
        Test convergence of a numerical method.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            grid_sizes: List of grid sizes to test
            test_params: Parameters for the test case
            error_norm: Error norm to use ('l1', 'l2', 'linf')

        Returns:
            Convergence test results
        """
        from ..utils.error_analysis import ErrorAnalyzer

        error_analyzer = ErrorAnalyzer(tolerance=self.tolerance)
        errors = []

        for N in grid_sizes:
            try:
                # Create grid
                x = np.linspace(0, 1, N)

                # Compute numerical solution
                numerical = method_func(x, **test_params)

                # Compute analytical solution
                analytical = analytical_func(x, **test_params)

                # Compute error
                if error_norm == "l1":
                    error = error_analyzer.l1_error(numerical, analytical)
                elif error_norm == "l2":
                    error = error_analyzer.l2_error(numerical, analytical)
                elif error_norm == "linf":
                    error = error_analyzer.linf_error(numerical, analytical)
                else:
                    raise ValueError(f"Unknown error norm: {error_norm}")

                errors.append(error)

            except Exception as e:
                warnings.warn(f"Failed to compute error for N={N}: {e}")
                errors.append(np.nan)

        # Remove any NaN values
        valid_indices = [i for i, e in enumerate(errors) if not np.isnan(e)]
        if len(valid_indices) < 2:
            raise ValueError(
                "Need at least 2 valid error measurements for convergence analysis"
            )

        valid_grid_sizes = [grid_sizes[i] for i in valid_indices]
        valid_errors = [errors[i] for i in valid_indices]

        # Compute convergence rate
        convergence_rate = self._compute_convergence_rate(
            valid_grid_sizes, valid_errors
        )

        return {
            "grid_sizes": valid_grid_sizes,
            "errors": valid_errors,
            "convergence_rate": convergence_rate,
            "error_norm": error_norm,
            "test_params": test_params,
            "all_grid_sizes": grid_sizes,
            "all_errors": errors,
        }

    def _compute_convergence_rate(
        self, grid_sizes: List[int], errors: List[float]
    ) -> float:
        """
        Compute convergence rate using linear regression on log-log plot.

        Args:
            grid_sizes: List of grid sizes
            errors: List of corresponding errors

        Returns:
            Convergence rate (order of accuracy)
        """
        if len(grid_sizes) < 2 or len(errors) < 2:
            raise ValueError(
                "Need at least 2 points to compute convergence rate")

        # Convert to log space
        log_n = np.log(np.array(grid_sizes))
        log_error = np.log(np.array(errors))

        # Linear regression: log(error) = -p * log(N) + C
        # where p is the convergence rate
        coeffs = np.polyfit(log_n, log_error, 1)
        # Negative because error decreases with N
        convergence_rate = -coeffs[0]

        return convergence_rate

    def test_multiple_norms(
        self,
        method_func: Callable,
        analytical_func: Callable,
        grid_sizes: List[int],
        test_params: Dict,
    ) -> Dict:
        """
        Test convergence using multiple error norms.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            grid_sizes: List of grid sizes to test
            test_params: Parameters for the test case

        Returns:
            Convergence test results for multiple norms
        """
        norms = ["l1", "l2", "linf"]
        results = {}

        for norm in norms:
            try:
                results[norm] = self.test_convergence(
                    method_func, analytical_func, grid_sizes, test_params, norm
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to test convergence for norm {norm}: {e}")
                results[norm] = None

        return results


class ConvergenceAnalyzer:
    """Analyzer for comprehensive convergence studies."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the convergence analyzer.

        Args:
            tolerance: Numerical tolerance for convergence calculations
        """
        self.tolerance = tolerance
        self.tester = ConvergenceTester(tolerance)

    def analyze_method_convergence(
        self,
        method_func: Callable,
        analytical_func: Callable,
        test_cases: List[Dict],
        grid_sizes: List[int] = None,
    ) -> Dict:
        """
        Analyze convergence for multiple test cases.

        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            test_cases: List of test case dictionaries
            grid_sizes: List of grid sizes to test (default: [50, 100, 200, 400])

        Returns:
            Comprehensive convergence analysis
        """
        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 400]

        results = {
            "test_cases": [],
            "summary": {},
            "grid_sizes": grid_sizes,
        }

        convergence_rates = []

        for i, test_case in enumerate(test_cases):
            try:
                # Test convergence for this case
                case_result = self.tester.test_multiple_norms(
                    method_func, analytical_func, grid_sizes, test_case
                )

                # Extract convergence rates
                rates = {}
                for norm, result in case_result.items():
                    if result is not None:
                        rates[norm] = result["convergence_rate"]
                    else:
                        rates[norm] = np.nan

                case_summary = {
                    "case_index": i,
                    "test_params": test_case,
                    "convergence_rates": rates,
                    "success": True,
                }

                # Add to overall rates
                for norm, rate in rates.items():
                    if not np.isnan(rate):
                        convergence_rates.append(rate)

            except Exception as e:
                case_summary = {
                    "case_index": i,
                    "test_params": test_case,
                    "convergence_rates": {},
                    "success": False,
                    "error": str(e),
                }

            results["test_cases"].append(case_summary)

        # Compute summary statistics
        if convergence_rates:
            results["summary"] = {
                "mean_convergence_rate": np.mean(convergence_rates),
                "std_convergence_rate": np.std(convergence_rates),
                "min_convergence_rate": np.min(convergence_rates),
                "max_convergence_rate": np.max(convergence_rates),
                "total_test_cases": len(test_cases),
                "successful_test_cases": len(
                    [c for c in results["test_cases"] if c["success"]]
                ),
            }
        else:
            results["summary"] = {
                "mean_convergence_rate": np.nan,
                "std_convergence_rate": np.nan,
                "min_convergence_rate": np.nan,
                "max_convergence_rate": np.nan,
                "total_test_cases": len(test_cases),
                "successful_test_cases": 0,
            }

        return results

    def estimate_optimal_grid_size(
        self,
        target_error: float,
        convergence_rate: float,
        reference_grid_size: int,
        reference_error: float,
    ) -> int:
        """
        Estimate optimal grid size for a target error.

        Args:
            target_error: Target error tolerance
            convergence_rate: Estimated convergence rate
            reference_grid_size: Reference grid size
            reference_error: Reference error

        Returns:
            Estimated optimal grid size
        """
        if convergence_rate <= 0:
            raise ValueError("Convergence rate must be positive")

        if target_error <= 0:
            raise ValueError("Target error must be positive")

        # Use the relationship: error ∝ N^(-convergence_rate)
        # N_opt = N_ref * (error_ref / target_error)^(1/convergence_rate)
        N_opt = int(
            reference_grid_size
            * (reference_error / target_error) ** (1 / convergence_rate)
        )

        return max(N_opt, 1)  # Ensure positive grid size

    def validate_convergence_order(
        self,
        observed_rate: float,
        expected_order: OrderOfAccuracy,
        tolerance: float = 0.5,
    ) -> Dict:
        """
        Validate if observed convergence rate matches expected order.

        Args:
            observed_rate: Observed convergence rate
            expected_order: Expected order of accuracy
            tolerance: Tolerance for validation

        Returns:
            Validation result
        """
        expected_rate = expected_order.value
        difference = abs(observed_rate - expected_rate)

        is_valid = difference <= tolerance

        return {
            "observed_rate": observed_rate,
            "expected_rate": expected_rate,
            "difference": difference,
            "tolerance": tolerance,
            "is_valid": is_valid,
            "order_achieved": OrderOfAccuracy(expected_rate) if is_valid else None,
        }


def run_convergence_study(
    method_func: Callable,
    analytical_func: Callable,
    test_cases: List[Dict],
    grid_sizes: List[int] = None,
) -> Dict:
    """
    Run a comprehensive convergence study.

    Args:
        method_func: Function that computes numerical solution
        analytical_func: Function that computes analytical solution
        test_cases: List of test case dictionaries
        grid_sizes: List of grid sizes to test

    Returns:
        Convergence study results
    """
    analyzer = ConvergenceAnalyzer()
    return analyzer.analyze_method_convergence(
        method_func, analytical_func, test_cases, grid_sizes
    )


def run_method_convergence_test(
    method_func: Callable,
    analytical_func: Callable,
    grid_sizes: List[int],
    test_params: Dict,
) -> Dict:
    """
    Test convergence of a numerical method.

    Args:
        method_func: Function that computes numerical solution
        analytical_func: Function that computes analytical solution
        grid_sizes: List of grid sizes to test
        test_params: Parameters for the test case

    Returns:
        Convergence test results
    """
    tester = ConvergenceTester()
    return tester.test_multiple_norms(
        method_func, analytical_func, grid_sizes, test_params
    )


def estimate_convergence_rate(
        grid_sizes: List[int],
        errors: List[float]) -> float:
    """
    Estimate convergence rate from grid sizes and errors.

    Args:
        grid_sizes: List of grid sizes
        errors: List of corresponding errors

    Returns:
        Estimated convergence rate
    """
    if len(grid_sizes) < 2 or len(errors) < 2:
        raise ValueError("Need at least 2 points to estimate convergence rate")

    # Convert to log space
    log_n = np.log(np.array(grid_sizes))
    log_error = np.log(np.array(errors))

    # Linear regression
    coeffs = np.polyfit(log_n, log_error, 1)
    convergence_rate = -coeffs[0]

    return convergence_rate
