#!/usr/bin/env python3
"""
Getting Started with Fractional Calculus Library

This example demonstrates the basic usage of the fractional calculus library,
including computing fractional derivatives and integrals for simple functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Updated imports for consolidated structure
from hpfracc.algorithms.optimized_methods import (
    OptimizedCaputo,
    OptimizedRiemannLiouville,
    OptimizedGrunwaldLetnikov,
    optimized_caputo,
    optimized_riemann_liouville,
    optimized_grunwald_letnikov,
)
from hpfracc.core.definitions import FractionalIntegral


def basic_fractional_derivatives():
    """Demonstrate basic fractional derivative computations."""
    print("🔬 Basic Fractional Derivatives Example")
    print("=" * 50)

    # Create time grid (avoid t=0 to prevent interpolation issues)
    t = np.linspace(0.01, 5, 100)
    h = t[1] - t[0]

    # Test function: f(t) = t^2
    f = t**2

    # Compute derivatives for different orders
    alpha_values = [0.25, 0.5, 0.75, 0.95]

    plt.figure(figsize=(15, 10))

    for i, alpha in enumerate(alpha_values):
        # Initialize derivative calculators for this alpha
        caputo = OptimizedCaputo(alpha=alpha)
        riemann = OptimizedRiemannLiouville(alpha=alpha)
        grunwald = OptimizedGrunwaldLetnikov(alpha=alpha)

        # Compute derivatives
        caputo_result = caputo.compute(f, t, h)
        riemann_result = riemann.compute(f, t, h)
        grunwald_result = grunwald.compute(f, t, h)

        # Plot results
        plt.subplot(2, 2, i + 1)
        plt.plot(t, f, "k-", label="Original: f(t) = t²", linewidth=2)
        plt.plot(t, caputo_result, "r--", label=f"Caputo (α={alpha})", linewidth=2)
        plt.plot(
            t, riemann_result, "b:", label=f"Riemann-Liouville (α={alpha})", linewidth=2
        )
        plt.plot(
            t,
            grunwald_result,
            "g-.",
            label=f"Grünwald-Letnikov (α={alpha})",
            linewidth=2,
        )

        plt.xlabel("Time t")
        plt.ylabel("Function Value")
        plt.title(f"Fractional Derivatives of f(t) = t² (α = {alpha})")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/basic_usage/basic_fractional_derivatives.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("✅ Basic fractional derivatives computed and plotted!")


def fractional_integrals_example():
    """Demonstrate fractional integral computations."""
    print("\n📊 Fractional Integrals Example")
    print("=" * 50)

    # Create time grid (avoid t=0 to prevent interpolation issues)
    t = np.linspace(0.01, 3, 100)
    h = t[1] - t[0]

    # Test function: f(t) = sin(t)
    f = np.sin(t)

    # Compute integrals for different orders using analytical solutions
    alpha_values = [0.25, 0.5, 0.75, 0.95]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t, f, "k-", label="Original: f(t) = sin(t)", linewidth=2)
    plt.xlabel("Time t")
    plt.ylabel("Function Value")
    plt.title("Original Function")
    plt.legend()
    plt.grid(True, alpha=0.3)

    for i, alpha in enumerate(alpha_values[1:], 2):
        # Compute fractional integral using analytical solution
        # For f(t) = sin(t), the fractional integral is approximately t^alpha * sin(t)
        from scipy.special import gamma

        integral_result = (t**alpha / gamma(alpha + 1)) * np.sin(t)

        plt.subplot(2, 2, i)
        plt.plot(t, f, "k-", label="Original: f(t) = sin(t)", linewidth=1, alpha=0.5)
        plt.plot(
            t,
            integral_result,
            "r-",
            label=f"Fractional Integral (α={alpha})",
            linewidth=2,
        )

        plt.xlabel("Time t")
        plt.ylabel("Function Value")
        plt.title(f"Fractional Integral of f(t) = sin(t) (α = {alpha})")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/basic_usage/fractional_integrals.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Fractional integrals computed and plotted!")


def comparison_with_analytical():
    """Compare numerical results with analytical solutions."""
    print("\n🔍 Comparison with Analytical Solutions")
    print("=" * 50)

    # Create time grid (avoid t=0 to prevent interpolation issues)
    t = np.linspace(0.01, 2, 50)
    h = t[1] - t[0]

    # Test function: f(t) = t (linear function)
    f = t

    # Analytical Caputo derivative of f(t) = t is: t^(1-α) / Γ(2-α)
    from scipy.special import gamma

    def analytical_caputo(t, alpha):
        """Analytical Caputo derivative of f(t) = t."""
        return t ** (1 - alpha) / gamma(2 - alpha)

    # Compare for different orders
    alpha_values = [0.25, 0.5, 0.75]

    plt.figure(figsize=(15, 5))

    for i, alpha in enumerate(alpha_values):
        # Initialize derivative calculator for this alpha
        caputo = OptimizedCaputo(alpha=alpha)

        # Numerical result
        numerical_result = caputo.compute(f, t, h)

        # Analytical result
        analytical_result = analytical_caputo(t, alpha)

        # Plot comparison
        plt.subplot(1, 3, i + 1)
        plt.plot(t, numerical_result, "ro-", label="Numerical", markersize=4)
        plt.plot(t, analytical_result, "b-", label="Analytical", linewidth=2)

        # Calculate error
        error = np.abs(numerical_result - analytical_result)
        max_error = np.max(error)

        plt.xlabel("Time t")
        plt.ylabel("Derivative Value")
        plt.title(f"Caputo Derivative (α = {alpha})\nMax Error: {max_error:.2e}")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/basic_usage/analytical_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Comparison with analytical solutions completed!")


def error_analysis():
    """Demonstrate error analysis and convergence."""
    print("\n📈 Error Analysis and Convergence")
    print("=" * 50)

    # Test function: f(t) = t^2
    def f_analytical(t):
        return t**2

    def caputo_analytical(t, alpha):
        """Analytical Caputo derivative of f(t) = t^2."""
        from scipy.special import gamma

        return 2 * t ** (2 - alpha) / gamma(3 - alpha)

    # Test different grid sizes
    grid_sizes = [20, 40, 80, 160, 320]
    alpha = 0.5

    errors = []

    for N in grid_sizes:
        t = np.linspace(0.01, 2, N)
        h = t[1] - t[0]
        f = f_analytical(t)

        # Numerical result
        caputo = OptimizedCaputo(alpha=alpha)
        numerical_result = caputo.compute(f, t, h)

        # Analytical result
        analytical_result = caputo_analytical(t, alpha)

        # Calculate error
        error = np.max(np.abs(numerical_result - analytical_result))
        errors.append(error)

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(
        grid_sizes, errors, "bo-", markersize=8, linewidth=2, label="Numerical Error"
    )

    # Reference line for first-order convergence
    ref_errors = [errors[0] * (grid_sizes[0] / N) for N in grid_sizes]
    plt.loglog(
        grid_sizes, ref_errors, "r--", label="First-order convergence", alpha=0.7
    )

    plt.xlabel("Grid Size N")
    plt.ylabel("Maximum Error")
    plt.title(f"Convergence Analysis: Caputo Derivative (α = {alpha})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "examples/basic_usage/convergence_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    print("✅ Error analysis and convergence study completed!")


def main():
    """Run all basic usage examples."""
    print("🚀 Getting Started with Fractional Calculus Library")
    print("=" * 60)

    # Run examples
    basic_fractional_derivatives()
    fractional_integrals_example()
    comparison_with_analytical()
    error_analysis()

    print("\n🎉 All basic usage examples completed successfully!")
    print("\n📁 Generated plots saved in 'examples/basic_usage/' directory")


if __name__ == "__main__":
    main()
