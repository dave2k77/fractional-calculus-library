"""
Mathematical definitions for fractional calculus.

This module provides the fundamental mathematical definitions and properties
of fractional derivatives and integrals, which form the theoretical foundation
for the numerical implementations.
"""

import numpy as np
from typing import Union, Dict, Any
from enum import Enum


class FractionalOrder:
    """
    Represents a fractional order with validation and properties.

    The fractional order α can be any real number, but typically
    we consider 0 < α < 2 for most applications.
    """

    def __init__(self,
                 alpha: Union[float, "FractionalOrder"],
                 validate: bool = True,
                 method: str = None):
        """
        Initialize fractional order.

        Args:
            alpha: Fractional order value
            validate: Whether to validate the order range
        """
        if isinstance(alpha, FractionalOrder):
            self.alpha = alpha.alpha
            # preserve method on copy if not explicitly provided
            self.method = method if method is not None else getattr(
                alpha, "method", None)
        else:
            self.alpha = float(alpha)
            self.method = method

        if validate:
            self._validate()

    def _validate(self):
        """Validate the fractional order."""
        if not np.isfinite(self.alpha):
            raise ValueError("Fractional order must be finite")

        if self.alpha < 0:
            raise ValueError("Fractional order must be non-negative")

    @property
    def value(self) -> float:
        """Get the fractional order value."""
        return self.alpha

    @property
    def is_integer(self) -> bool:
        """Check if the order is an integer."""
        return abs(self.alpha - round(self.alpha)) < 1e-10

    @property
    def is_fractional(self) -> bool:
        """Check if the order is fractional (not integer)."""
        return not self.is_integer

    @property
    def integer_part(self) -> int:
        """Get the integer part of the fractional order."""
        return int(np.floor(self.alpha))

    @property
    def fractional_part(self) -> float:
        """Get the fractional part of the fractional order."""
        return self.alpha - self.integer_part

    def __repr__(self) -> str:
        return f"FractionalOrder({self.alpha})"

    def __str__(self) -> str:
        return f"α = {self.alpha}"

    def __eq__(self, other) -> bool:
        if isinstance(other, FractionalOrder):
            return abs(self.alpha - other.alpha) < 1e-10
        return False  # Different types are not equal

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.alpha)


class DefinitionType(Enum):
    """Enumeration of fractional derivative definitions."""

    CAPUTO = "caputo"
    RIEMANN_LIOUVILLE = "riemann_liouville"
    GRUNWALD_LETNIKOV = "grunwald_letnikov"
    MILLER_ROSS = "miller_ross"
    WEYL = "weyl"
    MARCHAUD = "marchaud"


class FractionalDefinition:
    """
    Base class for fractional derivative definitions.

    This class provides the mathematical framework for different
    definitions of fractional derivatives and integrals.
    """

    def __init__(self, order: Union[float, FractionalOrder],
                 definition_type: DefinitionType):
        """
        Initialize fractional definition.

        Args:
            order: Fractional order
            definition_type: Type of fractional definition
        """
        self.order = FractionalOrder(order) if not isinstance(
            order, FractionalOrder) else order
        self.definition_type = definition_type
        # Keep alpha for backward compatibility
        self.alpha = self.order

    def get_definition_formula(self) -> str:
        """Get the mathematical formula for this definition."""
        formulas = {
            DefinitionType.CAPUTO: "D^α f(x) = (1/Γ(n-α)) ∫₀^x (x-t)^(n-α-1) f^(n)(t) dt",
            DefinitionType.RIEMANN_LIOUVILLE: "D^α f(x) = (1/Γ(n-α)) (d/dx)^n ∫₀^x (x-t)^(n-α-1) f(t) dt",
            DefinitionType.GRUNWALD_LETNIKOV: "D^α f(x) = lim_{h→0} h^(-α) Σ_{k=0}^∞ (-1)^k C(α,k) f(x-kh)",
            DefinitionType.MILLER_ROSS: "D^α f(x) = (1/Γ(1-α)) ∫₀^x (x-t)^(-α) f'(t) dt",
            DefinitionType.WEYL: "D^α f(x) = (1/Γ(n-α)) (d/dx)^n ∫_x^∞ (t-x)^(n-α-1) f(t) dt",
            DefinitionType.MARCHAUD: "D^α f(x) = (α/Γ(1-α)) ∫_0^∞ (f(x) - f(x-t)) / t^(α+1) dt",
        }
        return formulas.get(self.definition_type, "Formula not available")

    def get_properties(self) -> Dict[str, Any]:
        """Get mathematical properties of this definition."""
        properties = {
            DefinitionType.CAPUTO: {
                "linearity": True,
                "commutativity": False,
                "semigroup_property": False,
                "initial_conditions": "Standard",
                "applications": "Differential equations with standard initial conditions",
            },
            DefinitionType.RIEMANN_LIOUVILLE: {
                "linearity": True,
                "commutativity": False,
                "semigroup_property": True,
                "initial_conditions": "Fractional",
                "applications": "Theoretical analysis, semigroup properties",
            },
            DefinitionType.GRUNWALD_LETNIKOV: {
                "linearity": True,
                "commutativity": False,
                "semigroup_property": True,
                "initial_conditions": "Fractional",
                "applications": "Numerical methods, discrete approximations",
            },
        }
        return properties.get(self.definition_type, {})

    def __repr__(self) -> str:
        return f"{self.definition_type.value.title()}Definition(α={self.alpha})"


class CaputoDefinition(FractionalDefinition):
    """
    Caputo fractional derivative definition.

    The Caputo derivative is defined as:
    D^α f(x) = (1/Γ(n-α)) ∫₀^x (x-t)^(n-α-1) f^(n)(t) dt

    where n = ⌈α⌉ is the smallest integer greater than or equal to α.
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        super().__init__(order, DefinitionType.CAPUTO)

    @property
    def n(self) -> int:
        """Get the smallest integer n such that n-1 < α ≤ n."""
        return int(np.ceil(self.alpha.alpha))

    def get_advantages(self) -> list:
        """Get advantages of the Caputo definition."""
        return [
            "Standard initial conditions can be used",
            "Well-suited for differential equations",
            "Physical interpretation is clear",
            "Numerical implementation is straightforward",
        ]

    def get_limitations(self) -> list:
        """Get limitations of the Caputo definition."""
        return [
            "Does not satisfy the semigroup property",
            "Requires function to be n-times differentiable",
            "More complex than Riemann-Liouville for some applications",
        ]


class RiemannLiouvilleDefinition(FractionalDefinition):
    """
    Riemann-Liouville fractional derivative definition.

    The Riemann-Liouville derivative is defined as:
    D^α f(x) = (1/Γ(n-α)) (d/dx)^n ∫₀^x (x-t)^(n-α-1) f(t) dt

    where n = ⌈α⌉ is the smallest integer greater than or equal to α.
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        super().__init__(order, DefinitionType.RIEMANN_LIOUVILLE)

    @property
    def n(self) -> int:
        """Get the smallest integer n such that n-1 < α ≤ n."""
        return int(np.ceil(self.alpha.alpha))

    def get_advantages(self) -> list:
        """Get advantages of the Riemann-Liouville definition."""
        return [
            "Satisfies the semigroup property",
            "Mathematically elegant",
            "Well-established theoretical framework",
            "Suitable for theoretical analysis",
        ]

    def get_limitations(self) -> list:
        """Get limitations of the Riemann-Liouville definition."""
        return [
            "Requires fractional initial conditions",
            "Physical interpretation is less intuitive",
            "Numerical implementation can be complex",
        ]


class GrunwaldLetnikovDefinition(FractionalDefinition):
    """
    Grünwald-Letnikov fractional derivative definition.

    The Grünwald-Letnikov derivative is defined as:
    D^α f(x) = lim_{h→0} h^(-α) Σ_{k=0}^∞ (-1)^k C(α,k) f(x-kh)

    where C(α,k) are the generalized binomial coefficients.
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        super().__init__(order, DefinitionType.GRUNWALD_LETNIKOV)

    def get_advantages(self) -> list:
        """Get advantages of the Grünwald-Letnikov definition."""
        return [
            "Natural discrete approximation",
            "Satisfies the semigroup property",
            "Well-suited for numerical methods",
            "Direct extension of integer derivatives",
        ]

    def get_limitations(self) -> list:
        """Get limitations of the Grünwald-Letnikov definition."""
        return [
            "Requires function values at all previous points",
            "Memory requirements can be large",
            "Convergence can be slow for some functions",
        ]


class FractionalIntegral:
    """
    Fractional integral definition.

    The Riemann-Liouville fractional integral of order α > 0 is defined as:
    I^α f(x) = (1/Γ(α)) ∫₀^x (x-t)^(α-1) f(t) dt
    """

    def __init__(self, order: Union[float, FractionalOrder]):
        """
        Initialize fractional integral.

        Args:
            order: Fractional order (must be positive)
        """
        self.alpha = FractionalOrder(order)
        if self.alpha.alpha <= 0:
            raise ValueError("Fractional integral order must be positive")

    def get_formula(self) -> str:
        """Get the mathematical formula for the fractional integral."""
        return f"I^α f(x) = (1/Γ({self.alpha.alpha})) ∫₀^x (x-t)^({self.alpha.alpha}-1) f(t) dt"

    def get_properties(self) -> Dict[str, Any]:
        """Get mathematical properties of the fractional integral."""
        return {
            "linearity": True,
            "semigroup_property": True,
            "commutativity": True,
            "inverse_relationship": "I^α D^α f(x) = f(x) - Σ_{k=0}^{n-1} f^(k)(0) x^k/k!",
        }

    def __str__(self) -> str:
        """String representation of the fractional integral."""
        return f"FractionalIntegral(α={self.alpha.alpha})"


class FractionalCalculusProperties:
    """
    Mathematical properties and relationships in fractional calculus.
    """

    @staticmethod
    def linearity_property() -> str:
        """Get the linearity property."""
        return "D^α (af + bg) = a D^α f + b D^α g"

    @staticmethod
    def semigroup_property() -> str:
        """Get the semigroup property."""
        return "D^α D^β f = D^(α+β) f (for compatible definitions)"

    @staticmethod
    def leibniz_rule() -> str:
        """Get the generalized Leibniz rule."""
        return "D^α (fg) = Σ_{k=0}^∞ C(α,k) D^(α-k) f D^k g"

    @staticmethod
    def chain_rule() -> str:
        """Get the generalized chain rule."""
        return "D^α f(g(x)) = Σ_{k=0}^∞ C(α,k) D^(α-k) f(g(x)) D^k g(x)"

    @staticmethod
    def relationship_between_definitions() -> Dict[str, str]:
        """Get relationships between different definitions."""
        return {
            "Caputo_to_Riemann_Liouville": "D^α_C f = D^α_RL f - Σ_{k=0}^{n-1} f^(k)(0) x^(k-α)/Γ(k-α+1)",
            "Riemann_Liouville_to_Caputo": "D^α_RL f = D^α_C f + Σ_{k=0}^{n-1} f^(k)(0) x^(k-α)/Γ(k-α+1)",
            "Grünwald_Letnikov_equivalence": "D^α_GL f = D^α_RL f (under suitable conditions)",
        }

    @staticmethod
    def get_analytical_solutions() -> Dict[str, str]:
        """Get analytical solutions for common functions."""
        return {
            "power": "D^α x^n = Γ(n+1)/Γ(n-α+1) * x^(n-α)",
            "exponential": "D^α e^x = x^(-α)/Γ(1-α)",
            "constant": "D^α c = c * x^(-α)/Γ(1-α)",
            "linear": "D^α x = x^(1-α)/Γ(2-α)",
        }

    def get_properties(self) -> Dict[str, Any]:
        """Get all fractional calculus properties."""
        return {
            "definitions": self.get_definition_properties(),
            "integrals": self.get_integral_properties(),
            "relationships": self.relationship_between_definitions(),
            "analytical_solutions": self.get_analytical_solutions()
        }

    def get_definition_properties(self) -> Dict[str, str]:
        """Get properties specific to fractional derivative definitions."""
        return {
            "linearity": self.linearity_property(),
            "semigroup": self.semigroup_property(),
            "leibniz_rule": self.leibniz_rule(),
            "chain_rule": self.chain_rule(),
            "caputo": "D^α_C f = I^(n-α) f^(n)",
            "riemann_liouville": "D^α_RL f = d^n/dx^n I^(n-α) f",
            "grunwald_letnikov": "D^α_GL f = lim[h→0] (1/h^α) Σ[k=0]^∞ (-1)^k C(α,k) f(x-kh)"
        }

    def get_integral_properties(self) -> Dict[str, Any]:
        """Get properties specific to fractional integrals."""
        return {
            "linearity": True,
            "semigroup_property": True,
            "commutativity": True,
            "inverse_relationship": "I^α D^α f(x) = f(x) - Σ_{k=0}^{n-1} f^(k)(0) x^k/k!",
            "riemann_liouville": "I^α_RL f = (1/Γ(α)) ∫₀^x (x-t)^(α-1) f(t) dt",
            "caputo": "I^α_C f = (1/Γ(α)) ∫₀^x (x-t)^(α-1) f(t) dt"
        }


# Convenience functions
def create_definition(
    definition_type: Union[str, DefinitionType], alpha: Union[float, FractionalOrder]
) -> FractionalDefinition:
    """
    Create a fractional derivative definition.

    Args:
        definition_type: Type of definition
        alpha: Fractional order

    Returns:
        Fractional derivative definition object
    """
    if isinstance(definition_type, str):
        definition_type = DefinitionType(definition_type.lower())

    if definition_type == DefinitionType.CAPUTO:
        return CaputoDefinition(alpha)
    elif definition_type == DefinitionType.RIEMANN_LIOUVILLE:
        return RiemannLiouvilleDefinition(alpha)
    elif definition_type == DefinitionType.GRUNWALD_LETNIKOV:
        return GrunwaldLetnikovDefinition(alpha)
    else:
        raise ValueError(f"Unsupported definition type: {definition_type}")


def get_available_definitions() -> list:
    """Get list of available fractional derivative definitions."""
    return [dtype.value for dtype in DefinitionType]


def validate_fractional_order(
    alpha: float, min_value: float = 0.0, max_value: float = 2.0
) -> bool:
    """
    Validate fractional order within specified range.

    Args:
        alpha: Fractional order to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        True if valid, False otherwise
    """
    return min_value <= alpha <= max_value and np.isfinite(alpha)
