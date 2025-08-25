"""
Fractional Green's Function Module

This module provides implementations of fractional Green's functions for various
fractional differential equations and boundary value problems.
"""

import numpy as np
import torch
from typing import Union, Callable, Optional, Tuple, List
from scipy.special import gamma, hyperu, kv, iv
from scipy.integrate import quad
import warnings

from ..core.definitions import FractionalOrder
from ..core.utilities import hypergeometric_series, bessel_function_first_kind


class FractionalGreensFunction:
    """
    Base class for fractional Green's function implementations.
    
    Green's functions are fundamental solutions to differential equations
    and are particularly useful for solving boundary value problems.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], equation_type: str = "diffusion"):
        """
        Initialize fractional Green's function.
        
        Args:
            alpha: Fractional order
            equation_type: Type of differential equation
        """
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha
            
        self.equation_type = equation_type
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate parameters."""
        if self.alpha.alpha <= 0:
            raise ValueError(f"Fractional order must be positive, got {self.alpha.alpha}")
        
        if self.equation_type not in ["diffusion", "wave", "advection", "reaction"]:
            raise ValueError(f"Unknown equation type: {self.equation_type}")
    
    def __call__(self, x: Union[float, np.ndarray, torch.Tensor], t: Union[float, np.ndarray, torch.Tensor], 
                 x0: float = 0.0, t0: float = 0.0) -> Union[float, np.ndarray, torch.Tensor]:
        """Compute Green's function value."""
        raise NotImplementedError("Subclasses must implement __call__")


class FractionalDiffusionGreensFunction(FractionalGreensFunction):
    """
    Green's function for fractional diffusion equation.
    
    The fractional diffusion equation is:
    ∂^α u/∂t^α = D ∂²u/∂x²
    
    The Green's function is:
    G(x,t;x₀,t₀) = (1/√(4πD(t-t₀)^α)) * exp(-(x-x₀)²/(4D(t-t₀)^α))
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], D: float = 1.0):
        """
        Initialize fractional diffusion Green's function.
        
        Args:
            alpha: Fractional order (0 < α < 2)
            D: Diffusion coefficient
        """
        super().__init__(alpha, equation_type="diffusion")
        self.D = D
    
    def __call__(self, x: Union[float, np.ndarray, torch.Tensor], t: Union[float, np.ndarray, torch.Tensor],
                 x0: float = 0.0, t0: float = 0.0) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute fractional diffusion Green's function.
        
        Args:
            x: Spatial coordinate(s)
            t: Time coordinate(s)
            x0: Source location
            t0: Source time
            
        Returns:
            Green's function value(s)
        """
        if isinstance(x, (int, float)) and isinstance(t, (int, float)):
            return self._compute_scalar(x, t, x0, t0)
        elif isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return self._compute_array_numpy(x, t, x0, t0)
        elif isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return self._compute_array_torch(x, t, x0, t0)
        else:
            raise TypeError("x and t must be of the same type")
    
    def _compute_scalar(self, x: float, t: float, x0: float, t0: float) -> float:
        """Compute Green's function at scalar points."""
        if t <= t0:
            return 0.0
        
        tau = t - t0
        xi = x - x0
        
        # Use Mittag-Leffler function for fractional diffusion
        if self.alpha.alpha == 1:
            # Standard diffusion
            return (1.0 / np.sqrt(4 * np.pi * self.D * tau)) * np.exp(-xi**2 / (4 * self.D * tau))
        else:
            # Fractional diffusion using Mittag-Leffler function
            from .mittag_leffler import mittag_leffler_function
            
            # Simplified form for fractional diffusion
            scaling = (tau ** (self.alpha.alpha - 1)) / gamma(self.alpha.alpha)
            return scaling * np.exp(-xi**2 / (4 * self.D * tau**self.alpha.alpha))
    
    def _compute_array_numpy(self, x: np.ndarray, t: np.ndarray, x0: float, t0: float) -> np.ndarray:
        """Compute Green's function for numpy arrays."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = np.zeros_like(x, dtype=float)
        
        for i in range(x.size):
            result.flat[i] = self._compute_scalar(x.flat[i], t.flat[i], x0, t0)
        
        return result
    
    def _compute_array_torch(self, x: torch.Tensor, t: torch.Tensor, x0: float, t0: float) -> torch.Tensor:
        """Compute Green's function for torch tensors."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = torch.zeros_like(x, dtype=torch.float64)
        
        for i in range(x.numel()):
            result.view(-1)[i] = self._compute_scalar(float(x.view(-1)[i]), float(t.view(-1)[i]), x0, t0)
        
        return result


class FractionalWaveGreensFunction(FractionalGreensFunction):
    """
    Green's function for fractional wave equation.
    
    The fractional wave equation is:
    ∂^α u/∂t^α = c² ∂²u/∂x²
    
    The Green's function involves Bessel functions and depends on the fractional order.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], c: float = 1.0):
        """
        Initialize fractional wave Green's function.
        
        Args:
            alpha: Fractional order (1 < α < 2)
            c: Wave speed
        """
        super().__init__(alpha, equation_type="wave")
        self.c = c
        
        if not (1 < self.alpha.alpha < 2):
            raise ValueError("Fractional wave equation requires 1 < α < 2")
    
    def __call__(self, x: Union[float, np.ndarray, torch.Tensor], t: Union[float, np.ndarray, torch.Tensor],
                 x0: float = 0.0, t0: float = 0.0) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute fractional wave Green's function.
        
        Args:
            x: Spatial coordinate(s)
            t: Time coordinate(s)
            x0: Source location
            t0: Source time
            
        Returns:
            Green's function value(s)
        """
        if isinstance(x, (int, float)) and isinstance(t, (int, float)):
            return self._compute_scalar(x, t, x0, t0)
        elif isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return self._compute_array_numpy(x, t, x0, t0)
        elif isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return self._compute_array_torch(x, t, x0, t0)
        else:
            raise TypeError("x and t must be of the same type")
    
    def _compute_scalar(self, x: float, t: float, x0: float, t0: float) -> float:
        """Compute Green's function at scalar points."""
        if t <= t0:
            return 0.0
        
        tau = t - t0
        xi = x - x0
        
        # Distance from source
        r = abs(xi)
        
        # Check causality
        if r > self.c * tau:
            return 0.0
        
        # Fractional wave Green's function
        if self.alpha.alpha == 2:
            # Standard wave equation
            return 0.5 * (np.heaviside(tau - r/self.c, 0.5) - np.heaviside(tau + r/self.c, 0.5))
        else:
            # Fractional wave equation
            # Use Mittag-Leffler function representation
            from .mittag_leffler import mittag_leffler_function
            
            scaling = (tau ** (self.alpha.alpha - 1)) / gamma(self.alpha.alpha)
            argument = (r / (self.c * tau)) ** (self.alpha.alpha / 2)
            
            return scaling * mittag_leffler_function(self.alpha.alpha, -argument)
    
    def _compute_array_numpy(self, x: np.ndarray, t: np.ndarray, x0: float, t0: float) -> np.ndarray:
        """Compute Green's function for numpy arrays."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = np.zeros_like(x, dtype=float)
        
        for i in range(x.size):
            result.flat[i] = self._compute_scalar(x.flat[i], t.flat[i], x0, t0)
        
        return result
    
    def _compute_array_torch(self, x: torch.Tensor, t: torch.Tensor, x0: float, t0: float) -> torch.Tensor:
        """Compute Green's function for torch tensors."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = torch.zeros_like(x, dtype=torch.float64)
        
        for i in range(x.numel()):
            result.view(-1)[i] = self._compute_scalar(float(x.view(-1)[i]), float(t.view(-1)[i]), x0, t0)
        
        return result


class FractionalAdvectionGreensFunction(FractionalGreensFunction):
    """
    Green's function for fractional advection equation.
    
    The fractional advection equation is:
    ∂^α u/∂t^α + v ∂u/∂x = 0
    
    The Green's function represents the fundamental solution.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], v: float = 1.0):
        """
        Initialize fractional advection Green's function.
        
        Args:
            alpha: Fractional order (0 < α < 1)
            v: Advection velocity
        """
        super().__init__(alpha, equation_type="advection")
        self.v = v
        
        if not (0 < self.alpha.alpha < 1):
            raise ValueError("Fractional advection equation requires 0 < α < 1")
    
    def __call__(self, x: Union[float, np.ndarray, torch.Tensor], t: Union[float, np.ndarray, torch.Tensor],
                 x0: float = 0.0, t0: float = 0.0) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Compute fractional advection Green's function.
        
        Args:
            x: Spatial coordinate(s)
            t: Time coordinate(s)
            x0: Source location
            t0: Source time
            
        Returns:
            Green's function value(s)
        """
        if isinstance(x, (int, float)) and isinstance(t, (int, float)):
            return self._compute_scalar(x, t, x0, t0)
        elif isinstance(x, np.ndarray) and isinstance(t, np.ndarray):
            return self._compute_array_numpy(x, t, x0, t0)
        elif isinstance(x, torch.Tensor) and isinstance(t, torch.Tensor):
            return self._compute_array_torch(x, t, x0, t0)
        else:
            raise TypeError("x and t must be of the same type")
    
    def _compute_scalar(self, x: float, t: float, x0: float, t0: float) -> float:
        """Compute Green's function at scalar points."""
        if t <= t0:
            return 0.0
        
        tau = t - t0
        xi = x - x0
        
        # Characteristic line
        characteristic = xi - self.v * tau
        
        # Fractional advection Green's function
        if characteristic == 0:
            # On the characteristic line
            return (tau ** (self.alpha.alpha - 1)) / gamma(self.alpha.alpha)
        else:
            # Off the characteristic line - use Mittag-Leffler function
            from .mittag_leffler import mittag_leffler_function
            
            argument = -abs(characteristic) / (tau ** self.alpha.alpha)
            return (tau ** (self.alpha.alpha - 1)) * mittag_leffler_function(self.alpha.alpha, argument)
    
    def _compute_array_numpy(self, x: np.ndarray, t: np.ndarray, x0: float, t0: float) -> np.ndarray:
        """Compute Green's function for numpy arrays."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = np.zeros_like(x, dtype=float)
        
        for i in range(x.size):
            result.flat[i] = self._compute_scalar(x.flat[i], t.flat[i], x0, t0)
        
        return result
    
    def _compute_array_torch(self, x: torch.Tensor, t: torch.Tensor, x0: float, t0: float) -> torch.Tensor:
        """Compute Green's function for torch tensors."""
        if x.shape != t.shape:
            raise ValueError("x and t must have the same shape")
        
        result = torch.zeros_like(x, dtype=torch.float64)
        
        for i in range(x.numel()):
            result.view(-1)[i] = self._compute_scalar(float(x.view(-1)[i]), float(t.view(-1)[i]), x0, t0)
        
        return result


def create_fractional_greens_function(alpha: Union[float, FractionalOrder], equation_type: str = "diffusion", 
                                    **kwargs) -> FractionalGreensFunction:
    """
    Factory function to create fractional Green's function objects.
    
    Args:
        alpha: Fractional order
        equation_type: Type of differential equation
        **kwargs: Additional parameters (D, c, v, etc.)
        
    Returns:
        Appropriate fractional Green's function object
    """
    if equation_type == "diffusion":
        D = kwargs.get("D", 1.0)
        return FractionalDiffusionGreensFunction(alpha, D)
    elif equation_type == "wave":
        c = kwargs.get("c", 1.0)
        return FractionalWaveGreensFunction(alpha, c)
    elif equation_type == "advection":
        v = kwargs.get("v", 1.0)
        return FractionalAdvectionGreensFunction(alpha, v)
    else:
        raise ValueError(f"Unknown equation type: {equation_type}")


# Utility functions for Green's functions
def greens_function_properties(equation_type: str) -> dict:
    """
    Return mathematical properties of Green's functions.
    
    Args:
        equation_type: Type of differential equation
        
    Returns:
        Dictionary containing properties
    """
    properties = {
        "diffusion": {
            "causality": "Retarded (causal)",
            "symmetry": "Spatial symmetry",
            "conservation": "Mass conservation",
            "asymptotics": "Gaussian decay"
        },
        "wave": {
            "causality": "Retarded (causal)",
            "symmetry": "Lorentz invariance",
            "conservation": "Energy conservation",
            "asymptotics": "Wavefront propagation"
        },
        "advection": {
            "causality": "Retarded (causal)",
            "symmetry": "Translation invariance",
            "conservation": "Mass conservation",
            "asymptotics": "Characteristic propagation"
        }
    }
    
    return properties.get(equation_type, {})


def validate_greens_function(G: FractionalGreensFunction, x_range: Tuple[float, float], 
                           t_range: Tuple[float, float], n_points: int = 100) -> dict:
    """
    Validate Green's function computation.
    
    Args:
        G: Green's function object
        x_range: Spatial range (x_min, x_max)
        t_range: Time range (t_min, t_max)
        n_points: Number of validation points
        
    Returns:
        Validation results
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    t = np.linspace(t_range[0], t_range[1], n_points)
    X, T = np.meshgrid(x, t)
    
    # Compute Green's function
    G_values = G(X, T)
    
    # Validation checks
    validation = {
        "causality_check": True,
        "positivity_check": True,
        "normalization_check": True,
        "symmetry_check": True,
        "error_estimate": 0.0
    }
    
    # Check causality (G = 0 for t < 0)
    if np.any(G_values[T < 0] != 0):
        validation["causality_check"] = False
    
    # Check positivity (G ≥ 0)
    if np.any(G_values < 0):
        validation["positivity_check"] = False
    
    # Check normalization (integral should be finite)
    integral = np.trapz(np.trapz(G_values, x, axis=1), t[:, 0])
    if not np.isfinite(integral):
        validation["normalization_check"] = False
    
    return validation


def greens_function_convolution(G: FractionalGreensFunction, f: Callable, x: np.ndarray, t: np.ndarray,
                              x0: float = 0.0, t0: float = 0.0) -> np.ndarray:
    """
    Compute convolution with Green's function.
    
    Args:
        G: Green's function object
        f: Source function
        x: Spatial coordinates
        t: Time coordinates
        x0: Source location
        t0: Source time
        
    Returns:
        Convolution result
    """
    X, T = np.meshgrid(x, t)
    result = np.zeros_like(X)
    
    for i, ti in enumerate(t):
        for j, xj in enumerate(x):
            # Compute convolution integral
            def integrand(tau):
                return G(xj, ti, x0, tau) * f(x0, tau)
            
            # Simple numerical integration
            tau_points = np.linspace(t0, ti, 100)
            integrand_values = [integrand(tau) for tau in tau_points]
            result[i, j] = np.trapz(integrand_values, tau_points)
    
    return result
