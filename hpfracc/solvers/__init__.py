"""
Solvers Module

This module provides comprehensive solvers for fractional differential equations,
including ODE solvers, PDE solvers, and advanced predictor-corrector methods.
"""

# Import ODE solvers
from .ode_solvers import (
    FractionalODESolver,
    AdaptiveFractionalODESolver,
    solve_fractional_ode,
    solve_fractional_system,
)

# Import PDE solvers
from .pde_solvers import (
    FractionalPDESolver,
    FractionalDiffusionSolver,
    FractionalAdvectionSolver,
    FractionalReactionDiffusionSolver,
    solve_fractional_diffusion,
    solve_fractional_advection,
    solve_fractional_reaction_diffusion,
)

# Import predictor-corrector methods
from .predictor_corrector import (
    PredictorCorrectorSolver,
    AdamsBashforthMoultonSolver,
    VariableStepPredictorCorrector,
    solve_predictor_corrector,
    solve_adams_bashforth_moulton,
    solve_variable_step_predictor_corrector,
)

# Import advanced solvers
from .advanced_solvers import (
    AdvancedFractionalODESolver,
    HighOrderFractionalSolver,
    ErrorControlMethod,
    AdaptiveMethod,
    solve_advanced_fractional_ode,
    solve_high_order_fractional_ode,
)

# Define what gets imported with "from solvers import *"
__all__ = [
    # ODE solvers
    "FractionalODESolver",
    "AdaptiveFractionalODESolver",
    "solve_fractional_ode",
    "solve_fractional_system",
    # PDE solvers
    "FractionalPDESolver",
    "FractionalDiffusionSolver",
    "FractionalAdvectionSolver",
    "FractionalReactionDiffusionSolver",
    "solve_fractional_diffusion",
    "solve_fractional_advection",
    "solve_fractional_reaction_diffusion",
    # Predictor-corrector methods
    "PredictorCorrectorSolver",
    "AdamsBashforthMoultonSolver",
    "VariableStepPredictorCorrector",
    "solve_predictor_corrector",
    "solve_adams_bashforth_moulton",
    "solve_variable_step_predictor_corrector",
    # Advanced solvers
    "AdvancedFractionalODESolver",
    "HighOrderFractionalSolver",
    "ErrorControlMethod",
    "AdaptiveMethod",
    "solve_advanced_fractional_ode",
    "solve_high_order_fractional_ode",
]
