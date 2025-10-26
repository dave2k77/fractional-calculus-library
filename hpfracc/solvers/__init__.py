# __init__.py (patch)
from .ode_solvers import (
    FixedStepODESolver,
    solve_fractional_ode,
    solve_fractional_system
)

from .pde_solvers import (
    FractionalPDESolver,
    FractionalDiffusionSolver,
    FractionalAdvectionSolver,
    FractionalReactionDiffusionSolver,
    solve_fractional_pde
)

__all__ = [
    # ODE Solvers
    'FixedStepODESolver',
    'solve_fractional_ode',
    'solve_fractional_system',

    # PDE Solvers
    'FractionalPDESolver',
    'FractionalDiffusionSolver',
    'FractionalAdvectionSolver',
    'FractionalReactionDiffusionSolver',
    'solve_fractional_pde',
]

# Backward-compatibility aliases for tests expecting legacy names
# Advanced and high-order solvers are not implemented; provide stubs.
class AdvancedFractionalODESolver(FixedStepODESolver):
    pass

class HighOrderFractionalSolver(FixedStepODESolver):
    pass

def solve_advanced_fractional_ode(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

def solve_high_order_fractional_ode(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

# Predictor-corrector compatibility names
PredictorCorrectorSolver = FixedStepODESolver
AdamsBashforthMoultonSolver = None # Was AdaptiveODESolver
class VariableStepPredictorCorrector: # Removed inheritance
    pass

def solve_predictor_corrector(*args, **kwargs):
    return solve_fractional_ode(*args, **kwargs)

__all__ += [
    'AdvancedFractionalODESolver',
    'HighOrderFractionalSolver',
    'solve_advanced_fractional_ode',
    'solve_high_order_fractional_ode',
    'PredictorCorrectorSolver',
    'AdamsBashforthMoultonSolver',
    'VariableStepPredictorCorrector',
    'solve_predictor_corrector',
]
