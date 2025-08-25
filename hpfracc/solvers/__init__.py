"""
Solvers Module

This module provides various numerical and analytical solvers for fractional differential equations:
- ODE solvers
- PDE solvers
- Advanced solvers
- Predictor-corrector methods
- Homotopy Perturbation Method (HPM)
- Variational Iteration Method (VIM)
"""

# Import only the modules we know exist
# from .ode_solvers import (
#     FractionalODESolver,
#     AdamsBashforthMoulton,
#     FractionalEuler,
#     FractionalRungeKutta,
#     solve_fractional_ode
# )

# from .pde_solvers import (
#     FractionalPDESolver,
#     FractionalDiffusionSolver,
#     FractionalWaveSolver,
#     FractionalAdvectionSolver,
#     solve_fractional_pde
# )

# from .advanced_solvers import (
#     AdaptiveSolver,
#     MultiStepSolver,
#     ImplicitSolver,
#     ExplicitSolver,
#     solve_adaptive_fractional
# )

# from .predictor_corrector import (
#     PredictorCorrectorSolver,
#     AdamsPredictorCorrector,
#     FractionalPredictorCorrector,
#     solve_predictor_corrector
# )

from .homotopy_perturbation import (
    HomotopyPerturbationMethod,
    HPMFractionalDiffusion,
    HPMFractionalWave,
    validate_hpm_solution,
    hpm_convergence_analysis
)

from .variational_iteration import (
    VariationalIterationMethod,
    VIMFractionalDiffusion,
    VIMFractionalWave,
    VIMFractionalAdvection,
    validate_vim_solution,
    vim_convergence_analysis,
    compare_hpm_vim
)

__all__ = [
    # ODE Solvers
    'FractionalODESolver',
    'AdamsBashforthMoulton',
    'FractionalEuler',
    'FractionalRungeKutta',
    'solve_fractional_ode',
    
    # PDE Solvers
    'FractionalPDESolver',
    'FractionalDiffusionSolver',
    'FractionalWaveSolver',
    'FractionalAdvectionSolver',
    'solve_fractional_pde',
    
    # Advanced Solvers
    'AdaptiveSolver',
    'MultiStepSolver',
    'ImplicitSolver',
    'ExplicitSolver',
    'solve_adaptive_fractional',
    
    # Predictor-Corrector Methods
    'PredictorCorrectorSolver',
    'AdamsPredictorCorrector',
    'FractionalPredictorCorrector',
    'solve_predictor_corrector',
    
    # Homotopy Perturbation Method
    'HomotopyPerturbationMethod',
    'HPMFractionalDiffusion',
    'HPMFractionalWave',
    'validate_hpm_solution',
    'hpm_convergence_analysis',
    
    # Variational Iteration Method
    'VariationalIterationMethod',
    'VIMFractionalDiffusion',
    'VIMFractionalWave',
    'VIMFractionalAdvection',
    'validate_vim_solution',
    'vim_convergence_analysis',
    'compare_hpm_vim'
]
