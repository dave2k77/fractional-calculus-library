"""
Utility modules for the Fractional Calculus Library.

This package provides various utility functions for:
- Error analysis and validation
- Memory management and optimization
- Plotting and visualization
"""

from .error_analysis import (
    ErrorAnalyzer,
    ConvergenceAnalyzer,
    ValidationFramework,
    compute_error_metrics,
    analyze_convergence,
    validate_solution,
)

from .memory_management import (
    MemoryManager,
    CacheManager,
    optimize_memory_usage,
    clear_cache,
    get_memory_usage,
)

from .plotting import (
    PlotManager,
    create_comparison_plot,
    plot_convergence,
    plot_error_analysis,
    save_plot,
    setup_plotting_style,
)

__all__ = [
    # Error analysis
    "ErrorAnalyzer",
    "ConvergenceAnalyzer",
    "ValidationFramework",
    "compute_error_metrics",
    "analyze_convergence",
    "validate_solution",
    # Memory management
    "MemoryManager",
    "CacheManager",
    "optimize_memory_usage",
    "clear_cache",
    "get_memory_usage",
    # Plotting
    "PlotManager",
    "create_comparison_plot",
    "plot_convergence",
    "plot_error_analysis",
    "save_plot",
    "setup_plotting_style",
]
