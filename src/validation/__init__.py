"""
Validation Module

This module provides comprehensive validation tools for fractional calculus
computations, including analytical solutions, convergence tests, and benchmarks.
"""

from .analytical_solutions import (
    AnalyticalSolutions,
    PowerFunctionSolutions,
    ExponentialSolutions,
    TrigonometricSolutions,
    get_analytical_solution,
    validate_against_analytical,
)

from .convergence_tests import (
    ConvergenceTester,
    ConvergenceAnalyzer,
    OrderOfAccuracy,
    run_convergence_study,
    run_method_convergence_test,
    estimate_convergence_rate,
)

from .benchmarks import (
    BenchmarkSuite,
    PerformanceBenchmark,
    AccuracyBenchmark,
    run_benchmarks,
    compare_methods,
    generate_benchmark_report,
)

__all__ = [
    # Analytical solutions
    "AnalyticalSolutions",
    "PowerFunctionSolutions", 
    "ExponentialSolutions",
    "TrigonometricSolutions",
    "get_analytical_solution",
    "validate_against_analytical",
    
    # Convergence tests
    "ConvergenceTester",
    "ConvergenceAnalyzer",
    "OrderOfAccuracy", 
    "run_convergence_study",
    "run_method_convergence_test",
    "estimate_convergence_rate",
    
    # Benchmarks
    "BenchmarkSuite",
    "PerformanceBenchmark",
    "AccuracyBenchmark",
    "run_benchmarks",
    "compare_methods",
    "generate_benchmark_report",
]
