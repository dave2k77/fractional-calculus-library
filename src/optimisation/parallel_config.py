"""
Parallel Computing Configuration

This module provides configuration settings for parallel computing
in the fractional calculus library.
"""

# Default parallel computing backend
DEFAULT_BACKEND = "joblib"

# Default number of workers (None = auto-detect)
DEFAULT_NUM_WORKERS = None

# Backend preferences (in order of preference)
BACKEND_PREFERENCES = [
    "joblib",  # Best for scientific computing
    "dask",  # Best for large data
    "ray",  # Best for distributed computing
    "multiprocessing",  # Fallback
    "threading",  # Last resort
]

# Performance settings
PERFORMANCE_SETTINGS = {
    "joblib": {
        "n_jobs": -1,  # Use all available cores
        "backend": "multiprocessing",
        "prefer": "processes",
        "verbose": 0,
    },
    "dask": {
        "n_workers": None,  # Auto-detect
        "threads_per_worker": 2,
        "memory_limit": "2GB",
    },
    "ray": {"num_cpus": None, "object_store_memory": 1000000000},  # Auto-detect  # 1GB
    "multiprocessing": {"max_workers": None},  # Auto-detect
}

# Use case recommendations
USE_CASE_RECOMMENDATIONS = {
    "general": "joblib",
    "scientific": "joblib",
    "large_data": "dask",
    "distributed": "ray",
    "simple": "joblib",
    "io_bound": "threading",
    "cpu_intensive": "joblib",
}


def get_optimal_backend(use_case: str = "general") -> str:
    """
    Get the optimal backend for a given use case.

    Args:
        use_case: Type of computation

    Returns:
        Recommended backend name
    """
    return USE_CASE_RECOMMENDATIONS.get(use_case, "joblib")


def get_backend_settings(backend: str) -> dict:
    """
    Get settings for a specific backend.

    Args:
        backend: Backend name

    Returns:
        Dictionary of settings
    """
    return PERFORMANCE_SETTINGS.get(backend, {})
