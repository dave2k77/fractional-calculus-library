#!/usr/bin/env python3
"""
Pytest configuration and common fixtures for fractional calculus library tests.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_time_array():
    """Provide a sample time array for testing."""
    return np.linspace(0.1, 2.0, 50)


@pytest.fixture
def sample_function_values():
    """Provide sample function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return t  # Simple linear function


@pytest.fixture
def sample_quadratic_function():
    """Provide quadratic function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return t**2


@pytest.fixture
def sample_exponential_function():
    """Provide exponential function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return np.exp(-t)


@pytest.fixture
def sample_trigonometric_function():
    """Provide trigonometric function values for testing."""
    t = np.linspace(0.1, 2.0, 50)
    return np.sin(t)


@pytest.fixture
def fractional_orders():
    """Provide various fractional orders for testing."""
    return [0.25, 0.5, 0.75, 1.0, 1.5]


@pytest.fixture
def step_sizes():
    """Provide various step sizes for testing."""
    return [0.01, 0.05, 0.1]


@pytest.fixture
def grid_sizes():
    """Provide various grid sizes for testing."""
    return [25, 50, 100, 200]


@pytest.fixture
def tolerance():
    """Provide tolerance for numerical comparisons."""
    return 1e-10


@pytest.fixture
def analytical_solutions():
    """Provide analytical solutions for known test cases."""
    from scipy.special import gamma

    def get_caputo_analytical(t, alpha):
        """Analytical solution for Caputo derivative of f(t) = t."""
        return t ** (1 - alpha) / gamma(2 - alpha)

    def get_riemann_liouville_analytical(t, alpha):
        """Analytical solution for Riemann-Liouville derivative of f(t) = t."""
        return t ** (1 - alpha) / gamma(2 - alpha)

    return {
        "caputo_linear": get_caputo_analytical,
        "riemann_liouville_linear": get_riemann_liouville_analytical,
    }


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")

    for item in items:
        if "gpu" in item.keywords:
            try:
                import jax

                # Check if GPU is available
                if not jax.devices("gpu"):
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
