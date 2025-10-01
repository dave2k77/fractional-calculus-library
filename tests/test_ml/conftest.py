#!/usr/bin/env python3
"""
Pytest configuration specific to ML tests.

This conftest provides test fixtures specific to the ML module,
including backend state management to ensure test isolation.
"""

import pytest


@pytest.fixture(autouse=True, scope="function")
def reset_backend_for_ml_tests():
    """
    Automatically reset backend manager state before each ML test.
    
    This ensures test isolation by resetting the global backend manager singleton
    before and after each test. This prevents state pollution that can cause tests
    to pass in isolation but fail when run together.
    
    Scope: function (runs before/after every test in test_ml/)
    Autouse: True (automatically applies to all tests in this directory)
    """
    # Reset before test
    try:
        import hpfracc.ml.backends as backends_module
        backends_module._backend_manager = None
    except Exception:
        pass  # If import fails, that's okay for some tests
    
    yield
    
    # Reset after test
    try:
        import hpfracc.ml.backends as backends_module
        backends_module._backend_manager = None
    except Exception:
        pass

