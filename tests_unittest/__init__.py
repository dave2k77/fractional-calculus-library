"""
HPFRACC Unittest Test Suite

This package contains comprehensive unittest-based tests for the HPFRACC library,
providing an alternative to pytest that bypasses PyTorch import issues.
"""

__version__ = "1.0.0"
__author__ = "Davian R. Chin"

# Import test suites for easy discovery
from .test_core import *
from .test_ml import *
from .test_special import *
from .test_analytics import *
from .test_utils import *

__all__ = [
    "TestFractionalOrder",
    "TestFractionalDerivatives", 
    "TestTensorOps",
    "TestBackendManager",
    "TestMLayers",
    "TestSpecialFunctions",
    "TestAnalytics",
    "TestUtils"
]
