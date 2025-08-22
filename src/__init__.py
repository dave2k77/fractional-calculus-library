"""
Backward compatibility layer for tests that import from 'src.*'
This allows tests to continue working while maintaining proper package structure.
"""

# Import hpfracc first
import hpfracc

# Re-export all modules from hpfracc
from hpfracc import *
from hpfracc.algorithms import *
from hpfracc.core import *
from hpfracc.solvers import *
from hpfracc.special import *
from hpfracc.utils import *
from hpfracc.validation import *

# Create simple aliases for backward compatibility
src = hpfracc
src.algorithms = hpfracc.algorithms
src.core = hpfracc.core
src.solvers = hpfracc.solvers
src.special = hpfracc.special
src.utils = hpfracc.utils
src.validation = hpfracc.validation
