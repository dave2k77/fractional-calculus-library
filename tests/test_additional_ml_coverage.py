
"""
Additional ML tests to improve coverage.
"""

import pytest
import torch
import numpy as np
from hpfracc.ml.backends import BackendManager, BackendType
from hpfracc.ml.tensor_ops import get_tensor_ops

class TestAdditionalML:
    """Additional tests for ML modules."""
    
    def test_backend_manager_edge_cases(self):
        """Test backend manager with edge cases."""
        manager = BackendManager()
        
        # Test backend switching
        original_backend = manager.active_backend
        manager.switch_backend(BackendType.TORCH)
        assert manager.active_backend == BackendType.TORCH
        
        # Test configuration
        config = manager.get_backend_config()
        assert isinstance(config, dict)
    
    def test_tensor_ops_coverage(self):
        """Test tensor operations coverage."""
        tensor_ops = get_tensor_ops(BackendType.TORCH)
        
        # Test basic operations
        x = torch.randn(10)
        y = torch.randn(10)
        
        result = tensor_ops.add(x, y)
        assert result.shape == x.shape
        
        result = tensor_ops.multiply(x, y)
        assert result.shape == x.shape
