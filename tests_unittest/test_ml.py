"""
Unittest tests for HPFRACC ML functionality
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestTensorOps(unittest.TestCase):
    """Test tensor operations functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.tensor_ops import TensorOps
        self.TensorOps = TensorOps
    
    def test_tensor_ops_initialization(self):
        """Test tensor ops initialization"""
        # Test with different backends
        backends = ['numba', 'torch', 'jax']
        for backend in backends:
            try:
                ops = self.TensorOps(backend)
                self.assertIsNotNone(ops)
                self.assertEqual(ops.backend.value, backend)
            except Exception as e:
                # Some backends may not be available
                self.assertIsInstance(e, (RuntimeError, ValueError))
    
    def test_tensor_creation(self):
        """Test tensor creation methods"""
        ops = self.TensorOps('numba')
        
        # Test basic tensor creation
        x = ops.ones((3, 4))
        y = ops.zeros((2, 3))
        z = ops.eye(4)
        
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(y.shape, (2, 3))
        self.assertEqual(z.shape, (4, 4))
    
    def test_mathematical_operations(self):
        """Test mathematical operations"""
        ops = self.TensorOps('numba')
        x = ops.ones((3, 4))
        y = ops.ones((3, 4))
        
        # Test addition
        result = ops.add(x, y)
        self.assertEqual(result.shape, (3, 4))
        
        # Test multiplication
        result = ops.multiply(x, 2.0)
        self.assertEqual(result.shape, (3, 4))
        
        # Test subtraction
        result = ops.subtract(x, y)
        self.assertEqual(result.shape, (3, 4))
    
    def test_tensor_manipulation(self):
        """Test tensor manipulation operations"""
        ops = self.TensorOps('numba')
        x = ops.ones((3, 4))
        
        # Test reshape
        reshaped = ops.reshape(x, (12, 1))
        self.assertEqual(reshaped.shape, (12, 1))
        
        # Test transpose
        transposed = ops.transpose(x)
        self.assertEqual(transposed.shape, (4, 3))
    
    def test_reduction_operations(self):
        """Test reduction operations"""
        ops = self.TensorOps('numba')
        x = ops.ones((3, 4))
        
        # Test sum
        sum_result = ops.sum(x, dim=0)
        self.assertEqual(sum_result.shape, (4,))
        
        # Test mean
        mean_result = ops.mean(x, dim=1)
        self.assertEqual(mean_result.shape, (3,))
        
        # Test max
        max_result = ops.max(x, dim=0)
        self.assertEqual(max_result.shape, (4,))
        
        # Test min
        min_result = ops.min(x, dim=1)
        self.assertEqual(min_result.shape, (3,))

class TestBackendManager(unittest.TestCase):
    """Test backend manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.backends import get_backend_manager, BackendType
        self.manager = get_backend_manager()
        self.BackendType = BackendType
    
    def test_backend_availability(self):
        """Test backend availability"""
        available = self.manager.available_backends
        self.assertGreater(len(available), 0)
        self.assertIsInstance(available, list)
        
        # Check that backends are BackendType instances
        for backend in available:
            self.assertIsInstance(backend, self.BackendType)
    
    def test_backend_switching(self):
        """Test backend switching"""
        # Test switching to available backends
        available = self.manager.available_backends
        for backend in available:
            self.manager.switch_backend(backend)
            self.assertEqual(self.manager.active_backend, backend)
    
    def test_tensor_creation_through_manager(self):
        """Test tensor creation through backend manager"""
        self.manager.switch_backend(self.BackendType.NUMBA)
        tensor = self.manager.create_tensor([1, 2, 3, 4])
        self.assertIsNotNone(tensor)
    
    def test_backend_manager_singleton(self):
        """Test that backend manager is a singleton"""
        from hpfracc.ml.backends import get_backend_manager
        manager1 = get_backend_manager()
        manager2 = get_backend_manager()
        self.assertIs(manager1, manager2)

class TestMLayers(unittest.TestCase):
    """Test machine learning layers"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.layers import FractionalLSTM, FractionalPooling
        from hpfracc.core.definitions import FractionalOrder
        self.FractionalLSTM = FractionalLSTM
        self.FractionalPooling = FractionalPooling
        self.FractionalOrder = FractionalOrder
    
    def test_fractional_lstm_initialization(self):
        """Test fractional LSTM initialization"""
        # Test with different parameters
        input_size = 10
        hidden_size = 32
        
        lstm = self.FractionalLSTM(input_size, hidden_size)
        self.assertEqual(lstm.input_size, input_size)
        self.assertEqual(lstm.hidden_size, hidden_size)
    
    def test_fractional_pooling_initialization(self):
        """Test fractional pooling initialization"""
        # Test 1D pooling
        pool_1d = self.FractionalPooling(1, kernel_size=2, stride=2)
        self.assertEqual(pool_1d.kernel_size, 2)
        self.assertEqual(pool_1d.stride, 2)
        
        # Test 2D pooling
        pool_2d = self.FractionalPooling(2, kernel_size=3, stride=2)
        self.assertEqual(pool_2d.kernel_size, 3)
        self.assertEqual(pool_2d.stride, 2)
    
    def test_fractional_lstm_forward(self):
        """Test fractional LSTM forward pass"""
        lstm = self.FractionalLSTM(10, 32)
        
        # Create mock input (batch_size=2, seq_len=5, input_size=10)
        import numpy as np
        x = np.random.randn(2, 5, 10)
        
        # Test forward pass (should return only output by default)
        output = lstm.forward(x)
        self.assertEqual(output.shape, (2, 5, 32))
    
    def test_fractional_pooling_forward(self):
        """Test fractional pooling forward pass"""
        pool_1d = self.FractionalPooling(1, kernel_size=2, stride=2)
        
        # Create mock input (batch_size=2, channels=3, length=10)
        import numpy as np
        x = np.random.randn(2, 3, 10)
        
        # Test forward pass
        output = pool_1d.forward(x)
        # Output length should be reduced by stride
        expected_length = 10 // 2  # 10 / stride=2
        self.assertEqual(output.shape, (2, 3, expected_length))

class TestSpectralAutograd(unittest.TestCase):
    """Test spectral autograd functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.spectral_autograd import SpectralFractionalLayer
        self.SpectralFractionalLayer = SpectralFractionalLayer
    
    def test_spectral_fractional_layer_initialization(self):
        """Test spectral fractional layer initialization"""
        layer = self.SpectralFractionalLayer(input_size=10, output_size=5, alpha=0.5)
        self.assertEqual(layer.input_size, 10)
        self.assertEqual(layer.output_size, 5)
        self.assertEqual(layer.alpha, 0.5)
    
    def test_spectral_fractional_layer_forward(self):
        """Test spectral fractional layer forward pass"""
        layer = self.SpectralFractionalLayer(input_size=10, output_size=5, alpha=0.5)
        
        # Create mock input
        import numpy as np
        x = np.random.randn(3, 10)
        
        # Test forward pass
        output = layer.forward(x)
        self.assertEqual(output.shape, (3, 5))

class TestMLUtilities(unittest.TestCase):
    """Test ML utility functions"""
    
    def test_tensor_operations_compatibility(self):
        """Test tensor operations compatibility across backends"""
        from hpfracc.ml.tensor_ops import TensorOps
        
        # Test that operations work consistently
        ops = TensorOps('numba')
        
        # Test basic operations
        x = ops.ones((3, 4))
        y = ops.ones((3, 4))
        
        # All operations should return tensors with correct shapes
        operations = [
            lambda a, b: ops.add(a, b),
            lambda a, b: ops.subtract(a, b),
            lambda a, b: ops.multiply(a, b),
        ]
        
        for op in operations:
            result = op(x, y)
            self.assertEqual(result.shape, (3, 4))
    
    def test_backend_consistency(self):
        """Test backend consistency"""
        from hpfracc.ml.backends import get_backend_manager, BackendType
        
        manager = get_backend_manager()
        
        # Test that backend switching is consistent
        original_backend = manager.active_backend
        
        # Switch to NUMBA if available
        if BackendType.NUMBA in manager.available_backends:
            manager.switch_backend(BackendType.NUMBA)
            self.assertEqual(manager.active_backend, BackendType.NUMBA)
            
            # Switch back
            manager.switch_backend(original_backend)
            self.assertEqual(manager.active_backend, original_backend)

if __name__ == '__main__':
    unittest.main()
