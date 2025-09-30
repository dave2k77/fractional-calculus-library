"""
Comprehensive unittest tests for HPFRACC tensor operations
Focusing on expanding coverage from 28% to 50%+
"""

import unittest
import sys
import os
import numpy as np
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestTensorOpsComprehensive(unittest.TestCase):
    """Comprehensive tests for tensor operations to increase coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.tensor_ops import TensorOps
        self.TensorOps = TensorOps
    
    def test_backend_resolution_comprehensive(self):
        """Test comprehensive backend resolution scenarios"""
        # Test with explicit backend
        ops = self.TensorOps('numba')
        self.assertEqual(ops.backend.value, 'numba')
        
        # Test with BackendType enum
        from hpfracc.ml.backends import BackendType
        ops = self.TensorOps(BackendType.NUMBA)
        self.assertEqual(ops.backend, BackendType.NUMBA)
        
        # Test with None (should use fallback)
        ops = self.TensorOps(None)
        self.assertIn(ops.backend, [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA])
        
        # Test invalid backend string
        with self.assertRaises(ValueError):
            self.TensorOps('invalid_backend')
    
    def test_tensor_creation_comprehensive(self):
        """Test comprehensive tensor creation methods"""
        ops = self.TensorOps('numba')
        
        # Test create_tensor with various data types
        data_list = [1, 2, 3, 4]
        tensor1 = ops.create_tensor(data_list)
        self.assertEqual(tensor1.shape, (4,))
        
        # Test create_tensor with numpy array
        data_array = np.array([1, 2, 3, 4])
        tensor2 = ops.create_tensor(data_array)
        self.assertEqual(tensor2.shape, (4,))
        
        # Test tensor method (alias)
        tensor3 = ops.tensor([1, 2, 3, 4])
        self.assertEqual(tensor3.shape, (4,))
        
        # Test from_numpy
        tensor4 = ops.from_numpy(data_array)
        self.assertEqual(tensor4.shape, (4,))
    
    def test_numpy_conversion_comprehensive(self):
        """Test comprehensive numpy conversion methods"""
        ops = self.TensorOps('numba')
        
        # Test to_numpy
        tensor = ops.ones((3, 4))
        numpy_array = ops.to_numpy(tensor)
        self.assertEqual(numpy_array.shape, (3, 4))
        self.assertIsInstance(numpy_array, np.ndarray)
        
        # Test from_numpy round trip
        original = np.random.randn(2, 3)
        tensor = ops.from_numpy(original)
        converted = ops.to_numpy(tensor)
        np.testing.assert_array_almost_equal(original, converted)
    
    def test_context_managers(self):
        """Test context managers like no_grad"""
        ops = self.TensorOps('numba')
        
        # Test no_grad context manager
        with ops.no_grad():
            tensor = ops.ones((2, 3))
            self.assertIsNotNone(tensor)
        
        # Test that we can still create tensors after context
        tensor2 = ops.zeros((2, 3))
        self.assertIsNotNone(tensor2)
    
    def test_tensor_creation_methods_comprehensive(self):
        """Test all tensor creation methods"""
        ops = self.TensorOps('numba')
        
        # Test zeros
        zeros = ops.zeros((3, 4))
        self.assertEqual(zeros.shape, (3, 4))
        np.testing.assert_array_equal(ops.to_numpy(zeros), np.zeros((3, 4)))
        
        # Test ones
        ones = ops.ones((2, 5))
        self.assertEqual(ones.shape, (2, 5))
        np.testing.assert_array_equal(ops.to_numpy(ones), np.ones((2, 5)))
        
        # Test eye
        eye = ops.eye(4)
        self.assertEqual(eye.shape, (4, 4))
        np.testing.assert_array_equal(ops.to_numpy(eye), np.eye(4))
        
        # Test arange
        arange = ops.arange(0, 10, 2)
        self.assertEqual(arange.shape, (5,))
        np.testing.assert_array_equal(ops.to_numpy(arange), np.arange(0, 10, 2))
        
        # Test linspace
        linspace = ops.linspace(0, 1, 5)
        self.assertEqual(linspace.shape, (5,))
        expected = np.linspace(0, 1, 5)
        np.testing.assert_array_almost_equal(ops.to_numpy(linspace), expected)
        
        # Test zeros_like
        original = ops.ones((3, 4))
        zeros_like = ops.zeros_like(original)
        self.assertEqual(zeros_like.shape, (3, 4))
        np.testing.assert_array_equal(ops.to_numpy(zeros_like), np.zeros((3, 4)))
    
    def test_mathematical_operations_comprehensive(self):
        """Test comprehensive mathematical operations"""
        ops = self.TensorOps('numba')
        
        # Create test tensors
        a = ops.ones((3, 4))
        b = ops.ones((3, 4))
        c = ops.ones((4, 5))
        
        # Test addition
        result = ops.add(a, b)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), 2 * np.ones((3, 4)))
        
        # Test subtraction
        result = ops.subtract(a, b)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), np.zeros((3, 4)))
        
        # Test multiplication
        result = ops.multiply(a, 2.0)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), 2 * np.ones((3, 4)))
        
        # Test division
        result = ops.divide(a, 2.0)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), 0.5 * np.ones((3, 4)))
        
        # Test power
        result = ops.power(a, 2)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), np.ones((3, 4)))
        
        # Test matrix multiplication
        result = ops.matmul(a, c)
        self.assertEqual(result.shape, (3, 5))
        
        # Test element-wise operations
        result = ops.multiply(a, b)
        self.assertEqual(result.shape, (3, 4))
    
    def test_transcendental_functions(self):
        """Test transcendental functions"""
        ops = self.TensorOps('numba')
        
        # Create test tensor
        x = ops.create_tensor([0.0, 0.5, 1.0, 2.0])
        
        # Test sqrt
        result = ops.sqrt(x)
        expected = np.sqrt([0.0, 0.5, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test sin
        result = ops.sin(x)
        expected = np.sin([0.0, 0.5, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test cos
        result = ops.cos(x)
        expected = np.cos([0.0, 0.5, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test exp
        result = ops.exp(x)
        expected = np.exp([0.0, 0.5, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test log
        x_positive = ops.create_tensor([1.0, 2.0, 3.0, 4.0])
        result = ops.log(x_positive)
        expected = np.log([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test abs
        x_signed = ops.create_tensor([-1.0, -0.5, 0.5, 1.0])
        result = ops.abs(x_signed)
        expected = np.abs([-1.0, -0.5, 0.5, 1.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
    
    def test_activation_functions(self):
        """Test activation functions"""
        ops = self.TensorOps('numba')
        
        # Create test tensor
        x = ops.create_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test ReLU
        result = ops.relu(x)
        expected = np.maximum(0, [-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test sigmoid
        result = ops.sigmoid(x)
        expected = 1 / (1 + np.exp(-np.array([-2.0, -1.0, 0.0, 1.0, 2.0])))
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected, decimal=5)
        
        # Test tanh
        result = ops.tanh(x)
        expected = np.tanh([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
        
        # Test softmax
        result = ops.softmax(x, dim=0)
        expected = np.exp([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = expected / np.sum(expected)
        np.testing.assert_array_almost_equal(ops.to_numpy(result), expected)
    
    def test_tensor_manipulation_comprehensive(self):
        """Test comprehensive tensor manipulation operations"""
        ops = self.TensorOps('numba')
        
        # Create test tensor
        x = ops.ones((2, 3, 4))
        
        # Test reshape
        reshaped = ops.reshape(x, (6, 4))
        self.assertEqual(reshaped.shape, (6, 4))
        
        # Test transpose
        transposed = ops.transpose(x, 1, 0, 2)
        self.assertEqual(transposed.shape, (3, 2, 4))
        
        # Test squeeze
        squeezed = ops.squeeze(x)
        self.assertEqual(squeezed.shape, (2, 3, 4))  # No singleton dims to squeeze
        
        # Test unsqueeze
        unsqueezed = ops.unsqueeze(x, 1)
        self.assertEqual(unsqueezed.shape, (2, 1, 3, 4))
        
        # Test stack
        tensors = [ops.ones((2, 3)) for _ in range(4)]
        stacked = ops.stack(tensors, dim=0)
        self.assertEqual(stacked.shape, (4, 2, 3))
        
        # Test concatenate
        concatenated = ops.concatenate(tensors, dim=0)
        self.assertEqual(concatenated.shape, (8, 3))
        
        # Test repeat
        repeated = ops.repeat(x, 2, dim=1)
        self.assertEqual(repeated.shape, (2, 6, 4))
        
        # Test tile
        tiled = ops.tile(x, 2)
        self.assertEqual(tiled.shape, (2, 3, 8))
        
        # Test clip
        clipped = ops.clip(x, 0.5, 1.5)
        self.assertEqual(clipped.shape, (2, 3, 4))
        result_array = ops.to_numpy(clipped)
        self.assertTrue(np.all(result_array >= 0.5))
        self.assertTrue(np.all(result_array <= 1.5))
    
    def test_reduction_operations_comprehensive(self):
        """Test comprehensive reduction operations"""
        ops = self.TensorOps('numba')
        
        # Create test tensor
        x = ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        
        # Test sum
        sum_all = ops.sum(x)
        self.assertEqual(sum_all, 21)
        
        sum_dim0 = ops.sum(x, dim=0)
        self.assertEqual(sum_dim0.shape, (3,))
        np.testing.assert_array_equal(ops.to_numpy(sum_dim0), [5, 7, 9])
        
        # Test mean
        mean_all = ops.mean(x)
        self.assertAlmostEqual(mean_all, 3.5)
        
        mean_dim1 = ops.mean(x, dim=1)
        self.assertEqual(mean_dim1.shape, (2,))
        np.testing.assert_array_almost_equal(ops.to_numpy(mean_dim1), [2, 5])
        
        # Test max
        max_all = ops.max(x)
        self.assertEqual(max_all, 6)
        
        max_dim0 = ops.max(x, dim=0)
        self.assertEqual(max_dim0.shape, (3,))
        np.testing.assert_array_equal(ops.to_numpy(max_dim0), [4, 5, 6])
        
        # Test min
        min_all = ops.min(x)
        self.assertEqual(min_all, 1)
        
        min_dim1 = ops.min(x, dim=1)
        self.assertEqual(min_dim1.shape, (2,))
        np.testing.assert_array_equal(ops.to_numpy(min_dim1), [1, 4])
        
        # Test std
        std_all = ops.std(x)
        self.assertGreater(std_all, 0)
        
        # Test median
        median_all = ops.median(x)
        self.assertEqual(median_all, 3.5)
        
        # Test norm
        norm_result = ops.norm(x, p=2)
        self.assertGreater(norm_result, 0)
    
    def test_random_operations(self):
        """Test random operations"""
        ops = self.TensorOps('numba')
        
        # Test randn
        random_tensor = ops.randn((3, 4))
        self.assertEqual(random_tensor.shape, (3, 4))
        
        # Test randn_like
        template = ops.ones((2, 5))
        random_like = ops.randn_like(template)
        self.assertEqual(random_like.shape, (2, 5))
        
        # Test dropout
        x = ops.ones((10, 10))
        dropped = ops.dropout(x, p=0.5, training=True)
        self.assertEqual(dropped.shape, (10, 10))
        
        # Test dropout in eval mode
        dropped_eval = ops.dropout(x, p=0.5, training=False)
        np.testing.assert_array_equal(ops.to_numpy(dropped_eval), ops.to_numpy(x))
    
    def test_advanced_operations(self):
        """Test advanced operations"""
        ops = self.TensorOps('numba')
        
        # Test einsum
        a = ops.ones((3, 4))
        b = ops.ones((4, 5))
        result = ops.einsum('ij,jk->ik', a, b)
        self.assertEqual(result.shape, (3, 5))
        
        # Test gather
        x = ops.create_tensor([[1, 2, 3], [4, 5, 6]])
        indices = ops.create_tensor([[0, 2], [1, 0]])
        gathered = ops.gather(x, dim=1, index=indices)
        self.assertEqual(gathered.shape, (2, 2))
        
        # Test expand
        x = ops.ones((1, 3, 1))
        expanded = ops.expand(x, 2, 3, 4)
        self.assertEqual(expanded.shape, (2, 3, 4))
        
        # Test clone
        x = ops.ones((2, 3))
        cloned = ops.clone(x)
        self.assertEqual(cloned.shape, (2, 3))
        np.testing.assert_array_equal(ops.to_numpy(cloned), ops.to_numpy(x))
    
    def test_fft_operations(self):
        """Test FFT operations"""
        ops = self.TensorOps('numba')
        
        # Create test signal
        x = ops.create_tensor([1, 2, 3, 4, 5, 6, 7, 8])
        
        # Test FFT
        fft_result = ops.fft(x)
        self.assertEqual(fft_result.shape, (8,))
        
        # Test IFFT
        ifft_result = ops.ifft(fft_result)
        self.assertEqual(ifft_result.shape, (8,))
        
        # Test round trip (should recover original)
        recovered = ops.to_numpy(ifft_result)
        original = ops.to_numpy(x)
        np.testing.assert_array_almost_equal(recovered.real, original, decimal=10)
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        ops = self.TensorOps('numba')
        
        # Test empty tensor creation
        empty = ops.zeros((0,))
        self.assertEqual(empty.shape, (0,))
        
        # Test single element tensor
        single = ops.ones((1,))
        self.assertEqual(single.shape, (1,))
        
        # Test large tensor (should not crash)
        large = ops.zeros((100, 100))
        self.assertEqual(large.shape, (100, 100))
        
        # Test operations on edge cases
        empty_sum = ops.sum(empty)
        self.assertEqual(empty_sum, 0)
        
        # Test division by zero handling
        x = ops.create_tensor([1.0, 0.0, -1.0])
        y = ops.create_tensor([0.0, 1.0, 0.0])
        # This should handle division by zero gracefully
        try:
            result = ops.divide(x, y)
            # If it doesn't raise an exception, check that we get expected behavior
            self.assertEqual(result.shape, (3,))
        except (ZeroDivisionError, RuntimeError):
            # Division by zero should be handled gracefully
            pass
    
    def test_performance_characteristics(self):
        """Test performance characteristics"""
        ops = self.TensorOps('numba')
        
        # Test tensor creation performance
        start_time = time.time()
        for _ in range(100):
            ops.ones((100, 100))
        creation_time = time.time() - start_time
        
        # Should complete quickly (less than 1 second for 100 operations)
        self.assertLess(creation_time, 1.0)
        
        # Test mathematical operations performance
        a = ops.ones((1000, 1000))
        b = ops.ones((1000, 1000))
        
        start_time = time.time()
        result = ops.add(a, b)
        operation_time = time.time() - start_time
        
        # Should complete quickly (less than 1 second for large operation)
        self.assertLess(operation_time, 1.0)
        self.assertEqual(result.shape, (1000, 1000))
    
    def test_memory_management(self):
        """Test memory management characteristics"""
        ops = self.TensorOps('numba')
        
        # Test that we can create many tensors without memory issues
        tensors = []
        for i in range(50):
            tensor = ops.ones((100, 100))
            tensors.append(tensor)
            # Verify tensor is still valid
            self.assertEqual(tensor.shape, (100, 100))
        
        # Test that operations work with many tensors
        result = ops.add(tensors[0], tensors[1])
        self.assertEqual(result.shape, (100, 100))
        
        # Test cleanup (tensors should be garbage collected)
        del tensors
        # Should not crash
        new_tensor = ops.zeros((50, 50))
        self.assertEqual(new_tensor.shape, (50, 50))

class TestTensorOpsModuleFunctions(unittest.TestCase):
    """Test module-level functions"""
    
    def test_get_tensor_ops(self):
        """Test get_tensor_ops function"""
        from hpfracc.ml.tensor_ops import get_tensor_ops
        from hpfracc.ml.backends import BackendType
        
        # Test with explicit backend
        ops = get_tensor_ops(BackendType.NUMBA)
        self.assertEqual(ops.backend, BackendType.NUMBA)
        
        # Test with None
        ops = get_tensor_ops(None)
        self.assertIn(ops.backend, [BackendType.TORCH, BackendType.JAX, BackendType.NUMBA])
    
    def test_create_tensor_function(self):
        """Test create_tensor module function"""
        from hpfracc.ml.tensor_ops import create_tensor
        
        # Test basic creation
        tensor = create_tensor([1, 2, 3, 4])
        self.assertEqual(tensor.shape, (4,))
        
        # Test with numpy array
        import numpy as np
        array = np.array([1, 2, 3, 4])
        tensor = create_tensor(array)
        self.assertEqual(tensor.shape, (4,))
    
    def test_switch_backend_function(self):
        """Test switch_backend function"""
        from hpfracc.ml.tensor_ops import switch_backend
        from hpfracc.ml.backends import BackendType, get_backend_manager
        
        # Get initial backend
        manager = get_backend_manager()
        initial_backend = manager.active_backend
        
        try:
            # Switch to NUMBA
            switch_backend(BackendType.NUMBA)
            self.assertEqual(manager.active_backend, BackendType.NUMBA)
            
            # Switch back
            switch_backend(initial_backend)
            self.assertEqual(manager.active_backend, initial_backend)
        finally:
            # Ensure we restore the original backend
            manager.switch_backend(initial_backend)

if __name__ == '__main__':
    unittest.main()
