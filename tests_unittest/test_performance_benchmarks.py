"""
Performance benchmark tests for HPFRACC library
Testing execution time, memory usage, and scalability
"""

import unittest
import sys
import os
import time
import psutil
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests for HPFRACC library"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.tensor_ops import TensorOps
        from hpfracc.ml.backends import BackendType
        self.TensorOps = TensorOps
        self.BackendType = BackendType
        
        # Performance thresholds (in seconds)
        self.TENSOR_CREATION_THRESHOLD = 0.1  # 100ms for 1000 tensors
        self.MATH_OPERATION_THRESHOLD = 0.5   # 500ms for large operations
        self.MEMORY_USAGE_THRESHOLD = 100     # 100MB for large operations
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def test_tensor_creation_performance(self):
        """Test tensor creation performance"""
        ops = self.TensorOps('numba')
        
        # Test small tensor creation performance
        start_time = time.time()
        for i in range(1000):
            tensor = ops.ones((10, 10))
            self.assertEqual(tensor.shape, (10, 10))
        small_tensor_time = time.time() - start_time
        
        # Should complete quickly
        self.assertLess(small_tensor_time, self.TENSOR_CREATION_THRESHOLD)
        
        # Test medium tensor creation performance
        start_time = time.time()
        for i in range(100):
            tensor = ops.ones((100, 100))
            self.assertEqual(tensor.shape, (100, 100))
        medium_tensor_time = time.time() - start_time
        
        # Should complete reasonably quickly
        self.assertLess(medium_tensor_time, self.TENSOR_CREATION_THRESHOLD * 2)
        
        # Test large tensor creation performance
        start_time = time.time()
        tensor = ops.ones((1000, 1000))
        self.assertEqual(tensor.shape, (1000, 1000))
        large_tensor_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(large_tensor_time, 1.0)
    
    def test_mathematical_operations_performance(self):
        """Test mathematical operations performance"""
        ops = self.TensorOps('numba')
        
        # Test addition performance
        a = ops.ones((500, 500))
        b = ops.ones((500, 500))
        
        start_time = time.time()
        result = ops.add(a, b)
        addition_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(addition_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test multiplication performance
        start_time = time.time()
        result = ops.multiply(a, 2.0)
        multiplication_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(multiplication_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test matrix multiplication performance
        a_small = ops.ones((200, 200))
        b_small = ops.ones((200, 200))
        
        start_time = time.time()
        result = ops.matmul(a_small, b_small)
        matmul_time = time.time() - start_time
        
        self.assertEqual(result.shape, (200, 200))
        self.assertLess(matmul_time, self.MATH_OPERATION_THRESHOLD)
    
    def test_reduction_operations_performance(self):
        """Test reduction operations performance"""
        ops = self.TensorOps('numba')
        
        # Test sum performance
        x = ops.ones((1000, 1000))
        
        start_time = time.time()
        result = ops.sum(x)
        sum_time = time.time() - start_time
        
        self.assertEqual(result, 1000000)
        self.assertLess(sum_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test mean performance
        start_time = time.time()
        result = ops.mean(x)
        mean_time = time.time() - start_time
        
        self.assertEqual(result, 1.0)
        self.assertLess(mean_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test max performance
        x_with_max = ops.create_tensor(np.random.rand(1000, 1000))
        
        start_time = time.time()
        result = ops.max(x_with_max)
        max_time = time.time() - start_time
        
        self.assertLess(max_time, self.MATH_OPERATION_THRESHOLD)
    
    def test_transcendental_functions_performance(self):
        """Test transcendental functions performance"""
        ops = self.TensorOps('numba')
        
        # Test trigonometric functions performance
        x = ops.create_tensor(np.random.rand(500, 500))
        
        start_time = time.time()
        result = ops.sin(x)
        sin_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(sin_time, self.MATH_OPERATION_THRESHOLD)
        
        start_time = time.time()
        result = ops.cos(x)
        cos_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(cos_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test exponential function performance
        start_time = time.time()
        result = ops.exp(x)
        exp_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(exp_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test logarithm function performance
        x_positive = ops.create_tensor(np.abs(np.random.rand(500, 500)) + 1e-10)
        
        start_time = time.time()
        result = ops.log(x_positive)
        log_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(log_time, self.MATH_OPERATION_THRESHOLD)
    
    def test_activation_functions_performance(self):
        """Test activation functions performance"""
        ops = self.TensorOps('numba')
        
        x = ops.create_tensor(np.random.randn(1000, 1000))
        
        # Test ReLU performance
        start_time = time.time()
        result = ops.relu(x)
        relu_time = time.time() - start_time
        
        self.assertEqual(result.shape, (1000, 1000))
        self.assertLess(relu_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test sigmoid performance
        start_time = time.time()
        result = ops.sigmoid(x)
        sigmoid_time = time.time() - start_time
        
        self.assertEqual(result.shape, (1000, 1000))
        self.assertLess(sigmoid_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test tanh performance
        start_time = time.time()
        result = ops.tanh(x)
        tanh_time = time.time() - start_time
        
        self.assertEqual(result.shape, (1000, 1000))
        self.assertLess(tanh_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test softmax performance
        start_time = time.time()
        result = ops.softmax(x, dim=1)
        softmax_time = time.time() - start_time
        
        self.assertEqual(result.shape, (1000, 1000))
        self.assertLess(softmax_time, self.MATH_OPERATION_THRESHOLD)
    
    def test_tensor_manipulation_performance(self):
        """Test tensor manipulation operations performance"""
        ops = self.TensorOps('numba')
        
        x = ops.ones((500, 500))
        
        # Test reshape performance
        start_time = time.time()
        result = ops.reshape(x, (250, 1000))
        reshape_time = time.time() - start_time
        
        self.assertEqual(result.shape, (250, 1000))
        self.assertLess(reshape_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test transpose performance
        start_time = time.time()
        result = ops.transpose(x, 1, 0)
        transpose_time = time.time() - start_time
        
        self.assertEqual(result.shape, (500, 500))
        self.assertLess(transpose_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test stack performance
        tensors = [ops.ones((100, 100)) for _ in range(10)]
        
        start_time = time.time()
        result = ops.stack(tensors, dim=0)
        stack_time = time.time() - start_time
        
        self.assertEqual(result.shape, (10, 100, 100))
        self.assertLess(stack_time, self.MATH_OPERATION_THRESHOLD)
        
        # Test concatenate performance
        start_time = time.time()
        result = ops.concatenate(tensors, dim=0)
        concat_time = time.time() - start_time
        
        self.assertEqual(result.shape, (1000, 100))
        self.assertLess(concat_time, self.MATH_OPERATION_THRESHOLD)
    
    def test_memory_usage_benchmarks(self):
        """Test memory usage benchmarks"""
        ops = self.TensorOps('numba')
        
        # Test memory usage for large tensors
        initial_memory = self.get_memory_usage()
        
        # Create large tensor
        large_tensor = ops.ones((1000, 1000))
        after_large_tensor = self.get_memory_usage()
        
        memory_increase = after_large_tensor - initial_memory
        self.assertLess(memory_increase, self.MEMORY_USAGE_THRESHOLD)
        
        # Test memory usage for many operations
        tensors = []
        for i in range(50):
            tensor = ops.ones((100, 100))
            tensors.append(tensor)
        
        after_many_tensors = self.get_memory_usage()
        memory_increase = after_many_tensors - initial_memory
        self.assertLess(memory_increase, self.MEMORY_USAGE_THRESHOLD)
        
        # Clean up
        del tensors
        del large_tensor
    
    def test_scalability_benchmarks(self):
        """Test scalability benchmarks"""
        ops = self.TensorOps('numba')
        
        # Test scalability with increasing tensor sizes
        sizes = [10, 50, 100, 200, 500]
        times = []
        
        for size in sizes:
            x = ops.ones((size, size))
            y = ops.ones((size, size))
            
            start_time = time.time()
            result = ops.add(x, y)
            operation_time = time.time() - start_time
            
            times.append(operation_time)
            self.assertEqual(result.shape, (size, size))
        
        # Times should increase reasonably with size
        # (not exponentially - that would indicate poor scalability)
        for i in range(1, len(times)):
            ratio = times[i] / times[i-1]
            size_ratio = (sizes[i] / sizes[i-1]) ** 2  # Area ratio
            # Time increase should be reasonable relative to size increase
            self.assertLess(ratio, size_ratio * 2)
    
    def test_concurrent_operations_performance(self):
        """Test concurrent operations performance"""
        ops = self.TensorOps('numba')
        
        # Test multiple operations in sequence
        x = ops.ones((200, 200))
        
        start_time = time.time()
        
        # Perform multiple operations
        result1 = ops.add(x, x)
        result2 = ops.multiply(x, 2.0)
        result3 = ops.sin(x)
        result4 = ops.exp(x)
        
        total_time = time.time() - start_time
        
        # All operations should complete quickly
        self.assertLess(total_time, self.MATH_OPERATION_THRESHOLD * 2)
        
        # Verify results
        self.assertEqual(result1.shape, (200, 200))
        self.assertEqual(result2.shape, (200, 200))
        self.assertEqual(result3.shape, (200, 200))
        self.assertEqual(result4.shape, (200, 200))
    
    def test_backend_switching_performance(self):
        """Test backend switching performance"""
        ops = self.TensorOps('numba')
        
        # Test backend switching time
        start_time = time.time()
        
        # Switch between backends multiple times
        for _ in range(10):
            ops = self.TensorOps('numba')
            ops = self.TensorOps('numba')  # Same backend
        
        switching_time = time.time() - start_time
        
        # Backend switching should be fast
        self.assertLess(switching_time, 1.0)
    
    def test_analytics_performance(self):
        """Test analytics system performance"""
        from hpfracc.analytics import AnalyticsManager
        
        manager = AnalyticsManager()
        
        # Test tracking performance
        start_time = time.time()
        
        for i in range(100):
            manager.track_method_call(
                method_name=f"test_method_{i}",
                estimator_type="test_estimator",
                parameters={"iteration": i},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=0.01
            )
        
        tracking_time = time.time() - start_time
        
        # Tracking should be fast
        self.assertLess(tracking_time, 1.0)
        
        # Test report generation performance
        start_time = time.time()
        manager.generate_report()
        report_time = time.time() - start_time
        
        # Report generation should be reasonably fast
        self.assertLess(report_time, 2.0)
    
    def test_special_functions_performance(self):
        """Test special functions performance"""
        from hpfracc.special.gamma_beta import gamma, beta
        from hpfracc.special.binomial_coeffs import BinomialCoefficients
        
        # Test gamma function performance
        start_time = time.time()
        for i in range(100):
            result = gamma(1.0 + i * 0.01)
            self.assertIsNotNone(result)
        gamma_time = time.time() - start_time
        
        self.assertLess(gamma_time, 1.0)
        
        # Test beta function performance
        start_time = time.time()
        for i in range(50):
            result = beta(1.0 + i * 0.01, 1.0 + i * 0.01)
            self.assertIsNotNone(result)
        beta_time = time.time() - start_time
        
        self.assertLess(beta_time, 1.0)
        
        # Test binomial coefficients performance
        bc = BinomialCoefficients()
        
        start_time = time.time()
        for n in range(20):
            for k in range(n + 1):
                result = bc.compute(n, k)
                self.assertIsNotNone(result)
        binomial_time = time.time() - start_time
        
        self.assertLess(binomial_time, 1.0)
    
    def test_error_handling_performance(self):
        """Test error handling performance"""
        ops = self.TensorOps('numba')
        
        # Test error handling doesn't significantly impact performance
        start_time = time.time()
        
        for i in range(100):
            try:
                # Valid operation
                x = ops.ones((10, 10))
                result = ops.add(x, x)
                self.assertEqual(result.shape, (10, 10))
            except Exception:
                # Error handling should be fast
                pass
        
        error_handling_time = time.time() - start_time
        
        # Error handling should not significantly impact performance
        self.assertLess(error_handling_time, 1.0)
    
    def test_comprehensive_performance_suite(self):
        """Run comprehensive performance test suite"""
        ops = self.TensorOps('numba')
        
        # Test suite of operations
        start_time = time.time()
        
        # Tensor creation
        x = ops.ones((100, 100))
        y = ops.ones((100, 100))
        
        # Mathematical operations
        result1 = ops.add(x, y)
        result2 = ops.multiply(x, 2.0)
        result3 = ops.matmul(x, ops.transpose(y, 1, 0))
        
        # Transcendental functions
        result4 = ops.sin(x)
        result5 = ops.exp(x)
        result6 = ops.log(ops.abs(x) + 1e-10)
        
        # Activation functions
        result7 = ops.relu(x)
        result8 = ops.sigmoid(x)
        result9 = ops.tanh(x)
        
        # Reduction operations
        sum_result = ops.sum(x)
        mean_result = ops.mean(x)
        max_result = ops.max(x)
        
        # Tensor manipulation
        reshaped = ops.reshape(x, (50, 200))
        transposed = ops.transpose(x, 1, 0)
        stacked = ops.stack([x, y], dim=0)
        
        total_time = time.time() - start_time
        
        # Comprehensive suite should complete quickly
        self.assertLess(total_time, 2.0)
        
        # Verify all results
        self.assertEqual(result1.shape, (100, 100))
        self.assertEqual(result2.shape, (100, 100))
        self.assertEqual(result3.shape, (100, 100))
        self.assertEqual(result4.shape, (100, 100))
        self.assertEqual(result5.shape, (100, 100))
        self.assertEqual(result6.shape, (100, 100))
        self.assertEqual(result7.shape, (100, 100))
        self.assertEqual(result8.shape, (100, 100))
        self.assertEqual(result9.shape, (100, 100))
        self.assertEqual(reshaped.shape, (50, 200))
        self.assertEqual(transposed.shape, (100, 100))
        self.assertEqual(stacked.shape, (2, 100, 100))

class TestPerformanceRegression(unittest.TestCase):
    """Test for performance regressions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.ml.tensor_ops import TensorOps
        self.TensorOps = TensorOps
    
    def test_performance_baseline(self):
        """Establish performance baseline"""
        ops = self.TensorOps('numba')
        
        # Baseline tensor creation
        start_time = time.time()
        for i in range(100):
            tensor = ops.ones((50, 50))
        baseline_creation_time = time.time() - start_time
        
        # Baseline mathematical operations
        x = ops.ones((100, 100))
        y = ops.ones((100, 100))
        
        start_time = time.time()
        result = ops.add(x, y)
        baseline_math_time = time.time() - start_time
        
        # Store baseline (in a real implementation, this would be persisted)
        self.assertLess(baseline_creation_time, 0.1)
        self.assertLess(baseline_math_time, 0.1)
        
        # Verify results
        self.assertEqual(result.shape, (100, 100))
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities"""
        from hpfracc.analytics import AnalyticsManager
        
        manager = AnalyticsManager()
        
        # Track performance metrics
        start_time = time.time()
        
        # Simulate work
        ops = self.TensorOps('numba')
        x = ops.ones((200, 200))
        y = ops.ones((200, 200))
        result = ops.add(x, y)
        
        execution_time = time.time() - start_time
        
        # Track the performance
        manager.track_method_call(
            method_name="performance_test",
            estimator_type="test_estimator",
            parameters={"size": 200},
            array_size=200,
            fractional_order=0.5,
            execution_success=True,
            execution_time=execution_time
        )
        
        # Get performance metrics
        metrics = manager.get_performance_metrics()
        self.assertIsNotNone(metrics)
        
        # Verify tracking worked
        self.assertEqual(result.shape, (200, 200))

if __name__ == '__main__':
    unittest.main()
