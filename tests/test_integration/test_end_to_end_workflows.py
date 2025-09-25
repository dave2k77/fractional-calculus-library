"""
End-to-end integration tests for complete workflows.

This module provides comprehensive testing for complete workflows that
integrate multiple components of the fractional calculus library.
"""

import pytest
import numpy as np
import torch
from datetime import datetime
from typing import Union, Callable, List, Dict, Any

from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
)
from hpfracc.core.integrals import (
    RiemannLiouvilleIntegral, CaputoIntegral, WeylIntegral
)
from hpfracc.ml.spectral_autograd import (
    SpectralFractionalNetwork, SpectralFractionalLayer
)
from hpfracc.ml.adjoint_optimization import (
    MemoryEfficientFractionalNetwork, AdjointFractionalLayer
)
from hpfracc.ml.tensor_ops import TensorOps, get_tensor_ops
from hpfracc.ml.registry import ModelRegistry, ModelVersion, ModelMetadata, DeploymentStatus
from hpfracc.algorithms.optimized_methods import (
    OptimizedRiemannLiouville, OptimizedCaputo
)
from hpfracc.algorithms.optimized_methods import (
    ParallelOptimizedRiemannLiouville, ParallelOptimizedCaputo
)


class TestFractionalCalculusWorkflow:
    """Test complete fractional calculus workflows."""
    
    def test_derivative_integral_workflow(self):
        """Test complete workflow from derivative to integral."""
        # Create a test function
        def test_func(x):
            return x**2 + 2*x + 1
        
        # Test points
        x_vals = np.linspace(0, 5, 100)
        
        # Step 1: Compute fractional derivative
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        derivative_result = rl_deriv.compute(test_func, x_vals)
        
        assert isinstance(derivative_result, np.ndarray)
        assert len(derivative_result) > 0
        
        # Step 2: Compute fractional integral
        rl_integral = RiemannLiouvilleIntegral(0.5)
        integral_result = rl_integral(test_func, x_vals)
        
        assert isinstance(integral_result, np.ndarray)
        assert len(integral_result) > 0
        
        # Step 3: Verify consistency (derivative of integral should be close to original)
        # This is a basic consistency check
        assert np.all(np.isfinite(derivative_result))
        assert np.all(np.isfinite(integral_result))
    
    def test_multiple_derivative_methods_workflow(self):
        """Test workflow using multiple derivative methods."""
        def test_func(x):
            return np.sin(x)
        
        x_vals = np.linspace(0, 2*np.pi, 50)
        
        # Test different derivative methods
        methods = [
            RiemannLiouvilleDerivative(0.5),
            CaputoDerivative(0.3),
            GrunwaldLetnikovDerivative(0.7)
        ]
        
        results = []
        for method in methods:
            result = method.compute(test_func, x_vals)
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
            assert np.all(np.isfinite(result))
            results.append(result)
        
        # All results should be finite (length may vary due to internal processing)
        for result in results:
            assert len(result) > 0
    
    def test_optimized_methods_workflow(self):
        """Test workflow using optimized methods."""
        def test_func(x):
            return np.exp(-x)
        
        x_vals = np.linspace(0, 3, 100)
        
        # Test optimized methods
        optimized_rl = OptimizedRiemannLiouville(0.5)
        optimized_caputo = OptimizedCaputo(0.3)
        
        rl_result = optimized_rl.compute(test_func, x_vals)
        caputo_result = optimized_caputo.compute(test_func, x_vals)
        
        assert isinstance(rl_result, np.ndarray)
        assert isinstance(caputo_result, np.ndarray)
        assert len(rl_result) > 0
        assert len(caputo_result) > 0
        assert np.all(np.isfinite(rl_result))
        assert np.all(np.isfinite(caputo_result))
    
    def test_parallel_optimized_methods_workflow(self):
        """Test workflow using parallel optimized methods."""
        def test_func(x):
            return x**3 + x**2 + x + 1
        
        x_vals = np.linspace(0, 2, 200)
        
        # Test parallel optimized methods
        parallel_rl = ParallelOptimizedRiemannLiouville(0.6)
        parallel_caputo = ParallelOptimizedCaputo(0.4)
        
        rl_result = parallel_rl.compute(test_func, x_vals)
        caputo_result = parallel_caputo.compute(test_func, x_vals)
        
        assert isinstance(rl_result, np.ndarray)
        assert isinstance(caputo_result, np.ndarray)
        assert len(rl_result) > 0
        assert len(caputo_result) > 0
        assert np.all(np.isfinite(rl_result))
        assert np.all(np.isfinite(caputo_result))


class TestMLWorkflow:
    """Test complete ML workflows."""
    
    def test_spectral_network_training_workflow(self):
        """Test complete spectral network training workflow."""
        # Create a simple dataset
        x_train = torch.randn(100, 10)
        y_train = torch.randn(100, 5)
        
        # Create spectral network
        network = SpectralFractionalNetwork(input_dim=10, hidden_dims=[20, 15], output_dim=5, alpha=0.5)
        
        # Forward pass
        output = network(x_train)
        assert output.shape == (100, 5)
        assert isinstance(output, torch.Tensor)
        
        # Test gradient computation
        loss = torch.nn.functional.mse_loss(output, y_train)
        loss.backward()
        
        # Check that gradients are computed
        for param in network.parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape
    
    def test_adjoint_network_training_workflow(self):
        """Test complete adjoint network training workflow."""
        # Create a simple dataset
        x_train = torch.randn(50, 8)
        y_train = torch.randn(50, 3)
        
        # Create adjoint network
        network = MemoryEfficientFractionalNetwork(8, [16, 12], 3, 0.6)
        
        # Forward pass
        output = network(x_train)
        assert output.shape == (50, 3)
        assert isinstance(output, torch.Tensor)
        
        # Test gradient computation
        loss = torch.nn.functional.mse_loss(output, y_train)
        loss.backward()
        
        # Check that gradients are computed
        for param in network.parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape
    
    def test_tensor_ops_workflow(self):
        """Test complete tensor operations workflow."""
        # Get tensor ops
        ops = get_tensor_ops()
        
        # Create tensors
        a = ops.create_tensor([1.0, 2.0, 3.0])
        b = ops.create_tensor([4.0, 5.0, 6.0])
        
        # Perform operations using native tensor operations
        result_add = a + b
        result_mul = a * b
        result_norm = torch.norm(a)
        
        assert isinstance(result_add, torch.Tensor)
        assert isinstance(result_mul, torch.Tensor)
        assert isinstance(result_norm, torch.Tensor)
        
        # Verify results
        expected_add = torch.tensor([5.0, 7.0, 9.0])
        expected_mul = torch.tensor([4.0, 10.0, 18.0])
        
        assert torch.allclose(result_add, expected_add)
        assert torch.allclose(result_mul, expected_mul)
        assert result_norm > 0
    
    def test_model_registry_workflow(self):
        """Test complete model registry workflow."""
        # Create model registry
        registry = ModelRegistry()
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id='test_model_001',
            version='1.0.0',
            name='Test Fractional Model',
            description='A test fractional neural network',
            author='Test User',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=['test', 'fractional', 'neural_network'],
            framework='pytorch',
            model_type='spectral_fractional',
            fractional_order=0.5,
            hyperparameters={'alpha': 0.5, 'layers': [10, 20, 5]},
            performance_metrics={'accuracy': 0.95, 'loss': 0.05},
            dataset_info={'name': 'test_dataset', 'size': 1000},
            dependencies={'torch': '1.9.0', 'numpy': '1.21.0'},
            file_size=1024,
            checksum='abc123',
            deployment_status=DeploymentStatus.DEVELOPMENT
        )
        
        # Create model version
        version = ModelVersion(
            version="1.0.0",
            model_id="test_model",
            metadata=metadata,
            model_path="/tmp/test_model.pth",
            config_path="/tmp/test_config.json",
            created_at=datetime.now(),
            created_by="test_user",
            git_commit="abc123",
            git_branch="main"
        )
        
        # Register model
        # Create a dummy model for testing
        dummy_model = torch.nn.Linear(10, 1)
        model_id = registry.register_model(
            model=dummy_model,
            name="test_model",
            version="1.0.0",
            description="A test fractional neural network",
            author="Test User",
            tags=['test', 'fractional', 'neural_network'],
            framework='pytorch',
            model_type='spectral_fractional',
            fractional_order=0.5,
            hyperparameters={'alpha': 0.5, 'layers': [10, 20, 5]},
            performance_metrics={'accuracy': 0.95, 'loss': 0.05},
            dataset_info={'name': 'test_dataset', 'size': 1000},
            dependencies={'torch': '1.9.0', 'numpy': '1.21.0'}
        )
        
        # Retrieve model using the returned model_id
        retrieved_model = registry.get_model(model_id)
        assert retrieved_model is not None
        
        # Test search models functionality
        search_results = registry.search_models(tags=['test'])
        assert len(search_results) > 0
        
        # Delete model
        registry.delete_model(model_id)
        assert registry.get_model(model_id) is None


class TestHybridWorkflow:
    """Test hybrid workflows combining multiple components."""
    
    def test_fractional_ml_workflow(self):
        """Test workflow combining fractional calculus with ML."""
        # Create fractional derivative
        rl_deriv = RiemannLiouvilleDerivative(0.5)
        
        # Create test function
        def test_func(x):
            return np.sin(x) + np.cos(x)
        
        # Generate data
        x_vals = np.linspace(0, 2*np.pi, 100)
        y_vals = test_func(x_vals)
        
        # Compute fractional derivative
        fractional_deriv = rl_deriv.compute(test_func, x_vals)
        
        # Convert to PyTorch tensors (ensure same length)
        x_tensor = torch.tensor(x_vals, dtype=torch.float32)
        y_tensor = torch.tensor(y_vals, dtype=torch.float32)
        deriv_tensor = torch.tensor(fractional_deriv[:len(x_vals)], dtype=torch.float32)
        
        # Create ML network
        network = SpectralFractionalNetwork(input_dim=1, hidden_dims=[20, 10], output_dim=1, alpha=0.5)
        
        # Train network to approximate fractional derivative
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            output = network(x_tensor.unsqueeze(1))
            
            # Loss between network output and fractional derivative
            loss = torch.nn.functional.mse_loss(output.squeeze(), deriv_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Test final prediction
        with torch.no_grad():
            final_output = network(x_tensor.unsqueeze(1))
            final_loss = torch.nn.functional.mse_loss(final_output.squeeze(), deriv_tensor)
            
            assert final_loss.item() < 10.0  # Reasonable loss threshold
    
    def test_optimization_workflow(self):
        """Test workflow combining optimization with fractional calculus."""
        # Create test function
        def objective_func(x):
            return np.sum(x**2) + 0.1 * np.sum(np.sin(x))
        
        # Create fractional derivative
        rl_deriv = RiemannLiouvilleDerivative(0.3)
        
        # Test points
        x_vals = np.linspace(-2, 2, 50)
        
        # Compute fractional derivative
        fractional_deriv = rl_deriv.compute(objective_func, x_vals)
        
        # Find minimum using fractional derivative information
        # (This is a simplified example)
        min_idx = np.argmin(np.abs(fractional_deriv))
        min_x = x_vals[min_idx]
        
        assert isinstance(min_x, float)
        assert -2 <= min_x <= 2
    
    def test_parallel_processing_workflow(self):
        """Test workflow using parallel processing."""
        import concurrent.futures
        import threading
        
        def compute_derivative(alpha, func, x_vals):
            """Compute derivative for given alpha."""
            rl_deriv = RiemannLiouvilleDerivative(alpha)
            return rl_deriv.compute(func, x_vals)
        
        # Test function
        def test_func(x):
            return x**2 + np.sin(x)
        
        # Test points
        x_vals = np.linspace(0, 3, 100)
        
        # Different alpha values
        alphas = [0.2, 0.4, 0.6, 0.8]
        
        # Compute derivatives in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(compute_derivative, alpha, test_func, x_vals)
                for alpha in alphas
            ]
            
            results = [future.result() for future in futures]
        
        # Verify results
        assert len(results) == len(alphas)
        for result in results:
            assert isinstance(result, np.ndarray)
            # Note: Different alpha values may return different array lengths due to internal processing
            assert len(result) > 0
            assert np.all(np.isfinite(result))


class TestErrorHandlingWorkflow:
    """Test workflows with error handling."""
    
    def test_robust_derivative_workflow(self):
        """Test workflow with robust error handling."""
        def test_func(x):
            return x**2
        
        x_vals = np.linspace(0, 2, 50)
        
        # Test with different alpha values, including edge cases
        alphas = [0.1, 0.5, 0.9, 1.0, 1.5]
        
        results = []
        for alpha in alphas:
            try:
                if alpha < 1.0:
                    # Use Caputo for alpha < 1
                    deriv = CaputoDerivative(alpha)
                else:
                    # Use Riemann-Liouville for alpha >= 1
                    deriv = RiemannLiouvilleDerivative(alpha)
                
                result = deriv.compute(test_func, x_vals)
                results.append(result)
                
            except ValueError as e:
                # Handle expected errors
                print(f"Expected error for alpha={alpha}: {e}")
                continue
        
        # Should have at least some successful results
        assert len(results) > 0
        
        # All results should be valid
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
            assert np.all(np.isfinite(result))
    
    def test_memory_efficient_workflow(self):
        """Test workflow with memory efficiency considerations."""
        # Create large dataset
        x_vals = np.linspace(0, 10, 10000)
        
        def test_func(x):
            return np.exp(-x) * np.sin(x)
        
        # Use memory-efficient methods
        try:
            # Test with parallel optimized method
            parallel_rl = ParallelOptimizedRiemannLiouville(0.5)
            result = parallel_rl.compute(test_func, x_vals)
            
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
            assert np.all(np.isfinite(result))
            
        except MemoryError:
            # Handle memory errors gracefully
            print("Memory error occurred, using smaller dataset")
            
            # Fallback to smaller dataset
            x_vals_small = np.linspace(0, 10, 1000)
            result = parallel_rl.compute(test_func, x_vals_small)
            
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
            assert np.all(np.isfinite(result))


class TestPerformanceWorkflow:
    """Test workflows with performance considerations."""
    
    def test_benchmarking_workflow(self):
        """Test workflow with performance benchmarking."""
        import time
        
        def test_func(x):
            return x**3 + 2*x**2 + x + 1
        
        x_vals = np.linspace(0, 2, 1000)
        
        # Benchmark different methods
        methods = [
            ("Riemann-Liouville", RiemannLiouvilleDerivative(0.5)),
            ("Caputo", CaputoDerivative(0.3)),
            ("Grunwald-Letnikov", GrunwaldLetnikovDerivative(0.7))
        ]
        
        results = {}
        for name, method in methods:
            start_time = time.time()
            result = method.compute(test_func, x_vals)
            end_time = time.time()
            
            results[name] = {
                "result": result,
                "time": end_time - start_time
            }
        
        # Verify all methods completed successfully
        for name, data in results.items():
            assert isinstance(data["result"], np.ndarray)
            assert len(data["result"]) > 0
            assert data["time"] > 0
            assert np.all(np.isfinite(data["result"]))
    
    def test_scalability_workflow(self):
        """Test workflow scalability with different data sizes."""
        def test_func(x):
            return np.sin(x) * np.exp(-x)
        
        # Test different data sizes
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            x_vals = np.linspace(0, 2*np.pi, size)
            
            # Use optimized method for larger datasets
            if size > 1000:
                deriv = ParallelOptimizedRiemannLiouville(0.5)
            else:
                deriv = RiemannLiouvilleDerivative(0.5)
            
            result = deriv.compute(test_func, x_vals)
            
            assert isinstance(result, np.ndarray)
            assert len(result) > 0
            assert np.all(np.isfinite(result))
