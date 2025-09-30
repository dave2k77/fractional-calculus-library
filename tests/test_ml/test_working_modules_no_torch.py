#!/usr/bin/env python3
"""Quality tests for working ML modules without direct PyTorch imports."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Test imports without PyTorch
def test_working_modules_import():
    """Test that working modules can be imported."""
    working_modules = [
        'hpfracc.ml.gpu_optimization',
        'hpfracc.ml.variance_aware_training', 
        'hpfracc.ml.neural_ode',
        'hpfracc.ml.layers',
        'hpfracc.ml.data',
        'hpfracc.ml.registry',
        'hpfracc.ml.workflow',
        'hpfracc.ml.training'
    ]
    
    imported_modules = {}
    
    for module_name in working_modules:
        try:
            module = __import__(module_name)
            imported_modules[module_name] = module
            print(f"✓ {module_name} imported successfully")
        except Exception as e:
            print(f"✗ {module_name} failed to import: {str(e)[:100]}...")
            pytest.fail(f"Failed to import {module_name}")
    
    assert len(imported_modules) == 8


def test_gpu_optimization_module():
    """Test GPU optimization module functionality."""
    try:
        from hpfracc.ml.gpu_optimization import GPUOptimizer, GPUConfig, get_gpu_info
        
        # Test GPUConfig
        config = GPUConfig()
        assert config.device == "auto"
        assert config.memory_fraction == 1.0
        
        # Test custom config
        custom_config = GPUConfig(device="cuda:0", memory_fraction=0.8)
        assert custom_config.device == "cuda:0"
        assert custom_config.memory_fraction == 0.8
        
        # Test GPUOptimizer initialization
        optimizer = GPUOptimizer()
        assert optimizer is not None
        
        # Test with config
        optimizer_with_config = GPUOptimizer(config=custom_config)
        assert optimizer_with_config.config == custom_config
        
        # Test get_gpu_info (will use mocks)
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                gpu_info = get_gpu_info()
                assert isinstance(gpu_info, dict)
                
        print("✓ GPU optimization module tests passed")
        
    except Exception as e:
        pytest.fail(f"GPU optimization module test failed: {str(e)}")


def test_variance_aware_training_module():
    """Test variance aware training module functionality."""
    try:
        from hpfracc.ml.variance_aware_training import (
            VarianceAwareTrainer, VarianceMonitor, AdaptiveSampling, SeedManager
        )
        
        # Test VarianceMonitor
        monitor = VarianceMonitor()
        assert monitor is not None
        
        # Test variance update
        sample_losses = [0.5, 0.6, 0.4, 0.7, 0.3]
        monitor.update_variance(sample_losses)
        assert len(monitor.variance_history) == 1
        
        # Test summary
        summary = monitor.get_summary()
        assert isinstance(summary, dict)
        
        # Test AdaptiveSampling
        sampling = AdaptiveSampling(min_k=5, max_k=50, initial_k=10)
        assert sampling.min_k == 5
        assert sampling.max_k == 50
        assert sampling.current_k == 10
        
        # Test sampling adjustment
        sampling.adjust_sampling(variance_high=True)
        assert sampling.current_k >= 10
        
        # Test SeedManager
        seed_manager = SeedManager(base_seed=42)
        assert seed_manager.base_seed == 42
        
        seed1 = seed_manager.get_next_seed()
        seed2 = seed_manager.get_next_seed()
        assert seed1 == 42
        assert seed2 != seed1
        
        # Test VarianceAwareTrainer with mock model
        with patch('torch.nn.Module') as mock_model:
            trainer = VarianceAwareTrainer(mock_model)
            assert trainer is not None
            assert trainer.model == mock_model
            
        print("✓ Variance aware training module tests passed")
        
    except Exception as e:
        pytest.fail(f"Variance aware training module test failed: {str(e)}")


def test_neural_ode_module():
    """Test neural ODE module functionality."""
    try:
        from hpfracc.ml.neural_ode import NeuralODE, ODEConfig
        
        # Test ODEConfig
        config = ODEConfig()
        assert config is not None
        
        # Test NeuralODE with mock components
        with patch('torch.nn.Module') as mock_model:
            with patch('torch.nn.Module') as mock_ode_func:
                neural_ode = NeuralODE(ode_func=mock_ode_func, config=config)
                assert neural_ode is not None
                assert neural_ode.ode_func == mock_ode_func
                
        print("✓ Neural ODE module tests passed")
        
    except Exception as e:
        pytest.fail(f"Neural ODE module test failed: {str(e)}")


def test_layers_module():
    """Test layers module functionality."""
    try:
        from hpfracc.ml.layers import (
            FractionalConv1D, FractionalConv2D, FractionalLSTM,
            LayerConfig
        )
        
        # Test LayerConfig
        config = LayerConfig()
        assert config is not None
        
        # Test layer initialization with mock parameters
        conv1d = FractionalConv1D(3, 16, 3)
        assert conv1d is not None
        
        conv2d = FractionalConv2D(3, 16, 3)
        assert conv2d is not None
        
        lstm = FractionalLSTM(10, 16, 2)
        assert lstm is not None
        
        print("✓ Layers module tests passed")
        
    except Exception as e:
        pytest.fail(f"Layers module test failed: {str(e)}")


def test_data_module():
    """Test data module functionality."""
    try:
        from hpfracc.ml.data import FractionalTensorDataset, DataConfig
        
        # Test DataConfig
        config = DataConfig()
        assert config is not None
        
        # Test dataset creation with mock data
        with patch('torch.tensor') as mock_tensor:
            mock_tensor.return_value = np.array([1, 2, 3, 4, 5])
            
            data = mock_tensor()
            targets = mock_tensor()
            dataset = FractionalTensorDataset(data, targets)
            assert dataset is not None
            
        print("✓ Data module tests passed")
        
    except Exception as e:
        pytest.fail(f"Data module test failed: {str(e)}")


def test_registry_module():
    """Test registry module functionality."""
    try:
        from hpfracc.ml.registry import ModelRegistry, DeploymentStatus, ModelMetadata
        
        # Test DeploymentStatus enum
        assert hasattr(DeploymentStatus, 'DEVELOPMENT')
        assert hasattr(DeploymentStatus, 'STAGING')
        assert hasattr(DeploymentStatus, 'PRODUCTION')
        
        # Test ModelMetadata
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            description="Test model"
        )
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        
        # Test ModelRegistry
        registry = ModelRegistry()
        assert registry is not None
        
        print("✓ Registry module tests passed")
        
    except Exception as e:
        pytest.fail(f"Registry module test failed: {str(e)}")


def test_workflow_module():
    """Test workflow module functionality."""
    try:
        from hpfracc.ml.workflow import QualityMetric, QualityThreshold
        
        # Test QualityMetric enum
        assert hasattr(QualityMetric, 'ACCURACY')
        assert hasattr(QualityMetric, 'PRECISION')
        assert hasattr(QualityMetric, 'RECALL')
        
        # Test QualityThreshold
        threshold = QualityThreshold(
            metric=QualityMetric.ACCURACY,
            min_value=0.8
        )
        assert threshold.metric == QualityMetric.ACCURACY
        assert threshold.min_value == 0.8
        
        print("✓ Workflow module tests passed")
        
    except Exception as e:
        pytest.fail(f"Workflow module test failed: {str(e)}")


def test_training_module():
    """Test training module functionality."""
    try:
        from hpfracc.ml.training import FractionalTrainer, TrainingConfig
        
        # Test TrainingConfig
        config = TrainingConfig()
        assert config is not None
        
        # Test FractionalTrainer with mock model
        with patch('torch.nn.Module') as mock_model:
            trainer = FractionalTrainer(model=mock_model, config=config)
            assert trainer is not None
            assert trainer.model == mock_model
            
        print("✓ Training module tests passed")
        
    except Exception as e:
        pytest.fail(f"Training module test failed: {str(e)}")


def test_module_edge_cases():
    """Test edge cases for working modules."""
    try:
        # Test with None parameters
        from hpfracc.ml.gpu_optimization import GPUOptimizer
        from hpfracc.ml.variance_aware_training import VarianceAwareTrainer
        
        # Test GPUOptimizer with None config
        optimizer = GPUOptimizer(config=None)
        assert optimizer is not None
        
        # Test VarianceAwareTrainer with None optimizer
        with patch('torch.nn.Module') as mock_model:
            trainer = VarianceAwareTrainer(mock_model, optimizer=None)
            assert trainer is not None
            
        print("✓ Edge cases tests passed")
        
    except Exception as e:
        pytest.fail(f"Edge cases test failed: {str(e)}")


def test_module_configurations():
    """Test different configurations for working modules."""
    try:
        from hpfracc.ml.gpu_optimization import GPUConfig, GPUOptimizer
        from hpfracc.ml.variance_aware_training import VarianceAwareTrainer
        from hpfracc.ml.workflow import QualityMetric, QualityThreshold
        
        # Test different GPU configurations
        configs = [
            GPUConfig(device="auto"),
            GPUConfig(device="cuda:0"),
            GPUConfig(device="cpu"),
            GPUConfig(memory_fraction=0.5),
            GPUConfig(allow_growth=False),
            GPUConfig(optimize_memory=False)
        ]
        
        for config in configs:
            optimizer = GPUOptimizer(config=config)
            assert optimizer.config == config
            
        # Test different quality thresholds
        thresholds = [
            QualityThreshold(QualityMetric.ACCURACY, 0.8),
            QualityThreshold(QualityMetric.PRECISION, 0.7),
            QualityThreshold(QualityMetric.RECALL, 0.9),
            QualityThreshold(QualityMetric.F1_SCORE, 0.85)
        ]
        
        for threshold in thresholds:
            assert threshold.metric is not None
            assert threshold.min_value > 0
            
        print("✓ Configuration tests passed")
        
    except Exception as e:
        pytest.fail(f"Configuration test failed: {str(e)}")


def test_module_integration():
    """Test integration between working modules."""
    try:
        from hpfracc.ml.gpu_optimization import GPUOptimizer, GPUConfig
        from hpfracc.ml.variance_aware_training import VarianceAwareTrainer
        from hpfracc.ml.registry import ModelRegistry
        from hpfracc.ml.workflow import QualityMetric
        
        # Test integration scenario
        gpu_config = GPUConfig(device="cuda:0")
        gpu_optimizer = GPUOptimizer(config=gpu_config)
        
        with patch('torch.nn.Module') as mock_model:
            trainer = VarianceAwareTrainer(mock_model)
            
            registry = ModelRegistry()
            
            # Test that components work together
            assert gpu_optimizer is not None
            assert trainer is not None
            assert registry is not None
            
        print("✓ Integration tests passed")
        
    except Exception as e:
        pytest.fail(f"Integration test failed: {str(e)}")


def test_module_error_handling():
    """Test error handling in working modules."""
    try:
        from hpfracc.ml.gpu_optimization import GPUOptimizer
        from hpfracc.ml.variance_aware_training import VarianceMonitor, AdaptiveSampling
        
        # Test error handling scenarios
        
        # Test VarianceMonitor with invalid data
        monitor = VarianceMonitor()
        monitor.update_variance([])  # Empty list
        assert len(monitor.variance_history) == 1
        
        monitor.update_variance([0.5])  # Single value
        assert len(monitor.variance_history) == 2
        
        # Test AdaptiveSampling boundary conditions
        sampling = AdaptiveSampling(min_k=5, max_k=50, initial_k=10)
        
        # Test at boundaries
        sampling.current_k = sampling.min_k
        sampling.adjust_sampling(variance_high=False)
        assert sampling.current_k >= sampling.min_k
        
        sampling.current_k = sampling.max_k
        sampling.adjust_sampling(variance_high=True)
        assert sampling.current_k <= sampling.max_k
        
        print("✓ Error handling tests passed")
        
    except Exception as e:
        pytest.fail(f"Error handling test failed: {str(e)}")


def test_module_performance():
    """Test performance characteristics of working modules."""
    try:
        from hpfracc.ml.variance_aware_training import VarianceMonitor, AdaptiveSampling
        import time
        
        # Test VarianceMonitor performance
        monitor = VarianceMonitor()
        
        start_time = time.time()
        for i in range(100):
            sample_data = np.random.random(100)
            monitor.update_variance(sample_data)
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 5.0
        assert len(monitor.variance_history) == 100
        
        # Test AdaptiveSampling performance
        sampling = AdaptiveSampling()
        
        start_time = time.time()
        for i in range(1000):
            sampling.adjust_sampling(variance_high=(i % 2 == 0))
        end_time = time.time()
        
        # Should complete very quickly
        assert (end_time - start_time) < 1.0
        
        print("✓ Performance tests passed")
        
    except Exception as e:
        pytest.fail(f"Performance test failed: {str(e)}")
