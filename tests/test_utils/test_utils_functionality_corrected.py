"""
Corrected comprehensive test suite for the utilities module.

This module tests all functionality in the hpfracc.utils module using the actual API.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import gc
from unittest.mock import patch, MagicMock

# Import utilities module components
from hpfracc.utils import (
    ErrorAnalyzer,
    ConvergenceAnalyzer,
    ValidationFramework,
    compute_error_metrics,
    analyze_convergence,
    validate_solution,
    MemoryManager,
    CacheManager,
    optimize_memory_usage,
    clear_cache,
    get_memory_usage,
    PlotManager,
    create_comparison_plot,
    plot_convergence,
    plot_error_analysis,
    save_plot,
    setup_plotting_style,
)


class TestErrorAnalyzer:
    """Test ErrorAnalyzer functionality."""

    def test_error_analyzer_creation(self):
        """Test ErrorAnalyzer creation."""
        analyzer = ErrorAnalyzer()
        assert analyzer.tolerance == 1e-10
        
        analyzer_custom = ErrorAnalyzer(tolerance=1e-6)
        assert analyzer_custom.tolerance == 1e-6

    def test_absolute_error(self):
        """Test absolute error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        error = analyzer.absolute_error(numerical, analytical)
        expected = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_almost_equal(error, expected)

    def test_relative_error(self):
        """Test relative error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        error = analyzer.relative_error(numerical, analytical)
        # Relative error = |numerical - analytical| / |analytical|
        expected = np.array([0.1/1.1, 0.1/1.9, 0.1/3.1])
        np.testing.assert_array_almost_equal(error, expected)

    def test_relative_error_zero_analytical(self):
        """Test relative error with zero analytical values."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 0.0, 2.0])
        analytical = np.array([0.0, 0.0, 0.0])
        
        error = analyzer.relative_error(numerical, analytical)
        # Should handle division by zero gracefully
        assert np.all(np.isfinite(error))

    def test_l1_error(self):
        """Test L1 error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        l1_error = analyzer.l1_error(numerical, analytical)
        expected = np.mean(np.abs(numerical - analytical))
        assert abs(l1_error - expected) < 1e-10

    def test_l2_error(self):
        """Test L2 error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        l2_error = analyzer.l2_error(numerical, analytical)
        expected = np.sqrt(np.mean((numerical - analytical) ** 2))
        assert abs(l2_error - expected) < 1e-10

    def test_linf_error(self):
        """Test L-infinity error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        linf_error = analyzer.linf_error(numerical, analytical)
        expected = np.max(np.abs(numerical - analytical))
        assert abs(linf_error - expected) < 1e-10

    def test_compute_all_errors(self):
        """Test comprehensive error metrics computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0, 4.0])
        analytical = np.array([1.1, 1.9, 3.1, 3.9])
        
        metrics = analyzer.compute_all_errors(numerical, analytical)
        
        assert 'l1' in metrics
        assert 'l2' in metrics
        assert 'linf' in metrics
        assert 'mean_absolute' in metrics
        assert 'mean_relative' in metrics
        
        # Check that all metrics are finite
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value))
            else:
                assert np.isfinite(value)


class TestConvergenceAnalyzer:
    """Test ConvergenceAnalyzer functionality."""

    def test_convergence_analyzer_creation(self):
        """Test ConvergenceAnalyzer creation."""
        analyzer = ConvergenceAnalyzer()
        assert hasattr(analyzer, 'analyze_convergence')

    def test_analyze_convergence(self):
        """Test convergence analysis."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with known convergence data
        grid_sizes = np.array([10, 20, 40, 80])
        methods = ['method1', 'method2']
        errors = {
            'method1': np.array([1.0, 0.5, 0.25, 0.125]),
            'method2': np.array([1.0, 0.25, 0.0625, 0.015625])
        }
        
        analysis = analyzer.analyze_convergence(methods, grid_sizes, errors)
        
        assert isinstance(analysis, dict)
        # Should contain convergence rate information
        assert len(analysis) > 0


class TestValidationFramework:
    """Test ValidationFramework functionality."""

    def test_validation_framework_creation(self):
        """Test ValidationFramework creation."""
        framework = ValidationFramework()
        assert hasattr(framework, 'validate_method')

    def test_validate_method(self):
        """Test method validation."""
        framework = ValidationFramework()
        
        # Create mock functions
        def mock_method(x):
            return x ** 2
        
        def mock_analytical(x):
            return x ** 2
        
        test_cases = [{'input': 1.0, 'expected': 1.0}]
        
        result = framework.validate_method(mock_method, mock_analytical, test_cases)
        assert isinstance(result, dict)


class TestMemoryManager:
    """Test MemoryManager functionality."""

    def test_memory_manager_creation(self):
        """Test MemoryManager creation."""
        manager = MemoryManager()
        assert hasattr(manager, 'memory_limit_gb')
        assert hasattr(manager, 'memory_history')
        assert hasattr(manager, 'peak_memory')

    def test_memory_manager_with_limit(self):
        """Test MemoryManager with memory limit."""
        manager = MemoryManager(memory_limit_gb=1.0)
        assert manager.memory_limit_gb == 1.0

    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        manager = MemoryManager()
        usage = manager.get_memory_usage()
        
        assert 'rss' in usage
        assert 'vms' in usage
        assert 'percent' in usage
        assert 'available' in usage
        assert 'total' in usage
        
        # Check that values are reasonable
        assert usage['rss'] >= 0
        assert usage['vms'] >= 0
        assert 0 <= usage['percent'] <= 100
        assert usage['available'] >= 0
        assert usage['total'] > 0


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_cache_manager_creation(self):
        """Test CacheManager creation."""
        cache = CacheManager()
        assert hasattr(cache, 'cache')
        assert hasattr(cache, 'max_size')

    def test_cache_manager_with_size(self):
        """Test CacheManager with custom size."""
        cache = CacheManager(max_size=100)
        assert cache.max_size == 100

    def test_cache_operations(self):
        """Test cache operations."""
        cache = CacheManager()
        
        # Test setting and getting
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Test cache miss
        assert cache.get('nonexistent') is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CacheManager()
        
        # Add items
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Clear cache
        cache.clear()
        assert cache.get('key1') is None
        assert cache.get('key2') is None


class TestPlotManager:
    """Test PlotManager functionality."""

    def test_plot_manager_creation(self):
        """Test PlotManager creation."""
        manager = PlotManager()
        assert hasattr(manager, 'style')
        assert hasattr(manager, 'figsize')

    def test_plot_manager_custom(self):
        """Test PlotManager with custom parameters."""
        manager = PlotManager(style='scientific', figsize=(12, 8))
        assert manager.style == 'scientific'
        assert manager.figsize == (12, 8)

    def test_setup_plotting_style(self):
        """Test plotting style setup."""
        manager = PlotManager()
        
        # Test different styles
        for style in ['default', 'scientific', 'presentation']:
            manager.setup_plotting_style(style)
            # Style should be set (though the actual style might not change)
            assert hasattr(manager, 'style')

    def test_plot_convergence(self):
        """Test convergence plotting."""
        manager = PlotManager()
        
        # Create convergence data
        grid_sizes = [10, 20, 40, 80]
        errors = {
            'method1': [1.0, 0.5, 0.25, 0.125],
            'method2': [1.0, 0.25, 0.0625, 0.015625]
        }
        
        fig = manager.plot_convergence(grid_sizes, errors)
        assert fig is not None
        
        plt.close(fig)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_error_metrics(self):
        """Test compute_error_metrics function."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        metrics = compute_error_metrics(numerical, analytical)
        
        assert 'l1' in metrics
        assert 'l2' in metrics
        assert 'linf' in metrics
        assert 'mean_absolute' in metrics
        assert 'mean_relative' in metrics

    def test_analyze_convergence(self):
        """Test analyze_convergence function."""
        grid_sizes = [10, 20, 40, 80]
        errors = {
            'method1': [1.0, 0.5, 0.25, 0.125],
            'method2': [1.0, 0.25, 0.0625, 0.015625]
        }
        
        analysis = analyze_convergence(grid_sizes, errors)
        assert isinstance(analysis, dict)

    def test_validate_solution(self):
        """Test validate_solution function."""
        def mock_method(x):
            return x ** 2
        
        def mock_analytical(x):
            return x ** 2
        
        test_cases = [{'input': 1.0, 'expected': 1.0}]
        
        result = validate_solution(mock_method, mock_analytical, test_cases)
        assert isinstance(result, dict)

    def test_get_memory_usage(self):
        """Test get_memory_usage function."""
        usage = get_memory_usage()
        
        assert 'rss' in usage
        assert 'vms' in usage
        assert 'percent' in usage
        assert 'available' in usage
        assert 'total' in usage

    def test_optimize_memory_usage(self):
        """Test optimize_memory_usage function."""
        @optimize_memory_usage
        def test_func():
            return np.random.rand(1000)
        
        result = test_func()
        # The decorated function should return the result
        assert isinstance(result, np.ndarray)

    def test_clear_cache(self):
        """Test clear_cache function."""
        # This should not raise an exception
        clear_cache()

    def test_create_comparison_plot(self):
        """Test create_comparison_plot function."""
        x = np.linspace(0, 1, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        data = {
            'Method 1': y1,
            'Method 2': y2
        }
        
        fig, ax = create_comparison_plot(x, data, 'Test Comparison')
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_plot_convergence(self):
        """Test plot_convergence function."""
        grid_sizes = [10, 20, 40, 80]
        errors = {
            'method1': [1.0, 0.5, 0.25, 0.125],
            'method2': [1.0, 0.25, 0.0625, 0.015625]
        }
        
        fig, ax = plot_convergence(grid_sizes, errors)
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_plot_error_analysis(self):
        """Test plot_error_analysis function."""
        x = np.array([1.0, 2.0, 3.0])
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        fig, ax = plot_error_analysis(x, numerical, analytical)
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_save_plot(self):
        """Test save_plot function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple plot
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            
            # Save plot
            filename = os.path.join(temp_dir, 'test_plot.png')
            save_plot(fig, filename)
            
            # Check that file was created
            assert os.path.exists(filename)
            
            plt.close(fig)

    def test_setup_plotting_style(self):
        """Test setup_plotting_style function."""
        # This should not raise an exception
        setup_plotting_style('default')
        setup_plotting_style('scientific')
        setup_plotting_style('presentation')


class TestIntegration:
    """Test integration between utility components."""

    def test_error_analysis_with_memory_management(self):
        """Test error analysis with memory management."""
        # Create error analyzer
        analyzer = ErrorAnalyzer()
        
        # Create memory manager
        memory_manager = MemoryManager()
        
        # Get initial memory usage
        initial_memory = memory_manager.get_memory_usage()
        
        # Perform error analysis
        numerical = np.random.rand(1000)
        analytical = numerical + 0.01 * np.random.rand(1000)
        
        metrics = analyzer.compute_all_errors(numerical, analytical)
        
        # Check that memory usage is reasonable
        final_memory = memory_manager.get_memory_usage()
        memory_increase = final_memory['rss'] - initial_memory['rss']
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 0.1

    def test_plotting_with_error_analysis(self):
        """Test plotting with error analysis."""
        # Create error analyzer
        analyzer = ErrorAnalyzer()
        
        # Create plot manager
        plot_manager = PlotManager()
        
        # Generate test data
        numerical = np.random.rand(100)
        analytical = numerical + 0.01 * np.random.rand(100)
        
        # Compute error metrics
        metrics = analyzer.compute_all_errors(numerical, analytical)
        
        # Create error analysis plot
        x = np.linspace(0, 1, len(numerical))
        fig, ax = plot_error_analysis(x, numerical, analytical)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_comprehensive_workflow(self):
        """Test comprehensive workflow using all utilities."""
        # Create managers
        error_analyzer = ErrorAnalyzer()
        memory_manager = MemoryManager()
        plot_manager = PlotManager()
        
        # Generate test data
        x = np.linspace(0, 1, 100)
        numerical = np.sin(x) + 0.01 * np.random.rand(100)
        analytical = np.sin(x)
        
        # Perform error analysis
        metrics = error_analyzer.compute_all_errors(numerical, analytical)
        
        # Check memory usage
        memory_usage = memory_manager.get_memory_usage()
        assert memory_usage['rss'] > 0
        
        # Create visualization
        x = np.linspace(0, 1, len(numerical))
        fig, ax = plot_error_analysis(x, numerical, analytical)
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__])
