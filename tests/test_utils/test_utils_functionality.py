"""
Comprehensive test suite for the utilities module.

This module tests all functionality in the hpfracc.utils module including:
- Error analysis and validation
- Memory management and optimization
- Plotting and visualization
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

    def test_mean_squared_error(self):
        """Test mean squared error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        mse = analyzer.mean_squared_error(numerical, analytical)
        expected = np.mean((numerical - analytical) ** 2)
        assert abs(mse - expected) < 1e-10

    def test_root_mean_squared_error(self):
        """Test root mean squared error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        rmse = analyzer.root_mean_squared_error(numerical, analytical)
        expected = np.sqrt(np.mean((numerical - analytical) ** 2))
        assert abs(rmse - expected) < 1e-10

    def test_maximum_error(self):
        """Test maximum error computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        max_error = analyzer.maximum_error(numerical, analytical)
        expected = np.max(np.abs(numerical - analytical))
        assert abs(max_error - expected) < 1e-10

    def test_error_metrics_comprehensive(self):
        """Test comprehensive error metrics computation."""
        analyzer = ErrorAnalyzer()
        numerical = np.array([1.0, 2.0, 3.0, 4.0])
        analytical = np.array([1.1, 1.9, 3.1, 3.9])
        
        metrics = analyzer.compute_error_metrics(numerical, analytical)
        
        assert 'absolute_error' in metrics
        assert 'relative_error' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'max_error' in metrics
        
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
        assert hasattr(analyzer, 'tolerance')

    def test_convergence_rate(self):
        """Test convergence rate computation."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with known convergence sequence
        errors = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        h_values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        
        rate = analyzer.compute_convergence_rate(errors, h_values)
        # Should be approximately 1 (first order convergence)
        assert abs(rate - 1.0) < 0.1

    def test_convergence_analysis(self):
        """Test convergence analysis."""
        analyzer = ConvergenceAnalyzer()
        
        # Test with multiple methods
        methods = ['method1', 'method2']
        h_values = np.array([1.0, 0.5, 0.25, 0.125])
        errors = {
            'method1': np.array([1.0, 0.5, 0.25, 0.125]),
            'method2': np.array([1.0, 0.25, 0.0625, 0.015625])
        }
        
        analysis = analyzer.analyze_convergence(methods, h_values, errors)
        
        assert 'convergence_rates' in analysis
        assert 'best_method' in analysis
        assert 'convergence_orders' in analysis


class TestValidationFramework:
    """Test ValidationFramework functionality."""

    def test_validation_framework_creation(self):
        """Test ValidationFramework creation."""
        framework = ValidationFramework()
        assert hasattr(framework, 'tolerance')

    def test_validate_solution_basic(self):
        """Test basic solution validation."""
        framework = ValidationFramework()
        
        # Test with valid solution
        solution = np.array([1.0, 2.0, 3.0])
        is_valid = framework.validate_solution(solution)
        assert is_valid

    def test_validate_solution_invalid(self):
        """Test solution validation with invalid input."""
        framework = ValidationFramework()
        
        # Test with invalid solution (NaN)
        solution = np.array([1.0, np.nan, 3.0])
        is_valid = framework.validate_solution(solution)
        assert not is_valid

    def test_validate_solution_infinite(self):
        """Test solution validation with infinite values."""
        framework = ValidationFramework()
        
        # Test with infinite solution
        solution = np.array([1.0, np.inf, 3.0])
        is_valid = framework.validate_solution(solution)
        assert not is_valid


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

    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        manager = MemoryManager()
        
        # Start monitoring
        manager.start_monitoring()
        assert manager.monitoring
        
        # Stop monitoring
        manager.stop_monitoring()
        assert not manager.monitoring

    def test_memory_optimization(self):
        """Test memory optimization."""
        manager = MemoryManager()
        
        # Test memory optimization
        result = manager.optimize_memory()
        assert isinstance(result, dict)
        assert 'freed_memory' in result
        assert 'optimization_applied' in result


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
        
        # Test cache size
        assert cache.size() == 1

    def test_cache_eviction(self):
        """Test cache eviction."""
        cache = CacheManager(max_size=2)
        
        # Fill cache beyond limit
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')
        
        # Should evict oldest entry
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = CacheManager()
        
        # Add items
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        assert cache.size() == 2
        
        # Clear cache
        cache.clear()
        assert cache.size() == 0


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
            assert manager.style == style

    def test_create_plot(self):
        """Test plot creation."""
        manager = PlotManager()
        
        # Create simple plot
        x = np.linspace(0, 1, 100)
        y = np.sin(x)
        
        fig, ax = manager.create_plot(x, y, title='Test Plot')
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_create_comparison_plot(self):
        """Test comparison plot creation."""
        manager = PlotManager()
        
        # Create comparison data
        x = np.linspace(0, 1, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        data = {
            'Method 1': (x, y1),
            'Method 2': (x, y2)
        }
        
        fig, ax = manager.create_comparison_plot(data, title='Comparison')
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_error_metrics(self):
        """Test compute_error_metrics function."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        metrics = compute_error_metrics(numerical, analytical)
        
        assert 'absolute_error' in metrics
        assert 'relative_error' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'max_error' in metrics

    def test_analyze_convergence(self):
        """Test analyze_convergence function."""
        methods = ['method1', 'method2']
        h_values = np.array([1.0, 0.5, 0.25])
        errors = {
            'method1': np.array([1.0, 0.5, 0.25]),
            'method2': np.array([1.0, 0.25, 0.0625])
        }
        
        analysis = analyze_convergence(methods, h_values, errors)
        
        assert 'convergence_rates' in analysis
        assert 'best_method' in analysis

    def test_validate_solution(self):
        """Test validate_solution function."""
        # Test valid solution
        solution = np.array([1.0, 2.0, 3.0])
        assert validate_solution(solution)
        
        # Test invalid solution
        solution = np.array([1.0, np.nan, 3.0])
        assert not validate_solution(solution)

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
        result = optimize_memory_usage()
        
        assert isinstance(result, dict)
        assert 'freed_memory' in result
        assert 'optimization_applied' in result

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
            'Method 1': (x, y1),
            'Method 2': (x, y2)
        }
        
        fig, ax = create_comparison_plot(data, title='Test Comparison')
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_plot_convergence(self):
        """Test plot_convergence function."""
        methods = ['method1', 'method2']
        h_values = np.array([1.0, 0.5, 0.25])
        errors = {
            'method1': np.array([1.0, 0.5, 0.25]),
            'method2': np.array([1.0, 0.25, 0.0625])
        }
        
        fig, ax = plot_convergence(methods, h_values, errors)
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)

    def test_plot_error_analysis(self):
        """Test plot_error_analysis function."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        
        fig, ax = plot_error_analysis(numerical, analytical)
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
        
        metrics = analyzer.compute_error_metrics(numerical, analytical)
        
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
        metrics = analyzer.compute_error_metrics(numerical, analytical)
        
        # Create error analysis plot
        fig, ax = plot_manager.create_plot(
            range(len(metrics['absolute_error'])),
            metrics['absolute_error'],
            title='Absolute Error Analysis'
        )
        
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
        metrics = error_analyzer.compute_error_metrics(numerical, analytical)
        
        # Check memory usage
        memory_usage = memory_manager.get_memory_usage()
        assert memory_usage['rss'] > 0
        
        # Create visualization
        fig, ax = plot_manager.create_plot(
            x, metrics['absolute_error'],
            title='Error Analysis',
            xlabel='x',
            ylabel='Absolute Error'
        )
        
        assert fig is not None
        assert ax is not None
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__])
