"""
Tests for utility modules.
"""

import numpy as np
import pytest
from hpfracc.utils.error_analysis import (
    ErrorAnalyzer,
    ConvergenceAnalyzer,
    ValidationFramework,
)
from hpfracc.utils.memory_management import MemoryManager, CacheManager
from hpfracc.utils.plotting import PlotManager
import matplotlib.pyplot as plt


class TestErrorAnalysis:
    """Test error analysis utilities."""

    def test_error_analyzer_basic(self):
        """Test basic error analyzer functionality."""
        analyzer = ErrorAnalyzer()

        # Test data
        numerical = np.array([1.0, 2.0, 3.0, 4.0])
        analytical = np.array([1.1, 1.9, 3.1, 4.1])

        # Test absolute error
        abs_error = analyzer.absolute_error(numerical, analytical)
        assert np.allclose(abs_error, [0.1, 0.1, 0.1, 0.1])

        # Test relative error
        rel_error = analyzer.relative_error(numerical, analytical)
        expected_rel = [0.1 / 1.1, 0.1 / 1.9, 0.1 / 3.1, 0.1 / 4.1]
        assert np.allclose(rel_error, expected_rel)

        # Test error norms
        assert abs(analyzer.l1_error(numerical, analytical) - 0.1) < 1e-10
        assert abs(analyzer.l2_error(numerical, analytical) - 0.1) < 1e-10
        assert abs(analyzer.linf_error(numerical, analytical) - 0.1) < 1e-10

        # Test all errors
        all_errors = analyzer.compute_all_errors(numerical, analytical)
        assert "l1" in all_errors
        assert "l2" in all_errors
        assert "linf" in all_errors

    def test_convergence_analyzer(self):
        """Test convergence analyzer functionality."""
        analyzer = ConvergenceAnalyzer()

        # Test data
        grid_sizes = [10, 20, 40, 80]
        errors = [1.0, 0.5, 0.25, 0.125]  # Second order convergence

        # Test convergence rate
        rate = analyzer.compute_convergence_rate(grid_sizes, errors)
        assert abs(rate - -1.0) < 0.1  # Should be close to -1.0 (log-log slope)

        # Test convergence analysis
        error_dict = {"l2": errors}
        rates = analyzer.analyze_convergence(["l2"], grid_sizes, error_dict)
        assert "l2" in rates["convergence_rates"]

    def test_validation_framework(self):
        """Test validation framework functionality."""
        framework = ValidationFramework()

        # Mock functions
        def mock_method(**kwargs):
            return np.array([1.0, 2.0, 3.0])

        def mock_analytical(**kwargs):
            return np.array([1.1, 1.9, 3.1])

        # Test cases
        test_cases = [{"params": {"N": 10}}]

        # Validate method
        results = framework.validate_method(mock_method, mock_analytical, test_cases)

        assert "test_cases" in results
        assert "convergence_study" in results
        assert "overall_summary" in results
        assert len(results["test_cases"]) == 1
        assert results["test_cases"][0]["success"] is True


class TestMemoryManagement:
    """Test memory management utilities."""

    def test_memory_manager_basic(self):
        """Test basic memory manager functionality."""
        manager = MemoryManager()

        # Test memory usage
        usage = manager.get_memory_usage()
        assert "rss" in usage
        assert "vms" in usage
        assert "percent" in usage
        assert "available" in usage
        assert "total" in usage

        # Test memory recording
        recorded = manager.record_memory_usage()
        assert len(manager.memory_history) == 1

        # Test memory limit check
        assert manager.check_memory_limit() is True

        # Test memory optimization
        opt_result = manager.optimize_memory_usage()
        assert "before" in opt_result
        assert "after" in opt_result
        assert "freed" in opt_result

    def test_cache_manager_basic(self):
        """Test basic cache manager functionality."""
        cache = CacheManager(max_size=5, max_memory_gb=0.1)

        # Test setting and getting
        cache.set("test_key", np.array([1, 2, 3]))
        result = cache.get("test_key")
        assert np.array_equal(result, np.array([1, 2, 3]))

        # Test cache stats
        stats = cache.get_cache_stats()
        assert stats["size"] == 1
        assert stats["memory_gb"] > 0

        # Test cache clearing
        cache.clear()
        assert cache.get("test_key") is None
        assert cache.get_cache_stats()["size"] == 0


class TestPlotting:
    """Test plotting utilities."""

    def test_plot_manager_basic(self):
        """Test basic plot manager functionality."""
        manager = PlotManager()

        # Test style setup
        manager.setup_plotting_style("default")
        assert manager.style == "default"

        # Test comparison plot
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        data = {"test": (x, y)}
        fig, ax = manager.create_comparison_plot(data, "Test Plot")

        assert fig is not None
        plt.close(fig)

        # Test convergence plot
        grid_sizes = [10, 20, 40]
        errors = {"l2": [1.0, 0.5, 0.25]}
        fig = manager.plot_convergence(grid_sizes, errors, "Test Convergence")
        assert fig is not None
        plt.close(fig)

        # Test error analysis plot
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.1, 1.9, 3.1])
        x = np.array([0, 1, 2])
        fig = manager.plot_error_analysis(x, numerical, analytical, "Test Error")
        assert fig is not None
        plt.close(fig)


def test_convenience_functions():
    """Test convenience functions."""
    from hpfracc.utils.error_analysis import compute_error_metrics, analyze_convergence
    from hpfracc.utils.memory_management import get_memory_usage, clear_cache
    from hpfracc.utils.plotting import setup_plotting_style

    # Test error metrics
    numerical = np.array([1.0, 2.0, 3.0])
    analytical = np.array([1.1, 1.9, 3.1])
    errors = compute_error_metrics(numerical, analytical)
    assert "l1" in errors
    assert "l2" in errors
    assert "linf" in errors

    # Test convergence analysis
    grid_sizes = [10, 20, 40]
    error_dict = {"l2": [1.0, 0.5, 0.25]}
    rates = analyze_convergence(["l2"], grid_sizes, error_dict)
    assert "l2" in rates["convergence_rates"]

    # Test memory usage
    usage = get_memory_usage()
    assert "rss" in usage

    # Test cache clearing
    clear_cache()  # Should not raise an error

    # Test plotting style
    setup_plotting_style("default")  # Should not raise an error
