"""
Performance Testing Module

This module provides comprehensive performance regression testing for the
fractional calculus library, ensuring that optimization improvements are
maintained and performance doesn't degrade over time.
"""

from .performance_config import get_performance_config, PerformanceConfig
from .performance_monitor import get_performance_monitor, get_performance_profiler, PerformanceMonitor, PerformanceProfiler
from .test_performance_regression import (
    PerformanceBaseline,
    TestDerivativePerformanceRegression,
    TestOptimizedMethodsPerformanceRegression,
    TestNeuralNetworkPerformanceRegression,
    TestTensorOperationsPerformanceRegression,
    TestMemoryUsageRegression,
    TestGPUPerformanceRegression,
    TestScalabilityRegression,
    TestPerformanceRegressionSuite
)

__all__ = [
    'get_performance_config',
    'PerformanceConfig',
    'get_performance_monitor',
    'get_performance_profiler',
    'PerformanceMonitor',
    'PerformanceProfiler',
    'PerformanceBaseline',
    'TestDerivativePerformanceRegression',
    'TestOptimizedMethodsPerformanceRegression',
    'TestNeuralNetworkPerformanceRegression',
    'TestTensorOperationsPerformanceRegression',
    'TestMemoryUsageRegression',
    'TestGPUPerformanceRegression',
    'TestScalabilityRegression',
    'TestPerformanceRegressionSuite'
]

