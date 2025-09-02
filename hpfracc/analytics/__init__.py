"""
Analytics module for hpfracc library.

This module provides comprehensive tracking and analysis of:
- Usage patterns and estimator popularity
- Performance metrics and resource usage
- Error analysis and reliability scores
- Workflow insights and usage sequences
"""

from .usage_tracker import UsageTracker
from .performance_monitor import PerformanceMonitor
from .error_analyzer import ErrorAnalyzer
from .workflow_insights import WorkflowInsights
from .analytics_manager import AnalyticsManager

__all__ = [
    'UsageTracker',
    'PerformanceMonitor',
    'ErrorAnalyzer',
    'WorkflowInsights',
    'AnalyticsManager'
]
