"""
Simple tests for analytics modules.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch

from hpfracc.analytics.analytics_manager import AnalyticsManager, AnalyticsConfig
from hpfracc.analytics.error_analyzer import ErrorAnalyzer, ErrorEvent, ErrorStats
from hpfracc.analytics.performance_monitor import PerformanceMonitor, PerformanceEvent, PerformanceStats
from hpfracc.analytics.usage_tracker import UsageTracker, UsageEvent, UsageStats
from hpfracc.analytics.workflow_insights import WorkflowInsights


class TestAnalyticsConfig:
    """Test analytics configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AnalyticsConfig()
        
        assert config.enable_usage_tracking is True
        assert config.enable_performance_monitoring is True
        assert config.enable_error_analysis is True
        assert config.enable_workflow_insights is True
        assert config.data_retention_days == 30
        assert config.export_format == "json"
        assert config.generate_reports is True
        assert config.report_output_dir == "analytics_reports"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AnalyticsConfig(
            enable_usage_tracking=False,
            data_retention_days=60,
            export_format="csv"
        )
        
        assert config.enable_usage_tracking is False
        assert config.data_retention_days == 60
        assert config.export_format == "csv"


class TestAnalyticsManager:
    """Test analytics manager functionality."""
    
    def test_initialization(self):
        """Test analytics manager initialization."""
        manager = AnalyticsManager()
        
        assert manager.config is not None
        assert manager.session_id is not None
        assert hasattr(manager, 'usage_tracker')
        assert hasattr(manager, 'performance_monitor')
        assert hasattr(manager, 'error_analyzer')
        assert hasattr(manager, 'workflow_insights')
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AnalyticsConfig(enable_usage_tracking=False)
        manager = AnalyticsManager(config)
        
        assert manager.config == config
        assert manager.config.enable_usage_tracking is False
    
    def test_session_id_generation(self):
        """Test session ID generation."""
        manager1 = AnalyticsManager()
        manager2 = AnalyticsManager()
        
        # Session IDs should be different
        assert manager1.session_id != manager2.session_id
        # Session IDs should be strings
        assert isinstance(manager1.session_id, str)
        assert isinstance(manager2.session_id, str)


class TestErrorEvent:
    """Test error event functionality."""
    
    def test_error_event_creation(self):
        """Test error event creation."""
        event = ErrorEvent(
            timestamp=1234567890.0,
            method_name="test_method",
            estimator_type="riemann_liouville",
            error_type="ValueError",
            error_message="Test error",
            error_traceback="Traceback...",
            error_hash="abc123",
            parameters={"param1": "value1"},
            array_size=100,
            fractional_order=0.5
        )
        
        assert event.timestamp == 1234567890.0
        assert event.method_name == "test_method"
        assert event.estimator_type == "riemann_liouville"
        assert event.error_type == "ValueError"
        assert event.error_message == "Test error"
        assert event.error_hash == "abc123"
        assert event.parameters == {"param1": "value1"}
        assert event.array_size == 100
        assert event.fractional_order == 0.5


class TestErrorStats:
    """Test error statistics functionality."""
    
    def test_error_stats_creation(self):
        """Test error stats creation."""
        stats = ErrorStats(
            method_name="test_method",
            total_errors=10,
            error_rate=0.1,
            common_error_types=[("ValueError", 5), ("TypeError", 3)],
            avg_execution_time_before_error=1.5,
            common_parameters=[("param1", 8), ("param2", 6)],
            reliability_score=0.9,
            error_trends=[("2023-01-01", 2), ("2023-01-02", 3)]
        )
        
        assert stats.method_name == "test_method"
        assert stats.total_errors == 10
        assert stats.error_rate == 0.1
        assert stats.common_error_types == [("ValueError", 5), ("TypeError", 3)]
        assert stats.avg_execution_time_before_error == 1.5
        assert stats.common_parameters == [("param1", 8), ("param2", 6)]
        assert stats.reliability_score == 0.9
        assert stats.error_trends == [("2023-01-01", 2), ("2023-01-02", 3)]


class TestErrorAnalyzer:
    """Test error analyzer functionality."""
    
    def test_initialization(self):
        """Test error analyzer initialization."""
        analyzer = ErrorAnalyzer()
        
        assert analyzer.enable_analysis is True
        assert analyzer.db_path is not None
    
    def test_initialization_disabled(self):
        """Test error analyzer initialization when disabled."""
        analyzer = ErrorAnalyzer(enable_analysis=False)
        
        assert analyzer.enable_analysis is False
        assert analyzer.db_path is not None  # db_path is still set even when disabled


class TestPerformanceEvent:
    """Test performance event functionality."""
    
    def test_performance_event_creation(self):
        """Test performance event creation."""
        event = PerformanceEvent(
            timestamp=1234567890.0,
            method_name="test_method",
            estimator_type="riemann_liouville",
            array_size=100,
            fractional_order=0.5,
            execution_time=1.5,
            memory_before=1024.0,
            memory_after=2048.0,
            memory_peak=3072.0,
            cpu_percent=50.0,
            gc_collections=2,
            gc_time=0.1,
            parameters={"param1": "value1"},
            success=True
        )
        
        assert event.timestamp == 1234567890.0
        assert event.method_name == "test_method"
        assert event.estimator_type == "riemann_liouville"
        assert event.array_size == 100
        assert event.fractional_order == 0.5
        assert event.execution_time == 1.5
        assert event.memory_before == 1024.0
        assert event.memory_after == 2048.0
        assert event.memory_peak == 3072.0
        assert event.cpu_percent == 50.0
        assert event.gc_collections == 2
        assert event.gc_time == 0.1
        assert event.parameters == {"param1": "value1"}
        assert event.success is True


class TestPerformanceStats:
    """Test performance statistics functionality."""
    
    def test_performance_stats_creation(self):
        """Test performance stats creation."""
        stats = PerformanceStats(
            method_name="test_method",
            total_executions=100,
            avg_execution_time=1.5,
            std_execution_time=0.2,
            min_execution_time=1.0,
            max_execution_time=2.0,
            avg_memory_usage=2048.0,
            avg_cpu_usage=50.0,
            success_rate=0.95,
            performance_percentiles={"50th": 1.5, "90th": 2.0, "99th": 2.5},
            array_size_performance={100: 1.5, 200: 2.0}
        )
        
        assert stats.method_name == "test_method"
        assert stats.total_executions == 100
        assert stats.avg_execution_time == 1.5
        assert stats.std_execution_time == 0.2
        assert stats.min_execution_time == 1.0
        assert stats.max_execution_time == 2.0
        assert stats.avg_memory_usage == 2048.0
        assert stats.avg_cpu_usage == 50.0
        assert stats.success_rate == 0.95
        assert stats.performance_percentiles == {"50th": 1.5, "90th": 2.0, "99th": 2.5}
        assert stats.array_size_performance == {100: 1.5, 200: 2.0}


class TestPerformanceMonitor:
    """Test performance monitor functionality."""
    
    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.enable_monitoring is True
        assert monitor.db_path is not None
    
    def test_initialization_disabled(self):
        """Test performance monitor initialization when disabled."""
        monitor = PerformanceMonitor(enable_monitoring=False)
        
        assert monitor.enable_monitoring is False
        assert monitor.db_path is not None  # db_path is still set even when disabled


class TestUsageEvent:
    """Test usage event functionality."""
    
    def test_usage_event_creation(self):
        """Test usage event creation."""
        event = UsageEvent(
            timestamp=1234567890.0,
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters={"param1": "value1"},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            user_session_id="session123"
        )
        
        assert event.timestamp == 1234567890.0
        assert event.method_name == "test_method"
        assert event.estimator_type == "riemann_liouville"
        assert event.parameters == {"param1": "value1"}
        assert event.array_size == 100
        assert event.fractional_order == 0.5
        assert event.execution_success is True
        assert event.user_session_id == "session123"


class TestUsageStats:
    """Test usage statistics functionality."""
    
    def test_usage_stats_creation(self):
        """Test usage stats creation."""
        stats = UsageStats(
            method_name="test_method",
            total_calls=100,
            success_rate=0.95,
            avg_array_size=150.0,
            common_fractional_orders=[(0.5, 50), (0.3, 30)],
            peak_usage_hours=[(9, 20), (14, 15)],
            user_sessions=25
        )
        
        assert stats.method_name == "test_method"
        assert stats.total_calls == 100
        assert stats.success_rate == 0.95
        assert stats.avg_array_size == 150.0
        assert stats.common_fractional_orders == [(0.5, 50), (0.3, 30)]
        assert stats.peak_usage_hours == [(9, 20), (14, 15)]
        assert stats.user_sessions == 25


class TestUsageTracker:
    """Test usage tracker functionality."""
    
    def test_initialization(self):
        """Test usage tracker initialization."""
        tracker = UsageTracker()
        
        assert tracker.enable_tracking is True
        assert tracker.db_path is not None
    
    def test_initialization_disabled(self):
        """Test usage tracker initialization when disabled."""
        tracker = UsageTracker(enable_tracking=False)
        
        assert tracker.enable_tracking is False
        assert tracker.db_path is not None  # db_path is still set even when disabled


class TestWorkflowInsights:
    """Test workflow insights functionality."""
    
    def test_initialization(self):
        """Test workflow insights initialization."""
        insights = WorkflowInsights()
        
        assert insights.enable_insights is True
        assert insights.db_path is not None
    
    def test_initialization_disabled(self):
        """Test workflow insights initialization when disabled."""
        insights = WorkflowInsights(enable_insights=False)
        
        assert insights.enable_insights is False
        assert insights.db_path is not None  # db_path is still set even when disabled


if __name__ == "__main__":
    pytest.main([__file__])
