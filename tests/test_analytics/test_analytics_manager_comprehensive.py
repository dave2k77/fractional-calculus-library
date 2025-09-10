"""
Comprehensive tests for AnalyticsManager module.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from hpfracc.analytics.analytics_manager import (
    AnalyticsManager, 
    AnalyticsConfig
)


class TestAnalyticsConfig:
    """Test AnalyticsConfig dataclass."""
    
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
            export_format="csv",
            report_output_dir="custom_reports"
        )
        assert config.enable_usage_tracking is False
        assert config.data_retention_days == 60
        assert config.export_format == "csv"
        assert config.report_output_dir == "custom_reports"


class TestAnalyticsManager:
    """Test AnalyticsManager class."""
    
    def test_initialization_default(self):
        """Test initialization with default config."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            assert isinstance(manager.config, AnalyticsConfig)
            assert manager.session_id is not None
            assert len(manager.session_id) > 0
            mock_usage.assert_called_once()
            mock_perf.assert_called_once()
            mock_error.assert_called_once()
            mock_workflow.assert_called_once()
    
    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = AnalyticsConfig(enable_usage_tracking=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            
            assert manager.config == config
            mock_usage.assert_called_once_with(enable_tracking=False)
    
    def test_generate_session_id(self):
        """Test session ID generation."""
        manager = AnalyticsManager()
        session_id = manager._generate_session_id()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        # Should be different each time
        session_id2 = manager._generate_session_id()
        assert session_id != session_id2
    
    def test_track_method_call(self):
        """Test tracking method calls."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            manager.track_method_call(
                "test_function", 
                "estimator", 
                {"arg1": "value1"}, 
                100, 
                0.5, 
                True
            )
            
            # Check that track_usage was called with the expected parameters
            manager.usage_tracker.track_usage.assert_called_once()
            call_args = manager.usage_tracker.track_usage.call_args
            assert call_args[1]['method_name'] == "test_function"
            assert call_args[1]['estimator_type'] == "estimator"
            assert call_args[1]['parameters'] == {"arg1": "value1"}
            assert call_args[1]['array_size'] == 100
            assert call_args[1]['fractional_order'] == 0.5
            assert call_args[1]['execution_success'] == True
    
    def test_monitor_method_performance(self):
        """Test monitoring method performance."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test context manager usage
            with manager.monitor_method_performance("test_method", "estimator", 100, 0.5, {"param": "value"}):
                pass  # Context manager should work
            
            manager.performance_monitor.monitor_performance.assert_called_once_with(
                "test_method", "estimator", 100, 0.5, {"param": "value"}
            )
    
    def test_get_comprehensive_analytics(self):
        """Test getting comprehensive analytics."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            expected_analytics = {
                "usage": {"calls": 100},
                "performance": {"avg_time": 1.5},
                "errors": {"error_count": 5},
                "workflow": {"steps": 10}
            }
            
            with patch.object(manager, 'get_comprehensive_analytics') as mock_get:
                mock_get.return_value = expected_analytics
                analytics = manager.get_comprehensive_analytics()
                
                assert analytics == expected_analytics
                mock_get.assert_called_once()
    
    def test_generate_analytics_report(self):
        """Test generating analytics report."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            with patch.object(manager, 'generate_analytics_report') as mock_generate:
                mock_generate.return_value = "test_report"
                report = manager.generate_analytics_report()
                
                assert report == "test_report"
                mock_generate.assert_called_once()
    
    def test_export_all_data(self):
        """Test exporting all data."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            with patch.object(manager, 'export_all_data') as mock_export:
                expected_data = {"json": "data", "csv": "data", "html": "data"}
                mock_export.return_value = expected_data
                data = manager.export_all_data()
                
                assert data == expected_data
                mock_export.assert_called_once()
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            with patch.object(manager, 'cleanup_old_data') as mock_cleanup:
                manager.cleanup_old_data()
                mock_cleanup.assert_called_once()
    
    def test_private_methods(self):
        """Test private methods."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test _generate_session_id
            session_id = manager._generate_session_id()
            assert isinstance(session_id, str)
            assert len(session_id) > 0
            
            # Test _generate_json_report
            analytics = {"test": "data"}
            with patch.object(manager, '_generate_json_report') as mock_json:
                mock_json.return_value = '{"test": "data"}'
                result = manager._generate_json_report(analytics)
                assert result == '{"test": "data"}'
                mock_json.assert_called_once_with(analytics)




def mock_open():
    """Mock open function for testing."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open()
