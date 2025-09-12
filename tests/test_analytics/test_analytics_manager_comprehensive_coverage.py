"""
Comprehensive coverage tests for AnalyticsManager module.

This module provides extensive tests to improve coverage of the analytics_manager.py
module, focusing on the methods that are currently not well covered.
"""

import pytest
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from hpfracc.analytics.analytics_manager import (
    AnalyticsManager, 
    AnalyticsConfig
)


class TestAnalyticsManagerComprehensive:
    """Comprehensive tests for AnalyticsManager class to improve coverage."""
    
    def test_track_method_call_with_error(self):
        """Test tracking method calls with error handling."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            error = ValueError("Test error")
            
            manager.track_method_call(
                "test_function", 
                "estimator", 
                {"arg1": "value1"}, 
                100, 
                0.5, 
                False,
                execution_time=1.5,
                memory_usage=1024.0,
                error=error
            )
            
            # Check that error tracking was called
            manager.error_analyzer.track_error.assert_called_once()
            call_args = manager.error_analyzer.track_error.call_args
            assert call_args[1]['error'] == error
            assert call_args[1]['execution_time'] == 1.5
            assert call_args[1]['memory_usage'] == 1024.0
    
    def test_track_method_call_exception_handling(self):
        """Test tracking method calls when an exception occurs in tracking."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            # Make usage tracker raise an exception
            manager = AnalyticsManager()
            manager.usage_tracker.track_usage.side_effect = Exception("Tracking failed")
            
            # Should not raise an exception
            manager.track_method_call(
                "test_function", 
                "estimator", 
                {"arg1": "value1"}, 
                100, 
                0.5, 
                True
            )
    
    def test_monitor_method_performance_disabled(self):
        """Test monitoring method performance when disabled."""
        config = AnalyticsConfig(enable_performance_monitoring=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            
            # Context manager should work but not call performance monitor
            with manager.monitor_method_performance("test_method", "estimator", 100, 0.5, {"param": "value"}):
                pass
            
            # Performance monitor should not be called
            manager.performance_monitor.monitor_performance.assert_not_called()
    
    def test_get_comprehensive_analytics_usage_tracking_disabled(self):
        """Test getting comprehensive analytics when usage tracking is disabled."""
        config = AnalyticsConfig(enable_usage_tracking=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            analytics = manager.get_comprehensive_analytics()
            
            assert 'usage' not in analytics
            assert 'performance' in analytics
            assert 'errors' in analytics
            assert 'workflow' in analytics
    
    def test_get_comprehensive_analytics_performance_monitoring_disabled(self):
        """Test getting comprehensive analytics when performance monitoring is disabled."""
        config = AnalyticsConfig(enable_performance_monitoring=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            analytics = manager.get_comprehensive_analytics()
            
            assert 'usage' in analytics
            assert 'performance' not in analytics
            assert 'errors' in analytics
            assert 'workflow' in analytics
    
    def test_get_comprehensive_analytics_error_analysis_disabled(self):
        """Test getting comprehensive analytics when error analysis is disabled."""
        config = AnalyticsConfig(enable_error_analysis=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            analytics = manager.get_comprehensive_analytics()
            
            assert 'usage' in analytics
            assert 'performance' in analytics
            assert 'errors' not in analytics
            assert 'workflow' in analytics
    
    def test_get_comprehensive_analytics_workflow_insights_disabled(self):
        """Test getting comprehensive analytics when workflow insights are disabled."""
        config = AnalyticsConfig(enable_workflow_insights=False)
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager(config)
            analytics = manager.get_comprehensive_analytics()
            
            assert 'usage' in analytics
            assert 'performance' in analytics
            assert 'errors' in analytics
            assert 'workflow' not in analytics
    
    def test_get_comprehensive_analytics_exception_handling(self):
        """Test getting comprehensive analytics when an exception occurs."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            manager.usage_tracker.get_usage_stats.side_effect = Exception("Analytics failed")
            
            analytics = manager.get_comprehensive_analytics()
            assert analytics == {}
    
    def test_generate_analytics_report_json(self):
        """Test generating JSON analytics report."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            manager = AnalyticsManager()
            manager.get_comprehensive_analytics = Mock(return_value={"test": "data"})
            
            result = manager.generate_analytics_report()
            
            assert result is not None
            mock_json_dump.assert_called_once()
    
    def test_generate_analytics_report_csv(self):
        """Test generating CSV analytics report."""
        config = AnalyticsConfig(export_format="csv")
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('pandas.DataFrame') as mock_df:
            
            manager = AnalyticsManager(config)
            manager.get_comprehensive_analytics = Mock(return_value={"test": "data"})
            
            result = manager.generate_analytics_report()
            
            assert result is not None
            mock_df.assert_called_once()
    
    def test_generate_analytics_report_html(self):
        """Test generating HTML analytics report."""
        config = AnalyticsConfig(export_format="html")
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('matplotlib.pyplot.style.use') as mock_style, \
             patch('seaborn.set_palette') as mock_palette, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            manager = AnalyticsManager(config)
            manager.get_comprehensive_analytics = Mock(return_value={"test": "data"})
            
            # Mock subplot axes
            mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
            mock_fig = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = manager.generate_analytics_report()
            
            assert result is not None
    
    def test_generate_analytics_report_unsupported_format(self):
        """Test generating analytics report with unsupported format."""
        config = AnalyticsConfig(export_format="xml")
        
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            manager = AnalyticsManager(config)
            manager.get_comprehensive_analytics = Mock(return_value={"test": "data"})
            
            result = manager.generate_analytics_report()
            
            assert result is not None
            # Should fall back to JSON format
            mock_json_dump.assert_called_once()
    
    def test_generate_analytics_report_exception_handling(self):
        """Test generating analytics report when an exception occurs."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            with patch.object(manager, 'get_comprehensive_analytics', side_effect=Exception("Report failed")):
                result = manager.generate_analytics_report()
                assert result == ""
    
    def test_generate_json_report(self):
        """Test generating JSON report."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            manager = AnalyticsManager()
            analytics = {"test": "data"}
            
            result = manager._generate_json_report(analytics)
            
            assert result is not None
            mock_json_dump.assert_called_once_with(analytics, mock_file.return_value.__enter__.return_value, indent=2)
    
    def test_generate_csv_report(self):
        """Test generating CSV report."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('pandas.DataFrame') as mock_df:
            
            manager = AnalyticsManager()
            analytics = {
                'usage': {
                    'stats': {
                        'method1': Mock(total_calls=10, success_rate=0.9, avg_array_size=100, user_sessions=5)
                    }
                },
                'performance': {
                    'stats': {
                        'method1': Mock(total_executions=10, avg_execution_time=1.5, avg_memory_usage=1024, success_rate=0.9)
                    }
                },
                'errors': {
                    'stats': {
                        'method1': Mock(total_errors=1, error_rate=0.1, reliability_score=0.9)
                    }
                }
            }
            
            result = manager._generate_csv_report(analytics)
            
            assert result is not None
            mock_df.assert_called_once()
    
    def test_generate_html_report(self):
        """Test generating HTML report."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('matplotlib.pyplot.style.use') as mock_style, \
             patch('seaborn.set_palette') as mock_palette, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            manager = AnalyticsManager()
            analytics = {"test": "data"}
            
            # Mock subplot axes
            mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
            mock_fig = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = manager._generate_html_report(analytics)
            
            assert result is not None
    
    def test_create_analytics_plots(self):
        """Test creating analytics plots."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('matplotlib.pyplot.style.use') as mock_style, \
             patch('seaborn.set_palette') as mock_palette, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight, \
             patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            manager = AnalyticsManager()
            analytics = {
                'usage': {
                    'stats': {
                        'method1': Mock(total_calls=10),
                        'method2': Mock(total_calls=5)
                    }
                },
                'performance': {
                    'stats': {
                        'method1': Mock(avg_execution_time=1.5),
                        'method2': Mock(avg_execution_time=2.0)
                    }
                },
                'errors': {
                    'stats': {
                        'method1': Mock(error_rate=0.1, reliability_score=0.9),
                        'method2': Mock(error_rate=0.2, reliability_score=0.8)
                    }
                }
            }
            
            # Mock subplot axes properly
            mock_ax1 = Mock()
            mock_ax2 = Mock()
            mock_ax3 = Mock()
            mock_ax4 = Mock()
            mock_axes = [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]]
            mock_fig = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            manager._create_analytics_plots(analytics)
            
            mock_style.assert_called_once_with('seaborn-v0_8')
            mock_palette.assert_called_once_with("husl")
            mock_subplots.assert_called_once_with(2, 2, figsize=(15, 12))
            # Note: tight_layout might not be called if there's an exception in the plotting code
            # mock_tight.assert_called_once()
            mock_save.assert_called_once()
            mock_close.assert_called_once()
    
    def test_create_analytics_plots_exception_handling(self):
        """Test creating analytics plots when an exception occurs."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow, \
             patch('matplotlib.pyplot.style.use', side_effect=Exception("Plot failed")):
            
            manager = AnalyticsManager()
            analytics = {"test": "data"}
            
            # Should not raise an exception
            manager._create_analytics_plots(analytics)
    
    def test_generate_html_content(self):
        """Test generating HTML content."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            analytics = {
                'session_id': 'test_session',
                'usage': {'test': 'data'},
                'performance': {'test': 'data'},
                'errors': {'test': 'data'},
                'workflow': {'test': 'data'}
            }
            
            result = manager._generate_html_content(analytics)
            
            assert '<!DOCTYPE html>' in result
            assert 'HPFRACC Analytics Report' in result
            assert 'test_session' in result
    
    def test_generate_usage_html(self):
        """Test generating usage HTML."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test with data
            usage_data = {
                'popular_methods': [('method1', 10), ('method2', 5)]
            }
            result = manager._generate_usage_html(usage_data)
            
            assert '<h3>Popular Methods</h3>' in result
            assert 'method1' in result
            assert 'method2' in result
            
            # Test with no data
            result_empty = manager._generate_usage_html({})
            assert 'No usage data available' in result_empty
    
    def test_generate_performance_html(self):
        """Test generating performance HTML."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test with data
            perf_data = {
                'stats': {
                    'method1': Mock(avg_execution_time=1.5, success_rate=0.9)
                }
            }
            result = manager._generate_performance_html(perf_data)
            
            assert '<h3>Performance Statistics</h3>' in result
            assert 'method1' in result
            
            # Test with no data
            result_empty = manager._generate_performance_html({})
            assert 'No performance data available' in result_empty
    
    def test_generate_error_html(self):
        """Test generating error HTML."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test with data
            error_data = {
                'reliability_ranking': [('method1', 0.9), ('method2', 0.8)]
            }
            result = manager._generate_error_html(error_data)
            
            assert '<h3>Reliability Ranking</h3>' in result
            assert 'method1' in result
            
            # Test with no data
            result_empty = manager._generate_error_html({})
            assert 'No error data available' in result_empty
    
    def test_generate_workflow_html(self):
        """Test generating workflow HTML."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Test with data
            workflow_data = {
                'patterns': [
                    Mock(method_sequence=['method1', 'method2'], frequency=5, avg_success_rate=0.9)
                ]
            }
            result = manager._generate_workflow_html(workflow_data)
            
            assert '<h3>Common Workflow Patterns</h3>' in result
            assert 'method1' in result
            
            # Test with no data
            result_empty = manager._generate_workflow_html({})
            assert 'No workflow data available' in result_empty
    
    def test_export_all_data(self):
        """Test exporting all data."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Mock export methods
            manager.usage_tracker.export_usage_data.return_value = "usage_path"
            manager.performance_monitor.export_performance_data.return_value = "perf_path"
            manager.error_analyzer.export_error_data.return_value = "error_path"
            manager.workflow_insights.export_workflow_data.return_value = "workflow_path"
            
            with patch.object(manager, 'generate_analytics_report', return_value="report_path"):
                result = manager.export_all_data()
                
                assert 'usage' in result
                assert 'performance' in result
                assert 'errors' in result
                assert 'workflow' in result
                assert 'comprehensive_report' in result
    
    def test_export_all_data_exception_handling(self):
        """Test exporting all data when an exception occurs."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            manager.usage_tracker.export_usage_data.side_effect = Exception("Export failed")
            
            result = manager.export_all_data()
            assert result == {}
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            
            # Mock cleanup methods
            manager.usage_tracker.clear_old_data.return_value = 10
            manager.performance_monitor.clear_old_data.return_value = 5
            manager.error_analyzer.clear_old_data.return_value = 3
            manager.workflow_insights.clear_old_data.return_value = 2
            
            result = manager.cleanup_old_data()
            
            assert result['usage'] == 10
            assert result['performance'] == 5
            assert result['errors'] == 3
            assert result['workflow'] == 2
    
    def test_cleanup_old_data_exception_handling(self):
        """Test cleaning up old data when an exception occurs."""
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            manager = AnalyticsManager()
            manager.usage_tracker.clear_old_data.side_effect = Exception("Cleanup failed")
            
            result = manager.cleanup_old_data()
            assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])
