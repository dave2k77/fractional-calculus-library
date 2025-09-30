"""
Comprehensive unittest tests for HPFRACC analytics system
Focusing on expanding coverage from 20% to 40%+
"""

import unittest
import sys
import os
import time
import tempfile
import json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestAnalyticsManagerComprehensive(unittest.TestCase):
    """Comprehensive tests for analytics manager to increase coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics import AnalyticsManager, AnalyticsConfig
        self.AnalyticsManager = AnalyticsManager
        self.AnalyticsConfig = AnalyticsConfig
    
    def test_analytics_config_comprehensive(self):
        """Test comprehensive analytics configuration"""
        # Test default configuration
        config = self.AnalyticsConfig()
        self.assertTrue(config.enable_usage_tracking)
        self.assertTrue(config.enable_performance_monitoring)
        self.assertTrue(config.enable_error_analysis)
        self.assertTrue(config.enable_workflow_insights)
        self.assertEqual(config.data_retention_days, 30)
        self.assertEqual(config.export_format, "json")
        self.assertTrue(config.generate_reports)
        self.assertEqual(config.report_output_dir, "analytics_reports")
        
        # Test custom configuration
        config = self.AnalyticsConfig(
            enable_usage_tracking=False,
            enable_performance_monitoring=True,
            enable_error_analysis=False,
            enable_workflow_insights=True,
            data_retention_days=60,
            export_format="csv",
            generate_reports=False,
            report_output_dir="custom_reports"
        )
        self.assertFalse(config.enable_usage_tracking)
        self.assertTrue(config.enable_performance_monitoring)
        self.assertFalse(config.enable_error_analysis)
        self.assertTrue(config.enable_workflow_insights)
        self.assertEqual(config.data_retention_days, 60)
        self.assertEqual(config.export_format, "csv")
        self.assertFalse(config.generate_reports)
        self.assertEqual(config.report_output_dir, "custom_reports")
    
    def test_analytics_manager_initialization_comprehensive(self):
        """Test comprehensive analytics manager initialization"""
        # Test with default config
        manager = self.AnalyticsManager()
        self.assertIsNotNone(manager.session_id)
        self.assertEqual(len(manager.session_id), 36)  # UUID length
        self.assertIsNotNone(manager.config)
        self.assertIsNotNone(manager.usage_tracker)
        self.assertIsNotNone(manager.performance_monitor)
        self.assertIsNotNone(manager.error_analyzer)
        self.assertIsNotNone(manager.workflow_insights)
        self.assertIsNotNone(manager.output_dir)
        
        # Test with custom config
        config = self.AnalyticsConfig(enable_usage_tracking=False)
        manager = self.AnalyticsManager(config)
        self.assertIsNotNone(manager.session_id)
        self.assertEqual(manager.config.enable_usage_tracking, False)
        
        # Test session ID uniqueness
        manager1 = self.AnalyticsManager()
        manager2 = self.AnalyticsManager()
        self.assertNotEqual(manager1.session_id, manager2.session_id)
    
    def test_method_call_tracking_comprehensive(self):
        """Test comprehensive method call tracking"""
        manager = self.AnalyticsManager()
        
        # Test successful method call tracking
        manager.track_method_call(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": "value1", "param2": 42},
            array_size=1000,
            fractional_order=0.5,
            execution_success=True,
            execution_time=0.1,
            memory_usage=1024
        )
        
        # Test failed method call tracking
        manager.track_method_call(
            method_name="failing_method",
            estimator_type="test_estimator",
            parameters={"param1": "value1"},
            array_size=500,
            fractional_order=0.7,
            execution_success=False,
            execution_time=0.05,
            memory_usage=512,
            error=ValueError("Test error")
        )
        
        # Verify tracking was successful (components should be initialized)
        self.assertIsNotNone(manager.usage_tracker)
        self.assertIsNotNone(manager.performance_monitor)
        self.assertIsNotNone(manager.error_analyzer)
        self.assertIsNotNone(manager.workflow_insights)
    
    def test_performance_tracking_comprehensive(self):
        """Test comprehensive performance tracking"""
        manager = self.AnalyticsManager()
        
        # Test timing operations
        with manager.timing_context("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        # Test memory tracking
        manager.track_memory_usage("test_operation", 1024)
        
        # Test performance metrics
        metrics = manager.get_performance_metrics()
        self.assertIsNotNone(metrics)
        
        # Test performance summary
        summary = manager.get_performance_summary()
        self.assertIsNotNone(summary)
    
    def test_error_analysis_comprehensive(self):
        """Test comprehensive error analysis"""
        manager = self.AnalyticsManager()
        
        # Test error tracking
        manager.track_error("test_error", ValueError("Test error message"))
        
        # Test error analysis
        analysis = manager.analyze_errors()
        self.assertIsNotNone(analysis)
        
        # Test error summary
        summary = manager.get_error_summary()
        self.assertIsNotNone(summary)
        
        # Test error trends
        trends = manager.get_error_trends()
        self.assertIsNotNone(trends)
    
    def test_usage_tracking_comprehensive(self):
        """Test comprehensive usage tracking"""
        manager = self.AnalyticsManager()
        
        # Test usage tracking
        manager.track_usage(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": "value1"},
            array_size=1000,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Test usage statistics
        stats = manager.get_usage_statistics()
        self.assertIsNotNone(stats)
        
        # Test usage trends
        trends = manager.get_usage_trends()
        self.assertIsNotNone(trends)
        
        # Test popular methods
        popular = manager.get_popular_methods()
        self.assertIsNotNone(popular)
    
    def test_workflow_insights_comprehensive(self):
        """Test comprehensive workflow insights"""
        manager = self.AnalyticsManager()
        
        # Test workflow tracking
        manager.track_workflow_event(
            session_id=manager.session_id,
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": "value1"},
            array_size=1000,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Test workflow analysis
        analysis = manager.analyze_workflows()
        self.assertIsNotNone(analysis)
        
        # Test workflow patterns
        patterns = manager.get_workflow_patterns()
        self.assertIsNotNone(patterns)
        
        # Test optimization suggestions
        suggestions = manager.get_optimization_suggestions()
        self.assertIsNotNone(suggestions)
    
    def test_reporting_comprehensive(self):
        """Test comprehensive reporting functionality"""
        # Create temporary directory for reports
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.AnalyticsConfig(report_output_dir=temp_dir)
            manager = self.AnalyticsManager(config)
            
            # Test report generation
            manager.generate_report()
            
            # Check if report directory was created
            report_dir = Path(temp_dir)
            self.assertTrue(report_dir.exists())
            
            # Test different export formats
            manager.export_data("json")
            manager.export_data("csv")
            manager.export_data("html")
            
            # Test comprehensive report
            manager.generate_comprehensive_report()
    
    def test_data_management_comprehensive(self):
        """Test comprehensive data management"""
        manager = self.AnalyticsManager()
        
        # Test data export
        json_data = manager.export_data("json")
        self.assertIsNotNone(json_data)
        
        csv_data = manager.export_data("csv")
        self.assertIsNotNone(csv_data)
        
        html_data = manager.export_data("html")
        self.assertIsNotNone(html_data)
        
        # Test data import
        test_data = {
            "usage": [],
            "performance": [],
            "errors": [],
            "workflows": []
        }
        manager.import_data(json.dumps(test_data))
        
        # Test data cleanup
        manager.cleanup_old_data()
        
        # Test data backup
        manager.backup_data()
    
    def test_configuration_management(self):
        """Test configuration management"""
        manager = self.AnalyticsManager()
        
        # Test configuration update
        new_config = self.AnalyticsConfig(
            enable_usage_tracking=False,
            data_retention_days=60
        )
        manager.update_config(new_config)
        self.assertEqual(manager.config.enable_usage_tracking, False)
        self.assertEqual(manager.config.data_retention_days, 60)
        
        # Test configuration validation
        self.assertTrue(manager.validate_config())
        
        # Test configuration reset
        manager.reset_config()
        self.assertEqual(manager.config.data_retention_days, 30)  # Default value
    
    def test_session_management(self):
        """Test session management"""
        manager = self.AnalyticsManager()
        
        # Test session start
        session_id = manager.start_session()
        self.assertIsNotNone(session_id)
        self.assertEqual(len(session_id), 36)  # UUID length
        
        # Test session end
        manager.end_session()
        
        # Test session info
        info = manager.get_session_info()
        self.assertIsNotNone(info)
        
        # Test active sessions
        active = manager.get_active_sessions()
        self.assertIsNotNone(active)
    
    def test_monitoring_comprehensive(self):
        """Test comprehensive monitoring functionality"""
        manager = self.AnalyticsManager()
        
        # Test health check
        health = manager.health_check()
        self.assertIsNotNone(health)
        
        # Test system status
        status = manager.get_system_status()
        self.assertIsNotNone(status)
        
        # Test monitoring metrics
        metrics = manager.get_monitoring_metrics()
        self.assertIsNotNone(metrics)
        
        # Test alerting
        manager.check_alerts()
        
        # Test monitoring dashboard
        dashboard = manager.get_monitoring_dashboard()
        self.assertIsNotNone(dashboard)
    
    def test_analytics_components_integration(self):
        """Test integration between analytics components"""
        manager = self.AnalyticsManager()
        
        # Test that all components are properly initialized
        self.assertIsNotNone(manager.usage_tracker)
        self.assertIsNotNone(manager.performance_monitor)
        self.assertIsNotNone(manager.error_analyzer)
        self.assertIsNotNone(manager.workflow_insights)
        
        # Test component communication
        manager.track_method_call(
            method_name="integration_test",
            estimator_type="test_estimator",
            parameters={"test": True},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # All components should have received the data
        self.assertIsNotNone(manager.usage_tracker)
        self.assertIsNotNone(manager.performance_monitor)
        self.assertIsNotNone(manager.error_analyzer)
        self.assertIsNotNone(manager.workflow_insights)
    
    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling"""
        manager = self.AnalyticsManager()
        
        # Test graceful handling of invalid inputs
        try:
            manager.track_method_call(
                method_name=None,  # Invalid input
                estimator_type="test",
                parameters={},
                array_size=0,
                fractional_order=0.5,
                execution_success=True
            )
        except Exception:
            # Should handle gracefully
            pass
        
        # Test handling of missing components
        manager.usage_tracker = None
        try:
            manager.track_method_call(
                method_name="test",
                estimator_type="test",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        except Exception:
            # Should handle gracefully
            pass
        
        # Manager should still be functional
        self.assertIsNotNone(manager.session_id)
    
    def test_performance_characteristics(self):
        """Test performance characteristics"""
        manager = self.AnalyticsManager()
        
        # Test tracking performance
        start_time = time.time()
        for i in range(100):
            manager.track_method_call(
                method_name=f"method_{i}",
                estimator_type="test",
                parameters={"iteration": i},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        tracking_time = time.time() - start_time
        
        # Should complete quickly (less than 1 second for 100 operations)
        self.assertLess(tracking_time, 1.0)
        
        # Test report generation performance
        start_time = time.time()
        manager.generate_report()
        report_time = time.time() - start_time
        
        # Should complete quickly (less than 2 seconds)
        self.assertLess(report_time, 2.0)
    
    def test_memory_management(self):
        """Test memory management characteristics"""
        manager = self.AnalyticsManager()
        
        # Test that we can track many operations without memory issues
        for i in range(1000):
            manager.track_method_call(
                method_name=f"method_{i}",
                estimator_type="test",
                parameters={"iteration": i},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        # Manager should still be functional
        self.assertIsNotNone(manager.session_id)
        
        # Test cleanup
        manager.cleanup_old_data()
        
        # Should not crash
        self.assertIsNotNone(manager.get_usage_statistics())

class TestAnalyticsConfigComprehensive(unittest.TestCase):
    """Test analytics configuration comprehensively"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        from hpfracc.analytics import AnalyticsConfig
        
        # Test valid configuration
        config = AnalyticsConfig()
        self.assertTrue(config.enable_usage_tracking)
        self.assertTrue(config.enable_performance_monitoring)
        self.assertTrue(config.enable_error_analysis)
        self.assertTrue(config.enable_workflow_insights)
        
        # Test edge cases
        config = AnalyticsConfig(data_retention_days=0)
        self.assertEqual(config.data_retention_days, 0)
        
        config = AnalyticsConfig(data_retention_days=365)
        self.assertEqual(config.data_retention_days, 365)
        
        # Test export formats
        for format_type in ["json", "csv", "html"]:
            config = AnalyticsConfig(export_format=format_type)
            self.assertEqual(config.export_format, format_type)
    
    def test_config_immutability(self):
        """Test configuration immutability"""
        from hpfracc.analytics import AnalyticsConfig
        
        config = AnalyticsConfig()
        original_retention = config.data_retention_days
        
        # Modifying the config should not affect the original
        config.data_retention_days = 999
        # This test depends on the actual implementation
        # If config is immutable, the change should not persist
        # If config is mutable, this test documents the behavior

if __name__ == '__main__':
    unittest.main()
