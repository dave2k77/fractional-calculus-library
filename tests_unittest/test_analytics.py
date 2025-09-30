"""
Unittest tests for HPFRACC analytics functionality
"""

import unittest
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestAnalyticsManager(unittest.TestCase):
    """Test analytics manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics import AnalyticsManager, AnalyticsConfig
        self.AnalyticsManager = AnalyticsManager
        self.AnalyticsConfig = AnalyticsConfig
    
    def test_analytics_config_creation(self):
        """Test analytics configuration creation"""
        # Test default configuration
        config = self.AnalyticsConfig()
        self.assertIsNotNone(config)
        
        # Test custom configuration
        config = self.AnalyticsConfig(
            enable_usage_tracking=True,
            enable_performance_monitoring=True,
            enable_error_analysis=True,
            data_retention_days=30
        )
        self.assertTrue(config.enable_usage_tracking)
        self.assertTrue(config.enable_performance_monitoring)
        self.assertTrue(config.enable_error_analysis)
        self.assertEqual(config.data_retention_days, 30)
    
    def test_analytics_manager_creation(self):
        """Test analytics manager creation"""
        config = self.AnalyticsConfig(enable_usage_tracking=True)
        manager = self.AnalyticsManager(config)
        
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.session_id)
        self.assertIsInstance(manager.session_id, str)
        self.assertEqual(len(manager.session_id), 36)  # UUID length
    
    def test_analytics_manager_singleton(self):
        """Test that analytics manager behaves consistently"""
        config = self.AnalyticsConfig(enable_usage_tracking=True)
        manager1 = self.AnalyticsManager(config)
        manager2 = self.AnalyticsManager(config)
        
        # Should have same session ID (singleton behavior)
        self.assertEqual(manager1.session_id, manager2.session_id)

class TestUsageTracker(unittest.TestCase):
    """Test usage tracking functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics.usage_tracker import UsageTracker
        self.UsageTracker = UsageTracker
    
    def test_usage_tracker_initialization(self):
        """Test usage tracker initialization"""
        tracker = self.UsageTracker()
        self.assertIsNotNone(tracker)
        self.assertIsInstance(tracker.start_time, float)
    
    def test_usage_tracking_basic(self):
        """Test basic usage tracking"""
        tracker = self.UsageTracker()
        
        # Track usage
        tracker.track_operation("test_operation", {"param": "value"})
        
        # Check that usage was tracked
        self.assertIsNotNone(tracker.usage_data)
        self.assertGreater(len(tracker.usage_data), 0)
    
    def test_usage_tracking_metadata(self):
        """Test usage tracking with metadata"""
        tracker = self.UsageTracker()
        
        # Track operation with metadata
        metadata = {
            "input_size": 100,
            "output_size": 50,
            "backend": "numba"
        }
        tracker.track_operation("matrix_multiply", metadata)
        
        # Verify metadata was stored
        self.assertTrue(len(tracker.usage_data) > 0)
        latest_entry = tracker.usage_data[-1]
        self.assertEqual(latest_entry["operation"], "matrix_multiply")
        self.assertEqual(latest_entry["metadata"]["input_size"], 100)

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics.performance_monitor import PerformanceMonitor
        self.PerformanceMonitor = PerformanceMonitor
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization"""
        monitor = self.PerformanceMonitor()
        self.assertIsNotNone(monitor)
        self.assertIsInstance(monitor.start_time, float)
    
    def test_performance_monitoring_basic(self):
        """Test basic performance monitoring"""
        monitor = self.PerformanceMonitor()
        
        # Start timing
        monitor.start_timing("test_operation")
        
        # Simulate some work
        time.sleep(0.01)  # 10ms
        
        # End timing
        monitor.end_timing("test_operation")
        
        # Check that timing was recorded
        self.assertIsNotNone(monitor.timing_data)
        self.assertGreater(len(monitor.timing_data), 0)
    
    def test_performance_monitoring_metrics(self):
        """Test performance monitoring metrics"""
        monitor = self.PerformanceMonitor()
        
        # Record multiple timings
        for i in range(5):
            monitor.start_timing(f"operation_{i}")
            time.sleep(0.001)  # 1ms
            monitor.end_timing(f"operation_{i}")
        
        # Check metrics
        metrics = monitor.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(len(metrics), 0)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        monitor = self.PerformanceMonitor()
        
        # Track memory usage
        monitor.track_memory_usage("test_operation", 1024)  # 1KB
        
        # Check that memory usage was tracked
        self.assertIsNotNone(monitor.memory_data)
        self.assertGreater(len(monitor.memory_data), 0)

class TestErrorAnalyzer(unittest.TestCase):
    """Test error analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics.error_analyzer import ErrorAnalyzer
        self.ErrorAnalyzer = ErrorAnalyzer
    
    def test_error_analyzer_initialization(self):
        """Test error analyzer initialization"""
        analyzer = self.ErrorAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_error_analysis_basic(self):
        """Test basic error analysis"""
        analyzer = self.ErrorAnalyzer()
        
        # Simulate some errors
        errors = [0.1, 0.2, 0.05, 0.15, 0.08]
        
        # Analyze errors
        analysis = analyzer.analyze_errors(errors)
        
        self.assertIsNotNone(analysis)
        self.assertIn("mean_error", analysis)
        self.assertIn("max_error", analysis)
        self.assertIn("min_error", analysis)
    
    def test_error_analysis_statistics(self):
        """Test error analysis statistics"""
        analyzer = self.ErrorAnalyzer()
        
        # Test with known data
        errors = [0.1, 0.2, 0.05, 0.15, 0.08]
        analysis = analyzer.analyze_errors(errors)
        
        # Check statistics
        self.assertAlmostEqual(analysis["mean_error"], sum(errors) / len(errors), places=10)
        self.assertEqual(analysis["max_error"], max(errors))
        self.assertEqual(analysis["min_error"], min(errors))
        self.assertGreater(analysis["max_error"], analysis["min_error"])
    
    def test_convergence_analysis(self):
        """Test convergence analysis"""
        analyzer = self.ErrorAnalyzer()
        
        # Simulate convergence data
        iterations = list(range(1, 11))
        errors = [1.0 / i for i in iterations]  # Decreasing errors
        
        # Analyze convergence
        convergence = analyzer.analyze_convergence(errors)
        
        self.assertIsNotNone(convergence)
        self.assertIn("converged", convergence)
        self.assertIn("final_error", convergence)

class TestWorkflowInsights(unittest.TestCase):
    """Test workflow insights functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics.workflow_insights import WorkflowInsights
        self.WorkflowInsights = WorkflowInsights
    
    def test_workflow_insights_initialization(self):
        """Test workflow insights initialization"""
        insights = self.WorkflowInsights()
        self.assertIsNotNone(insights)
    
    def test_workflow_pattern_detection(self):
        """Test workflow pattern detection"""
        insights = self.WorkflowInsights()
        
        # Simulate workflow data
        workflow_data = [
            {"operation": "load_data", "timestamp": time.time()},
            {"operation": "preprocess", "timestamp": time.time() + 1},
            {"operation": "train_model", "timestamp": time.time() + 2},
            {"operation": "evaluate", "timestamp": time.time() + 3},
        ]
        
        # Analyze workflow
        patterns = insights.detect_patterns(workflow_data)
        
        self.assertIsNotNone(patterns)
        self.assertIsInstance(patterns, dict)
    
    def test_workflow_optimization_suggestions(self):
        """Test workflow optimization suggestions"""
        insights = self.WorkflowInsights()
        
        # Simulate performance data
        performance_data = {
            "load_data": {"time": 1.0, "memory": 100},
            "preprocess": {"time": 2.0, "memory": 200},
            "train_model": {"time": 10.0, "memory": 1000},
            "evaluate": {"time": 0.5, "memory": 50},
        }
        
        # Get optimization suggestions
        suggestions = insights.get_optimization_suggestions(performance_data)
        
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)

class TestAnalyticsIntegration(unittest.TestCase):
    """Test analytics system integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics import AnalyticsManager, AnalyticsConfig
        self.config = AnalyticsConfig(
            enable_usage_tracking=True,
            enable_performance_monitoring=True,
            enable_error_analysis=True
        )
        self.manager = AnalyticsManager(self.config)
    
    def test_analytics_system_workflow(self):
        """Test complete analytics system workflow"""
        # Test that all components work together
        self.assertIsNotNone(self.manager.session_id)
        
        # Test usage tracking
        self.manager.track_usage("test_operation", {"param": "value"})
        
        # Test performance monitoring
        self.manager.start_timing("test_operation")
        time.sleep(0.001)
        self.manager.end_timing("test_operation")
        
        # Test error analysis
        errors = [0.1, 0.2, 0.05]
        self.manager.analyze_errors(errors)
    
    def test_analytics_data_consistency(self):
        """Test analytics data consistency"""
        # Ensure that data is stored consistently across components
        session_id = self.manager.session_id
        
        # All operations should be associated with the same session
        self.manager.track_usage("operation1", {})
        self.manager.track_usage("operation2", {})
        
        # Session ID should remain consistent
        self.assertEqual(self.manager.session_id, session_id)
    
    def test_analytics_error_handling(self):
        """Test analytics error handling"""
        # Test that analytics system handles errors gracefully
        try:
            # Test with invalid data
            self.manager.track_usage(None, None)
            self.manager.analyze_errors([])
            self.manager.start_timing("")
            self.manager.end_timing("")
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, (ValueError, TypeError, AttributeError))

class TestAnalyticsPerformance(unittest.TestCase):
    """Test analytics system performance"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.analytics import AnalyticsManager, AnalyticsConfig
        self.config = AnalyticsConfig(enable_usage_tracking=True)
        self.manager = AnalyticsManager(self.config)
    
    def test_analytics_performance_overhead(self):
        """Test that analytics doesn't add significant overhead"""
        import time
        
        # Test without analytics
        start_time = time.time()
        for _ in range(100):
            pass  # Dummy operation
        baseline_time = time.time() - start_time
        
        # Test with analytics
        start_time = time.time()
        for i in range(100):
            self.manager.track_usage(f"operation_{i}", {"iteration": i})
        analytics_time = time.time() - start_time
        
        # Analytics overhead should be reasonable
        overhead_ratio = analytics_time / baseline_time
        self.assertLess(overhead_ratio, 10)  # Should be less than 10x overhead
    
    def test_analytics_memory_usage(self):
        """Test that analytics doesn't consume excessive memory"""
        # Track many operations
        for i in range(1000):
            self.manager.track_usage(f"operation_{i}", {"data": "x" * 100})
        
        # System should still be responsive
        self.assertIsNotNone(self.manager.session_id)
        
        # Should be able to continue tracking
        self.manager.track_usage("final_operation", {})

if __name__ == '__main__':
    unittest.main()
