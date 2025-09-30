"""
Unittest tests for HPFRACC utility functions
"""

import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestErrorAnalysis(unittest.TestCase):
    """Test error analysis utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.error_analysis import ErrorAnalyzer
        self.ErrorAnalyzer = ErrorAnalyzer
    
    def test_error_analyzer_initialization(self):
        """Test error analyzer initialization"""
        analyzer = self.ErrorAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_error_analysis_basic(self):
        """Test basic error analysis"""
        analyzer = self.ErrorAnalyzer()
        
        # Test with known data
        errors = np.array([0.1, 0.2, 0.05, 0.15, 0.08])
        
        # Analyze errors
        analysis = analyzer.analyze_errors(errors)
        
        self.assertIsNotNone(analysis)
        self.assertIn("mean_error", analysis)
        self.assertIn("max_error", analysis)
        self.assertIn("min_error", analysis)
        self.assertIn("std_error", analysis)
    
    def test_error_analysis_statistics(self):
        """Test error analysis statistics"""
        analyzer = self.ErrorAnalyzer()
        
        # Test with known data
        errors = np.array([0.1, 0.2, 0.05, 0.15, 0.08])
        analysis = analyzer.analyze_errors(errors)
        
        # Check statistics
        expected_mean = np.mean(errors)
        expected_max = np.max(errors)
        expected_min = np.min(errors)
        expected_std = np.std(errors)
        
        self.assertAlmostEqual(analysis["mean_error"], expected_mean, places=10)
        self.assertEqual(analysis["max_error"], expected_max)
        self.assertEqual(analysis["min_error"], expected_min)
        self.assertAlmostEqual(analysis["std_error"], expected_std, places=10)
    
    def test_convergence_analysis(self):
        """Test convergence analysis"""
        analyzer = self.ErrorAnalyzer()
        
        # Simulate convergence data
        iterations = np.arange(1, 11)
        errors = 1.0 / iterations  # Decreasing errors
        
        # Analyze convergence
        convergence = analyzer.analyze_convergence(errors)
        
        self.assertIsNotNone(convergence)
        self.assertIn("converged", convergence)
        self.assertIn("final_error", convergence)
        self.assertIn("convergence_rate", convergence)
    
    def test_error_distribution_analysis(self):
        """Test error distribution analysis"""
        analyzer = self.ErrorAnalyzer()
        
        # Test with normal distribution-like data
        np.random.seed(42)
        errors = np.random.normal(0.1, 0.05, 100)
        errors = np.abs(errors)  # Make positive
        
        # Analyze distribution
        distribution = analyzer.analyze_error_distribution(errors)
        
        self.assertIsNotNone(distribution)
        self.assertIn("mean", distribution)
        self.assertIn("std", distribution)
        self.assertIn("percentiles", distribution)

class TestConvergenceAnalysis(unittest.TestCase):
    """Test convergence analysis utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.convergence_analysis import ConvergenceAnalyzer
        self.ConvergenceAnalyzer = ConvergenceAnalyzer
    
    def test_convergence_analyzer_initialization(self):
        """Test convergence analyzer initialization"""
        analyzer = self.ConvergenceAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_convergence_detection(self):
        """Test convergence detection"""
        analyzer = self.ConvergenceAnalyzer()
        
        # Test converging sequence
        values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        converged = analyzer.detect_convergence(values, tolerance=0.1)
        
        self.assertTrue(converged)
    
    def test_convergence_rate_analysis(self):
        """Test convergence rate analysis"""
        analyzer = self.ConvergenceAnalyzer()
        
        # Test geometric convergence
        values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        rate = analyzer.analyze_convergence_rate(values)
        
        self.assertIsNotNone(rate)
        self.assertIn("rate", rate)
        self.assertIn("type", rate)
    
    def test_oscillatory_convergence_detection(self):
        """Test oscillatory convergence detection"""
        analyzer = self.ConvergenceAnalyzer()
        
        # Test oscillatory sequence
        values = np.array([1.0, 0.9, 0.8, 0.85, 0.82, 0.83, 0.825, 0.826])
        oscillatory = analyzer.detect_oscillatory_convergence(values)
        
        self.assertIsNotNone(oscillatory)
        self.assertIsInstance(oscillatory, bool)

class TestMemoryManager(unittest.TestCase):
    """Test memory management utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.memory_manager import MemoryManager
        self.MemoryManager = MemoryManager
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization"""
        manager = self.MemoryManager()
        self.assertIsNotNone(manager)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        manager = self.MemoryManager()
        
        # Track memory usage
        manager.track_usage("operation1", 1024)  # 1KB
        manager.track_usage("operation2", 2048)  # 2KB
        
        # Get memory report
        report = manager.get_memory_report()
        
        self.assertIsNotNone(report)
        self.assertIn("total_memory", report)
        self.assertIn("operation_count", report)
    
    def test_memory_optimization_suggestions(self):
        """Test memory optimization suggestions"""
        manager = self.MemoryManager()
        
        # Track high memory usage
        manager.track_usage("large_operation", 10000)  # 10KB
        manager.track_usage("small_operation", 100)    # 100B
        
        # Get optimization suggestions
        suggestions = manager.get_optimization_suggestions()
        
        self.assertIsNotNone(suggestions)
        self.assertIsInstance(suggestions, list)
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        manager = self.MemoryManager()
        
        # Track some usage
        manager.track_usage("temp_operation", 1024)
        
        # Clean up
        manager.cleanup()
        
        # Should still be functional
        self.assertIsNotNone(manager)

class TestCacheManager(unittest.TestCase):
    """Test cache management utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.cache_manager import CacheManager
        self.CacheManager = CacheManager
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization"""
        manager = self.CacheManager()
        self.assertIsNotNone(manager)
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        manager = self.CacheManager()
        
        # Test cache set and get
        key = "test_key"
        value = {"data": [1, 2, 3, 4, 5]}
        
        manager.set(key, value)
        retrieved = manager.get(key)
        
        self.assertEqual(retrieved, value)
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        manager = self.CacheManager(max_age=0.1)  # 100ms expiration
        
        # Set a value
        manager.set("expiring_key", "test_value")
        
        # Should be available immediately
        self.assertEqual(manager.get("expiring_key"), "test_value")
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        # Should be expired
        self.assertIsNone(manager.get("expiring_key"))
    
    def test_cache_size_limits(self):
        """Test cache size limits"""
        manager = self.CacheManager(max_size=2)
        
        # Add items up to limit
        manager.set("key1", "value1")
        manager.set("key2", "value2")
        
        # Should be available
        self.assertEqual(manager.get("key1"), "value1")
        self.assertEqual(manager.get("key2"), "value2")
        
        # Add one more item (should evict oldest)
        manager.set("key3", "value3")
        
        # key1 should be evicted
        self.assertIsNone(manager.get("key1"))
        self.assertEqual(manager.get("key2"), "value2")
        self.assertEqual(manager.get("key3"), "value3")

class TestPlotManager(unittest.TestCase):
    """Test plotting utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.plot_manager import PlotManager
        self.PlotManager = PlotManager
    
    def test_plot_manager_initialization(self):
        """Test plot manager initialization"""
        manager = self.PlotManager()
        self.assertIsNotNone(manager)
    
    def test_plot_data_preparation(self):
        """Test plot data preparation"""
        manager = self.PlotManager()
        
        # Test data preparation
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        plot_data = manager.prepare_plot_data(x, y, title="Test Plot")
        
        self.assertIsNotNone(plot_data)
        self.assertIn("x", plot_data)
        self.assertIn("y", plot_data)
        self.assertIn("title", plot_data)
        self.assertEqual(plot_data["title"], "Test Plot")
    
    def test_plot_configuration(self):
        """Test plot configuration"""
        manager = self.PlotManager()
        
        # Test configuration
        config = manager.get_plot_config(
            plot_type="line",
            xlabel="Time",
            ylabel="Value",
            grid=True
        )
        
        self.assertIsNotNone(config)
        self.assertEqual(config["plot_type"], "line")
        self.assertEqual(config["xlabel"], "Time")
        self.assertEqual(config["ylabel"], "Value")
        self.assertTrue(config["grid"])
    
    def test_plot_export_formats(self):
        """Test plot export formats"""
        manager = self.PlotManager()
        
        # Test supported formats
        formats = manager.get_supported_formats()
        
        self.assertIsNotNone(formats)
        self.assertIsInstance(formats, list)
        self.assertIn("png", formats)
        self.assertIn("pdf", formats)
        self.assertIn("svg", formats)

class TestUtilityIntegration(unittest.TestCase):
    """Test utility functions integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.error_analysis import ErrorAnalyzer
        from hpfracc.utils.convergence_analysis import ConvergenceAnalyzer
        from hpfracc.utils.memory_manager import MemoryManager
        from hpfracc.utils.cache_manager import CacheManager
        
        self.error_analyzer = ErrorAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
    
    def test_utility_workflow(self):
        """Test complete utility workflow"""
        # Test error analysis
        errors = np.array([0.1, 0.2, 0.05, 0.15])
        error_analysis = self.error_analyzer.analyze_errors(errors)
        
        # Test convergence analysis
        values = np.array([1.0, 0.5, 0.25, 0.125])
        convergence = self.convergence_analyzer.detect_convergence(values)
        
        # Test memory management
        self.memory_manager.track_usage("error_analysis", 1024)
        memory_report = self.memory_manager.get_memory_report()
        
        # Test caching
        self.cache_manager.set("error_analysis", error_analysis)
        cached_analysis = self.cache_manager.get("error_analysis")
        
        # All should work together
        self.assertIsNotNone(error_analysis)
        self.assertIsInstance(convergence, bool)
        self.assertIsNotNone(memory_report)
        self.assertEqual(cached_analysis, error_analysis)
    
    def test_utility_performance(self):
        """Test utility functions performance"""
        import time
        
        # Test error analysis performance
        start_time = time.time()
        for _ in range(100):
            errors = np.random.random(50)
            self.error_analyzer.analyze_errors(errors)
        error_time = time.time() - start_time
        
        # Test convergence analysis performance
        start_time = time.time()
        for _ in range(100):
            values = np.random.random(20)
            self.convergence_analyzer.detect_convergence(values)
        convergence_time = time.time() - start_time
        
        # Should complete quickly
        self.assertLess(error_time, 1.0)
        self.assertLess(convergence_time, 1.0)
    
    def test_utility_error_handling(self):
        """Test utility functions error handling"""
        # Test with invalid inputs
        try:
            self.error_analyzer.analyze_errors([])
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError))
        
        try:
            self.convergence_analyzer.detect_convergence([])
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError))
        
        # Should handle gracefully
        self.assertTrue(True)

class TestUtilityMathematicalCorrectness(unittest.TestCase):
    """Test utility functions mathematical correctness"""
    
    def setUp(self):
        """Set up test fixtures"""
        from hpfracc.utils.error_analysis import ErrorAnalyzer
        from hpfracc.utils.convergence_analysis import ConvergenceAnalyzer
        self.error_analyzer = ErrorAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def test_error_analysis_mathematical_correctness(self):
        """Test error analysis mathematical correctness"""
        # Test with known mathematical properties
        errors = np.array([0.1, 0.2, 0.05, 0.15, 0.08])
        
        analysis = self.error_analyzer.analyze_errors(errors)
        
        # Check mathematical properties
        self.assertGreaterEqual(analysis["max_error"], analysis["mean_error"])
        self.assertLessEqual(analysis["min_error"], analysis["mean_error"])
        self.assertGreaterEqual(analysis["std_error"], 0)
        
        # Check that mean is correct
        expected_mean = np.mean(errors)
        self.assertAlmostEqual(analysis["mean_error"], expected_mean, places=10)
    
    def test_convergence_analysis_mathematical_correctness(self):
        """Test convergence analysis mathematical correctness"""
        # Test geometric convergence
        values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        
        convergence = self.convergence_analyzer.detect_convergence(values, tolerance=0.1)
        self.assertTrue(convergence)
        
        # Test non-converging sequence
        non_converging = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        convergence = self.convergence_analyzer.detect_convergence(non_converging, tolerance=0.1)
        self.assertFalse(convergence)
    
    def test_utility_functions_numerical_stability(self):
        """Test utility functions numerical stability"""
        # Test with very small numbers
        small_errors = np.array([1e-10, 2e-10, 3e-10])
        analysis = self.error_analyzer.analyze_errors(small_errors)
        
        self.assertTrue(np.isfinite(analysis["mean_error"]))
        self.assertTrue(np.isfinite(analysis["std_error"]))
        
        # Test with very large numbers
        large_errors = np.array([1e10, 2e10, 3e10])
        analysis = self.error_analyzer.analyze_errors(large_errors)
        
        self.assertTrue(np.isfinite(analysis["mean_error"]))
        self.assertTrue(np.isfinite(analysis["std_error"]))

if __name__ == '__main__':
    unittest.main()
