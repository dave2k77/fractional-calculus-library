"""
Comprehensive tests for ErrorAnalyzer to improve coverage from 33% to 85%.
"""

import pytest
import tempfile
import os
import json
import time
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict
from hpfracc.analytics.error_analyzer import ErrorAnalyzer, ErrorEvent, ErrorStats


class TestErrorAnalyzerComprehensive:
    """Comprehensive tests for ErrorAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.analyzer = ErrorAnalyzer(db_path=self.temp_db.name, enable_analysis=True)
        
        # Sample error data
        self.sample_error = ValueError("Test error message")
        self.sample_parameters = {"alpha": 0.5, "method": "riemann_liouville"}
        self.sample_array_size = 100
        self.sample_fractional_order = 0.5

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_initialization_with_analysis_disabled(self):
        """Test ErrorAnalyzer initialization with analysis disabled."""
        analyzer = ErrorAnalyzer(enable_analysis=False)
        assert analyzer.enable_analysis is False
        assert analyzer.db_path == "error_analytics.db"

    def test_initialization_with_custom_db_path(self):
        """Test ErrorAnalyzer initialization with custom database path."""
        custom_path = "/tmp/custom_error_analytics.db"
        analyzer = ErrorAnalyzer(db_path=custom_path, enable_analysis=True)
        assert analyzer.db_path == custom_path
        assert analyzer.enable_analysis is True

    def test_database_setup_success(self):
        """Test successful database setup."""
        # Database should be created and tables should exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check if error_events table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='error_events'")
        assert cursor.fetchone() is not None
        
        # Check if indices exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        index_names = [row[0] for row in cursor.fetchall()]
        expected_indices = ['idx_method_name', 'idx_error_type', 'idx_timestamp', 'idx_error_hash']
        for idx in expected_indices:
            assert any(idx in name for name in index_names)
        
        conn.close()

    def test_database_setup_failure(self):
        """Test database setup failure handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            analyzer = ErrorAnalyzer(enable_analysis=True)
            assert analyzer.enable_analysis is False

    def test_track_error_success(self):
        """Test successful error tracking."""
        self.analyzer.track_error(
            method_name="test_method",
            estimator_type="riemann_liouville",
            error=self.sample_error,
            parameters=self.sample_parameters,
            array_size=self.sample_array_size,
            fractional_order=self.sample_fractional_order,
            execution_time=1.5,
            memory_usage=1024.0,
            user_session_id="test_session"
        )
        
        # Verify error was stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM error_events")
        assert cursor.fetchone()[0] == 1
        
        # Verify error details
        cursor.execute("SELECT * FROM error_events")
        event = cursor.fetchone()
        assert event[2] == "test_method"  # method_name
        assert event[3] == "riemann_liouville"  # estimator_type
        assert event[4] == "ValueError"  # error_type
        assert event[5] == "Test error message"  # error_message
        assert event[7] == self.analyzer._generate_error_hash(
            self.sample_error, "test_method", self.sample_parameters
        )  # error_hash
        assert event[9] == self.sample_array_size  # array_size
        assert event[10] == self.sample_fractional_order  # fractional_order
        assert event[11] == 1.5  # execution_time
        assert event[12] == 1024.0  # memory_usage
        assert event[13] == "test_session"  # user_session_id
        
        conn.close()

    def test_track_error_with_analysis_disabled(self):
        """Test error tracking when analysis is disabled."""
        analyzer = ErrorAnalyzer(enable_analysis=False)
        analyzer.track_error(
            method_name="test_method",
            estimator_type="riemann_liouville",
            error=self.sample_error,
            parameters=self.sample_parameters,
            array_size=self.sample_array_size,
            fractional_order=self.sample_fractional_order
        )
        
        # Should not store anything
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM error_events")
        assert cursor.fetchone()[0] == 0
        conn.close()

    def test_track_error_exception_handling(self):
        """Test error tracking exception handling."""
        with patch.object(self.analyzer, '_store_error_event', side_effect=Exception("Storage error")):
            # Should not raise exception
            self.analyzer.track_error(
                method_name="test_method",
                estimator_type="riemann_liouville",
                error=self.sample_error,
                parameters=self.sample_parameters,
                array_size=self.sample_array_size,
                fractional_order=self.sample_fractional_order
            )

    def test_generate_error_hash(self):
        """Test error hash generation."""
        hash1 = self.analyzer._generate_error_hash(
            ValueError("Error 1"), "method1", {"param": "value1"}
        )
        hash2 = self.analyzer._generate_error_hash(
            ValueError("Error 2"), "method2", {"param": "value2"}
        )
        hash3 = self.analyzer._generate_error_hash(
            ValueError("Error 1"), "method1", {"param": "value1"}
        )
        
        # Different errors should have different hashes
        assert hash1 != hash2
        # Same errors should have same hashes
        assert hash1 == hash3

    def test_store_error_event_success(self):
        """Test successful error event storage."""
        event = ErrorEvent(
            timestamp=time.time(),
            method_name="test_method",
            estimator_type="riemann_liouville",
            error_type="ValueError",
            error_message="Test error",
            error_traceback="Traceback...",
            error_hash="test_hash",
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_time=1.0,
            memory_usage=512.0,
            user_session_id="session123"
        )
        
        self.analyzer._store_error_event(event)
        
        # Verify storage
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM error_events")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_store_error_event_failure(self):
        """Test error event storage failure handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            event = ErrorEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="riemann_liouville",
                error_type="ValueError",
                error_message="Test error",
                error_traceback="Traceback...",
                error_hash="test_hash",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5
            )
            
            # Should not raise exception
            self.analyzer._store_error_event(event)

    def test_get_error_stats_no_errors(self):
        """Test getting error stats when no errors exist."""
        stats = self.analyzer.get_error_stats()
        assert stats == {}

    def test_get_error_stats_with_errors(self):
        """Test getting error stats with existing errors."""
        # Add some test errors
        for i in range(3):
            self.analyzer.track_error(
                method_name=f"method_{i}",
                estimator_type="riemann_liouville",
                error=ValueError(f"Error {i}"),
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_time=1.0 + i
            )
        
        stats = self.analyzer.get_error_stats()
        assert len(stats) == 3
        assert "method_0" in stats
        assert "method_1" in stats
        assert "method_2" in stats
        
        # Check stats structure
        for method_name, stat in stats.items():
            assert isinstance(stat, ErrorStats)
            assert stat.method_name == method_name
            assert stat.total_errors == 1
            assert stat.error_rate == 0.001  # 1/1000
            assert stat.reliability_score >= 0.0
            assert stat.reliability_score <= 1.0

    def test_get_error_stats_with_time_window(self):
        """Test getting error stats with time window filter."""
        # Add errors at different times
        current_time = time.time()
        
        # Mock time.time() to control timestamps
        with patch('time.time', return_value=current_time - 3600):  # 1 hour ago
            self.analyzer.track_error(
                method_name="old_method",
                estimator_type="riemann_liouville",
                error=ValueError("Old error"),
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5
            )
        
        with patch('time.time', return_value=current_time):  # Now
            self.analyzer.track_error(
                method_name="new_method",
                estimator_type="riemann_liouville",
                error=ValueError("New error"),
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5
            )
        
        # Get stats for last 30 minutes (should only include new error)
        stats = self.analyzer.get_error_stats(time_window_hours=0.5)
        assert len(stats) == 1
        assert "new_method" in stats
        assert "old_method" not in stats

    def test_get_error_stats_database_error(self):
        """Test error stats retrieval with database error."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            stats = self.analyzer.get_error_stats()
            assert stats == {}

    def test_process_events_to_stats(self):
        """Test processing raw events to statistics."""
        # Create mock events
        events = [
            (1, time.time(), "method1", "riemann_liouville", "ValueError", "Error 1", 
             "Traceback...", "hash1", '{"alpha": 0.5}', 100, 0.5, 1.0, 512.0, "session1"),
            (2, time.time(), "method1", "riemann_liouville", "TypeError", "Error 2", 
             "Traceback...", "hash2", '{"alpha": 0.7}', 200, 0.7, 2.0, 1024.0, "session2"),
            (3, time.time(), "method2", "caputo", "ValueError", "Error 3", 
             "Traceback...", "hash3", '{"alpha": 0.3}', 150, 0.3, 1.5, 768.0, "session3")
        ]
        
        stats = self.analyzer._process_events_to_stats(events)
        
        assert len(stats) == 2
        assert "method1" in stats
        assert "method2" in stats
        
        # Check method1 stats
        method1_stats = stats["method1"]
        assert method1_stats.total_errors == 2
        assert method1_stats.avg_execution_time_before_error == 1.5
        assert len(method1_stats.common_error_types) == 2
        assert len(method1_stats.common_parameters) == 2

    def test_get_error_trends_success(self):
        """Test getting error trends successfully."""
        # Add some test errors
        self.analyzer.track_error(
            method_name="test_method",
            estimator_type="riemann_liouville",
            error=ValueError("Test error"),
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5
        )
        
        trends = self.analyzer.get_error_trends("test_method", days=7)
        assert isinstance(trends, list)
        # Should have at least one entry for today
        assert len(trends) >= 1

    def test_get_error_trends_database_error(self):
        """Test error trends retrieval with database error."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            trends = self.analyzer.get_error_trends("test_method")
            assert trends == []

    def test_get_common_error_patterns_success(self):
        """Test getting common error patterns successfully."""
        # Add some test errors
        for i in range(5):
            self.analyzer.track_error(
                method_name=f"method_{i}",
                estimator_type="riemann_liouville",
                error=ValueError("Common error"),
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5
            )
        
        patterns = self.analyzer.get_common_error_patterns()
        assert isinstance(patterns, dict)
        assert "common_error_types" in patterns
        assert "common_error_messages" in patterns
        assert "error_prone_parameters" in patterns

    def test_get_common_error_patterns_database_error(self):
        """Test common error patterns retrieval with database error."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            patterns = self.analyzer.get_common_error_patterns()
            assert patterns == {}

    def test_get_reliability_ranking_success(self):
        """Test getting reliability ranking successfully."""
        # Add some test errors
        for i in range(3):
            self.analyzer.track_error(
                method_name=f"method_{i}",
                estimator_type="riemann_liouville",
                error=ValueError(f"Error {i}"),
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5
            )
        
        ranking = self.analyzer.get_reliability_ranking()
        assert isinstance(ranking, list)
        assert len(ranking) == 3
        # Should be sorted by reliability score (descending)
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_get_reliability_ranking_database_error(self):
        """Test reliability ranking retrieval with database error."""
        with patch.object(self.analyzer, 'get_error_stats', side_effect=Exception("Stats error")):
            ranking = self.analyzer.get_reliability_ranking()
            assert ranking == []

    def test_get_error_correlation_analysis_success(self):
        """Test getting error correlation analysis successfully."""
        # Add test errors with different characteristics
        test_cases = [
            (100, 0.5, 0.1),  # small array, medium order, fast
            (1000, 0.3, 1.0),  # large array, low order, medium
            (500, 0.8, 2.0),   # medium array, high order, slow
        ]
        
        for array_size, fractional_order, execution_time in test_cases:
            self.analyzer.track_error(
                method_name="test_method",
                estimator_type="riemann_liouville",
                error=ValueError("Test error"),
                parameters={"alpha": fractional_order},
                array_size=array_size,
                fractional_order=fractional_order,
                execution_time=execution_time
            )
        
        correlations = self.analyzer.get_error_correlation_analysis()
        assert isinstance(correlations, dict)
        assert "array_size_correlation" in correlations
        assert "fractional_order_correlation" in correlations
        assert "execution_time_correlation" in correlations

    def test_get_error_correlation_analysis_database_error(self):
        """Test error correlation analysis with database error."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            correlations = self.analyzer.get_error_correlation_analysis()
            assert correlations == {}

    def test_export_error_data_success(self):
        """Test successful error data export."""
        # Add some test errors
        self.analyzer.track_error(
            method_name="test_method",
            estimator_type="riemann_liouville",
            error=ValueError("Test error"),
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            self.analyzer.export_error_data(temp_path)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "export_timestamp" in data
            assert "total_methods_with_errors" in data
            assert "methods" in data
            assert data["total_methods_with_errors"] == 1
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_error_data_failure(self):
        """Test error data export failure handling."""
        with patch.object(self.analyzer, 'get_error_stats', side_effect=Exception("Stats error")):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                temp_path = temp_file.name
            
            try:
                # Should not raise exception
                self.analyzer.export_error_data(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_error_event_dataclass(self):
        """Test ErrorEvent dataclass functionality."""
        event = ErrorEvent(
            timestamp=1234567890.0,
            method_name="test_method",
            estimator_type="riemann_liouville",
            error_type="ValueError",
            error_message="Test error",
            error_traceback="Traceback...",
            error_hash="test_hash",
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_time=1.0,
            memory_usage=512.0,
            user_session_id="session123"
        )
        
        # Test asdict conversion
        event_dict = asdict(event)
        assert event_dict["method_name"] == "test_method"
        assert event_dict["error_type"] == "ValueError"
        assert event_dict["array_size"] == 100

    def test_error_stats_dataclass(self):
        """Test ErrorStats dataclass functionality."""
        stats = ErrorStats(
            method_name="test_method",
            total_errors=10,
            error_rate=0.01,
            common_error_types=[("ValueError", 5), ("TypeError", 3)],
            avg_execution_time_before_error=1.5,
            common_parameters=[('{"alpha": 0.5}', 8)],
            reliability_score=0.9,
            error_trends=[("2023-01-01", 2), ("2023-01-02", 3)]
        )
        
        assert stats.method_name == "test_method"
        assert stats.total_errors == 10
        assert stats.reliability_score == 0.9
        assert len(stats.common_error_types) == 2

    def test_database_connection_error_handling(self):
        """Test various database connection error scenarios."""
        # Test connection error in get_error_stats
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            stats = self.analyzer.get_error_stats()
            assert stats == {}
        
        # Test connection error in get_error_trends
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            trends = self.analyzer.get_error_trends("test_method")
            assert trends == []
        
        # Test connection error in get_common_error_patterns
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            patterns = self.analyzer.get_common_error_patterns()
            assert patterns == {}
        
        # Test connection error in get_error_correlation_analysis
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            correlations = self.analyzer.get_error_correlation_analysis()
            assert correlations == {}

    def test_json_serialization_handling(self):
        """Test JSON serialization of parameters."""
        # Test with complex parameters
        complex_params = {
            "alpha": 0.5,
            "method": "riemann_liouville",
            "nested": {"key": "value", "number": 42},
            "list": [1, 2, 3]
        }
        
        self.analyzer.track_error(
            method_name="test_method",
            estimator_type="riemann_liouville",
            error=ValueError("Test error"),
            parameters=complex_params,
            array_size=100,
            fractional_order=0.5
        )
        
        # Verify the error was stored correctly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT parameters FROM error_events")
        stored_params = json.loads(cursor.fetchone()[0])
        assert stored_params == complex_params
        conn.close()

    def test_time_window_edge_cases(self):
        """Test time window filtering edge cases."""
        # Test with very small time window
        stats = self.analyzer.get_error_stats(time_window_hours=0.001)  # ~3.6 seconds
        assert stats == {}
        
        # Test with very large time window
        stats = self.analyzer.get_error_stats(time_window_hours=8760)  # 1 year
        assert isinstance(stats, dict)

    def test_error_hash_collision_handling(self):
        """Test error hash collision scenarios."""
        # Test with identical errors
        error1 = ValueError("Same error")
        error2 = ValueError("Same error")
        
        hash1 = self.analyzer._generate_error_hash(error1, "method1", {"param": "value"})
        hash2 = self.analyzer._generate_error_hash(error2, "method1", {"param": "value"})
        
        assert hash1 == hash2  # Should be identical
        
        # Test with different parameters
        hash3 = self.analyzer._generate_error_hash(error1, "method1", {"param": "different"})
        assert hash1 != hash3  # Should be different

    def test_memory_usage_tracking(self):
        """Test memory usage tracking in error events."""
        self.analyzer.track_error(
            method_name="memory_test",
            estimator_type="riemann_liouville",
            error=ValueError("Memory error"),
            parameters={"alpha": 0.5},
            array_size=1000,
            fractional_order=0.5,
            memory_usage=2048.0
        )
        
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT memory_usage FROM error_events WHERE method_name = 'memory_test'")
        memory_usage = cursor.fetchone()[0]
        assert memory_usage == 2048.0
        conn.close()

    def test_user_session_tracking(self):
        """Test user session tracking in error events."""
        session_id = "user_session_12345"
        self.analyzer.track_error(
            method_name="session_test",
            estimator_type="riemann_liouville",
            error=ValueError("Session error"),
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            user_session_id=session_id
        )
        
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT user_session_id FROM error_events WHERE method_name = 'session_test'")
        stored_session = cursor.fetchone()[0]
        assert stored_session == session_id
        conn.close()

    def test_execution_time_tracking(self):
        """Test execution time tracking in error events."""
        execution_time = 3.14159
        self.analyzer.track_error(
            method_name="timing_test",
            estimator_type="riemann_liouville",
            error=ValueError("Timing error"),
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_time=execution_time
        )
        
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_time FROM error_events WHERE method_name = 'timing_test'")
        stored_time = cursor.fetchone()[0]
        assert abs(stored_time - execution_time) < 1e-6  # Account for floating point precision
        conn.close()
