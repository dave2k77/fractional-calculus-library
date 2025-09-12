"""
Comprehensive tests for UsageTracker to improve coverage from 38% to 85%.
"""

import pytest
import tempfile
import os
import json
import time
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict
from hpfracc.analytics.usage_tracker import UsageTracker, UsageEvent, UsageStats


class TestUsageTrackerComprehensive:
    """Comprehensive tests for UsageTracker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.tracker = UsageTracker(db_path=self.temp_db.name, enable_tracking=True)
        
        # Sample parameters
        self.sample_parameters = {"alpha": 0.5, "method": "riemann_liouville"}
        self.sample_array_size = 100
        self.sample_fractional_order = 0.5

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_initialization_with_tracking_disabled(self):
        """Test UsageTracker initialization with tracking disabled."""
        tracker = UsageTracker(enable_tracking=False)
        assert tracker.enable_tracking is False
        assert tracker.db_path == "usage_analytics.db"
        assert tracker.session_id is not None

    def test_initialization_with_custom_db_path(self):
        """Test UsageTracker initialization with custom database path."""
        custom_path = "/tmp/custom_usage_analytics.db"
        tracker = UsageTracker(db_path=custom_path, enable_tracking=True)
        assert tracker.db_path == custom_path
        assert tracker.enable_tracking is True
        assert tracker.session_id is not None

    def test_session_id_generation(self):
        """Test session ID generation."""
        tracker1 = UsageTracker(enable_tracking=True)
        tracker2 = UsageTracker(enable_tracking=True)
        
        # Session IDs should be unique
        assert tracker1.session_id != tracker2.session_id
        assert len(tracker1.session_id) > 0
        assert len(tracker2.session_id) > 0

    def test_database_setup_success(self):
        """Test successful database setup."""
        # Database should be created and tables should exist
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check if usage_events table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage_events'")
        assert cursor.fetchone() is not None
        
        # Check if indices exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        index_names = [row[0] for row in cursor.fetchall()]
        expected_indices = ['idx_method_name', 'idx_timestamp', 'idx_session_id']
        for idx in expected_indices:
            assert any(idx in name for name in index_names)
        
        conn.close()

    def test_database_setup_failure(self):
        """Test database setup failure handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            tracker = UsageTracker(enable_tracking=True)
            assert tracker.enable_tracking is False

    def test_track_usage_success(self):
        """Test successful usage tracking."""
        self.tracker.track_usage(
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters=self.sample_parameters,
            array_size=self.sample_array_size,
            fractional_order=self.sample_fractional_order,
            execution_success=True,
            user_session_id="test_session",
            ip_address="192.168.1.1"
        )
        
        # Verify usage event was stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage_events")
        assert cursor.fetchone()[0] == 1
        
        # Verify event details
        cursor.execute("SELECT * FROM usage_events")
        event = cursor.fetchone()
        assert event[2] == "test_method"  # method_name
        assert event[3] == "riemann_liouville"  # estimator_type
        assert event[4] == json.dumps(self.sample_parameters)  # parameters
        assert event[5] == self.sample_array_size  # array_size
        assert event[6] == self.sample_fractional_order  # fractional_order
        assert event[7] == 1  # execution_success (True)
        assert event[8] == "test_session"  # user_session_id
        assert event[9] == "192.168.1.1"  # ip_address
        
        conn.close()

    def test_track_usage_with_default_session_id(self):
        """Test usage tracking with default session ID."""
        self.tracker.track_usage(
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters=self.sample_parameters,
            array_size=self.sample_array_size,
            fractional_order=self.sample_fractional_order,
            execution_success=True
        )
        
        # Verify default session ID was used
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT user_session_id FROM usage_events")
        stored_session_id = cursor.fetchone()[0]
        assert stored_session_id == self.tracker.session_id
        conn.close()

    def test_track_usage_with_tracking_disabled(self):
        """Test usage tracking when tracking is disabled."""
        tracker = UsageTracker(enable_tracking=False)
        tracker.track_usage(
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters=self.sample_parameters,
            array_size=self.sample_array_size,
            fractional_order=self.sample_fractional_order,
            execution_success=True
        )
        
        # Should not store anything
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage_events")
        assert cursor.fetchone()[0] == 0
        conn.close()

    def test_track_usage_exception_handling(self):
        """Test usage tracking exception handling."""
        with patch.object(self.tracker, '_store_event', side_effect=Exception("Storage error")):
            # Should not raise exception
            self.tracker.track_usage(
                method_name="test_method",
                estimator_type="riemann_liouville",
                parameters=self.sample_parameters,
                array_size=self.sample_array_size,
                fractional_order=self.sample_fractional_order,
                execution_success=True
            )

    def test_store_event_success(self):
        """Test successful usage event storage."""
        event = UsageEvent(
            timestamp=time.time(),
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            user_session_id="test_session",
            ip_address="192.168.1.1"
        )
        
        self.tracker._store_event(event)
        
        # Verify storage
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage_events")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_store_event_failure(self):
        """Test usage event storage failure handling."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            event = UsageEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            
            # Should not raise exception
            self.tracker._store_event(event)

    def test_get_usage_stats_no_events(self):
        """Test getting usage stats when no events exist."""
        stats = self.tracker.get_usage_stats()
        assert stats == {}

    def test_get_usage_stats_with_events(self):
        """Test getting usage stats with existing events."""
        # Add some test events
        for i in range(3):
            self.tracker.track_usage(
                method_name=f"method_{i}",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100 + i * 10,
                fractional_order=0.5,
                execution_success=True,
                user_session_id=f"session_{i}"
            )
        
        stats = self.tracker.get_usage_stats()
        assert len(stats) == 3
        assert "method_0" in stats
        assert "method_1" in stats
        assert "method_2" in stats
        
        # Check stats structure
        for method_name, stat in stats.items():
            assert isinstance(stat, UsageStats)
            assert stat.method_name == method_name
            assert stat.total_calls == 1
            assert stat.success_rate == 1.0
            assert stat.avg_array_size > 0
            assert stat.user_sessions == 1

    def test_get_usage_stats_with_time_window(self):
        """Test getting usage stats with time window filter."""
        current_time = time.time()
        
        # Add events at different times
        with patch('time.time', return_value=current_time - 3600):  # 1 hour ago
            self.tracker.track_usage(
                method_name="old_method",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        with patch('time.time', return_value=current_time):  # Now
            self.tracker.track_usage(
                method_name="new_method",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        # Get stats for last 30 minutes (should only include new event)
        stats = self.tracker.get_usage_stats(time_window_hours=0.5)
        assert len(stats) == 1
        assert "new_method" in stats
        assert "old_method" not in stats

    def test_get_usage_stats_database_error(self):
        """Test usage stats retrieval with database error."""
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Database error")):
            stats = self.tracker.get_usage_stats()
            assert stats == {}

    def test_process_events_to_stats(self):
        """Test processing raw events to statistics."""
        # Create mock events
        current_time = time.time()
        events = [
            (1, current_time, "method1", "riemann_liouville", '{"alpha": 0.5}', 100, 0.5, 1, "session1", "192.168.1.1"),
            (2, current_time, "method1", "riemann_liouville", '{"alpha": 0.7}', 200, 0.7, 0, "session2", "192.168.1.2"),
            (3, current_time, "method2", "caputo", '{"alpha": 0.3}', 150, 0.3, 1, "session3", "192.168.1.3")
        ]
        
        stats = self.tracker._process_events_to_stats(events)
        
        assert len(stats) == 2
        assert "method1" in stats
        assert "method2" in stats
        
        # Check method1 stats
        method1_stats = stats["method1"]
        assert method1_stats.total_calls == 2
        assert method1_stats.success_rate == 0.5  # 1 success out of 2 calls
        assert method1_stats.avg_array_size == 150.0  # (100 + 200) / 2
        assert method1_stats.user_sessions == 2  # 2 unique sessions

    def test_usage_event_dataclass(self):
        """Test UsageEvent dataclass functionality."""
        event = UsageEvent(
            timestamp=1234567890.0,
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            user_session_id="test_session",
            ip_address="192.168.1.1"
        )
        
        # Test asdict conversion
        event_dict = asdict(event)
        assert event_dict["method_name"] == "test_method"
        assert event_dict["execution_success"] is True
        assert event_dict["user_session_id"] == "test_session"

    def test_usage_stats_dataclass(self):
        """Test UsageStats dataclass functionality."""
        stats = UsageStats(
            method_name="test_method",
            total_calls=100,
            success_rate=0.95,
            avg_array_size=150.0,
            common_fractional_orders=[(0.5, 50), (0.3, 30), (0.7, 20)],
            peak_usage_hours=[(14, 25), (15, 20), (16, 15)],
            user_sessions=25
        )
        
        assert stats.method_name == "test_method"
        assert stats.total_calls == 100
        assert stats.success_rate == 0.95
        assert stats.avg_array_size == 150.0
        assert len(stats.common_fractional_orders) == 3
        assert len(stats.peak_usage_hours) == 3
        assert stats.user_sessions == 25

    def test_common_fractional_orders_calculation(self):
        """Test common fractional orders calculation."""
        # Add events with different fractional orders
        fractional_orders = [0.5, 0.5, 0.3, 0.7, 0.5, 0.3, 0.3, 0.7, 0.7, 0.7]
        for i, order in enumerate(fractional_orders):
            self.tracker.track_usage(
                method_name="order_test",
                estimator_type="riemann_liouville",
                parameters={"alpha": order},
                array_size=100,
                fractional_order=order,
                execution_success=True
            )
        
        stats = self.tracker.get_usage_stats()
        order_test_stats = stats["order_test"]
        
        # Check that common fractional orders were calculated correctly
        assert len(order_test_stats.common_fractional_orders) == 3
        # Should be sorted by frequency: 0.7 (4), 0.5 (3), 0.3 (3)
        assert order_test_stats.common_fractional_orders[0] == (0.7, 4)
        assert order_test_stats.common_fractional_orders[1] == (0.5, 3)
        assert order_test_stats.common_fractional_orders[2] == (0.3, 3)

    def test_peak_usage_hours_calculation(self):
        """Test peak usage hours calculation."""
        # Add events at different hours
        hours = [14, 14, 15, 16, 14, 15, 15, 16, 16, 16]
        for i, hour in enumerate(hours):
            with patch('time.time', return_value=time.mktime((2023, 1, 1, hour, 0, 0, 0, 0, 0))):
                self.tracker.track_usage(
                    method_name="hour_test",
                    estimator_type="riemann_liouville",
                    parameters={"alpha": 0.5},
                    array_size=100,
                    fractional_order=0.5,
                    execution_success=True
                )
        
        stats = self.tracker.get_usage_stats()
        hour_test_stats = stats["hour_test"]
        
        # Check that peak usage hours were calculated correctly
        assert len(hour_test_stats.peak_usage_hours) == 3
        # Should be sorted by frequency: 16 (4), 14 (3), 15 (3)
        assert hour_test_stats.peak_usage_hours[0] == (16, 4)
        assert hour_test_stats.peak_usage_hours[1] == (14, 3)
        assert hour_test_stats.peak_usage_hours[2] == (15, 3)

    def test_user_sessions_tracking(self):
        """Test unique user sessions tracking."""
        # Add events with different session IDs
        session_ids = ["session1", "session2", "session1", "session3", "session2", "session4"]
        for i, session_id in enumerate(session_ids):
            self.tracker.track_usage(
                method_name="session_test",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                user_session_id=session_id
            )
        
        stats = self.tracker.get_usage_stats()
        session_test_stats = stats["session_test"]
        
        # Should have 4 unique sessions
        assert session_test_stats.user_sessions == 4

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Add events with different success rates
        success_values = [True, True, False, True, False, True, False, True, True, True]
        for i, success in enumerate(success_values):
            self.tracker.track_usage(
                method_name="success_test",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=success
            )
        
        stats = self.tracker.get_usage_stats()
        success_test_stats = stats["success_test"]
        
        # Should have 7 successes out of 10 calls = 0.7
        assert success_test_stats.success_rate == 0.7

    def test_average_array_size_calculation(self):
        """Test average array size calculation."""
        # Add events with different array sizes
        array_sizes = [100, 200, 150, 300, 250, 180, 220, 160, 280, 190]
        for i, size in enumerate(array_sizes):
            self.tracker.track_usage(
                method_name="size_test",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=size,
                fractional_order=0.5,
                execution_success=True
            )
        
        stats = self.tracker.get_usage_stats()
        size_test_stats = stats["size_test"]
        
        # Should have average of 203.0 (100+200+150+300+250+180+220+160+280+190)/10
        assert abs(size_test_stats.avg_array_size - 203.0) < 0.1

    def test_parameter_serialization(self):
        """Test parameter serialization and storage."""
        complex_params = {
            "alpha": 0.5,
            "method": "riemann_liouville",
            "nested": {"key": "value", "number": 42},
            "list": [1, 2, 3]
        }
        
        self.tracker.track_usage(
            method_name="param_test",
            estimator_type="riemann_liouville",
            parameters=complex_params,
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify parameters were stored correctly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT parameters FROM usage_events WHERE method_name = 'param_test'")
        stored_params = json.loads(cursor.fetchone()[0])
        assert stored_params == complex_params
        conn.close()

    def test_ip_address_tracking(self):
        """Test IP address tracking."""
        ip_addresses = ["192.168.1.1", "10.0.0.1", "172.16.0.1"]
        for i, ip in enumerate(ip_addresses):
            self.tracker.track_usage(
                method_name="ip_test",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                ip_address=ip
            )
        
        # Verify IP addresses were stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT ip_address FROM usage_events WHERE method_name = 'ip_test' ORDER BY id")
        stored_ips = [row[0] for row in cursor.fetchall()]
        assert stored_ips == ip_addresses
        conn.close()

    def test_database_connection_error_handling(self):
        """Test various database connection error scenarios."""
        # Test connection error in get_usage_stats
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            stats = self.tracker.get_usage_stats()
            assert stats == {}
        
        # Test connection error in _store_event
        with patch('sqlite3.connect', side_effect=sqlite3.Error("Connection failed")):
            event = UsageEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="riemann_liouville",
                parameters={"alpha": 0.5},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            # Should not raise exception
            self.tracker._store_event(event)

    def test_time_window_edge_cases(self):
        """Test time window filtering edge cases."""
        # Test with very small time window
        stats = self.tracker.get_usage_stats(time_window_hours=0.001)  # ~3.6 seconds
        assert stats == {}
        
        # Test with very large time window
        stats = self.tracker.get_usage_stats(time_window_hours=8760)  # 1 year
        assert isinstance(stats, dict)

    def test_json_serialization_handling(self):
        """Test JSON serialization of complex parameters."""
        # Test with parameters that are JSON serializable
        complex_params = {
            "alpha": 0.5,
            "method": "riemann_liouville",
            "nested": {"key": "value", "number": 42},
            "list": [1, 2, 3]
        }
        
        self.tracker.track_usage(
            method_name="json_test",
            estimator_type="riemann_liouville",
            parameters=complex_params,
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify the parameters were stored correctly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT parameters FROM usage_events WHERE method_name = 'json_test'")
        stored_params = json.loads(cursor.fetchone()[0])
        
        # Check that complex types were handled
        assert stored_params["alpha"] == 0.5
        assert stored_params["nested"]["key"] == "value"
        assert stored_params["list"] == [1, 2, 3]
        conn.close()

    def test_empty_parameters_handling(self):
        """Test handling of empty parameters."""
        self.tracker.track_usage(
            method_name="empty_param_test",
            estimator_type="riemann_liouville",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify empty parameters were stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT parameters FROM usage_events WHERE method_name = 'empty_param_test'")
        stored_params = json.loads(cursor.fetchone()[0])
        assert stored_params == {}
        conn.close()

    def test_none_values_handling(self):
        """Test handling of None values in optional fields."""
        self.tracker.track_usage(
            method_name="none_test",
            estimator_type="riemann_liouville",
            parameters={"alpha": 0.5},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            user_session_id=None,
            ip_address=None
        )
        
        # Verify None values were stored correctly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT user_session_id, ip_address FROM usage_events WHERE method_name = 'none_test'")
        session_id, ip_address = cursor.fetchone()
        # user_session_id should be the default session ID (not None)
        assert session_id is not None
        assert ip_address is None
        conn.close()

    def test_large_array_size_handling(self):
        """Test handling of large array sizes."""
        large_size = 1000000
        self.tracker.track_usage(
            method_name="large_size_test",
            estimator_type="riemann_liouville",
            parameters={"alpha": 0.5},
            array_size=large_size,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify large array size was stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT array_size FROM usage_events WHERE method_name = 'large_size_test'")
        stored_size = cursor.fetchone()[0]
        assert stored_size == large_size
        conn.close()

    def test_extreme_fractional_orders(self):
        """Test handling of extreme fractional orders."""
        extreme_orders = [0.0, 0.001, 0.999, 1.0, 2.5, 10.0]
        for i, order in enumerate(extreme_orders):
            self.tracker.track_usage(
                method_name=f"extreme_order_test_{i}",
                estimator_type="riemann_liouville",
                parameters={"alpha": order},
                array_size=100,
                fractional_order=order,
                execution_success=True
            )
        
        # Verify all extreme orders were stored
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT fractional_order FROM usage_events WHERE method_name LIKE 'extreme_order_test_%' ORDER BY id")
        stored_orders = [row[0] for row in cursor.fetchall()]
        assert stored_orders == extreme_orders
        conn.close()

    def test_unicode_parameter_handling(self):
        """Test handling of Unicode parameters."""
        unicode_params = {
            "method": "riemann_liouville",
            "description": "Fractional derivative with α = 0.5",
            "symbols": ["α", "β", "γ", "∂"],
            "unicode_key": "αβγ∂"
        }
        
        self.tracker.track_usage(
            method_name="unicode_test",
            estimator_type="riemann_liouville",
            parameters=unicode_params,
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify Unicode parameters were stored correctly
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        cursor.execute("SELECT parameters FROM usage_events WHERE method_name = 'unicode_test'")
        stored_params = json.loads(cursor.fetchone()[0])
        assert stored_params == unicode_params
        conn.close()
