"""
Comprehensive tests for WorkflowInsights module.

This module provides extensive tests to improve coverage of the workflow_insights.py
module, focusing on the methods that are currently not well covered.
"""

import pytest
import tempfile
import json
import os
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from hpfracc.analytics.workflow_insights import (
    WorkflowInsights,
    WorkflowEvent,
    WorkflowPattern,
    WorkflowSummary
)


class TestWorkflowEvent:
    """Test WorkflowEvent dataclass."""
    
    def test_workflow_event_creation(self):
        """Test creating a workflow event."""
        event = WorkflowEvent(
            timestamp=1234567890.0,
            session_id="session123",
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters={"param1": "value1"},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            execution_time=1.5,
            user_agent="test_agent",
            ip_address="192.168.1.1"
        )
        
        assert event.timestamp == 1234567890.0
        assert event.session_id == "session123"
        assert event.method_name == "test_method"
        assert event.estimator_type == "riemann_liouville"
        assert event.parameters == {"param1": "value1"}
        assert event.array_size == 100
        assert event.fractional_order == 0.5
        assert event.execution_success is True
        assert event.execution_time == 1.5
        assert event.user_agent == "test_agent"
        assert event.ip_address == "192.168.1.1"
    
    def test_workflow_event_optional_fields(self):
        """Test creating a workflow event with optional fields."""
        event = WorkflowEvent(
            timestamp=1234567890.0,
            session_id="session123",
            method_name="test_method",
            estimator_type="riemann_liouville",
            parameters={"param1": "value1"},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        assert event.execution_time is None
        assert event.user_agent is None
        assert event.ip_address is None


class TestWorkflowPattern:
    """Test WorkflowPattern dataclass."""
    
    def test_workflow_pattern_creation(self):
        """Test creating a workflow pattern."""
        pattern = WorkflowPattern(
            pattern_id="pattern123",
            method_sequence=["method1", "method2", "method3"],
            frequency=10,
            avg_success_rate=0.9,
            avg_execution_time=2.5,
            common_parameters={"param1": "value1"},
            user_sessions={"session1", "session2"},
            first_seen=1234567890.0,
            last_seen=1234567891.0
        )
        
        assert pattern.pattern_id == "pattern123"
        assert pattern.method_sequence == ["method1", "method2", "method3"]
        assert pattern.frequency == 10
        assert pattern.avg_success_rate == 0.9
        assert pattern.avg_execution_time == 2.5
        assert pattern.common_parameters == {"param1": "value1"}
        assert pattern.user_sessions == {"session1", "session2"}
        assert pattern.first_seen == 1234567890.0
        assert pattern.last_seen == 1234567891.0


class TestWorkflowSummary:
    """Test WorkflowSummary dataclass."""
    
    def test_workflow_summary_creation(self):
        """Test creating a workflow summary."""
        pattern = WorkflowPattern(
            pattern_id="pattern123",
            method_sequence=["method1", "method2"],
            frequency=5,
            avg_success_rate=0.8,
            avg_execution_time=1.5,
            common_parameters={},
            user_sessions={"session1"},
            first_seen=1234567890.0,
            last_seen=1234567891.0
        )
        
        summary = WorkflowSummary(
            total_sessions=10,
            total_workflows=5,
            common_patterns=[pattern],
            method_transitions={"method1": {"method2": 3}},
            session_durations={"session1": 100.0},
            user_behavior_clusters={"power_users": ["session1"]}
        )
        
        assert summary.total_sessions == 10
        assert summary.total_workflows == 5
        assert len(summary.common_patterns) == 1
        assert summary.method_transitions == {"method1": {"method2": 3}}
        assert summary.session_durations == {"session1": 100.0}
        assert summary.user_behavior_clusters == {"power_users": ["session1"]}


class TestWorkflowInsights:
    """Test WorkflowInsights class."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        insights = WorkflowInsights()
        
        assert insights.db_path == "workflow_analytics.db"
        assert insights.enable_insights is True
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        insights = WorkflowInsights(
            db_path="custom_workflow.db",
            enable_insights=False
        )
        
        assert insights.db_path == "custom_workflow.db"
        assert insights.enable_insights is False
    
    def test_setup_database_disabled(self):
        """Test database setup when insights are disabled."""
        insights = WorkflowInsights(enable_insights=False)
        
        # Should not raise an exception
        insights._setup_database()
    
    def test_setup_database_exception_handling(self):
        """Test database setup when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            # Should not raise an exception
            insights._setup_database()
            assert insights.enable_insights is False
    
    def test_track_workflow_event_disabled(self):
        """Test tracking workflow event when insights are disabled."""
        insights = WorkflowInsights(enable_insights=False)
        
        # Should not raise an exception
        insights.track_workflow_event(
            "session123",
            "test_method",
            "riemann_liouville",
            {"param": "value"},
            100,
            0.5,
            True
        )
    
    def test_track_workflow_event_exception_handling(self):
        """Test tracking workflow event when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            # Should not raise an exception
            insights.track_workflow_event(
                "session123",
                "test_method",
                "riemann_liouville",
                {"param": "value"},
                100,
                0.5,
                True
            )
    
    def test_store_workflow_event(self):
        """Test storing a workflow event."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            insights = WorkflowInsights(db_path=db_path)
            
            event = WorkflowEvent(
                timestamp=1234567890.0,
                session_id="session123",
                method_name="test_method",
                estimator_type="riemann_liouville",
                parameters={"param1": "value1"},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=1.5,
                user_agent="test_agent",
                ip_address="192.168.1.1"
            )
            
            insights._store_workflow_event(event)
            
            # Verify the event was stored
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM workflow_events")
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 1
            
        finally:
            os.unlink(db_path)
    
    def test_store_workflow_event_exception_handling(self):
        """Test storing a workflow event when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            event = WorkflowEvent(
                timestamp=1234567890.0,
                session_id="session123",
                method_name="test_method",
                estimator_type="riemann_liouville",
                parameters={"param1": "value1"},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            
            # Should not raise an exception
            insights._store_workflow_event(event)
    
    def test_get_workflow_patterns(self):
        """Test getting workflow patterns."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            insights = WorkflowInsights(db_path=db_path)
            
            # Add some test data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert test events
            test_events = [
                ("session1", "method1", 1234567890.0, True, 1.0),
                ("session1", "method2", 1234567891.0, True, 1.5),
                ("session2", "method1", 1234567892.0, True, 1.2),
                ("session2", "method2", 1234567893.0, True, 1.8),
            ]
            
            for session_id, method_name, timestamp, success, exec_time in test_events:
                cursor.execute('''
                    INSERT INTO workflow_events
                    (timestamp, session_id, method_name, estimator_type, parameters,
                     array_size, fractional_order, execution_success, execution_time,
                     user_agent, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, session_id, method_name, "riemann_liouville", "{}",
                      100, 0.5, success, exec_time, None, None))
            
            conn.commit()
            conn.close()
            
            # Get patterns
            patterns = insights.get_workflow_patterns(min_frequency=1, max_pattern_length=2)
            
            assert len(patterns) > 0
            assert all(isinstance(pattern, WorkflowPattern) for pattern in patterns)
            
        finally:
            os.unlink(db_path)
    
    def test_get_workflow_patterns_exception_handling(self):
        """Test getting workflow patterns when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            patterns = insights.get_workflow_patterns()
            assert patterns == []
    
    def test_find_patterns_of_length(self):
        """Test finding patterns of a specific length."""
        insights = WorkflowInsights(enable_insights=False)
        
        session_sequences = {
            "session1": [
                {"method": "method1", "timestamp": 1234567890.0, "success": True, "exec_time": 1.0},
                {"method": "method2", "timestamp": 1234567891.0, "success": True, "exec_time": 1.5},
                {"method": "method3", "timestamp": 1234567892.0, "success": True, "exec_time": 2.0}
            ],
            "session2": [
                {"method": "method1", "timestamp": 1234567893.0, "success": True, "exec_time": 1.2},
                {"method": "method2", "timestamp": 1234567894.0, "success": True, "exec_time": 1.8}
            ]
        }
        
        patterns = insights._find_patterns_of_length(session_sequences, 2, 1)
        
        assert len(patterns) > 0
        assert all(isinstance(pattern, WorkflowPattern) for pattern in patterns)
        assert all(len(pattern.method_sequence) == 2 for pattern in patterns)
    
    def test_get_method_transitions(self):
        """Test getting method transitions."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            insights = WorkflowInsights(db_path=db_path)
            
            # Add some test data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert test events
            test_events = [
                ("session1", "method1", 1234567890.0),
                ("session1", "method2", 1234567891.0),
                ("session2", "method1", 1234567892.0),
                ("session2", "method3", 1234567893.0),
            ]
            
            for session_id, method_name, timestamp in test_events:
                cursor.execute('''
                    INSERT INTO workflow_events
                    (timestamp, session_id, method_name, estimator_type, parameters,
                     array_size, fractional_order, execution_success, execution_time,
                     user_agent, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, session_id, method_name, "riemann_liouville", "{}",
                      100, 0.5, True, None, None, None))
            
            conn.commit()
            conn.close()
            
            # Get transitions
            transitions = insights.get_method_transitions()
            
            assert isinstance(transitions, dict)
            assert "method1" in transitions
            assert "method2" in transitions["method1"]
            assert "method3" in transitions["method1"]
            
        finally:
            os.unlink(db_path)
    
    def test_get_method_transitions_exception_handling(self):
        """Test getting method transitions when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            transitions = insights.get_method_transitions()
            assert transitions == {}
    
    def test_get_session_insights(self):
        """Test getting session insights."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            insights = WorkflowInsights(db_path=db_path)
            
            # Add some test data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert test events
            test_events = [
                ("session1", 1234567890.0, 1234567892.0, 2, 1.5, 2),
                ("session2", 1234567893.0, 1234567895.0, 1, 2.0, 1),
            ]
            
            for session_id, start_time, end_time, event_count, avg_exec_time, success_count in test_events:
                cursor.execute('''
                    INSERT INTO workflow_events
                    (timestamp, session_id, method_name, estimator_type, parameters,
                     array_size, fractional_order, execution_success, execution_time,
                     user_agent, ip_address)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (start_time, session_id, "method1", "riemann_liouville", "{}",
                      100, 0.5, True, avg_exec_time, None, None))
            
            conn.commit()
            conn.close()
            
            # Get session insights
            insights_data = insights.get_session_insights()
            
            assert "total_sessions" in insights_data
            assert "session_durations" in insights_data
            assert "event_counts" in insights_data
            assert "success_rates" in insights_data
            assert "avg_execution_times" in insights_data
            
        finally:
            os.unlink(db_path)
    
    def test_get_session_insights_exception_handling(self):
        """Test getting session insights when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            insights_data = insights.get_session_insights()
            assert insights_data == {}
    
    def test_get_user_behavior_clusters(self):
        """Test getting user behavior clusters."""
        with patch.object(WorkflowInsights, 'get_session_insights') as mock_insights:
            mock_insights.return_value = {
                'session_durations': {
                    'session1': 3600.0,  # 1 hour
                    'session2': 300.0,   # 5 minutes
                    'session3': 7200.0   # 2 hours
                }
            }
            
            insights = WorkflowInsights(enable_insights=False)
            
            # Test with empty session durations
            clusters = insights.get_user_behavior_clusters()
            assert "power_users" in clusters
            assert "regular_users" in clusters
            assert "casual_users" in clusters
    
    def test_get_user_behavior_clusters_exception_handling(self):
        """Test getting user behavior clusters when an exception occurs."""
        with patch.object(WorkflowInsights, 'get_session_insights', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            clusters = insights.get_user_behavior_clusters()
            assert clusters == {}
    
    def test_get_workflow_recommendations(self):
        """Test getting workflow recommendations."""
        with patch.object(WorkflowInsights, 'get_method_transitions') as mock_transitions:
            mock_transitions.return_value = {
                "method1": {"method2": 5, "method3": 3}
            }
            
            insights = WorkflowInsights(enable_insights=False)
            
            recommendations = insights.get_workflow_recommendations("method1", ["method1"])
            
            assert len(recommendations) == 2
            assert ("method2", 5/8) in recommendations
            assert ("method3", 3/8) in recommendations
    
    def test_get_workflow_recommendations_no_transitions(self):
        """Test getting workflow recommendations when no transitions exist."""
        with patch.object(WorkflowInsights, 'get_method_transitions') as mock_transitions:
            mock_transitions.return_value = {}
            
            insights = WorkflowInsights(enable_insights=False)
            
            recommendations = insights.get_workflow_recommendations("method1", ["method1"])
            assert recommendations == []
    
    def test_get_workflow_recommendations_exception_handling(self):
        """Test getting workflow recommendations when an exception occurs."""
        with patch.object(WorkflowInsights, 'get_method_transitions', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            recommendations = insights.get_workflow_recommendations("method1", ["method1"])
            assert recommendations == []
    
    def test_export_workflow_data(self):
        """Test exporting workflow data."""
        with patch.object(WorkflowInsights, 'get_workflow_patterns') as mock_patterns, \
             patch.object(WorkflowInsights, 'get_method_transitions') as mock_transitions, \
             patch.object(WorkflowInsights, 'get_session_insights') as mock_insights, \
             patch.object(WorkflowInsights, 'get_user_behavior_clusters') as mock_clusters, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_json_dump:
            
            mock_patterns.return_value = []
            mock_transitions.return_value = {}
            mock_insights.return_value = {}
            mock_clusters.return_value = {}
            
            insights = WorkflowInsights(enable_insights=False)
            
            result = insights.export_workflow_data("test_export.json")
            
            assert result is True
            mock_json_dump.assert_called_once()
    
    def test_export_workflow_data_exception_handling(self):
        """Test exporting workflow data when an exception occurs."""
        with patch.object(WorkflowInsights, 'get_workflow_patterns', side_effect=Exception("Export error")):
            insights = WorkflowInsights()
            
            result = insights.export_workflow_data("test_export.json")
            assert result is False
    
    def test_clear_old_data(self):
        """Test clearing old data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        try:
            insights = WorkflowInsights(db_path=db_path)
            
            # Add some test data
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert old event (more than 30 days ago)
            old_timestamp = time.time() - (31 * 24 * 3600)
            cursor.execute('''
                INSERT INTO workflow_events
                (timestamp, session_id, method_name, estimator_type, parameters,
                 array_size, fractional_order, execution_success, execution_time,
                 user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (old_timestamp, "session1", "method1", "riemann_liouville", "{}",
                  100, 0.5, True, None, None, None))
            
            conn.commit()
            conn.close()
            
            # Clear old data
            deleted_count = insights.clear_old_data(days_to_keep=30)
            
            assert deleted_count == 1
            
        finally:
            os.unlink(db_path)
    
    def test_clear_old_data_exception_handling(self):
        """Test clearing old data when an exception occurs."""
        with patch('sqlite3.connect', side_effect=Exception("Database error")):
            insights = WorkflowInsights()
            
            deleted_count = insights.clear_old_data()
            assert deleted_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
