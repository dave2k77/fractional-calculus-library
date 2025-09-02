"""
Workflow insights system for hpfracc library.

Tracks common usage sequences, workflow patterns, and user behavior
to identify optimization opportunities and improve user experience.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkflowEvent:
    """Represents a single workflow event."""
    timestamp: float
    session_id: str
    method_name: str
    estimator_type: str
    parameters: Dict[str, Any]
    array_size: int
    fractional_order: float
    execution_success: bool
    execution_time: Optional[float] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class WorkflowPattern:
    """Represents a discovered workflow pattern."""
    pattern_id: str
    method_sequence: List[str]
    frequency: int
    avg_success_rate: float
    avg_execution_time: float
    common_parameters: Dict[str, Any]
    user_sessions: Set[str]
    first_seen: float
    last_seen: float


@dataclass
class WorkflowSummary:
    """Aggregated workflow insights summary."""
    total_sessions: int
    total_workflows: int
    common_patterns: List[WorkflowPattern]
    method_transitions: Dict[str, Dict[str, int]]
    session_durations: Dict[str, float]
    user_behavior_clusters: Dict[str, List[str]]


class WorkflowInsights:
    """Analyzes workflow patterns and user behavior."""

    def __init__(
            self,
            db_path: str = "workflow_analytics.db",
            enable_insights: bool = True):
        self.db_path = db_path
        self.enable_insights = enable_insights
        self._setup_database()

    def _setup_database(self):
        """Initialize the SQLite database for workflow tracking."""
        if not self.enable_insights:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create workflow events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    estimator_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    array_size INTEGER NOT NULL,
                    fractional_order REAL NOT NULL,
                    execution_success BOOLEAN NOT NULL,
                    execution_time REAL,
                    user_agent TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indices for faster queries
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_session_id ON workflow_events(session_id)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_timestamp ON workflow_events(timestamp)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_method_name ON workflow_events(method_name)')

            conn.commit()
            conn.close()
            logger.info(
                f"Workflow insights database initialized at {self.db_path}")

        except Exception as e:
            logger.error(
                f"Failed to initialize workflow insights database: {e}")
            self.enable_insights = False

    def track_workflow_event(self,
                             session_id: str,
                             method_name: str,
                             estimator_type: str,
                             parameters: Dict[str,
                                              Any],
                             array_size: int,
                             fractional_order: float,
                             execution_success: bool,
                             execution_time: Optional[float] = None,
                             user_agent: Optional[str] = None,
                             ip_address: Optional[str] = None):
        """Track a workflow event."""
        if not self.enable_insights:
            return

        try:
            event = WorkflowEvent(
                timestamp=time.time(),
                session_id=session_id,
                method_name=method_name,
                estimator_type=estimator_type,
                parameters=parameters,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_success=execution_success,
                execution_time=execution_time,
                user_agent=user_agent,
                ip_address=ip_address
            )

            self._store_workflow_event(event)

        except Exception as e:
            logger.error(f"Failed to track workflow event: {e}")

    def _store_workflow_event(self, event: WorkflowEvent):
        """Store a workflow event in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO workflow_events
                (timestamp, session_id, method_name, estimator_type, parameters,
                 array_size, fractional_order, execution_success, execution_time,
                 user_agent, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.session_id,
                event.method_name,
                event.estimator_type,
                json.dumps(event.parameters),
                event.array_size,
                event.fractional_order,
                event.execution_success,
                event.execution_time,
                event.user_agent,
                event.ip_address
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store workflow event: {e}")

    def get_workflow_patterns(
            self,
            min_frequency: int = 2,
            max_pattern_length: int = 5) -> List[WorkflowPattern]:
        """Discover common workflow patterns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all sessions with their method sequences
            cursor.execute('''
                SELECT session_id, method_name, timestamp, execution_success, execution_time
                FROM workflow_events
                ORDER BY session_id, timestamp
            ''')

            events = cursor.fetchall()
            conn.close()

            # Group events by session
            session_sequences = defaultdict(list)
            for event in events:
                session_id, method_name, timestamp, success, exec_time = event
                session_sequences[session_id].append({
                    'method': method_name,
                    'timestamp': timestamp,
                    'success': success,
                    'exec_time': exec_time
                })

            # Find patterns of different lengths
            patterns = []
            for length in range(2, max_pattern_length + 1):
                length_patterns = self._find_patterns_of_length(
                    session_sequences, length, min_frequency)
                patterns.extend(length_patterns)

            # Sort by frequency
            patterns.sort(key=lambda x: x.frequency, reverse=True)

            return patterns

        except Exception as e:
            logger.error(f"Failed to get workflow patterns: {e}")
            return []

    def _find_patterns_of_length(self,
                                 session_sequences: Dict[str,
                                                         List],
                                 pattern_length: int,
                                 min_frequency: int) -> List[WorkflowPattern]:
        """Find patterns of a specific length."""
        pattern_counts = defaultdict(lambda: {
            'count': 0,
            'sessions': set(),
            'successes': 0,
            'execution_times': [],
            'first_seen': float('inf'),
            'last_seen': 0.0,
            'parameters': defaultdict(int)
        })

        for session_id, sequence in session_sequences.items():
            if len(sequence) < pattern_length:
                continue

            # Check all possible subsequences of the given length
            for i in range(len(sequence) - pattern_length + 1):
                subsequence = sequence[i:i + pattern_length]
                method_sequence = tuple(event['method']
                                        for event in subsequence)

                # Create pattern key
                pattern_key = ' -> '.join(method_sequence)

                # Update pattern statistics
                pattern_counts[pattern_key]['count'] += 1
                pattern_counts[pattern_key]['sessions'].add(session_id)

                # Track success rate
                success_count = sum(
                    1 for event in subsequence if event['success'])
                pattern_counts[pattern_key]['successes'] += success_count

                # Track execution times
                exec_times = [event['exec_time']
                              for event in subsequence if event['exec_time']]
                pattern_counts[pattern_key]['execution_times'].extend(
                    exec_times)

                # Track timestamps
                first_timestamp = min(event['timestamp']
                                      for event in subsequence)
                last_timestamp = max(event['timestamp']
                                     for event in subsequence)
                pattern_counts[pattern_key]['first_seen'] = min(
                    pattern_counts[pattern_key]['first_seen'], first_timestamp
                )
                pattern_counts[pattern_key]['last_seen'] = max(
                    pattern_counts[pattern_key]['last_seen'], last_timestamp
                )

        # Convert to WorkflowPattern objects
        patterns = []
        for pattern_key, data in pattern_counts.items():
            if data['count'] >= min_frequency:
                method_sequence = pattern_key.split(' -> ')

                avg_success_rate = data['successes'] / \
                    (data['count'] * pattern_length)
                avg_execution_time = (sum(data['execution_times']) / len(
                    data['execution_times']) if data['execution_times'] else 0.0)

                pattern = WorkflowPattern(
                    pattern_id=pattern_key,
                    method_sequence=method_sequence,
                    frequency=data['count'],
                    avg_success_rate=avg_success_rate,
                    avg_execution_time=avg_execution_time,
                    common_parameters={},  # Simplified for now
                    user_sessions=data['sessions'],
                    first_seen=data['first_seen'],
                    last_seen=data['last_seen']
                )

                patterns.append(pattern)

        return patterns

    def get_method_transitions(self) -> Dict[str, Dict[str, int]]:
        """Analyze transitions between different methods."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get method sequences by session
            cursor.execute('''
                SELECT session_id, method_name, timestamp
                FROM workflow_events
                ORDER BY session_id, timestamp
            ''')

            events = cursor.fetchall()
            conn.close()

            # Build transition matrix
            transitions = defaultdict(lambda: defaultdict(int))

            # Group by session
            session_sequences = defaultdict(list)
            for event in events:
                session_id, method_name, timestamp = event
                session_sequences[session_id].append(method_name)

            # Count transitions
            for session_sequence in session_sequences.values():
                for i in range(len(session_sequence) - 1):
                    current_method = session_sequence[i]
                    next_method = session_sequence[i + 1]
                    transitions[current_method][next_method] += 1

            return dict(transitions)

        except Exception as e:
            logger.error(f"Failed to get method transitions: {e}")
            return {}

    def get_session_insights(self) -> Dict[str, Any]:
        """Get insights about user sessions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get session statistics
            cursor.execute('''
                SELECT
                    session_id,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time,
                    COUNT(*) as event_count,
                    AVG(execution_time) as avg_exec_time,
                    SUM(CASE WHEN execution_success THEN 1 ELSE 0 END) as success_count
                FROM workflow_events
                GROUP BY session_id
            ''')

            session_stats = cursor.fetchall()
            conn.close()

            # Process session statistics
            insights = {
                'total_sessions': len(session_stats),
                'session_durations': {},
                'event_counts': [],
                'success_rates': [],
                'avg_execution_times': []
            }

            for session_data in session_stats:
                session_id, start_time, end_time, event_count, avg_exec_time, success_count = session_data

                duration = end_time - start_time if start_time and end_time else 0.0
                insights['session_durations'][session_id] = duration
                insights['event_counts'].append(event_count)

                if event_count > 0:
                    success_rate = success_count / event_count
                    insights['success_rates'].append(success_rate)

                if avg_exec_time:
                    insights['avg_execution_times'].append(avg_exec_time)

            # Calculate summary statistics
            if insights['event_counts']:
                insights['avg_events_per_session'] = sum(
                    insights['event_counts']) / len(insights['event_counts'])
                insights['max_events_per_session'] = max(
                    insights['event_counts'])
                insights['min_events_per_session'] = min(
                    insights['event_counts'])

            if insights['success_rates']:
                insights['avg_success_rate'] = sum(
                    insights['success_rates']) / len(insights['success_rates'])

            if insights['avg_execution_times']:
                insights['avg_execution_time'] = sum(
                    insights['avg_execution_times']) / len(insights['avg_execution_times'])

            return insights

        except Exception as e:
            logger.error(f"Failed to get session insights: {e}")
            return {}

    def get_user_behavior_clusters(self) -> Dict[str, List[str]]:
        """Cluster users based on their behavior patterns."""
        try:
            # Get session insights
            session_insights = self.get_session_insights()

            if not session_insights.get('session_durations'):
                return {}

            # Simple clustering based on session duration and event count
            clusters = {
                'power_users': [],
                'regular_users': [],
                'casual_users': []
            }

            for session_id, duration in session_insights['session_durations'].items(
            ):
                # Find corresponding event count
                session_index = None
                if 'session_stats' not in locals():
                    session_stats = []
                for i, (sid, _, _, event_count, _,
                        _) in enumerate(session_stats):
                    if sid == session_id:
                        session_index = i
                        break

                if session_index is not None:
                    event_count = session_stats[session_index][3]

                    # Simple clustering logic
                    if duration > 3600 and event_count > 20:  # >1 hour and >20 events
                        clusters['power_users'].append(session_id)
                    elif duration > 600 and event_count > 5:   # >10 min and >5 events
                        clusters['regular_users'].append(session_id)
                    else:
                        clusters['casual_users'].append(session_id)

            return clusters

        except Exception as e:
            logger.error(f"Failed to cluster user behavior: {e}")
            return {}

    def get_workflow_recommendations(
            self, current_method: str, user_history: List[str]) -> List[Tuple[str, float]]:
        """Get workflow recommendations based on current context."""
        try:
            # Get method transitions
            transitions = self.get_method_transitions()

            if current_method not in transitions:
                return []

            # Get transition probabilities
            method_transitions = transitions[current_method]
            total_transitions = sum(method_transitions.values())

            if total_transitions == 0:
                return []

            # Calculate transition probabilities
            recommendations = []
            for next_method, count in method_transitions.items():
                probability = count / total_transitions
                recommendations.append((next_method, probability))

            # Sort by probability
            recommendations.sort(key=lambda x: x[1], reverse=True)

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get workflow recommendations: {e}")
            return []

    def export_workflow_data(
            self,
            output_path: str = "workflow_analytics.json"):
        """Export workflow data to JSON format."""
        try:
            # Convert dataclasses to dictionaries
            patterns = self.get_workflow_patterns()
            pattern_data = []
            for pattern in patterns:
                pattern_dict = asdict(pattern)
                pattern_dict['user_sessions'] = list(
                    pattern.user_sessions)  # Convert set to list
                pattern_data.append(pattern_dict)

            export_data = {
                'export_timestamp': time.time(),
                'workflow_patterns': pattern_data,
                'method_transitions': self.get_method_transitions(),
                'session_insights': self.get_session_insights(),
                'user_behavior_clusters': self.get_user_behavior_clusters()
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Workflow data exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export workflow data: {e}")
            return False

    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old workflow data to manage database size."""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'DELETE FROM workflow_events WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Cleared {deleted_count} old workflow events")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")
            return 0
