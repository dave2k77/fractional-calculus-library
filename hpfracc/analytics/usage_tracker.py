"""
Usage tracking system for hpfracc library.

Tracks which estimators, methods, and parameters are used most frequently
to identify popular usage patterns and optimization opportunities.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """Represents a single usage event."""
    timestamp: float
    method_name: str
    estimator_type: str
    parameters: Dict[str, Any]
    array_size: int
    fractional_order: float
    execution_success: bool
    user_session_id: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class UsageStats:
    """Aggregated usage statistics."""
    method_name: str
    total_calls: int
    success_rate: float
    avg_array_size: float
    common_fractional_orders: List[Tuple[float, int]]
    peak_usage_hours: List[Tuple[int, int]]
    user_sessions: int


class UsageTracker:
    """Tracks usage patterns of different estimators and methods."""

    def __init__(
            self,
            db_path: str = "usage_analytics.db",
            enable_tracking: bool = True):
        self.db_path = db_path
        self.enable_tracking = enable_tracking
        self.session_id = self._generate_session_id()
        self._setup_database()

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def _setup_database(self):
        """Initialize the SQLite database for usage tracking."""
        if not self.enable_tracking:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create usage events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    method_name TEXT NOT NULL,
                    estimator_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    array_size INTEGER NOT NULL,
                    fractional_order REAL NOT NULL,
                    execution_success BOOLEAN NOT NULL,
                    user_session_id TEXT,
                    ip_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indices for faster queries
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_method_name ON usage_events(method_name)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_events(timestamp)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_session_id ON usage_events(user_session_id)')

            conn.commit()
            conn.close()
            logger.info(
                f"Usage tracking database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize usage tracking database: {e}")
            self.enable_tracking = False

    def track_usage(self,
                    method_name: str,
                    estimator_type: str,
                    parameters: Dict[str,
                                     Any],
                    array_size: int,
                    fractional_order: float,
                    execution_success: bool,
                    user_session_id: Optional[str] = None,
                    ip_address: Optional[str] = None):
        """Track a usage event."""
        if not self.enable_tracking:
            return

        try:
            event = UsageEvent(
                timestamp=time.time(),
                method_name=method_name,
                estimator_type=estimator_type,
                parameters=parameters,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_success=execution_success,
                user_session_id=user_session_id or self.session_id,
                ip_address=ip_address
            )

            self._store_event(event)

        except Exception as e:
            logger.error(f"Failed to track usage event: {e}")

    def _store_event(self, event: UsageEvent):
        """Store a usage event in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO usage_events
                (timestamp, method_name, estimator_type, parameters, array_size,
                 fractional_order, execution_success, user_session_id, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.method_name,
                event.estimator_type,
                json.dumps(event.parameters),
                event.array_size,
                event.fractional_order,
                event.execution_success,
                event.user_session_id,
                event.ip_address
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store usage event: {e}")

    def get_usage_stats(
            self, time_window_hours: Optional[int] = None) -> Dict[str, UsageStats]:
        """Get aggregated usage statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build time filter
            time_filter = ""
            params = []
            if time_window_hours:
                cutoff_time = time.time() - (time_window_hours * 3600)
                time_filter = "WHERE timestamp > ?"
                params.append(cutoff_time)

            # Get all events
            query = f"SELECT * FROM usage_events {time_filter} ORDER BY timestamp"
            cursor.execute(query, params)
            events = cursor.fetchall()

            conn.close()

            # Process events into statistics
            return self._process_events_to_stats(events)

        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {}

    def _process_events_to_stats(
            self, events: List[Tuple]) -> Dict[str, UsageStats]:
        """Process raw database events into UsageStats objects."""
        method_stats = defaultdict(lambda: {
            'calls': [],
            'successes': 0,
            'array_sizes': [],
            'fractional_orders': [],
            'hours': [],
            'sessions': set()
        })

        for event in events:
            method_name = event[2]  # method_name column

            method_stats[method_name]['calls'].append(event)
            if event[7]:  # execution_success column
                method_stats[method_name]['successes'] += 1

            method_stats[method_name]['array_sizes'].append(
                event[5])  # array_size
            method_stats[method_name]['fractional_orders'].append(
                event[6])  # fractional_order

            # Extract hour from timestamp
            hour = int(time.localtime(event[1]).tm_hour)
            method_stats[method_name]['hours'].append(hour)

            # Track unique sessions
            if event[8]:  # user_session_id
                method_stats[method_name]['sessions'].add(event[8])

        # Convert to UsageStats objects
        stats = {}
        for method_name, data in method_stats.items():
            total_calls = len(data['calls'])
            success_rate = data['successes'] / \
                total_calls if total_calls > 0 else 0.0
            avg_array_size = sum(
                data['array_sizes']) / len(data['array_sizes']) if data['array_sizes'] else 0.0

            # Most common fractional orders
            order_counter = Counter(data['fractional_orders'])
            common_orders = order_counter.most_common(5)

            # Peak usage hours
            hour_counter = Counter(data['hours'])
            peak_hours = hour_counter.most_common(5)

            stats[method_name] = UsageStats(
                method_name=method_name,
                total_calls=total_calls,
                success_rate=success_rate,
                avg_array_size=avg_array_size,
                common_fractional_orders=common_orders,
                peak_usage_hours=peak_hours,
                user_sessions=len(data['sessions'])
            )

        return stats

    def get_popular_methods(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most popular methods by usage count."""
        stats = self.get_usage_stats()
        sorted_methods = sorted(
            [(name, stat.total_calls) for name, stat in stats.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_methods[:limit]

    def get_method_trends(self, method_name: str,
                          days: int = 7) -> List[Tuple[str, int]]:
        """Get usage trends for a specific method over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_time = time.time() - (days * 24 * 3600)

            cursor.execute('''
                SELECT DATE(datetime(timestamp, 'unixepoch')) as date, COUNT(*) as count
                FROM usage_events
                WHERE method_name = ? AND timestamp > ?
                GROUP BY DATE(datetime(timestamp, 'unixepoch'))
                ORDER BY date
            ''', (method_name, cutoff_time))

            trends = cursor.fetchall()
            conn.close()

            return [(date, count) for date, count in trends]

        except Exception as e:
            logger.error(f"Failed to get method trends: {e}")
            return []

    def export_usage_data(self, output_path: str = "usage_analytics.json"):
        """Export usage data to JSON format."""
        try:
            stats = self.get_usage_stats()

            # Convert dataclasses to dictionaries
            export_data = {
                'export_timestamp': time.time(),
                'total_methods': len(stats),
                'methods': {name: asdict(stat) for name, stat in stats.items()}
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Usage data exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export usage data: {e}")
            return False

    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old usage data to manage database size."""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'DELETE FROM usage_events WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Cleared {deleted_count} old usage events")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")
            return 0
