"""
Error analysis system for hpfracc library.

Tracks failure patterns, error types, and reliability scores
to identify common issues and improve system stability.
"""

import time
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from collections import defaultdict, Counter
import logging
import hashlib
import os
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """Represents a single error event."""
    timestamp: float
    method_name: str
    estimator_type: str
    error_type: str
    error_message: str
    error_traceback: str
    error_hash: str
    parameters: Dict[str, Any]
    array_size: int
    fractional_order: float
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    user_session_id: Optional[str] = None


@dataclass
class ErrorStats:
    """Aggregated error statistics."""
    method_name: str
    total_errors: int
    error_rate: float
    common_error_types: List[Tuple[str, int]]
    avg_execution_time_before_error: float
    common_parameters: List[Tuple[str, int]]
    reliability_score: float
    error_trends: List[Tuple[str, int]]


class ErrorAnalyzer:
    """Analyzes error patterns and reliability of different estimators and methods."""

    def __init__(
            self,
            db_path: str = "error_analytics.db",
            enable_analysis: bool = True):
        self.db_path = db_path
        self.enable_analysis = enable_analysis
        # Internal active path used for sqlite connections; may fall back to temp dir
        self._active_db_path = db_path
        self._setup_database()

    def _setup_database(self):
        """Initialize the SQLite database for error tracking."""
        if not self.enable_analysis:
            return

        # Ensure parent directory exists if provided
        try:
            parent_dir = os.path.dirname(self._active_db_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        except Exception:
            # Directory creation best-effort; continue to attempt sqlite open
            pass

        try:
            conn = sqlite3.connect(self._active_db_path)
            cursor = conn.cursor()

            # Create error events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    method_name TEXT NOT NULL,
                    estimator_type TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_traceback TEXT NOT NULL,
                    error_hash TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    array_size INTEGER NOT NULL,
                    fractional_order REAL NOT NULL,
                    execution_time REAL,
                    memory_usage REAL,
                    user_session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indices for faster queries
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_method_name ON error_events(method_name)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_error_type ON error_events(error_type)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_timestamp ON error_events(timestamp)')
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_error_hash ON error_events(error_hash)')

            conn.commit()
            conn.close()
            logger.info(
                f"Error analysis database initialized at {self._active_db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize error analysis database: {e}")
            # Attempt fallback to a writable temp directory without changing public db_path
            try:
                fallback_dir = tempfile.gettempdir()
                fallback_path = os.path.join(fallback_dir, os.path.basename(self.db_path))
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                conn = sqlite3.connect(fallback_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS error_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        method_name TEXT NOT NULL,
                        estimator_type TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        error_traceback TEXT NOT NULL,
                        error_hash TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        array_size INTEGER NOT NULL,
                        fractional_order REAL NOT NULL,
                        execution_time REAL,
                        memory_usage REAL,
                        user_session_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_name ON error_events(method_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_type ON error_events(error_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON error_events(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_hash ON error_events(error_hash)')
                conn.commit()
                conn.close()
                self._active_db_path = fallback_path
                logger.info(f"Error analysis database initialized at fallback path {self._active_db_path}")
            except Exception as e2:
                logger.error(f"Fallback database initialization failed: {e2}")
                self.enable_analysis = False

    def track_error(self,
                    method_name: str,
                    estimator_type: str,
                    error: Exception,
                    parameters: Dict[str,
                                     Any],
                    array_size: int,
                    fractional_order: float,
                    execution_time: Optional[float] = None,
                    memory_usage: Optional[float] = None,
                    user_session_id: Optional[str] = None):
        """Track an error event."""
        if not self.enable_analysis:
            return

        try:
            # Generate error hash for deduplication
            error_hash = self._generate_error_hash(
                error, method_name, parameters)

            event = ErrorEvent(
                timestamp=time.time(),
                method_name=method_name,
                estimator_type=estimator_type,
                error_type=type(error).__name__,
                error_message=str(error),
                error_traceback=traceback.format_exc(),
                error_hash=error_hash,
                parameters=parameters,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_time=execution_time,
                memory_usage=memory_usage,
                user_session_id=user_session_id
            )

            self._store_error_event(event)

        except Exception as e:
            logger.error(f"Failed to track error event: {e}")

    def _generate_error_hash(self,
                             error: Exception,
                             method_name: str,
                             parameters: Dict[str,
                                              Any]) -> str:
        """Generate a hash for error deduplication."""
        error_info = f"{type(error).__name__}:{str(error)}:{method_name}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.md5(error_info.encode()).hexdigest()

    def _store_error_event(self, event: ErrorEvent):
        """Store an error event in the database."""
        try:
            conn = sqlite3.connect(self._active_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO error_events
                (timestamp, method_name, estimator_type, error_type, error_message,
                 error_traceback, error_hash, parameters, array_size, fractional_order,
                 execution_time, memory_usage, user_session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.method_name,
                event.estimator_type,
                event.error_type,
                event.error_message,
                event.error_traceback,
                event.error_hash,
                json.dumps(event.parameters),
                event.array_size,
                event.fractional_order,
                event.execution_time,
                event.memory_usage,
                event.user_session_id
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store error event: {e}")

    def get_error_stats(
            self, time_window_hours: Optional[int] = None) -> Dict[str, ErrorStats]:
        """Get aggregated error statistics."""
        try:
            conn = sqlite3.connect(self._active_db_path)
            cursor = conn.cursor()

            # Build time filter
            time_filter = ""
            params = []
            if time_window_hours:
                cutoff_time = time.time() - (time_window_hours * 3600)
                time_filter = "WHERE timestamp > ?"
                params.append(cutoff_time)

            # Get all error events
            query = f"SELECT * FROM error_events {time_filter} ORDER BY timestamp"
            cursor.execute(query, params)
            events = cursor.fetchall()

            conn.close()

            # Process events into statistics
            return self._process_events_to_stats(events)

        except Exception as e:
            logger.error(f"Failed to get error stats: {e}")
            return {}

    def _process_events_to_stats(
            self, events: List[Tuple]) -> Dict[str, ErrorStats]:
        """Process raw database events into ErrorStats objects."""
        method_stats = defaultdict(lambda: {
            'errors': [],
            'error_types': [],
            'execution_times': [],
            'parameters': [],
            'dates': []
        })

        for event in events:
            method_name = event[2]  # method_name column

            method_stats[method_name]['errors'].append(event)
            method_stats[method_name]['error_types'].append(
                event[4])  # error_type

            if event[11]:  # execution_time
                method_stats[method_name]['execution_times'].append(event[11])

            # Track parameter combinations
            param_str = json.dumps(event[8], sort_keys=True)  # parameters
            method_stats[method_name]['parameters'].append(param_str)

            # Track dates for trends
            date = time.strftime('%Y-%m-%d', time.localtime(event[1]))
            method_stats[method_name]['dates'].append(date)

        # Convert to ErrorStats objects
        stats = {}
        for method_name, data in method_stats.items():
            total_errors = len(data['errors'])

            # Error type analysis
            error_type_counter = Counter(data['error_types'])
            common_error_types = error_type_counter.most_common(5)

            # Parameter analysis
            param_counter = Counter(data['parameters'])
            common_parameters = param_counter.most_common(5)

            # Execution time analysis
            avg_execution_time = 0.0
            if data['execution_times']:
                avg_execution_time = sum(
                    data['execution_times']) / len(data['execution_times'])

            # Date trends
            date_counter = Counter(data['dates'])
            error_trends = sorted(date_counter.items())

            # Calculate reliability score (simplified)
            # This would ideally be compared against total usage
            reliability_score = max(
                0.0, 1.0 - (total_errors / 1000))  # Normalize to 0-1

            stats[method_name] = ErrorStats(
                method_name=method_name,
                total_errors=total_errors,
                error_rate=total_errors / 1000.0,  # Normalized rate
                common_error_types=common_error_types,
                avg_execution_time_before_error=avg_execution_time,
                common_parameters=common_parameters,
                reliability_score=reliability_score,
                error_trends=error_trends
            )

        return stats

    def get_error_trends(self, method_name: str,
                         days: int = 7) -> List[Tuple[str, int]]:
        """Get error trends for a specific method over time."""
        try:
            conn = sqlite3.connect(self._active_db_path)
            cursor = conn.cursor()

            cutoff_time = time.time() - (days * 24 * 3600)

            cursor.execute('''
                SELECT DATE(datetime(timestamp, 'unixepoch')) as date, COUNT(*) as count
                FROM error_events
                WHERE method_name = ? AND timestamp > ?
                GROUP BY DATE(datetime(timestamp, 'unixepoch'))
                ORDER BY date
            ''', (method_name, cutoff_time))

            trends = cursor.fetchall()
            conn.close()

            return [(date, count) for date, count in trends]

        except Exception as e:
            logger.error(f"Failed to get error trends: {e}")
            return []

    def get_common_error_patterns(self) -> Dict[str, Any]:
        """Analyze common error patterns across methods."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get most common error types
            cursor.execute('''
                SELECT error_type, COUNT(*) as count
                FROM error_events
                GROUP BY error_type
                ORDER BY count DESC
                LIMIT 10
            ''')

            common_error_types = cursor.fetchall()

            # Get most common error messages
            cursor.execute('''
                SELECT error_message, COUNT(*) as count
                FROM error_events
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 10
            ''')

            common_error_messages = cursor.fetchall()

            # Get error-prone parameter combinations
            cursor.execute('''
                SELECT parameters, COUNT(*) as count
                FROM error_events
                GROUP BY parameters
                ORDER BY count DESC
                LIMIT 10
            ''')

            error_prone_parameters = cursor.fetchall()

            conn.close()

            return {
                'common_error_types': common_error_types,
                'common_error_messages': common_error_messages,
                'error_prone_parameters': error_prone_parameters
            }

        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return {}

    def get_reliability_ranking(self) -> List[Tuple[str, float]]:
        """Get methods ranked by reliability score."""
        try:
            stats = self.get_error_stats()

            reliability_scores = [(name, stat.reliability_score)
                                  for name, stat in stats.items()]
            reliability_scores.sort(key=lambda x: x[1], reverse=True)

            return reliability_scores

        except Exception as e:
            logger.error(f"Failed to get reliability ranking: {e}")
            return []

    def get_error_correlation_analysis(self) -> Dict[str, Any]:
        """Analyze correlations between errors and other factors."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Error correlation with array size
            cursor.execute('''
                SELECT array_size, COUNT(*) as error_count
                FROM error_events
                GROUP BY array_size
                ORDER BY array_size
            ''')

            array_size_correlation = cursor.fetchall()

            # Error correlation with fractional order
            cursor.execute('''
                SELECT fractional_order, COUNT(*) as error_count
                FROM error_events
                GROUP BY fractional_order
                ORDER BY fractional_order
            ''')

            fractional_order_correlation = cursor.fetchall()

            # Error correlation with execution time
            cursor.execute('''
                SELECT
                    CASE
                        WHEN execution_time < 0.1 THEN 'fast'
                        WHEN execution_time < 1.0 THEN 'medium'
                        ELSE 'slow'
                    END as speed_category,
                    COUNT(*) as error_count
                FROM error_events
                WHERE execution_time IS NOT NULL
                GROUP BY speed_category
            ''')

            execution_time_correlation = cursor.fetchall()

            conn.close()

            return {
                'array_size_correlation': array_size_correlation,
                'fractional_order_correlation': fractional_order_correlation,
                'execution_time_correlation': execution_time_correlation
            }

        except Exception as e:
            logger.error(f"Failed to analyze error correlations: {e}")
            return {}

    def export_error_data(self, output_path: str = "error_analytics.json"):
        """Export error data to JSON format."""
        try:
            stats = self.get_error_stats()

            # Convert dataclasses to dictionaries
            export_data = {
                'export_timestamp': time.time(),
                'total_methods_with_errors': len(stats),
                'methods': {
                    name: asdict(stat) for name,
                    stat in stats.items()},
                'common_error_patterns': self.get_common_error_patterns(),
                'reliability_ranking': self.get_reliability_ranking(),
                'error_correlation_analysis': self.get_error_correlation_analysis()}

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Error data exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export error data: {e}")
            return False

    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old error data to manage database size."""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                'DELETE FROM error_events WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            logger.info(f"Cleared {deleted_count} old error events")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")
            return 0
