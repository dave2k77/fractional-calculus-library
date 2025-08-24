"""
Performance monitoring system for hpfracc library.

Tracks execution times, resource usage, and performance metrics
to identify bottlenecks and optimization opportunities.
"""

import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import json
import logging
from contextlib import contextmanager
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceEvent:
    """Represents a single performance measurement."""
    timestamp: float
    method_name: str
    estimator_type: str
    array_size: int
    fractional_order: float
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    gc_collections: int
    gc_time: float
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    method_name: str
    total_executions: int
    avg_execution_time: float
    std_execution_time: float
    min_execution_time: float
    max_execution_time: float
    avg_memory_usage: float
    avg_cpu_usage: float
    success_rate: float
    performance_percentiles: Dict[str, float]
    array_size_performance: Dict[int, float]


class PerformanceMonitor:
    """Monitors performance metrics of different estimators and methods."""
    
    def __init__(self, db_path: str = "performance_analytics.db", enable_monitoring: bool = True):
        self.db_path = db_path
        self.enable_monitoring = enable_monitoring
        self._setup_database()
        
    def _setup_database(self):
        """Initialize the SQLite database for performance tracking."""
        if not self.enable_monitoring:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    method_name TEXT NOT NULL,
                    estimator_type TEXT NOT NULL,
                    array_size INTEGER NOT NULL,
                    fractional_order REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_before REAL NOT NULL,
                    memory_after REAL NOT NULL,
                    memory_peak REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    gc_collections INTEGER NOT NULL,
                    gc_time REAL NOT NULL,
                    parameters TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_name ON performance_events(method_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_array_size ON performance_events(array_size)')
            
            conn.commit()
            conn.close()
            logger.info(f"Performance monitoring database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring database: {e}")
            self.enable_monitoring = False
    
    @contextmanager
    def monitor_performance(self, method_name: str, estimator_type: str, 
                          array_size: int, fractional_order: float, 
                          parameters: Dict[str, Any]):
        """Context manager for monitoring performance of a method execution."""
        if not self.enable_monitoring:
            yield
            return
        
        # Record initial state
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = process.cpu_percent()
        
        # GC monitoring
        gc_before = gc.get_count()
        gc_time_before = time.time()
        
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            # Record final state
            end_time = time.time()
            execution_time = end_time - start_time
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_peak = process.memory_info().rss / 1024 / 1024  # MB  # Simplified for now
            
            # GC monitoring
            gc_after = gc.get_count()
            gc_time_after = time.time()
            gc_collections = sum(gc_after) - sum(gc_before)
            gc_time = gc_time_after - gc_time_before
            
            # Create and store performance event
            event = PerformanceEvent(
                timestamp=start_time,
                method_name=method_name,
                estimator_type=estimator_type,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_percent=cpu_percent,
                gc_collections=gc_collections,
                gc_time=gc_time,
                parameters=parameters,
                success=success,
                error_message=error_message
            )
            
            self._store_performance_event(event)
    
    def _store_performance_event(self, event: PerformanceEvent):
        """Store a performance event in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_events 
                (timestamp, method_name, estimator_type, array_size, fractional_order,
                 execution_time, memory_before, memory_after, memory_peak, cpu_percent,
                 gc_collections, gc_time, parameters, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp,
                event.method_name,
                event.estimator_type,
                event.array_size,
                event.fractional_order,
                event.execution_time,
                event.memory_before,
                event.memory_after,
                event.memory_peak,
                event.cpu_percent,
                event.gc_collections,
                event.gc_time,
                json.dumps(event.parameters),
                event.success,
                event.error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store performance event: {e}")
    
    def get_performance_stats(self, time_window_hours: Optional[int] = None) -> Dict[str, PerformanceStats]:
        """Get aggregated performance statistics."""
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
            query = f"SELECT * FROM performance_events {time_filter} ORDER BY timestamp"
            cursor.execute(query, params)
            events = cursor.fetchall()
            
            conn.close()
            
            # Process events into statistics
            return self._process_events_to_stats(events)
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    def _process_events_to_stats(self, events: List[Tuple]) -> Dict[str, PerformanceStats]:
        """Process raw database events into PerformanceStats objects."""
        method_stats = defaultdict(lambda: {
            'execution_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'successes': 0,
            'array_sizes': defaultdict(list)
        })
        
        for event in events:
            method_name = event[2]  # method_name column
            
            method_stats[method_name]['execution_times'].append(event[6])  # execution_time
            memory_usage = event[8] - event[7]  # memory_after - memory_before
            method_stats[method_name]['memory_usage'].append(memory_usage)
            method_stats[method_name]['cpu_usage'].append(event[10])  # cpu_percent
            
            if event[14]:  # success column
                method_stats[method_name]['successes'] += 1
            
            array_size = event[4]  # array_size
            method_stats[method_name]['array_sizes'][array_size].append(event[6])  # execution_time
        
        # Convert to PerformanceStats objects
        stats = {}
        for method_name, data in method_stats.items():
            execution_times = data['execution_times']
            memory_usage = data['memory_usage']
            cpu_usage = data['cpu_usage']
            
            total_executions = len(execution_times)
            success_rate = data['successes'] / total_executions if total_executions > 0 else 0.0
            
            # Execution time statistics
            avg_execution_time = np.mean(execution_times)
            std_execution_time = np.std(execution_times)
            min_execution_time = np.min(execution_times)
            max_execution_time = np.max(execution_times)
            
            # Percentiles
            percentiles = [25, 50, 75, 90, 95, 99]
            performance_percentiles = {
                f"p{p}": np.percentile(execution_times, p) for p in percentiles
            }
            
            # Array size performance mapping
            array_size_performance = {}
            for size, times in data['array_sizes'].items():
                array_size_performance[size] = np.mean(times)
            
            stats[method_name] = PerformanceStats(
                method_name=method_name,
                total_executions=total_executions,
                avg_execution_time=avg_execution_time,
                std_execution_time=std_execution_time,
                min_execution_time=min_execution_time,
                max_execution_time=max_execution_time,
                avg_memory_usage=np.mean(memory_usage) if memory_usage else 0.0,
                avg_cpu_usage=np.mean(cpu_usage) if cpu_usage else 0.0,
                success_rate=success_rate,
                performance_percentiles=performance_percentiles,
                array_size_performance=array_size_performance
            )
        
        return stats
    
    def get_performance_trends(self, method_name: str, days: int = 7) -> List[Tuple[str, float]]:
        """Get performance trends for a specific method over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (days * 24 * 3600)
            
            cursor.execute('''
                SELECT DATE(datetime(timestamp, 'unixepoch')) as date, AVG(execution_time) as avg_time
                FROM performance_events 
                WHERE method_name = ? AND timestamp > ? AND success = 1
                GROUP BY DATE(datetime(timestamp, 'unixepoch'))
                ORDER BY date
            ''', (method_name, cutoff_time))
            
            trends = cursor.fetchall()
            conn.close()
            
            return [(date, avg_time) for date, avg_time in trends]
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return []
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks across methods."""
        try:
            stats = self.get_performance_stats()
            
            bottlenecks = {
                'slowest_methods': [],
                'memory_intensive_methods': [],
                'cpu_intensive_methods': [],
                'unreliable_methods': []
            }
            
            # Find slowest methods
            method_times = [(name, stat.avg_execution_time) for name, stat in stats.items()]
            method_times.sort(key=lambda x: x[1], reverse=True)
            bottlenecks['slowest_methods'] = method_times[:5]
            
            # Find memory intensive methods
            method_memory = [(name, stat.avg_memory_usage) for name, stat in stats.items()]
            method_memory.sort(key=lambda x: x[1], reverse=True)
            bottlenecks['memory_intensive_methods'] = method_memory[:5]
            
            # Find CPU intensive methods
            method_cpu = [(name, stat.avg_cpu_usage) for name, stat in stats.items()]
            method_cpu.sort(key=lambda x: x[1], reverse=True)
            bottlenecks['cpu_intensive_methods'] = method_cpu[:5]
            
            # Find unreliable methods (low success rate)
            method_reliability = [(name, stat.success_rate) for name, stat in stats.items()]
            method_reliability.sort(key=lambda x: x[1])
            bottlenecks['unreliable_methods'] = method_reliability[:5]
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Failed to analyze bottlenecks: {e}")
            return {}
    
    def export_performance_data(self, output_path: str = "performance_analytics.json"):
        """Export performance data to JSON format."""
        try:
            stats = self.get_performance_stats()
            
            # Convert dataclasses to dictionaries
            export_data = {
                'export_timestamp': time.time(),
                'total_methods': len(stats),
                'methods': {name: asdict(stat) for name, stat in stats.items()},
                'bottleneck_analysis': self.get_bottleneck_analysis()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")
            return False
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old performance data to manage database size."""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM performance_events WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted_count} old performance events")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")
            return 0
