"""
Performance Configuration

This module provides configuration for performance regression testing,
including baseline management and performance thresholds.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class PerformanceConfig:
    """Configuration manager for performance regression testing."""
    
    def __init__(self, config_path: str = "performance_baselines.json"):
        self.config_path = Path(config_path)
        self.baselines = {}
        self.thresholds = {
            'derivative_computation': 0.15,  # 15% tolerance
            'optimized_methods': 0.10,       # 10% tolerance
            'neural_network_training': 0.20,  # 20% tolerance
            'tensor_operations': 0.10,       # 10% tolerance
            'memory_usage': 0.20,            # 20% tolerance
            'gpu_operations': 0.15,          # 15% tolerance
            'comprehensive': 0.20             # 20% tolerance
        }
        self.load_baselines()
    
    def load_baselines(self):
        """Load performance baselines from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.baselines = data.get('baselines', {})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load baselines: {e}")
                self.baselines = {}
    
    def save_baselines(self):
        """Save performance baselines to file."""
        data = {
            'baselines': self.baselines,
            'last_updated': datetime.now().isoformat(),
            'thresholds': self.thresholds
        }
        
        # Create directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_baseline(self, test_name: str) -> Optional[Dict[str, float]]:
        """Get baseline metrics for a test."""
        return self.baselines.get(test_name)
    
    def set_baseline(self, test_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for a test."""
        self.baselines[test_name] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.save_baselines()
    
    def get_threshold(self, test_name: str) -> float:
        """Get performance threshold for a test."""
        return self.thresholds.get(test_name, 0.15)
    
    def set_threshold(self, test_name: str, threshold: float):
        """Set performance threshold for a test."""
        self.thresholds[test_name] = threshold
        self.save_baselines()
    
    def check_regression(self, test_name: str, current_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if current performance has regressed from baseline."""
        baseline_data = self.get_baseline(test_name)
        if not baseline_data or 'metrics' not in baseline_data:
            return {}
        
        baseline_metrics = baseline_data['metrics']
        threshold = self.get_threshold(test_name)
        regression_flags = {}
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                # For time-based metrics, higher is worse (regression)
                # For throughput metrics, lower is worse (regression)
                if 'time' in metric or 'duration' in metric:
                    regression = current_value > baseline_value * (1 + threshold)
                else:
                    regression = current_value < baseline_value * (1 - threshold)
                regression_flags[metric] = regression
        
        return regression_flags
    
    def update_baseline_if_improved(self, test_name: str, current_metrics: Dict[str, float]):
        """Update baseline if current performance is significantly better."""
        baseline_data = self.get_baseline(test_name)
        if not baseline_data or 'metrics' not in baseline_data:
            self.set_baseline(test_name, current_metrics)
            return True
        
        baseline_metrics = baseline_data['metrics']
        threshold = self.get_threshold(test_name)
        improved = False
        
        for metric, current_value in current_metrics.items():
            if metric in baseline_metrics:
                baseline_value = baseline_metrics[metric]
                # For time-based metrics, lower is better
                # For throughput metrics, higher is better
                if 'time' in metric or 'duration' in metric:
                    if current_value < baseline_value * (1 - threshold):
                        improved = True
                        break
                else:
                    if current_value > baseline_value * (1 + threshold):
                        improved = True
                        break
        
        if improved:
            self.set_baseline(test_name, current_metrics)
            return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance baselines."""
        summary = {
            'total_tests': len(self.baselines),
            'last_updated': None,
            'tests': {}
        }
        
        for test_name, data in self.baselines.items():
            if 'timestamp' in data:
                summary['last_updated'] = data['timestamp']
            
            summary['tests'][test_name] = {
                'metrics_count': len(data.get('metrics', {})),
                'timestamp': data.get('timestamp', 'Unknown')
            }
        
        return summary
    
    def reset_baselines(self):
        """Reset all performance baselines."""
        self.baselines = {}
        self.save_baselines()
    
    def export_baselines(self, export_path: str):
        """Export baselines to a different file."""
        data = {
            'baselines': self.baselines,
            'last_updated': datetime.now().isoformat(),
            'thresholds': self.thresholds
        }
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_baselines(self, import_path: str):
        """Import baselines from a file."""
        if os.path.exists(import_path):
            with open(import_path, 'r') as f:
                data = json.load(f)
                self.baselines = data.get('baselines', {})
                self.thresholds.update(data.get('thresholds', {}))
                self.save_baselines()


# Global configuration instance
performance_config = PerformanceConfig()


def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration instance."""
    return performance_config


def set_performance_config(config: PerformanceConfig):
    """Set the global performance configuration instance."""
    global performance_config
    performance_config = config

