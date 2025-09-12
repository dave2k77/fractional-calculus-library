#!/usr/bin/env python3
"""
Performance Test Runner

This script runs performance regression tests and generates reports.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_performance.performance_config import get_performance_config
from tests.test_performance.performance_monitor import get_performance_monitor, get_performance_profiler


def run_performance_tests(test_pattern: str = None, verbose: bool = False) -> Dict[str, Any]:
    """Run performance regression tests."""
    import pytest
    
    # Set up test arguments
    test_args = [
        'tests/test_performance/test_performance_regression.py',
        '-v' if verbose else '-q',
        '--tb=short',
        '--durations=10'
    ]
    
    if test_pattern:
        test_args.extend(['-k', test_pattern])
    
    # Run tests
    result = pytest.main(test_args)
    
    return {
        'exit_code': result,
        'test_pattern': test_pattern,
        'timestamp': datetime.now().isoformat()
    }


def generate_performance_report() -> Dict[str, Any]:
    """Generate a comprehensive performance report."""
    config = get_performance_config()
    monitor = get_performance_monitor()
    profiler = get_performance_profiler()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'configuration': {
            'baselines_count': len(config.baselines),
            'thresholds': config.thresholds
        },
        'monitoring': {
            'metrics_count': len(monitor.metrics_history),
            'operations': list(set(entry['operation'] for entry in monitor.metrics_history))
        },
        'profiling': profiler.get_profile_summary()
    }
    
    return report


def update_baselines(force: bool = False) -> Dict[str, Any]:
    """Update performance baselines."""
    config = get_performance_config()
    
    if not force and config.baselines:
        print("Baselines already exist. Use --force to overwrite.")
        return {'status': 'skipped', 'reason': 'baselines_exist'}
    
    # Run tests to generate new baselines
    print("Running performance tests to generate baselines...")
    result = run_performance_tests(verbose=True)
    
    if result['exit_code'] == 0:
        print("Baselines updated successfully.")
        return {'status': 'success', 'result': result}
    else:
        print("Failed to update baselines.")
        return {'status': 'failed', 'result': result}


def export_performance_data(output_dir: str = "performance_data") -> Dict[str, Any]:
    """Export performance data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    config = get_performance_config()
    monitor = get_performance_monitor()
    profiler = get_performance_profiler()
    
    # Export baselines
    baselines_file = output_path / "baselines.json"
    config.export_baselines(str(baselines_file))
    
    # Export metrics
    metrics_file = output_path / "metrics.json"
    monitor.export_metrics(str(metrics_file))
    
    # Export profiles
    profiles_file = output_path / "profiles.json"
    profiler.export_profiles(str(profiles_file))
    
    # Generate report
    report = generate_performance_report()
    report_file = output_path / "performance_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return {
        'status': 'success',
        'output_directory': str(output_path),
        'files_created': [
            str(baselines_file),
            str(metrics_file),
            str(profiles_file),
            str(report_file)
        ]
    }


def main():
    """Main entry point for the performance test runner."""
    parser = argparse.ArgumentParser(description="Run performance regression tests")
    parser.add_argument('--test-pattern', '-k', help='Test pattern to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--update-baselines', action='store_true', help='Update performance baselines')
    parser.add_argument('--force', action='store_true', help='Force update baselines')
    parser.add_argument('--export', help='Export performance data to directory')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    
    args = parser.parse_args()
    
    if args.update_baselines:
        result = update_baselines(force=args.force)
        print(json.dumps(result, indent=2))
        return result.get('status') == 'success'
    
    if args.export:
        result = export_performance_data(args.export)
        print(json.dumps(result, indent=2))
        return result.get('status') == 'success'
    
    if args.report:
        report = generate_performance_report()
        print(json.dumps(report, indent=2))
        return True
    
    # Run performance tests
    result = run_performance_tests(
        test_pattern=args.test_pattern,
        verbose=args.verbose
    )
    
    print(f"Performance tests completed with exit code: {result['exit_code']}")
    return result['exit_code'] == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

