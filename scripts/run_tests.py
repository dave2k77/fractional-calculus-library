#!/usr/bin/env python3
"""
Test runner script for the fractional calculus library.

This script provides a convenient way to run different types of tests
and generate reports.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def run_linting():
    """Run code linting checks."""
    print("\n🔍 Running Code Quality Checks...")
    
    # Run flake8
    success = run_command(
        ["flake8", "src", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
        "Flake8 syntax check"
    )
    
    if success:
        run_command(
            ["flake8", "src", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=88", "--statistics"],
            "Flake8 style check"
        )
    
    # Run black check
    run_command(
        ["black", "--check", "src", "tests"],
        "Black formatting check"
    )
    
    # Run mypy
    run_command(
        ["mypy", "src", "--ignore-missing-imports"],
        "MyPy type checking"
    )


def run_tests(test_type="all", coverage=True, verbose=True):
    """Run tests with specified options."""
    print(f"\n🧪 Running Tests ({test_type})...")
    
    cmd = ["pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    if test_type == "unit":
        cmd.extend(["-m", "not integration and not benchmark and not slow"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "benchmark":
        cmd.extend(["-m", "benchmark", "--benchmark-only"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow and not benchmark"])
    elif test_type == "gpu":
        cmd.extend(["-m", "gpu"])
    
    return run_command(cmd, f"Pytest {test_type} tests")


def run_benchmarks():
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks...")
    
    # Run pytest benchmarks
    success = run_command(
        ["pytest", "tests/", "-m", "benchmark", "--benchmark-only", "--benchmark-sort=mean"],
        "Pytest benchmarks"
    )
    
    if success:
        # Run custom benchmarks
        run_command(
            ["python", "benchmarks/accuracy_comparisons.py"],
            "Accuracy comparisons"
        )
        
        run_command(
            ["python", "benchmarks/performance_tests.py"],
            "Performance tests"
        )


def run_examples():
    """Run example scripts to ensure they work."""
    print("\n📚 Running Examples...")
    
    examples_dir = Path("examples")
    if examples_dir.exists():
        for example_file in examples_dir.rglob("*.py"):
            if example_file.name != "__init__.py":
                print(f"Running example: {example_file}")
                run_command(
                    ["python", str(example_file)],
                    f"Example: {example_file.name}"
                )


def generate_reports():
    """Generate test and coverage reports."""
    print("\n📊 Generating Reports...")
    
    # Generate coverage report
    run_command(
        ["coverage", "html", "--directory=htmlcov"],
        "Coverage HTML report"
    )
    
    # Generate test report
    run_command(
        ["pytest", "tests/", "--junitxml=test-results.xml", "--html=test-report.html"],
        "Test report generation"
    )


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run tests for fractional calculus library")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "benchmark", "fast", "gpu"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="Skip coverage reporting"
    )
    parser.add_argument(
        "--no-lint", 
        action="store_true",
        help="Skip linting checks"
    )
    parser.add_argument(
        "--no-examples", 
        action="store_true",
        help="Skip running examples"
    )
    parser.add_argument(
        "--reports", 
        action="store_true",
        help="Generate detailed reports"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    print("🚀 Fractional Calculus Library - Test Runner")
    print("=" * 60)
    
    success = True
    
    # Run linting
    if not args.no_lint:
        run_linting()
    
    # Run tests
    if not run_tests(args.type, not args.no_coverage, not args.quiet):
        success = False
    
    # Run benchmarks if requested
    if args.type in ["all", "benchmark"]:
        run_benchmarks()
    
    # Run examples
    if not args.no_examples:
        run_examples()
    
    # Generate reports
    if args.reports:
        generate_reports()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
