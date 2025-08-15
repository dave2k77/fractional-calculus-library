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
        [
            "python",
            "-m",
            "flake8",
            "src",
            "--count",
            "--select=E9,F63,F7,F82",
            "--show-source",
            "--statistics",
        ],
        "Flake8 syntax check",
    )

    if success:
        run_command(
            [
                "python",
                "-m",
                "flake8",
                "src",
                "--count",
                "--exit-zero",
                "--max-complexity=10",
                "--max-line-length=88",
                "--statistics",
            ],
            "Flake8 style check",
        )

    # Run black check
    run_command(
        ["python", "-m", "black", "--check", "src", "tests"], "Black formatting check"
    )

    # Run mypy
    run_command(
        ["python", "-m", "mypy", "src", "--ignore-missing-imports"],
        "MyPy type checking",
    )


def run_tests(test_type="all", coverage=True, verbose=True):
    """Run tests with specified options."""
    print(f"\n🧪 Running Tests ({test_type})...")

    cmd = ["python", "-m", "pytest", "tests/"]

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
    elif test_type == "algorithms":
        cmd.extend(["tests/test_algorithms/"])
    elif test_type == "consolidated":
        # Test the consolidated structure specifically
        cmd.extend(
            [
                "tests/test_algorithms/test_caputo.py",
                "tests/test_algorithms/test_riemann_liouville.py",
                "tests/test_algorithms/test_grunwald_letnikov.py",
                "tests/test_algorithms/test_fft_methods.py",
            ]
        )

    return run_command(cmd, f"Pytest {test_type} tests")


def run_benchmarks():
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks...")

    # Run pytest benchmarks
    success = run_command(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-m",
            "benchmark",
            "--benchmark-only",
            "--benchmark-sort=mean",
        ],
        "Pytest benchmarks",
    )

    if success:
        # Run custom benchmarks with updated imports
        run_command(
            ["python", "benchmarks/accuracy_comparisons.py"],
            "Accuracy comparisons benchmark",
        )

        run_command(
            ["python", "benchmarks/performance_tests.py"], "Performance tests benchmark"
        )

        run_command(
            ["python", "benchmarks/scaling_analysis.py"], "Scaling analysis benchmark"
        )


def run_examples():
    """Run example scripts to verify they work with consolidated structure."""
    print("\n📚 Running Examples...")

    examples = [
        "examples/basic_usage/getting_started.py",
        "examples/advanced_methods_demo.py",
        "examples/parallel_examples/parallel_computing_demo.py",
        "examples/jax_examples/jax_optimization_demo.py",
    ]

    for example in examples:
        if os.path.exists(example):
            run_command(["python", example], f"Running {example}")
        else:
            print(f"⚠️  Example file not found: {example}")


def main():
    """Main function to run tests and benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run tests and benchmarks for fractional calculus library"
    )
    parser.add_argument(
        "--test-type",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "benchmark",
            "fast",
            "gpu",
            "algorithms",
            "consolidated",
        ],
        help="Type of tests to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Disable verbose output"
    )
    parser.add_argument(
        "--lint-only", action="store_true", help="Run only linting checks"
    )
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Run only benchmarks"
    )
    parser.add_argument(
        "--examples-only", action="store_true", help="Run only examples"
    )

    args = parser.parse_args()

    print("🧪 Fractional Calculus Library Test Runner")
    print("=" * 60)
    print(f"📋 Test type: {args.test_type}")
    print(f"📊 Coverage: {'Disabled' if args.no_coverage else 'Enabled'}")
    print(f"🔍 Verbose: {'Disabled' if args.no_verbose else 'Enabled'}")
    print("=" * 60)

    if args.lint_only:
        run_linting()
        return

    if args.benchmark_only:
        run_benchmarks()
        return

    if args.examples_only:
        run_examples()
        return

    # Run linting first
    run_linting()

    # Run tests
    run_tests(args.test_type, not args.no_coverage, not args.no_verbose)

    # Run benchmarks
    run_benchmarks()

    # Run examples
    run_examples()

    print("\n🎉 All tests and benchmarks completed!")
    print("📁 Coverage report available in htmlcov/")
    print("📊 Performance results saved in benchmarks/")


if __name__ == "__main__":
    main()
