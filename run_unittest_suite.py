#!/usr/bin/env python3
"""
HPFRACC Unittest Test Suite Runner

This script provides a comprehensive test runner for the HPFRACC library
using Python's built-in unittest framework, bypassing pytest import issues.
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def discover_and_run_tests():
    """Discover and run all unittest tests"""
    print("=" * 80)
    print("HPFRACC UNITTEST TEST SUITE RUNNER")
    print("=" * 80)
    print()
    
    # Test discovery
    test_dir = project_root / "tests_unittest"
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    print(f"📁 Test directory: {test_dir}")
    print(f"🔍 Discovering tests...")
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = str(test_dir)
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"📊 Found {test_count} test cases")
    print()
    
    if test_count == 0:
        print("❌ No tests found!")
        return False
    
    # Run tests
    print("🚀 Running tests...")
    print("-" * 80)
    
    start_time = time.time()
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"📊 Tests run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Print failures
    if result.failures:
        print()
        print("❌ FAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"FAIL: {test}")
            print(traceback)
            print()
    
    # Print errors
    if result.errors:
        print()
        print("💥 ERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(traceback)
            print()
    
    # Overall result
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {len(result.failures + result.errors)} test(s) failed")
        return False

def run_specific_test_module(module_name):
    """Run tests for a specific module"""
    print(f"🎯 Running tests for module: {module_name}")
    print("-" * 80)
    
    try:
        # Import and run specific test module
        module = __import__(f"tests_unittest.{module_name}", fromlist=[module_name])
        suite = unittest.TestLoader().loadTestsFromModule(module)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"❌ Could not import test module {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running tests for {module_name}: {e}")
        return False

def run_coverage_tests():
    """Run tests with coverage reporting"""
    print("📊 Running tests with coverage reporting...")
    print("-" * 80)
    
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage(source=['hpfracc'])
        cov.start()
        
        # Run tests
        success = discover_and_run_tests()
        
        # Stop coverage
        cov.stop()
        
        # Generate report
        print()
        print("📊 COVERAGE REPORT:")
        print("-" * 40)
        cov.report(show_missing=True)
        
        # Save coverage data
        cov.save()
        print(f"💾 Coverage data saved")
        
        return success
    except ImportError:
        print("❌ Coverage module not available. Install with: pip install coverage")
        return discover_and_run_tests()
    except Exception as e:
        print(f"❌ Coverage error: {e}")
        return discover_and_run_tests()

def list_available_test_modules():
    """List all available test modules"""
    test_dir = project_root / "tests_unittest"
    if not test_dir.exists():
        print("❌ Test directory not found")
        return
    
    print("📋 Available test modules:")
    print("-" * 40)
    
    for test_file in test_dir.glob("test_*.py"):
        module_name = test_file.stem
        print(f"  • {module_name}")
    
    print()

def main():
    """Main function"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "coverage":
            success = run_coverage_tests()
        elif command == "list":
            list_available_test_modules()
            return
        elif command.startswith("module:"):
            module_name = command.split(":", 1)[1]
            success = run_specific_test_module(module_name)
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands:")
            print("  python run_unittest_suite.py           - Run all tests")
            print("  python run_unittest_suite.py coverage  - Run tests with coverage")
            print("  python run_unittest_suite.py list      - List available test modules")
            print("  python run_unittest_suite.py module:test_core - Run specific module")
            return
    else:
        success = discover_and_run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
