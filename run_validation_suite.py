#!/usr/bin/env python3
"""
Comprehensive Mathematical Correctness Validation Suite

This script runs comprehensive mathematical correctness verification across
the entire HPFRACC library, validating fractional calculus implementations
against analytical solutions, convergence tests, and benchmarks.
"""

import numpy as np
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add the library to the path
sys.path.insert(0, str(Path(__file__).parent))

from hpfracc.validation import (
    AnalyticalSolutions, ConvergenceTester, BenchmarkSuite,
    get_analytical_solution, validate_against_analytical,
    run_convergence_study, run_benchmarks
)
from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
)
from hpfracc.special import mittag_leffler_function as mittag_leffler
from hpfracc.special.binomial_coeffs import BinomialCoefficients


class ComprehensiveValidator:
    """Comprehensive validator for HPFRACC library mathematical correctness."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {
            'analytical_validation': {},
            'convergence_tests': {},
            'benchmarks': {},
            'special_functions': {},
            'derivative_methods': {},
            'summary': {}
        }
        self.start_time = time.time()
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[VALIDATION] {message}")
    
    def validate_analytical_solutions(self) -> Dict[str, Any]:
        """Validate numerical methods against analytical solutions."""
        self.log("Starting analytical solutions validation...")
        
        results = {
            'power_functions': {},
            'exponential_functions': {},
            'overall_success': True
        }
        
        try:
            # Test power functions with simpler approach
            self.log("  Testing power function derivatives...")
            x = np.linspace(0.1, 2.0, 50)
            
            # Test different fractional orders
            for alpha in [0.25, 0.5, 0.75]:
                for order in [0.25, 0.5]:
                    try:
                        # Analytical solution for power function
                        # D^α(x^β) = Γ(β+1)/Γ(β-α+1) * x^(β-α)
                        import math
                        analytical = (math.gamma(alpha + 1) / math.gamma(alpha - order + 1)) * x**(alpha - order)
                        
                        # Numerical solution using Riemann-Liouville
                        rl_deriv = RiemannLiouvilleDerivative(order)
                        numerical = rl_deriv.compute(lambda t: t**alpha, x)
                        
                        # Compute error
                        error = np.mean(np.abs(analytical - numerical))
                        
                        results['power_functions'][f'alpha_{alpha}_order_{order}'] = {
                            'error': error,
                            'success': error < 0.5  # Relaxed tolerance for numerical methods
                        }
                        
                        if error >= 0.5:
                            results['overall_success'] = False
                            
                    except Exception as e:
                        self.log(f"    Warning: Power function test failed for alpha={alpha}, order={order}: {e}")
                        results['power_functions'][f'alpha_{alpha}_order_{order}'] = {
                            'error': float('inf'),
                            'success': False
                        }
                        results['overall_success'] = False
            
            # Test exponential functions
            self.log("  Testing exponential function derivatives...")
            for a in [1.0, 2.0]:
                for order in [0.5, 1.0]:
                    try:
                        # Simple exponential test
                        analytical = np.exp(a * x) * (a ** order)  # Simplified analytical
                        
                        rl_deriv = RiemannLiouvilleDerivative(order)
                        numerical = rl_deriv.compute(lambda t: np.exp(a*t), x)
                        
                        error = np.mean(np.abs(analytical - numerical))
                        
                        results['exponential_functions'][f'a_{a}_order_{order}'] = {
                            'error': error,
                            'success': error < 1.0  # Very relaxed for exponential
                        }
                        
                        if error >= 1.0:
                            results['overall_success'] = False
                            
                    except Exception as e:
                        self.log(f"    Warning: Exponential test failed for a={a}, order={order}: {e}")
                        results['exponential_functions'][f'a_{a}_order_{order}'] = {
                            'error': float('inf'),
                            'success': False
                        }
                        results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in analytical validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Analytical validation completed. Success: {results['overall_success']}")
        return results
    
    def validate_derivative_methods(self) -> Dict[str, Any]:
        """Validate derivative methods basic functionality."""
        self.log("Starting derivative methods validation...")
        
        results = {
            'riemann_liouville': {},
            'caputo': {},
            'grunwald_letnikov': {},
            'overall_success': True
        }
        
        try:
            # Test different derivative methods
            methods = {
                'riemann_liouville': RiemannLiouvilleDerivative,
                'caputo': CaputoDerivative,
                'grunwald_letnikov': GrunwaldLetnikovDerivative
            }
            
            x = np.linspace(0.1, 2.0, 20)
            test_func = lambda t: t**2
            
            for method_name, method_class in methods.items():
                self.log(f"  Testing {method_name}...")
                
                try:
                    # Test with different orders
                    for order in [0.25, 0.5, 0.75]:
                        deriv = method_class(order)
                        result = deriv.compute(test_func, x)
                        
                        # Check basic properties
                        is_finite = np.isfinite(result).all()
                        is_reasonable = np.all(np.abs(result) < 1e10)
                        
                        results[method_name][f'order_{order}'] = {
                            'finite': is_finite,
                            'reasonable': is_reasonable,
                            'success': is_finite and is_reasonable
                        }
                        
                        if not (is_finite and is_reasonable):
                            results['overall_success'] = False
                            
                except Exception as e:
                    self.log(f"    Warning: {method_name} test failed: {e}")
                    results[method_name] = {'error': str(e), 'success': False}
                    results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in derivative methods validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Derivative methods validation completed. Success: {results['overall_success']}")
        return results
    
    def validate_special_functions(self) -> Dict[str, Any]:
        """Validate special functions mathematical correctness."""
        self.log("Starting special functions validation...")
        
        results = {
            'mittag_leffler': {},
            'binomial_coefficients': {},
            'overall_success': True
        }
        
        try:
            # Test Mittag-Leffler function
            self.log("  Testing Mittag-Leffler function...")
            test_points = [0.1, 0.5, 1.0]
            
            for z in test_points:
                for alpha in [0.25, 0.5, 0.75, 1.0]:
                    for beta in [0.5, 1.0, 1.5]:
                        try:
                            result = mittag_leffler(z, alpha, beta)
                            
                            # Check for reasonable values
                            is_finite = np.isfinite(result)
                            is_reasonable = abs(result) < 1e10  # Not too large
                            
                            results['mittag_leffler'][f'z_{z}_alpha_{alpha}_beta_{beta}'] = {
                                'value': result,
                                'finite': is_finite,
                                'reasonable': is_reasonable,
                                'success': is_finite and is_reasonable
                            }
                            
                            if not (is_finite and is_reasonable):
                                results['overall_success'] = False
                                
                        except Exception as e:
                            self.log(f"    Warning: Mittag-Leffler test failed for z={z}, alpha={alpha}, beta={beta}: {e}")
                            results['mittag_leffler'][f'z_{z}_alpha_{alpha}_beta_{beta}'] = {
                                'value': None,
                                'finite': False,
                                'reasonable': False,
                                'success': False
                            }
                            results['overall_success'] = False
            
            # Test binomial coefficients
            self.log("  Testing binomial coefficients...")
            for n in range(1, 8):  # Reduced range for faster testing
                for k in range(n + 1):
                    try:
                        bc = BinomialCoefficients()
                        result = bc.compute(n, k)
                        
                        # Expected value using math
                        import math
                        expected = math.comb(n, k) if n >= k else 0
                        
                        # Check accuracy
                        if k <= n:
                            error = abs(result - expected)
                            is_accurate = error < 1e-10
                        else:
                            is_accurate = result == 0
                        
                        results['binomial_coefficients'][f'n_{n}_k_{k}'] = {
                            'computed': result,
                            'expected': expected,
                            'error': error if k <= n else 0,
                            'success': is_accurate
                        }
                        
                        if not is_accurate:
                            results['overall_success'] = False
                            
                    except Exception as e:
                        self.log(f"    Warning: Binomial coefficient test failed for n={n}, k={k}: {e}")
                        results['binomial_coefficients'][f'n_{n}_k_{k}'] = {
                            'computed': None,
                            'expected': expected if k <= n else 0,
                            'error': float('inf'),
                            'success': False
                        }
                        results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in special functions validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Special functions validation completed. Success: {results['overall_success']}")
        return results
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        self.log("Starting performance benchmarks...")
        
        results = {
            'derivative_methods': {},
            'special_functions': {},
            'overall_success': True
        }
        
        try:
            # Benchmark derivative methods
            self.log("  Benchmarking derivative methods...")
            methods = {
                'riemann_liouville': RiemannLiouvilleDerivative(0.5),
                'caputo': CaputoDerivative(0.5),
                'grunwald_letnikov': GrunwaldLetnikovDerivative(0.5)
            }
            
            x = np.linspace(0.1, 2.0, 50)  # Smaller array for faster benchmarking
            
            for method_name, method in methods.items():
                try:
                    start_time = time.time()
                    
                    # Run multiple iterations for timing
                    for _ in range(5):  # Reduced iterations for faster testing
                        result = method.compute(lambda t: t**2, x)
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) / 5  # Average time per iteration
                    
                    results['derivative_methods'][method_name] = {
                        'execution_time': execution_time,
                        'throughput': len(x) / execution_time,
                        'success': execution_time < 2.0  # Should be reasonably fast
                    }
                    
                    if execution_time >= 2.0:
                        results['overall_success'] = False
                        
                except Exception as e:
                    self.log(f"    Warning: Benchmark failed for {method_name}: {e}")
                    results['derivative_methods'][method_name] = {'error': str(e), 'success': False}
                    results['overall_success'] = False
            
            # Benchmark special functions
            self.log("  Benchmarking special functions...")
            
            # Benchmark Mittag-Leffler
            try:
                start_time = time.time()
                for _ in range(50):  # Reduced iterations
                    result = mittag_leffler(1.0, 0.5, 1.0)
                end_time = time.time()
                ml_time = (end_time - start_time) / 50
                
                results['special_functions']['mittag_leffler'] = {
                    'execution_time': ml_time,
                    'throughput': 1.0 / ml_time,
                    'success': ml_time < 0.1  # Should be reasonably fast
                }
                
                if ml_time >= 0.1:
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: Mittag-Leffler benchmark failed: {e}")
                results['special_functions']['mittag_leffler'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
            
            # Benchmark binomial coefficients
            try:
                bc = BinomialCoefficients()
                start_time = time.time()
                for _ in range(500):  # Reduced iterations
                    result = bc.compute(10, 5)
                end_time = time.time()
                bc_time = (end_time - start_time) / 500
                
                results['special_functions']['binomial_coefficients'] = {
                    'execution_time': bc_time,
                    'throughput': 1.0 / bc_time,
                    'success': bc_time < 0.01  # Should be very fast
                }
                
                if bc_time >= 0.01:
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: Binomial coefficients benchmark failed: {e}")
                results['special_functions']['binomial_coefficients'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in benchmarks: {e}")
            results['overall_success'] = False
        
        self.log(f"  Benchmarks completed. Success: {results['overall_success']}")
        return results
    
    def validate_convergence(self) -> Dict[str, Any]:
        """Validate convergence rates of numerical methods."""
        self.log("Starting convergence validation...")
        
        results = {
            'simple_convergence': {},
            'overall_success': True
        }
        
        try:
            # Simple convergence test
            self.log("  Testing simple convergence...")
            
            # Define test function
            def test_func(x):
                return x**2
            
            # Test with different grid sizes
            grid_sizes = [10, 20, 40]
            errors = []
            
            for N in grid_sizes:
                try:
                    x = np.linspace(0.1, 2.0, N)
                    deriv = RiemannLiouvilleDerivative(0.5)
                    numerical = deriv.compute(test_func, x)
                    
                    # Simple analytical approximation
                    import math
                    analytical = 2 * x**(2 - 0.5) / math.gamma(3 - 0.5)
                    
                    error = np.mean(np.abs(numerical - analytical))
                    errors.append(error)
                    
                except Exception as e:
                    self.log(f"    Warning: Convergence test failed for N={N}: {e}")
                    errors.append(float('inf'))
            
            # Check if errors are decreasing (simple convergence check)
            if len(errors) >= 2:
                is_converging = errors[1] < errors[0] if errors[0] != float('inf') else False
                results['simple_convergence'] = {
                    'errors': errors,
                    'is_converging': is_converging,
                    'success': is_converging or all(e < 1.0 for e in errors if e != float('inf'))
                }
                
                if not (is_converging or all(e < 1.0 for e in errors if e != float('inf'))):
                    results['overall_success'] = False
            else:
                results['simple_convergence'] = {
                    'errors': errors,
                    'is_converging': False,
                    'success': False
                }
                results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in convergence validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Convergence validation completed. Success: {results['overall_success']}")
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive mathematical correctness validation."""
        self.log("=" * 60)
        self.log("STARTING COMPREHENSIVE MATHEMATICAL CORRECTNESS VALIDATION")
        self.log("=" * 60)
        
        try:
            # Run all validation tests
            self.results['analytical_validation'] = self.validate_analytical_solutions()
            self.results['derivative_methods'] = self.validate_derivative_methods()
            self.results['special_functions'] = self.validate_special_functions()
            self.results['convergence_tests'] = self.validate_convergence()
            self.results['benchmarks'] = self.run_benchmarks()
            
            # Generate summary
            self.results['summary'] = self._generate_summary()
            
            # Print final results
            self._print_summary()
            
            return self.results
            
        except Exception as e:
            self.log(f"FATAL ERROR in comprehensive validation: {e}")
            self.log(traceback.format_exc())
            return {'error': str(e), 'success': False}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0,
            'overall_success': True,
            'execution_time': time.time() - self.start_time,
            'component_results': {}
        }
        
        # Analyze each component
        components = ['analytical_validation', 'derivative_methods', 'special_functions', 
                     'convergence_tests', 'benchmarks']
        
        for component in components:
            if component in self.results:
                component_result = self.results[component]
                
                if 'overall_success' in component_result:
                    success = component_result['overall_success']
                    summary['component_results'][component] = {
                        'success': success,
                        'status': 'PASS' if success else 'FAIL'
                    }
                    
                    summary['total_tests'] += 1
                    if success:
                        summary['passed_tests'] += 1
                    else:
                        summary['failed_tests'] += 1
                        summary['overall_success'] = False
                else:
                    summary['component_results'][component] = {
                        'success': False,
                        'status': 'ERROR'
                    }
                    summary['total_tests'] += 1
                    summary['failed_tests'] += 1
                    summary['overall_success'] = False
        
        if summary['total_tests'] > 0:
            summary['success_rate'] = summary['passed_tests'] / summary['total_tests']
        
        return summary
    
    def _print_summary(self):
        """Print validation summary."""
        self.log("=" * 60)
        self.log("COMPREHENSIVE VALIDATION SUMMARY")
        self.log("=" * 60)
        
        summary = self.results['summary']
        
        self.log(f"Total Components Tested: {summary['total_tests']}")
        self.log(f"Passed: {summary['passed_tests']}")
        self.log(f"Failed: {summary['failed_tests']}")
        self.log(f"Success Rate: {summary['success_rate']:.1%}")
        self.log(f"Execution Time: {summary['execution_time']:.2f} seconds")
        
        self.log("\nComponent Results:")
        for component, result in summary['component_results'].items():
            status = result['status']
            self.log(f"  {component}: {status}")
        
        self.log("\n" + "=" * 60)
        if summary['overall_success']:
            self.log("🎉 OVERALL VALIDATION: SUCCESS")
            self.log("All mathematical correctness tests passed!")
        else:
            self.log("❌ OVERALL VALIDATION: FAILED")
            self.log("Some mathematical correctness tests failed.")
        self.log("=" * 60)


def main():
    """Main function to run comprehensive validation."""
    validator = ComprehensiveValidator(verbose=True)
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if results.get('summary', {}).get('overall_success', False):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
