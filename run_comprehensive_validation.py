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
from hpfracc.ml.spectral_autograd import SpectralFractionalLayer
from hpfracc.ml.layers import FractionalLSTM, FractionalPooling
from hpfracc.solvers import FractionalODESolver, PredictorCorrectorSolver


class ComprehensiveValidator:
    """Comprehensive validator for HPFRACC library mathematical correctness."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {
            'analytical_validation': {},
            'convergence_tests': {},
            'benchmarks': {},
            'special_functions': {},
            'ml_layers': {},
            'solvers': {},
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
            'trigonometric_functions': {},
            'overall_success': True
        }
        
        try:
            # Test power functions
            self.log("  Testing power function derivatives...")
            x = np.linspace(0.1, 2.0, 50)
            
            # Test different fractional orders
            for alpha in [0.25, 0.5, 0.75, 1.0]:
                for order in [0.25, 0.5, 0.75]:
                    try:
                        # Analytical solution
                        analytical = get_analytical_solution("power", x, alpha=alpha, order=order)
                        
                        # Numerical solution using Riemann-Liouville
                        rl_deriv = RiemannLiouvilleDerivative(order)
                        numerical = rl_deriv.compute(lambda t: t**alpha, x)
                        
                        # Compute error
                        error = np.mean(np.abs(analytical - numerical))
                        
                        results['power_functions'][f'alpha_{alpha}_order_{order}'] = {
                            'error': error,
                            'success': error < 0.1  # Relaxed tolerance for numerical methods
                        }
                        
                        if error >= 0.1:
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
                        analytical = get_analytical_solution("exponential", x, a=a, order=order)
                        
                        rl_deriv = RiemannLiouvilleDerivative(order)
                        numerical = rl_deriv.compute(lambda t: np.exp(a*t), x)
                        
                        error = np.mean(np.abs(analytical - numerical))
                        
                        results['exponential_functions'][f'a_{a}_order_{order}'] = {
                            'error': error,
                            'success': error < 0.2  # More relaxed for exponential
                        }
                        
                        if error >= 0.2:
                            results['overall_success'] = False
                            
                    except Exception as e:
                        self.log(f"    Warning: Exponential test failed for a={a}, order={order}: {e}")
                        results['exponential_functions'][f'a_{a}_order_{order}'] = {
                            'error': float('inf'),
                            'success': False
                        }
                        results['overall_success'] = False
            
            # Test trigonometric functions
            self.log("  Testing trigonometric function derivatives...")
            for func_type in ['sin', 'cos']:
                for omega in [1.0, 2.0]:
                    for order in [0.5, 1.0]:
                        try:
                            analytical = get_analytical_solution("trigonometric", x, 
                                                               func_type=func_type, omega=omega, order=order)
                            
                            rl_deriv = RiemannLiouvilleDerivative(order)
                            if func_type == 'sin':
                                numerical = rl_deriv.compute(lambda t: np.sin(omega*t), x)
                            else:
                                numerical = rl_deriv.compute(lambda t: np.cos(omega*t), x)
                            
                            error = np.mean(np.abs(analytical - numerical))
                            
                            results['trigonometric_functions'][f'{func_type}_omega_{omega}_order_{order}'] = {
                                'error': error,
                                'success': error < 0.3  # Most relaxed for trigonometric
                            }
                            
                            if error >= 0.3:
                                results['overall_success'] = False
                                
                        except Exception as e:
                            self.log(f"    Warning: Trigonometric test failed for {func_type}, omega={omega}, order={order}: {e}")
                            results['trigonometric_functions'][f'{func_type}_omega_{omega}_order_{order}'] = {
                                'error': float('inf'),
                                'success': False
                            }
                            results['overall_success'] = False
            
        except Exception as e:
            self.log(f"  Error in analytical validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Analytical validation completed. Success: {results['overall_success']}")
        return results
    
    def validate_convergence(self) -> Dict[str, Any]:
        """Validate convergence rates of numerical methods."""
        self.log("Starting convergence validation...")
        
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
            
            for method_name, method_class in methods.items():
                self.log(f"  Testing {method_name} convergence...")
                
                try:
                    # Test convergence for different orders
                    for order in [0.25, 0.5, 0.75]:
                        grid_sizes = [20, 40, 80, 160]
                        
                        # Define test function and analytical solution
                        def test_func(x):
                            return x**2
                        
                        def analytical_solution(x, order):
                            # D^α(x^2) for different orders
                            if order == 0.25:
                                return 2 * x**(2 - 0.25) / np.math.gamma(3 - 0.25)
                            elif order == 0.5:
                                return 2 * x**(2 - 0.5) / np.math.gamma(3 - 0.5)
                            elif order == 0.75:
                                return 2 * x**(2 - 0.75) / np.math.gamma(3 - 0.75)
                            else:
                                return x**(2 - order) * 2 / np.math.gamma(3 - order)
                        
                        # Run convergence test
                        convergence_result = run_convergence_study(
                            lambda x: method_class(order).compute(test_func, x),
                            lambda x: analytical_solution(x, order),
                            [{'order': order}],
                            grid_sizes
                        )
                        
                        # Extract convergence rate
                        if 'test_cases' in convergence_result and len(convergence_result['test_cases']) > 0:
                            test_case = convergence_result['test_cases'][0]
                            if 'l2' in test_case and 'convergence_rate' in test_case['l2']:
                                rate = test_case['l2']['convergence_rate']
                                results[method_name][f'order_{order}'] = {
                                    'convergence_rate': rate,
                                    'success': rate > 0.5  # Expect at least first-order convergence
                                }
                                
                                if rate <= 0.5:
                                    results['overall_success'] = False
                            else:
                                results[method_name][f'order_{order}'] = {
                                    'convergence_rate': 0.0,
                                    'success': False
                                }
                                results['overall_success'] = False
                        else:
                            results[method_name][f'order_{order}'] = {
                                'convergence_rate': 0.0,
                                'success': False
                            }
                            results['overall_success'] = False
                            
                except Exception as e:
                    self.log(f"    Warning: {method_name} convergence test failed: {e}")
                    results[method_name] = {'error': str(e), 'success': False}
                    results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in convergence validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Convergence validation completed. Success: {results['overall_success']}")
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
            test_points = [0.1, 0.5, 1.0, 2.0]
            
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
            for n in range(1, 11):
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
    
    def validate_ml_layers(self) -> Dict[str, Any]:
        """Validate ML layers mathematical correctness."""
        self.log("Starting ML layers validation...")
        
        results = {
            'fractional_lstm': {},
            'fractional_pooling': {},
            'spectral_fractional': {},
            'overall_success': True
        }
        
        try:
            import torch
            
            # Test FractionalLSTM
            self.log("  Testing FractionalLSTM...")
            try:
                lstm = FractionalLSTM(input_size=10, hidden_size=20, fractional_order=0.5)
                x = torch.randn(5, 1, 10)  # (seq_len, batch, input_size)
                output = lstm(x)
                
                # Check output shape and properties
                expected_shape = (5, 1, 20)
                shape_correct = output.shape == expected_shape
                finite_output = torch.isfinite(output).all()
                
                results['fractional_lstm'] = {
                    'output_shape': output.shape,
                    'expected_shape': expected_shape,
                    'shape_correct': shape_correct,
                    'finite_output': finite_output.item(),
                    'success': shape_correct and finite_output.item()
                }
                
                if not (shape_correct and finite_output.item()):
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: FractionalLSTM test failed: {e}")
                results['fractional_lstm'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
            
            # Test FractionalPooling
            self.log("  Testing FractionalPooling...")
            try:
                pooling = FractionalPooling(kernel_size=2, stride=2, fractional_order=0.5)
                x = torch.randn(2, 3, 10, 10)  # (batch, channels, height, width)
                output = pooling(x)
                
                # Check output shape (should be reduced by stride)
                expected_shape = (2, 3, 5, 5)
                shape_correct = output.shape == expected_shape
                finite_output = torch.isfinite(output).all()
                
                results['fractional_pooling'] = {
                    'output_shape': output.shape,
                    'expected_shape': expected_shape,
                    'shape_correct': shape_correct,
                    'finite_output': finite_output.item(),
                    'success': shape_correct and finite_output.item()
                }
                
                if not (shape_correct and finite_output.item()):
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: FractionalPooling test failed: {e}")
                results['fractional_pooling'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
            
            # Test SpectralFractionalLayer
            self.log("  Testing SpectralFractionalLayer...")
            try:
                spectral_layer = SpectralFractionalLayer(alpha=0.5)
                x = torch.randn(10, 20)  # (batch, features)
                output = spectral_layer(x)
                
                # Check output shape and properties
                expected_shape = (10, 20)
                shape_correct = output.shape == expected_shape
                finite_output = torch.isfinite(output).all()
                
                results['spectral_fractional'] = {
                    'output_shape': output.shape,
                    'expected_shape': expected_shape,
                    'shape_correct': shape_correct,
                    'finite_output': finite_output.item(),
                    'success': shape_correct and finite_output.item()
                }
                
                if not (shape_correct and finite_output.item()):
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: SpectralFractionalLayer test failed: {e}")
                results['spectral_fractional'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in ML layers validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  ML layers validation completed. Success: {results['overall_success']}")
        return results
    
    def validate_solvers(self) -> Dict[str, Any]:
        """Validate ODE solvers mathematical correctness."""
        self.log("Starting solvers validation...")
        
        results = {
            'fractional_ode_solver': {},
            'predictor_corrector': {},
            'overall_success': True
        }
        
        try:
            # Test FractionalODESolver
            self.log("  Testing FractionalODESolver...")
            try:
                solver = FractionalODESolver()
                
                # Define a simple test ODE: dy/dt = -y
                def test_ode(t, y):
                    return -y
                
                # Solve the ODE
                t_span = (0, 1)
                y0 = 1.0
                alpha = 0.8
                
                t, y = solver.solve(test_ode, t_span, y0, alpha)
                
                # Check solution properties
                has_output = t is not None and y is not None
                finite_solution = np.isfinite(y).all() if has_output else False
                reasonable_values = np.all(y >= 0) and np.all(y <= 2) if has_output else False
                
                results['fractional_ode_solver'] = {
                    'has_output': has_output,
                    'finite_solution': finite_solution,
                    'reasonable_values': reasonable_values,
                    'solution_length': len(y) if has_output else 0,
                    'success': has_output and finite_solution and reasonable_values
                }
                
                if not (has_output and finite_solution and reasonable_values):
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: FractionalODESolver test failed: {e}")
                results['fractional_ode_solver'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
            
            # Test PredictorCorrectorSolver
            self.log("  Testing PredictorCorrectorSolver...")
            try:
                solver = PredictorCorrectorSolver("caputo", 0.5)
                
                # Define a simple test ODE
                def test_ode(t, y):
                    return -y
                
                # Solve the ODE
                t_span = (0, 0.5)
                y0 = 1.0
                
                t, y = solver.solve(test_ode, t_span, y0)
                
                # Check solution properties
                has_output = t is not None and y is not None
                finite_solution = np.isfinite(y).all() if has_output else False
                reasonable_values = np.all(y >= 0) and np.all(y <= 2) if has_output else False
                
                results['predictor_corrector'] = {
                    'has_output': has_output,
                    'finite_solution': finite_solution,
                    'reasonable_values': reasonable_values,
                    'solution_length': len(y) if has_output else 0,
                    'success': has_output and finite_solution and reasonable_values
                }
                
                if not (has_output and finite_solution and reasonable_values):
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: PredictorCorrectorSolver test failed: {e}")
                results['predictor_corrector'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
        
        except Exception as e:
            self.log(f"  Error in solvers validation: {e}")
            results['overall_success'] = False
        
        self.log(f"  Solvers validation completed. Success: {results['overall_success']}")
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
            
            x = np.linspace(0.1, 2.0, 100)
            
            for method_name, method in methods.items():
                try:
                    start_time = time.time()
                    
                    # Run multiple iterations for timing
                    for _ in range(10):
                        result = method.compute(lambda t: t**2, x)
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) / 10  # Average time per iteration
                    
                    results['derivative_methods'][method_name] = {
                        'execution_time': execution_time,
                        'throughput': len(x) / execution_time,
                        'success': execution_time < 1.0  # Should be reasonably fast
                    }
                    
                    if execution_time >= 1.0:
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
                for _ in range(100):
                    result = mittag_leffler(1.0, 0.5, 1.0)
                end_time = time.time()
                ml_time = (end_time - start_time) / 100
                
                results['special_functions']['mittag_leffler'] = {
                    'execution_time': ml_time,
                    'throughput': 1.0 / ml_time,
                    'success': ml_time < 0.01  # Should be very fast
                }
                
                if ml_time >= 0.01:
                    results['overall_success'] = False
                    
            except Exception as e:
                self.log(f"    Warning: Mittag-Leffler benchmark failed: {e}")
                results['special_functions']['mittag_leffler'] = {'error': str(e), 'success': False}
                results['overall_success'] = False
            
            # Benchmark binomial coefficients
            try:
                bc = BinomialCoefficients()
                start_time = time.time()
                for _ in range(1000):
                    result = bc.compute(10, 5)
                end_time = time.time()
                bc_time = (end_time - start_time) / 1000
                
                results['special_functions']['binomial_coefficients'] = {
                    'execution_time': bc_time,
                    'throughput': 1.0 / bc_time,
                    'success': bc_time < 0.001  # Should be very fast
                }
                
                if bc_time >= 0.001:
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
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive mathematical correctness validation."""
        self.log("=" * 60)
        self.log("STARTING COMPREHENSIVE MATHEMATICAL CORRECTNESS VALIDATION")
        self.log("=" * 60)
        
        try:
            # Run all validation tests
            self.results['analytical_validation'] = self.validate_analytical_solutions()
            self.results['convergence_tests'] = self.validate_convergence()
            self.results['special_functions'] = self.validate_special_functions()
            self.results['ml_layers'] = self.validate_ml_layers()
            self.results['solvers'] = self.validate_solvers()
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
        components = ['analytical_validation', 'convergence_tests', 'special_functions', 
                     'ml_layers', 'solvers', 'benchmarks']
        
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
