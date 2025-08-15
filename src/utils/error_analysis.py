"""
Error analysis and validation tools for fractional calculus computations.

This module provides tools for:
- Computing error metrics between numerical and analytical solutions
- Analyzing convergence rates
- Validating numerical methods against known solutions
- Error estimation and bounds
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings


class ErrorAnalyzer:
    """Analyzer for computing various error metrics."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the error analyzer.
        
        Args:
            tolerance: Numerical tolerance for avoiding division by zero
        """
        self.tolerance = tolerance
    
    def absolute_error(self, numerical: np.ndarray, analytical: np.ndarray) -> np.ndarray:
        """
        Compute absolute error between numerical and analytical solutions.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            Absolute error array
        """
        return np.abs(numerical - analytical)
    
    def relative_error(self, numerical: np.ndarray, analytical: np.ndarray) -> np.ndarray:
        """
        Compute relative error between numerical and analytical solutions.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            Relative error array
        """
        # Avoid division by zero
        denominator = np.abs(analytical)
        denominator = np.where(denominator < self.tolerance, self.tolerance, denominator)
        return np.abs(numerical - analytical) / denominator
    
    def l1_error(self, numerical: np.ndarray, analytical: np.ndarray) -> float:
        """
        Compute L1 error norm.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            L1 error norm
        """
        return np.mean(np.abs(numerical - analytical))
    
    def l2_error(self, numerical: np.ndarray, analytical: np.ndarray) -> float:
        """
        Compute L2 error norm.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            L2 error norm
        """
        return np.sqrt(np.mean((numerical - analytical) ** 2))
    
    def linf_error(self, numerical: np.ndarray, analytical: np.ndarray) -> float:
        """
        Compute L-infinity error norm.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            L-infinity error norm
        """
        return np.max(np.abs(numerical - analytical))
    
    def compute_all_errors(self, numerical: np.ndarray, analytical: np.ndarray) -> Dict[str, float]:
        """
        Compute all error metrics.
        
        Args:
            numerical: Numerical solution array
            analytical: Analytical solution array
            
        Returns:
            Dictionary containing all error metrics
        """
        return {
            'l1': self.l1_error(numerical, analytical),
            'l2': self.l2_error(numerical, analytical),
            'linf': self.linf_error(numerical, analytical),
            'mean_absolute': np.mean(self.absolute_error(numerical, analytical)),
            'mean_relative': np.mean(self.relative_error(numerical, analytical)),
        }


class ConvergenceAnalyzer:
    """Analyzer for studying convergence rates of numerical methods."""
    
    def __init__(self):
        """Initialize the convergence analyzer."""
        pass
    
    def compute_convergence_rate(self, grid_sizes: List[int], errors: List[float]) -> float:
        """
        Compute convergence rate using linear regression on log-log plot.
        
        Args:
            grid_sizes: List of grid sizes (N)
            errors: List of corresponding errors
            
        Returns:
            Convergence rate (order of accuracy)
        """
        if len(grid_sizes) < 2 or len(errors) < 2:
            raise ValueError("Need at least 2 points to compute convergence rate")
        
        # Convert to log space
        log_n = np.log(np.array(grid_sizes))
        log_error = np.log(np.array(errors))
        
        # Linear regression: log(error) = -p * log(N) + C
        # where p is the convergence rate
        coeffs = np.polyfit(log_n, log_error, 1)
        convergence_rate = -coeffs[0]  # Negative because error decreases with N
        
        return convergence_rate
    
    def analyze_convergence(self, grid_sizes: List[int], errors: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Analyze convergence for multiple error metrics.
        
        Args:
            grid_sizes: List of grid sizes
            errors: Dictionary of error lists for different metrics
            
        Returns:
            Dictionary of convergence rates for each metric
        """
        convergence_rates = {}
        
        for metric, error_list in errors.items():
            try:
                convergence_rates[metric] = self.compute_convergence_rate(grid_sizes, error_list)
            except (ValueError, np.linalg.LinAlgError) as e:
                warnings.warn(f"Could not compute convergence rate for {metric}: {e}")
                convergence_rates[metric] = np.nan
        
        return convergence_rates
    
    def estimate_optimal_grid_size(self, target_error: float, grid_sizes: List[int], 
                                 errors: List[float]) -> int:
        """
        Estimate optimal grid size for a target error.
        
        Args:
            target_error: Target error tolerance
            grid_sizes: List of grid sizes used in convergence study
            errors: List of corresponding errors
            
        Returns:
            Estimated optimal grid size
        """
        if len(grid_sizes) < 2:
            raise ValueError("Need at least 2 points to estimate optimal grid size")
        
        convergence_rate = self.compute_convergence_rate(grid_sizes, errors)
        
        # Use the last point as reference
        N_ref = grid_sizes[-1]
        error_ref = errors[-1]
        
        # Estimate: N_opt = N_ref * (error_ref / target_error)^(1/p)
        N_opt = int(N_ref * (error_ref / target_error) ** (1 / convergence_rate))
        
        return max(N_opt, 1)  # Ensure positive grid size


class ValidationFramework:
    """Framework for validating numerical methods against analytical solutions."""
    
    def __init__(self, error_analyzer: Optional[ErrorAnalyzer] = None):
        """
        Initialize the validation framework.
        
        Args:
            error_analyzer: Error analyzer instance (optional)
        """
        self.error_analyzer = error_analyzer or ErrorAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()
    
    def validate_method(self, method_func: Callable, analytical_func: Callable,
                       test_cases: List[Dict], grid_sizes: List[int] = None) -> Dict:
        """
        Validate a numerical method against analytical solutions.
        
        Args:
            method_func: Function that computes numerical solution
            analytical_func: Function that computes analytical solution
            test_cases: List of test case dictionaries
            grid_sizes: List of grid sizes for convergence study
            
        Returns:
            Validation results dictionary
        """
        if grid_sizes is None:
            grid_sizes = [50, 100, 200, 400]
        
        results = {
            'test_cases': [],
            'convergence_study': {},
            'overall_summary': {}
        }
        
        # Test individual cases
        for i, test_case in enumerate(test_cases):
            case_result = self._validate_single_case(method_func, analytical_func, test_case)
            results['test_cases'].append(case_result)
        
        # Convergence study
        if len(test_cases) > 0:
            convergence_result = self._convergence_study(method_func, analytical_func, 
                                                       test_cases[0], grid_sizes)
            results['convergence_study'] = convergence_result
        
        # Overall summary
        results['overall_summary'] = self._compute_summary(results['test_cases'])
        
        return results
    
    def _validate_single_case(self, method_func: Callable, analytical_func: Callable,
                             test_case: Dict) -> Dict:
        """Validate a single test case."""
        try:
            # Compute numerical solution
            numerical = method_func(**test_case['params'])
            
            # Compute analytical solution
            analytical = analytical_func(**test_case['params'])
            
            # Compute errors
            errors = self.error_analyzer.compute_all_errors(numerical, analytical)
            
            return {
                'case_name': test_case.get('name', f'Case_{len(test_case)}'),
                'success': True,
                'errors': errors,
                'numerical_shape': numerical.shape,
                'analytical_shape': analytical.shape,
            }
            
        except Exception as e:
            return {
                'case_name': test_case.get('name', f'Case_{len(test_case)}'),
                'success': False,
                'error': str(e),
                'errors': None,
            }
    
    def _convergence_study(self, method_func: Callable, analytical_func: Callable,
                          test_case: Dict, grid_sizes: List[int]) -> Dict:
        """Perform convergence study."""
        errors_by_metric = {'l2': [], 'linf': []}
        
        for N in grid_sizes:
            try:
                # Update test case with new grid size
                params = test_case['params'].copy()
                params['N'] = N
                
                numerical = method_func(**params)
                analytical = analytical_func(**params)
                
                errors = self.error_analyzer.compute_all_errors(numerical, analytical)
                errors_by_metric['l2'].append(errors['l2'])
                errors_by_metric['linf'].append(errors['linf'])
                
            except Exception as e:
                errors_by_metric['l2'].append(np.nan)
                errors_by_metric['linf'].append(np.nan)
        
        # Compute convergence rates
        convergence_rates = self.convergence_analyzer.analyze_convergence(
            grid_sizes, errors_by_metric
        )
        
        return {
            'grid_sizes': grid_sizes,
            'errors': errors_by_metric,
            'convergence_rates': convergence_rates,
        }
    
    def _compute_summary(self, test_cases: List[Dict]) -> Dict:
        """Compute overall summary of validation results."""
        successful_cases = [case for case in test_cases if case['success']]
        
        if not successful_cases:
            return {'success_rate': 0.0, 'average_errors': None}
        
        # Compute average errors across all successful cases
        error_metrics = ['l1', 'l2', 'linf', 'mean_absolute', 'mean_relative']
        average_errors = {}
        
        for metric in error_metrics:
            values = [case['errors'][metric] for case in successful_cases 
                     if case['errors'] and metric in case['errors']]
            if values:
                average_errors[metric] = np.mean(values)
        
        return {
            'success_rate': len(successful_cases) / len(test_cases),
            'total_cases': len(test_cases),
            'successful_cases': len(successful_cases),
            'average_errors': average_errors,
        }


# Convenience functions
def compute_error_metrics(numerical: np.ndarray, analytical: np.ndarray) -> Dict[str, float]:
    """Compute all error metrics for given solutions."""
    analyzer = ErrorAnalyzer()
    return analyzer.compute_all_errors(numerical, analytical)


def analyze_convergence(grid_sizes: List[int], errors: Dict[str, List[float]]) -> Dict[str, float]:
    """Analyze convergence rates for given data."""
    analyzer = ConvergenceAnalyzer()
    return analyzer.analyze_convergence(grid_sizes, errors)


def validate_solution(method_func: Callable, analytical_func: Callable,
                     test_cases: List[Dict]) -> Dict:
    """Validate a numerical method against analytical solutions."""
    framework = ValidationFramework()
    return framework.validate_method(method_func, analytical_func, test_cases)
