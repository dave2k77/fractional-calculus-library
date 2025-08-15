"""
Tests for Advanced Features

This module tests the advanced features implemented in Phase 5:
- Advanced solvers with error control
- GPU optimization
- High-order methods
- Real-world applications
"""

import pytest
import numpy as np
from typing import Dict, Any
import warnings

# Import advanced features
from src.solvers.advanced_solvers import (
    AdvancedFractionalODESolver,
    HighOrderFractionalSolver,
    ErrorControlMethod,
    AdaptiveMethod,
    solve_advanced_fractional_ode,
    solve_high_order_fractional_ode,
)

# Import GPU optimization
from src.optimisation.gpu_optimization import (
    GPUOptimizer,
    MultiGPUManager,
    GPUBackend,
    MemoryStrategy,
    create_gpu_optimizer,
    create_multi_gpu_manager,
    optimize_fractional_derivative_gpu,
)

# Import real-world applications
from examples.real_world_applications.financial_modeling import (
    FractionalBlackScholesModel,
    FractionalVolatilityModel,
    FractionalRiskManager,
)


class TestAdvancedSolvers:
    """Test advanced fractional ODE solvers."""

    def test_advanced_solver_initialization(self):
        """Test advanced solver initialization."""
        solver = AdvancedFractionalODESolver(
            derivative_type="caputo",
            method="embedded_pairs",
            error_control=ErrorControlMethod.LOCAL_ERROR,
            adaptive_method=AdaptiveMethod.PID_CONTROL,
            tol=1e-6,
            rtol=1e-6,
            atol=1e-8,
        )
        
        assert solver.derivative_type == "caputo"
        assert solver.method == "embedded_pairs"
        assert solver.error_control == ErrorControlMethod.LOCAL_ERROR
        assert solver.adaptive_method == AdaptiveMethod.PID_CONTROL
        assert solver.tol == 1e-6
        assert solver.rtol == 1e-6
        assert solver.atol == 1e-8

    def test_advanced_solver_parameter_validation(self):
        """Test parameter validation in advanced solver."""
        # Test invalid derivative type
        with pytest.raises(ValueError):
            AdvancedFractionalODESolver(derivative_type="invalid")
        
        # Test invalid tolerances
        with pytest.raises(ValueError):
            AdvancedFractionalODESolver(tol=0)
        
        with pytest.raises(ValueError):
            AdvancedFractionalODESolver(rtol=-1)
        
        # Test invalid step size bounds
        with pytest.raises(ValueError):
            AdvancedFractionalODESolver(min_step=1.0, max_step=0.5)
        
        # Test invalid safety factor
        with pytest.raises(ValueError):
            AdvancedFractionalODESolver(safety_factor=1.5)

    def test_advanced_solver_simple_ode(self):
        """Test advanced solver on simple ODE."""
        def simple_ode(t, y):
            return -y
        
        solver = AdvancedFractionalODESolver(
            method="embedded_pairs",
            tol=1e-4,
            max_iter=100,
        )
        
        solution = solver.solve(
            simple_ode,
            t_span=(0, 1),
            y0=1.0,
            alpha=1.0,
        )
        
        assert 't' in solution
        assert 'y' in solution
        assert 'h' in solution
        assert 'error_estimates' in solution
        assert 'order_estimates' in solution
        assert 'iterations' in solution
        assert 'converged' in solution
        
        # Check that solution converged
        assert solution['converged'] is True
        assert len(solution['t']) > 1
        assert len(solution['y']) > 1

    def test_advanced_solver_error_control(self):
        """Test different error control methods."""
        def test_ode(t, y):
            return np.sin(t) * y
        
        # Test local error control
        solver_local = AdvancedFractionalODESolver(
            error_control=ErrorControlMethod.LOCAL_ERROR,
            tol=1e-4,
        )
        solution_local = solver_local.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # Test global error control
        solver_global = AdvancedFractionalODESolver(
            error_control=ErrorControlMethod.GLOBAL_ERROR,
            tol=1e-4,
        )
        solution_global = solver_global.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # Test mixed error control
        solver_mixed = AdvancedFractionalODESolver(
            error_control=ErrorControlMethod.MIXED_ERROR,
            tol=1e-4,
        )
        solution_mixed = solver_mixed.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # All should converge
        assert solution_local['converged']
        assert solution_global['converged']
        assert solution_mixed['converged']

    def test_advanced_solver_adaptive_methods(self):
        """Test different adaptive methods."""
        def test_ode(t, y):
            return np.cos(t) * y
        
        # Test PID control
        solver_pid = AdvancedFractionalODESolver(
            adaptive_method=AdaptiveMethod.PID_CONTROL,
            tol=1e-4,
        )
        solution_pid = solver_pid.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # Test embedded pairs
        solver_embedded = AdvancedFractionalODESolver(
            adaptive_method=AdaptiveMethod.EMBEDDED_PAIRS,
            tol=1e-4,
        )
        solution_embedded = solver_embedded.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # Test variable order
        solver_variable = AdvancedFractionalODESolver(
            adaptive_method=AdaptiveMethod.VARIABLE_ORDER,
            tol=1e-4,
        )
        solution_variable = solver_variable.solve(test_ode, (0, 1), 1.0, 0.5)
        
        # All should converge
        assert solution_pid['converged']
        assert solution_embedded['converged']
        assert solution_variable['converged']

    def test_convenience_functions(self):
        """Test convenience functions for advanced solvers."""
        def test_ode(t, y):
            return np.exp(-t) * y
        
        # Test solve_advanced_fractional_ode
        solution = solve_advanced_fractional_ode(
            test_ode,
            t_span=(0, 1),
            y0=1.0,
            alpha=0.5,
            method="embedded_pairs",
            tol=1e-4,
        )
        
        assert 't' in solution
        assert 'y' in solution
        assert solution['converged']


class TestHighOrderSolvers:
    """Test high-order fractional solvers."""

    def test_high_order_solver_initialization(self):
        """Test high-order solver initialization."""
        solver = HighOrderFractionalSolver(
            method="spectral",
            order=4,
            collocation_points=10,
        )
        
        assert solver.method == "spectral"
        assert solver.order == 4
        assert solver.collocation_points == 10

    def test_high_order_solver_parameter_validation(self):
        """Test parameter validation in high-order solver."""
        # Test invalid method
        with pytest.raises(ValueError):
            HighOrderFractionalSolver(method="invalid")

    def test_spectral_method(self):
        """Test spectral method solver."""
        def test_ode(t, y):
            return -y
        
        solver = HighOrderFractionalSolver(method="spectral", collocation_points=8)
        solution = solver.solve(test_ode, (0, 1), 1.0, 0.5)
        
        assert 't' in solution
        assert 'y' in solution
        assert 'method' in solution
        assert 'order' in solution
        assert solution['method'] == 'spectral'

    def test_multistep_method(self):
        """Test multi-step method solver."""
        def test_ode(t, y):
            return np.sin(t)
        
        solver = HighOrderFractionalSolver(method="multistep", order=3)
        solution = solver.solve(test_ode, (0, 1), 0.0, 0.5)
        
        assert 't' in solution
        assert 'y' in solution
        assert 'method' in solution
        assert 'order' in solution
        assert solution['method'] == 'multistep'

    def test_collocation_method(self):
        """Test collocation method solver."""
        def test_ode(t, y):
            return t * y
        
        solver = HighOrderFractionalSolver(method="collocation", collocation_points=6)
        solution = solver.solve(test_ode, (0, 1), 1.0, 0.5)
        
        assert 't' in solution
        assert 'y' in solution
        assert 'method' in solution
        assert 'order' in solution
        assert solution['method'] == 'collocation'

    def test_convenience_function(self):
        """Test convenience function for high-order solvers."""
        def test_ode(t, y):
            return y
        
        solution = solve_high_order_fractional_ode(
            test_ode,
            t_span=(0, 1),
            y0=1.0,
            alpha=0.5,
            method="spectral",
        )
        
        assert 't' in solution
        assert 'y' in solution


class TestGPUOptimization:
    """Test GPU optimization features."""

    def test_gpu_optimizer_initialization(self):
        """Test GPU optimizer initialization."""
        try:
            optimizer = GPUOptimizer(
                backend=GPUBackend.AUTO,
                memory_strategy=MemoryStrategy.ADAPTIVE,
                max_memory_usage=0.8,
                batch_size=1000,
                performance_monitoring=True,
            )
            
            assert optimizer.backend in [GPUBackend.JAX, GPUBackend.CUPY]
            assert optimizer.memory_strategy == MemoryStrategy.ADAPTIVE
            assert optimizer.max_memory_usage == 0.8
            assert optimizer.batch_size == 1000
            assert optimizer.performance_monitoring is True
            
        except RuntimeError as e:
            if "No GPU backend available" in str(e):
                pytest.skip("No GPU backend available")
            else:
                raise

    def test_gpu_optimizer_parameter_validation(self):
        """Test GPU optimizer parameter validation."""
        # Test invalid memory usage
        with pytest.raises(ValueError):
            GPUOptimizer(max_memory_usage=1.5)
        
        with pytest.raises(ValueError):
            GPUOptimizer(max_memory_usage=-0.1)

    def test_gpu_fractional_derivative(self):
        """Test GPU-optimized fractional derivative computation."""
        try:
            optimizer = create_gpu_optimizer(backend=GPUBackend.AUTO)
            
            # Test data
            x = np.random.random(100)
            alpha = 0.5
            
            # Compute fractional derivative
            result = optimizer.optimize_fractional_derivative(x, alpha, method="caputo")
            
            assert result.shape == x.shape
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
            
        except RuntimeError as e:
            if "No GPU backend available" in str(e):
                pytest.skip("No GPU backend available")
            else:
                raise

    def test_gpu_performance_monitoring(self):
        """Test GPU performance monitoring."""
        try:
            optimizer = create_gpu_optimizer(backend=GPUBackend.AUTO, performance_monitoring=True)
            
            # Run some computations
            x = np.random.random(1000)
            alpha = 0.5
            
            for _ in range(3):
                optimizer.optimize_fractional_derivative(x, alpha, method="caputo")
            
            # Get performance report
            report = optimizer.get_performance_report()
            
            assert 'total_time' in report
            assert 'gpu_time' in report
            assert 'backend' in report
            assert 'device_count' in report
            
            # Reset performance stats
            optimizer.reset_performance_stats()
            new_report = optimizer.get_performance_report()
            assert new_report['total_time'] == 0.0
            
        except RuntimeError as e:
            if "No GPU backend available" in str(e):
                pytest.skip("No GPU backend available")
            else:
                raise

    def test_multi_gpu_manager(self):
        """Test multi-GPU manager."""
        try:
            manager = create_multi_gpu_manager(backend=GPUBackend.AUTO)
            
            assert manager.device_count >= 1
            assert manager.backend in [GPUBackend.JAX, GPUBackend.CUPY]
            
            # Test distribution
            x = np.random.random(100)
            alpha = 0.5
            
            result = manager.distribute_computation(x, alpha, method="caputo")
            
            assert result.shape == x.shape
            assert not np.any(np.isnan(result))
            
        except RuntimeError as e:
            if "No GPU backend available" in str(e):
                pytest.skip("No GPU backend available")
            else:
                raise

    def test_convenience_functions(self):
        """Test GPU optimization convenience functions."""
        try:
            # Test create_gpu_optimizer
            optimizer = create_gpu_optimizer(backend=GPUBackend.AUTO)
            assert isinstance(optimizer, GPUOptimizer)
            
            # Test create_multi_gpu_manager
            manager = create_multi_gpu_manager(backend=GPUBackend.AUTO)
            assert isinstance(manager, MultiGPUManager)
            
            # Test optimize_fractional_derivative_gpu
            x = np.random.random(50)
            alpha = 0.5
            
            result = optimize_fractional_derivative_gpu(x, alpha, method="caputo")
            assert result.shape == x.shape
            
        except RuntimeError as e:
            if "No GPU backend available" in str(e):
                pytest.skip("No GPU backend available")
            else:
                raise


class TestRealWorldApplications:
    """Test real-world applications."""

    def test_fractional_black_scholes_model(self):
        """Test fractional Black-Scholes model."""
        model = FractionalBlackScholesModel(
            alpha=0.7,
            r=0.05,
            sigma=0.2,
            use_gpu=False,
        )
        
        assert model.alpha == 0.7
        assert model.r == 0.05
        assert model.sigma == 0.2
        assert model.use_gpu is False

    def test_option_pricing(self):
        """Test option pricing with fractional Black-Scholes."""
        model = FractionalBlackScholesModel(alpha=0.7, r=0.05, sigma=0.2)
        
        # Test call option pricing
        call_result = model.price_european_call(S0=100, K=100, T=1.0, t_points=50)
        
        assert 't' in call_result
        assert 'S_t' in call_result
        assert 'payoff' in call_result
        assert 'option_price' in call_result
        assert 'final_price' in call_result
        assert 'solution_metadata' in call_result
        
        # Test put option pricing
        put_result = model.price_european_put(S0=100, K=100, T=1.0, t_points=50)
        
        assert 't' in put_result
        assert 'S_t' in put_result
        assert 'payoff' in put_result
        assert 'option_price' in put_result
        assert 'final_price' in put_result
        
        # Check that prices are reasonable
        assert call_result['final_price'] >= 0
        assert put_result['final_price'] >= 0

    def test_volatility_surface_analysis(self):
        """Test volatility surface analysis."""
        model = FractionalBlackScholesModel(alpha=0.7, r=0.05, sigma=0.2)
        
        K_range = np.linspace(80, 120, 5)
        T_range = np.linspace(0.1, 1.0, 5)
        
        result = model.analyze_volatility_surface(S0=100, K_range=K_range, T_range=T_range)
        
        assert 'volatility_surface' in result
        assert 'implied_volatilities' in result
        assert 'K_range' in result
        assert 'T_range' in result
        assert 'moneyness' in result
        
        # Check shapes
        assert result['volatility_surface'].shape == (len(T_range), len(K_range))
        assert result['implied_volatilities'].shape == (len(T_range), len(K_range))

    def test_fractional_volatility_model(self):
        """Test fractional volatility model."""
        model = FractionalVolatilityModel(alpha=0.6, beta=0.1)
        
        assert model.alpha == 0.6
        assert model.beta == 0.1

    def test_volatility_simulation(self):
        """Test volatility simulation."""
        model = FractionalVolatilityModel(alpha=0.6, beta=0.1)
        
        result = model.simulate_volatility(v0=0.2, T=1.0, n_steps=100)
        
        assert 't' in result
        assert 'volatility' in result
        assert 'solution_metadata' in result
        
        # Check that volatility is reasonable
        assert np.all(result['volatility'] >= 0)
        assert len(result['t']) == len(result['volatility'])

    def test_fractional_risk_manager(self):
        """Test fractional risk manager."""
        manager = FractionalRiskManager(alpha=0.5)
        
        assert manager.alpha == 0.5

    def test_risk_measures(self):
        """Test fractional risk measures."""
        manager = FractionalRiskManager(alpha=0.5)
        
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        # Calculate VaR
        var = manager.calculate_fractional_var(returns, confidence_level=0.95)
        assert isinstance(var, float)
        assert not np.isnan(var)
        assert not np.isinf(var)
        
        # Calculate CVaR
        cvar = manager.calculate_fractional_cvar(returns, confidence_level=0.95)
        assert isinstance(cvar, float)
        assert not np.isnan(cvar)
        assert not np.isinf(cvar)

    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        manager = FractionalRiskManager(alpha=0.5)
        
        # Generate sample returns matrix
        np.random.seed(42)
        returns_matrix = np.random.normal(0.001, 0.02, (100, 3))
        
        # Optimize portfolio weights
        weights = manager.optimize_portfolio_weights(returns_matrix, target_return=0.1)
        
        assert len(weights) == 3
        assert np.all(weights >= 0)  # No short selling
        assert abs(np.sum(weights) - 1.0) < 1e-6  # Weights sum to 1


def test_integration():
    """Test integration of advanced features."""
    # Test that advanced solvers work with GPU optimization
    def test_ode(t, y):
        return -y
    
    # Test advanced solver
    solution = solve_advanced_fractional_ode(
        test_ode,
        t_span=(0, 1),
        y0=1.0,
        alpha=0.5,
        method="embedded_pairs",
        tol=1e-4,
    )
    
    assert solution['converged']
    
    # Test high-order solver
    solution_high = solve_high_order_fractional_ode(
        test_ode,
        t_span=(0, 1),
        y0=1.0,
        alpha=0.5,
        method="spectral",
    )
    
    assert 't' in solution_high
    assert 'y' in solution_high


if __name__ == "__main__":
    pytest.main([__file__])
