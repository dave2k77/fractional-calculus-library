"""
Tests for Stochastic Differential Equation (SDE) solvers.

This module contains comprehensive tests for all SDE implementations
including Euler-Maruyama, Milstein, and Heun methods.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.solvers.sde_solvers import (
    BaseSDESolver,
    EulerMaruyama,
    Milstein,
    Heun,
    create_sde_solver,
    get_sde_solver_properties,
    validate_sde_parameters
)


class TestBaseSDESolver:
    """Test base SDE solver class."""
    
    def test_base_solver_creation(self):
        """Test creating BaseSDESolver instances."""
        solver = BaseSDESolver(dt=0.01, seed=42)
        assert solver.dt == 0.01
        
    def test_wiener_process_generation(self):
        """Test Wiener process generation."""
        solver = BaseSDESolver(dt=0.01, seed=42)
        increments = solver.generate_wiener_process(n_steps=100, n_paths=5)
        
        assert increments.shape == (5, 100)
        assert np.all(np.isfinite(increments))
        
        # Check that increments have correct variance
        expected_variance = 0.01
        actual_variance = np.var(increments)
        assert abs(actual_variance - expected_variance) < 0.1
        
    def test_error_estimation(self):
        """Test error estimation."""
        solver = BaseSDESolver(dt=0.01)
        
        # Create a simple solution
        solution = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        
        # Test without analytical solution
        error_est = solver.estimate_error(solution)
        assert "mean_error" in error_est
        assert "max_error" in error_est
        assert "rmse" in error_est
        assert error_est["dt"] == 0.01
        
    def test_stability_check(self):
        """Test stability checking."""
        solver = BaseSDESolver(dt=0.01)
        
        # Test stable solution
        stable_solution = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        stability = solver.check_stability(stable_solution)
        assert stability["is_finite"] is True
        assert stability["is_stable"] is True
        
        # Test unstable solution
        unstable_solution = np.array([1e10, 1e11, 1e12, 1e13, 1e14])
        stability = solver.check_stability(unstable_solution)
        assert stability["is_finite"] is True
        assert stability["is_stable"] is False


class TestEulerMaruyama:
    """Test Euler-Maruyama solver."""
    
    def test_euler_maruyama_creation(self):
        """Test creating EulerMaruyama instances."""
        solver = EulerMaruyama(dt=0.01, seed=42)
        assert solver.dt == 0.01
        
    def test_simple_sde_solution(self):
        """Test solving a simple SDE."""
        solver = EulerMaruyama(dt=0.01, seed=42)
        
        # Define simple drift and diffusion functions
        def drift(x, t):
            return -0.1 * x  # Mean reversion
        
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)  # Constant volatility
        
        # Solve SDE
        result = solver.solve(drift, diffusion, x0=1.0, t_span=(0, 1), n_steps=100)
        
        assert "solution" in result
        assert "t" in result
        assert "error_estimate" in result
        assert "stability" in result
        assert result["method"] == "Euler-Maruyama"
        
        # Check solution shape
        solution = result["solution"]
        assert solution.shape[0] == 1  # n_paths
        assert solution.shape[1] == 101  # n_steps + 1
        assert solution.shape[2] == 1  # dimension
        
        # Check that solution is finite
        assert np.all(np.isfinite(solution))


class TestMilstein:
    """Test Milstein solver."""
    
    def test_milstein_creation(self):
        """Test creating Milstein instances."""
        solver = Milstein(dt=0.01, seed=42)
        assert solver.dt == 0.01
        
    def test_milstein_solution(self):
        """Test solving SDE with Milstein method."""
        solver = Milstein(dt=0.01, seed=42)
        
        # Define drift and diffusion functions
        def drift(x, t):
            return -0.1 * x
        
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        def diffusion_derivative(x, t):
            return np.zeros_like(x)  # Constant diffusion
        
        # Solve SDE
        result = solver.solve(drift, diffusion, diffusion_derivative, 
                            x0=1.0, t_span=(0, 1), n_steps=100)
        
        assert result["method"] == "Milstein"
        assert "solution" in result
        
        # Check solution shape
        solution = result["solution"]
        assert solution.shape[0] == 1  # n_paths
        assert solution.shape[1] == 101  # n_steps + 1
        assert solution.shape[2] == 1  # dimension


class TestHeun:
    """Test Heun solver."""
    
    def test_heun_creation(self):
        """Test creating Heun instances."""
        solver = Heun(dt=0.01, seed=42)
        assert solver.dt == 0.01
        
    def test_heun_solution(self):
        """Test solving SDE with Heun method."""
        solver = Heun(dt=0.01, seed=42)
        
        # Define drift and diffusion functions
        def drift(x, t):
            return -0.1 * x
        
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        # Solve SDE
        result = solver.solve(drift, diffusion, x0=1.0, t_span=(0, 1), n_steps=100)
        
        assert result["method"] == "Heun"
        assert "solution" in result
        
        # Check solution shape
        solution = result["solution"]
        assert solution.shape[0] == 1  # n_paths
        assert solution.shape[1] == 101  # n_steps + 1
        assert solution.shape[2] == 1  # dimension


class TestFactoryFunctions:
    """Test factory functions for SDE solvers."""
    
    def test_create_sde_solver(self):
        """Test create_sde_solver factory function."""
        # Test Euler-Maruyama
        solver = create_sde_solver("euler_maruyama", dt=0.01)
        assert isinstance(solver, EulerMaruyama)
        assert solver.dt == 0.01
        
        # Test Milstein
        solver = create_sde_solver("milstein", dt=0.01)
        assert isinstance(solver, Milstein)
        assert solver.dt == 0.01
        
        # Test Heun
        solver = create_sde_solver("heun", dt=0.01)
        assert isinstance(solver, Heun)
        assert solver.dt == 0.01
        
        # Test invalid solver type
        with pytest.raises(ValueError):
            create_sde_solver("invalid_solver")
    
    def test_get_sde_solver_properties(self):
        """Test get_sde_solver_properties function."""
        solver = EulerMaruyama(dt=0.01, seed=42)
        properties = get_sde_solver_properties(solver)
        
        assert properties["dt"] == 0.01
        assert properties["solver_type"] == "EulerMaruyama"
        assert properties["method"] == "Euler-Maruyama"
    
    def test_validate_sde_parameters(self):
        """Test validate_sde_parameters function."""
        # Test valid parameters
        validation = validate_sde_parameters(dt=0.01, n_steps=100, n_paths=5)
        assert validation["valid"] is True
        assert validation["dt"] == 0.01
        assert validation["n_steps"] == 100
        assert validation["n_paths"] == 5
        
        # Test invalid dt
        with pytest.raises(ValueError):
            validate_sde_parameters(dt=-0.01)
        
        # Test invalid n_steps
        with pytest.raises(ValueError):
            validate_sde_parameters(dt=0.01, n_steps=0)
        
        # Test invalid n_paths
        with pytest.raises(ValueError):
            validate_sde_parameters(dt=0.01, n_paths=0)


class TestSDEComparison:
    """Test comparison between different SDE methods."""
    
    def test_method_comparison(self):
        """Test that different methods produce similar results for simple SDE."""
        # Define simple SDE
        def drift(x, t):
            return -0.1 * x
        
        def diffusion(x, t):
            return 0.1 * np.ones_like(x)
        
        def diffusion_derivative(x, t):
            return np.zeros_like(x)
        
        # Solve with different methods
        euler_solver = EulerMaruyama(dt=0.01, seed=42)
        milstein_solver = Milstein(dt=0.01, seed=42)
        heun_solver = Heun(dt=0.01, seed=42)
        
        euler_result = euler_solver.solve(drift, diffusion, x0=1.0, t_span=(0, 1), n_steps=100)
        milstein_result = milstein_solver.solve(drift, diffusion, diffusion_derivative, 
                                              x0=1.0, t_span=(0, 1), n_steps=100)
        heun_result = heun_solver.solve(drift, diffusion, x0=1.0, t_span=(0, 1), n_steps=100)
        
        # All solutions should have similar structure
        assert euler_result["solution"].shape == milstein_result["solution"].shape
        assert euler_result["solution"].shape == heun_result["solution"].shape
        
        # All solutions should be finite
        assert np.all(np.isfinite(euler_result["solution"]))
        assert np.all(np.isfinite(milstein_result["solution"]))
        assert np.all(np.isfinite(heun_result["solution"]))
