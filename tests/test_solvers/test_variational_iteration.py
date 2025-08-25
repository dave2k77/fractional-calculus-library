"""
Tests for Variational Iteration Method (VIM) solvers.

This module contains comprehensive tests for all VIM implementations
including general FDE solvers and specialized solvers for diffusion, wave, and advection equations.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.solvers.variational_iteration import (
    VariationalIterationMethod,
    VIMFractionalDiffusion,
    VIMFractionalWave,
    VIMFractionalAdvection,
    create_vim_solver,
    get_vim_properties,
    validate_vim_parameters
)


class TestVariationalIterationMethod:
    """Test base VariationalIterationMethod class."""
    
    def test_vim_creation(self):
        """Test creating VariationalIterationMethod instances."""
        vim = VariationalIterationMethod(order=0.5, max_iterations=10)
        assert vim.order == 0.5
        assert vim.max_iterations == 10
        assert vim.tolerance == 1e-6
    
    def test_vim_validation(self):
        """Test VariationalIterationMethod validation."""
        # Test valid parameters
        vim = VariationalIterationMethod(order=0.5, max_iterations=10)
        assert vim.order == 0.5
        
        # Test invalid order
        with pytest.raises(ValueError):
            VariationalIterationMethod(order=-1.0, max_iterations=10)
        
        # Test invalid max_iterations
        with pytest.raises(ValueError):
            VariationalIterationMethod(order=0.5, max_iterations=0)
        
        # Test invalid tolerance
        with pytest.raises(ValueError):
            VariationalIterationMethod(order=0.5, max_iterations=10, tolerance=-1e-6)
    
    def test_vim_repr(self):
        """Test VariationalIterationMethod string representation."""
        vim = VariationalIterationMethod(order=0.5, max_iterations=10, tolerance=1e-8)
        repr_str = repr(vim)
        assert "VariationalIterationMethod" in repr_str
        assert "0.5" in repr_str
        assert "10" in repr_str
    
    def test_compute_lagrange_multiplier(self):
        """Test Lagrange multiplier computation."""
        vim = VariationalIterationMethod(order=0.5, max_iterations=5)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        x = np.linspace(0, 5, 10)
        lagrange_multiplier = vim.compute_lagrange_multiplier(differential_operator, x)
        
        # Lagrange multiplier should be a function
        assert callable(lagrange_multiplier)
        
        # Test evaluation
        result = lagrange_multiplier(x)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_construct_correction_functional(self):
        """Test correction functional construction."""
        vim = VariationalIterationMethod(order=0.5, max_iterations=5)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        def initial_guess(x):
            return 1.0
        
        x = np.linspace(0, 5, 10)
        correction_functional = vim.construct_correction_functional(
            differential_operator, initial_guess, x
        )
        
        # Correction functional should be callable
        assert callable(correction_functional)
        
        # Test evaluation
        result = correction_functional(x)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_solve(self):
        """Test general VIM solve method."""
        vim = VariationalIterationMethod(order=0.5, max_iterations=5)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        def initial_guess(x):
            return 1.0
        
        x = np.linspace(0, 5, 20)
        solution = vim.solve(differential_operator, initial_guess, x)
        
        assert len(solution) == len(x)
        assert np.all(np.isfinite(solution))
        
        # Test convergence
        solution_5_iter = vim.solve(differential_operator, initial_guess, x)
        vim.max_iterations = 10
        solution_10_iter = vim.solve(differential_operator, initial_guess, x)
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5_iter, solution_10_iter, rtol=1e-2)


class TestVIMFractionalDiffusion:
    """Test VIM solver for fractional diffusion equation."""
    
    def test_vim_diffusion_creation(self):
        """Test creating VIMFractionalDiffusion instances."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=10)
        assert isinstance(vim, VariationalIterationMethod)
        assert vim.order == 0.5
        assert vim.diffusion_coefficient == 1.0
        assert vim.max_iterations == 10
    
    def test_vim_diffusion_equation(self):
        """Test VIM for fractional diffusion equation."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        # Define initial condition: u(x,0) = exp(-x^2)
        def initial_condition(x):
            return np.exp(-x**2)
        
        # Define boundary conditions
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
        
        # Test initial condition
        np.testing.assert_allclose(solution[0, :], initial_condition(x), rtol=1e-10)
        
        # Test boundary conditions
        for i, t_val in enumerate(t):
            np.testing.assert_allclose(solution[i, 0], boundary_condition_left(t_val), rtol=1e-10)
            np.testing.assert_allclose(solution[i, -1], boundary_condition_right(t_val), rtol=1e-10)
    
    def test_vim_diffusion_convergence(self):
        """Test convergence of VIM for diffusion equation."""
        vim_5 = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        vim_10 = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=10)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 30)
        t = np.linspace(0, 1, 10)
        
        solution_5 = vim_5.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        solution_10 = vim_10.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5, solution_10, rtol=1e-1)
    
    def test_vim_diffusion_properties(self):
        """Test properties of VIM diffusion solution."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Test symmetry (for symmetric initial condition)
        mid_point = len(x) // 2
        for i in range(len(t)):
            left_half = solution[i, :mid_point]
            right_half = solution[i, mid_point:][::-1]
            np.testing.assert_allclose(left_half, right_half, rtol=1e-10)
        
        # Test maximum principle (solution should not exceed initial maximum)
        initial_max = np.max(initial_condition(x))
        solution_max = np.max(solution)
        assert solution_max <= initial_max * 1.1  # Allow small numerical error


class TestVIMFractionalWave:
    """Test VIM solver for fractional wave equation."""
    
    def test_vim_wave_creation(self):
        """Test creating VIMFractionalWave instances."""
        vim = VIMFractionalWave(order=0.5, wave_speed=1.0, max_iterations=10)
        assert isinstance(vim, VariationalIterationMethod)
        assert vim.order == 0.5
        assert vim.wave_speed == 1.0
        assert vim.max_iterations == 10
    
    def test_vim_wave_equation(self):
        """Test VIM for fractional wave equation."""
        vim = VIMFractionalWave(order=0.5, wave_speed=1.0, max_iterations=5)
        
        # Define initial conditions
        def initial_displacement(x):
            return np.exp(-x**2)
        
        def initial_velocity(x):
            return np.zeros_like(x)
        
        # Define boundary conditions
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
        
        # Test initial displacement
        np.testing.assert_allclose(solution[0, :], initial_displacement(x), rtol=1e-10)
        
        # Test boundary conditions
        for i, t_val in enumerate(t):
            np.testing.assert_allclose(solution[i, 0], boundary_condition_left(t_val), rtol=1e-10)
            np.testing.assert_allclose(solution[i, -1], boundary_condition_right(t_val), rtol=1e-10)
    
    def test_vim_wave_convergence(self):
        """Test convergence of VIM for wave equation."""
        vim_5 = VIMFractionalWave(order=0.5, wave_speed=1.0, max_iterations=5)
        vim_10 = VIMFractionalWave(order=0.5, wave_speed=1.0, max_iterations=10)
        
        def initial_displacement(x):
            return np.exp(-x**2)
        
        def initial_velocity(x):
            return np.zeros_like(x)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 30)
        t = np.linspace(0, 1, 10)
        
        solution_5 = vim_5.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        solution_10 = vim_10.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5, solution_10, rtol=1e-1)
    
    def test_vim_wave_properties(self):
        """Test properties of VIM wave solution."""
        vim = VIMFractionalWave(order=0.5, wave_speed=1.0, max_iterations=5)
        
        def initial_displacement(x):
            return np.exp(-x**2)
        
        def initial_velocity(x):
            return np.zeros_like(x)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Test symmetry (for symmetric initial condition)
        mid_point = len(x) // 2
        for i in range(len(t)):
            left_half = solution[i, :mid_point]
            right_half = solution[i, mid_point:][::-1]
            np.testing.assert_allclose(left_half, right_half, rtol=1e-10)


class TestVIMFractionalAdvection:
    """Test VIM solver for fractional advection equation."""
    
    def test_vim_advection_creation(self):
        """Test creating VIMFractionalAdvection instances."""
        vim = VIMFractionalAdvection(order=0.5, advection_velocity=1.0, max_iterations=10)
        assert isinstance(vim, VariationalIterationMethod)
        assert vim.order == 0.5
        assert vim.advection_velocity == 1.0
        assert vim.max_iterations == 10
    
    def test_vim_advection_equation(self):
        """Test VIM for fractional advection equation."""
        vim = VIMFractionalAdvection(order=0.5, advection_velocity=1.0, max_iterations=5)
        
        # Define initial condition
        def initial_condition(x):
            return np.exp(-x**2)
        
        # Define boundary conditions
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_advection(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
        
        # Test initial condition
        np.testing.assert_allclose(solution[0, :], initial_condition(x), rtol=1e-10)
        
        # Test boundary conditions
        for i, t_val in enumerate(t):
            np.testing.assert_allclose(solution[i, 0], boundary_condition_left(t_val), rtol=1e-10)
            np.testing.assert_allclose(solution[i, -1], boundary_condition_right(t_val), rtol=1e-10)
    
    def test_vim_advection_convergence(self):
        """Test convergence of VIM for advection equation."""
        vim_5 = VIMFractionalAdvection(order=0.5, advection_velocity=1.0, max_iterations=5)
        vim_10 = VIMFractionalAdvection(order=0.5, advection_velocity=1.0, max_iterations=10)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 30)
        t = np.linspace(0, 1, 10)
        
        solution_5 = vim_5.solve_advection(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        solution_10 = vim_10.solve_advection(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5, solution_10, rtol=1e-1)
    
    def test_vim_advection_properties(self):
        """Test properties of VIM advection solution."""
        vim = VIMFractionalAdvection(order=0.5, advection_velocity=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = vim.solve_advection(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Test advection behavior
        # For advection equation, the peak should move with velocity v
        v = 1.0
        for i, t_val in enumerate(t):
            if t_val > 0:
                # Find peak position
                peak_index = np.argmax(solution[i, :])
                peak_x = x[peak_index]
                
                # Peak should be close to x = vt (allowing for boundary effects)
                if abs(peak_x) < 4:  # Not too close to boundaries
                    np.testing.assert_allclose(peak_x, v * t_val, rtol=0.5)


class TestCreateVIMSolver:
    """Test factory function for creating VIM solvers."""
    
    def test_create_general_vim(self):
        """Test creating general VIM solver via factory."""
        vim = create_vim_solver("general", order=0.5, max_iterations=10)
        assert isinstance(vim, VariationalIterationMethod)
        assert vim.order == 0.5
        assert vim.max_iterations == 10
    
    def test_create_diffusion_vim(self):
        """Test creating diffusion VIM solver via factory."""
        vim = create_vim_solver("diffusion", order=0.5, diffusion_coefficient=1.0, max_iterations=10)
        assert isinstance(vim, VIMFractionalDiffusion)
        assert vim.order == 0.5
        assert vim.diffusion_coefficient == 1.0
    
    def test_create_wave_vim(self):
        """Test creating wave VIM solver via factory."""
        vim = create_vim_solver("wave", order=0.5, wave_speed=1.0, max_iterations=10)
        assert isinstance(vim, VIMFractionalWave)
        assert vim.order == 0.5
        assert vim.wave_speed == 1.0
    
    def test_create_advection_vim(self):
        """Test creating advection VIM solver via factory."""
        vim = create_vim_solver("advection", order=0.5, advection_velocity=1.0, max_iterations=10)
        assert isinstance(vim, VIMFractionalAdvection)
        assert vim.order == 0.5
        assert vim.advection_velocity == 1.0
    
    def test_create_invalid_type(self):
        """Test creating VIM solver with invalid type."""
        with pytest.raises(ValueError):
            create_vim_solver("invalid", order=0.5)


class TestVIMUtilities:
    """Test utility functions for VIM solvers."""
    
    def test_get_vim_properties(self):
        """Test getting VIM properties."""
        # Test general VIM properties
        properties = get_vim_properties("general")
        assert isinstance(properties, dict)
        assert "parameters" in properties
        assert "applications" in properties
        
        # Test diffusion VIM properties
        properties = get_vim_properties("diffusion")
        assert isinstance(properties, dict)
        
        # Test wave VIM properties
        properties = get_vim_properties("wave")
        assert isinstance(properties, dict)
        
        # Test advection VIM properties
        properties = get_vim_properties("advection")
        assert isinstance(properties, dict)
        
        # Test invalid type
        properties = get_vim_properties("invalid")
        assert properties is None
    
    def test_validate_vim_parameters(self):
        """Test validation of VIM parameters."""
        # Test valid general parameters
        params = {"order": 0.5, "max_iterations": 10}
        assert validate_vim_parameters("general", params) is None
        
        # Test valid diffusion parameters
        params = {"order": 0.5, "diffusion_coefficient": 1.0, "max_iterations": 10}
        assert validate_vim_parameters("diffusion", params) is None
        
        # Test valid wave parameters
        params = {"order": 0.5, "wave_speed": 1.0, "max_iterations": 10}
        assert validate_vim_parameters("wave", params) is None
        
        # Test valid advection parameters
        params = {"order": 0.5, "advection_velocity": 1.0, "max_iterations": 10}
        assert validate_vim_parameters("advection", params) is None
        
        # Test invalid parameters
        params = {"order": -0.5, "max_iterations": 10}
        with pytest.raises(ValueError):
            validate_vim_parameters("general", params)
        
        # Test missing parameters
        params = {"order": 0.5}
        with pytest.raises(ValueError):
            validate_vim_parameters("general", params)


class TestVIMPerformance:
    """Test performance characteristics of VIM solvers."""
    
    def test_large_grid_performance(self):
        """Test performance with large grids."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        # Large grid
        x = np.linspace(-10, 10, 200)
        t = np.linspace(0, 5, 100)
        
        # Should complete within reasonable time
        import time
        start_time = time.time()
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        end_time = time.time()
        
        assert end_time - start_time < 60.0  # Should complete within 60 seconds
        assert solution.shape == (len(t), len(x))
    
    def test_memory_usage(self):
        """Test memory usage with large grids."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        # Large grid
        x = np.linspace(-10, 10, 200)
        t = np.linspace(0, 5, 100)
        
        # Should not cause memory issues
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.nbytes < 1e7  # Should use less than 10MB


class TestVIMEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_order(self):
        """Test behavior with zero order."""
        vim = VIMFractionalDiffusion(order=0.0, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 20)
        t = np.linspace(0, 1, 10)
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Zero order should behave like standard diffusion
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
    
    def test_single_iteration(self):
        """Test behavior with single iteration."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=1)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 20)
        t = np.linspace(0, 1, 10)
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
    
    def test_empty_grids(self):
        """Test behavior with empty grids."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.array([])
        t = np.array([])
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.shape == (0, 0)
    
    def test_single_points(self):
        """Test behavior with single points."""
        vim = VIMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_iterations=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.array([0.0])
        t = np.array([0.5])
        
        solution = vim.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.shape == (1, 1)
        assert np.isfinite(solution[0, 0])


if __name__ == "__main__":
    pytest.main([__file__])
