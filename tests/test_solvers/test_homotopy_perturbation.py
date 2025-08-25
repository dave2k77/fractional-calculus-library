"""
Tests for Homotopy Perturbation Method (HPM) solvers.

This module contains comprehensive tests for all HPM implementations
including general FDE solvers and specialized solvers for diffusion and wave equations.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from hpfracc.solvers.homotopy_perturbation import (
    HomotopyPerturbationMethod,
    HPMFractionalDiffusion,
    HPMFractionalWave,
    create_hpm_solver,
    get_hpm_properties,
    validate_hpm_parameters
)


class TestHomotopyPerturbationMethod:
    """Test base HomotopyPerturbationMethod class."""
    
    def test_hpm_creation(self):
        """Test creating HomotopyPerturbationMethod instances."""
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=10)
        assert hpm.order == 0.5
        assert hpm.max_terms == 10
        assert hpm.tolerance == 1e-6
    
    def test_hpm_validation(self):
        """Test HomotopyPerturbationMethod validation."""
        # Test valid parameters
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=10)
        assert hpm.order == 0.5
        
        # Test invalid order
        with pytest.raises(ValueError):
            HomotopyPerturbationMethod(order=-1.0, max_terms=10)
        
        # Test invalid max_terms
        with pytest.raises(ValueError):
            HomotopyPerturbationMethod(order=0.5, max_terms=0)
        
        # Test invalid tolerance
        with pytest.raises(ValueError):
            HomotopyPerturbationMethod(order=0.5, max_terms=10, tolerance=-1e-6)
    
    def test_hpm_repr(self):
        """Test HomotopyPerturbationMethod string representation."""
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=10, tolerance=1e-8)
        repr_str = repr(hpm)
        assert "HomotopyPerturbationMethod" in repr_str
        assert "0.5" in repr_str
        assert "10" in repr_str
    
    def test_construct_homotopy(self):
        """Test homotopy construction."""
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=5)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        def initial_guess(x):
            return 1.0
        
        homotopy = hpm.construct_homotopy(differential_operator, initial_guess)
        
        # Test that homotopy is callable
        x = np.array([0.0, 1.0, 2.0])
        p = 0.5
        result = homotopy(x, p)
        assert len(result) == len(x)
        assert np.all(np.isfinite(result))
    
    def test_compute_series_terms(self):
        """Test series terms computation."""
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=3)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        def initial_guess(x):
            return 1.0
        
        x = np.linspace(0, 5, 10)
        terms = hpm.compute_series_terms(differential_operator, initial_guess, x)
        
        assert len(terms) == 3  # max_terms
        for term in terms:
            assert len(term) == len(x)
            assert np.all(np.isfinite(term))
    
    def test_solve(self):
        """Test general HPM solve method."""
        hpm = HomotopyPerturbationMethod(order=0.5, max_terms=5)
        
        # Define a simple FDE: D^α u + u = 0
        def differential_operator(u, x):
            return u  # Simplified for testing
        
        def initial_guess(x):
            return 1.0
        
        x = np.linspace(0, 5, 20)
        solution = hpm.solve(differential_operator, initial_guess, x)
        
        assert len(solution) == len(x)
        assert np.all(np.isfinite(solution))
        
        # Test convergence
        solution_5_terms = hpm.solve(differential_operator, initial_guess, x)
        hpm.max_terms = 10
        solution_10_terms = hpm.solve(differential_operator, initial_guess, x)
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5_terms, solution_10_terms, rtol=1e-2)


class TestHPMFractionalDiffusion:
    """Test HPM solver for fractional diffusion equation."""
    
    def test_hpm_diffusion_creation(self):
        """Test creating HPMFractionalDiffusion instances."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=10)
        assert isinstance(hpm, HomotopyPerturbationMethod)
        assert hpm.order == 0.5
        assert hpm.diffusion_coefficient == 1.0
        assert hpm.max_terms == 10
    
    def test_hpm_diffusion_equation(self):
        """Test HPM for fractional diffusion equation."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
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
        
        solution = hpm.solve_diffusion(
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
    
    def test_hpm_diffusion_convergence(self):
        """Test convergence of HPM for diffusion equation."""
        hpm_5 = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        hpm_10 = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=10)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 30)
        t = np.linspace(0, 1, 10)
        
        solution_5 = hpm_5.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        solution_10 = hpm_10.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5, solution_10, rtol=1e-1)
    
    def test_hpm_diffusion_properties(self):
        """Test properties of HPM diffusion solution."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-5, 5, 50)
        t = np.linspace(0, 2, 20)
        
        solution = hpm.solve_diffusion(
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


class TestHPMFractionalWave:
    """Test HPM solver for fractional wave equation."""
    
    def test_hpm_wave_creation(self):
        """Test creating HPMFractionalWave instances."""
        hpm = HPMFractionalWave(order=0.5, wave_speed=1.0, max_terms=10)
        assert isinstance(hpm, HomotopyPerturbationMethod)
        assert hpm.order == 0.5
        assert hpm.wave_speed == 1.0
        assert hpm.max_terms == 10
    
    def test_hpm_wave_equation(self):
        """Test HPM for fractional wave equation."""
        hpm = HPMFractionalWave(order=0.5, wave_speed=1.0, max_terms=5)
        
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
        
        solution = hpm.solve_wave(
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
    
    def test_hpm_wave_convergence(self):
        """Test convergence of HPM for wave equation."""
        hpm_5 = HPMFractionalWave(order=0.5, wave_speed=1.0, max_terms=5)
        hpm_10 = HPMFractionalWave(order=0.5, wave_speed=1.0, max_terms=10)
        
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
        
        solution_5 = hpm_5.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        solution_10 = hpm_10.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Solutions should be similar (convergence)
        np.testing.assert_allclose(solution_5, solution_10, rtol=1e-1)
    
    def test_hpm_wave_properties(self):
        """Test properties of HPM wave solution."""
        hpm = HPMFractionalWave(order=0.5, wave_speed=1.0, max_terms=5)
        
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
        
        solution = hpm.solve_wave(
            initial_displacement, initial_velocity,
            boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Test symmetry (for symmetric initial condition)
        mid_point = len(x) // 2
        for i in range(len(t)):
            left_half = solution[i, :mid_point]
            right_half = solution[i, mid_point:][::-1]
            np.testing.assert_allclose(left_half, right_half, rtol=1e-10)


class TestCreateHPMSolver:
    """Test factory function for creating HPM solvers."""
    
    def test_create_general_hpm(self):
        """Test creating general HPM solver via factory."""
        hpm = create_hpm_solver("general", order=0.5, max_terms=10)
        assert isinstance(hpm, HomotopyPerturbationMethod)
        assert hpm.order == 0.5
        assert hpm.max_terms == 10
    
    def test_create_diffusion_hpm(self):
        """Test creating diffusion HPM solver via factory."""
        hpm = create_hpm_solver("diffusion", order=0.5, diffusion_coefficient=1.0, max_terms=10)
        assert isinstance(hpm, HPMFractionalDiffusion)
        assert hpm.order == 0.5
        assert hpm.diffusion_coefficient == 1.0
    
    def test_create_wave_hpm(self):
        """Test creating wave HPM solver via factory."""
        hpm = create_hpm_solver("wave", order=0.5, wave_speed=1.0, max_terms=10)
        assert isinstance(hpm, HPMFractionalWave)
        assert hpm.order == 0.5
        assert hpm.wave_speed == 1.0
    
    def test_create_invalid_type(self):
        """Test creating HPM solver with invalid type."""
        with pytest.raises(ValueError):
            create_hpm_solver("invalid", order=0.5)


class TestHPMUtilities:
    """Test utility functions for HPM solvers."""
    
    def test_get_hpm_properties(self):
        """Test getting HPM properties."""
        # Test general HPM properties
        properties = get_hpm_properties("general")
        assert isinstance(properties, dict)
        assert "parameters" in properties
        assert "applications" in properties
        
        # Test diffusion HPM properties
        properties = get_hpm_properties("diffusion")
        assert isinstance(properties, dict)
        
        # Test wave HPM properties
        properties = get_hpm_properties("wave")
        assert isinstance(properties, dict)
        
        # Test invalid type
        properties = get_hpm_properties("invalid")
        assert properties is None
    
    def test_validate_hpm_parameters(self):
        """Test validation of HPM parameters."""
        # Test valid general parameters
        params = {"order": 0.5, "max_terms": 10}
        assert validate_hpm_parameters("general", params) is None
        
        # Test valid diffusion parameters
        params = {"order": 0.5, "diffusion_coefficient": 1.0, "max_terms": 10}
        assert validate_hpm_parameters("diffusion", params) is None
        
        # Test valid wave parameters
        params = {"order": 0.5, "wave_speed": 1.0, "max_terms": 10}
        assert validate_hpm_parameters("wave", params) is None
        
        # Test invalid parameters
        params = {"order": -0.5, "max_terms": 10}
        with pytest.raises(ValueError):
            validate_hpm_parameters("general", params)
        
        # Test missing parameters
        params = {"order": 0.5}
        with pytest.raises(ValueError):
            validate_hpm_parameters("general", params)


class TestHPMPerformance:
    """Test performance characteristics of HPM solvers."""
    
    def test_large_grid_performance(self):
        """Test performance with large grids."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
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
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        end_time = time.time()
        
        assert end_time - start_time < 30.0  # Should complete within 30 seconds
        assert solution.shape == (len(t), len(x))
    
    def test_memory_usage(self):
        """Test memory usage with large grids."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
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
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.nbytes < 1e7  # Should use less than 10MB


class TestHPMEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_order(self):
        """Test behavior with zero order."""
        hpm = HPMFractionalDiffusion(order=0.0, diffusion_coefficient=1.0, max_terms=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 20)
        t = np.linspace(0, 1, 10)
        
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        # Zero order should behave like standard diffusion
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
    
    def test_single_term(self):
        """Test behavior with single term."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=1)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.linspace(-3, 3, 20)
        t = np.linspace(0, 1, 10)
        
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        
        assert solution.shape == (len(t), len(x))
        assert np.all(np.isfinite(solution))
    
    def test_empty_grids(self):
        """Test behavior with empty grids."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.array([])
        t = np.array([])
        
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.shape == (0, 0)
    
    def test_single_points(self):
        """Test behavior with single points."""
        hpm = HPMFractionalDiffusion(order=0.5, diffusion_coefficient=1.0, max_terms=5)
        
        def initial_condition(x):
            return np.exp(-x**2)
        
        def boundary_condition_left(t):
            return 0.0
        
        def boundary_condition_right(t):
            return 0.0
        
        x = np.array([0.0])
        t = np.array([0.5])
        
        solution = hpm.solve_diffusion(
            initial_condition, boundary_condition_left, boundary_condition_right, x, t
        )
        assert solution.shape == (1, 1)
        assert np.isfinite(solution[0, 0])


if __name__ == "__main__":
    pytest.main([__file__])
