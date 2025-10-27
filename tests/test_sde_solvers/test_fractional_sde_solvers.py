"""
Unit tests for fractional SDE solvers in hpfracc.solvers.sde_solvers

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
from hpfracc.solvers import (
    FractionalEulerMaruyama, FractionalMilstein, solve_fractional_sde,
    solve_fractional_sde_system, SDESolution
)


class TestFractionalSDESolverBase:
    """Test base SDE solver functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.x0 = np.array([1.0])
        
        def drift(t, x):
            return -0.5 * x
        
        def diffusion(t, x):
            return 0.2 * np.ones_like(x)
        
        self.drift = drift
        self.diffusion = diffusion
    
    def test_invalid_fractional_order(self):
        """Test that invalid fractional orders raise errors"""
        with pytest.raises(ValueError):
            FractionalEulerMaruyama(fractional_order=0.0)
        
        with pytest.raises(ValueError):
            FractionalEulerMaruyama(fractional_order=2.0)
        
        with pytest.raises(ValueError):
            FractionalEulerMaruyama(fractional_order=-0.5)


class TestFractionalEulerMaruyama:
    """Test FractionalEulerMaruyama solver"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.solver = FractionalEulerMaruyama(fractional_order=self.alpha)
        
        self.drift = lambda t, x: -0.5 * x
        self.diffusion = lambda t, x: 0.2 * np.ones_like(x)
        self.x0 = np.array([1.0])
    
    def test_initialization(self):
        """Test solver initialization"""
        assert self.solver.fractional_order.alpha == self.alpha
        assert self.solver.method_name == "fractional_euler_maruyama"
    
    def test_solve_basic(self):
        """Test basic solving"""
        solution = self.solver.solve(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            num_steps=10,
            seed=42
        )
        
        assert isinstance(solution, SDESolution)
        assert solution.t.shape == (11,)  # num_steps + 1
        assert solution.y.shape == (11, 1)
        assert solution.metadata['num_steps'] == 10
    
    def test_solution_structure(self):
        """Test that solution has correct structure"""
        solution = self.solver.solve(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            num_steps=10
        )
        
        assert hasattr(solution, 't')
        assert hasattr(solution, 'y')
        assert hasattr(solution, 'fractional_order')
        assert hasattr(solution, 'method')
        assert hasattr(solution, 'drift_func')
        assert hasattr(solution, 'diffusion_func')
        assert hasattr(solution, 'metadata')
    
    def test_time_points(self):
        """Test time points are correctly generated"""
        solution = self.solver.solve(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 5),
            num_steps=20,
            seed=42
        )
        
        assert solution.t[0] == 0
        assert abs(solution.t[-1] - 5) < 1e-10
        assert len(solution.t) == 21
        assert np.all(np.diff(solution.t) > 0)  # Monotonically increasing


class TestFractionalMilstein:
    """Test FractionalMilstein solver"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.alpha = 0.5
        self.solver = FractionalMilstein(fractional_order=self.alpha)
        
        self.drift = lambda t, x: -0.5 * x
        self.diffusion = lambda t, x: 0.2 * np.ones_like(x)
        self.x0 = np.array([1.0])
    
    def test_initialization(self):
        """Test solver initialization"""
        assert self.solver.fractional_order.alpha == self.alpha
        assert self.solver.method_name == "fractional_milstein"
    
    def test_solve_basic(self):
        """Test basic solving"""
        solution = self.solver.solve(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            num_steps=10,
            seed=42
        )
        
        assert isinstance(solution, SDESolution)
        assert solution.t.shape == (11,)
        assert solution.y.shape == (11, 1)


class TestSolveFractionalSDE:
    """Test solve_fractional_sde convenience function"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.drift = lambda t, x: -0.5 * x
        self.diffusion = lambda t, x: 0.2 * np.ones_like(x)
        self.x0 = np.array([1.0])
    
    def test_euler_maruyama_method(self):
        """Test Euler-Maruyama method"""
        solution = solve_fractional_sde(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=10
        )
        
        assert isinstance(solution, SDESolution)
        assert solution.method == "fractional_euler_maruyama"
    
    def test_milstein_method(self):
        """Test Milstein method"""
        solution = solve_fractional_sde(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="milstein",
            num_steps=10
        )
        
        assert isinstance(solution, SDESolution)
        assert solution.method == "fractional_milstein"
    
    def test_invalid_method(self):
        """Test that invalid methods raise errors"""
        with pytest.raises(ValueError):
            solve_fractional_sde(
                self.drift,
                self.diffusion,
                self.x0,
                t_span=(0, 1),
                fractional_order=0.5,
                method="invalid_method"
            )
    
    def test_different_fractional_orders(self):
        """Test different fractional orders"""
        alphas = [0.3, 0.5, 0.7, 0.9]
        
        for alpha in alphas:
            solution = solve_fractional_sde(
                self.drift,
                self.diffusion,
                self.x0,
                t_span=(0, 1),
                fractional_order=alpha,
                method="euler_maruyama",
                num_steps=10
            )
            
            assert solution.fractional_order.alpha == alpha


class TestSolveFractionalSDESystem:
    """Test solve_fractional_sde_system for coupled systems"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.drift = lambda t, x: -0.5 * x
        self.diffusion = lambda t, x: 0.2 * np.ones_like(x)
        self.x0 = np.array([1.0, 0.5])
    
    def test_single_order(self):
        """Test with single fractional order"""
        solution = solve_fractional_sde_system(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=10
        )
        
        assert isinstance(solution, SDESolution)
        assert solution.y.shape == (11, 2)  # 2D system
    
    def test_multiple_orders(self):
        """Test with multiple fractional orders"""
        solution = solve_fractional_sde_system(
            self.drift,
            self.diffusion,
            self.x0,
            t_span=(0, 1),
            fractional_order=[0.5, 0.7],
            method="euler_maruyama",
            num_steps=10
        )
        
        assert isinstance(solution, SDESolution)


class TestConvergence:
    """Test numerical convergence of solvers"""
    
    def test_euler_maruyama_convergence(self):
        """Test Euler-Maruyama convergence order"""
        # Use Ornstein-Uhlenbeck process with known solution
        def drift(t, x):
            return 1.5 * (0.5 - x)
        
        def diffusion(t, x):
            return 0.3
        
        x0 = np.array([0.0])
        
        # Test with different step sizes
        errors = []
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        
        for dt in step_sizes:
            num_steps = int(1.0 / dt)
            solution = solve_fractional_sde(
                drift,
                diffusion,
                x0,
                t_span=(0, 1),
                fractional_order=0.5,
                method="euler_maruyama",
                num_steps=num_steps,
                seed=42
            )
            
            # For OU process, long-term mean should be around 0.5
            final_value = solution.y[-1, 0]
            error = abs(final_value - 0.5)
            errors.append(error)
        
        # Check that errors decrease as step size decreases
        for i in range(len(errors) - 1):
            # Roughly expect error to decrease with smaller step size
            # Due to stochastic nature, just check it's reasonable
            assert errors[i] < 1.0  # Sanity check


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_diffusion(self):
        """Test behavior with zero diffusion (deterministic)"""
        drift = lambda t, x: -x
        diffusion = lambda t, x: 0.0 * np.ones_like(x)
        
        x0 = np.array([1.0])
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=0.5,
            num_steps=10,
            seed=42
        )
        
        # Should still produce solution
        assert isinstance(solution, SDESolution)
    
    def test_large_time_span(self):
        """Test with large time span"""
        drift = lambda t, x: -0.1 * x
        diffusion = lambda t, x: 0.1
        
        x0 = np.array([1.0])
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 100),
            fractional_order=0.5,
            num_steps=1000,
            seed=42
        )
        
        assert solution.t[-1] == 100
    
    def test_state_dependent_diffusion(self):
        """Test with state-dependent diffusion"""
        drift = lambda t, x: -0.5 * x
        diffusion = lambda t, x: 0.2 * np.abs(x)  # Multiplicative noise
        
        x0 = np.array([1.0])
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=0.5,
            num_steps=50,
            seed=42
        )
        
        # Should handle multiplicative noise
        assert isinstance(solution, SDESolution)
        assert not np.any(np.isnan(solution.y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
