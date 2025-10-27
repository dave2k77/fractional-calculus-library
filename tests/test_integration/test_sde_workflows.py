"""
Integration tests for end-to-end SDE workflows

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
from hpfracc.solvers import (
    solve_fractional_sde, BrownianMotion, 
    FractionalEulerMaruyama, FractionalMilstein
)


class TestOrnsteinUhlenbeckWorkflow:
    """Test complete Ornstein-Uhlenbeck process workflow"""
    
    def test_ornstein_uhlenbeck(self):
        """Test solving Ornstein-Uhlenbeck process end-to-end"""
        # OU process: dX = θ(μ - X)dt + σ dW
        theta = 1.5
        mu = 0.5
        sigma = 0.3
        
        def drift(t, x):
            return theta * (mu - x)
        
        def diffusion(t, x):
            return sigma * np.ones_like(x)
        
        x0 = np.array([0.0])
        
        # Solve using Euler-Maruyama
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 5),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=100,
            seed=42
        )
        
        # Check solution properties
        assert solution.t.shape == (101,)
        assert solution.y.shape == (101, 1)
        assert solution.t[0] == 0
        assert abs(solution.t[-1] - 5) < 1e-10
        
        # Long-term mean should approach μ
        final_value = solution.y[-1, 0]
        # With variance, should be within reasonable bounds
        assert -2 < final_value < 2
    
    def test_ornstein_uhlenbeck_multiple_trajectories(self):
        """Test generating multiple OU trajectories"""
        theta = 1.0
        mu = 0.0
        sigma = 1.0
        
        def drift(t, x):
            return theta * (mu - x)
        
        def diffusion(t, x):
            return sigma * np.ones_like(x)
        
        x0 = np.array([1.0])
        
        # Generate multiple trajectories
        trajectories = []
        for seed in range(10):
            solution = solve_fractional_sde(
                drift,
                diffusion,
                x0,
                t_span=(0, 1),
                fractional_order=0.5,
                method="euler_maruyama",
                num_steps=50,
                seed=seed
            )
            trajectories.append(solution.y)
        
        trajectories = np.array(trajectories)
        
        # Check shapes
        assert trajectories.shape == (10, 51, 1)
        
        # Final values should vary (variance of OU process)
        final_values = trajectories[:, -1, 0]
        std_final = np.std(final_values)
        
        # Should have non-zero variance
        assert std_final > 0.1


class TestGeometricBrownianMotionWorkflow:
    """Test complete geometric Brownian motion workflow"""
    
    def test_geometric_brownian_motion(self):
        """Test solving geometric Brownian motion end-to-end"""
        # GBM: dS = μS dt + σS dW
        mu = 0.1
        sigma = 0.2
        
        def drift(t, x):
            return mu * x
        
        def diffusion(t, x):
            return sigma * x  # Multiplicative noise
        
        x0 = np.array([100.0])
        
        # Solve using Milstein method (better for multiplicative noise)
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="milstein",
            num_steps=100,
            seed=42
        )
        
        # Check solution
        assert solution.y.shape == (101, 1)
        
        # GBM should stay positive
        assert np.all(solution.y > 0)
    
    def test_geometric_brownian_motion_monte_carlo(self):
        """Test Monte Carlo simulation of GBM"""
        mu = 0.08
        sigma = 0.15
        
        def drift(t, x):
            return mu * x
        
        def diffusion(t, x):
            return sigma * x
        
        x0 = np.array([100.0])
        
        # Generate many paths
        n_paths = 100
        final_values = []
        
        for seed in range(n_paths):
            solution = solve_fractional_sde(
                drift,
                diffusion,
                x0,
                t_span=(0, 1),
                fractional_order=0.5,
                method="milstein",
                num_steps=100,
                seed=seed
            )
            final_values.append(solution.y[-1, 0])
        
        final_values = np.array(final_values)
        
        # Check statistical properties
        mean_final = np.mean(final_values)
        std_final = np.std(final_values)
        
        # Expected mean: S_0 * exp((μ - σ²/2) * t)
        # For GBM at t=1
        expected_mean = x0[0] * np.exp((mu - 0.5 * sigma**2) * 1)
        
        # Should be close to expected mean (with variance due to randomness)
        assert abs(mean_final - expected_mean) < 50  # Reasonable tolerance


class TestMultiDimensionalWorkflow:
    """Test multi-dimensional SDE systems"""
    
    def test_coupled_2d_system(self):
        """Test 2D coupled SDE system"""
        def drift(t, x):
            # Simple coupling: x influences y
            dx_dt = -0.5 * x[0] + 0.2 * x[1]
            dy_dt = -0.3 * x[1] - 0.1 * x[0]
            return np.array([dx_dt, dy_dt])
        
        def diffusion(t, x):
            return np.array([0.1, 0.15])
        
        x0 = np.array([1.0, 0.5])
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=50,
            seed=42
        )
        
        # Check 2D solution
        assert solution.y.shape == (51, 2)
    
    def test_nd_system(self):
        """Test higher-dimensional system"""
        def drift(t, x):
            return -0.1 * x
        
        def diffusion(t, x):
            return 0.05 * np.ones_like(x)
        
        n = 5
        x0 = np.ones(n)
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=50,
            seed=42
        )
        
        # Check n-dimensional solution
        assert solution.y.shape == (51, n)


class TestDifferentFractionalOrders:
    """Test workflows with different fractional orders"""
    
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9, 1.0])
    def test_various_fractional_orders(self, alpha):
        """Test solving SDE with various fractional orders"""
        def drift(t, x):
            return -0.5 * x
        
        def diffusion(t, x):
            return 0.2 * np.ones_like(x)
        
        x0 = np.array([1.0])
        
        solution = solve_fractional_sde(
            drift,
            diffusion,
            x0,
            t_span=(0, 1),
            fractional_order=alpha,
            method="euler_maruyama",
            num_steps=50,
            seed=42
        )
        
        # All should produce valid solutions
        assert isinstance(solution.t, np.ndarray)
        assert isinstance(solution.y, np.ndarray)
        assert solution.fractional_order.alpha == alpha


class TestSolverComparison:
    """Test comparing different solvers on same problem"""
    
    def test_euler_vs_milstein(self):
        """Compare Euler-Maruyama and Milstein methods"""
        def drift(t, x):
            return 1.0 - 0.5 * x
        
        def diffusion(t, x):
            return 0.3 * (1.0 + 0.5 * x)  # State-dependent
        
        x0 = np.array([1.0])
        
        # Solve with both methods
        sol_euler = solve_fractional_sde(
            drift, diffusion, x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="euler_maruyama",
            num_steps=100,
            seed=42
        )
        
        sol_milstein = solve_fractional_sde(
            drift, diffusion, x0,
            t_span=(0, 1),
            fractional_order=0.5,
            method="milstein",
            num_steps=100,
            seed=42
        )
        
        # Both should produce solutions
        assert sol_euler.y.shape == sol_milstein.y.shape
        
        # Solutions should be finite
        assert np.all(np.isfinite(sol_euler.y))
        assert np.all(np.isfinite(sol_milstein.y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
