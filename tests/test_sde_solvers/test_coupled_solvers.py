"""
Unit tests for coupled system SDE solvers in hpfracc.solvers.coupled_solvers

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import pytest
import torch
from hpfracc.solvers.coupled_solvers import (
    CoupledSystemSolver, OperatorSplittingSolver, MonolithicSolver,
    solve_coupled_graph_sde, CoupledSolution
)


class TestCoupledSystemSolver:
    """Test CoupledSystemSolver base class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.solver = CoupledSystemSolver()
    
    def test_initialization(self):
        """Test solver initialization"""
        assert self.solver is not None
        assert hasattr(self.solver, 'solve')
    
    def test_abstract_method(self):
        """Test that solve method is abstract"""
        # Should raise NotImplementedError when called
        with pytest.raises(NotImplementedError):
            self.solver.solve(None, None, None, None)


class TestOperatorSplittingSolver:
    """Test OperatorSplittingSolver"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.solver = OperatorSplittingSolver()
    
    def test_initialization(self):
        """Test solver initialization"""
        assert self.solver is not None
        assert hasattr(self.solver, 'solve')
    
    def test_solve_basic(self):
        """Test basic solving with operator splitting"""
        # Define coupled system
        def drift(t, x):
            # Simple coupled system: dx/dt = -x + y, dy/dt = -y + x
            return np.array([-x[0] + x[1], -x[1] + x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        num_steps = 10
        
        solution = self.solver.solve(drift, diffusion, x0, t_span, num_steps=num_steps)
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (num_steps + 1,)
        assert solution.y.shape == (num_steps + 1, 2)
    
    def test_solve_with_splitting_method(self):
        """Test solving with different splitting methods"""
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test Strang splitting
        solver_strang = OperatorSplittingSolver(splitting_method="strang")
        solution_strang = solver_strang.solve(drift, diffusion, x0, t_span, num_steps=10)
        
        # Test Lie splitting
        solver_lie = OperatorSplittingSolver(splitting_method="lie")
        solution_lie = solver_lie.solve(drift, diffusion, x0, t_span, num_steps=10)
        
        assert isinstance(solution_strang, CoupledSolution)
        assert isinstance(solution_lie, CoupledSolution)
    
    def test_invalid_splitting_method(self):
        """Test with invalid splitting method"""
        with pytest.raises(ValueError):
            OperatorSplittingSolver(splitting_method="invalid")


class TestMonolithicSolver:
    """Test MonolithicSolver"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.solver = MonolithicSolver()
    
    def test_initialization(self):
        """Test solver initialization"""
        assert self.solver is not None
        assert hasattr(self.solver, 'solve')
    
    def test_solve_basic(self):
        """Test basic solving with monolithic method"""
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        num_steps = 10
        
        solution = self.solver.solve(drift, diffusion, x0, t_span, num_steps=num_steps)
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (num_steps + 1,)
        assert solution.y.shape == (num_steps + 1, 2)
    
    def test_solve_with_coupling_strength(self):
        """Test solving with different coupling strengths"""
        def drift(t, x):
            coupling_strength = 0.5
            return np.array([-x[0] + coupling_strength * x[1], 
                           -x[1] + coupling_strength * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        solution = self.solver.solve(drift, diffusion, x0, t_span, num_steps=10)
        
        assert isinstance(solution, CoupledSolution)


class TestSolveCoupledGraphSDE:
    """Test solve_coupled_graph_sde function"""
    
    def test_basic_graph_sde_solving(self):
        """Test basic graph-SDE solving"""
        # Simple 2-node graph
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def drift(t, x):
            # Graph-coupled dynamics
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        solution = solve_coupled_graph_sde(
            drift, diffusion, x0, adjacency_matrix,
            t_span=t_span, num_steps=10
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.t.shape == (11,)
        assert solution.y.shape == (11, 2)
    
    def test_graph_sde_with_different_methods(self):
        """Test graph-SDE with different solving methods"""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        # Test with operator splitting
        solution_split = solve_coupled_graph_sde(
            drift, diffusion, x0, adjacency_matrix,
            t_span=t_span, num_steps=10, method="operator_splitting"
        )
        
        # Test with monolithic solver
        solution_mono = solve_coupled_graph_sde(
            drift, diffusion, x0, adjacency_matrix,
            t_span=t_span, num_steps=10, method="monolithic"
        )
        
        assert isinstance(solution_split, CoupledSolution)
        assert isinstance(solution_mono, CoupledSolution)
    
    def test_invalid_method(self):
        """Test with invalid method"""
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        
        with pytest.raises(ValueError):
            solve_coupled_graph_sde(
                drift, diffusion, x0, adjacency_matrix,
                t_span=t_span, method="invalid_method"
            )


class TestCoupledSolution:
    """Test CoupledSolution dataclass"""
    
    def test_solution_creation(self):
        """Test solution object creation"""
        t = np.linspace(0, 1, 11)
        y = np.random.randn(11, 2)
        
        solution = CoupledSolution(
            t=t,
            y=y,
            coupling_matrix=np.array([[0, 1], [1, 0]]),
            method="operator_splitting"
        )
        
        assert solution.t.shape == (11,)
        assert solution.y.shape == (11, 2)
        assert solution.coupling_matrix.shape == (2, 2)
        assert solution.method == "operator_splitting"
    
    def test_solution_properties(self):
        """Test solution properties"""
        t = np.linspace(0, 1, 11)
        y = np.random.randn(11, 2)
        
        solution = CoupledSolution(
            t=t,
            y=y,
            coupling_matrix=np.array([[0, 1], [1, 0]]),
            method="monolithic"
        )
        
        # Test final state
        final_state = solution.get_final_state()
        assert final_state.shape == (2,)
        assert np.allclose(final_state, y[-1])
        
        # Test state at specific time
        state_at_half = solution.get_state_at_time(0.5)
        assert state_at_half.shape == (2,)


class TestCoupledSolverIntegration:
    """Test integration between coupled solvers"""
    
    def test_solver_comparison(self):
        """Test comparison between different solvers"""
        def drift(t, x):
            return np.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        t_span = (0, 1)
        num_steps = 20
        
        # Solve with operator splitting
        solver_split = OperatorSplittingSolver()
        solution_split = solver_split.solve(drift, diffusion, x0, t_span, num_steps=num_steps)
        
        # Solve with monolithic solver
        solver_mono = MonolithicSolver()
        solution_mono = solver_mono.solve(drift, diffusion, x0, t_span, num_steps=num_steps)
        
        # Both should produce valid solutions
        assert isinstance(solution_split, CoupledSolution)
        assert isinstance(solution_mono, CoupledSolution)
        
        # Solutions should have same shape
        assert solution_split.y.shape == solution_mono.y.shape
    
    def test_graph_coupling_integration(self):
        """Test integration with graph coupling"""
        # Create a more complex graph (3 nodes)
        adjacency_matrix = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        def drift(t, x):
            # Graph-coupled dynamics for 3 nodes
            return np.array([
                -x[0] + 0.3 * x[1],
                -x[1] + 0.3 * x[0] + 0.3 * x[2],
                -x[2] + 0.3 * x[1]
            ])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1, 0.1])
        
        x0 = np.array([1.0, 0.5, 0.8])
        t_span = (0, 1)
        
        solution = solve_coupled_graph_sde(
            drift, diffusion, x0, adjacency_matrix,
            t_span=t_span, num_steps=15
        )
        
        assert isinstance(solution, CoupledSolution)
        assert solution.y.shape == (16, 3)  # 15 steps + 1 initial


class TestCoupledSolverEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_system(self):
        """Test with empty system"""
        def drift(t, x):
            return np.array([])
        
        def diffusion(t, x):
            return np.array([])
        
        x0 = np.array([])
        t_span = (0, 1)
        
        solver = OperatorSplittingSolver()
        
        with pytest.raises(ValueError):
            solver.solve(drift, diffusion, x0, t_span, num_steps=10)
    
    def test_mismatched_dimensions(self):
        """Test with mismatched dimensions"""
        def drift(t, x):
            return np.array([-x[0], -x[1]])  # 2D drift
        
        def diffusion(t, x):
            return np.array([0.1])  # 1D diffusion
        
        x0 = np.array([1.0, 0.5])  # 2D initial condition
        
        solver = OperatorSplittingSolver()
        
        with pytest.raises(ValueError):
            solver.solve(drift, diffusion, x0, (0, 1), num_steps=10)
    
    def test_invalid_time_span(self):
        """Test with invalid time span"""
        def drift(t, x):
            return np.array([-x[0]])
        
        def diffusion(t, x):
            return np.array([0.1])
        
        x0 = np.array([1.0])
        
        solver = OperatorSplittingSolver()
        
        with pytest.raises(ValueError):
            solver.solve(drift, diffusion, x0, (1, 0), num_steps=10)  # Invalid: tf < t0
    
    def test_invalid_adjacency_matrix(self):
        """Test with invalid adjacency matrix"""
        adjacency_matrix = np.array([[0, 1], [1, 0], [0, 0]])  # Wrong shape
        
        def drift(t, x):
            return np.array([-x[0], -x[1]])
        
        def diffusion(t, x):
            return np.array([0.1, 0.1])
        
        x0 = np.array([1.0, 0.5])
        
        with pytest.raises(ValueError):
            solve_coupled_graph_sde(
                drift, diffusion, x0, adjacency_matrix,
                t_span=(0, 1), num_steps=10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
