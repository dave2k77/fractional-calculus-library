"""
Coupled System Solvers for Graph-SDE Dynamics

This module provides numerical solvers for systems of coupled spatial-temporal
dynamics, integrating graph-based spatial evolution with fractional SDE temporal evolution.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any, Union, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.definitions import FractionalOrder


@dataclass
class CoupledSolution:
    """Solution object for coupled graph-SDE systems."""
    t: np.ndarray
    spatial: np.ndarray  # Spatial (graph) state trajectory
    temporal: np.ndarray  # Temporal (SDE) state trajectory
    coupling: np.ndarray  # Coupling strength trajectory
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CoupledSystemSolver(ABC):
    """Base class for coupled system solvers."""
    
    def __init__(
        self,
        fractional_orders: Union[float, FractionalOrder, Dict[str, float]],
        coupling_strength: float = 1.0
    ):
        """
        Initialize coupled system solver.
        
        Args:
            fractional_orders: Fractional order(s) for system
            coupling_strength: Strength of spatial-temporal coupling
        """
        # Handle different types of fractional orders
        if isinstance(fractional_orders, dict):
            self.fractional_orders = fractional_orders
        elif isinstance(fractional_orders, (float, FractionalOrder)):
            self.fractional_orders = {
                'spatial': fractional_orders,
                'temporal': fractional_orders
            }
        else:
            raise ValueError("Invalid fractional_orders type")
        
        self.coupling_strength = coupling_strength
    
    @abstractmethod
    def solve(
        self,
        graph_dynamics: Callable,
        sde_drift: Callable,
        sde_diffusion: Callable,
        adjacency: np.ndarray,
        node_features: np.ndarray,
        t_span: Tuple[float, float],
        **kwargs
    ) -> CoupledSolution:
        """Solve coupled system."""
        pass


class OperatorSplittingSolver(CoupledSystemSolver):
    """
    Operator splitting solver for graph-SDE dynamics.
    
    Uses Strang splitting for second-order accuracy by splitting
    spatial and temporal operators.
    """
    
    def __init__(
        self,
        fractional_orders: Union[float, FractionalOrder, Dict[str, float]],
        coupling_strength: float = 1.0,
        split_order: int = 2
    ):
        """
        Initialize operator splitting solver.
        
        Args:
            fractional_orders: Fractional order(s)
            coupling_strength: Coupling strength
            split_order: Splitting order (1=Lie-Trotter, 2=Strang)
        """
        super().__init__(fractional_orders, coupling_strength)
        self.split_order = split_order
    
    def solve(
        self,
        graph_dynamics: Callable,
        sde_drift: Callable,
        sde_diffusion: Callable,
        adjacency: np.ndarray,
        node_features: np.ndarray,
        t_span: Tuple[float, float],
        num_steps: int = 100,
        seed: Optional[int] = None,
        **kwargs
    ) -> CoupledSolution:
        """
        Solve using operator splitting.
        
        For Strang splitting (order 2):
        - Half step of spatial dynamics
        - Full step of temporal dynamics
        - Half step of spatial dynamics
        """
        t0, tf = t_span
        dt = (tf - t0) / num_steps
        t = np.linspace(t0, tf, num_steps + 1)
        
        # Initialize state
        spatial_state = node_features.copy()
        temporal_state = node_features.copy()
        
        # Storage
        spatial_traj = np.zeros((num_steps + 1, *spatial_state.shape))
        temporal_traj = np.zeros((num_steps + 1, *temporal_state.shape))
        coupling_traj = np.zeros(num_steps + 1)
        
        spatial_traj[0] = spatial_state
        temporal_traj[0] = temporal_state
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Time stepping with operator splitting
        for i in range(num_steps):
            if self.split_order == 2:
                # Strang splitting: 0.5*spatial -> temporal -> 0.5*spatial
                # Half step spatial
                spatial_state = self._spatial_step(
                    graph_dynamics, adjacency, spatial_state, dt/2
                )
                
                # Full step temporal (SDE)
                temporal_state = self._temporal_step(
                    sde_drift, sde_diffusion, temporal_state, t[i], dt
                )
                
                # Half step spatial
                spatial_state = self._spatial_step(
                    graph_dynamics, adjacency, spatial_state, dt/2
                )
            else:
                # Lie-Trotter splitting: spatial -> temporal
                spatial_state = self._spatial_step(
                    graph_dynamics, adjacency, spatial_state, dt
                )
                temporal_state = self._temporal_step(
                    sde_drift, sde_diffusion, temporal_state, t[i], dt
                )
            
            # Save trajectory
            spatial_traj[i+1] = spatial_state
            temporal_traj[i+1] = temporal_state
            coupling_traj[i+1] = np.mean(np.abs(spatial_state - temporal_state))
        
        # Create solution
        solution = CoupledSolution(
            t=t,
            spatial=spatial_traj,
            temporal=temporal_traj,
            coupling=coupling_traj,
            metadata={
                'solver': 'operator_splitting',
                'split_order': self.split_order,
                'num_steps': num_steps
            }
        )
        
        return solution
    
    def _spatial_step(
        self,
        graph_dynamics: Callable,
        adjacency: np.ndarray,
        state: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Single spatial (graph) evolution step."""
        # Apply graph dynamics
        dspatial = graph_dynamics(state, adjacency)
        return state + dt * dspatial
    
    def _temporal_step(
        self,
        drift: Callable,
        diffusion: Callable,
        state: np.ndarray,
        t: float,
        dt: float
    ) -> np.ndarray:
        """Single temporal (SDE) evolution step."""
        # Euler-Maruyama step
        drift_val = drift(t, state)
        diffusion_val = diffusion(t, state)
        
        # Generate noise
        dW = np.random.normal(0, np.sqrt(dt), size=state.shape)
        
        # Update
        alpha = self.fractional_orders.get('temporal', 0.5)
        if isinstance(alpha, FractionalOrder):
            alpha = alpha.alpha
        
        return state + dt**alpha * drift_val + diffusion_val * dW


class MonolithicSolver(CoupledSystemSolver):
    """
    Monolithic solver for strongly coupled graph-SDE systems.
    
    Solves the full coupled system simultaneously for better accuracy
    in strongly coupled regimes, at the cost of higher memory usage.
    """
    
    def solve(
        self,
        graph_dynamics: Callable,
        sde_drift: Callable,
        sde_diffusion: Callable,
        adjacency: np.ndarray,
        node_features: np.ndarray,
        t_span: Tuple[float, float],
        num_steps: int = 100,
        seed: Optional[int] = None,
        **kwargs
    ) -> CoupledSolution:
        """Solve monolithic coupled system."""
        t0, tf = t_span
        dt = (tf - t0) / num_steps
        t = np.linspace(t0, tf, num_steps + 1)
        
        # Combined state: [spatial; temporal]
        combined_state = np.concatenate([node_features, node_features], axis=-1)
        
        # Storage
        combined_traj = np.zeros((num_steps + 1, *combined_state.shape))
        combined_traj[0] = combined_state
        
        # Random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Time stepping
        for i in range(num_steps):
            # Split state
            spatial_state = combined_state[..., :combined_state.shape[-1]//2]
            temporal_state = combined_state[..., combined_state.shape[-1]//2:]
            
            # Compute coupled dynamics
            dspatial = graph_dynamics(spatial_state, adjacency)
            dspatial += self.coupling_strength * (temporal_state - spatial_state)
            
            drift_val = sde_drift(t[i], temporal_state)
            diffusion_val = sde_diffusion(t[i], temporal_state)
            drift_val += self.coupling_strength * (spatial_state - temporal_state)
            
            # Generate noise
            dW = np.random.normal(0, np.sqrt(dt), size=temporal_state.shape)
            
            # Update
            dtemporal = drift_val + diffusion_val * dW
            
            # Get fractional order
            alpha = self.fractional_orders.get('temporal', 0.5)
            if isinstance(alpha, FractionalOrder):
                alpha = alpha.alpha
            
            # Combine and update
            dcombined = np.concatenate([dspatial * dt, dtemporal * dt**alpha], axis=-1)
            combined_state = combined_state + dcombined
            
            # Save
            combined_traj[i+1] = combined_state
        
        # Split trajectories
        spatial_traj = combined_traj[..., :combined_traj.shape[-1]//2]
        temporal_traj = combined_traj[..., combined_traj.shape[-1]//2:]
        coupling_traj = np.mean(np.abs(spatial_traj - temporal_traj), axis=(-2, -1))
        
        solution = CoupledSolution(
            t=t,
            spatial=spatial_traj,
            temporal=temporal_traj,
            coupling=coupling_traj,
            metadata={'solver': 'monolithic', 'num_steps': num_steps}
        )
        
        return solution


def solve_coupled_graph_sde(
    graph_dynamics: Callable,
    sde_drift: Callable,
    sde_diffusion: Callable,
    adjacency: np.ndarray,
    node_features: np.ndarray,
    t_span: Tuple[float, float],
    fractional_orders: Union[float, FractionalOrder, Dict[str, float]] = 0.5,
    coupling_type: str = "bidirectional",
    coupling_strength: float = 1.0,
    solver: str = "operator_splitting",
    **kwargs
) -> CoupledSolution:
    """
    Solve coupled graph-SDE system.
    
    Args:
        graph_dynamics: Spatial dynamics function f(spatial, adjacency)
        sde_drift: Temporal drift function f_spatial(t, temporal)
        sde_diffusion: Temporal diffusion function g_temporal(t, temporal)
        adjacency: Graph adjacency matrix
        node_features: Initial node features
        t_span: Time interval
        fractional_orders: Fractional order(s)
        coupling_type: Coupling type ("bidirectional", "spatial_to_temporal", etc.)
        coupling_strength: Strength of coupling
        solver: Solver type ("operator_splitting", "monolithic", "multiscale")
        **kwargs: Additional solver parameters
        
    Returns:
        CoupledSolution object
    """
    if solver == "operator_splitting":
        solver_obj = OperatorSplittingSolver(fractional_orders, coupling_strength)
    elif solver == "monolithic":
        solver_obj = MonolithicSolver(fractional_orders, coupling_strength)
    else:
        raise ValueError(f"Unknown solver type: {solver}")
    
    return solver_obj.solve(
        graph_dynamics,
        sde_drift,
        sde_diffusion,
        adjacency,
        node_features,
        t_span,
        **kwargs
    )
