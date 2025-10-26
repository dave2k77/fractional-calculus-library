"""
Tutorial 03: Fractional State Space Modeling using HPFRACC
=========================================================

This tutorial demonstrates advanced fractional state space modeling techniques
using the HPFRACC library. The tutorial covers:

1. Fractional-Order State Space (FOSS) reconstruction
2. Multi-span Transition Entropy Component Method (MTECM-FOSS)
3. Stability analysis of fractional state space systems
4. Parameter estimation for fractional state space models
5. Applications to complex dynamical systems

References:
- Xie, Y., et al. (2024). Fractional-Order State Space (FOSS) reconstruction method
- Chen, Y., et al. (2023). FPGA Implementation of Non-Commensurate Fractional-Order State-Space Models
- Wang, Y., et al. (2023). Parameter estimation in fractional-order Hammerstein state space systems
- Busłowicz, M. (2023). Practical stability of discrete fractional-order state space models
- Zhang, Y., et al. (2025). Fractional-order Wiener state space systems

Author: HPFRACC Development Team
Date: January 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import os
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from typing import Dict

# JAX will automatically use the GPU if available, otherwise it will default to CPU.

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import HPFRACC components
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.derivatives import create_fractional_derivative
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.special import gamma
from hpfracc.core.utilities import validate_fractional_order, timing_decorator
from hpfracc.solvers import solve_fractional_ode

# Set up plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 11


class FractionalStateSpaceModel:
    """
    Advanced fractional state space modeling using HPFRACC.
    """

    def __init__(self, alpha=0.5, dim=3):
        """
        Initialize fractional state space model.

        Parameters:
        -----------
        alpha : float
            Fractional order
        dim : int
            State space dimension
        """
        self.alpha = FractionalOrder(alpha)
        self.dim = dim

        # Validate parameters
        if not validate_fractional_order(alpha):
            raise ValueError(f"Invalid fractional order: {alpha}")

        # Initialize fractional operators
        self.derivative = create_fractional_derivative(
            definition_type="riemann_liouville", alpha=alpha
        )
        self.integral = create_fractional_integral(order=alpha, method="RL")

        # Initialize state space matrices
        self.A = None  # State matrix
        self.B = None  # Input matrix
        self.C = None  # Output matrix
        self.D = None  # Feedthrough matrix

        print(f"FractionalStateSpaceModel initialized (α={alpha}, dim={dim})")

    def generate_lorenz_data(self, duration=100, dt=0.01, noise_level=0.01):
        """
        Generate Lorenz system data with fractional dynamics using the library's solver.
        """
        # Lorenz parameters
        sigma = 10.0
        rho = 28.0
        beta_lorenz = 8/3

        def lorenz_system(t, y):
            x, y, z = y
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta_lorenz * z
            return np.array([dxdt, dydt, dzdt])

        # Initial conditions
        y0 = np.array([1.0, 1.0, 1.0])
        t_span = (0, duration)
        
        # Solve the fractional ODE
        t, x = solve_fractional_ode(
            lorenz_system,
            t_span,
            y0,
            self.alpha.alpha,
            h=dt,
            adaptive=True
        )

        # Add noise
        if noise_level > 0:
            x += noise_level * np.random.randn(*x.shape)

        # Check for and handle numerical instability
        if not np.all(np.isfinite(x)):
            print("Warning: Numerical instability detected in Lorenz attractor. Replacing non-finite values.")
            x = np.nan_to_num(x)

        return t, x

    @timing_decorator
    def foss_reconstruction(self, time_series, embedding_dim=3, delay=1,
                            alpha_values=None):
        """
        Fractional-Order State Space (FOSS) reconstruction.

        Parameters:
        -----------
        time_series : array
            Input time series
        embedding_dim : int
            Embedding dimension
        delay : int
            Time delay
        alpha_values : list
            Fractional orders to test

        Returns:
        --------
        state_spaces : dict
            Dictionary of reconstructed state spaces for different α
        """
        if alpha_values is None:
            alpha_values = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]

        state_spaces = {}
        n_samples = len(time_series)

        for alpha in alpha_values:
            # Create fractional derivative operator for this alpha
            derivative_op = create_fractional_derivative(
                definition_type="riemann_liouville", alpha=alpha
            )

            # Apply fractional derivative to time series
            time = np.arange(n_samples)

            def series_func(t):
                idx = int(t)
                if idx >= n_samples:
                    idx = n_samples - 1
                return time_series[idx]

            fractional_series = derivative_op.compute(series_func, time)

            # Traditional delay embedding with fractional signal
            n_vectors = n_samples - (embedding_dim - 1) * delay
            state_space = np.zeros((n_vectors, embedding_dim))

            for i in range(embedding_dim):
                start_idx = i * delay
                end_idx = start_idx + n_vectors
                state_space[:, i] = fractional_series[start_idx:end_idx]

            state_spaces[alpha] = state_space

        return state_spaces

    def mtecm_foss_analysis(self, state_spaces, n_clusters=5):
        """
        Multi-span Transition Entropy Component Method (MTECM-FOSS).

        Parameters:
        -----------
        state_spaces : dict
            Dictionary of state spaces for different α
        n_clusters : int
            Number of clusters for state classification

        Returns:
        --------
        results : dict
            MTECM-FOSS analysis results
        """
        results = {}
        max_samples = 3000

        for alpha, state_space in state_spaces.items():
            # Downsample for speed if needed
            if state_space.shape[0] > max_samples:
                idx = np.linspace(0, state_space.shape[0] - 1, max_samples, dtype=int)
                state_space_downsampled = state_space[idx]
                cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(state_space_downsampled)
            else:
                state_space_downsampled = state_space
                cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(state_space)

            # Normalize state space
            scaler = StandardScaler()
            state_space_norm = scaler.fit_transform(state_space_downsampled)

            # Cluster states
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=200, random_state=42)
            cluster_labels = kmeans.fit_predict(state_space_norm)

            # Compute transition matrix
            transition_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(len(cluster_labels) - 1):
                current_cluster = cluster_labels[i]
                next_cluster = cluster_labels[i + 1]
                transition_matrix[current_cluster, next_cluster] += 1

            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
            transition_matrix = np.nan_to_num(transition_matrix, 0)

            # Compute entropy measures
            # Intra-sample entropy (within clusters)
            intra_entropy = 0
            for i in range(n_clusters):
                cluster_points = state_space_norm[cluster_labels == i]
                if len(cluster_points) > 0:
                    # Compute variance within cluster
                    cluster_var = np.var(cluster_points, axis=0)
                    cluster_entropy = np.sum(cluster_var)
                    intra_entropy += cluster_entropy * \
                        len(cluster_points) / len(state_space_norm)

            # Inter-sample entropy (between clusters)
            inter_entropy = 0
            cluster_centers = kmeans.cluster_centers_
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    distance = np.linalg.norm(
                        cluster_centers[i] - cluster_centers[j])
                    inter_entropy += distance

            # Transition entropy
            transition_entropy = 0
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if transition_matrix[i, j] > 0:
                        transition_entropy -= transition_matrix[i, j] * np.log(
                            transition_matrix[i, j])

            results[alpha] = {
                'intra_entropy': intra_entropy,
                'inter_entropy': inter_entropy,
                'transition_entropy': transition_entropy,
                'total_entropy': intra_entropy + inter_entropy + transition_entropy,
                'transition_matrix': transition_matrix,
                'cluster_labels': cluster_labels,
                'cluster_centers': cluster_centers,
                'state_space_downsampled': state_space_downsampled # Added for plotting
            }

        return results

    def estimate_fractional_parameters(self, time_series, method='ls'):
        """
        Estimate parameters of fractional state space model.

        Parameters:
        -----------
        time_series : array
            Input time series
        method : str
            Estimation method ('ls' for least squares, 'kalman' for Kalman filter)

        Returns:
        --------
        params : dict
            Estimated parameters
        """
        if method == 'ls':
            return self._least_squares_estimation(time_series)
        elif method == 'kalman':
            return self._kalman_filter_estimation(time_series)
        elif method == 'bayesian':
            return self._bayesian_estimation(time_series)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _least_squares_estimation(self, time_series: np.ndarray) -> Dict[str, float]:
        """Estimate fractional order using least squares."""
        time = np.linspace(0, 1, len(time_series))
        series_func = interp1d(time, time_series, kind='cubic', fill_value="extrapolate")
        
        def residuals(alpha):
            self.derivative.alpha = FractionalOrder(alpha[0])
            frac_deriv = self.derivative.compute(series_func, time[:-1])
            return frac_deriv - np.gradient(time_series, time)[1:]
        
        result = least_squares(residuals, [0.5], bounds=(0.1, 1.9))
        return {'alpha': result.x[0]}

    def _kalman_filter_estimation(self, time_series):
        """
        Kalman filter parameter estimation (simplified).
        """
        # Simplified Kalman filter implementation
        n_samples = len(time_series)

        # Initialize state and covariance
        x_est = np.zeros(self.dim)
        P = np.eye(self.dim) * 0.1

        # Process and measurement noise
        Q = np.eye(self.dim) * 0.01
        R = 0.1

        # Storage
        state_history = np.zeros((n_samples, self.dim))

        # Measurement matrix
        H = np.zeros((1, self.dim))
        H[0, 0] = 1.0

        for k in range(1, n_samples):
            # Prediction step
            # Simplified state transition (would be more complex in practice)
            x_pred = x_est
            P_pred = P + Q

            # Update step
            H = np.atleast_2d(H)
            y_res = time_series[k] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_est = x_pred + K @ y_res
            P_est = (np.eye(self.dim) - K @ H) @ P_pred

        return {'alpha': x_est[0]}

    def _bayesian_estimation(self, time_series):
        """
        Bayesian parameter estimation (simplified).
        """
        # Simplified Bayesian estimation
        n_samples = len(time_series)

        # Initialize state and covariance
        x_est = np.zeros(self.dim)
        P_est = np.eye(self.dim) * 0.1

        # Process and measurement noise
        Q = np.eye(self.dim) * 0.01
        R = 0.1

        # Storage
        state_history = np.zeros((n_samples, self.dim))

        # Measurement matrix
        H = np.zeros((1, self.dim))
        H[0, 0] = 1.0

        for k in range(n_samples):
            # Prediction step
            x_pred = x_est
            P_pred = P_est + Q

            # Update step
            H = np.atleast_2d(H)
            y_res = time_series[k] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_est = x_pred + K @ y_res
            P_est = (np.eye(self.dim) - K @ H) @ P_pred

        return {'alpha': x_est[0]}

    def stability_analysis(self, A_matrix=None):
        """
        Analyze stability of fractional state space system.

        Parameters:
        -----------
        A_matrix : array, optional
            State matrix (if None, uses estimated matrix)

        Returns:
        --------
        stability_info : dict
            Stability analysis results
        """
        if A_matrix is None:
            # Use identity matrix as default
            A_matrix = np.eye(self.dim)

        # Compute eigenvalues
        eigenvals = linalg.eigvals(A_matrix)

        # For fractional systems, stability condition is more complex
        # |arg(λ)| > απ/2 for all eigenvalues λ

        alpha_threshold = self.alpha.alpha * np.pi / 2
        stability_margins = []

        for eig in eigenvals:
            arg_eig = np.abs(np.angle(eig))
            margin = arg_eig - alpha_threshold
            stability_margins.append(margin)

        # Determine stability
        min_margin = min(stability_margins)
        is_stable = min_margin > 0

        # Compute stability measures
        stability_radius = min(np.abs(eigenvals))
        condition_number = np.linalg.cond(A_matrix)

        return {
            'eigenvalues': eigenvals,
            'stability_margins': stability_margins,
            'is_stable': is_stable,
            'min_stability_margin': min_margin,
            'stability_radius': stability_radius,
            'condition_number': condition_number,
            'alpha_threshold': alpha_threshold
        }

    def simulate_fractional_system(self, u, x0=None, dt=0.01):
        """
        Simulate fractional state space system.

        Parameters:
        -----------
        u : array
            Input signal
        x0 : array, optional
            Initial state
        dt : float
            Time step

        Returns:
        --------
        t : array
            Time vector
        x : array
            State history
        y : array
            Output history
        """
        n_steps = len(u)
        t = np.arange(n_steps) * dt

        # Initialize state
        if x0 is None:
            x0 = np.zeros(self.dim)

        x = np.zeros((n_steps, self.dim))
        y = np.zeros(n_steps)
        x[0] = x0

        # Use default matrices if not set
        if self.A is None:
            self.A = -np.eye(self.dim) * 0.1
        if self.B is None:
            self.B = np.ones(self.dim)
        if self.C is None:
            self.C = np.array([1, 0, 0])
        if self.D is None:
            self.D = 0

        # Simulation loop
        for k in range(1, n_steps):
            # Fractional state equation
            # D^α x(t) = Ax(t) + Bu(t)

            # Simplified fractional integration
            # In practice, this would use proper fractional integration
            frac_factor = (dt ** self.alpha.alpha) / \
                gamma(self.alpha.alpha + 1)

            # State update
            dx = self.A @ x[k-1] + self.B * u[k-1]
            x[k] = x[k-1] + frac_factor * dx

            # Output equation
            y[k] = self.C @ x[k] + self.D * u[k]

        return t, x, y

    def plot_foss_analysis(self, state_spaces, mtecm_results):
        """
        Plot comprehensive FOSS analysis results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FOSS Analysis', fontsize=16)

        for i, alpha in enumerate(state_spaces.keys()):
            mtecm_result = mtecm_results[alpha]
            
            # Use the downsampled data and cluster labels from the MTECM results
            state_space_plot = mtecm_result['state_space_downsampled']
            cluster_labels = mtecm_result['cluster_labels']

            # FOSS State Space
            axes[0, i].scatter(state_space_plot[:, 0], state_space_plot[:, 1], c=state_space_plot[:, 2], cmap='viridis', alpha=0.6, s=1)
            axes[0, i].set_title(f'FOSS State Space (α={alpha})')
            
            # MTECM-FOSS Analysis
            scatter = axes[1, i].scatter(state_space_plot[:, 0], state_space_plot[:, 1], c=cluster_labels, cmap='tab10', alpha=0.6, s=1)
            axes[1, i].set_title(f'MTECM-FOSS (α={alpha})')
            
        # Remove unused axes
        if len(state_spaces) < 2:
            fig.delaxes(axes[0, 1])
            fig.delaxes(axes[1, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('foss_analysis.png')
        plt.close()

    def plot_parameter_estimation(self, params_ls, params_kf, params_bayes):
        """
        Plot parameter estimation results.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Parameter Estimation Results', fontsize=16)

        # Plot 1: Least Squares Estimation
        axes[0].plot(params_ls['alpha'], 'b-o', label='Least Squares')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Estimated α')
        axes[0].set_title('Least Squares Estimation')
        axes[0].grid(True)

        # Plot 2: Kalman Filter Estimation
        axes[1].plot(params_kf['alpha'], 'r-s', label='Kalman Filter')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Estimated α')
        axes[1].set_title('Kalman Filter Estimation')
        axes[1].grid(True)

        # Plot 3: Bayesian Estimation (simplified)
        axes[2].plot(params_bayes['alpha'], 'g-^', label='Bayesian')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Estimated α')
        axes[2].set_title('Bayesian Estimation')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('parameter_estimation.png')
        plt.close()


def main():
    """
    Main tutorial demonstration.
    """
    print("=" * 60)
    print("TUTORIAL 03: FRACTIONAL STATE SPACE MODELING")
    print("=" * 60)

    # Initialize model
    model = FractionalStateSpaceModel(alpha=0.5, dim=3)

    # Generate Lorenz data
    print("Generating fractional Lorenz system data...")
    t_lorenz, x_lorenz = model.generate_lorenz_data(duration=5, dt=0.001)

    print(f"Lorenz data generated: {len(t_lorenz)} time steps")

    # FOSS reconstruction
    print("Performing FOSS reconstruction...")
    time_series = x_lorenz[:, 0]  # Use x-component
    state_spaces = model.foss_reconstruction(
        time_series, embedding_dim=3, delay=10, alpha_values=[0.5, 0.7])

    print(
        f"FOSS reconstruction completed for {len(state_spaces)} fractional orders")

    # MTECM-FOSS analysis
    print("Performing MTECM-FOSS analysis...")
    mtecm_results = model.mtecm_foss_analysis(state_spaces, n_clusters=5)

    print("\nMTECM-FOSS Results:")
    for alpha, results in mtecm_results.items():
        print(f"  α={alpha}: Total Entropy = {results['total_entropy']:.4f}")

    # Parameter estimation
    print("\nEstimating fractional parameters...")
    params_ls = model.estimate_fractional_parameters(time_series, method='ls')
    params_kf = model.estimate_fractional_parameters(
        time_series, method='kalman')

    print("Least Squares Estimation:")
    for key, value in params_ls.items():
        if key != 'residuals':
            print(f"  {key}: {value:.4f}")

    # Stability analysis
    print("\nPerforming stability analysis...")
    stability_info = model.stability_analysis()

    print(
        f"System Stability: {'Stable' if stability_info['is_stable'] else 'Unstable'}")
    print(
        f"Min Stability Margin: {stability_info['min_stability_margin']:.4f}")
    print(f"Stability Radius: {stability_info['stability_radius']:.4f}")

    # System simulation
    print("\nSimulating fractional system...")
    t_sim = np.linspace(0, 10, 1000)
    u_sim = np.sin(2 * np.pi * 0.5 * t_sim)

    t_out, x_out, y_out = model.simulate_fractional_system(u_sim)

    print(f"Simulation completed: {len(t_out)} time steps")

    # Plot results
    print("\nGenerating analysis plots...")
    model.plot_foss_analysis(state_spaces, mtecm_results)

    # Estimate Bayesian parameters and plot all estimations
    params_bayes = model.estimate_fractional_parameters(time_series, method='bayesian')
    model.plot_parameter_estimation(params_ls, params_kf, params_bayes)

    # Demonstrate different fractional orders
    print("\n--- Analysis with Different Fractional Orders ---")
    for alpha in [0.5, 0.7]:
        print(f"\nα = {alpha}:")

        # Create model with different alpha
        model_alpha = FractionalStateSpaceModel(alpha=alpha, dim=3)

        # FOSS reconstruction
        state_spaces_alpha = model_alpha.foss_reconstruction(time_series, alpha_values=[alpha])

        # MTECM analysis
        mtecm_alpha = model_alpha.mtecm_foss_analysis(state_spaces_alpha)

        # Find best alpha based on total entropy
        best_entropy = max([results['total_entropy']
                           for results in mtecm_alpha.values()])
        print(f"  Best Total Entropy: {best_entropy:.4f}")

        # Stability analysis
        stability_alpha = model_alpha.stability_analysis()
        print(
            f"  Stability: {'Stable' if stability_alpha['is_stable'] else 'Unstable'}")

    print(f"\n" + "=" * 60)
    print("TUTORIAL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
