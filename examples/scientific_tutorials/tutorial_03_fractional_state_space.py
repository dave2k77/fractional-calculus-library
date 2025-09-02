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
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import HPFRACC components
from hpfracc.core.definitions import FractionalOrder
from hpfracc.core.derivatives import create_fractional_derivative
from hpfracc.core.integrals import create_fractional_integral
from hpfracc.special import gamma
from hpfracc.core.utilities import validate_fractional_order, timing_decorator

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
        self.derivative = create_fractional_derivative(alpha, method="RL")
        self.integral = create_fractional_integral(alpha, method="RL")

        # Initialize state space matrices
        self.A = None  # State matrix
        self.B = None  # Input matrix
        self.C = None  # Output matrix
        self.D = None  # Feedthrough matrix

        print(f"FractionalStateSpaceModel initialized (α={alpha}, dim={dim})")

    def generate_lorenz_data(self, duration=100, dt=0.01, noise_level=0.01):
        """
        Generate Lorenz system data with fractional dynamics.

        Parameters:
        -----------
        duration : float
            Simulation duration
        dt : float
            Time step
        noise_level : float
            Noise amplitude

        Returns:
        --------
        t : array
            Time vector
        x : array
            State variables
        """
        n_steps = int(duration / dt)
        t = np.linspace(0, duration, n_steps)

        # Lorenz parameters
        sigma = 10.0
        rho = 28.0
        beta_lorenz = 8/3

        # Initialize state
        x = np.zeros((n_steps, 3))
        x[0] = [1.0, 1.0, 1.0]

        # Generate fractional Lorenz dynamics
        for i in range(1, n_steps):
            # Standard Lorenz equations
            dx1 = sigma * (x[i-1, 1] - x[i-1, 0])
            dx2 = x[i-1, 0] * (rho - x[i-1, 2]) - x[i-1, 1]
            dx3 = x[i-1, 0] * x[i-1, 1] - beta_lorenz * x[i-1, 2]

            # Add fractional dynamics
            if self.alpha.alpha != 1.0:
                # Apply fractional derivative effect
                frac_factor = (dt ** (self.alpha.alpha - 1)) / \
                    gamma(self.alpha.alpha)
                dx1 *= frac_factor
                dx2 *= frac_factor
                dx3 *= frac_factor

            # Euler integration
            x[i, 0] = x[i-1, 0] + dt * dx1
            x[i, 1] = x[i-1, 1] + dt * dx2
            x[i, 2] = x[i-1, 2] + dt * dx3

            # Add noise
            x[i] += noise_level * np.random.randn(3)

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
            # Create fractional derivative operator
            derivative_op = create_fractional_derivative(alpha, method="RL")

            # Apply fractional derivative to time series
            time = np.arange(n_samples)

            def series_func(t):
                idx = int(t)
                if idx >= n_samples:
                    idx = n_samples - 1
                return time_series[idx]

            fractional_series = derivative_op(series_func, time)

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

        for alpha, state_space in state_spaces.items():
            # Normalize state space
            scaler = StandardScaler()
            state_space_norm = scaler.fit_transform(state_space)

            # Cluster states
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
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
                'cluster_centers': cluster_centers
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
        else:
            raise ValueError(f"Unknown method: {method}")

    def _least_squares_estimation(self, time_series):
        """
        Least squares parameter estimation.
        """
        n_samples = len(time_series)

        # Create feature matrix
        X = np.zeros((n_samples - 1, 3))
        X[:, 0] = time_series[:-1]  # Previous value
        X[:, 1] = np.arange(n_samples - 1)  # Time
        X[:, 2] = 1  # Constant term

        # Target vector (fractional derivative)
        time = np.arange(n_samples)

        def series_func(t):
            idx = int(t)
            if idx >= n_samples:
                idx = n_samples - 1
            return time_series[idx]

        frac_deriv = self.derivative(series_func, time[:-1])
        y = frac_deriv

        # Solve least squares problem
        params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        return {
            'A': params[0],
            'B': params[1],
            'C': params[2],
            'residuals': residuals[0] if len(residuals) > 0 else 0
        }

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

        for k in range(n_samples):
            # Prediction step
            # Simplified state transition (would be more complex in practice)
            x_pred = x_est
            P_pred = P + Q

            # Update step
            # Simplified measurement model
            H = np.array([1, 0, 0])  # Only first state is observed
            y = time_series[k]

            # Kalman gain
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)

            # State update
            x_est = x_pred + K * (y - H @ x_pred)
            P = (np.eye(self.dim) - K @ H) @ P_pred

            state_history[k] = x_est

        return {
            'state_history': state_history,
            'final_state': x_est,
            'final_covariance': P
        }

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
        condition_number = linalg.cond(A_matrix)

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
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Plot 1: State spaces for different α
        alphas = list(state_spaces.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

        for i, alpha in enumerate(alphas):
            state_space = state_spaces[alpha]
            axes[0, 0].scatter(state_space[:, 0], state_space[:, 1],
                               c=colors[i], alpha=0.6, s=1, label=f'α={alpha}')

        axes[0, 0].set_xlabel('x(t)')
        axes[0, 0].set_ylabel('x(t+τ)')
        axes[0, 0].set_title('FOSS Reconstruction')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Entropy measures vs α
        alphas = list(mtecm_results.keys())
        intra_entropy = [mtecm_results[alpha]['intra_entropy']
                         for alpha in alphas]
        inter_entropy = [mtecm_results[alpha]['inter_entropy']
                         for alpha in alphas]
        transition_entropy = [mtecm_results[alpha]
                              ['transition_entropy'] for alpha in alphas]
        total_entropy = [mtecm_results[alpha]['total_entropy']
                         for alpha in alphas]

        axes[0, 1].plot(alphas, intra_entropy, 'b-o', label='Intra-sample')
        axes[0, 1].plot(alphas, inter_entropy, 'r-s', label='Inter-sample')
        axes[0, 1].plot(alphas, transition_entropy, 'g-^', label='Transition')
        axes[0, 1].plot(alphas, total_entropy, 'k-*', label='Total')
        axes[0, 1].set_xlabel('Fractional Order α')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].set_title('MTECM-FOSS Entropy Analysis')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Transition matrices
        best_alpha = alphas[np.argmax(total_entropy)]
        transition_matrix = mtecm_results[best_alpha]['transition_matrix']

        im = axes[0, 2].imshow(
            transition_matrix, cmap='viridis', aspect='auto')
        axes[0, 2].set_xlabel('Next State')
        axes[0, 2].set_ylabel('Current State')
        axes[0, 2].set_title(f'Transition Matrix (α={best_alpha})')
        plt.colorbar(im, ax=axes[0, 2])

        # Plot 4: State clustering
        cluster_labels = mtecm_results[best_alpha]['cluster_labels']
        cluster_centers = mtecm_results[best_alpha]['cluster_centers']
        state_space = state_spaces[best_alpha]

        scatter = axes[1, 0].scatter(state_space[:, 0], state_space[:, 1],
                                     c=cluster_labels, cmap='tab10', alpha=0.6, s=1)
        axes[1, 0].scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                           c='red', s=100, marker='x', linewidths=3)
        axes[1, 0].set_xlabel('x(t)')
        axes[1, 0].set_ylabel('x(t+τ)')
        axes[1, 0].set_title(f'State Clustering (α={best_alpha})')
        plt.colorbar(scatter, ax=axes[1, 0])

        # Plot 5: Stability analysis
        stability_info = self.stability_analysis()
        eigenvals = stability_info['eigenvalues']

        # Plot eigenvalues in complex plane
        axes[1, 1].scatter(eigenvals.real, eigenvals.imag, c='red', s=100)
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Draw stability region
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        axes[1, 1].plot(x_circle, y_circle, 'k--',
                        alpha=0.5, label='Unit Circle')

        axes[1, 1].set_xlabel('Real Part')
        axes[1, 1].set_ylabel('Imaginary Part')
        axes[1, 1].set_title('Eigenvalue Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_aspect('equal')

        # Plot 6: Parameter estimation results
        # Generate test data
        t_test = np.linspace(0, 10, 1000)
        u_test = np.sin(2 * np.pi * 0.5 * t_test)

        # Simulate system
        t_sim, x_sim, y_sim = self.simulate_fractional_system(u_test)

        axes[1, 2].plot(t_sim, y_sim, 'b-', linewidth=2, label='Output')
        axes[1, 2].plot(t_sim, u_test, 'r--', linewidth=1, label='Input')
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Amplitude')
        axes[1, 2].set_title('Fractional System Simulation')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        # Plot 7: Lorenz attractor
        t_lorenz, x_lorenz = self.generate_lorenz_data(duration=50, dt=0.01)

        axes[2, 0].plot(x_lorenz[:, 0], x_lorenz[:, 1], 'b-', linewidth=0.5)
        axes[2, 0].set_xlabel('x')
        axes[2, 0].set_ylabel('y')
        axes[2, 0].set_title('Fractional Lorenz Attractor')
        axes[2, 0].grid(True)

        # Plot 8: Time series analysis
        time_series = x_lorenz[:, 0]  # Use x-component
        state_spaces_lorenz = self.foss_reconstruction(time_series)

        # Plot reconstruction for α=0.5
        if 0.5 in state_spaces_lorenz:
            state_space_05 = state_spaces_lorenz[0.5]
            axes[2, 1].scatter(state_space_05[:, 0], state_space_05[:, 1],
                               c=state_space_05[:, 2], cmap='viridis', alpha=0.6, s=1)
            axes[2, 1].set_xlabel('x(t)')
            axes[2, 1].set_ylabel('x(t+τ)')
            axes[2, 1].set_title('Lorenz FOSS (α=0.5)')
            plt.colorbar(axes[2, 1].collections[0], ax=axes[2, 1])

        # Plot 9: Complexity measures
        complexity_measures = []
        for alpha in alphas:
            if alpha in mtecm_results:
                complexity = mtecm_results[alpha]['total_entropy']
                complexity_measures.append(complexity)

        axes[2, 2].plot(alphas[:len(complexity_measures)],
                        complexity_measures, 'bo-', linewidth=2)
        axes[2, 2].set_xlabel('Fractional Order α')
        axes[2, 2].set_ylabel('Complexity Measure')
        axes[2, 2].set_title('System Complexity vs α')
        axes[2, 2].grid(True)

        plt.tight_layout()
        plt.show()


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
    t_lorenz, x_lorenz = model.generate_lorenz_data(duration=100, dt=0.01)

    print(f"Lorenz data generated: {len(t_lorenz)} time steps")

    # FOSS reconstruction
    print("Performing FOSS reconstruction...")
    time_series = x_lorenz[:, 0]  # Use x-component
    state_spaces = model.foss_reconstruction(
        time_series, embedding_dim=3, delay=10)

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

    # Demonstrate different fractional orders
    print("\n--- Analysis with Different Fractional Orders ---")
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.3]:
        print(f"\nα = {alpha}:")

        # Create model with different alpha
        model_alpha = FractionalStateSpaceModel(alpha=alpha, dim=3)

        # FOSS reconstruction
        state_spaces_alpha = model_alpha.foss_reconstruction(time_series)

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
