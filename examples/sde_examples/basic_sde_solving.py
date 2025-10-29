"""
Basic SDE Solving Examples

Demonstrates solving simple fractional stochastic differential equations
using the hpfracc SDE solvers.

Examples:
1. Ornstein-Uhlenbeck process (mean-reverting)
2. Geometric Brownian motion (exponential growth)
3. Mean-reverting process with fractional order

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
from hpfracc.solvers import solve_fractional_sde

# Set random seed for reproducibility
np.random.seed(42)


def ornstein_uhlenbeck_drift(t, x):
    """Drift for Ornstein-Uhlenbeck process: dx = theta*(mu - x)dt + sigma*dW"""
    theta = 1.5  # Mean reversion rate
    mu = 0.5     # Long-term mean
    return theta * (mu - x)


def ornstein_uhlenbeck_diffusion(t, x):
    """Diffusion for Ornstein-Uhlenbeck process"""
    sigma = 0.3  # Volatility
    return sigma * np.ones_like(x)


def geometric_brownian_drift(t, x):
    """Drift for Geometric Brownian Motion: dX = mu*X*dt + sigma*X*dW"""
    mu = 0.1  # Drift rate
    return mu * x


def geometric_brownian_diffusion(t, x):
    """Diffusion for Geometric Brownian Motion"""
    sigma = 0.2  # Volatility
    return sigma * x


def mean_reverting_drift(t, x):
    """Mean-reverting drift with time-dependent rate"""
    theta = 2.0
    mu = 1.0
    return theta * (mu - x) * np.exp(-0.1 * t)


def mean_reverting_diffusion(t, x):
    """State-dependent diffusion"""
    return 0.2 * np.sqrt(np.abs(x)) + 0.05


def example_1_ornstein_uhlenbeck():
    """Example 1: Ornstein-Uhlenbeck Process"""
    print("=" * 80)
    print("Example 1: Ornstein-Uhlenbeck Process (Mean-Reverting)")
    print("=" * 80)
    
    # Initial condition
    x0 = np.array([0.0])
    
    # Solve with different fractional orders
    fractional_orders = [0.5, 0.7, 1.0]
    results = {}
    
    for alpha in fractional_orders:
        print(f"\nSolving with fractional order α = {alpha}")
        solution = solve_fractional_sde(
            drift=ornstein_uhlenbeck_drift,
            diffusion=ornstein_uhlenbeck_diffusion,
            x0=x0,
            t_span=(0, 5),
            fractional_order=alpha,
            method="euler_maruyama",
            num_steps=200
        )
        results[alpha] = solution
        
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot trajectories
    ax = axes[0]
    for alpha, solution in results.items():
        ax.plot(solution.t, solution.y[:, 0], label=f'α = {alpha}', linewidth=1.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Long-term mean')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State x(t)', fontsize=12)
    ax.set_title('Ornstein-Uhlenbeck Process Trajectories', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot mean and std
    ax = axes[1]
    for alpha, solution in results.items():
        mean = np.mean(solution.y, axis=0)
        std = np.std(solution.y, axis=0)
        ax.plot(solution.t, mean, label=f'Mean (α={alpha})', linewidth=1.5)
        ax.fill_between(solution.t, mean - std, mean + std, alpha=0.2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Theoretical mean')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mean ± Std', fontsize=12)
    ax.set_title('Statistical Properties', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/sde_examples/ornstein_uhlenbeck.png', dpi=150)
    print("\n✓ Saved: examples/sde_examples/ornstein_uhlenbeck.png")
    
    return results


def example_2_geometric_brownian_motion():
    """Example 2: Geometric Brownian Motion"""
    print("\n" + "=" * 80)
    print("Example 2: Geometric Brownian Motion (Exponential Growth)")
    print("=" * 80)
    
    # Initial condition (positive)
    x0 = np.array([1.0])
    
    # Solve with fractional order
    solution = solve_fractional_sde(
        drift=geometric_brownian_drift,
        diffusion=geometric_brownian_diffusion,
        x0=x0,
        t_span=(0, 2),
        fractional_order=0.5,
        method="milstein",  # Use higher-order method
        num_steps=200
    )
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(solution.t, solution.y[:, 0], 'b-', linewidth=1.5, label='GBM Trajectory')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State X(t)', fontsize=12)
    ax.set_title('Geometric Brownian Motion', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('examples/sde_examples/geometric_brownian_motion.png', dpi=150)
    print("\n✓ Saved: examples/sde_examples/geometric_brownian_motion.png")
    
    return solution


def example_3_mean_reverting_fractional():
    """Example 3: Mean-Reverting Process with Fractional Order"""
    print("\n" + "=" * 80)
    print("Example 3: Mean-Reverting Process (Fractional Order)")
    print("=" * 80)
    
    # Initial condition
    x0 = np.array([0.5])
    
    # Solve with different fractional orders
    alphas = [0.3, 0.5, 0.7, 1.0]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alpha in alphas:
        print(f"\nSolving with fractional order α = {alpha}")
        solution = solve_fractional_sde(
            drift=mean_reverting_drift,
            diffusion=mean_reverting_diffusion,
            x0=x0,
            t_span=(0, 3),
            fractional_order=alpha,
            method="euler_maruyama",
            num_steps=150
        )
        ax.plot(solution.t, solution.y[:, 0], label=f'α = {alpha}', linewidth=1.5)
    
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target mean')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State x(t)', fontsize=12)
    ax.set_title('Mean-Reverting Process with Fractional Derivatives', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/sde_examples/mean_reverting_fractional.png', dpi=150)
    print("\n✓ Saved: examples/sde_examples/mean_reverting_fractional.png")


def example_4_trajectory_comparison():
    """Example 4: Comparison of Multiple Trajectories"""
    print("\n" + "=" * 80)
    print("Example 4: Multiple Stochastic Trajectories")
    print("=" * 80)
    
    # Generate multiple trajectories
    n_trajectories = 10
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Ornstein-Uhlenbeck trajectories
    ax = axes[0, 0]
    for i in range(n_trajectories):
        solution = solve_fractional_sde(
            drift=ornstein_uhlenbeck_drift,
            diffusion=ornstein_uhlenbeck_diffusion,
            x0=np.array([np.random.uniform(-1, 1)]),
            t_span=(0, 3),
            fractional_order=0.5,
            num_steps=150,
            seed=i
        )
        ax.plot(solution.t, solution.y[:, 0], alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Mean')
    ax.set_title('Ornstein-Uhlenbeck Trajectories', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Geometric Brownian Motion trajectories
    ax = axes[0, 1]
    for i in range(n_trajectories):
        solution = solve_fractional_sde(
            drift=geometric_brownian_drift,
            diffusion=geometric_brownian_diffusion,
            x0=np.array([1.0]),
            t_span=(0, 2),
            fractional_order=0.5,
            num_steps=150,
            seed=i
        )
        ax.plot(solution.t, solution.y[:, 0], alpha=0.5, linewidth=0.8)
    ax.set_title('Geometric Brownian Motion Trajectories', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('State (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Statistical summary - OU process
    ax = axes[1, 0]
    all_trajectories = []
    for i in range(50):
        solution = solve_fractional_sde(
            drift=ornstein_uhlenbeck_drift,
            diffusion=ornstein_uhlenbeck_diffusion,
            x0=np.array([0.0]),
            t_span=(0, 3),
            fractional_order=0.5,
            num_steps=150,
            seed=i
        )
        all_trajectories.append(solution.y[:, 0])
    
    all_trajectories = np.array(all_trajectories)
    mean = np.mean(all_trajectories, axis=0)
    std = np.std(all_trajectories, axis=0)
    
    ax.plot(solution.t, mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(solution.t, mean - std, mean + std, alpha=0.3, label='±1 Std')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Theoretical mean')
    ax.set_title('Statistical Properties (50 trajectories)', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution at final time
    ax = axes[1, 1]
    final_values = all_trajectories[:, -1]
    ax.hist(final_values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Theoretical mean')
    ax.set_title('Final State Distribution', fontweight='bold')
    ax.set_xlabel('State Value')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/sde_examples/trajectory_comparison.png', dpi=150)
    print("\n✓ Saved: examples/sde_examples/trajectory_comparison.png")


def main():
    """Run all examples"""
    print("=" * 80)
    print("Basic SDE Solving Examples")
    print("=" * 80)
    
    # Run examples
    example_1_ornstein_uhlenbeck()
    example_2_geometric_brownian_motion()
    example_3_mean_reverting_fractional()
    example_4_trajectory_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    
    plt.close('all')


if __name__ == "__main__":
    main()
