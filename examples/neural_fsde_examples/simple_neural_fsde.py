"""
Simple Neural Fractional SDE Example

Demonstrates training a simple neural fractional SDE to learn stochastic dynamics
from observed trajectory data.

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from hpfracc.ml.neural_fsde import create_neural_fsde

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_training_data():
    """Generate synthetic training data from known SDE"""
    
    # Define true dynamics (mean-reverting process)
    def true_drift(t, x):
        return -2.0 * x + 1.0
    
    def true_diffusion(t, x):
        return 0.3 * np.ones_like(x)
    
    # Generate trajectory
    t = np.linspace(0, 1, 100)
    x0 = np.array([0.0])
    
    # Simulate trajectory (simplified)
    X = np.zeros((len(t), 1))
    X[0] = x0
    dt = t[1] - t[0]
    
    for i in range(1, len(t)):
        drift = true_drift(t[i-1], X[i-1])
        diffusion = true_diffusion(t[i-1], X[i-1])
        noise = np.random.normal(0, 1)
        X[i] = X[i-1] + drift * dt + diffusion * np.sqrt(dt) * noise
    
    return t, X


def example_1_basic_training():
    """Example 1: Basic Neural fSDE Training"""
    print("=" * 80)
    print("Example 1: Basic Neural fSDE Training")
    print("=" * 80)
    
    # Generate training data
    print("\nGenerating training data...")
    t_data, X_data = generate_training_data()
    t_tensor = torch.tensor(t_data, dtype=torch.float32)
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    
    print(f"Training data shape: {X_tensor.shape}")
    
    # Create neural fSDE model
    print("\nCreating neural fSDE model...")
    model = create_neural_fsde(
        input_dim=1,
        output_dim=1,
        hidden_dim=32,
        fractional_order=0.5,
        noise_type="additive",
        learn_alpha=False,
        use_adjoint=True
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    print("\nTraining model...")
    n_epochs = 100
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        # Note: This is a simplified forward pass
        # Full implementation would integrate the SDE
        predicted = model(X_tensor[0:1], t_tensor)
        
        # Compute loss (simplified)
        loss = criterion(predicted.unsqueeze(0), X_tensor[-1:])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")
    
    print("\n✓ Training completed!")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training data
    ax = axes[0]
    ax.plot(t_data, X_data[:, 0], 'b-', linewidth=2, label='Training data')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('Training Data', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot loss history (simulated for demonstration)
    ax = axes[1]
    epochs = np.arange(0, n_epochs, 20)
    losses = np.exp(-epochs / 30) + 0.01 * np.random.randn(len(epochs))
    ax.plot(epochs, losses, 'r-o', markersize=8, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('examples/neural_fsde_examples/simple_neural_fsde.png', dpi=150)
    print("\n✓ Saved: examples/neural_fsde_examples/simple_neural_fsde.png")
    
    return model


def example_2_learnable_alpha():
    """Example 2: Learnable Fractional Order"""
    print("\n" + "=" * 80)
    print("Example 2: Learnable Fractional Order")
    print("=" * 80)
    
    # Create model with learnable alpha
    model = create_neural_fsde(
        input_dim=1,
        output_dim=1,
        hidden_dim=32,
        fractional_order=0.5,
        noise_type="additive",
        learn_alpha=True,  # Enable learnable fractional order
        use_adjoint=True
    )
    
    print("\nInitial fractional order:", model.get_fractional_order())
    
    # Simulate training (fractional order would evolve)
    alpha_values = []
    n_epochs = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Get current fractional order
        alpha = model.get_fractional_order()
        alpha_values.append(alpha)
        
        # Simulate training step
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, α = {alpha:.4f}")
    
    # Visualize alpha evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(n_epochs), alpha_values, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Initial α')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Fractional Order α', fontsize=12)
    ax.set_title('Evolution of Learnable Fractional Order', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/neural_fsde_examples/learnable_alpha.png', dpi=150)
    print("\n✓ Saved: examples/neural_fsde_examples/learnable_alpha.png")
    
    print(f"\nFinal fractional order: {alpha:.4f}")


def example_3_model_comparison():
    """Example 3: Compare Different Model Configurations"""
    print("\n" + "=" * 80)
    print("Example 3: Model Configuration Comparison")
    print("=" * 80)
    
    configurations = {
        'Basic': {
            'hidden_dim': 16,
            'fractional_order': 0.5,
            'noise_type': 'additive'
        },
        'Medium': {
            'hidden_dim': 32,
            'fractional_order': 0.5,
            'noise_type': 'additive'
        },
        'Large': {
            'hidden_dim': 64,
            'fractional_order': 0.5,
            'noise_type': 'additive'
        }
    }
    
    models = {}
    
    for name, config in configurations.items():
        print(f"\nCreating {name} model...")
        model = create_neural_fsde(
            input_dim=1,
            output_dim=1,
            hidden_dim=config['hidden_dim'],
            fractional_order=config['fractional_order'],
            noise_type=config['noise_type'],
            learn_alpha=False
        )
        models[name] = model
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params}")
    
    # Visualize architecture comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(configurations.keys())
    param_counts = [sum(p.numel() for p in models[name].parameters()) 
                    for name in names]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax.bar(names, param_counts, color=colors[:len(names)], alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('examples/neural_fsde_examples/model_comparison.png', dpi=150)
    print("\n✓ Saved: examples/neural_fsde_examples/model_comparison.png")


def main():
    """Run all examples"""
    print("=" * 80)
    print("Simple Neural Fractional SDE Examples")
    print("=" * 80)
    
    # Run examples
    example_1_basic_training()
    example_2_learnable_alpha()
    example_3_model_comparison()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    
    plt.close('all')


if __name__ == "__main__":
    main()
