"""
Fractional Physics-Informed Neural Operator (Fractional PINO) Experiment

This experiment demonstrates the use of hpfracc library for implementing
a Fractional PINO that can learn and solve fractional differential equations
using the fractional Laplacian and fractional Fourier Transform operators.

Author: David
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from typing import Tuple, Callable, Optional

# Import hpfracc library components
try:
    from algorithms import (
        fractional_laplacian,
        fractional_fourier_transform,
        optimized_riemann_liouville,
        optimized_caputo
    )
    print("✅ hpfracc library imported successfully!")
except ImportError:
    print("❌ hpfracc library not found. Please install with: pip install hpfracc")
    exit(1)


class FractionalPINO(nn.Module):
    """
    Fractional Physics-Informed Neural Operator
    
    This model combines neural networks with fractional calculus operators
    from the hpfracc library to solve fractional differential equations.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
        fractional_order: float = 0.5,
        operator_type: str = "laplacian"
    ):
        super(FractionalPINO, self).__init__()
        
        self.fractional_order = fractional_order
        self.operator_type = operator_type
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.module.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder"""
        z = self.encode(x)
        return self.decode(z)
    
    def fractional_operator(self, x: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Apply fractional operator using hpfracc library
        
        Args:
            x: Input tensor
            alpha: Fractional order (uses self.fractional_order if None)
        
        Returns:
            Tensor with fractional operator applied
        """
        if alpha is None:
            alpha = self.fractional_order
        
        # Convert to numpy for hpfracc operations
        x_np = x.detach().cpu().numpy().flatten()
        
        if self.operator_type == "laplacian":
            # Use fractional Laplacian
            result = fractional_laplacian(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                alpha,
                method="spectral"
            )
        elif self.operator_type == "fourier":
            # Use fractional Fourier transform
            u, result = fractional_fourier_transform(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                alpha,
                method="fast"
            )
        elif self.operator_type == "riemann_liouville":
            # Use Riemann-Liouville derivative
            h = 1.0 / (len(x_np) - 1)
            result = optimized_riemann_liouville(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                alpha,
                h
            )
        elif self.operator_type == "caputo":
            # Use Caputo derivative
            h = 1.0 / (len(x_np) - 1)
            result = optimized_caputo(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                alpha,
                h
            )
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")
        
        # Convert back to tensor
        return torch.tensor(result, dtype=x.dtype, device=x.device).unsqueeze(-1)


class EnhancedFractionalPINOTrainer:
    """
    Enhanced training scheme for Fractional PINO with adaptive learning rates,
    curriculum learning, and physics-informed loss functions.
    """
    
    def __init__(
        self,
        model: FractionalPINO,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        physics_weight: float = 1.0,
        reconstruction_weight: float = 1.0
    ):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.physics_weight = physics_weight
        self.reconstruction_weight = reconstruction_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def physics_informed_loss(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics-informed loss combining reconstruction and physics constraints
        
        Args:
            x: Input coordinates
            y_pred: Predicted solution
            y_true: True solution (if available)
        
        Returns:
            Total loss and loss components dictionary
        """
        # Reconstruction loss
        if y_true is not None:
            recon_loss = self.mse_loss(y_pred, y_true)
        else:
            recon_loss = torch.tensor(0.0, device=x.device)
        
        # Physics loss: fractional operator should satisfy the equation
        # For example: (-Δ)^α u + λu = f
        alpha = self.model.fractional_order
        lambda_param = 1.0  # Can be made configurable
        
        # Apply fractional operator to prediction
        frac_op_pred = self.model.fractional_operator(y_pred, alpha)
        
        # Physics constraint: (-Δ)^α u + λu = f
        # For simplicity, we assume f = 0 (homogeneous equation)
        physics_residual = frac_op_pred + lambda_param * y_pred
        
        physics_loss = self.mse_loss(physics_residual, torch.zeros_like(physics_residual))
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.physics_weight * physics_loss
        )
        
        loss_components = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'physics': physics_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'reconstruction': 0.0, 'physics': 0.0}
        
        for batch_idx, (x, y) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(x)
            
            # Compute loss
            loss, components = self.physics_informed_loss(x, y_pred, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
        
        # Average losses
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # Update learning rate
        self.scheduler.step(avg_loss)
        
        return {
            'epoch': epoch,
            'total_loss': avg_loss,
            **loss_components
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'reconstruction': 0.0, 'physics': 0.0}
        
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = self.model(x)
                loss, components = self.physics_informed_loss(x, y_pred, y)
                
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += components[key]
        
        # Average losses
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }


def generate_fractional_pde_data(
    n_points: int = 1000,
    n_samples: int = 100,
    fractional_order: float = 0.5,
    operator_type: str = "laplacian"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for fractional PDE experiments
    
    Args:
        n_points: Number of spatial points
        n_samples: Number of samples
        fractional_order: Fractional order
        operator_type: Type of fractional operator
    
    Returns:
        Tuple of (x_coordinates, y_solutions)
    """
    print(f"Generating {n_samples} samples with {n_points} points each...")
    
    # Spatial domain
    x = np.linspace(0, 1, n_points)
    
    # Generate random initial conditions
    solutions = []
    
    for i in range(n_samples):
        # Random Fourier coefficients
        n_modes = 10
        coeffs = np.random.normal(0, 1, n_modes)
        
        # Construct solution as sum of sine functions
        solution = np.zeros_like(x)
        for j in range(n_modes):
            freq = (j + 1) * np.pi
            solution += coeffs[j] * np.sin(freq * x)
        
        # Normalize
        solution = solution / np.max(np.abs(solution))
        solutions.append(solution)
    
    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).repeat(n_samples, 1)
    y_tensor = torch.tensor(np.array(solutions), dtype=torch.float32).unsqueeze(-1)
    
    return x_tensor, y_tensor


def benchmark_fractional_pino(
    operator_types: list = ["laplacian", "fourier", "riemann_liouville", "caputo"],
    fractional_orders: list = [0.25, 0.5, 0.75],
    n_points: int = 1000,
    n_samples: int = 50,
    epochs: int = 100
) -> dict:
    """
    Benchmark Fractional PINO against different operators and orders
    
    Args:
        operator_types: List of fractional operators to test
        fractional_orders: List of fractional orders to test
        n_points: Number of spatial points
        n_samples: Number of samples
        epochs: Number of training epochs
    
    Returns:
        Dictionary with benchmark results
    """
    print("🚀 Starting Fractional PINO Benchmark...")
    print("=" * 60)
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for operator in operator_types:
        results[operator] = {}
        
        for alpha in fractional_orders:
            print(f"\n🔬 Testing {operator} operator with α = {alpha}")
            print("-" * 40)
            
            # Generate data
            x, y = generate_fractional_pde_data(
                n_points=n_points,
                n_samples=n_samples,
                fractional_order=alpha,
                operator_type=operator
            )
            
            # Split data
            train_size = int(0.8 * n_samples)
            x_train, x_val = x[:train_size], x[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Create data loaders
            train_dataset = TensorDataset(x_train, y_train)
            val_dataset = TensorDataset(x_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Initialize model
            model = FractionalPINO(
                input_dim=1,
                hidden_dim=64,
                output_dim=1,
                num_layers=4,
                fractional_order=alpha,
                operator_type=operator
            ).to(device)
            
            # Initialize trainer
            trainer = EnhancedFractionalPINOTrainer(
                model=model,
                learning_rate=1e-3,
                physics_weight=1.0,
                reconstruction_weight=1.0
            )
            
            # Training loop
            train_losses = []
            val_losses = []
            
            start_time = time.time()
            
            for epoch in range(epochs):
                # Train
                train_metrics = trainer.train_epoch(train_loader, epoch)
                train_losses.append(train_metrics['total_loss'])
                
                # Validate
                if epoch % 10 == 0:
                    val_metrics = trainer.validate(val_loader)
                    val_losses.append(val_metrics['total_loss'])
                    
                    print(f"Epoch {epoch:3d}: Train Loss = {train_metrics['total_loss']:.6f}, "
                          f"Val Loss = {val_metrics['total_loss']:.6f}")
            
            training_time = time.time() - start_time
            
            # Final evaluation
            final_val_metrics = trainer.validate(val_loader)
            
            # Store results
            results[operator][alpha] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': final_val_metrics['total_loss'],
                'training_time': training_time,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'model': model
            }
            
            print(f"✅ {operator} (α={alpha}) completed in {training_time:.2f}s")
            print(f"   Final validation loss: {final_val_metrics['total_loss']:.6f}")
    
    return results


def plot_benchmark_results(results: dict):
    """Plot benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fractional PINO Benchmark Results', fontsize=16)
    
    # Plot 1: Final validation losses
    ax1 = axes[0, 0]
    operators = list(results.keys())
    alphas = list(results[operators[0]].keys())
    
    for operator in operators:
        losses = [results[operator][alpha]['final_val_loss'] for alpha in alphas]
        ax1.plot(alphas, losses, 'o-', label=operator, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Fractional Order α')
    ax1.set_ylabel('Final Validation Loss')
    ax1.set_title('Final Validation Loss vs Fractional Order')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training times
    ax2 = axes[0, 1]
    for operator in operators:
        times = [results[operator][alpha]['training_time'] for alpha in alphas]
        ax2.bar([f"{operator}\nα={alpha}" for alpha in alphas], times, alpha=0.7)
    
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Training curves for best performing model
    ax3 = axes[1, 0]
    best_operator = min(results.keys(), key=lambda op: 
                       min(results[op][alpha]['final_val_loss'] for alpha in results[op].keys()))
    best_alpha = min(results[best_operator].keys(), 
                    key=lambda a: results[best_operator][a]['final_val_loss'])
    
    train_losses = results[best_operator][best_alpha]['train_losses']
    val_losses = results[best_operator][best_alpha]['val_losses']
    
    ax3.plot(train_losses, label='Training Loss', linewidth=2)
    ax3.plot(range(0, len(train_losses), 10), val_losses, 'o-', label='Validation Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title(f'Training Curves: {best_operator} (α={best_alpha})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Sample predictions
    ax4 = axes[1, 1]
    model = results[best_operator][best_alpha]['model']
    model.eval()
    
    with torch.no_grad():
        x_test = torch.linspace(0, 1, 100).unsqueeze(-1)
        y_pred = model(x_test)
        
        ax4.plot(x_test.numpy(), y_pred.numpy(), 'b-', linewidth=2, label='Prediction')
        ax4.set_xlabel('x')
        ax4.set_ylabel('u(x)')
        ax4.set_title(f'Sample Prediction: {best_operator} (α={best_alpha})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fractional_pino_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_classical_methods(
    fractional_order: float = 0.5,
    n_points: int = 1000
) -> dict:
    """
    Compare Fractional PINO with classical numerical methods
    
    Args:
        fractional_order: Fractional order to test
        n_points: Number of spatial points
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n🔬 Comparing with Classical Methods (α = {fractional_order})")
    print("=" * 60)
    
    # Generate test problem
    x = np.linspace(0, 1, n_points)
    f = lambda t: np.sin(np.pi * t)  # Test function
    
    results = {}
    
    # Test hpfracc operators directly
    print("Testing hpfracc operators...")
    
    # Fractional Laplacian
    start_time = time.time()
    laplacian_result = fractional_laplacian(f, x, fractional_order, method="spectral")
    laplacian_time = time.time() - start_time
    
    # Fractional Fourier Transform
    start_time = time.time()
    u, fourier_result = fractional_fourier_transform(f, x, fractional_order, method="fast")
    fourier_time = time.time() - start_time
    
    # Riemann-Liouville derivative
    start_time = time.time()
    h = x[1] - x[0]
    rl_result = optimized_riemann_liouville(f, x, fractional_order, h)
    rl_time = time.time() - start_time
    
    # Caputo derivative
    start_time = time.time()
    caputo_result = optimized_caputo(f, x, fractional_order, h)
    caputo_time = time.time() - start_time
    
    results['hpfracc_operators'] = {
        'laplacian': {'result': laplacian_result, 'time': laplacian_time},
        'fourier': {'result': fourier_result, 'time': fourier_time},
        'riemann_liouville': {'result': rl_result, 'time': rl_time},
        'caputo': {'result': caputo_result, 'time': caputo_time}
    }
    
    print(f"✅ hpfracc operators completed:")
    print(f"   Laplacian: {laplacian_time:.4f}s")
    print(f"   Fourier: {fourier_time:.4f}s")
    print(f"   Riemann-Liouville: {rl_time:.4f}s")
    print(f"   Caputo: {caputo_time:.4f}s")
    
    return results


def main():
    """Main experiment function"""
    print("🚀 Fractional PINO Experiment with hpfracc Library")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Benchmark Fractional PINO
    benchmark_results = benchmark_fractional_pino(
        operator_types=["laplacian", "fourier", "riemann_liouville", "caputo"],
        fractional_orders=[0.25, 0.5, 0.75],
        n_points=500,  # Reduced for faster execution
        n_samples=30,  # Reduced for faster execution
        epochs=50      # Reduced for faster execution
    )
    
    # Plot results
    plot_benchmark_results(benchmark_results)
    
    # Compare with classical methods
    classical_results = compare_with_classical_methods(
        fractional_order=0.5,
        n_points=1000
    )
    
    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    best_operator = min(benchmark_results.keys(), key=lambda op: 
                       min(benchmark_results[op][alpha]['final_val_loss'] 
                           for alpha in benchmark_results[op].keys()))
    best_alpha = min(benchmark_results[best_operator].keys(), 
                    key=lambda a: benchmark_results[best_operator][a]['final_val_loss'])
    
    print(f"🏆 Best performing model: {best_operator} (α = {best_alpha})")
    print(f"   Final validation loss: {benchmark_results[best_operator][best_alpha]['final_val_loss']:.6f}")
    print(f"   Training time: {benchmark_results[best_operator][best_alpha]['training_time']:.2f}s")
    
    print(f"\n⚡ hpfracc Performance:")
    for op, data in classical_results['hpfracc_operators'].items():
        print(f"   {op}: {data['time']:.4f}s")
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"📈 Results saved to: fractional_pino_benchmark_results.png")


if __name__ == "__main__":
    main()
