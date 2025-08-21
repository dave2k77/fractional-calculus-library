"""
Simple Fractional PINO Implementation

A Physics-Informed Neural Operator that uses fractional calculus operators
for solving fractional differential equations.

Author: David
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from scipy.fft import fft, ifft, fftfreq

def fractional_laplacian_spectral(f, x, alpha, method="spectral"):
    """Compute fractional Laplacian using spectral method"""
    if callable(f):
        f_array = np.array([f(xi) for xi in x])
    else:
        f_array = f
    
    N = len(x)
    dx = x[1] - x[0]
    
    if method == "spectral":
        k = 2 * np.pi * fftfreq(N, dx)
        f_hat = fft(f_array)
        laplacian_hat = -np.power(np.abs(k), alpha) * f_hat
        laplacian_hat[0] = 0
        result = np.real(ifft(laplacian_hat))
    
    return result

def fractional_fourier_transform_fast(f, x, alpha, method="fast"):
    """Compute fractional Fourier transform using fast method"""
    if callable(f):
        f_array = np.array([f(xi) for xi in x])
    else:
        f_array = f
    
    N = len(x)
    dx = x[1] - x[0]
    
    if method == "fast":
        k = 2 * np.pi * fftfreq(N, dx)
        f_hat = fft(f_array)
        phase = np.exp(1j * np.pi * alpha / 2)
        kernel = np.exp(-1j * np.pi * alpha * k**2 / 2)
        result_hat = phase * kernel * f_hat
        result = ifft(result_hat)
        u = x
    
    return u, result

class SimpleFractionalPINO(nn.Module):
    """Simple Fractional Physics-Informed Neural Operator"""
    
    def __init__(self, hidden_dim=32, fractional_order=0.5, operator_type="laplacian"):
        super().__init__()
        self.fractional_order = fractional_order
        self.operator_type = operator_type
        
        # Neural network
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def apply_fractional_operator(self, x, alpha=None):
        """Apply fractional operator"""
        if alpha is None:
            alpha = self.fractional_order
        
        # Convert to numpy for operator computation
        x_np = x.detach().cpu().numpy().flatten()
        
        if self.operator_type == "laplacian":
            # Create function from array
            def f(t):
                idx = int(t * (len(x_np) - 1))
                return x_np[idx] if 0 <= idx < len(x_np) else 0
            
            # Apply fractional Laplacian
            result = fractional_laplacian_spectral(f, np.linspace(0, 1, len(x_np)), alpha)
            
        elif self.operator_type == "fourier":
            # Create function from array
            def f(t):
                idx = int(t * (len(x_np) - 1))
                return x_np[idx] if 0 <= idx < len(x_np) else 0
            
            # Apply fractional Fourier transform
            u, result = fractional_fourier_transform_fast(f, np.linspace(0, 1, len(x_np)), alpha)
            result = np.real(result)  # Take real part for simplicity
        
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")
        
        # Convert back to tensor
        return torch.tensor(result, dtype=x.dtype, device=x.device).unsqueeze(-1)

def generate_fractional_pde_data(n_points=100, n_samples=50):
    """Generate synthetic data for fractional PDE experiments"""
    print(f"Generating {n_samples} samples with {n_points} points each...")
    
    x = np.linspace(0, 1, n_points)
    solutions = []
    
    for i in range(n_samples):
        # Random Fourier coefficients
        n_modes = 5
        coeffs = np.random.normal(0, 1, n_modes)
        
        # Construct solution as sum of sine functions
        solution = np.zeros_like(x)
        for j in range(n_modes):
            freq = (j + 1) * np.pi
            solution += coeffs[j] * np.sin(freq * x)
        
        # Normalize
        solution = solution / np.max(np.abs(solution))
        solutions.append(solution)
    
    # Convert to tensors - reshape for batch processing
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (n_points, 1)
    y_tensor = torch.tensor(np.array(solutions), dtype=torch.float32)  # (n_samples, n_points)
    
    return x_tensor, y_tensor

def train_fractional_pino(operator_type="laplacian", fractional_order=0.5, epochs=100):
    """Train a Fractional PINO"""
    print(f"\n🚀 Training Fractional PINO ({operator_type}, α={fractional_order})")
    print("=" * 60)
    
    # Generate data
    x, y = generate_fractional_pde_data(n_points=100, n_samples=30)
    
    # Initialize model
    model = SimpleFractionalPINO(
        hidden_dim=32, 
        fractional_order=fractional_order,
        operator_type=operator_type
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    physics_losses = []
    recon_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Process each sample in the batch
        total_recon_loss = 0
        total_physics_loss = 0
        
        for i in range(y.shape[0]):  # For each sample
            # Forward pass for this sample
            y_pred_sample = model(x)  # (n_points, 1)
            y_true_sample = y[i]  # (n_points,)
            
            # Reconstruction loss
            recon_loss_sample = criterion(y_pred_sample.squeeze(), y_true_sample)
            total_recon_loss += recon_loss_sample
            
            # Physics loss: fractional operator constraint
            frac_op_pred = model.apply_fractional_operator(y_pred_sample, fractional_order)
            lambda_param = 1.0
            physics_residual = frac_op_pred + lambda_param * y_pred_sample
            physics_loss_sample = criterion(physics_residual, torch.zeros_like(physics_residual))
            total_physics_loss += physics_loss_sample
        
        # Average losses
        avg_recon_loss = total_recon_loss / y.shape[0]
        avg_physics_loss = total_physics_loss / y.shape[0]
        
        # Total loss
        total_loss = avg_recon_loss + 0.1 * avg_physics_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        losses.append(total_loss.item())
        physics_losses.append(avg_physics_loss.item())
        recon_losses.append(avg_recon_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Total Loss = {total_loss.item():.6f}, "
                  f"Recon = {avg_recon_loss.item():.6f}, Physics = {avg_physics_loss.item():.6f}")
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(x)  # (n_points, 1)
        final_recon_loss = criterion(y_pred.squeeze(), y[0]).item()  # Compare with first sample
        print(f"Final reconstruction loss: {final_recon_loss:.6f}")
    
    return {
        'model': model,
        'training_time': training_time,
        'final_loss': final_recon_loss,
        'losses': losses,
        'physics_losses': physics_losses,
        'recon_losses': recon_losses,
        'x': x,
        'y': y,
        'y_pred': y_pred
    }

def plot_training_results(results, operator_type, fractional_order):
    """Plot training results"""
    print("\n📊 Plotting training results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Fractional PINO Training Results ({operator_type}, α={fractional_order})', fontsize=16)
    
    # Plot 1: Training losses
    axes[0, 0].plot(results['losses'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].plot(results['recon_losses'], 'r-', linewidth=2, label='Reconstruction Loss')
    axes[0, 0].plot(results['physics_losses'], 'g-', linewidth=2, label='Physics Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Sample predictions
    sample_idx = 0
    x_sample = results['x'][sample_idx].numpy()
    y_true = results['y'][sample_idx].numpy()
    y_pred = results['y_pred'][sample_idx].numpy()
    
    axes[0, 1].plot(x_sample, y_true, 'b-', linewidth=2, label='True')
    axes[0, 1].plot(x_sample, y_pred, 'r--', linewidth=2, label='Predicted')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u(x)')
    axes[0, 1].set_title('Sample Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Physics residual
    model = results['model']
    model.eval()
    with torch.no_grad():
        frac_op_pred = model.apply_fractional_operator(results['y_pred'][sample_idx:sample_idx+1])
        physics_residual = frac_op_pred + results['y_pred'][sample_idx:sample_idx+1]
        
    axes[1, 0].plot(x_sample, physics_residual.numpy().flatten(), 'g-', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Physics Residual')
    axes[1, 0].set_title('Physics Constraint Residual')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Loss convergence
    axes[1, 1].plot(results['losses'][-50:], 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch (last 50)')
    axes[1, 1].set_ylabel('Total Loss')
    axes[1, 1].set_title('Loss Convergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fractional_pino_{operator_type}_alpha_{fractional_order}_results.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Results saved to 'fractional_pino_{operator_type}_alpha_{fractional_order}_results.png'")

def benchmark_fractional_pino():
    """Benchmark different Fractional PINO configurations"""
    print("\n⚡ Fractional PINO Benchmark")
    print("=" * 40)
    
    operators = ["laplacian", "fourier"]
    alphas = [0.25, 0.5, 0.75]
    
    benchmark_results = {}
    
    for operator in operators:
        benchmark_results[operator] = {}
        
        for alpha in alphas:
            print(f"\n🔬 Testing {operator} operator with α = {alpha}")
            print("-" * 40)
            
            result = train_fractional_pino(operator, alpha, epochs=50)
            benchmark_results[operator][alpha] = result
            
            print(f"✅ {operator} (α={alpha}) completed in {result['training_time']:.2f}s")
            print(f"   Final loss: {result['final_loss']:.6f}")
    
    return benchmark_results

def plot_benchmark_results(benchmark_results):
    """Plot benchmark results"""
    print("\n📊 Plotting benchmark results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fractional PINO Benchmark Results', fontsize=16)
    
    operators = list(benchmark_results.keys())
    alphas = list(benchmark_results[operators[0]].keys())
    
    # Plot 1: Final losses
    ax1 = axes[0, 0]
    for operator in operators:
        losses = [benchmark_results[operator][alpha]['final_loss'] for alpha in alphas]
        ax1.plot(alphas, losses, 'o-', label=operator, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Fractional Order α')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Final Loss vs Fractional Order')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training times
    ax2 = axes[0, 1]
    for operator in operators:
        times = [benchmark_results[operator][alpha]['training_time'] for alpha in alphas]
        ax2.bar([f"{operator}\nα={alpha}" for alpha in alphas], times, alpha=0.7)
    
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Best model predictions
    ax3 = axes[1, 0]
    best_operator = min(operators, key=lambda op: 
                       min(benchmark_results[op][alpha]['final_loss'] for alpha in benchmark_results[op].keys()))
    best_alpha = min(alphas, key=lambda a: benchmark_results[best_operator][a]['final_loss'])
    
    result = benchmark_results[best_operator][best_alpha]
    x_sample = result['x'].numpy().flatten()
    y_true = result['y'][0].numpy()  # First sample
    y_pred = result['y_pred'].numpy().flatten()
    
    ax3.plot(x_sample, y_true, 'b-', linewidth=2, label='True')
    ax3.plot(x_sample, y_pred, 'r--', linewidth=2, label='Predicted')
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x)')
    ax3.set_title(f'Best Model: {best_operator} (α={best_alpha})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss convergence for best model
    ax4 = axes[1, 1]
    ax4.plot(result['losses'], 'g-', linewidth=2, label='Total Loss')
    ax4.plot(result['recon_losses'], 'b-', linewidth=2, label='Reconstruction')
    ax4.plot(result['physics_losses'], 'r-', linewidth=2, label='Physics')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Curves (Best Model)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fractional_pino_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Benchmark results saved to 'fractional_pino_benchmark_results.png'")

def main():
    """Main function"""
    print("🚀 Simple Fractional PINO Experiment")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Benchmark different configurations
    benchmark_results = benchmark_fractional_pino()
    
    # Plot benchmark results
    plot_benchmark_results(benchmark_results)
    
    # Summary
    print("\n🎉 Experiment Summary")
    print("=" * 25)
    
    best_operator = min(benchmark_results.keys(), key=lambda op: 
                       min(benchmark_results[op][alpha]['final_loss'] 
                           for alpha in benchmark_results[op].keys()))
    best_alpha = min(benchmark_results[best_operator].keys(), 
                    key=lambda a: benchmark_results[best_operator][a]['final_loss'])
    
    print(f"🏆 Best performing model: {best_operator} (α = {best_alpha})")
    print(f"   Final loss: {benchmark_results[best_operator][best_alpha]['final_loss']:.6f}")
    print(f"   Training time: {benchmark_results[best_operator][best_alpha]['training_time']:.2f}s")
    
    print(f"\n✅ Fractional PINO experiment completed successfully!")
    print(f"📈 Results demonstrate physics-informed learning with fractional operators")

if __name__ == "__main__":
    main()
