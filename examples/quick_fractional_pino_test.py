"""
Quick Fractional PINO Test

A simplified version of the Fractional PINO experiment for quick testing
and demonstration of the hpfracc library capabilities.

Author: David
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

# Import hpfracc library
try:
    from algorithms import fractional_laplacian, fractional_fourier_transform
    print("✅ hpfracc library imported successfully!")
except ImportError:
    print("❌ hpfracc library not found. Please install with: pip install hpfracc")
    exit(1)


class SimpleFractionalPINO(nn.Module):
    """Simplified Fractional PINO for quick testing"""
    
    def __init__(self, hidden_dim=32, fractional_order=0.5):
        super().__init__()
        self.fractional_order = fractional_order
        
        # Simple neural network
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def apply_fractional_operator(self, x, operator_type="laplacian"):
        """Apply fractional operator using hpfracc"""
        x_np = x.detach().cpu().numpy().flatten()
        
        if operator_type == "laplacian":
            result = fractional_laplacian(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                self.fractional_order,
                method="spectral"
            )
        elif operator_type == "fourier":
            u, result = fractional_fourier_transform(
                lambda t: x_np[int(t * (len(x_np) - 1))] if t <= 1 else 0,
                np.linspace(0, 1, len(x_np)),
                self.fractional_order,
                method="fast"
            )
        else:
            raise ValueError(f"Unknown operator: {operator_type}")
        
        return torch.tensor(result, dtype=x.dtype, device=x.device).unsqueeze(-1)


def generate_test_data(n_points=100):
    """Generate simple test data"""
    x = np.linspace(0, 1, n_points)
    # Simple sine function
    y = np.sin(2 * np.pi * x)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


def test_fractional_operators():
    """Test hpfracc fractional operators directly"""
    print("\n🔬 Testing hpfracc Fractional Operators")
    print("=" * 40)
    
    # Generate test data
    x = np.linspace(0, 1, 100)
    f = lambda t: np.sin(2 * np.pi * t)
    
    # Test fractional Laplacian
    print("Testing Fractional Laplacian...")
    start_time = time.time()
    laplacian_result = fractional_laplacian(f, x, 0.5, method="spectral")
    laplacian_time = time.time() - start_time
    print(f"✅ Laplacian completed in {laplacian_time:.4f}s")
    
    # Test fractional Fourier transform
    print("Testing Fractional Fourier Transform...")
    start_time = time.time()
    u, fourier_result = fractional_fourier_transform(f, x, 0.5, method="fast")
    fourier_time = time.time() - start_time
    print(f"✅ Fourier Transform completed in {fourier_time:.4f}s")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, f(x), 'b-', linewidth=2, label='Original')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Original Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(x, laplacian_result, 'r-', linewidth=2, label='Fractional Laplacian')
    plt.xlabel('x')
    plt.ylabel('(-Δ)^0.5 f(x)')
    plt.title('Fractional Laplacian (α=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(u, np.real(fourier_result), 'g-', linewidth=2, label='Real part')
    plt.plot(u, np.imag(fourier_result), 'g--', linewidth=2, label='Imaginary part')
    plt.xlabel('u')
    plt.ylabel('FrFT(f)(u)')
    plt.title('Fractional Fourier Transform (α=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fractional_operators_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'laplacian_time': laplacian_time,
        'fourier_time': fourier_time,
        'laplacian_result': laplacian_result,
        'fourier_result': fourier_result
    }


def train_simple_fractional_pino(operator_type="laplacian", epochs=100):
    """Train a simple Fractional PINO"""
    print(f"\n🚀 Training Simple Fractional PINO ({operator_type})")
    print("=" * 50)
    
    # Generate data
    x, y = generate_test_data(100)
    
    # Initialize model
    model = SimpleFractionalPINO(hidden_dim=32, fractional_order=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x)
        
        # Reconstruction loss
        recon_loss = criterion(y_pred, y)
        
        # Physics loss (fractional operator constraint)
        frac_op_pred = model.apply_fractional_operator(y_pred, operator_type)
        physics_loss = criterion(frac_op_pred, torch.zeros_like(frac_op_pred))
        
        # Total loss
        total_loss = recon_loss + 0.1 * physics_loss
        losses.append(total_loss.item())
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.6f}")
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        final_loss = criterion(y_pred, y).item()
        print(f"Final reconstruction loss: {final_loss:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x.numpy(), y.numpy(), 'b-', linewidth=2, label='True')
    plt.plot(x.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Fractional PINO Results ({operator_type})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'fractional_pino_{operator_type}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'model': model,
        'training_time': training_time,
        'final_loss': final_loss,
        'losses': losses
    }


def benchmark_operators():
    """Quick benchmark of different operators"""
    print("\n⚡ Quick Operator Benchmark")
    print("=" * 30)
    
    operators = ["laplacian", "fourier"]
    results = {}
    
    for operator in operators:
        print(f"\nTesting {operator} operator...")
        result = train_simple_fractional_pino(operator, epochs=50)
        results[operator] = result
    
    # Compare results
    print("\n📊 Benchmark Results")
    print("-" * 20)
    for operator, result in results.items():
        print(f"{operator:15s}: Loss = {result['final_loss']:.6f}, Time = {result['training_time']:.2f}s")
    
    return results


def main():
    """Main test function"""
    print("🚀 Quick Fractional PINO Test with hpfracc")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test fractional operators
    operator_results = test_fractional_operators()
    
    # Benchmark different operators
    benchmark_results = benchmark_operators()
    
    # Summary
    print("\n🎉 Test Summary")
    print("=" * 20)
    print(f"✅ hpfracc operators tested successfully")
    print(f"✅ Fractional PINO trained with multiple operators")
    print(f"✅ Results saved as PNG files")
    
    best_operator = min(benchmark_results.keys(), 
                       key=lambda op: benchmark_results[op]['final_loss'])
    print(f"🏆 Best operator: {best_operator} (loss: {benchmark_results[best_operator]['final_loss']:.6f})")


if __name__ == "__main__":
    main()
