#!/usr/bin/env python3
"""
Fractional Physics Examples using HPFRACC
==========================================

This script demonstrates the application of HPFRACC to various fractional physics problems:
1. Anomalous Diffusion
2. Advection-Diffusion 
3. Fractional Wave Equation
4. Fractional Heat Equation

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import gamma
import time
import warnings
warnings.filterwarnings('ignore')

# Import HPFRACC components
try:
    from hpfracc.ml.spectral_autograd import SpectralFractionalDerivative, BoundedAlphaParameter
    from hpfracc.algorithms.gpu_optimized_methods import GPUOptimizedRiemannLiouville, GPUConfig
    print("✅ HPFRACC imported successfully")
except ImportError as e:
    print(f"❌ HPFRACC import failed: {e}")
    print("Please ensure HPFRACC is properly installed")
    exit(1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

class FractionalPhysicsExamples:
    """Comprehensive fractional physics examples using HPFRACC"""
    
    def __init__(self):
        self.device = device
        self.results = {}
        
    def anomalous_diffusion(self, alpha=0.5, D=1.0, L=1.0, T=1.0, nx=100, nt=100):
        """
        Solve the fractional diffusion equation:
        ∂^α u/∂t^α = D ∂²u/∂x²
        
        This models anomalous diffusion where the mean square displacement
        grows as ⟨x²(t)⟩ ~ t^α instead of linear growth.
        """
        print(f"\n🔬 Anomalous Diffusion (α={alpha})")
        print("=" * 50)
        
        # Create spatial and temporal grids
        x = torch.linspace(0, L, nx, device=self.device, requires_grad=True)
        t = torch.linspace(0, T, nt, device=self.device)
        
        # Initial condition: Gaussian pulse
        x0 = L/2
        sigma = 0.1
        u0 = torch.exp(-((x - x0)**2) / (2 * sigma**2))
        
        # Analytical solution for fractional diffusion
        def analytical_solution(x, t, alpha, D):
            """Analytical solution using Mittag-Leffler function approximation"""
            # For small times, use asymptotic expansion
            if t < 0.1:
                return u0 * torch.exp(-D * t**alpha * (x - x0)**2 / (2 * sigma**2))
            else:
                # Use Green's function approximation
                return u0 * torch.exp(-(x - x0)**2 / (4 * D * t**alpha))
        
        # Neural network for learning the solution
        class DiffusionNet(nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = BoundedAlphaParameter(alpha)
                self.net = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                )
                
            def forward(self, x, t):
                inputs = torch.stack([x, t], dim=-1)
                return self.net(inputs).squeeze()
        
        # Initialize network
        net = DiffusionNet(alpha).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # Training loop
        print("Training neural network...")
        start_time = time.time()
        
        for epoch in range(1000):
            optimizer.zero_grad()
            
            # Sample random points in space-time
            x_sample = torch.rand(nx, device=self.device) * L
            t_sample = torch.rand(nt, device=self.device) * T
            
            # Compute neural network solution
            u_nn = net(x_sample, t_sample)
            
            # Compute fractional time derivative using spectral autograd
            u_t_alpha = SpectralFractionalDerivative.apply(u_nn, net.alpha(), -1, "fft")
            
            # Compute spatial second derivative
            u_x = torch.autograd.grad(u_nn, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_nn), allow_unused=True)[0]
            if u_x is not None:
                u_xx = torch.autograd.grad(u_x, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_x), allow_unused=True)[0]
                if u_xx is None:
                    u_xx = torch.zeros_like(u_nn)
            else:
                u_xx = torch.zeros_like(u_nn)
            
            # Physics loss: ∂^α u/∂t^α = D ∂²u/∂x²
            physics_loss = torch.mean((u_t_alpha - D * u_xx)**2)
            
            # Initial condition loss
            u_ic = net(x, torch.zeros_like(x))
            ic_loss = torch.mean((u_ic - u0)**2)
            
            # Boundary condition loss (u=0 at x=0, L)
            u_bc1 = net(torch.zeros(1, device=self.device), torch.zeros(1, device=self.device))
            u_bc2 = net(torch.tensor([L], device=self.device), torch.zeros(1, device=self.device))
            bc_loss = torch.mean(u_bc1**2) + torch.mean(u_bc2**2)
            
            # Total loss
            total_loss = physics_loss + 10 * ic_loss + 10 * bc_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate solution
        with torch.no_grad():
            x_eval = torch.linspace(0, L, nx, device=self.device)
            t_eval = torch.linspace(0, T, nt, device=self.device)
            X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')
            
            u_nn_eval = net(X.flatten(), T_grid.flatten()).reshape(nx, nt)
            
            # Analytical solution
            u_analytical = torch.zeros_like(u_nn_eval)
            for i, t_val in enumerate(t_eval):
                u_analytical[:, i] = analytical_solution(x_eval, t_val, alpha, D)
        
        # Compute error
        error = torch.mean((u_nn_eval - u_analytical)**2).item()
        print(f"Mean squared error: {error:.6f}")
        
        # Store results
        self.results['anomalous_diffusion'] = {
            'alpha': alpha,
            'error': error,
            'training_time': training_time,
            'solution': u_nn_eval.cpu().numpy(),
            'analytical': u_analytical.cpu().numpy(),
            'x': x_eval.cpu().numpy(),
            't': t_eval.cpu().numpy()
        }
        
        return self.results['anomalous_diffusion']
    
    def advection_diffusion(self, alpha=0.7, beta=0.5, v=1.0, D=0.1, L=2.0, T=1.0, nx=100, nt=100):
        """
        Solve the fractional advection-diffusion equation:
        ∂^α u/∂t^α + v ∂^β u/∂x^β = D ∂²u/∂x²
        
        This models transport with memory effects in both time and space.
        """
        print(f"\n🌊 Advection-Diffusion (α={alpha}, β={beta})")
        print("=" * 50)
        
        # Create spatial and temporal grids
        x = torch.linspace(0, L, nx, device=self.device, requires_grad=True)
        t = torch.linspace(0, T, nt, device=self.device)
        
        # Initial condition: traveling wave
        k = 2 * np.pi / L
        u0 = torch.sin(k * x)
        
        # Neural network
        class AdvectionDiffusionNet(nn.Module):
            def __init__(self, alpha, beta):
                super().__init__()
                self.alpha = BoundedAlphaParameter(alpha)
                self.beta = BoundedAlphaParameter(beta)
                self.net = nn.Sequential(
                    nn.Linear(2, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x, t):
                inputs = torch.stack([x, t], dim=-1)
                return self.net(inputs).squeeze()
        
        # Initialize network
        net = AdvectionDiffusionNet(alpha, beta).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # Training loop
        print("Training neural network...")
        start_time = time.time()
        
        for epoch in range(1000):
            optimizer.zero_grad()
            
            # Sample random points
            x_sample = torch.rand(nx, device=self.device) * L
            t_sample = torch.rand(nt, device=self.device) * T
            
            # Compute solution
            u_nn = net(x_sample, t_sample)
            
            # Fractional time derivative
            u_t_alpha = SpectralFractionalDerivative.apply(u_nn, net.alpha(), -1, "fft")
            
            # Fractional spatial derivative
            u_x_beta = SpectralFractionalDerivative.apply(u_nn, net.beta(), 0, "fft")
            
            # Second spatial derivative
            u_xx = torch.autograd.grad(u_nn, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_nn))[0]
            u_xx = torch.autograd.grad(u_xx, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_xx))[0]
            
            # Physics loss: ∂^α u/∂t^α + v ∂^β u/∂x^β = D ∂²u/∂x²
            physics_loss = torch.mean((u_t_alpha + v * u_x_beta - D * u_xx)**2)
            
            # Initial condition loss
            u_ic = net(x, torch.zeros_like(x))
            ic_loss = torch.mean((u_ic - u0)**2)
            
            # Total loss
            total_loss = physics_loss + 10 * ic_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate solution
        with torch.no_grad():
            x_eval = torch.linspace(0, L, nx, device=self.device)
            t_eval = torch.linspace(0, T, nt, device=self.device)
            X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')
            
            u_nn_eval = net(X.flatten(), T_grid.flatten()).reshape(nx, nt)
        
        # Store results
        self.results['advection_diffusion'] = {
            'alpha': alpha,
            'beta': beta,
            'training_time': training_time,
            'solution': u_nn_eval.cpu().numpy(),
            'x': x_eval.cpu().numpy(),
            't': t_eval.cpu().numpy()
        }
        
        return self.results['advection_diffusion']
    
    def fractional_wave_equation(self, alpha=1.5, c=1.0, L=2.0, T=2.0, nx=100, nt=100):
        """
        Solve the fractional wave equation:
        ∂^α u/∂t^α = c² ∂²u/∂x²
        
        This models wave propagation with memory effects.
        """
        print(f"\n🌊 Fractional Wave Equation (α={alpha})")
        print("=" * 50)
        
        # Create grids
        x = torch.linspace(0, L, nx, device=self.device, requires_grad=True)
        t = torch.linspace(0, T, nt, device=self.device)
        
        # Initial condition: Gaussian pulse
        x0 = L/2
        sigma = 0.2
        u0 = torch.exp(-((x - x0)**2) / (2 * sigma**2))
        
        # Neural network
        class WaveNet(nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = BoundedAlphaParameter(alpha)
                self.net = nn.Sequential(
                    nn.Linear(2, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x, t):
                inputs = torch.stack([x, t], dim=-1)
                return self.net(inputs).squeeze()
        
        # Initialize network
        net = WaveNet(alpha).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # Training loop
        print("Training neural network...")
        start_time = time.time()
        
        for epoch in range(1000):
            optimizer.zero_grad()
            
            # Sample random points
            x_sample = torch.rand(nx, device=self.device) * L
            t_sample = torch.rand(nt, device=self.device) * T
            
            # Compute solution
            u_nn = net(x_sample, t_sample)
            
            # Fractional time derivative
            u_t_alpha = SpectralFractionalDerivative.apply(u_nn, net.alpha(), -1, "fft")
            
            # Second spatial derivative
            u_xx = torch.autograd.grad(u_nn, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_nn))[0]
            u_xx = torch.autograd.grad(u_xx, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_xx))[0]
            
            # Physics loss: ∂^α u/∂t^α = c² ∂²u/∂x²
            physics_loss = torch.mean((u_t_alpha - c**2 * u_xx)**2)
            
            # Initial condition loss
            u_ic = net(x, torch.zeros_like(x))
            ic_loss = torch.mean((u_ic - u0)**2)
            
            # Total loss
            total_loss = physics_loss + 10 * ic_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate solution
        with torch.no_grad():
            x_eval = torch.linspace(0, L, nx, device=self.device)
            t_eval = torch.linspace(0, T, nt, device=self.device)
            X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')
            
            u_nn_eval = net(X.flatten(), T_grid.flatten()).reshape(nx, nt)
        
        # Store results
        self.results['fractional_wave'] = {
            'alpha': alpha,
            'training_time': training_time,
            'solution': u_nn_eval.cpu().numpy(),
            'x': x_eval.cpu().numpy(),
            't': t_eval.cpu().numpy()
        }
        
        return self.results['fractional_wave']
    
    def fractional_heat_equation(self, alpha=0.8, kappa=1.0, L=1.0, T=1.0, nx=100, nt=100):
        """
        Solve the fractional heat equation:
        ∂^α u/∂t^α = κ ∂²u/∂x²
        
        This models heat conduction with memory effects.
        """
        print(f"\n🔥 Fractional Heat Equation (α={alpha})")
        print("=" * 50)
        
        # Create grids
        x = torch.linspace(0, L, nx, device=self.device, requires_grad=True)
        t = torch.linspace(0, T, nt, device=self.device)
        
        # Initial condition: step function
        u0 = torch.where(x < L/2, torch.ones_like(x), torch.zeros_like(x))
        
        # Neural network
        class HeatNet(nn.Module):
            def __init__(self, alpha):
                super().__init__()
                self.alpha = BoundedAlphaParameter(alpha)
                self.net = nn.Sequential(
                    nn.Linear(2, 128),
                    nn.Tanh(),
                    nn.Linear(128, 128),
                    nn.Tanh(),
                    nn.Linear(128, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, x, t):
                inputs = torch.stack([x, t], dim=-1)
                return self.net(inputs).squeeze()
        
        # Initialize network
        net = HeatNet(alpha).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # Training loop
        print("Training neural network...")
        start_time = time.time()
        
        for epoch in range(1000):
            optimizer.zero_grad()
            
            # Sample random points
            x_sample = torch.rand(nx, device=self.device) * L
            t_sample = torch.rand(nt, device=self.device) * T
            
            # Compute solution
            u_nn = net(x_sample, t_sample)
            
            # Fractional time derivative
            u_t_alpha = SpectralFractionalDerivative.apply(u_nn, net.alpha(), -1, "fft")
            
            # Second spatial derivative
            u_xx = torch.autograd.grad(u_nn, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_nn))[0]
            u_xx = torch.autograd.grad(u_xx, x_sample, create_graph=True, grad_outputs=torch.ones_like(u_xx))[0]
            
            # Physics loss: ∂^α u/∂t^α = κ ∂²u/∂x²
            physics_loss = torch.mean((u_t_alpha - kappa * u_xx)**2)
            
            # Initial condition loss
            u_ic = net(x, torch.zeros_like(x))
            ic_loss = torch.mean((u_ic - u0)**2)
            
            # Boundary conditions (u=0 at x=0, L)
            u_bc1 = net(torch.zeros(1, device=self.device), torch.zeros(1, device=self.device))
            u_bc2 = net(torch.tensor([L], device=self.device), torch.zeros(1, device=self.device))
            bc_loss = torch.mean(u_bc1**2) + torch.mean(u_bc2**2)
            
            # Total loss
            total_loss = physics_loss + 10 * ic_loss + 10 * bc_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {training_time:.2f} seconds")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate solution
        with torch.no_grad():
            x_eval = torch.linspace(0, L, nx, device=self.device)
            t_eval = torch.linspace(0, T, nt, device=self.device)
            X, T_grid = torch.meshgrid(x_eval, t_eval, indexing='ij')
            
            u_nn_eval = net(X.flatten(), T_grid.flatten()).reshape(nx, nt)
        
        # Store results
        self.results['fractional_heat'] = {
            'alpha': alpha,
            'training_time': training_time,
            'solution': u_nn_eval.cpu().numpy(),
            'x': x_eval.cpu().numpy(),
            't': t_eval.cpu().numpy()
        }
        
        return self.results['fractional_heat']
    
    def plot_results(self):
        """Plot all results"""
        print("\n📊 Plotting Results")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fractional Physics Examples using HPFRACC', fontsize=16)
        
        # Plot anomalous diffusion
        if 'anomalous_diffusion' in self.results:
            ax = axes[0, 0]
            result = self.results['anomalous_diffusion']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['solution'], levels=20, cmap='viridis')
            ax.set_title(f'Anomalous Diffusion (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot advection-diffusion
        if 'advection_diffusion' in self.results:
            ax = axes[0, 1]
            result = self.results['advection_diffusion']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['solution'], levels=20, cmap='plasma')
            ax.set_title(f'Advection-Diffusion (α={result["alpha"]}, β={result["beta"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot fractional wave
        if 'fractional_wave' in self.results:
            ax = axes[1, 0]
            result = self.results['fractional_wave']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['solution'], levels=20, cmap='coolwarm')
            ax.set_title(f'Fractional Wave (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot fractional heat
        if 'fractional_heat' in self.results:
            ax = axes[1, 1]
            result = self.results['fractional_heat']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['solution'], levels=20, cmap='hot')
            ax.set_title(f'Fractional Heat (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('fractional_physics_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Results plotted and saved as 'fractional_physics_examples.png'")
    
    def run_all_examples(self):
        """Run all fractional physics examples"""
        print("🚀 Running All Fractional Physics Examples")
        print("=" * 60)
        
        # Run all examples
        self.anomalous_diffusion()
        self.advection_diffusion()
        self.fractional_wave_equation()
        self.fractional_heat_equation()
        
        # Plot results
        self.plot_results()
        
        # Print summary
        print("\n📋 Summary of Results")
        print("=" * 50)
        for name, result in self.results.items():
            print(f"{name}: α={result['alpha']}, Training time: {result['training_time']:.2f}s")
        
        return self.results

def main():
    """Main function to run fractional physics examples"""
    print("🔬 Fractional Physics Examples using HPFRACC")
    print("=" * 60)
    print("Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>")
    print(f"Device: {device}")
    print()
    
    # Create examples instance
    examples = FractionalPhysicsExamples()
    
    # Run all examples
    results = examples.run_all_examples()
    
    print("\n✅ All fractional physics examples completed successfully!")
    print("Results saved and plotted.")

if __name__ == "__main__":
    main()
