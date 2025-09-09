#!/usr/bin/env python3
"""
Fractional Physics Demonstration using HPFRACC
==============================================

This script demonstrates fractional physics concepts using HPFRACC's spectral autograd framework:
1. Anomalous Diffusion
2. Fractional Wave Equation
3. Fractional Heat Equation

Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# Import HPFRACC components
try:
    from hpfracc.ml.spectral_autograd import SpectralFractionalDerivative, BoundedAlphaParameter
    print("✅ HPFRACC imported successfully")
except ImportError as e:
    print(f"❌ HPFRACC import failed: {e}")
    exit(1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

class FractionalPhysicsDemo:
    """Demonstration of fractional physics using HPFRACC"""
    
    def __init__(self):
        self.device = device
        self.results = {}
        
    def anomalous_diffusion_demo(self, alpha=0.5, nx=50, nt=50):
        """
        Demonstrate anomalous diffusion using spectral autograd.
        Shows how fractional derivatives capture memory effects.
        """
        print(f"\n🔬 Anomalous Diffusion Demo (α={alpha})")
        print("=" * 50)
        
        # Create simple 1D signal
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create a simple function: u(x,t) = exp(-x²) * sin(2πt)
        X, T = torch.meshgrid(x, t, indexing='ij')
        u = torch.exp(-X**2) * torch.sin(2 * np.pi * T)
        
        # Compute fractional time derivative using spectral autograd
        print("Computing fractional time derivative...")
        start_time = time.time()
        
        # Reshape for spectral autograd
        u_flat = u.flatten()
        u_t_alpha = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_alpha = u_t_alpha.reshape(nx, nt)
        
        compute_time = time.time() - start_time
        print(f"Fractional derivative computed in {compute_time:.4f} seconds")
        
        # Store results
        self.results['anomalous_diffusion'] = {
            'alpha': alpha,
            'u': u.cpu().numpy(),
            'u_t_alpha': u_t_alpha.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'compute_time': compute_time
        }
        
        return self.results['anomalous_diffusion']
    
    def fractional_wave_demo(self, alpha=1.5, nx=50, nt=50):
        """
        Demonstrate fractional wave equation.
        Shows wave propagation with memory effects.
        """
        print(f"\n🌊 Fractional Wave Demo (α={alpha})")
        print("=" * 50)
        
        # Create wave-like signal
        x = torch.linspace(0, 2*np.pi, nx, device=self.device)
        t = torch.linspace(0, 2*np.pi, nt, device=self.device)
        
        # Create wave: u(x,t) = sin(x) * cos(t)
        X, T = torch.meshgrid(x, t, indexing='ij')
        u = torch.sin(X) * torch.cos(T)
        
        # Compute fractional time derivative
        print("Computing fractional time derivative...")
        start_time = time.time()
        
        u_flat = u.flatten()
        u_t_alpha = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_alpha = u_t_alpha.reshape(nx, nt)
        
        compute_time = time.time() - start_time
        print(f"Fractional derivative computed in {compute_time:.4f} seconds")
        
        # Store results
        self.results['fractional_wave'] = {
            'alpha': alpha,
            'u': u.cpu().numpy(),
            'u_t_alpha': u_t_alpha.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'compute_time': compute_time
        }
        
        return self.results['fractional_wave']
    
    def fractional_heat_demo(self, alpha=0.8, nx=50, nt=50):
        """
        Demonstrate fractional heat equation.
        Shows heat conduction with memory effects.
        """
        print(f"\n🔥 Fractional Heat Demo (α={alpha})")
        print("=" * 50)
        
        # Create heat-like signal
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create heat profile: u(x,t) = exp(-x²/4t) / sqrt(4πt)
        X, T = torch.meshgrid(x, t, indexing='ij')
        # Avoid division by zero
        T_safe = torch.clamp(T, min=0.01)
        u = torch.exp(-X**2 / (4 * T_safe)) / torch.sqrt(4 * np.pi * T_safe)
        
        # Compute fractional time derivative
        print("Computing fractional time derivative...")
        start_time = time.time()
        
        u_flat = u.flatten()
        u_t_alpha = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_alpha = u_t_alpha.reshape(nx, nt)
        
        compute_time = time.time() - start_time
        print(f"Fractional derivative computed in {compute_time:.4f} seconds")
        
        # Store results
        self.results['fractional_heat'] = {
            'alpha': alpha,
            'u': u.cpu().numpy(),
            'u_t_alpha': u_t_alpha.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'compute_time': compute_time
        }
        
        return self.results['fractional_heat']
    
    def learnable_alpha_demo(self, nx=50, nt=50):
        """
        Demonstrate learnable fractional order using BoundedAlphaParameter.
        Shows how the framework can adapt fractional orders during training.
        """
        print(f"\n🎯 Learnable Alpha Demo")
        print("=" * 50)
        
        # Create simple signal
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create target signal
        X, T = torch.meshgrid(x, t, indexing='ij')
        u_target = torch.exp(-X**2) * torch.sin(2 * np.pi * T)
        
        # Create learnable alpha parameter
        alpha_param = BoundedAlphaParameter(0.5).to(self.device)
        optimizer = torch.optim.Adam(alpha_param.parameters(), lr=0.01)
        
        print("Training learnable alpha...")
        start_time = time.time()
        
        for epoch in range(100):
            optimizer.zero_grad()
            
            # Compute fractional derivative with current alpha
            u_flat = u_target.flatten()
            u_t_alpha = SpectralFractionalDerivative.apply(u_flat, alpha_param(), -1, "fft")
            
            # Simple loss: minimize the magnitude of the derivative
            loss = torch.mean(u_t_alpha**2)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: α={alpha_param().item():.4f}, Loss={loss.item():.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.4f} seconds")
        print(f"Final alpha: {alpha_param().item():.4f}")
        
        # Store results
        self.results['learnable_alpha'] = {
            'initial_alpha': 0.5,
            'final_alpha': alpha_param().item(),
            'training_time': training_time
        }
        
        return self.results['learnable_alpha']
    
    def plot_results(self):
        """Plot all demonstration results"""
        print("\n📊 Plotting Results")
        print("=" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fractional Physics Demonstrations using HPFRACC', fontsize=16)
        
        # Plot anomalous diffusion
        if 'anomalous_diffusion' in self.results:
            ax = axes[0, 0]
            result = self.results['anomalous_diffusion']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['u'], levels=20, cmap='viridis')
            ax.set_title(f'Anomalous Diffusion (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot fractional wave
        if 'fractional_wave' in self.results:
            ax = axes[0, 1]
            result = self.results['fractional_wave']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['u'], levels=20, cmap='coolwarm')
            ax.set_title(f'Fractional Wave (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot fractional heat
        if 'fractional_heat' in self.results:
            ax = axes[1, 0]
            result = self.results['fractional_heat']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            im = ax.contourf(T, X, result['u'], levels=20, cmap='hot')
            ax.set_title(f'Fractional Heat (α={result["alpha"]})')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            plt.colorbar(im, ax=ax)
        
        # Plot learnable alpha results
        if 'learnable_alpha' in self.results:
            ax = axes[1, 1]
            result = self.results['learnable_alpha']
            
            alphas = [result['initial_alpha'], result['final_alpha']]
            labels = ['Initial α', 'Final α']
            colors = ['red', 'blue']
            
            ax.bar(labels, alphas, color=colors, alpha=0.7)
            ax.set_title('Learnable Alpha Training')
            ax.set_ylabel('Alpha Value')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, (label, alpha) in enumerate(zip(labels, alphas)):
                ax.text(i, alpha + 0.01, f'{alpha:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('fractional_physics_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Results plotted and saved as 'fractional_physics_demo.png'")
    
    def run_all_demos(self):
        """Run all fractional physics demonstrations"""
        print("🚀 Running All Fractional Physics Demonstrations")
        print("=" * 60)
        
        # Run all demonstrations
        self.anomalous_diffusion_demo()
        self.fractional_wave_demo()
        self.fractional_heat_demo()
        self.learnable_alpha_demo()
        
        # Plot results
        self.plot_results()
        
        # Print summary
        print("\n📋 Summary of Results")
        print("=" * 50)
        for name, result in self.results.items():
            if 'compute_time' in result:
                print(f"{name}: α={result['alpha']}, Compute time: {result['compute_time']:.4f}s")
            elif 'final_alpha' in result:
                print(f"{name}: Initial α={result['initial_alpha']}, Final α={result['final_alpha']:.4f}")
        
        return self.results

def main():
    """Main function to run fractional physics demonstrations"""
    print("🔬 Fractional Physics Demonstrations using HPFRACC")
    print("=" * 60)
    print("Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>")
    print(f"Device: {device}")
    print()
    
    # Create demo instance
    demo = FractionalPhysicsDemo()
    
    # Run all demonstrations
    results = demo.run_all_demos()
    
    print("\n✅ All fractional physics demonstrations completed successfully!")
    print("Results saved and plotted.")

if __name__ == "__main__":
    main()
