#!/usr/bin/env python3
"""
Fractional vs Integer-Order Physics Comparison
==============================================

This script compares fractional physics models with their integer-order counterparts
to demonstrate the key differences and advantages of fractional calculus.

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

class FractionalVsIntegerComparison:
    """Comprehensive comparison between fractional and integer-order physics models"""
    
    def __init__(self):
        self.device = device
        self.results = {}
        
    def diffusion_comparison(self, alpha=0.5, nx=100, nt=100):
        """
        Compare fractional diffusion (α=0.5) vs integer diffusion (α=1.0)
        """
        print(f"\n🔬 Diffusion Comparison: Fractional (α={alpha}) vs Integer (α=1.0)")
        print("=" * 70)
        
        # Create spatial and temporal grids
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create initial condition: Gaussian pulse
        x0 = 0.5
        sigma = 0.1
        u0 = torch.exp(-((x - x0)**2) / (2 * sigma**2))
        
        # Create test function: u(x,t) = exp(-x²/4t) / sqrt(4πt)
        X, T = torch.meshgrid(x, t, indexing='ij')
        T_safe = torch.clamp(T, min=0.01)
        u_test = torch.exp(-X**2 / (4 * T_safe)) / torch.sqrt(4 * np.pi * T_safe)
        
        # Compute fractional time derivative (α=0.5)
        print("Computing fractional time derivative (α=0.5)...")
        start_time = time.time()
        u_flat = u_test.flatten()
        u_t_frac = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_frac = u_t_frac.reshape(nx, nt)
        frac_time = time.time() - start_time
        
        # Compute integer time derivative (α=1.0)
        print("Computing integer time derivative (α=1.0)...")
        start_time = time.time()
        u_t_int = SpectralFractionalDerivative.apply(u_flat, 1.0, -1, "fft")
        u_t_int = u_t_int.reshape(nx, nt)
        int_time = time.time() - start_time
        
        # Compute analytical integer derivative for comparison
        u_t_analytical = -u_test / (2 * T_safe) + X**2 * u_test / (4 * T_safe**2)
        
        # Store results
        self.results['diffusion'] = {
            'alpha_frac': alpha,
            'alpha_int': 1.0,
            'u_test': u_test.cpu().numpy(),
            'u_t_frac': u_t_frac.cpu().numpy(),
            'u_t_int': u_t_int.cpu().numpy(),
            'u_t_analytical': u_t_analytical.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'frac_time': frac_time,
            'int_time': int_time
        }
        
        print(f"Fractional derivative time: {frac_time:.4f}s")
        print(f"Integer derivative time: {int_time:.4f}s")
        
        return self.results['diffusion']
    
    def wave_comparison(self, alpha=1.5, nx=100, nt=100):
        """
        Compare fractional wave (α=1.5) vs integer wave (α=2.0)
        """
        print(f"\n🌊 Wave Comparison: Fractional (α={alpha}) vs Integer (α=2.0)")
        print("=" * 70)
        
        # Create grids
        x = torch.linspace(0, 2*np.pi, nx, device=self.device)
        t = torch.linspace(0, 2*np.pi, nt, device=self.device)
        
        # Create wave function: u(x,t) = sin(x) * cos(t)
        X, T = torch.meshgrid(x, t, indexing='ij')
        u_wave = torch.sin(X) * torch.cos(T)
        
        # Compute fractional time derivative (α=1.5)
        print("Computing fractional time derivative (α=1.5)...")
        start_time = time.time()
        u_flat = u_wave.flatten()
        u_t_frac = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_frac = u_t_frac.reshape(nx, nt)
        frac_time = time.time() - start_time
        
        # Compute integer time derivative (α=2.0)
        print("Computing integer time derivative (α=2.0)...")
        start_time = time.time()
        u_t_int = SpectralFractionalDerivative.apply(u_flat, 2.0, -1, "fft")
        u_t_int = u_t_int.reshape(nx, nt)
        int_time = time.time() - start_time
        
        # Analytical integer derivative: ∂²u/∂t² = -sin(x) * cos(t)
        u_t_analytical = -torch.sin(X) * torch.cos(T)
        
        # Store results
        self.results['wave'] = {
            'alpha_frac': alpha,
            'alpha_int': 2.0,
            'u_wave': u_wave.cpu().numpy(),
            'u_t_frac': u_t_frac.cpu().numpy(),
            'u_t_int': u_t_int.cpu().numpy(),
            'u_t_analytical': u_t_analytical.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'frac_time': frac_time,
            'int_time': int_time
        }
        
        print(f"Fractional derivative time: {frac_time:.4f}s")
        print(f"Integer derivative time: {int_time:.4f}s")
        
        return self.results['wave']
    
    def heat_comparison(self, alpha=0.8, nx=100, nt=100):
        """
        Compare fractional heat (α=0.8) vs integer heat (α=1.0)
        """
        print(f"\n🔥 Heat Comparison: Fractional (α={alpha}) vs Integer (α=1.0)")
        print("=" * 70)
        
        # Create grids
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create heat function: u(x,t) = exp(-x²/4t) / sqrt(4πt)
        X, T = torch.meshgrid(x, t, indexing='ij')
        T_safe = torch.clamp(T, min=0.01)
        u_heat = torch.exp(-X**2 / (4 * T_safe)) / torch.sqrt(4 * np.pi * T_safe)
        
        # Compute fractional time derivative (α=0.8)
        print("Computing fractional time derivative (α=0.8)...")
        start_time = time.time()
        u_flat = u_heat.flatten()
        u_t_frac = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
        u_t_frac = u_t_frac.reshape(nx, nt)
        frac_time = time.time() - start_time
        
        # Compute integer time derivative (α=1.0)
        print("Computing integer time derivative (α=1.0)...")
        start_time = time.time()
        u_t_int = SpectralFractionalDerivative.apply(u_flat, 1.0, -1, "fft")
        u_t_int = u_t_int.reshape(nx, nt)
        int_time = time.time() - start_time
        
        # Analytical integer derivative
        u_t_analytical = -u_heat / (2 * T_safe) + X**2 * u_heat / (4 * T_safe**2)
        
        # Store results
        self.results['heat'] = {
            'alpha_frac': alpha,
            'alpha_int': 1.0,
            'u_heat': u_heat.cpu().numpy(),
            'u_t_frac': u_t_frac.cpu().numpy(),
            'u_t_int': u_t_int.cpu().numpy(),
            'u_t_analytical': u_t_analytical.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy(),
            'frac_time': frac_time,
            'int_time': int_time
        }
        
        print(f"Fractional derivative time: {frac_time:.4f}s")
        print(f"Integer derivative time: {int_time:.4f}s")
        
        return self.results['heat']
    
    def memory_effects_analysis(self, nx=100, nt=100):
        """
        Analyze memory effects by comparing different fractional orders
        """
        print(f"\n🧠 Memory Effects Analysis")
        print("=" * 70)
        
        # Create simple signal
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        
        # Create test function
        X, T = torch.meshgrid(x, t, indexing='ij')
        u_test = torch.exp(-X**2) * torch.sin(2 * np.pi * T)
        
        # Test different fractional orders
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0]
        derivatives = {}
        compute_times = {}
        
        print("Computing derivatives for different fractional orders...")
        for alpha in alphas:
            start_time = time.time()
            u_flat = u_test.flatten()
            u_t_alpha = SpectralFractionalDerivative.apply(u_flat, alpha, -1, "fft")
            u_t_alpha = u_t_alpha.reshape(nx, nt)
            compute_time = time.time() - start_time
            
            derivatives[alpha] = u_t_alpha.cpu().numpy()
            compute_times[alpha] = compute_time
            
            print(f"α={alpha}: {compute_time:.4f}s")
        
        # Store results
        self.results['memory_effects'] = {
            'alphas': alphas,
            'derivatives': derivatives,
            'compute_times': compute_times,
            'u_test': u_test.cpu().numpy(),
            'x': x.cpu().numpy(),
            't': t.cpu().numpy()
        }
        
        return self.results['memory_effects']
    
    def performance_comparison(self, nx=100, nt=100):
        """
        Compare computational performance between fractional and integer derivatives
        """
        print(f"\n⚡ Performance Comparison")
        print("=" * 70)
        
        # Create test signal
        x = torch.linspace(0, 1, nx, device=self.device)
        t = torch.linspace(0, 1, nt, device=self.device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        u_test = torch.exp(-X**2) * torch.sin(2 * np.pi * T)
        
        # Test different problem sizes
        sizes = [50, 100, 200, 500, 1000]
        frac_times = []
        int_times = []
        
        print("Testing performance across different problem sizes...")
        for size in sizes:
            # Create signal of given size
            x_small = torch.linspace(0, 1, size, device=self.device)
            t_small = torch.linspace(0, 1, size, device=self.device)
            X_small, T_small = torch.meshgrid(x_small, t_small, indexing='ij')
            u_small = torch.exp(-X_small**2) * torch.sin(2 * np.pi * T_small)
            
            # Fractional derivative (α=0.5)
            start_time = time.time()
            u_flat = u_small.flatten()
            _ = SpectralFractionalDerivative.apply(u_flat, 0.5, -1, "fft")
            frac_time = time.time() - start_time
            
            # Integer derivative (α=1.0)
            start_time = time.time()
            _ = SpectralFractionalDerivative.apply(u_flat, 1.0, -1, "fft")
            int_time = time.time() - start_time
            
            frac_times.append(frac_time)
            int_times.append(int_time)
            
            print(f"Size {size}: Fractional={frac_time:.4f}s, Integer={int_time:.4f}s")
        
        # Store results
        self.results['performance'] = {
            'sizes': sizes,
            'frac_times': frac_times,
            'int_times': int_times
        }
        
        return self.results['performance']
    
    def plot_comparison_results(self):
        """Plot comprehensive comparison results"""
        print("\n📊 Plotting Comparison Results")
        print("=" * 70)
        
        fig = plt.figure(figsize=(20, 16))
        
        # Diffusion comparison
        if 'diffusion' in self.results:
            result = self.results['diffusion']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            # Original function
            ax1 = plt.subplot(3, 4, 1)
            im1 = ax1.contourf(T, X, result['u_test'], levels=20, cmap='viridis')
            ax1.set_title('Original Function')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Position')
            plt.colorbar(im1, ax=ax1)
            
            # Fractional derivative
            ax2 = plt.subplot(3, 4, 2)
            im2 = ax2.contourf(T, X, result['u_t_frac'], levels=20, cmap='plasma')
            ax2.set_title(f'Fractional Derivative (α={result["alpha_frac"]})')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Position')
            plt.colorbar(im2, ax=ax2)
            
            # Integer derivative
            ax3 = plt.subplot(3, 4, 3)
            im3 = ax3.contourf(T, X, result['u_t_int'], levels=20, cmap='plasma')
            ax3.set_title(f'Integer Derivative (α={result["alpha_int"]})')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Position')
            plt.colorbar(im3, ax=ax3)
            
            # Difference
            ax4 = plt.subplot(3, 4, 4)
            diff = result['u_t_frac'] - result['u_t_int']
            im4 = ax4.contourf(T, X, diff, levels=20, cmap='RdBu_r')
            ax4.set_title('Difference (Fractional - Integer)')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Position')
            plt.colorbar(im4, ax=ax4)
        
        # Wave comparison
        if 'wave' in self.results:
            result = self.results['wave']
            X, T = np.meshgrid(result['x'], result['t'], indexing='ij')
            
            # Original function
            ax5 = plt.subplot(3, 4, 5)
            im5 = ax5.contourf(T, X, result['u_wave'], levels=20, cmap='coolwarm')
            ax5.set_title('Wave Function')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Position')
            plt.colorbar(im5, ax=ax5)
            
            # Fractional derivative
            ax6 = plt.subplot(3, 4, 6)
            im6 = ax6.contourf(T, X, result['u_t_frac'], levels=20, cmap='plasma')
            ax6.set_title(f'Fractional Wave (α={result["alpha_frac"]})')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Position')
            plt.colorbar(im6, ax=ax6)
            
            # Integer derivative
            ax7 = plt.subplot(3, 4, 7)
            im7 = ax7.contourf(T, X, result['u_t_int'], levels=20, cmap='plasma')
            ax7.set_title(f'Integer Wave (α={result["alpha_int"]})')
            ax7.set_xlabel('Time')
            ax7.set_ylabel('Position')
            plt.colorbar(im7, ax=ax7)
            
            # Difference
            ax8 = plt.subplot(3, 4, 8)
            diff = result['u_t_frac'] - result['u_t_int']
            im8 = ax8.contourf(T, X, diff, levels=20, cmap='RdBu_r')
            ax8.set_title('Difference (Fractional - Integer)')
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Position')
            plt.colorbar(im8, ax=ax8)
        
        # Memory effects analysis
        if 'memory_effects' in self.results:
            result = self.results['memory_effects']
            
            # Plot derivatives for different alphas
            ax9 = plt.subplot(3, 4, 9)
            for i, alpha in enumerate([0.3, 0.5, 0.7, 1.0, 1.5]):
                if alpha in result['derivatives']:
                    # Take a slice at x=0.5
                    x_idx = len(result['x']) // 2
                    ax9.plot(result['t'], result['derivatives'][alpha][x_idx, :], 
                            label=f'α={alpha}', linewidth=2)
            ax9.set_title('Memory Effects: Different α Values')
            ax9.set_xlabel('Time')
            ax9.set_ylabel('Derivative Value')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
            
            # Performance vs alpha
            ax10 = plt.subplot(3, 4, 10)
            alphas = result['alphas']
            times = [result['compute_times'][alpha] for alpha in alphas]
            ax10.plot(alphas, times, 'o-', linewidth=2, markersize=6)
            ax10.set_title('Compute Time vs Fractional Order')
            ax10.set_xlabel('Fractional Order α')
            ax10.set_ylabel('Compute Time (s)')
            ax10.grid(True, alpha=0.3)
        
        # Performance comparison
        if 'performance' in self.results:
            result = self.results['performance']
            
            # Performance vs problem size
            ax11 = plt.subplot(3, 4, 11)
            ax11.loglog(result['sizes'], result['frac_times'], 'o-', 
                       label='Fractional (α=0.5)', linewidth=2, markersize=6)
            ax11.loglog(result['sizes'], result['int_times'], 's-', 
                       label='Integer (α=1.0)', linewidth=2, markersize=6)
            ax11.set_title('Performance vs Problem Size')
            ax11.set_xlabel('Problem Size')
            ax11.set_ylabel('Compute Time (s)')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
            
            # Speedup ratio
            ax12 = plt.subplot(3, 4, 12)
            speedup = [int_time / frac_time for int_time, frac_time in 
                      zip(result['int_times'], result['frac_times'])]
            ax12.semilogx(result['sizes'], speedup, 'o-', linewidth=2, markersize=6)
            ax12.set_title('Speedup Ratio (Integer/Fractional)')
            ax12.set_xlabel('Problem Size')
            ax12.set_ylabel('Speedup Ratio')
            ax12.grid(True, alpha=0.3)
            ax12.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('fractional_vs_integer_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison results plotted and saved as 'fractional_vs_integer_comparison.png'")
    
    def run_all_comparisons(self):
        """Run all fractional vs integer comparisons"""
        print("🚀 Running All Fractional vs Integer Comparisons")
        print("=" * 70)
        
        # Run all comparisons
        self.diffusion_comparison()
        self.wave_comparison()
        self.heat_comparison()
        self.memory_effects_analysis()
        self.performance_comparison()
        
        # Plot results
        self.plot_comparison_results()
        
        # Print summary
        print("\n📋 Summary of Comparisons")
        print("=" * 70)
        
        if 'diffusion' in self.results:
            result = self.results['diffusion']
            print(f"Diffusion: Fractional (α={result['alpha_frac']}) vs Integer (α={result['alpha_int']})")
            print(f"  Fractional time: {result['frac_time']:.4f}s")
            print(f"  Integer time: {result['int_time']:.4f}s")
            print(f"  Speedup: {result['int_time']/result['frac_time']:.2f}x")
        
        if 'wave' in self.results:
            result = self.results['wave']
            print(f"Wave: Fractional (α={result['alpha_frac']}) vs Integer (α={result['alpha_int']})")
            print(f"  Fractional time: {result['frac_time']:.4f}s")
            print(f"  Integer time: {result['int_time']:.4f}s")
            print(f"  Speedup: {result['int_time']/result['frac_time']:.2f}x")
        
        if 'heat' in self.results:
            result = self.results['heat']
            print(f"Heat: Fractional (α={result['alpha_frac']}) vs Integer (α={result['alpha_int']})")
            print(f"  Fractional time: {result['frac_time']:.4f}s")
            print(f"  Integer time: {result['int_time']:.4f}s")
            print(f"  Speedup: {result['int_time']/result['frac_time']:.2f}x")
        
        if 'performance' in self.results:
            result = self.results['performance']
            avg_speedup = np.mean([int_time / frac_time for int_time, frac_time in 
                                 zip(result['int_times'], result['frac_times'])])
            print(f"Average speedup across all problem sizes: {avg_speedup:.2f}x")
        
        return self.results

def main():
    """Main function to run fractional vs integer comparisons"""
    print("🔬 Fractional vs Integer-Order Physics Comparison")
    print("=" * 70)
    print("Author: Davian R. Chin <d.r.chin@pgr.reading.ac.uk>")
    print(f"Device: {device}")
    print()
    
    # Create comparison instance
    comparison = FractionalVsIntegerComparison()
    
    # Run all comparisons
    results = comparison.run_all_comparisons()
    
    print("\n✅ All fractional vs integer comparisons completed successfully!")
    print("Results saved and plotted.")

if __name__ == "__main__":
    main()
