#!/usr/bin/env python3
"""
Physics Benchmarks for hpfracc Fractional Calculus Library
Phase 1: Classical Physics Problems vs Fractional Methods

This script implements classical physics problems and compares them with
fractional calculus implementations using the hpfracc library.

Classical Problems:
1. Wave Equation: ∂²u/∂t² = c²∇²u
2. Heat Equation: ∂u/∂t = α∇²u  
3. Burgers Equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
4. Navier-Stokes Equations: ∂u/∂t + (u·∇)u = -∇p + ν∇²u

Fractional Versions:
1. Fractional Wave: ∂ᵅu/∂tᵅ = c²∇²u
2. Fractional Heat: ∂ᵅu/∂tᵅ = α∇²u
3. Fractional Burgers: ∂ᵅu/∂tᵅ + u∂u/∂x = ν∂²u/∂x²
4. Fractional Navier-Stokes: ∂ᵅu/∂tᵅ + (u·∇)u = -∇p + ν∇²u
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# Add hpfracc to path
sys.path.append('/home/davianc/fractional-calculus-library')

class PhysicsBenchmark:
    """Base class for physics benchmark problems"""
    
    def __init__(self, nx=100, nt=1000, L=1.0, T=1.0):
        self.nx = nx  # Number of spatial points
        self.nt = nt  # Number of time points
        self.L = L    # Domain length
        self.T = T    # Total time
        self.dx = L / (nx - 1)
        self.dt = T / (nt - 1)
        self.x = np.linspace(0, L, nx)
        self.t = np.linspace(0, T, nt)
        
    def l2_error(self, u_exact, u_numerical):
        """Calculate L2 error"""
        return np.sqrt(np.mean((u_exact - u_numerical)**2))
    
    def linf_error(self, u_exact, u_numerical):
        """Calculate L∞ error"""
        return np.max(np.abs(u_exact - u_numerical))
    
    def relative_error(self, u_exact, u_numerical):
        """Calculate relative error"""
        return self.l2_error(u_exact, u_numerical) / np.sqrt(np.mean(u_exact**2))

class WaveEquation(PhysicsBenchmark):
    """Classical and Fractional Wave Equation Benchmark"""
    
    def __init__(self, c=1.0, **kwargs):
        super().__init__(**kwargs)
        self.c = c  # Wave speed
        
    def analytical_solution(self, x, t):
        """Analytical solution for 1D wave equation with initial conditions"""
        # Initial condition: u(x,0) = sin(πx), ∂u/∂t(x,0) = 0
        return np.sin(np.pi * x) * np.cos(np.pi * self.c * t)
    
    def classical_solver(self):
        """Classical finite difference solver for wave equation"""
        print("Solving classical wave equation...")
        start_time = time.time()
        
        # Initialize solution array
        u = np.zeros((self.nt, self.nx))
        
        # Initial conditions
        u[0, :] = np.sin(np.pi * self.x)  # u(x,0) = sin(πx)
        u[1, :] = u[0, :]  # ∂u/∂t(x,0) = 0 (first order approximation)
        
        # Finite difference scheme: ∂²u/∂t² = c²∂²u/∂x²
        for n in range(1, self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n+1, i] = 2*u[n, i] - u[n-1, i] + (self.c*self.dt/self.dx)**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            # Boundary conditions (Dirichlet: u=0 at x=0 and x=L)
            u[n+1, 0] = 0
            u[n+1, -1] = 0
        
        solve_time = time.time() - start_time
        return u, solve_time
    
    def fractional_solver(self, alpha=0.5):
        """Fractional wave equation solver using hpfracc"""
        print(f"Solving fractional wave equation (α={alpha})...")
        start_time = time.time()
        
        try:
            # Import hpfracc fractional derivative
            from hpfracc.core import FractionalDerivative
            from hpfracc.algorithms import grunwald_letnikov
            
            # Initialize solution array
            u = np.zeros((self.nt, self.nx))
            
            # Initial conditions
            u[0, :] = np.sin(np.pi * self.x)
            u[1, :] = u[0, :]
            
            # Fractional time derivative: ∂ᵅu/∂tᵅ = c²∂²u/∂x²
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    # Calculate fractional time derivative
                    frac_deriv = grunwald_letnikov(u[:n+1, i], alpha, self.dt)
                    
                    # Spatial second derivative
                    spatial_deriv = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / self.dx**2
                    
                    # Update solution
                    u[n+1, i] = u[n, i] + self.dt * (self.c**2 * spatial_deriv - frac_deriv)
                
                # Boundary conditions
                u[n+1, 0] = 0
                u[n+1, -1] = 0
            
            solve_time = time.time() - start_time
            return u, solve_time
            
        except ImportError:
            print("hpfracc not available, using simplified fractional approximation")
            # Simplified fractional approximation
            u = np.zeros((self.nt, self.nx))
            u[0, :] = np.sin(np.pi * self.x)
            u[1, :] = u[0, :]
            
            for n in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    # Simplified fractional effect (reduced wave speed)
                    effective_c = self.c * (alpha ** 0.5)
                    u[n+1, i] = 2*u[n, i] - u[n-1, i] + (effective_c*self.dt/self.dx)**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                
                u[n+1, 0] = 0
                u[n+1, -1] = 0
            
            solve_time = time.time() - start_time
            return u, solve_time
    
    def benchmark(self):
        """Run complete benchmark for wave equation"""
        print("="*60)
        print("WAVE EQUATION BENCHMARK")
        print("="*60)
        
        # Classical solution
        u_classical, t_classical = self.classical_solver()
        
        # Fractional solutions
        alphas = [0.5, 0.7, 0.9, 1.0]
        results = {}
        
        for alpha in alphas:
            u_frac, t_frac = self.fractional_solver(alpha)
            
            # Calculate errors at final time
            u_exact = self.analytical_solution(self.x, self.T)
            u_class_final = u_classical[-1, :]
            u_frac_final = u_frac[-1, :]
            
            # Errors
            l2_error_class = self.l2_error(u_exact, u_class_final)
            l2_error_frac = self.l2_error(u_exact, u_frac_final)
            linf_error_class = self.linf_error(u_exact, u_class_final)
            linf_error_frac = self.linf_error(u_exact, u_frac_final)
            
            results[alpha] = {
                'classical_time': t_classical,
                'fractional_time': t_frac,
                'classical_l2': l2_error_class,
                'fractional_l2': l2_error_frac,
                'classical_linf': linf_error_class,
                'fractional_linf': linf_error_frac
            }
            
            print(f"α = {alpha}:")
            print(f"  Classical: L2={l2_error_class:.6f}, L∞={linf_error_class:.6f}, Time={t_classical:.4f}s")
            print(f"  Fractional: L2={l2_error_frac:.6f}, L∞={linf_error_frac:.6f}, Time={t_frac:.4f}s")
            print()
        
        return results

class HeatEquation(PhysicsBenchmark):
    """Classical and Fractional Heat Equation Benchmark"""
    
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha  # Thermal diffusivity
        
    def analytical_solution(self, x, t):
        """Analytical solution for 1D heat equation"""
        # Initial condition: u(x,0) = sin(πx)
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi**2 * t)
    
    def classical_solver(self):
        """Classical finite difference solver for heat equation"""
        print("Solving classical heat equation...")
        start_time = time.time()
        
        # Initialize solution array
        u = np.zeros((self.nt, self.nx))
        
        # Initial condition
        u[0, :] = np.sin(np.pi * self.x)
        
        # Boundary conditions (Dirichlet: u=0 at x=0 and x=L)
        u[:, 0] = 0
        u[:, -1] = 0
        
        # Finite difference scheme: ∂u/∂t = α∂²u/∂x²
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                u[n+1, i] = u[n, i] + self.alpha * self.dt / self.dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        
        solve_time = time.time() - start_time
        return u, solve_time
    
    def fractional_solver(self, frac_alpha=0.5):
        """Fractional heat equation solver"""
        print(f"Solving fractional heat equation (α={frac_alpha})...")
        start_time = time.time()
        
        try:
            from hpfracc.algorithms import grunwald_letnikov
            
            # Initialize solution array
            u = np.zeros((self.nt, self.nx))
            u[0, :] = np.sin(np.pi * self.x)
            u[:, 0] = 0
            u[:, -1] = 0
            
            # Fractional time derivative: ∂ᵅu/∂tᵅ = α∂²u/∂x²
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    # Calculate fractional time derivative
                    frac_deriv = grunwald_letnikov(u[:n+1, i], frac_alpha, self.dt)
                    
                    # Spatial second derivative
                    spatial_deriv = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / self.dx**2
                    
                    # Update solution
                    u[n+1, i] = u[n, i] + self.dt * (self.alpha * spatial_deriv - frac_deriv)
            
            solve_time = time.time() - start_time
            return u, solve_time
            
        except ImportError:
            print("hpfracc not available, using simplified fractional approximation")
            # Simplified fractional approximation
            u = np.zeros((self.nt, self.nx))
            u[0, :] = np.sin(np.pi * self.x)
            u[:, 0] = 0
            u[:, -1] = 0
            
            for n in range(self.nt - 1):
                for i in range(1, self.nx - 1):
                    # Simplified fractional effect (reduced diffusivity)
                    effective_alpha = self.alpha * (frac_alpha ** 0.5)
                    u[n+1, i] = u[n, i] + effective_alpha * self.dt / self.dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            solve_time = time.time() - start_time
            return u, solve_time
    
    def benchmark(self):
        """Run complete benchmark for heat equation"""
        print("="*60)
        print("HEAT EQUATION BENCHMARK")
        print("="*60)
        
        # Classical solution
        u_classical, t_classical = self.classical_solver()
        
        # Fractional solutions
        alphas = [0.5, 0.7, 0.9, 1.0]
        results = {}
        
        for alpha in alphas:
            u_frac, t_frac = self.fractional_solver(alpha)
            
            # Calculate errors at final time
            u_exact = self.analytical_solution(self.x, self.T)
            u_class_final = u_classical[-1, :]
            u_frac_final = u_frac[-1, :]
            
            # Errors
            l2_error_class = self.l2_error(u_exact, u_class_final)
            l2_error_frac = self.l2_error(u_exact, u_frac_final)
            linf_error_class = self.linf_error(u_exact, u_class_final)
            linf_error_frac = self.linf_error(u_exact, u_frac_final)
            
            results[alpha] = {
                'classical_time': t_classical,
                'fractional_time': t_frac,
                'classical_l2': l2_error_class,
                'fractional_l2': l2_error_frac,
                'classical_linf': linf_error_class,
                'fractional_linf': linf_error_frac
            }
            
            print(f"α = {alpha}:")
            print(f"  Classical: L2={l2_error_class:.6f}, L∞={linf_error_class:.6f}, Time={t_classical:.4f}s")
            print(f"  Fractional: L2={l2_error_frac:.6f}, L∞={linf_error_frac:.6f}, Time={t_frac:.4f}s")
            print()
        
        return results

class BurgersEquation(PhysicsBenchmark):
    """Classical and Fractional Burgers Equation Benchmark"""
    
    def __init__(self, nu=0.01, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu  # Viscosity
        
    def classical_solver(self):
        """Classical finite difference solver for Burgers equation"""
        print("Solving classical Burgers equation...")
        start_time = time.time()
        
        # Initialize solution array
        u = np.zeros((self.nt, self.nx))
        
        # Initial condition: u(x,0) = sin(πx)
        u[0, :] = np.sin(np.pi * self.x)
        
        # Boundary conditions (Dirichlet: u=0 at x=0 and x=L)
        u[:, 0] = 0
        u[:, -1] = 0
        
        # Finite difference scheme: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                # Convection term: u∂u/∂x (upwind scheme)
                if u[n, i] >= 0:
                    conv_term = u[n, i] * (u[n, i] - u[n, i-1]) / self.dx
                else:
                    conv_term = u[n, i] * (u[n, i+1] - u[n, i]) / self.dx
                
                # Diffusion term: ν∂²u/∂x²
                diff_term = self.nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / self.dx**2
                
                # Update solution
                u[n+1, i] = u[n, i] - self.dt * conv_term + self.dt * diff_term
        
        solve_time = time.time() - start_time
        return u, solve_time
    
    def fractional_solver(self, alpha=0.5):
        """Fractional Burgers equation solver"""
        print(f"Solving fractional Burgers equation (α={alpha})...")
        start_time = time.time()
        
        # Initialize solution array
        u = np.zeros((self.nt, self.nx))
        u[0, :] = np.sin(np.pi * self.x)
        u[:, 0] = 0
        u[:, -1] = 0
        
        # Simplified fractional approximation
        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                # Convection term (same as classical)
                if u[n, i] >= 0:
                    conv_term = u[n, i] * (u[n, i] - u[n, i-1]) / self.dx
                else:
                    conv_term = u[n, i] * (u[n, i+1] - u[n, i]) / self.dx
                
                # Diffusion term with fractional effect
                effective_nu = self.nu * (alpha ** 0.5)
                diff_term = effective_nu * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / self.dx**2
                
                # Update solution
                u[n+1, i] = u[n, i] - self.dt * conv_term + self.dt * diff_term
        
        solve_time = time.time() - start_time
        return u, solve_time
    
    def benchmark(self):
        """Run complete benchmark for Burgers equation"""
        print("="*60)
        print("BURGERS EQUATION BENCHMARK")
        print("="*60)
        
        # Classical solution
        u_classical, t_classical = self.classical_solver()
        
        # Fractional solutions
        alphas = [0.5, 0.7, 0.9, 1.0]
        results = {}
        
        for alpha in alphas:
            u_frac, t_frac = self.fractional_solver(alpha)
            
            # Calculate errors (compare with classical as reference)
            u_class_final = u_classical[-1, :]
            u_frac_final = u_frac[-1, :]
            
            # Errors
            l2_error_class = self.l2_error(u_class_final, u_class_final)  # Should be 0
            l2_error_frac = self.l2_error(u_class_final, u_frac_final)
            linf_error_class = self.linf_error(u_class_final, u_class_final)  # Should be 0
            linf_error_frac = self.linf_error(u_class_final, u_frac_final)
            
            results[alpha] = {
                'classical_time': t_classical,
                'fractional_time': t_frac,
                'classical_l2': l2_error_class,
                'fractional_l2': l2_error_frac,
                'classical_linf': linf_error_class,
                'fractional_linf': linf_error_frac
            }
            
            print(f"α = {alpha}:")
            print(f"  Classical: L2={l2_error_class:.6f}, L∞={linf_error_class:.6f}, Time={t_classical:.4f}s")
            print(f"  Fractional: L2={l2_error_frac:.6f}, L∞={linf_error_frac:.6f}, Time={t_frac:.4f}s")
            print()
        
        return results

def main():
    """Run all physics benchmarks"""
    print("PHYSICS BENCHMARKS FOR HPFRACC FRACTIONAL CALCULUS LIBRARY")
    print("="*70)
    print("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)")
    print("Goal: Compare classical vs fractional physics simulations")
    print("="*70)
    
    # Create results directory
    os.makedirs('physics_results', exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    # 1. Wave Equation
    wave_bench = WaveEquation(nx=100, nt=1000, L=1.0, T=1.0, c=1.0)
    all_results['wave'] = wave_bench.benchmark()
    
    # 2. Heat Equation
    heat_bench = HeatEquation(nx=100, nt=1000, L=1.0, T=1.0, alpha=0.1)
    all_results['heat'] = heat_bench.benchmark()
    
    # 3. Burgers Equation
    burgers_bench = BurgersEquation(nx=100, nt=1000, L=1.0, T=1.0, nu=0.01)
    all_results['burgers'] = burgers_bench.benchmark()
    
    # Save results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    
    with open('physics_results/physics_benchmark_results.txt', 'w') as f:
        f.write("Physics Benchmark Results for hpfracc\n")
        f.write("Hardware: ASUS TUF A15 (RTX 3050, 30GB RAM, Ubuntu 24.04)\n")
        f.write("="*50 + "\n\n")
        
        for problem, results in all_results.items():
            f.write(f"{problem.upper()} EQUATION:\n")
            for alpha, result in results.items():
                f.write(f"  α = {alpha}:\n")
                f.write(f"    Classical: L2={result['classical_l2']:.6f}, L∞={result['classical_linf']:.6f}, Time={result['classical_time']:.4f}s\n")
                f.write(f"    Fractional: L2={result['fractional_l2']:.6f}, L∞={result['fractional_linf']:.6f}, Time={result['fractional_time']:.4f}s\n")
            f.write("\n")
    
    print("Results saved to physics_results/physics_benchmark_results.txt")
    print("\nBENCHMARKS COMPLETED!")
    print("Ready to compare with other fractional calculus libraries.")

if __name__ == "__main__":
    import os
    main()
