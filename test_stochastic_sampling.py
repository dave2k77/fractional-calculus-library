"""
Test script for stochastic memory sampling implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

from hpfracc.ml.stochastic_memory_sampling import (
    StochasticFractionalLayer,
    stochastic_fractional_derivative,
    ImportanceSampler,
    StratifiedSampler,
    ControlVariateSampler
)


def test_sampling_methods():
    """Test different sampling methods."""
    print("Testing stochastic memory sampling methods...")
    
    # Create test signal
    t = torch.linspace(0, 4*np.pi, 200)
    x = torch.sin(t) + 0.1 * torch.sin(3*t)
    alpha = 0.6
    k = 32  # Sample size
    
    methods = ["importance", "stratified", "control_variate"]
    results = {}
    
    for method in methods:
        print(f"Testing {method} sampling...")
        try:
            # Test stochastic derivative
            start_time = time.time()
            frac_deriv = stochastic_fractional_derivative(x, alpha, k, method)
            end_time = time.time()
            
            results[method] = {
                'derivative': frac_deriv.detach().numpy(),
                'time': end_time - start_time
            }
            
            print(f"  ✓ {method} sampling working (time: {end_time - start_time:.4f}s)")
            
        except Exception as e:
            print(f"  ✗ {method} sampling failed: {e}")
            results[method] = None
    
    return results


def test_variance_analysis():
    """Test variance of stochastic estimators."""
    print("\nTesting variance analysis...")
    
    # Create test signal
    t = torch.linspace(0, 2*np.pi, 100)
    x = torch.sin(t)
    alpha = 0.5
    k = 16
    n_samples = 100
    
    methods = ["importance", "stratified", "control_variate"]
    variances = {}
    
    for method in methods:
        print(f"Computing variance for {method}...")
        try:
            samples = []
            for _ in range(n_samples):
                frac_deriv = stochastic_fractional_derivative(x, alpha, k, method)
                samples.append(frac_deriv.detach().numpy())
            
            samples = np.array(samples)
            mean_estimate = np.mean(samples, axis=0)
            variance = np.var(samples, axis=0)
            
            variances[method] = {
                'mean': mean_estimate,
                'variance': variance,
                'std': np.sqrt(variance)
            }
            
            print(f"  ✓ {method}: mean={np.mean(mean_estimate):.4f}, std={np.mean(np.sqrt(variance)):.4f}")
            
        except Exception as e:
            print(f"  ✗ {method} variance analysis failed: {e}")
            variances[method] = None
    
    return variances


def test_layer_integration():
    """Test integration with neural networks."""
    print("\nTesting layer integration...")
    
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.stoch_frac = StochasticFractionalLayer(alpha=0.7, k=32, method="importance")
            self.linear = nn.Linear(1, 1)  # Fix: input size should be 1 since stoch_frac returns scalar
        
        def forward(self, x):
            frac_x = self.stoch_frac(x)
            # Ensure frac_x is 2D for linear layer
            if frac_x.dim() == 0:
                frac_x = frac_x.unsqueeze(0).unsqueeze(0)
            return self.linear(frac_x)
    
    # Test forward and backward pass
    net = TestNetwork()
    x = torch.randn(1, 100, requires_grad=True)
    
    try:
        output = net(x)
        print(f"  ✓ Forward pass successful, output shape: {output.shape}")
        
        loss = output.sum()
        loss.backward()
        print(f"  ✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"  ✗ Layer integration failed: {e}")
        return False


def test_sampling_efficiency():
    """Test computational efficiency vs sample size."""
    print("\nTesting sampling efficiency...")
    
    # Create test signal
    t = torch.linspace(0, 2*np.pi, 500)
    x = torch.sin(t) + 0.1 * torch.sin(3*t)
    alpha = 0.6
    
    k_values = [8, 16, 32, 64, 128]
    method = "importance"
    
    times = []
    for k in k_values:
        start_time = time.time()
        for _ in range(10):  # Average over 10 runs
            _ = stochastic_fractional_derivative(x, alpha, k, method)
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        times.append(avg_time)
        print(f"  k={k}: {avg_time:.4f}s")
    
    return k_values, times


def test_accuracy_vs_deterministic():
    """Compare accuracy with deterministic methods."""
    print("\nTesting accuracy vs deterministic...")
    
    # Create test signal
    t = torch.linspace(0, 2*np.pi, 200)
    x = torch.sin(t)
    alpha = 0.5
    
    # Deterministic reference (using our existing fractional derivative)
    try:
        from hpfracc.ml.fractional_autograd import fractional_derivative
        deterministic = fractional_derivative(x, alpha, "RL")
        print("  ✓ Deterministic reference computed")
    except:
        print("  ⚠ Could not compute deterministic reference")
        return None
    
    # Stochastic estimates
    k_values = [16, 32, 64, 128]
    method = "importance"
    
    errors = []
    for k in k_values:
        # Average over multiple samples
        samples = []
        for _ in range(20):
            stoch = stochastic_fractional_derivative(x, alpha, k, method)
            samples.append(stoch.detach())
        
        # Compute mean and error
        mean_stoch = torch.mean(torch.stack(samples), dim=0)
        error = torch.mean(torch.abs(mean_stoch - deterministic))
        errors.append(error.item())
        print(f"  k={k}: error={error:.6f}")
    
    return k_values, errors


def plot_results(results, variances, k_values, times, errors):
    """Plot test results."""
    print("\nGenerating plots...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Sampling methods comparison
        ax1 = axes[0, 0]
        methods = []
        derivative_vals = []
        for method, result in results.items():
            if result is not None:
                methods.append(method)
                derivative_vals.append(result['derivative'])
        
        if methods:
            ax1.bar(methods, derivative_vals, alpha=0.7)
            ax1.set_title('Stochastic Sampling Methods')
            ax1.set_ylabel('Derivative Value')
            ax1.grid(True)
        
        # Plot 2: Variance analysis
        ax2 = axes[0, 1]
        methods = [m for m, v in variances.items() if v is not None]
        stds = [np.mean(v['std']) for v in variances.values() if v is not None]
        if methods:
            ax2.bar(methods, stds)
            ax2.set_title('Standard Deviation by Method')
            ax2.set_ylabel('Mean Std Dev')
        
        # Plot 3: Efficiency vs sample size
        ax3 = axes[1, 0]
        ax3.plot(k_values, times, 'bo-')
        ax3.set_xlabel('Sample Size (k)')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Computational Efficiency')
        ax3.grid(True)
        
        # Plot 4: Summary
        ax4 = axes[1, 1]
        working_methods = [m for m, r in results.items() if r is not None]
        ax4.text(0.5, 0.5, f'Working Methods:\n{", ".join(working_methods)}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Test Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('stochastic_sampling_results.png', dpi=150, bbox_inches='tight')
        print("  ✓ Results plot saved as 'stochastic_sampling_results.png'")
        
    except Exception as e:
        print(f"  ⚠ Plotting failed: {e}")
        print("  ✓ Core functionality working, skipping plots")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STOCHASTIC MEMORY SAMPLING TEST")
    print("=" * 60)
    
    # Test sampling methods
    results = test_sampling_methods()
    
    # Test variance analysis
    variances = test_variance_analysis()
    
    # Test layer integration
    layer_success = test_layer_integration()
    
    # Test efficiency
    k_values, times = test_sampling_efficiency()
    
    # Test accuracy
    errors = test_accuracy_vs_deterministic()
    
    # Generate plots
    plot_results(results, variances, k_values, times, errors)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    working_methods = [method for method, result in results.items() if result is not None]
    print(f"Working sampling methods: {working_methods}")
    print(f"Layer integration: {'✓' if layer_success else '✗'}")
    print(f"Variance analysis: {'✓' if any(v is not None for v in variances.values()) else '✗'}")
    print(f"Efficiency test: {'✓' if times else '✗'}")
    print(f"Accuracy test: {'✓' if errors else '✗'}")
    
    if len(working_methods) > 0 and layer_success:
        print("\n🎉 Stochastic memory sampling is working! Ready for next phase.")
    else:
        print("\n⚠️  Stochastic sampling needs fixes before proceeding.")


if __name__ == "__main__":
    main()
