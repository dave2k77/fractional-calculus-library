#!/usr/bin/env python3
"""
Spectral Autograd Comparison Test

This script compares the new spectral autograd implementation with the previous
fractional autograd implementation to verify that gradients flow correctly.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

# Import both implementations
from hpfracc.ml.spectral_autograd import SpectralFractionalDerivative, SpectralFractionalLayer
from hpfracc.ml.fractional_autograd import fractional_derivative

class SpectralAutogradComparisonTest:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
        # Test configurations
        self.test_sizes = [32, 64, 128, 256, 512]
        self.fractional_orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        print(f"Running spectral autograd comparison tests on {self.device}")
        print(f"Test sizes: {self.test_sizes}")
        print(f"Fractional orders: {self.fractional_orders}")
    
    def test_gradient_flow(self) -> Dict:
        """Test that gradients flow correctly through both implementations"""
        print("\n=== Gradient Flow Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'orders': self.fractional_orders,
            'spectral_gradients': [],
            'old_gradients': [],
            'gradient_ratios': []
        }
        
        for size in self.test_sizes:
            print(f"Testing gradient flow for size {size}...")
            
            for alpha in self.fractional_orders:
                # Create test data
                x = torch.randn(1, size, requires_grad=True, device=self.device)
                target = torch.randn(1, size, device=self.device)
                
                # Test spectral autograd
                try:
                    x.grad = None
                    spectral_result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
                    spectral_loss = torch.nn.functional.mse_loss(spectral_result, target)
                    spectral_loss.backward()
                    
                    if x.grad is not None:
                        spectral_grad_norm = x.grad.norm().item()
                    else:
                        spectral_grad_norm = 0.0
                        
                except Exception as e:
                    print(f"  Spectral autograd failed for size {size}, alpha {alpha}: {e}")
                    spectral_grad_norm = 0.0
                
                # Test old fractional autograd
                try:
                    x.grad = None
                    old_result = fractional_derivative(x, alpha, method="RL")
                    old_loss = torch.nn.functional.mse_loss(old_result, target)
                    old_loss.backward()
                    
                    if x.grad is not None:
                        old_grad_norm = x.grad.norm().item()
                    else:
                        old_grad_norm = 0.0
                        
                except Exception as e:
                    print(f"  Old fractional autograd failed for size {size}, alpha {alpha}: {e}")
                    old_grad_norm = 0.0
                
                # Calculate gradient ratio
                if old_grad_norm > 0:
                    gradient_ratio = spectral_grad_norm / old_grad_norm
                else:
                    gradient_ratio = float('inf') if spectral_grad_norm > 0 else 0.0
                
                results['spectral_gradients'].append(spectral_grad_norm)
                results['old_gradients'].append(old_grad_norm)
                results['gradient_ratios'].append(gradient_ratio)
                
                print(f"  Size {size}, α={alpha}: Spectral {spectral_grad_norm:.6f}, Old {old_grad_norm:.6f}, Ratio {gradient_ratio:.2f}")
        
        return results
    
    def test_performance_comparison(self) -> Dict:
        """Compare performance between spectral and old autograd"""
        print("\n=== Performance Comparison Test ===")
        
        results = {
            'sizes': self.test_sizes,
            'spectral_times': [],
            'old_times': [],
            'speedup_ratios': []
        }
        
        for size in self.test_sizes:
            print(f"Testing performance for size {size}...")
            
            # Test spectral autograd performance
            x = torch.randn(1, size, requires_grad=True, device=self.device)
            target = torch.randn(1, size, device=self.device)
            
            start_time = time.time()
            for _ in range(10):  # Multiple runs
                x.grad = None
                spectral_result = SpectralFractionalDerivative.apply(x, 0.5, -1, "fft")
                spectral_loss = torch.nn.functional.mse_loss(spectral_result, target)
                spectral_loss.backward()
            spectral_time = (time.time() - start_time) / 10
            
            # Test old fractional autograd performance
            start_time = time.time()
            for _ in range(10):  # Multiple runs
                x.grad = None
                old_result = fractional_derivative(x, 0.5, method="RL")
                old_loss = torch.nn.functional.mse_loss(old_result, target)
                old_loss.backward()
            old_time = (time.time() - start_time) / 10
            
            speedup = old_time / spectral_time if spectral_time > 0 else float('inf')
            
            results['spectral_times'].append(spectral_time)
            results['old_times'].append(old_time)
            results['speedup_ratios'].append(speedup)
            
            print(f"  Size {size}: Spectral {spectral_time:.4f}s, Old {old_time:.4f}s, Speedup {speedup:.2f}x")
        
        return results
    
    def test_accuracy_comparison(self) -> Dict:
        """Compare accuracy between spectral and old autograd"""
        print("\n=== Accuracy Comparison Test ===")
        
        results = {
            'orders': self.fractional_orders,
            'spectral_errors': [],
            'old_errors': [],
            'error_ratios': []
        }
        
        # Test function: f(x) = x^2
        x = torch.tensor([1.0], requires_grad=True, device=self.device)
        
        for alpha in self.fractional_orders:
            print(f"Testing accuracy for α = {alpha}...")
            
            # Analytical fractional derivative of x^2
            from scipy.special import gamma
            analytical_grad = (2.0 / gamma(3 - alpha)) * (x.item() ** (2 - alpha))
            
            # Test spectral autograd accuracy
            try:
                x.grad = None
                spectral_result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")
                spectral_error = abs(spectral_result.item() - analytical_grad) / abs(analytical_grad)
            except Exception as e:
                print(f"  Spectral autograd failed for α={alpha}: {e}")
                spectral_error = float('inf')
            
            # Test old fractional autograd accuracy
            try:
                x.grad = None
                old_result = fractional_derivative(x, alpha, method="RL")
                old_error = abs(old_result.item() - analytical_grad) / abs(analytical_grad)
            except Exception as e:
                print(f"  Old fractional autograd failed for α={alpha}: {e}")
                old_error = float('inf')
            
            # Calculate error ratio
            if old_error > 0:
                error_ratio = spectral_error / old_error
            else:
                error_ratio = float('inf') if spectral_error > 0 else 1.0
            
            results['spectral_errors'].append(spectral_error)
            results['old_errors'].append(old_error)
            results['error_ratios'].append(error_ratio)
            
            print(f"  α = {alpha}: Spectral error {spectral_error:.2e}, Old error {old_error:.2e}, Ratio {error_ratio:.2f}")
        
        return results
    
    def test_neural_network_integration(self) -> Dict:
        """Test integration with neural networks"""
        print("\n=== Neural Network Integration Test ===")
        
        results = {
            'convergence_spectral': [],
            'convergence_old': [],
            'final_loss_spectral': [],
            'final_loss_old': []
        }
        
        # Create simple neural network with spectral fractional layer
        class SpectralFractionalNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, alpha):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.frac_layer = SpectralFractionalLayer(alpha, method="fft")
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.frac_layer(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        # Create simple neural network with old fractional layer
        class OldFractionalNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, alpha):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.alpha = alpha
                self.linear2 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.linear1(x)
                x = fractional_derivative(x, self.alpha, method="RL")
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        # Test parameters
        input_size = 32
        hidden_size = 64
        output_size = 10
        alpha = 0.5
        num_epochs = 10
        
        # Create test data
        X = torch.randn(100, input_size, device=self.device)
        y = torch.randint(0, output_size, (100,), device=self.device)
        
        # Test spectral network
        try:
            spectral_net = SpectralFractionalNet(input_size, hidden_size, output_size, alpha).to(self.device)
            spectral_optimizer = optim.Adam(spectral_net.parameters(), lr=0.001)
            spectral_criterion = nn.CrossEntropyLoss()
            
            spectral_losses = []
            for epoch in range(num_epochs):
                spectral_optimizer.zero_grad()
                output = spectral_net(X)
                loss = spectral_criterion(output, y)
                loss.backward()
                spectral_optimizer.step()
                spectral_losses.append(loss.item())
            
            results['convergence_spectral'] = spectral_losses
            results['final_loss_spectral'] = [spectral_losses[-1]]
            
        except Exception as e:
            print(f"  Spectral network failed: {e}")
            results['convergence_spectral'] = [float('inf')] * num_epochs
            results['final_loss_spectral'] = [float('inf')]
        
        # Test old network
        try:
            old_net = OldFractionalNet(input_size, hidden_size, output_size, alpha).to(self.device)
            old_optimizer = optim.Adam(old_net.parameters(), lr=0.001)
            old_criterion = nn.CrossEntropyLoss()
            
            old_losses = []
            for epoch in range(num_epochs):
                old_optimizer.zero_grad()
                output = old_net(X)
                loss = old_criterion(output, y)
                loss.backward()
                old_optimizer.step()
                old_losses.append(loss.item())
            
            results['convergence_old'] = old_losses
            results['final_loss_old'] = [old_losses[-1]]
            
        except Exception as e:
            print(f"  Old network failed: {e}")
            results['convergence_old'] = [float('inf')] * num_epochs
            results['final_loss_old'] = [float('inf')]
        
        print(f"  Spectral final loss: {results['final_loss_spectral'][0]:.6f}")
        print(f"  Old final loss: {results['final_loss_old'][0]:.6f}")
        
        return results
    
    def run_all_tests(self) -> Dict:
        """Run all comparison tests"""
        print("Starting comprehensive spectral autograd comparison testing...")
        
        all_results = {
            'gradient_flow': self.test_gradient_flow(),
            'performance': self.test_performance_comparison(),
            'accuracy': self.test_accuracy_comparison(),
            'neural_network': self.test_neural_network_integration()
        }
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = "spectral_autograd_comparison_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def plot_results(self, results: Dict):
        """Create comparison visualization plots"""
        print("\nCreating comparison plots...")
        
        plt.figure(figsize=(15, 10))
        
        # Gradient flow comparison
        plt.subplot(2, 3, 1)
        sizes = results['gradient_flow']['sizes']
        spectral_grads = results['gradient_flow']['spectral_gradients']
        old_grads = results['gradient_flow']['old_gradients']
        
        # Reshape data for plotting (5 sizes x 5 orders = 25 points)
        spectral_grads_reshaped = np.array(spectral_grads).reshape(len(sizes), len(results['gradient_flow']['orders']))
        old_grads_reshaped = np.array(old_grads).reshape(len(sizes), len(results['gradient_flow']['orders']))
        
        # Plot average gradient norm for each size
        avg_spectral_grads = np.mean(spectral_grads_reshaped, axis=1)
        avg_old_grads = np.mean(old_grads_reshaped, axis=1)
        
        plt.plot(sizes, avg_spectral_grads, 'b-o', label='Spectral Autograd')
        plt.plot(sizes, avg_old_grads, 'r-s', label='Old Fractional Autograd')
        plt.xlabel('Problem Size')
        plt.ylabel('Average Gradient Norm')
        plt.title('Gradient Flow Comparison')
        plt.legend()
        plt.grid(True)
        
        # Performance comparison
        plt.subplot(2, 3, 2)
        spectral_times = results['performance']['spectral_times']
        old_times = results['performance']['old_times']
        
        plt.plot(sizes, spectral_times, 'b-o', label='Spectral Autograd')
        plt.plot(sizes, old_times, 'r-s', label='Old Fractional Autograd')
        plt.xlabel('Problem Size')
        plt.ylabel('Time (s)')
        plt.title('Performance Comparison')
        plt.legend()
        plt.grid(True)
        
        # Accuracy comparison
        plt.subplot(2, 3, 3)
        orders = results['accuracy']['orders']
        spectral_errors = results['accuracy']['spectral_errors']
        old_errors = results['accuracy']['old_errors']
        
        plt.semilogy(orders, spectral_errors, 'b-o', label='Spectral Autograd')
        plt.semilogy(orders, old_errors, 'r-s', label='Old Fractional Autograd')
        plt.xlabel('Fractional Order α')
        plt.ylabel('Relative Error')
        plt.title('Accuracy Comparison')
        plt.legend()
        plt.grid(True)
        
        # Neural network convergence
        plt.subplot(2, 3, 4)
        epochs = range(len(results['neural_network']['convergence_spectral']))
        spectral_losses = results['neural_network']['convergence_spectral']
        old_losses = results['neural_network']['convergence_old']
        
        plt.plot(epochs, spectral_losses, 'b-o', label='Spectral Network')
        plt.plot(epochs, old_losses, 'r-s', label='Old Network')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Convergence')
        plt.legend()
        plt.grid(True)
        
        # Speedup ratios
        plt.subplot(2, 3, 5)
        speedup_ratios = results['performance']['speedup_ratios']
        
        plt.plot(sizes, speedup_ratios, 'g-o')
        plt.xlabel('Problem Size')
        plt.ylabel('Speedup Ratio')
        plt.title('Spectral vs Old Speedup')
        plt.grid(True)
        
        # Error ratios
        plt.subplot(2, 3, 6)
        error_ratios = results['accuracy']['error_ratios']
        
        plt.plot(orders, error_ratios, 'g-o')
        plt.xlabel('Fractional Order α')
        plt.ylabel('Error Ratio (Spectral/Old)')
        plt.title('Accuracy Ratio')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('spectral_autograd_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('spectral_autograd_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved as 'spectral_autograd_comparison.png' and 'spectral_autograd_comparison.pdf'")


def main():
    """Main function to run spectral autograd comparison tests"""
    print("=" * 60)
    print("SPECTRAL AUTOGRAD COMPARISON TEST")
    print("=" * 60)
    
    # Initialize test
    test = SpectralAutogradComparisonTest()
    
    # Run all tests
    results = test.run_all_tests()
    
    # Save results
    test.save_results(results)
    
    # Create plots
    test.plot_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON TEST SUMMARY")
    print("=" * 60)
    
    # Gradient flow summary
    spectral_grads = results['gradient_flow']['spectral_gradients']
    old_grads = results['gradient_flow']['old_gradients']
    avg_spectral_grad = np.mean([g for g in spectral_grads if g > 0])
    avg_old_grad = np.mean([g for g in old_grads if g > 0])
    print(f"Average Spectral Gradient Norm: {avg_spectral_grad:.6f}")
    print(f"Average Old Gradient Norm: {avg_old_grad:.6f}")
    
    # Performance summary
    spectral_times = results['performance']['spectral_times']
    old_times = results['performance']['old_times']
    avg_spectral_time = np.mean(spectral_times)
    avg_old_time = np.mean(old_times)
    avg_speedup = avg_old_time / avg_spectral_time
    print(f"Average Spectral Time: {avg_spectral_time:.4f}s")
    print(f"Average Old Time: {avg_old_time:.4f}s")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    # Accuracy summary
    spectral_errors = results['accuracy']['spectral_errors']
    old_errors = results['accuracy']['old_errors']
    avg_spectral_error = np.mean([e for e in spectral_errors if e < float('inf')])
    avg_old_error = np.mean([e for e in old_errors if e < float('inf')])
    print(f"Average Spectral Error: {avg_spectral_error:.2e}")
    print(f"Average Old Error: {avg_old_error:.2e}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
