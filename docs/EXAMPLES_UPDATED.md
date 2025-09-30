# HPFRACC Examples & Tutorials (Updated)

## Table of Contents
1. [Quick Start Examples](#quick-start-examples)
2. [Spectral Autograd Framework](#spectral-autograd-framework)
3. [Machine Learning Integration](#machine-learning-integration)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Advanced Applications](#advanced-applications)
6. [Scientific Computing](#scientific-computing)
7. [Validation & Testing](#validation--testing)

---

## Quick Start Examples

### Basic Fractional Derivative Computation

```python
import hpfracc as hpc
import numpy as np
import matplotlib.pyplot as plt

# Create test function
x = np.linspace(0, 2*np.pi, 100)
f = np.sin(x)

# Compute fractional derivatives
alpha_values = [0.25, 0.5, 0.75, 1.0]
derivatives = {}

for alpha in alpha_values:
    derivatives[alpha] = hpc.fractional_derivative(f, alpha, method="caputo")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, f, 'k-', label='Original function sin(x)', linewidth=2)

for alpha, deriv in derivatives.items():
    plt.plot(x, deriv, '--', label=f'D^{alpha} sin(x)')

plt.xlabel('x')
plt.ylabel('Function value')
plt.legend()
plt.grid(True)
plt.title('Fractional Derivatives of sin(x)')
plt.show()
```

### Multiple Derivative Definitions

```python
from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
)

# Test function
def f(x):
    return x**2

x = np.linspace(0, 5, 100)
alpha = 0.5

# Compute using different definitions
rl_deriv = RiemannLiouvilleDerivative(alpha)
caputo_deriv = CaputoDerivative(alpha)
gl_deriv = GrunwaldLetnikovDerivative(alpha)

rl_result = rl_deriv.compute(f, x)
caputo_result = caputo_deriv.compute(f, x)
gl_result = gl_deriv.compute(f, x)

# Compare results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, rl_result, 'b-', label='Riemann-Liouville')
plt.title('Riemann-Liouville Derivative')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, caputo_result, 'r-', label='Caputo')
plt.title('Caputo Derivative')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, gl_result, 'g-', label='Grünwald-Letnikov')
plt.title('Grünwald-Letnikov Derivative')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Spectral Autograd Framework

### Basic Spectral Autograd

```python
import torch
import torch.nn as nn
from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

# Create input with gradient support
x = torch.randn(32, requires_grad=True)
alpha = 0.5

# Apply spectral fractional derivative
result = SpectralFractionalDerivative.apply(x, alpha, -1, "fft")

print(f"Input shape: {x.shape}")
print(f"Output shape: {result.shape}")
print(f"Requires grad: {result.requires_grad}")

# Compute gradients
loss = torch.sum(result)
loss.backward()

print(f"Input gradient norm: {x.grad.norm().item():.6f}")
```

### Learnable Fractional Orders

```python
import torch
import torch.nn as nn
import torch.optim as optim
from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

class FractionalNeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.alpha_param = BoundedAlphaParameter(alpha_init=1.0)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Apply spectral fractional derivative with learnable alpha
        alpha_val = self.alpha_param()
        x_frac = SpectralFractionalDerivative.apply(x, alpha_val, -1, "fft")
        
        # Standard neural network layers
        x = torch.relu(self.linear1(x_frac))
        x = self.linear2(x)
        return x

# Create model and data
model = FractionalNeuralNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate synthetic data
x_train = torch.randn(100, 10, requires_grad=True)
y_train = torch.sum(x_train**2, dim=1, keepdim=True)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    output = model(x_train)
    loss = nn.MSELoss()(output, y_train)
    loss.backward()
    
    optimizer.step()
    
    if epoch % 20 == 0:
        alpha_val = model.alpha_param.get_alpha()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Alpha: {alpha_val:.4f}")
```

### Spectral Autograd with Different Methods

```python
import torch
from hpfracc.ml import SpectralFractionalDerivative

# Test different spectral methods
x = torch.randn(64, requires_grad=True)
alpha = 0.7

methods = ["fft", "mellin"]
results = {}

for method in methods:
    try:
        result = SpectralFractionalDerivative.apply(x, alpha, -1, method)
        
        # Compute gradients
        loss = torch.sum(result)
        loss.backward()
        
        results[method] = {
            'output': result.detach(),
            'gradient_norm': x.grad.norm().item()
        }
        
        # Clear gradients for next iteration
        x.grad.zero_()
        
    except Exception as e:
        print(f"Method {method} failed: {e}")

# Compare results
for method, data in results.items():
    print(f"{method.upper()} Method:")
    print(f"  Output shape: {data['output'].shape}")
    print(f"  Gradient norm: {data['gradient_norm']:.6f}")
```

---

## Machine Learning Integration

### Fractional LSTM Network

```python
import torch
import torch.nn as nn
from hpfracc.ml.layers import FractionalLSTM

class FractionalLSTMNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, alpha=0.5):
        super().__init__()
        self.lstm = FractionalLSTM(input_size, hidden_size, alpha=alpha)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        output = self.fc(lstm_out)  # (batch_size, seq_len, output_size)
        return output

# Create model and test data
model = FractionalLSTMNetwork(input_size=1, hidden_size=64, alpha=0.7)
x = torch.randn(32, 10, 1)  # batch_size=32, seq_len=10, input_size=1

# Forward pass
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Requires grad: {output.requires_grad}")
```

### Fractional Pooling

```python
import torch
from hpfracc.ml.layers import FractionalPooling

# Test 1D pooling
x_1d = torch.randn(2, 3, 10)  # (batch, channels, length)
pool_1d = FractionalPooling(kernel_size=2, stride=2, alpha=0.5)
output_1d = pool_1d(x_1d)

print(f"1D Input shape: {x_1d.shape}")
print(f"1D Output shape: {output_1d.shape}")

# Test 2D pooling
x_2d = torch.randn(2, 3, 10, 10)  # (batch, channels, height, width)
pool_2d = FractionalPooling(kernel_size=2, stride=2, alpha=0.5)
output_2d = pool_2d(x_2d)

print(f"2D Input shape: {x_2d.shape}")
print(f"2D Output shape: {output_2d.shape}")
```

### Backend Management

```python
from hpfracc.ml.backends import BackendManager, BackendType

# Check available backends
available = BackendManager.get_available_backends()
print(f"Available backends: {available}")

# Set backend
if BackendType.TORCH in available:
    BackendManager.set_backend(BackendType.TORCH)
    print("Using PyTorch backend")
elif BackendType.JAX in available:
    BackendManager.set_backend(BackendType.JAX)
    print("Using JAX backend")
else:
    BackendManager.set_backend(BackendType.NUMPY)
    print("Using NumPy backend")

# Get current backend
current = BackendManager.get_current_backend()
print(f"Current backend: {current}")
```

---

## Performance Benchmarking

### Comprehensive Benchmarking

```python
import time
import numpy as np
from hpfracc.core.fractional_implementations import (
    RiemannLiouvilleDerivative, CaputoDerivative, GrunwaldLetnikovDerivative
)

def benchmark_derivative_methods():
    """Benchmark different derivative methods."""
    
    # Test parameters
    sizes = [100, 500, 1000, 2000]
    alpha_values = [0.25, 0.5, 0.75]
    
    methods = {
        'Riemann-Liouville': RiemannLiouvilleDerivative,
        'Caputo': CaputoDerivative,
        'Grünwald-Letnikov': GrunwaldLetnikovDerivative
    }
    
    results = {}
    
    for method_name, method_class in methods.items():
        results[method_name] = {}
        
        for size in sizes:
            # Generate test data
            x = np.linspace(0, 2*np.pi, size)
            f = np.sin(x)
            
            # Benchmark
            times = []
            for alpha in alpha_values:
                deriv = method_class(alpha)
                
                start_time = time.perf_counter()
                result = deriv.compute(lambda t: np.sin(t), x)
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = size / avg_time
            
            results[method_name][size] = {
                'avg_time': avg_time,
                'throughput': throughput
            }
    
    return results

# Run benchmarks
benchmark_results = benchmark_derivative_methods()

# Display results
for method_name, method_results in benchmark_results.items():
    print(f"\n{method_name}:")
    for size, metrics in method_results.items():
        print(f"  Size {size}: {metrics['throughput']:.0f} samples/sec")
```

### Special Functions Benchmarking

```python
from hpfracc.special import mittag_leffler_function as mittag_leffler
from hpfracc.special.binomial_coeffs import BinomialCoefficients

def benchmark_special_functions():
    """Benchmark special functions."""
    
    # Benchmark Mittag-Leffler function
    print("Benchmarking Mittag-Leffler function...")
    z_values = [0.1, 0.5, 1.0, 2.0]
    alpha_values = [0.25, 0.5, 0.75, 1.0]
    
    ml_times = []
    for z in z_values:
        for alpha in alpha_values:
            start_time = time.perf_counter()
            result = mittag_leffler(z, alpha, 1.0)
            end_time = time.perf_counter()
            ml_times.append(end_time - start_time)
    
    avg_ml_time = np.mean(ml_times)
    print(f"Average Mittag-Leffler time: {avg_ml_time*1000:.3f} ms")
    
    # Benchmark binomial coefficients
    print("Benchmarking binomial coefficients...")
    bc = BinomialCoefficients()
    test_cases = [(10, 5), (20, 10), (50, 25), (100, 50)]
    
    bc_times = []
    for n, k in test_cases:
        start_time = time.perf_counter()
        result = bc.compute(n, k)
        end_time = time.perf_counter()
        bc_times.append(end_time - start_time)
    
    avg_bc_time = np.mean(bc_times)
    print(f"Average binomial coefficient time: {avg_bc_time*1000:.3f} ms")

# Run special functions benchmarks
benchmark_special_functions()
```

---

## Advanced Applications

### Fractional Signal Processing

```python
import numpy as np
import matplotlib.pyplot as plt
from hpfracc.core.fractional_implementations import RiemannLiouvilleDerivative

def fractional_signal_processing():
    """Apply fractional derivatives to signal processing."""
    
    # Generate test signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2*np.pi*t) + 0.5*np.sin(4*np.pi*t) + 0.1*np.random.randn(len(t))
    
    # Apply fractional derivatives
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    derivatives = {}
    
    for alpha in alpha_values:
        deriv = RiemannLiouvilleDerivative(alpha)
        derivatives[alpha] = deriv.compute(lambda x: signal, t)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, 'k-', label='Original signal')
    plt.title('Original Signal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    for alpha, deriv in derivatives.items():
        plt.plot(t, deriv, '--', label=f'α = {alpha}')
    plt.title('Fractional Derivatives')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    # Show frequency content
    fft_original = np.abs(np.fft.fft(signal))
    fft_deriv = np.abs(np.fft.fft(derivatives[0.5]))
    freqs = np.fft.fftfreq(len(t), t[1]-t[0])
    
    plt.plot(freqs[:len(freqs)//2], fft_original[:len(freqs)//2], 'k-', label='Original')
    plt.plot(freqs[:len(freqs)//2], fft_deriv[:len(freqs)//2], 'r--', label='α = 0.5')
    plt.title('Frequency Content')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run signal processing example
fractional_signal_processing()
```

### Fractional Image Processing

```python
import numpy as np
import matplotlib.pyplot as plt
from hpfracc.core.fractional_implementations import RiemannLiouvilleDerivative

def fractional_image_processing():
    """Apply fractional derivatives to image processing."""
    
    # Create test image
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    image = np.sin(x) * np.cos(y) + 0.1 * np.random.randn(100, 100)
    
    # Apply fractional derivatives
    alpha = 0.5
    deriv_x = RiemannLiouvilleDerivative(alpha)
    deriv_y = RiemannLiouvilleDerivative(alpha)
    
    # Compute fractional gradients
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        gradient_x[i, :] = deriv_x.compute(lambda t: image[i, :], np.arange(image.shape[1]))
    
    for j in range(image.shape[1]):
        gradient_y[:, j] = deriv_y.compute(lambda y: image[:, j], np.arange(image.shape[0]))
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(gradient_x, cmap='gray')
    plt.title('Fractional Gradient X')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# Run image processing example
fractional_image_processing()
```

---

## Scientific Computing

### Fractional ODE Solving

```python
from hpfracc.solvers import FractionalODESolver, PredictorCorrectorSolver

def solve_fractional_ode():
    """Solve a fractional ordinary differential equation."""
    
    # Define the fractional ODE: D^α y(t) = -y(t)
    def ode_func(t, y):
        return -y
    
    # Initial conditions
    t_span = (0, 5)
    y0 = 1.0
    alpha = 0.8
    
    # Solve using different methods
    solver1 = FractionalODESolver()
    t1, y1 = solver1.solve(ode_func, t_span, y0, alpha)
    
    solver2 = PredictorCorrectorSolver("caputo", alpha)
    t2, y2 = solver2.solve(ode_func, t_span, y0)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t1, y1, 'b-', label=f'Fractional ODE Solver (α={alpha})')
    plt.plot(t2, y2, 'r--', label='Predictor-Corrector Solver')
    plt.xlabel('Time')
    plt.ylabel('Solution')
    plt.title('Fractional ODE Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

# Solve fractional ODE
solve_fractional_ode()
```

### Convergence Analysis

```python
from hpfracc.validation import ConvergenceTester, run_convergence_study

def convergence_analysis():
    """Analyze convergence of fractional derivative methods."""
    
    # Define test function and analytical solution
    def test_func(x):
        return x**2
    
    def analytical_solution(x, alpha):
        # Analytical solution for D^α(x²)
        import math
        return 2 * x**(2 - alpha) / math.gamma(3 - alpha)
    
    # Test convergence with different grid sizes
    grid_sizes = [20, 40, 80, 160, 320]
    
    # Run convergence study
    results = run_convergence_study(
        lambda x: RiemannLiouvilleDerivative(0.5).compute(test_func, x),
        lambda x: analytical_solution(x, 0.5),
        [{'alpha': 0.5}],
        grid_sizes
    )
    
    # Extract convergence data
    if 'test_cases' in results and len(results['test_cases']) > 0:
        test_case = results['test_cases'][0]
        if 'l2' in test_case and 'convergence_rate' in test_case['l2']:
            rate = test_case['l2']['convergence_rate']
            print(f"Convergence rate: {rate:.3f}")
    
    return results

# Run convergence analysis
convergence_results = convergence_analysis()
```

---

## Validation & Testing

### Mathematical Validation

```python
from hpfracc.validation import validate_against_analytical, get_analytical_solution

def mathematical_validation():
    """Validate numerical methods against analytical solutions."""
    
    # Test power function derivatives
    x = np.linspace(0.1, 2.0, 50)
    
    def numerical_method(x, alpha=0.5):
        deriv = RiemannLiouvilleDerivative(alpha)
        return deriv.compute(lambda t: t**1.5, x)
    
    def analytical_method(x, alpha=0.5):
        import math
        return 1.5 * x**(1.5 - alpha) / math.gamma(2.5 - alpha)
    
    # Validate against analytical solution
    test_params = [{'alpha': 0.5}]
    validation_results = validate_against_analytical(
        numerical_method,
        analytical_method,
        test_params
    )
    
    print("Validation Results:")
    print(f"Success rate: {validation_results['summary']['success_rate']:.1%}")
    
    for i, result in enumerate(validation_results['results']):
        if result['success']:
            print(f"Test case {i}: PASSED (max error: {result['max_error']:.6f})")
        else:
            print(f"Test case {i}: FAILED")
    
    return validation_results

# Run mathematical validation
validation_results = mathematical_validation()
```

### Performance Testing

```python
def performance_testing():
    """Test performance across different problem sizes."""
    
    sizes = [100, 500, 1000, 2000, 5000]
    alpha = 0.5
    
    times = []
    throughputs = []
    
    for size in sizes:
        x = np.linspace(0, 2*np.pi, size)
        f = np.sin(x)
        
        deriv = RiemannLiouvilleDerivative(alpha)
        
        # Time the computation
        start_time = time.perf_counter()
        result = deriv.compute(lambda t: np.sin(t), x)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        throughput = size / execution_time
        
        times.append(execution_time)
        throughputs.append(throughput)
        
        print(f"Size {size:4d}: {execution_time*1000:6.2f} ms, {throughput:8.0f} samples/sec")
    
    # Plot performance
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'bo-')
    plt.xlabel('Problem Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Problem Size')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sizes, throughputs, 'ro-')
    plt.xlabel('Problem Size')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Throughput vs Problem Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run performance testing
performance_testing()
```

---

## Complete Workflow Example

### End-to-End Fractional Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hpfracc.ml import SpectralFractionalDerivative, BoundedAlphaParameter

class CompleteFractionalNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super().__init__()
        
        # Learnable fractional order
        self.alpha_param = BoundedAlphaParameter(alpha_init=1.0)
        
        # Network layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Input processing
        x = self.activation(self.input_layer(x))
        
        # Apply spectral fractional derivative
        alpha_val = self.alpha_param()
        x_frac = SpectralFractionalDerivative.apply(x, alpha_val, -1, "fft")
        
        # Hidden processing
        x = self.activation(self.hidden_layer(x_frac))
        
        # Output
        x = self.output_layer(x)
        return x

def train_fractional_network():
    """Train a complete fractional neural network."""
    
    # Create model and optimizer
    model = CompleteFractionalNetwork()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Generate training data
    n_samples = 1000
    X = torch.randn(n_samples, 10)
    y = torch.sum(X**2, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        
        optimizer.step()
        
        if epoch % 20 == 0:
            alpha_val = model.alpha_param.get_alpha()
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}, Alpha = {alpha_val:.4f}")
    
    return model

# Train the network
trained_model = train_fractional_network()

# Test the trained model
test_input = torch.randn(10, 10)
with torch.no_grad():
    test_output = trained_model(test_input)
    final_alpha = trained_model.alpha_param.get_alpha()

print(f"\nFinal Results:")
print(f"Test output shape: {test_output.shape}")
print(f"Final alpha value: {final_alpha:.4f}")
print(f"Model trained successfully!")
```

---

*These examples demonstrate the full capabilities of HPFRACC, from basic fractional calculus operations to advanced machine learning applications. Each example is designed to be runnable and provides a foundation for more complex applications.*

