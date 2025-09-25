# Fractional Autograd Guide

This guide provides a comprehensive overview of the fractional autograd framework in HPFRACC, including spectral domain computation, stochastic memory sampling, probabilistic fractional orders, and GPU optimization.

## Table of Contents

1. [Overview](#overview)
2. [Spectral Autograd Engines](#spectral-autograd-engines)
3. [Stochastic Memory Sampling](#stochastic-memory-sampling)
4. [Probabilistic Fractional Orders](#probabilistic-fractional-orders)
5. [Variance-Aware Training](#variance-aware-training)
6. [GPU Optimization](#gpu-optimization)
7. [Complete Examples](#complete-examples)
8. [Best Practices](#best-practices)

## Overview

The fractional autograd framework extends PyTorch's automatic differentiation to fractional calculus operations. It provides three main approaches:

- **Spectral Domain Computation**: Efficient fractional derivatives using FFT, Mellin, and Laplacian transforms
- **Stochastic Memory Sampling**: Approximate fractional operators by sampling from memory history
- **Probabilistic Fractional Orders**: Treat fractional orders as random variables for uncertainty quantification

## Spectral Autograd Engines (Unified by Default)

### Basic Usage (Unified)

```python
import torch
from hpfracc.ml.spectral_autograd import FractionalAutogradLayer, SpectralFractionalNetwork

# Create a spectral fractional layer (engine default: FFT)
layer = FractionalAutogradLayer(
    engine_type="fft",  # or "mellin", "laplacian"
    alpha=0.5,
    method="RL"  # Riemann-Liouville
)

# Forward pass
x = torch.randn(32, 128, requires_grad=True)
output = layer(x)

# Backward pass (automatic)
loss = output.sum()
loss.backward()
print(f"Input gradient shape: {x.grad.shape}")
### Unified Network Helper

```python
# Unified by default: specify dims
net = SpectralFractionalNetwork(
    input_dim=128, hidden_dims=[256, 256], output_dim=10,
    alpha=0.5, mode="unified"
)
y = net(x)
```

### Model-Specific (Legacy) Mode

```python
# Opt-in to legacy/coverage-style args
net_legacy = SpectralFractionalNetwork(
    input_size=128, hidden_sizes=[64, 64], output_size=10,
    alpha=0.5, mode="model"
)
```

### Backends
- Supported backends: `pytorch` (default), `jax`, `numba`.
- Fallbacks: if a backend is unavailable, CPU-safe paths are used.
- FFT fallbacks use NumPy FFTs isolated from PyTorch FFT to avoid mock leakage.
```

### Engine Types

#### FFT Engine
- **Best for**: Large sequences, periodic functions
- **Method**: Frequency domain multiplication
- **Performance**: O(N log N) complexity

```python
from hpfracc.ml.spectral_autograd import FFTEngine

engine = FFTEngine()
result = engine.forward(x, alpha=0.5)
```

#### Mellin Engine
- **Best for**: Power-law functions, scale-invariant problems
- **Method**: Mellin transform domain
- **Performance**: O(N log N) complexity

```python
from hpfracc.ml.spectral_autograd import MellinEngine

engine = MellinEngine()
result = engine.forward(x, alpha=0.5)
```

#### Laplacian Engine
- **Best for**: Spatial problems, diffusion equations
- **Method**: Fractional Laplacian in frequency domain
- **Performance**: O(N log N) complexity

```python
from hpfracc.ml.spectral_autograd import LaplacianEngine

engine = LaplacianEngine()
result = engine.forward(x, alpha=0.5)
```

## Stochastic Memory Sampling

### Basic Usage

```python
from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer

# Create stochastic fractional layer
layer = StochasticFractionalLayer(
    alpha=0.5,
    k=32,  # Number of samples
    method="importance"  # or "stratified", "control_variate"
)

# Forward pass
x = torch.randn(32, 128)
output = layer(x)
```

### Sampling Methods

#### Importance Sampling
- **Best for**: General purpose, power-law distributions
- **Variance**: Moderate
- **Computational cost**: O(K)

```python
from hpfracc.ml.stochastic_memory_sampling import ImportanceSampler

sampler = ImportanceSampler(alpha=0.5, k=32)
indices, weights = sampler.sample_indices(n=128, k=32)
estimate = sampler.estimate_derivative(x, indices, weights)
```

#### Stratified Sampling
- **Best for**: When recent history is important
- **Variance**: Low for recent-heavy distributions
- **Computational cost**: O(K)

```python
from hpfracc.ml.stochastic_memory_sampling import StratifiedSampler

sampler = StratifiedSampler(alpha=0.5, k=32, recent_window=16)
indices, weights = sampler.sample_indices(n=128, k=32)
estimate = sampler.estimate_derivative(x, indices, weights)
```

#### Control Variate Sampling
- **Best for**: When baseline estimates are available
- **Variance**: Lowest (with good baseline)
- **Computational cost**: O(K + baseline)

```python
from hpfracc.ml.stochastic_memory_sampling import ControlVariateSampler

sampler = ControlVariateSampler(alpha=0.5, k=32, baseline_window=8)
indices, weights = sampler.sample_indices(n=128, k=32)
estimate = sampler.estimate_derivative(x, indices, weights)
```

### Choosing K (Number of Samples)

The choice of K affects both accuracy and computational cost:

- **K = 8-16**: Fast, suitable for real-time applications
- **K = 32-64**: Balanced accuracy and speed
- **K = 128-256**: High accuracy, slower computation

```python
# Benchmark different K values
from hpfracc.ml.stochastic_memory_sampling import benchmark_k_values

results = benchmark_k_values(
    sequence_length=128,
    alpha=0.5,
    k_values=[8, 16, 32, 64, 128]
)
```

## Probabilistic Fractional Orders

### Basic Usage

```python
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

# Create probabilistic fractional layer
layer = create_normal_alpha_layer(
    mean=0.5,
    std=0.1,
    learnable=True
)

# Forward pass
x = torch.randn(32, 128)
output = layer(x)
```

### Distribution Types

#### Normal Distribution
- **Best for**: Continuous fractional orders
- **Parameters**: mean, std
- **Uncertainty**: Gaussian

```python
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

layer = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=True)
```

#### Uniform Distribution
- **Best for**: Bounded fractional orders
- **Parameters**: low, high
- **Uncertainty**: Uniform

```python
from hpfracc.ml.probabilistic_fractional_orders import create_uniform_alpha_layer

layer = create_uniform_alpha_layer(low=0.1, high=0.9, learnable=True)
```

#### Beta Distribution
- **Best for**: Fractional orders in [0, 1]
- **Parameters**: concentration1, concentration0
- **Uncertainty**: Beta-shaped

```python
from hpfracc.ml.probabilistic_fractional_orders import create_beta_alpha_layer

layer = create_beta_alpha_layer(concentration1=2.0, concentration0=2.0, learnable=True)
```

### Gradient Estimation Methods

#### Reparameterization Trick
- **Best for**: Continuous distributions
- **Variance**: Low
- **Requirements**: Reparameterizable distribution

```python
from hpfracc.ml.probabilistic_fractional_orders import ReparameterizedFractionalDerivative

# Use in custom autograd function
result = ReparameterizedFractionalDerivative.apply(x, alpha_dist, epsilon, method, k)
```

#### Score Function Estimator
- **Best for**: Any distribution
- **Variance**: Higher
- **Requirements**: None

```python
from hpfracc.ml.probabilistic_fractional_orders import ScoreFunctionFractionalDerivative

# Use in custom autograd function
result = ScoreFunctionFractionalDerivative.apply(x, alpha_dist, method, k)
```

## Variance-Aware Training

### Basic Usage

```python
from hpfracc.ml.variance_aware_training import create_variance_aware_trainer

# Create variance-aware trainer
trainer = create_variance_aware_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    base_seed=42,
    variance_threshold=0.1,
    log_interval=10
)

# Train with variance monitoring
results = trainer.train(dataloader, num_epochs=100)
```

### Variance Monitoring

```python
from hpfracc.ml.variance_aware_training import VarianceMonitor

monitor = VarianceMonitor(window_size=100)

# Monitor variance during training
for batch in dataloader:
    output = model(batch)
    monitor.update("output", output)
    
    # Check variance summary
    summary = monitor.get_summary()
    for name, metrics in summary.items():
        if metrics['cv'] > 0.5:  # High variance
            print(f"High variance in {name}: CV={metrics['cv']:.3f}")
```

### Adaptive Sampling

```python
from hpfracc.ml.variance_aware_training import AdaptiveSamplingManager

sampling_manager = AdaptiveSamplingManager(
    initial_k=32,
    min_k=8,
    max_k=128,
    variance_threshold=0.1
)

# Adjust K based on variance
new_k = sampling_manager.update_k(variance=0.2, current_k=32)
print(f"Adjusted K: {new_k}")
```

## GPU Optimization

### Automatic Mixed Precision (AMP)

```python
from hpfracc.ml.gpu_optimization import GPUOptimizedSpectralEngine

# Create GPU-optimized engine
engine = GPUOptimizedSpectralEngine(
    engine_type="fft",
    use_amp=True,
    dtype=torch.float16
)

# Forward pass with AMP
x = torch.randn(32, 1024, device='cuda')
result = engine.forward(x, alpha=0.5)
```

### Chunked FFT for Large Sequences

```python
from hpfracc.ml.gpu_optimization import ChunkedFFT

# Create chunked FFT processor
chunked_fft = ChunkedFFT(chunk_size=1024, overlap=128)

# Process large sequences
x = torch.randn(32, 8192, device='cuda')
x_fft = chunked_fft.fft_chunked(x)
x_reconstructed = chunked_fft.ifft_chunked(x_fft)
```

### Performance Profiling

```python
from hpfracc.ml.gpu_optimization import GPUProfiler

profiler = GPUProfiler(device="cuda")

# Profile operations
profiler.start_timer("fractional_derivative")
result = fractional_derivative(x, alpha=0.5)
profiler.end_timer(x, result)

# Get performance summary
summary = profiler.get_summary()
print(f"Execution time: {summary['fractional_derivative']['execution_time']:.4f}s")
```

## Complete Examples

### End-to-End Training

```python
import torch
import torch.nn as nn
from hpfracc.ml.spectral_autograd import FractionalAutogradLayer
from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer
from hpfracc.ml.variance_aware_training import create_variance_aware_trainer

class FractionalNeuralNetwork(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=1):
        super().__init__()
        
        # Standard layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size + 3, output_size)  # +3 for fractional outputs
        
        # Fractional layers
        self.spectral_layer = FractionalAutogradLayer(engine_type="fft", alpha=0.5)
        self.stochastic_layer = StochasticFractionalLayer(alpha=0.5, k=32, method="importance")
        self.probabilistic_layer = create_normal_alpha_layer(0.5, 0.1, learnable=True)
        
    def forward(self, x):
        # Standard forward pass
        x = torch.relu(self.linear1(x))
        
        # Apply fractional layers
        spectral_out = self.spectral_layer(x)
        stochastic_out = self.stochastic_layer(x)
        probabilistic_out = self.probabilistic_layer(x)
        
        # Handle different output shapes
        if spectral_out.dim() == 2:
            spectral_out = spectral_out.mean(dim=-1, keepdim=True)
        if stochastic_out.dim() == 0:
            stochastic_out = stochastic_out.unsqueeze(0).unsqueeze(-1)
        if probabilistic_out.dim() == 0:
            probabilistic_out = probabilistic_out.unsqueeze(0).unsqueeze(-1)
        
        # Combine features
        x_combined = torch.cat([x, spectral_out, stochastic_out, probabilistic_out], dim=1)
        
        # Final output
        output = self.linear2(x_combined)
        return output

# Create model and trainer
model = FractionalNeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

trainer = create_variance_aware_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    base_seed=42,
    variance_threshold=0.1
)

# Train the model
results = trainer.train(dataloader, num_epochs=100)
```

### Benchmarking Performance

```python
from hpfracc.ml.gpu_optimization import benchmark_gpu_optimization

# Benchmark different configurations
results = benchmark_gpu_optimization()

# Print results
for length, configs in results.items():
    print(f"Sequence length {length}:")
    for config, alphas in configs.items():
        avg_time = np.mean(list(alphas.values()))
        print(f"  {config}: {avg_time:.4f}s average")
```

## Best Practices

### 1. Choosing the Right Approach

- **Spectral engines**: Use for large sequences (>1000 points) or when high accuracy is needed
- **Stochastic sampling**: Use for real-time applications or when memory is limited
- **Probabilistic orders**: Use when uncertainty quantification is important

### 2. Performance Optimization

- **GPU acceleration**: Always use GPU for large computations
- **AMP**: Enable for 2x speedup with minimal accuracy loss
- **Chunked FFT**: Use for sequences >4096 points
- **Batch processing**: Process multiple sequences together

### 3. Variance Control

- **Monitor variance**: Use variance-aware training for stochastic methods
- **Adaptive sampling**: Adjust K based on variance levels
- **Seed management**: Use reproducible seeds for debugging

### 4. Memory Management

- **Chunked processing**: Use for large sequences
- **Gradient checkpointing**: Enable for memory-constrained training
- **Mixed precision**: Use AMP to reduce memory usage

### 5. Debugging Tips

- **Start simple**: Begin with spectral engines before adding stochastic/probabilistic components
- **Check gradients**: Verify gradient flow with `torch.autograd.gradcheck`
- **Monitor variance**: Watch for high variance in stochastic methods
- **Profile performance**: Use GPU profiler to identify bottlenecks

## Troubleshooting

### Common Issues

1. **High variance in stochastic sampling**
   - Increase K (number of samples)
   - Use stratified or control variate sampling
   - Check if alpha is too extreme

2. **Memory issues with large sequences**
   - Use chunked FFT
   - Enable gradient checkpointing
   - Reduce batch size

3. **Slow performance**
   - Enable GPU acceleration
   - Use AMP (Automatic Mixed Precision)
   - Profile to identify bottlenecks

4. **Gradient issues**
   - Check if tensors require gradients
   - Verify autograd function implementation
   - Use `torch.autograd.gradcheck`

### Getting Help

- Check the examples in `examples/` directory
- Run the test scripts to verify functionality
- Use the benchmark scripts to measure performance
- Consult the API documentation for detailed parameter descriptions

## Conclusion

The fractional autograd framework provides powerful tools for incorporating fractional calculus into neural networks. By combining spectral domain computation, stochastic memory sampling, and probabilistic fractional orders, you can build sophisticated models that capture the non-local and memory-dependent nature of fractional systems.

Start with the basic examples and gradually incorporate more advanced features as needed. The framework is designed to be modular, so you can mix and match different approaches based on your specific requirements.

