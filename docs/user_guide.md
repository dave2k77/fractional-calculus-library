# HPFRACC User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Fractional Calculus](#basic-fractional-calculus)
4. [Machine Learning Integration](#machine-learning-integration)
5. [Performance Optimization](#performance-optimization)
6. [Production Workflow](#production-workflow)
7. [Benchmarking](#benchmarking)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install from Source
```bash
git clone https://github.com/your-username/hpfracc.git
cd hpfracc
pip install -e .
```

### Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib seaborn
pip install psutil optuna scikit-learn
```

---

## Quick Start

### Basic Fractional Derivative
```python
import torch
from hpfracc.core import fractional_derivative

# Create input tensor
x = torch.randn(100, 50)

# Compute fractional derivative
result = fractional_derivative(x, alpha=0.5, method="RL")
print(f"Input shape: {x.shape}")
print(f"Output shape: {result.shape}")
```

### Simple Fractional Neural Network
```python
from hpfracc.ml import FractionalNeuralNetwork, FractionalAdam

# Create network
net = FractionalNeuralNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5
)

# Create optimizer
optimizer = FractionalAdam(net.parameters(), lr=0.001)

# Forward pass
x = torch.randn(32, 100)  # batch_size=32, input_size=100
output = net(x)
print(f"Output shape: {output.shape}")
```

---

## Basic Fractional Calculus

### Understanding Fractional Derivatives

Fractional derivatives extend the concept of integer-order derivatives to non-integer orders. The library supports several methods:

- **Riemann-Liouville (RL)**: Most general, works for 0 < α < 2
- **Caputo**: Better for initial value problems, works for 0 < α < 1
- **Grünwald-Letnikov (GL)**: Numerical approximation, works for 0 < α < 2
- **Weyl**: For periodic functions
- **Marchaud**: For functions with specific decay properties
- **Hadamard**: Logarithmic fractional derivative

### Choosing the Right Method

```python
from hpfracc.core import fractional_derivative

# For general purposes
result_rl = fractional_derivative(x, alpha=0.5, method="RL")

# For initial value problems (e.g., differential equations)
result_caputo = fractional_derivative(x, alpha=0.3, method="Caputo")

# For numerical stability
result_gl = fractional_derivative(x, alpha=0.7, method="GL")
```

### Fractional Order Validation

```python
from hpfracc.core.definitions import FractionalOrder

# Create and validate fractional order
alpha = FractionalOrder(0.5)
print(f"Alpha: {alpha.alpha}")
print(f"Is valid: {alpha.is_valid}")

# Invalid order will raise error
try:
    invalid_alpha = FractionalOrder(2.5)  # Out of range for most methods
except ValueError as e:
    print(f"Error: {e}")
```

---

## Machine Learning Integration

### Creating Fractional Neural Networks

#### Standard Network
```python
from hpfracc.ml import FractionalNeuralNetwork

net = FractionalNeuralNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5
)

# The network automatically applies fractional derivatives
# to inputs and intermediate activations
```

#### Memory-Efficient Adjoint Network
```python
from hpfracc.ml.adjoint_optimization import (
    MemoryEfficientFractionalNetwork, 
    AdjointConfig
)

# Configure adjoint optimization
adjoint_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=5,
    gradient_accumulation=True,
    accumulation_steps=4
)

# Create optimized network
net = MemoryEfficientFractionalNetwork(
    input_size=100,
    hidden_sizes=[512, 256, 128, 64],
    output_size=10,
    fractional_order=0.5,
    adjoint_config=adjoint_config
)
```

### Training with Fractional Optimizers

```python
import torch.nn as nn
from hpfracc.ml import FractionalAdam, FractionalMSELoss

# Loss function with fractional derivatives
loss_fn = FractionalMSELoss(fractional_order=0.5, method="RL")

# Optimizer with fractional gradient updates
optimizer = FractionalAdam(
    net.parameters(),
    lr=0.001,
    fractional_order=0.5,
    method="RL"
)

# Training loop
net.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    output = net(x)
    loss = loss_fn(output, target)
    
    # Backward pass
    loss.backward()
    
    # Update with fractional gradients
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using Fractional Layers

#### Convolutional Layers
```python
from hpfracc.ml.layers import FractionalConv1D, FractionalConv2D, LayerConfig
from hpfracc.core.definitions import FractionalOrder

# Configure layer
config = LayerConfig(
    fractional_order=FractionalOrder(0.5),
    method="RL",
    use_fractional=True,
    activation="relu",
    dropout=0.1
)

# 1D convolution
conv1d = FractionalConv1D(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    config=config
)

# 2D convolution
conv2d = FractionalConv2D(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    config=config
)
```

#### Transformer Layer
```python
from hpfracc.ml.layers import FractionalTransformer

transformer = FractionalTransformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    config=config
)

# Encoder-only mode (single input)
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
output = transformer(x)  # No target needed

# Full transformer mode (encoder-decoder)
src = torch.randn(32, 100, 512)
tgt = torch.randn(32, 50, 512)
output = transformer(src, tgt)
```

---

## Performance Optimization

### Adjoint Method Benefits

The adjoint method provides significant performance improvements:

- **Training Speed**: Up to 19.7x faster training
- **Memory Usage**: Up to 81% reduction in memory consumption
- **Scalability**: Better performance on large models

### Configuration Options

```python
from hpfracc.ml.adjoint_optimization import AdjointConfig

# Memory optimization
memory_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=5
)

# Gradient accumulation
accumulation_config = AdjointConfig(
    use_adjoint=True,
    gradient_accumulation=True,
    accumulation_steps=8
)

# Balanced configuration
balanced_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=3,
    gradient_accumulation=True,
    accumulation_steps=4
)
```

### When to Use Adjoint Optimization

- **Use adjoint optimization when**:
  - Training large models (>100M parameters)
  - Limited GPU memory
  - Need faster training times
  - Working with long sequences

- **Stick with standard methods when**:
  - Small models (<10M parameters)
  - Sufficient memory available
  - Need maximum numerical precision
  - Debugging or prototyping

---

## Production Workflow

### Model Registry Setup

```python
from hpfracc.ml import ModelRegistry, ModelValidator
from hpfracc.ml.workflow import DevelopmentWorkflow, ProductionWorkflow

# Initialize components
registry = ModelRegistry()
validator = ModelValidator()
dev_workflow = DevelopmentWorkflow(registry, validator)
prod_workflow = ProductionWorkflow(registry, validator)
```

### Development Phase

```python
# Train and validate model
validation_results = dev_workflow.train_model(
    model=net,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100
)

# Run quality gates
quality_result = dev_workflow.run_quality_gates(
    model_id=model_id,
    validation_results=validation_results
)

if quality_result["passed"]:
    print("Model passed quality gates!")
else:
    print(f"Quality gate failed: {quality_result['reason']}")
```

### Production Deployment

```python
# Promote to production
promotion_result = prod_workflow.promote_to_production(
    model_id=model_id,
    version="1.0.0",
    test_data=X_test,
    test_labels=y_test,
    custom_metrics={},
    force=False
)

if promotion_result["promoted"]:
    print("Model promoted to production!")
    
    # Deploy model
    deployment_result = prod_workflow.deploy_model(
        model_id=model_id,
        version="1.0.0",
        deployment_config={"environment": "production"}
    )
else:
    print(f"Promotion failed: {promotion_result['reason']}")
```

### Monitoring Production Models

```python
# Monitor model performance
prod_workflow.monitor_model(
    model_id=model_id,
    metrics={
        "accuracy": 0.95,
        "latency": 0.1,
        "throughput": 1000
    }
)

# Get production model
production_model = registry.reconstruct_model(model_id, "1.0.0")
production_model.eval()

# Run inference
with torch.no_grad():
    predictions = production_model(X_new)
```

---

## Benchmarking

### Performance Benchmarking

```python
from hpfracc.benchmarks import MLPerformanceBenchmark

# Initialize benchmark
benchmark = MLPerformanceBenchmark(
    device="cuda",
    num_runs=10,
    warmup_runs=3
)

# Benchmark neural networks
network_results = benchmark.benchmark_fractional_networks(
    input_sizes=[50, 100, 200],
    hidden_sizes_list=[[128, 64], [256, 128, 64]],
    fractional_orders=[0.1, 0.5, 0.9],
    methods=["RL", "Caputo"]
)

# Benchmark attention mechanisms
attention_results = benchmark.benchmark_fractional_attention(
    batch_sizes=[16, 32, 64],
    seq_lengths=[100, 200],
    d_models=[256, 512],
    fractional_orders=[0.1, 0.5, 0.9],
    methods=["RL", "Caputo"]
)

# Generate comprehensive report
benchmark.generate_report("benchmark_results")
```

### Interpreting Results

The benchmark generates several metrics:

- **Execution Time**: Wall-clock time for operations
- **Memory Usage**: Peak memory consumption
- **Throughput**: Samples processed per second
- **Speedup**: Performance improvement over baseline

### Performance Comparison

```python
# Compare standard vs. adjoint methods
standard_time = network_results["standard"]["execution_time"]
adjoint_time = network_results["adjoint"]["execution_time"]

speedup = standard_time / adjoint_time
print(f"Adjoint method is {speedup:.1f}x faster")

memory_reduction = (
    (network_results["standard"]["memory_usage"] - 
     network_results["adjoint"]["memory_usage"]) / 
    network_results["standard"]["memory_usage"] * 100
)
print(f"Memory usage reduced by {memory_reduction:.1f}%")
```

---

## Troubleshooting

### Common Issues

#### 1. Fractional Order Range Errors
```python
# Error: "L1 scheme requires 0 < α < 1"
# Solution: Use valid range for Caputo method
result = fractional_derivative(x, alpha=0.5, method="Caputo")  # Valid
# result = fractional_derivative(x, alpha=1.0, method="Caputo")  # Invalid
```

#### 2. Tensor Shape Mismatches
```python
# Error: "mat1 and mat2 shapes cannot be multiplied"
# Solution: Ensure input dimensions match network architecture
net = FractionalNeuralNetwork(input_size=100, ...)
x = torch.randn(32, 100)  # batch_size=32, input_size=100
output = net(x)
```

#### 3. Memory Issues
```python
# Error: "CUDA out of memory"
# Solution: Use adjoint optimization
adjoint_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=3
)
net = MemoryEfficientFractionalNetwork(..., adjoint_config=adjoint_config)
```

#### 4. Optimizer Not Updating Parameters
```python
# Issue: Parameters not updating during training
# Solution: Ensure proper gradient flow
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check gradients
for param in net.parameters():
    if param.grad is not None:
        print(f"Gradient norm: {param.grad.norm()}")
```

### Debugging Tips

1. **Check Tensor Shapes**: Print shapes at each step
2. **Verify Gradients**: Ensure gradients are computed and non-zero
3. **Monitor Memory**: Use `torch.cuda.memory_summary()` for GPU memory
4. **Test Components**: Test individual layers before full network
5. **Use Small Data**: Start with small datasets for debugging

### Getting Help

- **Documentation**: Check the API reference for detailed information
- **Examples**: Review the examples directory for working code
- **Issues**: Report bugs on the GitHub repository
- **Community**: Join discussions in the project forum

---

## Best Practices

### Code Organization
1. **Separate Concerns**: Keep data loading, model definition, and training separate
2. **Configuration Files**: Use configuration files for hyperparameters
3. **Logging**: Implement proper logging for experiments
4. **Version Control**: Use git for model and code versioning

### Performance
1. **Profile First**: Benchmark before optimization
2. **Use Adjoint**: Enable adjoint optimization for large models
3. **Batch Processing**: Use appropriate batch sizes
4. **Memory Management**: Monitor and optimize memory usage

### Production
1. **Quality Gates**: Always run quality gates before deployment
2. **Monitoring**: Implement continuous monitoring
3. **Rollback Plan**: Have rollback strategies ready
4. **Documentation**: Document all production models

---

## Next Steps

After mastering the basics:

1. **Advanced Architectures**: Experiment with custom fractional layers
2. **Research**: Explore novel fractional derivative methods
3. **Applications**: Apply to your specific domain problems
4. **Contributions**: Contribute to the library development

The HPFRACC library provides a solid foundation for fractional calculus in machine learning. Start simple, experiment, and gradually explore more advanced features as you become comfortable with the basics.
