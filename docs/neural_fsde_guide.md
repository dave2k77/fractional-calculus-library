# Neural Fractional SDE Guide

## Introduction to Neural Fractional SDEs

### What are Neural Fractional SDEs?

Neural Fractional Stochastic Differential Equations (Neural fSDEs) combine the power of neural networks with fractional calculus and stochastic dynamics. They extend neural ODEs by incorporating:

1. **Stochasticity**: Random noise terms for modeling uncertainty
2. **Memory effects**: Fractional derivatives capture long-range temporal dependencies
3. **Learnable dynamics**: Neural networks parameterize drift and diffusion functions

### Mathematical Foundation

A neural fractional SDE takes the form:

$$
D_t^\alpha X(t) = f_\theta(t, X(t)) dt + g_\theta(t, X(t)) dW(t)
$$

where:
- $\alpha \in (0, 2)$ is the fractional order
- $D_t^\alpha$ is the Caputo or Riemann-Liouville fractional derivative
- $f_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d}$ is the learnable drift function (neural network)
- $g_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d \times m}$ is the learnable diffusion function
- $W(t)$ is a Wiener process (or more general noise)

### When to Use Neural fSDEs

**Ideal for:**
- Systems with memory effects (viscoelastic materials, anomalous diffusion)
- Data with stochastic uncertainty
- Continuous-time dynamics with long-range dependence
- Spatio-temporal systems (graph-SDE coupling)
- When uncertainty quantification is crucial

**Compared to Neural ODEs:**
- Neural ODEs: Deterministic, integer-order derivatives, limited memory
- Neural fSDEs: Stochastic, fractional-order, long memory, uncertainty

## Quick Start

### Installation

```bash
pip install hpfracc
# Optional for Bayesian inference
pip install numpyro
```

### Basic Example

```python
import numpy as np
import torch
from hpfracc.ml.neural_fsde import create_neural_fsde

# Create a simple neural fSDE
model = create_neural_fsde(
    input_dim=2,
    output_dim=2,
    hidden_dim=64,
    fractional_order=0.5,
    noise_type="additive"
)

# Forward pass
x0 = torch.randn(32, 2)
t = torch.linspace(0, 1, 50)
trajectory = model(x0, t, method="euler_maruyama", num_steps=50)

print(f"Trajectory shape: {trajectory.shape}")  # (32, 2)
```

### Training Example

```python
import torch.nn as nn

model = create_neural_fsde(input_dim=2, output_dim=2, fractional_order=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x0, t)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Core Concepts

### Fractional Orders in SDEs

The fractional order $\alpha$ controls memory effects:

- **$\alpha \to 0$**: Nearly instantaneous response (no memory)
- **$\alpha = 0.5$**: Subdiffusion (slower than normal diffusion)
- **$\alpha = 1$**: Standard first-order dynamics
- **$\alpha = 1.5$**: Superdiffusion (faster than normal)
- **$\alpha \to 2$**: Wave-like behavior

```python
# Compare different fractional orders
for alpha in [0.3, 0.5, 0.7, 1.0]:
    model = create_neural_fsde(
        input_dim=2,
        output_dim=2,
        fractional_order=alpha
    )
    # Different models exhibit different memory behavior
```

### Drift and Diffusion Functions

**Drift $f_\theta$**: Determines deterministic dynamics
```python
# Custom drift network
drift_net = nn.Sequential(
    nn.Linear(3, 64),  # 2 features + time
    nn.Tanh(),
    nn.Linear(64, 2)
)

model = create_neural_fsde(
    input_dim=2,
    output_dim=2,
    drift_net=drift_net  # Use custom network
)
```

**Diffusion $g_\theta$**: Controls stochastic noise magnitude
```python
# Custom diffusion network
diffusion_net = nn.Sequential(
    nn.Linear(3, 64),
    nn.Tanh(),
    nn.Linear(64, 2)
)

model = create_neural_fsde(
    input_dim=2,
    output_dim=2,
    diffusion_net=diffusion_net
)
```

### Stochastic Noise Modeling

Choose noise type based on problem:

```python
from hpfracc.solvers import BrownianMotion, FractionalBrownianMotion

# Standard Brownian motion (independent increments)
brownian = BrownianMotion(scale=1.0)

# Fractional Brownian motion (correlated increments, Hurst H)
fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)

# Coloured noise (exponential autocorrelation)
from hpfracc.solvers import ColouredNoise
coloured = ColouredNoise(correlation_time=1.0, amplitude=1.0)
```

### Memory Effects

Fractional derivatives introduce memory through convolution:

$$
D_t^\alpha x(t) = \int_0^t \frac{(t-\tau)^{-\alpha-1}}{\Gamma(-\alpha)} x(\tau) d\tau
$$

This requires storing history, but FFT-based computation gives $O(N \log N)$ complexity.

## Building Neural fSDE Models

### Model Architecture Design

```python
from hpfracc.ml.neural_fsde import NeuralFSDEConfig

# Configure model
config = NeuralFSDEConfig(
    input_dim=10,
    output_dim=10,
    hidden_dim=128,
    num_layers=4,
    fractional_order=0.5,
    diffusion_dim=1,
    noise_type="multiplicative",  # or "additive"
    learn_alpha=True  # Make fractional order learnable
)

model = create_neural_fsde(
    input_dim=config.input_dim,
    output_dim=config.output_dim,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    fractional_order=config.fractional_order,
    diffusion_dim=config.diffusion_dim,
    noise_type=config.noise_type,
    learn_alpha=config.learn_alpha
)
```

### Network Configuration

**Drift Network Tips:**
- Include time as input: concatenate `[t, x]`
- Use smooth activations (Tanh, Swish) for drift
- Avoid ReLU for drift (creates discontinuities)

**Diffusion Network Tips:**
- Use positive activations (Softplus, Sigmoid)
- Scale output appropriately for your data
- Consider log-scale for diffusion

```python
# Example: Positive diffusion network
def create_diffusion_net(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim + 1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, output_dim),
        nn.Softplus()  # Ensure positivity
    )
```

### Learnable Parameters

```python
# Fixed fractional order
model_fixed = create_neural_fsde(..., learn_alpha=False)

# Learnable fractional order
model_learnable = create_neural_fsde(..., learn_alpha=True)

# Access learned order
alpha = model_learnable.get_fractional_order()
print(f"Current fractional order: {alpha}")
```

### Backend Selection

The intelligent backend selector automatically chooses optimal computation:

```python
from hpfracc.ml.intelligent_backend_selector import select_optimal_backend

# Automatic selection
backend = select_optimal_backend(
    operation_type="neural_sde",
    data_shape=(1000, 10),
    requires_gradient=True
)
```

## Training with Adjoint Methods

### Why Adjoint Methods?

Standard backpropagation through SDEs stores full trajectory (memory-intensive). Adjoint methods solve the adjoint equation backwards.

### Basic Adjoint Training

```python
from hpfracc.ml.adjoint_optimization import AdjointConfig

config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=10
)

model = create_neural_fsde(
    ...,
    use_adjoint=config.use_adjoint
)
```

### Memory-Efficient Checkpointing

```python
from hpfracc.ml.sde_adjoint_utils import (
    SDEAdjointOptimizer, CheckpointConfig
)

checkpoint_config = CheckpointConfig(
    checkpoint_frequency=10,
    checkpoint_strategy="adaptive",
    max_checkpoints=100
)

optimizer = torch.optim.Adam(model.parameters())
sde_optimizer = SDEAdjointOptimizer(
    model, optimizer,
    checkpoint_config=checkpoint_config
)
```

### Mixed Precision Training

```python
from hpfracc.ml.sde_adjoint_utils import MixedPrecisionConfig

mixed_precision_config = MixedPrecisionConfig(
    enable_amp=True,  # Automatic Mixed Precision
    half_precision=False,
    loss_scaling=1.0
)

sde_optimizer = SDEAdjointOptimizer(
    model, optimizer,
    mixed_precision_config=mixed_precision_config
)
```

### Gradient Accumulation

For large models or limited memory:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = compute_loss(model, batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Graph-SDE Coupling

### Spatio-Temporal Dynamics

Combine graph neural networks with temporal SDE dynamics:

```python
from hpfracc.ml.graph_sde_coupling import GraphFractionalSDELayer

layer = GraphFractionalSDELayer(
    input_dim=10,
    output_dim=10,
    fractional_order=0.5,
    coupling_type="bidirectional",
    use_gated_coupling=True
)

# Apply to graph
features = torch.randn(32, 10)  # Node features
adjacency = torch.ones(32, 32)  # Graph edges
output = layer(features, adjacency)
```

### Coupling Mechanisms

1. **Bidirectional**: Information flows both ways
2. **Gated**: Learnable gating controls coupling strength
3. **Attention-based**: Self-attention for coupling

### Multi-Scale Modeling

```python
from hpfracc.ml.graph_sde_coupling import MultiScaleGraphSDE

model = MultiScaleGraphSDE(
    node_dim=10,
    num_scales=3,
    fractional_orders=[0.3, 0.5, 0.7]
)
```

### Operator Splitting

For weakly coupled systems:

```python
from hpfracc.solvers import OperatorSplittingSolver

solver = OperatorSplittingSolver(
    fractional_order=0.5,
    splitting_method="strang"  # Strang splitting
)
```

## Uncertainty Quantification

### Bayesian Neural fSDEs

```python
from hpfracc.ml.probabilistic_sde import create_bayesian_fsde

model = create_bayesian_fsde(
    input_dim=2,
    output_dim=2,
    hidden_dim=64,
    fractional_order=0.5,
    use_guide=True
)
```

### NumPyro Integration

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

# Define guide (variational posterior)
guide = model.create_guide()

# Stochastic Variational Inference
svi = SVI(
    model.model,
    guide,
    numpyro.optim.Adam(step_size=1e-3),
    loss=Trace_ELBO()
)

# Training loop
for epoch in range(1000):
    elbo = svi.step(x0, t, observations)
```

### Posterior Predictive Distributions

```python
# Generate samples from posterior
predictive = numpyro.infer.Predictive(model.model, guide=guide, num_samples=1000)
samples = predictive(rng_key, x0, t)

# Compute statistics
mean = samples.mean(axis=0)
std = samples.std(axis=0)
```

### Confidence Intervals

```python
# 95% confidence interval
lower = np.percentile(samples, 2.5, axis=0)
upper = np.percentile(samples, 97.5, axis=0)
```

## Performance Optimization

### Intelligent Backend Selection

```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector

selector = IntelligentBackendSelector(enable_learning=True)

# It learns optimal backend over time
backend = selector.select_backend(workload)
```

### Memory Management

1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision**: Use float16 for activations
3. **Sparse Gradients**: Store only non-zero gradients

```python
from hpfracc.ml.sde_adjoint_utils import SparseGradientAccumulator

accumulator = SparseGradientAccumulator(sparsity_threshold=1e-6)

for param in model.parameters():
    if param.grad is not None:
        accumulator.accumulate(param.grad)
```

### GPU Utilization

```python
# Move model to GPU
model = model.cuda()

# Use DataParallel for multiple GPUs
model = nn.DataParallel(model)
```

### Batch Processing

Process multiple trajectories efficiently:

```python
# Batch of 100 initial conditions
x0_batch = torch.randn(100, 32, 2)
trajectories = model(x0_batch, t)
```

## Best Practices

### Model Selection

1. **Start simple**: Begin with additive noise
2. **Add complexity gradually**: Introduce multiplicative noise if needed
3. **Validate architecture**: Use validation set to tune architecture

### Hyperparameter Tuning

**Critical parameters:**
- Fractional order $\alpha$: Try 0.3, 0.5, 0.7
- Hidden dimension: 32, 64, 128
- Number of layers: 2-4
- Learning rate: 1e-4 to 1e-2

```python
# Grid search example
for alpha in [0.3, 0.5, 0.7]:
    for hidden_dim in [32, 64, 128]:
        model = create_neural_fsde(
            fractional_order=alpha,
            hidden_dim=hidden_dim
        )
        # Train and evaluate
```

### Numerical Stability

1. **Clip gradients**: Prevent exploding gradients
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

2. **Use stable activations**: Tanh, Swish over ReLU for drift
3. **Check for NaNs**: Monitor during training
```python
if torch.isnan(loss):
    print("NaN detected!")
```

### Validation Strategies

```python
# Train-validation split
train_size = int(0.8 * len(data))
train_data, val_data = torch.utils.data.random_split(data, [train_size, len(data) - train_size])

# Early stopping
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_data)
    val_loss = validate(model, val_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improve += 1
        if no_improve >= patience:
            break
```

### Debugging Tips

1. **Visualize trajectories**: Plot $\partial_t X(t)$
2. **Check noise magnitude**: Scale diffusion appropriately
3. **Monitor memory**: Use `torch.cuda.memory_allocated()`
4. **Profile code**: Use `torch.profiler` for bottlenecks

```python
# Profile example
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)
) as prof:
    output = model(x0, t)
    
prof.export_chrome_trace('trace.json')
```

## Advanced Topics

### Learnable Fractional Orders

Make $\alpha$ a learnable parameter:

```python
model = create_neural_fsde(..., learn_alpha=True)

# During training, alpha updates automatically
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Access learned order
alpha = model.get_fractional_order()
print(f"Learned order: {alpha}")  # e.g., 0.523
```

### Multi-Equation Systems

Different fractional orders per equation:

```python
solution = solve_fractional_sde_system(
    drift, diffusion, x0,
    t_span=(0, 1),
    fractional_order=[0.3, 0.5, 0.7],  # Different orders
    method="euler_maruyama"
)
```

### Custom Noise Processes

Implement your own noise:

```python
from hpfracc.solvers import NoiseModel

class CustomNoise(NoiseModel):
    def generate_increment(self, t, dt, size, seed=None):
        # Your custom noise generation
        return custom_noise
```

## References

- [Neural ODEs](https://arxiv.org/abs/1806.07366)
- [Neural SDEs](https://arxiv.org/abs/1905.09883)
- [Fractional Calculus](https://www.elsevier.com/books/handbook-of-fractional-calculus-with-applications/kochubei/978-0-12-817856-1)

## Getting Help

- **Documentation**: [ReadTheDocs](https://hpfracc.readthedocs.io/)
- **Examples**: `examples/neural_fsde_examples/`
- **Issues**: [GitHub Issues](https://github.com/dave2k77/fractional-calculus-library/issues)

