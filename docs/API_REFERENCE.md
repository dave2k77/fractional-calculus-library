# HPFRACC API Reference

## Core API

### Main Package
```python
import hpfracc

# Version information
hpfracc.__version__  # "2.2.0"
hpfracc.__author__    # "Davian R. Chin"
hpfracc.__email__    # "d.r.chin@pgr.reading.ac.uk"
```

### Core Fractional Calculus

#### Fractional Derivatives
```python
from hpfracc.core import create_fractional_derivative, create_derivative_operator

# Create fractional derivative operators
caputo_deriv = create_fractional_derivative(alpha=0.5, definition="caputo")
rl_deriv = create_fractional_derivative(alpha=0.5, definition="riemann_liouville")
gl_deriv = create_fractional_derivative(alpha=0.5, definition="grunwald_letnikov")

# Use with functions
import numpy as np
def f(x):
    return np.sin(x)

x = np.linspace(0, 2*np.pi, 100)
result = caputo_deriv(f, x)
```

#### Fractional Integrals
```python
from hpfracc.core import create_fractional_integral, RiemannLiouvilleIntegral, CaputoIntegral

# Create fractional integral operators
rl_integral = create_fractional_integral(alpha=0.5, definition="riemann_liouville")
caputo_integral = create_fractional_integral(alpha=0.5, definition="caputo")

# Use with functions
result = rl_integral(f, x)
```

#### Advanced Derivatives
```python
from hpfracc.algorithms import (
    WeylDerivative, MarchaudDerivative, HadamardDerivative,
    ReizFellerDerivative, FractionalLaplacian
)

# Advanced fractional derivatives
weyl = WeylDerivative(alpha=0.5)
marchaud = MarchaudDerivative(alpha=0.5)
hadamard = HadamardDerivative(alpha=0.5)
reiz_feller = ReizFellerDerivative(alpha=0.5, beta=0.3)

# Fractional Laplacian
frac_laplacian = FractionalLaplacian(alpha=0.5)
result = frac_laplacian.compute(f, x)
```

#### Special Functions
```python
from hpfracc.special import (
    mittag_leffler, gamma_function, beta_function,
    binomial_coefficient, pochhammer_symbol
)

# Mittag-Leffler function
ml_result = mittag_leffler(alpha=0.5, beta=1.0, z=x)

# Gamma and Beta functions
gamma_val = gamma_function(0.5)
beta_val = beta_function(0.5, 0.3)
```

## Machine Learning API

### Intelligent Backend Selection

#### Quick Selection
```python
from hpfracc.ml.intelligent_backend_selector import select_optimal_backend
from hpfracc.ml.backends import BackendType

# Quick backend selection
backend = select_optimal_backend(
    operation_type="matmul",
    data_shape=(1000, 1000),
    requires_gradient=True
)
print(f"Selected backend: {backend}")
```

#### Advanced Selection with Learning
```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector, WorkloadCharacteristics

# Create selector with learning enabled
selector = IntelligentBackendSelector(enable_learning=True)

# Define workload characteristics
workload = WorkloadCharacteristics(
    operation_type="fractional_derivative",
    data_size=10000,
    data_shape=(100, 100),
    requires_gradient=True
)

# Select optimal backend
backend = selector.select_backend(workload)

# Monitor performance
selector.record_performance(
    backend=backend,
    operation="fractional_derivative",
    data_size=10000,
    execution_time=0.1,
    success=True
)
```

### Neural Network Layers

#### Basic Fractional Layers
```python
import torch
from hpfracc.ml.layers import FractionalLayer, FractionalConv1D, FractionalConv2D

# Basic fractional layer (automatic backend selection)
layer = FractionalLayer(alpha=0.5, input_dim=10, output_dim=5)
input_data = torch.randn(32, 10)
output = layer(input_data)

# Fractional convolutional layers
conv1d = FractionalConv1D(in_channels=3, out_channels=16, kernel_size=3, alpha=0.5)
conv2d = FractionalConv2D(in_channels=3, out_channels=16, kernel_size=3, alpha=0.5)
```

#### Advanced Neural Networks
```python
from hpfracc.ml.core import FractionalNeuralNetwork, FractionalAttention

# Fractional neural network
fnn = FractionalNeuralNetwork(
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=1,
    fractional_order=0.5,
    activation="relu"
)

# Fractional attention mechanism
attention = FractionalAttention(
    d_model=512,
    n_heads=8,
    fractional_order=0.5
)
```

### Optimizers
```python
from hpfracc.ml.optimized_optimizers import (
    OptimizedFractionalAdam, OptimizedFractionalSGD, OptimizedFractionalRMSprop
)

# Fractional Adam optimizer
optimizer = OptimizedFractionalAdam(
    params=model.parameters(),
    lr=0.001,
    fractional_order=0.5,
    use_fractional=True
)
```

### Loss Functions
```python
from hpfracc.ml.losses import (
    FractionalMSELoss, FractionalCrossEntropyLoss, FractionalHuberLoss
)

# Fractional loss functions
mse_loss = FractionalMSELoss(fractional_order=0.5)
ce_loss = FractionalCrossEntropyLoss(fractional_order=0.5)
huber_loss = FractionalHuberLoss(fractional_order=0.5, delta=1.0)
```

## Solver API

### Fractional ODE Solvers
```python
from hpfracc.solvers import FixedStepODESolver, solve_fractional_ode

# Define right-hand side function
def rhs(t, y, alpha):
    return -y**alpha

# Solve fractional ODE
solver = FixedStepODESolver(
    derivative_type="caputo",
    method="predictor_corrector"
)

t, y = solver.solve(
    rhs, 
    t_span=(0.0, 5.0), 
    y0=1.0, 
    alpha=0.5, 
    h=0.01
)

# Or use convenience function
t, y = solve_fractional_ode(
    rhs, 
    t_span=(0.0, 5.0), 
    y0=1.0, 
    alpha=0.5, 
    h=0.01,
    derivative_type="caputo"
)
```

### Fractional PDE Solvers
```python
from hpfracc.solvers import (
    FractionalPDESolver, FractionalDiffusionSolver, solve_fractional_pde
)

# Define PDE parameters
def initial_condition(x):
    return np.exp(-x**2)

def boundary_conditions(t):
    return 0.0, 0.0

# Solve fractional diffusion equation
solver = FractionalDiffusionSolver(
    alpha=0.5,
    diffusion_coefficient=1.0,
    domain=(0, 1),
    nx=100
)

solution = solver.solve(
    initial_condition=initial_condition,
    boundary_conditions=boundary_conditions,
    t_span=(0, 1),
    nt=100
)
```

## Neural Fractional SDE Solvers API

### Fractional SDE Solvers
```python
from hpfracc.solvers import (
    FractionalSDESolver, FractionalEulerMaruyama, FractionalMilstein,
    solve_fractional_sde, solve_fractional_sde_system, SDESolution
)

# Solve a fractional SDE using Euler-Maruyama method
def drift(t, x):
    return -0.5 * x

def diffusion(t, x):
    return 0.2 * np.ones_like(x)

x0 = np.array([1.0])
solution = solve_fractional_sde(
    drift, diffusion, x0, 
    t_span=(0, 1), 
    fractional_order=0.5,
    method="euler_maruyama",
    num_steps=100
)

# Access solution
t = solution.t          # Time points
y = solution.y          # Trajectory
print(f"Final value: {y[-1]}")

# Use Milstein method for higher accuracy
solution_milstein = solve_fractional_sde(
    drift, diffusion, x0,
    t_span=(0, 1),
    fractional_order=0.5,
    method="milstein",
    num_steps=100
)

# Solve coupled system of fractional SDEs
solution_system = solve_fractional_sde_system(
    drift, diffusion, x0,
    t_span=(0, 1),
    fractional_order=[0.5, 0.7],  # Different orders per equation
    method="euler_maruyama"
)
```

### Stochastic Noise Models
```python
from hpfracc.solvers import (
    BrownianMotion, FractionalBrownianMotion, LevyNoise, ColouredNoise,
    NoiseConfig, create_noise_model
)

# Standard Brownian motion
brownian = BrownianMotion(scale=1.0)
dW = brownian.generate_increment(t=0.0, dt=0.01, size=(100,))

# Fractional Brownian motion with Hurst exponent H=0.7
fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)
dW_fbm = fbm.generate_increment(t=0.0, dt=0.01, size=(100,))

# Lévy noise for jump processes
levy = LevyNoise(alpha=1.5, beta=0.0, scale=1.0)
dW_levy = levy.generate_increment(t=0.0, dt=0.01, size=(100,))

# Coloured noise (Ornstein-Uhlenbeck process)
coloured = ColouredNoise(correlation_time=1.0, amplitude=1.0)
dW_coloured = coloured.generate_increment(t=0.0, dt=0.01, size=(100,))

# Create noise model from configuration
config = NoiseConfig(
    noise_type="fractional_brownian",
    hurst=0.6,
    scale=1.0
)
noise = create_noise_model(config)
```

### Neural Fractional SDE Models
```python
from hpfracc.ml.neural_fsde import NeuralFractionalSDE, create_neural_fsde
import torch

# Create a neural fractional SDE
model = create_neural_fsde(
    input_dim=2,
    output_dim=2,
    hidden_dim=64,
    num_layers=3,
    fractional_order=0.5,
    diffusion_dim=1,
    noise_type="additive",
    learn_alpha=False,
    use_adjoint=True
)

# Forward pass
x0 = torch.randn(32, 2)  # Batch of initial conditions
t = torch.linspace(0, 1, 50)
trajectory = model(x0, t, method="euler_maruyama", num_steps=50)

# Training example
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x0, t)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
```

### SDE Adjoint Methods and Optimization
```python
from hpfracc.ml.adjoint_optimization import AdjointConfig, adjoint_sde_gradient
from hpfracc.ml.sde_adjoint_utils import (
    SDEAdjointOptimizer, CheckpointConfig, MixedPrecisionConfig
)

# Configure adjoint method
adjoint_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=10,
    sde_noise_type="itô",
    bsde_method="fd"
)

# Create optimized optimizer
checkpoint_config = CheckpointConfig(
    checkpoint_frequency=10,
    checkpoint_strategy="adaptive"
)

mixed_precision_config = MixedPrecisionConfig(
    enable_amp=True,
    half_precision=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
sde_optimizer = SDEAdjointOptimizer(
    model, optimizer,
    checkpoint_config=checkpoint_config,
    mixed_precision_config=mixed_precision_config,
    enable_sparse_gradients=True
)

# Training with optimized optimizer
for epoch in range(100):
    loss = loss_fn(model(x0, t), target)
    sde_optimizer.step(loss)
```

### SDE Loss Functions
```python
from hpfracc.ml.losses import (
    FractionalSDEMSELoss, FractionalKLDivergenceLoss,
    FractionalPathwiseLoss, FractionalMomentMatchingLoss
)

# MSE loss for SDE trajectory matching
mse_loss = FractionalSDEMSELoss(
    num_samples=10,  # Average over stochastic samples
    fractional_order=0.5
)

# KL divergence for distribution matching
kl_loss = FractionalKLDivergenceLoss(
    eps=1e-8,
    fractional_order=0.5
)

# Pathwise loss with uncertainty weighting
pathwise_loss = FractionalPathwiseLoss(
    uncertainty_weight=1.0,
    fractional_order=0.5
)

# Moment matching loss
moment_loss = FractionalMomentMatchingLoss(
    moments=[1, 2],  # Mean and variance
    weights=[1.0, 1.0],
    fractional_order=0.5
)

# Usage in training
loss = mse_loss(predictions, targets)
loss.backward()
```

### Graph-SDE Coupling
```python
from hpfracc.ml.graph_sde_coupling import (
    GraphFractionalSDELayer, SpatialTemporalCoupling, CouplingType
)
import torch.nn as nn

# Create graph-SDE layer
layer = GraphFractionalSDELayer(
    input_dim=10,
    output_dim=10,
    fractional_order=0.5,
    coupling_type="bidirectional",
    use_gated_coupling=True
)

# Forward pass with graph connectivity
graph_features = torch.randn(32, 10)  # Node features
adjacency = torch.ones(32, 32)  # Graph adjacency
output = layer(graph_features, adjacency)
```

### Coupled System Solvers
```python
from hpfracc.solvers import (
    OperatorSplittingSolver, MonolithicSolver, solve_coupled_graph_sde
)

# Operator splitting for coupled graph-SDE
splitting_solver = OperatorSplittingSolver(
    fractional_order=0.5,
    splitting_method="strang"
)

# Monolithic solver for strongly coupled systems
monolithic_solver = MonolithicSolver(
    fractional_order=[0.5, 0.7],
    coupling_strength=0.5
)

# High-level interface
solution = solve_coupled_graph_sde(
    spatial_dynamics=lambda x, adj: x @ adj,  # Graph convolution
    temporal_dynamics=lambda t, x: -x,  # Temporal evolution
    initial_condition=x0,
    adjacency_matrix=adj,
    t_span=(0, 1),
    num_steps=100
)
```

### Bayesian Neural Fractional SDE
```python
from hpfracc.ml.probabilistic_sde import (
    BayesianNeuralFractionalSDE, create_bayesian_fsde
)

# Create Bayesian neural fSDE
bayesian_model = create_bayesian_fsde(
    input_dim=2,
    output_dim=2,
    hidden_dim=64,
    fractional_order=0.5,
    use_guide=True
)

# Bayesian inference with NumPyro
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

# Define guide (variational posterior)
guide = bayesian_model.create_guide()

# Stochastic Variational Inference
svi = SVI(
    bayesian_model.model,
    guide,
    numpyro.optim.Adam(step_size=1e-3),
    loss=Trace_ELBO()
)

# Train
for epoch in range(1000):
    svi.step(x0, t, observations)
```

## Backend Management API

### Backend Control
```python
from hpfracc.ml.backends import (
    BackendType, BackendManager, get_backend_manager, switch_backend
)

# Get backend manager
manager = get_backend_manager()

# Switch backend
switch_backend(BackendType.TORCH)
switch_backend(BackendType.JAX)
switch_backend(BackendType.NUMBA)

# Check available backends
available = manager.available_backends
print(f"Available backends: {available}")
```

### Tensor Operations
```python
from hpfracc.ml.tensor_ops import get_tensor_ops

# Get unified tensor operations
tensor_ops = get_tensor_ops()

# Cross-backend tensor operations
a = tensor_ops.zeros((10, 10))
b = tensor_ops.ones((10, 10))
c = tensor_ops.matmul(a, b)
d = tensor_ops.transpose(c, dims=(1, 0))
```

## Performance Monitoring

### Analytics and Monitoring
```python
from hpfracc.analytics import PerformanceMonitor, UsageTracker

# Performance monitoring
monitor = PerformanceMonitor()
monitor.start_timing("operation_name")
# ... perform operation ...
execution_time = monitor.end_timing("operation_name")

# Usage tracking
tracker = UsageTracker()
tracker.record_usage("fractional_derivative", data_size=1000)
tracker.get_statistics()
```

## Configuration

### Environment Variables
```bash
# Backend selection
export HPFRACC_FORCE_JAX=1        # Force JAX backend
export HPFRACC_DISABLE_TORCH=1    # Disable PyTorch
export JAX_PLATFORM_NAME=cpu      # Force CPU mode

# Performance tuning
export HPFRACC_GPU_MEMORY_LIMIT=0.8  # GPU memory limit (80%)
export HPFRACC_ENABLE_LEARNING=1      # Enable performance learning
```

### Programmatic Configuration
```python
from hpfracc.ml.intelligent_backend_selector import IntelligentBackendSelector

# Configure intelligent selector
selector = IntelligentBackendSelector(
    enable_learning=True,
    gpu_memory_limit=0.8,
    performance_threshold=0.1
)

# Set global configuration
import os
os.environ['HPFRACC_FORCE_JAX'] = '1'
```

## Error Handling

### Custom Exceptions
```python
from hpfracc.core import (
    FractionalCalculusError, ConvergenceError, ValidationError
)

try:
    result = fractional_operation()
except ConvergenceError as e:
    print(f"Convergence failed: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except FractionalCalculusError as e:
    print(f"General error: {e}")
```

## Examples and Tutorials

### Basic Usage
```python
import hpfracc
import numpy as np

# Simple fractional derivative
frac_deriv = hpfracc.create_fractional_derivative(alpha=0.5, definition="caputo")
def f(x):
    return np.sin(x)

x = np.linspace(0, 2*np.pi, 100)
result = frac_deriv(f, x)
```

### Machine Learning Pipeline
```python
import torch
from hpfracc.ml.layers import FractionalLayer
from hpfracc.ml.optimized_optimizers import OptimizedFractionalAdam

# Create model with fractional layer
model = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    FractionalLayer(alpha=0.5, input_dim=64, output_dim=32),
    torch.nn.Linear(32, 1)
)

# Use fractional optimizer
optimizer = OptimizedFractionalAdam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Research Applications
```python
# Protein folding kinetics
from hpfracc.solvers import solve_fractional_ode

def protein_kinetics(t, y, alpha):
    return -y**alpha + 0.1 * np.sin(t)

t, y = solve_fractional_ode(
    protein_kinetics,
    t_span=(0, 10),
    y0=1.0,
    alpha=0.7,
    h=0.01
)
```

## Performance Tips

1. **Use Intelligent Backend Selection**: Let HPFRACC automatically choose the optimal backend
2. **Enable Learning**: Use `IntelligentBackendSelector(enable_learning=True)` for adaptive performance
3. **GPU Memory Management**: Set appropriate memory limits to avoid OOM errors
4. **Batch Processing**: Process multiple operations together when possible
5. **Profile Your Code**: Use `PerformanceMonitor` to identify bottlenecks

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **GPU Issues**: Check CUDA installation and memory availability
- **Performance**: Use intelligent backend selection for optimal performance
- **Convergence**: Adjust tolerance parameters for numerical methods

### Getting Help
- **Documentation**: [ReadTheDocs](https://hpfracc.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/dave2k77/fractional-calculus-library/issues)
- **Examples**: See `examples/` directory for comprehensive examples
