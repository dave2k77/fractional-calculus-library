# HPFRACC API Reference

## Table of Contents
1. [Core Module](#core-module)
2. [ML Module](#ml-module)
3. [Graph Neural Networks](#graph-neural-networks)
4. [Benchmarks Module](#benchmarks-module)
5. [Analytics Module](#analytics-module)

---

## Core Module

### Fractional Derivatives

#### `fractional_derivative(x, alpha, method="RL")`
Compute fractional derivative of input tensor.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `alpha` (float): Fractional order (0 < α < 2)
- `method` (str): Derivative method ("RL", "Caputo", "GL", "Weyl", "Marchaud", "Hadamard")

**Returns:**
- `torch.Tensor`: Fractional derivative result

**Example:**
```python
import torch
from hpfracc.core import fractional_derivative

x = torch.randn(100, 50)
result = fractional_derivative(x, alpha=0.5, method="RL")
```

#### `riemann_liouville_derivative(x, alpha)`
Compute Riemann-Liouville fractional derivative.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `alpha` (float): Fractional order (0 < α < 2)

**Returns:**
- `torch.Tensor`: Riemann-Liouville derivative

#### `caputo_derivative(x, alpha)`
Compute Caputo fractional derivative.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `alpha` (float): Fractional order (0 < α < 1)

**Returns:**
- `torch.Tensor`: Caputo derivative

#### `grunwald_letnikov_derivative(x, alpha)`
Compute Grünwald-Letnikov fractional derivative.

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `alpha` (float): Fractional order (0 < α < 2)

**Returns:**
- `torch.Tensor`: Grünwald-Letnikov derivative

### Fractional Order

#### `FractionalOrder(alpha)`
Class representing fractional order with validation.

**Parameters:**
- `alpha` (float): Fractional order value

**Attributes:**
- `alpha` (float): The fractional order value
- `is_valid` (bool): Whether the order is valid

**Methods:**
- `validate()`: Validate the fractional order
- `__str__()`: String representation

---

## ML Module

### Neural Networks

#### `FractionalNeuralNetwork(input_size, hidden_sizes, output_size, fractional_order=0.5)`
Standard fractional neural network.

**Parameters:**
- `input_size` (int): Input dimension
- `hidden_sizes` (List[int]): Hidden layer dimensions
- `output_size` (int): Output dimension
- `fractional_order` (float): Fractional order for derivatives

**Example:**
```python
from hpfracc.ml import FractionalNeuralNetwork

net = FractionalNeuralNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5
)
```

#### `MemoryEfficientFractionalNetwork(input_size, hidden_sizes, output_size, fractional_order=0.5, adjoint_config=None)`
Memory-efficient adjoint-optimized fractional network.

**Parameters:**
- `input_size` (int): Input dimension
- `hidden_sizes` (List[int]): Hidden layer dimensions
- `output_size` (int): Output dimension
- `fractional_order` (float): Fractional order for derivatives
- `adjoint_config` (AdjointConfig): Adjoint optimization configuration

**Example:**
```python
from hpfracc.ml.adjoint_optimization import MemoryEfficientFractionalNetwork, AdjointConfig

adjoint_config = AdjointConfig(
    use_adjoint=True,
    memory_efficient=True,
    checkpoint_frequency=5
)

net = MemoryEfficientFractionalNetwork(
    input_size=100,
    hidden_sizes=[256, 128, 64],
    output_size=10,
    fractional_order=0.5,
    adjoint_config=adjoint_config
)
```

### Fractional Layers

#### `FractionalConv1D(in_channels, out_channels, kernel_size, fractional_order=0.5, config=None)`
1D fractional convolutional layer.

**Parameters:**
- `in_channels` (int): Input channels
- `out_channels` (int): Output channels
- `kernel_size` (int): Kernel size
- `fractional_order` (float): Fractional order
- `config` (LayerConfig): Layer configuration

#### `FractionalConv2D(in_channels, out_channels, kernel_size, fractional_order=0.5, config=None)`
2D fractional convolutional layer.

**Parameters:**
- `in_channels` (int): Input channels
- `out_channels` (int): Output channels
- `kernel_size` (int): Kernel size
- `fractional_order` (float): Fractional order
- `config` (LayerConfig): Layer configuration

#### `FractionalLSTM(input_size, hidden_size, num_layers=1, fractional_order=0.5, config=None)`
Fractional LSTM layer.

**Parameters:**
- `input_size` (int): Input dimension
- `hidden_size` (int): Hidden dimension
- `num_layers` (int): Number of LSTM layers
- `fractional_order` (float): Fractional order
- `config` (LayerConfig): Layer configuration

#### `FractionalTransformer(d_model, nhead, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", custom_encoder=None, custom_decoder=None, config=None)`
Fractional transformer with optional fractional derivatives.

**Parameters:**
- `d_model` (int): Model dimension
- `nhead` (int): Number of attention heads
- `num_encoder_layers` (int): Number of encoder layers
- `num_decoder_layers` (int): Number of decoder layers
- `dim_feedforward` (int): Feedforward dimension
- `dropout` (float): Dropout rate
- `activation` (str): Activation function
- `custom_encoder` (nn.Module): Custom encoder
- `custom_decoder` (nn.Module): Custom decoder
- `config` (LayerConfig): Layer configuration

**Forward Method:**
```python
def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, 
            memory_mask=None, src_key_padding_mask=None, 
            tgt_key_padding_mask=None, memory_key_padding_mask=None):
    """
    Forward pass with optional fractional derivative.
    
    Args:
        src: Source tensor (required)
        tgt: Target tensor (optional, if None uses encoder-only mode)
        ...: Other transformer arguments
    
    Returns:
        Output tensor
    """
```

### Optimizers

#### `FractionalAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False, fractional_order=0.5, method="RL")`
Adam optimizer with fractional gradient updates.

**Parameters:**
- `params` (iterable): Model parameters
- `lr` (float): Learning rate
- `betas` (tuple): Beta parameters
- `eps` (float): Epsilon for numerical stability
- `weight_decay` (float): Weight decay
- `amsgrad` (bool): Use AMSGrad variant
- `fractional_order` (float): Fractional order for gradients
- `method` (str): Fractional derivative method

**Example:**
```python
from hpfracc.ml import FractionalAdam

optimizer = FractionalAdam(
    model.parameters(),
    lr=0.001,
    fractional_order=0.5,
    method="RL"
)
```

#### `FractionalSGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, fractional_order=0.5, method="RL")`
SGD optimizer with fractional gradient updates.

#### `FractionalRMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False, fractional_order=0.5, method="RL")`
RMSprop optimizer with fractional gradient updates.

### Loss Functions

#### `FractionalMSELoss(fractional_order=0.5, method="RL")`
Mean squared error loss with fractional derivatives.

**Parameters:**
- `fractional_order` (float): Fractional order
- `method` (str): Fractional derivative method

#### `FractionalCrossEntropyLoss(fractional_order=0.5, method="RL")`
Cross-entropy loss with fractional derivatives.

#### `FractionalL1Loss(fractional_order=0.5, method="RL")`
L1 loss with fractional derivatives.

### Model Registry

#### `ModelRegistry(db_path="models/registry.db", storage_path="models/")`
Central model registry for tracking and managing models.

**Methods:**
- `register_model(model, name, version, description, author, tags, framework, model_type, fractional_order, hyperparameters, performance_metrics, dataset_info, dependencies, notes="", git_commit="", git_branch="main")`: Register a new model
- `get_model(model_id)`: Get model metadata by ID
- `get_model_versions(model_id)`: Get all versions of a model
- `reconstruct_model(model_id, version=None)`: Reconstruct a model from saved state
- `promote_to_production(model_id, version)`: Promote a model to production
- `search_models(name=None, tags=None, model_type=None, deployment_status=None, author=None)`: Search for models

**Example:**
```python
from hpfracc.ml import ModelRegistry

registry = ModelRegistry()
model_id = registry.register_model(
    model=net,
    name="MyModel",
    version="1.0.0",
    description="A fractional neural network",
    author="Developer",
    tags=["fractional", "neural-network"],
    framework="pytorch",
    model_type="fractional_neural_network",
    fractional_order=0.5,
    hyperparameters={"input_size": 100, "hidden_sizes": [256, 128]},
    performance_metrics={"accuracy": 0.95},
    dataset_info={"samples": 10000},
    dependencies={"torch": "2.0.0"}
)
```

### Workflows

#### `DevelopmentWorkflow(model_registry, model_validator)`
Development workflow for model training and validation.

**Methods:**
- `train_model(model, train_data, val_data, epochs=100, optimizer=None, loss_fn=None)`: Train a model
- `validate_model(model, val_data, metrics=None)`: Validate a model
- `run_quality_gates(model_id, validation_results)`: Run quality gates

#### `ProductionWorkflow(model_registry, model_validator)`
Production workflow for model deployment and monitoring.

**Methods:**
- `promote_to_production(model_id, version, test_data, test_labels, custom_metrics, force=False)`: Promote model to production
- `deploy_model(model_id, version, deployment_config)`: Deploy a model
- `monitor_model(model_id, metrics)`: Monitor production model

---

## Graph Neural Networks

### GNN Factory

#### `FractionalGNNFactory`
Factory class for creating fractional GNN models.

**Static Methods:**
- `create_model(model_type, input_dim, hidden_dim, output_dim, **kwargs)`: Create GNN model
- `get_available_models()`: Get list of available model types
- `get_model_info(model_type)`: Get information about a specific model type

**Example:**
```python
from hpfracc.ml import FractionalGNNFactory, BackendType
from hpfracc.core.definitions import FractionalOrder

# Create GCN model
gcn = FractionalGNNFactory.create_model(
    model_type='gcn',
    input_dim=16,
    hidden_dim=32,
    output_dim=4,
    fractional_order=FractionalOrder(0.5),
    backend=BackendType.TORCH
)
```

### Available GNN Models

#### `FractionalGCN(input_dim, hidden_dim, output_dim, num_layers=3, fractional_order=0.5, method="RL", use_fractional=True, activation="relu", dropout=0.1, backend=None)`
Graph Convolutional Network with fractional calculus integration.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `output_dim` (int): Output dimension
- `num_layers` (int): Number of layers
- `fractional_order` (Union[float, FractionalOrder]): Fractional order
- `method` (str): Fractional derivative method
- `use_fractional` (bool): Whether to use fractional derivatives
- `activation` (str): Activation function
- `dropout` (float): Dropout rate
- `backend` (Optional[BackendType]): Computation backend

#### `FractionalGAT(input_dim, hidden_dim, output_dim, num_layers=3, num_heads=8, fractional_order=0.5, method="RL", use_fractional=True, activation="relu", dropout=0.1, backend=None)`
Graph Attention Network with fractional calculus integration.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `output_dim` (int): Output dimension
- `num_layers` (int): Number of layers
- `num_heads` (int): Number of attention heads
- `fractional_order` (Union[float, FractionalOrder]): Fractional order
- `method` (str): Fractional derivative method
- `use_fractional` (bool): Whether to use fractional derivatives
- `activation` (str): Activation function
- `dropout` (float): Dropout rate
- `backend` (Optional[BackendType]): Computation backend

#### `FractionalGraphSAGE(input_dim, hidden_dim, output_dim, num_layers=3, num_samples=25, fractional_order=0.5, method="RL", use_fractional=True, activation="relu", dropout=0.1, backend=None)`
GraphSAGE network with fractional calculus integration.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `output_dim` (int): Output dimension
- `num_layers` (int): Number of layers
- `num_samples` (int): Number of neighbor samples
- `fractional_order` (Union[float, FractionalOrder]): Fractional order
- `method` (str): Fractional derivative method
- `use_fractional` (bool): Whether to use fractional derivatives
- `activation` (str): Activation function
- `dropout` (float): Dropout rate
- `backend` (Optional[BackendType]): Computation backend

#### `FractionalGraphUNet(input_dim, hidden_dim, output_dim, num_layers=4, pooling_ratio=0.5, fractional_order=0.5, method="RL", use_fractional=True, activation="relu", dropout=0.1, backend=None)`
Graph U-Net with fractional calculus integration.

**Parameters:**
- `input_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden layer dimension
- `output_dim` (int): Output dimension
- `num_layers` (int): Number of layers
- `pooling_ratio` (float): Pooling ratio for hierarchical structure
- `fractional_order` (Union[float, FractionalOrder]): Fractional order
- `method` (str): Fractional derivative method
- `use_fractional` (bool): Whether to use fractional derivatives
- `activation` (str): Activation function
- `dropout` (float): Dropout rate
- `backend` (Optional[BackendType]): Computation backend

### Backend Support

All GNN models support multiple computation backends:

- **PyTorch** (`BackendType.TORCH`): Full-featured deep learning with GPU acceleration
- **JAX** (`BackendType.JAX`): High-performance numerical computing with automatic differentiation
- **NUMBA** (`BackendType.NUMBA`): JIT compilation for CPU optimization

### Usage Example

```python
from hpfracc.ml import FractionalGNNFactory, BackendType
from hpfracc.core.definitions import FractionalOrder
import torch

# Create synthetic graph data
num_nodes = 100
num_features = 16
node_features = torch.randn(num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, 200))

# Create GNN model
gnn = FractionalGNNFactory.create_model(
    model_type='gcn',
    input_dim=num_features,
    hidden_dim=32,
    output_dim=4,
    fractional_order=FractionalOrder(0.5),
    backend=BackendType.TORCH
)

# Forward pass
output = gnn(node_features, edge_index)
print(f"Output shape: {output.shape}")
```

---

## Benchmarks Module

### ML Performance Benchmark

#### `MLPerformanceBenchmark(device="auto", num_runs=5, warmup_runs=3)`
Comprehensive ML performance benchmarking system.

**Parameters:**
- `device` (str): Device to run benchmarks on ("auto", "cpu", "cuda")
- `num_runs` (int): Number of measurement runs
- `warmup_runs` (int): Number of warmup runs

**Methods:**
- `benchmark_fractional_networks(input_sizes, hidden_sizes_list, fractional_orders, methods)`: Benchmark neural networks
- `benchmark_fractional_attention(batch_sizes, seq_lengths, d_models, fractional_orders, methods)`: Benchmark attention mechanisms
- `benchmark_fractional_layers(layer_types, configs, fractional_orders, methods)`: Benchmark individual layers
- `generate_report(output_dir="benchmark_results")`: Generate comprehensive report

**Example:**
```python
from hpfracc.benchmarks import MLPerformanceBenchmark

benchmark = MLPerformanceBenchmark(device="cuda", num_runs=10)

# Benchmark networks
results = benchmark.benchmark_fractional_networks(
    input_sizes=[50, 100, 200],
    hidden_sizes_list=[[128, 64], [256, 128, 64]],
    fractional_orders=[0.1, 0.5, 0.9],
    methods=["RL", "Caputo"]
)

# Generate report
benchmark.generate_report("ml_benchmarks")
```

---

## Analytics Module

### Usage Analytics

#### `UsageAnalytics(db_path="analytics/usage.db")`
Track usage patterns and model performance.

**Methods:**
- `track_usage(operation, model_id, parameters, execution_time, success)`: Track operation usage
- `get_usage_stats(time_period="7d")`: Get usage statistics
- `get_popular_models(limit=10)`: Get most popular models

### Performance Monitoring

#### `PerformanceMonitor(db_path="analytics/performance.db")`
Monitor model performance and resource usage.

**Methods:**
- `record_metric(model_id, metric_name, value, timestamp=None)`: Record performance metric
- `get_performance_history(model_id, metric_name, time_period="30d")`: Get performance history
- `generate_performance_report(model_id)`: Generate performance report

### Error Analysis

#### `ErrorAnalyzer(db_path="analytics/errors.db")`
Analyze error patterns and reliability.

**Methods:**
- `record_error(operation, model_id, error_type, error_message, stack_trace)`: Record error
- `get_error_summary(time_period="7d")`: Get error summary
- `get_reliability_score(model_id)`: Calculate reliability score

---

## Configuration Classes

### Layer Configuration

#### `LayerConfig(fractional_order=None, method="RL", use_fractional=True, activation="relu", dropout=0.1)`
Configuration for fractional layers.

**Parameters:**
- `fractional_order` (FractionalOrder): Fractional order object
- `method` (str): Fractional derivative method
- `use_fractional` (bool): Whether to use fractional derivatives
- `activation` (str): Activation function
- `dropout` (float): Dropout rate

### Adjoint Configuration

#### `AdjointConfig(use_adjoint=True, memory_efficient=True, checkpoint_frequency=5, gradient_accumulation=False, accumulation_steps=4)`
Configuration for adjoint optimization.

**Parameters:**
- `use_adjoint` (bool): Whether to use adjoint methods
- `memory_efficient` (bool): Enable memory optimization
- `checkpoint_frequency` (int): Checkpointing frequency
- `gradient_accumulation` (bool): Enable gradient accumulation
- `accumulation_steps` (int): Number of accumulation steps

---

## Data Types

### Deployment Status

```python
class DeploymentStatus(Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"
```

### Model Metadata

```python
@dataclass
class ModelMetadata:
    model_id: str
    version: str
    name: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    framework: str
    model_type: str
    fractional_order: float
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    dependencies: Dict[str, str]
    file_size: int
    checksum: str
    deployment_status: DeploymentStatus
    notes: str = ""
```

---

## Error Handling

The library provides comprehensive error handling with informative error messages:

- **ValueError**: Invalid parameters (e.g., fractional order out of range)
- **RuntimeError**: Runtime issues (e.g., tensor shape mismatches)
- **ImportError**: Missing dependencies
- **FileNotFoundError**: Missing model files

## Best Practices

1. **Fractional Orders**: Use values in valid ranges (0 < α < 2 for most methods)
2. **Memory Management**: Use adjoint optimization for large models
3. **Model Registry**: Always register models with comprehensive metadata
4. **Performance**: Run benchmarks before production deployment
5. **Monitoring**: Continuously monitor production models

## Examples

See the `examples/` directory for comprehensive usage examples:
- `ml_integration_demo.py`: Complete ML workflow demonstration
- `fractional_calculus_examples.py`: Basic fractional calculus usage
- `performance_benchmarks.py`: Performance testing examples
