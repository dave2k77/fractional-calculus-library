# HPFRACC Examples

## Table of Contents
1. [Basic Fractional Calculus](#basic-fractional-calculus)
2. [Neural Network Examples](#neural-network-examples)
3. [Advanced Architectures](#advanced-architectures)
4. [Training Examples](#training-examples)
5. [Production Workflow Examples](#production-workflow-examples)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Real-World Applications](#real-world-applications)

---

## Basic Fractional Calculus

### Simple Fractional Derivatives

```python
import torch
from hpfracc.core import fractional_derivative

# Create sample data
t = torch.linspace(0, 10, 1000)
f = torch.sin(t) + 0.1 * torch.randn_like(t)

# Compute fractional derivatives with different methods
alpha = 0.5

# Riemann-Liouville derivative
rl_derivative = fractional_derivative(f, alpha, method="RL")

# Caputo derivative
caputo_derivative = fractional_derivative(f, alpha, method="Caputo")

# Grünwald-Letnikov derivative
gl_derivative = fractional_derivative(f, alpha, method="GL")

print(f"Original function shape: {f.shape}")
print(f"RL derivative shape: {rl_derivative.shape}")
print(f"Caputo derivative shape: {caputo_derivative.shape}")
print(f"GL derivative shape: {gl_derivative.shape}")
```

### Fractional Order Validation

```python
from hpfracc.core.definitions import FractionalOrder

# Test different fractional orders
orders = [0.1, 0.5, 0.9, 1.0, 1.5, 2.0]

for alpha in orders:
    try:
        frac_order = FractionalOrder(alpha)
        print(f"α = {alpha}: Valid = {frac_order.is_valid}")
    except ValueError as e:
        print(f"α = {alpha}: Error - {e}")

# Test specific method constraints
def test_method_constraints():
    """Test method-specific constraints"""
    
    # Caputo method requires 0 < α < 1
    try:
        result = fractional_derivative(torch.randn(100), 1.5, method="Caputo")
        print("Caputo with α=1.5: Success")
    except ValueError as e:
        print(f"Caputo with α=1.5: {e}")
    
    # RL and GL methods work for 0 < α < 2
    try:
        result = fractional_derivative(torch.randn(100), 1.5, method="RL")
        print("RL with α=1.5: Success")
    except ValueError as e:
        print(f"RL with α=1.5: {e}")

test_method_constraints()
```

### Visualizing Fractional Derivatives

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_fractional_derivatives():
    """Visualize different fractional derivatives"""
    
    # Create smooth function
    t = torch.linspace(0, 4*np.pi, 1000)
    f = torch.sin(t) * torch.exp(-0.1*t)
    
    # Compute derivatives
    orders = [0.1, 0.5, 1.0, 1.5]
    derivatives = {}
    
    for alpha in orders:
        derivatives[f'α={alpha}'] = fractional_derivative(f, alpha, method="RL")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t.numpy(), f.numpy(), 'k-', linewidth=2, label='Original Function')
    plt.title('Original Function: sin(t) * exp(-0.1t)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for label, derivative in derivatives.items():
        plt.plot(t.numpy(), derivative.numpy(), label=label, linewidth=2)
    
    plt.title('Fractional Derivatives (Riemann-Liouville)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run visualization
visualize_fractional_derivatives()
```

---

## Neural Network Examples

### Basic Fractional Neural Network

```python
import torch
import torch.nn as nn
from hpfracc.ml import FractionalNeuralNetwork, FractionalAdam, FractionalMSELoss

def basic_fractional_network_example():
    """Basic example of creating and using a fractional neural network"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    net = FractionalNeuralNetwork(
        input_size=50,
        hidden_sizes=[256, 128, 64],
        output_size=10,
        fractional_order=0.5
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = FractionalAdam(net.parameters(), lr=0.001)
    loss_fn = FractionalMSELoss(fractional_order=0.5)
    
    # Generate sample data
    batch_size = 32
    x = torch.randn(batch_size, 50).to(device)
    y = torch.randn(batch_size, 10).to(device)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    
    # Training loop
    net.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        output = net(x)
        loss = loss_fn(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test inference
    net.eval()
    with torch.no_grad():
        test_output = net(x)
        print(f"Test output shape: {test_output.shape}")
        print(f"Final loss: {loss_fn(test_output, y).item():.4f}")
    
    return net

# Run example
net = basic_fractional_network_example()
```

### Memory-Efficient Adjoint Network

```python
from hpfracc.ml.adjoint_optimization import (
    MemoryEfficientFractionalNetwork,
    AdjointConfig
)

def adjoint_network_example():
    """Example of using memory-efficient adjoint networks"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure adjoint optimization
    adjoint_config = AdjointConfig(
        use_adjoint=True,
        memory_efficient=True,
        checkpoint_frequency=3,
        gradient_accumulation=True,
        accumulation_steps=4
    )
    
    # Create large network
    net = MemoryEfficientFractionalNetwork(
        input_size=200,
        hidden_sizes=[1024, 512, 256, 128, 64],
        output_size=20,
        fractional_order=0.5,
        adjoint_config=adjoint_config
    ).to(device)
    
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Generate large dataset
    batch_size = 64
    x = torch.randn(batch_size, 200).to(device)
    y = torch.randn(batch_size, 20).to(device)
    
    # Training with gradient accumulation
    optimizer = FractionalAdam(net.parameters(), lr=0.001)
    loss_fn = FractionalMSELoss(fractional_order=0.5)
    
    net.train()
    for step in range(20):
        # Forward pass
        output = net(x)
        loss = loss_fn(output, y) / adjoint_config.accumulation_steps
        loss.backward()
        
        # Update every accumulation_steps
        if (step + 1) % adjoint_config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Step {step+1}: Loss = {loss.item() * adjoint_config.accumulation_steps:.4f}")
    
    return net

# Run example
adjoint_net = adjoint_network_example()
```

---

## Advanced Architectures

### Fractional Convolutional Network

```python
from hpfracc.ml.layers import FractionalConv1D, FractionalConv2D, LayerConfig
from hpfracc.core.definitions import FractionalOrder

def fractional_convolutional_example():
    """Example of fractional convolutional networks"""
    
    # Configure layers
    config = LayerConfig(
        fractional_order=FractionalOrder(0.5),
        method="RL",
        use_fractional=True,
        activation="relu",
        dropout=0.1
    )
    
    # 1D Convolution for time series
    conv1d = FractionalConv1D(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        config=config
    )
    
    # 2D Convolution for images
    conv2d = FractionalConv2D(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        config=config
    )
    
    # Test 1D convolution
    x1d = torch.randn(32, 64, 100)  # (batch, channels, length)
    output1d = conv1d(x1d)
    print(f"1D Conv input: {x1d.shape}")
    print(f"1D Conv output: {output1d.shape}")
    
    # Test 2D convolution
    x2d = torch.randn(32, 3, 64, 64)  # (batch, channels, height, width)
    output2d = conv2d(x2d)
    print(f"2D Conv input: {x2d.shape}")
    print(f"2D Conv output: {output2d.shape}")
    
    return conv1d, conv2d

# Run example
conv1d, conv2d = fractional_convolutional_example()
```

### Fractional LSTM Network

```python
from hpfracc.ml.layers import FractionalLSTM

def fractional_lstm_example():
    """Example of fractional LSTM networks"""
    
    # Create fractional LSTM
    lstm = FractionalLSTM(
        input_size=100,
        hidden_size=256,
        num_layers=2,
        fractional_order=0.5,
        config=config
    )
    
    # Test with sequence data
    batch_size = 16
    seq_length = 50
    x = torch.randn(batch_size, seq_length, 100)
    
    # Forward pass
    output, (hidden, cell) = lstm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Cell state shape: {cell.shape}")
    
    # Test with different sequence lengths
    x_variable = torch.randn(batch_size, 30, 100)
    output_variable, _ = lstm(x_variable)
    print(f"Variable length output: {output_variable.shape}")
    
    return lstm

# Run example
lstm_net = fractional_lstm_example()
```

### Fractional Transformer

```python
from hpfracc.ml.layers import FractionalTransformer

def fractional_transformer_example():
    """Example of fractional transformer networks"""
    
    # Create fractional transformer
    transformer = FractionalTransformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        config=config
    )
    
    batch_size = 16
    seq_length = 100
    
    # Test encoder-only mode
    x_encoder = torch.randn(batch_size, seq_length, 512)
    output_encoder = transformer(x_encoder)
    print(f"Encoder input: {x_encoder.shape}")
    print(f"Encoder output: {output_encoder.shape}")
    
    # Test full transformer mode
    src = torch.randn(batch_size, seq_length, 512)
    tgt = torch.randn(batch_size, 50, 512)
    output_full = transformer(src, tgt)
    print(f"Full transformer src: {src.shape}")
    print(f"Full transformer tgt: {tgt.shape}")
    print(f"Full transformer output: {output_full.shape}")
    
    return transformer

# Run example
transformer_net = fractional_transformer_example()
```

---

## Training Examples

### Complete Training Pipeline

```python
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_synthetic_dataset(n_samples=10000, input_dim=100, output_dim=10):
    """Create synthetic dataset for training"""
    
    # Generate features
    X = torch.randn(n_samples, input_dim)
    
    # Generate targets (non-linear transformation)
    y = torch.zeros(n_samples, output_dim)
    for i in range(output_dim):
        y[:, i] = torch.sin(X[:, i % input_dim]) + 0.1 * torch.randn(n_samples)
    
    return X, y

def fractional_network_training_example():
    """Complete training example with data loading and validation"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset
    X, y = create_synthetic_dataset(n_samples=10000, input_dim=100, output_dim=10)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_val_scaled = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    
    # Create data loaders
    train_dataset = data.TensorDataset(X_train_scaled, y_train)
    val_dataset = data.TensorDataset(X_val_scaled, y_val)
    
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create network
    net = FractionalNeuralNetwork(
        input_size=100,
        hidden_sizes=[512, 256, 128, 64],
        output_size=10,
        fractional_order=0.5
    ).to(device)
    
    # Setup training
    optimizer = FractionalAdam(net.parameters(), lr=0.001)
    loss_fn = FractionalMSELoss(fractional_order=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(100):
        # Training phase
        net.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = net(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = net(batch_x)
                val_loss += loss_fn(output, batch_y).item()
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val Loss = {val_loss/len(val_loader):.4f}")
    
    # Load best model
    net.load_state_dict(torch.load('best_model.pth'))
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return net

# Run training example
trained_net = fractional_network_training_example()
```

### Hyperparameter Optimization

```python
import optuna
from sklearn.metrics import mean_squared_error

def hyperparameter_optimization_example():
    """Example of hyperparameter optimization with Optuna"""
    
    def objective(trial):
        """Objective function for hyperparameter optimization"""
        
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
        fractional_order = trial.suggest_float('fractional_order', 0.1, 0.9)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        
        # Create network
        hidden_sizes = [hidden_size // (2**i) for i in range(num_layers)]
        net = FractionalNeuralNetwork(
            input_size=100,
            hidden_sizes=hidden_sizes,
            output_size=10,
            fractional_order=fractional_order
        ).to(device)
        
        # Setup training
        optimizer = FractionalAdam(net.parameters(), lr=lr)
        loss_fn = FractionalMSELoss(fractional_order=fractional_order)
        
        # Quick training for optimization
        net.train()
        for epoch in range(5):  # Reduced epochs for optimization
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = net(batch_x)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        net.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = net(batch_x)
                val_predictions.append(output.cpu())
                val_targets.append(batch_y.cpu())
        
        # Calculate MSE
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        mse = mean_squared_error(val_targets.numpy(), val_predictions.numpy())
        
        return mse
    
    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print(f"Best trial: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")
    
    return study

# Run optimization example
study = hyperparameter_optimization_example()
```

---

## Production Workflow Examples

### Model Registry and Validation

```python
from hpfracc.ml import ModelRegistry, ModelValidator
from hpfracc.ml.workflow import DevelopmentWorkflow, ProductionWorkflow

def production_workflow_example():
    """Complete production workflow example"""
    
    # Initialize components
    registry = ModelRegistry()
    validator = ModelValidator()
    dev_workflow = DevelopmentWorkflow(registry, validator)
    prod_workflow = ProductionWorkflow(registry, validator)
    
    # Train a model (using the trained network from previous example)
    net = trained_net  # From previous example
    
    # Register model in development
    model_id = registry.register_model(
        model=net,
        name="FractionalTimeSeriesPredictor",
        version="1.0.0",
        description="Fractional neural network for time series prediction",
        author="ML Team",
        tags=["fractional", "neural-network", "time-series", "production"],
        framework="pytorch",
        model_type="fractional_neural_network",
        fractional_order=0.5,
        hyperparameters={
            "input_size": 100,
            "hidden_sizes": [512, 256, 128, 64],
            "output_size": 10,
            "learning_rate": 0.001
        },
        performance_metrics={
            "train_loss": 0.0234,
            "val_loss": 0.0456,
            "mse": 0.0456
        },
        dataset_info={
            "samples": 8000,
            "features": 100,
            "classes": 10,
            "train_val_split": "80/20"
        },
        dependencies={
            "torch": "2.0.0",
            "hpfracc": "1.0.0"
        }
    )
    
    print(f"Model registered with ID: {model_id}")
    
    # Validate model
    validation_results = dev_workflow.train_model(
        model=net,
        train_data=(X_train_scaled, y_train),
        val_data=(X_val_scaled, y_val),
        epochs=10  # Reduced for example
    )
    
    # Run quality gates
    quality_result = dev_workflow.run_quality_gates(
        model_id=model_id,
        validation_results=validation_results
    )
    
    if quality_result["passed"]:
        print("Model passed quality gates!")
        
        # Promote to production
        promotion_result = prod_workflow.promote_to_production(
            model_id=model_id,
            version="1.0.0",
            test_data=X_val_scaled,
            test_labels=y_val,
            custom_metrics={},
            force=False
        )
        
        if promotion_result["promoted"]:
            print("Model promoted to production!")
            
            # Deploy model
            deployment_result = prod_workflow.deploy_model(
                model_id=model_id,
                version="1.0.0",
                deployment_config={
                    "environment": "production",
                    "replicas": 3,
                    "resources": {"cpu": "2", "memory": "4Gi"}
                }
            )
            
            print(f"Deployment result: {deployment_result}")
        else:
            print(f"Promotion failed: {promotion_result['reason']}")
    else:
        print(f"Quality gate failed: {quality_result['reason']}")
    
    return registry, dev_workflow, prod_workflow

# Run production workflow example
registry, dev_workflow, prod_workflow = production_workflow_example()
```

### Model Monitoring and Inference

```python
def production_inference_example():
    """Example of production inference and monitoring"""
    
    # Load production model
    production_model = registry.reconstruct_model(model_id, "1.0.0")
    production_model.eval()
    
    # Generate new data for inference
    X_new = torch.randn(100, 100)  # 100 new samples
    X_new_scaled = torch.tensor(scaler.transform(X_new), dtype=torch.float32)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        predictions = production_model(X_new_scaled.to(device))
    inference_time = time.time() - start_time
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Throughput: {len(X_new)/inference_time:.1f} samples/second")
    
    # Monitor performance
    prod_workflow.monitor_model(
        model_id=model_id,
        metrics={
            "latency": inference_time / len(X_new),
            "throughput": len(X_new) / inference_time,
            "accuracy": 0.95,  # Example accuracy
            "memory_usage": 0.5  # GB
        }
    )
    
    # Get production model info
    production_models = registry.get_production_models()
    for model in production_models:
        print(f"Production model: {model.metadata.name} v{model.version}")
        print(f"Deployment status: {model.metadata.deployment_status}")
    
    return predictions

# Run inference example
predictions = production_inference_example()
```

---

## Performance Benchmarking

### ML Performance Benchmark

```python
from hpfracc.benchmarks import MLPerformanceBenchmark
import time

def performance_benchmark_example():
    """Example of running performance benchmarks"""
    
    # Initialize benchmark
    benchmark = MLPerformanceBenchmark(
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_runs=5,
        warmup_runs=2
    )
    
    print("Running ML Performance Benchmark...")
    
    # Benchmark neural networks
    network_results = benchmark.benchmark_fractional_networks(
        input_sizes=[50, 100, 200],
        hidden_sizes_list=[[128, 64], [256, 128, 64]],
        fractional_orders=[0.1, 0.5, 0.9],
        methods=["RL", "Caputo"]
    )
    
    print("Neural network benchmarking completed!")
    
    # Benchmark attention mechanisms
    attention_results = benchmark.benchmark_fractional_attention(
        batch_sizes=[16, 32, 64],
        seq_lengths=[100, 200],
        d_models=[256, 512],
        fractional_orders=[0.1, 0.5, 0.9],
        methods=["RL", "Caputo"]
    )
    
    print("Attention mechanism benchmarking completed!")
    
    # Benchmark individual layers
    layer_results = benchmark.benchmark_fractional_layers(
        layer_types=["FractionalConv1D", "FractionalConv2D", "FractionalLSTM"],
        configs=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": 3},
            {"in_channels": 3, "out_channels": 64, "kernel_size": 3},
            {"input_size": 100, "hidden_size": 256, "num_layers": 2}
        ],
        fractional_orders=[0.1, 0.5, 0.9],
        methods=["RL", "Caputo"]
    )
    
    print("Individual layer benchmarking completed!")
    
    # Generate comprehensive report
    benchmark.generate_report("ml_performance_benchmarks")
    
    print("Benchmark report generated in 'ml_performance_benchmarks' directory")
    
    return network_results, attention_results, layer_results

# Run benchmark example
network_results, attention_results, layer_results = performance_benchmark_example()
```

### Custom Benchmarking

```python
def custom_benchmark_example():
    """Example of custom benchmarking for specific use cases"""
    
    def benchmark_model_variants():
        """Benchmark different model variants"""
        
        variants = {
            "Standard": FractionalNeuralNetwork(100, [256, 128], 10, 0.5),
            "Deep": FractionalNeuralNetwork(100, [512, 256, 128, 64], 10, 0.5),
            "Wide": FractionalNeuralNetwork(100, [1024, 512], 10, 0.5)
        }
        
        results = {}
        input_data = torch.randn(64, 100)
        
        for name, model in variants.items():
            model.to(device)
            model.eval()
            
            # Warmup
            for _ in range(3):
                _ = model(input_data.to(device))
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                _ = model(input_data.to(device))
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            results[name] = {
                "avg_time": avg_time,
                "throughput": 64 / avg_time,
                "parameters": sum(p.numel() for p in model.parameters())
            }
            
            print(f"{name}: {avg_time:.4f}s, {results[name]['throughput']:.1f} samples/s")
        
        return results
    
    def benchmark_fractional_orders():
        """Benchmark different fractional orders"""
        
        orders = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}
        
        model = FractionalNeuralNetwork(100, [256, 128], 10, 0.5).to(device)
        input_data = torch.randn(64, 100).to(device)
        
        for alpha in orders:
            # Update fractional order
            model.fractional_order = alpha
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                _ = model(input_data)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            results[alpha] = avg_time
            
            print(f"α={alpha}: {avg_time:.4f}s")
        
        return results
    
    # Run custom benchmarks
    print("Benchmarking model variants...")
    variant_results = benchmark_model_variants()
    
    print("\nBenchmarking fractional orders...")
    order_results = benchmark_fractional_orders()
    
    return variant_results, order_results

# Run custom benchmark example
variant_results, order_results = custom_benchmark_example()
```

---

## Real-World Applications

### Time Series Prediction

```python
def time_series_prediction_example():
    """Example of time series prediction with fractional networks"""
    
    # Generate synthetic time series data
    def generate_time_series(n_samples=1000, seq_length=100):
        """Generate synthetic time series with seasonality and trend"""
        t = torch.linspace(0, 10, seq_length)
        
        # Base signal: trend + seasonality + noise
        trend = 0.1 * t
        seasonality = torch.sin(2 * torch.pi * t) + 0.5 * torch.sin(4 * torch.pi * t)
        noise = 0.05 * torch.randn_like(t)
        
        signal = trend + seasonality + noise
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(n_samples):
            start_idx = torch.randint(0, seq_length - 20, (1,)).item()
            seq = signal[start_idx:start_idx + 80]  # Input sequence
            target = signal[start_idx + 80:start_idx + 100]  # Target sequence
            
            sequences.append(seq)
            targets.append(target)
        
        return torch.stack(sequences), torch.stack(targets)
    
    # Generate data
    X_ts, y_ts = generate_time_series(n_samples=1000, seq_length=100)
    
    # Split data
    train_size = int(0.8 * len(X_ts))
    X_train_ts, X_val_ts = X_ts[:train_size], X_ts[train_size:]
    y_train_ts, y_val_ts = y_ts[:train_size], y_ts[train_size:]
    
    # Create fractional LSTM for time series
    ts_model = FractionalLSTM(
        input_size=80,
        hidden_size=128,
        num_layers=2,
        fractional_order=0.5,
        config=config
    ).to(device)
    
    # Training
    optimizer = torch.optim.Adam(ts_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    ts_model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = ts_model(X_train_ts.to(device))
        loss = loss_fn(output, y_train_ts.to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test prediction
    ts_model.eval()
    with torch.no_grad():
        test_output, _ = ts_model(X_val_ts[:5].to(device))
        
        print(f"Input sequence shape: {X_val_ts[:5].shape}")
        print(f"Predicted shape: {test_output.shape}")
        print(f"Actual shape: {y_val_ts[:5].shape}")
    
    return ts_model

# Run time series example
ts_model = time_series_prediction_example()
```

### Image Classification

```python
def image_classification_example():
    """Example of image classification with fractional CNNs"""
    
    # Generate synthetic image data
    batch_size = 32
    channels = 3
    height, width = 64, 64
    num_classes = 10
    
    X_img = torch.randn(batch_size, channels, height, width)
    y_img = torch.randint(0, num_classes, (batch_size,))
    
    # Create fractional CNN
    class FractionalCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            
            self.fractional_conv1 = FractionalConv2D(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                config=config
            )
            
            self.fractional_conv2 = FractionalConv2D(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                config=config
            )
            
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.fractional_conv1(x)
            x = torch.relu(x)
            x = self.fractional_conv2(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    # Create model
    img_model = FractionalCNN(num_classes=num_classes).to(device)
    
    # Training
    optimizer = torch.optim.Adam(img_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    img_model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        
        # Forward pass
        output = img_model(X_img.to(device))
        loss = loss_fn(output, y_img.to(device))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Test
    img_model.eval()
    with torch.no_grad():
        test_output = img_model(X_img[:5].to(device))
        predictions = torch.argmax(test_output, dim=1)
        
        print(f"Input image shape: {X_img[:5].shape}")
        print(f"Output shape: {test_output.shape}")
        print(f"Predictions: {predictions}")
        print(f"Actual labels: {y_img[:5]}")
    
    return img_model

# Run image classification example
img_model = image_classification_example()
```

---

## Conclusion

These examples demonstrate the versatility and power of the HPFRACC library across various machine learning tasks:

1. **Basic Operations**: Fractional derivatives, validation, and visualization
2. **Neural Networks**: Standard and memory-efficient architectures
3. **Advanced Architectures**: Convolutional, recurrent, and transformer networks
4. **Training**: Complete pipelines with optimization and hyperparameter tuning
5. **Production**: End-to-end workflow from development to deployment
6. **Benchmarking**: Performance analysis and optimization
7. **Applications**: Real-world use cases in time series and image processing

The library provides a comprehensive framework that combines mathematical rigor with practical usability, making it suitable for both research and production applications.

For more examples and advanced usage patterns, refer to the API reference and ML integration guide.
