#!/usr/bin/env python3
"""
Machine Learning Integration Demo

This script demonstrates the comprehensive ML integration system for hpfracc,
including fractional neural networks, model registry, and development vs.
production workflow management.

NOTE: The MLOps features in this demo are currently under development.
This example is a placeholder for future functionality.
"""

# from hpfracc.ml import (
#     FractionalNeuralNetwork,
#     FractionalAttention,
#     FractionalMSELoss,
#     FractionalAdam,
#     ModelRegistry,
#     DevelopmentWorkflow,
#     ProductionWorkflow,
#     ModelValidator,
#     FractionalConv1D,
#     FractionalConv2D,
#     FractionalLSTM,
#     FractionalTransformer,
#     FractionalPooling,
#     FractionalBatchNorm1d
# )
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_demo.log')
        ]
    )


def create_sample_data(n_samples=1000, input_size=10, output_size=3):
    """Create sample training and validation data"""
    print("🔧 Creating sample data...")

    # Generate random input data
    X = torch.randn(n_samples, input_size)

    # Create target data with some non-linear relationships
    y = torch.zeros(n_samples, output_size)
    y[:, 0] = torch.sin(X[:, 0]) + 0.1 * torch.randn(n_samples)
    y[:, 1] = X[:, 1] ** 2 + 0.1 * torch.randn(n_samples)
    y[:, 2] = torch.exp(-X[:, 2]) + 0.1 * torch.randn(n_samples)

    # Split into train and validation
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    print(f"✅ Created {n_samples} samples")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")

    return (X_train, y_train), (X_val, y_val)


def create_fractional_model(input_size=10, hidden_sizes=[64, 32], output_size=3, fractional_order=0.5):
    """Create a fractional neural network model"""
    print(f"🧠 Creating fractional neural network with α={fractional_order}...")

    model = FractionalNeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        fractional_order=fractional_order,
        activation="relu",
        dropout=0.1
    )

    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train_model(model, train_data, val_data, epochs=50, lr=0.001):
    """Train the fractional neural network"""
    print("🚀 Starting model training...")

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Setup loss function and optimizer
    criterion = FractionalMSELoss(fractional_order=0.5, method="RL")
    optimizer = FractionalAdam(
        model.parameters(),
        lr=lr,
        fractional_order=0.5,
        method="RL",
        use_fractional=True
    )

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train, use_fractional=True, method="RL")
        loss = criterion(outputs, y_train, use_fractional=True)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, use_fractional=True, method="RL")
            val_loss = criterion(val_outputs, y_val, use_fractional=True)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {loss.item():.6f}, "
                  f"Val Loss: {val_loss.item():.6f}")

    print("✅ Training completed!")
    return train_losses, val_losses


def evaluate_model(model, val_data):
    """Evaluate the trained model"""
    print("📊 Evaluating model performance...")

    X_val, y_val = val_data
    model.eval()

    with torch.no_grad():
        predictions = model(X_val, use_fractional=True, method="RL")

        # Calculate metrics
        mse = nn.functional.mse_loss(predictions, y_val)
        mae = nn.functional.l1_loss(predictions, y_val)

        # Calculate R² score
        ss_res = torch.sum((y_val - predictions) ** 2)
        ss_tot = torch.sum((y_val - y_val.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    print(f"✅ Evaluation Results:")
    print(f"   Mean Squared Error: {mse.item():.6f}")
    print(f"   Mean Absolute Error: {mae.item():.6f}")
    print(f"   R² Score: {r2.item():.6f}")

    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'r2': r2.item()
    }


def demonstrate_model_registry(model, performance_metrics):
    """Demonstrate the model registry system"""
    print("\n📚 Demonstrating Model Registry System...")

    # Initialize registry
    registry = ModelRegistry(storage_path="models/")

    # Register the model in development
    model_id = registry.register_model(
        model=model,
        name="fractional_neural_network",
        version="1.0.0",
        description="Fractional neural network with α=0.5",
        author="Davian R. Chin",
        tags=["fractional", "neural_network", "regression"],
        framework="PyTorch",
        model_type="FractionalNeuralNetwork",
        fractional_order=0.5,
        hyperparameters={
            "input_size": 10,
            "hidden_sizes": [64, 32],
            "output_size": 3,
            "fractional_order": 0.5,
            "method": "RL"
        },
        performance_metrics=performance_metrics,
        dataset_info={
            "num_samples": 1000,
            "train_split": 0.8,
            "val_split": 0.2,
            "features": 10
        },
        dependencies={
            "torch": ">=2.0.0",
            "numpy": ">=1.21.0",
            "hpfracc": ">=0.1.0"
        },
        notes="Trained with fractional calculus integration",
        git_commit="demo_commit",
        git_branch="dev"
    )

    print(f"✅ Model registered with ID: {model_id}")

    # Get model information
    model_info = registry.get_model(model_id)
    print(f"📋 Model Info:")
    print(f"   Name: {model_info.name}")
    print(f"   Version: {model_info.version}")
    print(f"   Status: {model_info.deployment_status.value}")
    print(f"   Author: {model_info.author}")
    print(f"   Created: {model_info.created_at}")

    # Get registry summary
    summary = registry.get_registry_summary()
    print(f"📊 Registry Summary:")
    print(f"   Total Models: {summary['total_models']}")
    print(f"   Total Versions: {summary['total_versions']}")
    print(f"   Production Models: {summary['production_models']}")

    return registry, model_id


def demonstrate_development_workflow(registry, model_id, val_data):
    """Demonstrate the development workflow"""
    print("\n🔬 Demonstrating Development Workflow...")

    # Initialize validator and workflow
    validator = ModelValidator()
    dev_workflow = DevelopmentWorkflow(registry, validator)

    # Validate the development model
    X_val, y_val = val_data
    validation_results = dev_workflow.validate_development_model(
        model_id=model_id,
        test_data=X_val,
        test_labels=y_val
    )

    print(f"✅ Development Validation Results:")
    print(f"   Validation Passed: {validation_results['validation_passed']}")
    print(f"   Final Score: {validation_results['final_score']:.3f}")
    print(
        f"   Required Gates Passed: {validation_results['required_gates_passed']}")

    # Show quality gate results
    print(f"🔍 Quality Gate Results:")
    for gate_result in validation_results['gate_results']:
        print(
            f"   {gate_result['gate_name']}: {'✅ PASSED' if gate_result['passed'] else '❌ FAILED'}")
        if not gate_result['passed']:
            for metric, result in gate_result['results'].items():
                if not result['passed']:
                    print(
                        f"     {metric}: {result['value']} (expected: {result['threshold']})")

    return validation_results


def demonstrate_production_workflow(registry, model_id, val_data):
    """Demonstrate the production workflow"""
    print("\n🚀 Demonstrating Production Workflow...")

    # Initialize production workflow
    validator = ModelValidator()
    prod_workflow = ProductionWorkflow(registry, validator)

    # Promote model to production
    promotion_results = prod_workflow.promote_to_production(
        model_id=model_id,
        version="1.0.0",
        test_data=val_data[0],
        test_labels=val_data[1]
    )

    if promotion_results['promoted']:
        print("✅ Model successfully promoted to production!")
        print(f"   Model ID: {promotion_results['model_id']}")
        print(f"   Version: {promotion_results['version']}")
        print(f"   Promoted At: {promotion_results['promoted_at']}")
    else:
        print("❌ Model promotion failed!")
        print(f"   Reason: {promotion_results['reason']}")

    # Get production status
    prod_status = prod_workflow.get_production_status()
    print(f"📊 Production Status:")
    print(
        f"   Total Production Models: {prod_status['total_production_models']}")
    for model_info in prod_status['models']:
        print(
            f"   - {model_info['name']} v{model_info['version']} ({model_info['deployment_status']})")

    return promotion_results


def demonstrate_model_monitoring(prod_workflow):
    """Demonstrate production model monitoring"""
    print("\n📡 Demonstrating Model Monitoring...")

    # Simulate monitoring data
    monitoring_data = {
        'model_id_1': {
            'accuracy': 0.85,
            'inference_time': 45.2,
            'memory_usage': 128.5
        },
        'model_id_2': {
            'accuracy': 0.72,  # Below threshold
            'inference_time': 156.8,  # Above threshold
            'memory_usage': 256.0
        }
    }

    # Monitor production models
    monitoring_results = prod_workflow.monitor_production_models(
        monitoring_data)

    print(f"📊 Monitoring Results:")
    print(
        f"   Models Monitored: {monitoring_results['total_models_monitored']}")
    print(f"   Alerts Generated: {monitoring_results['alert_count']}")

    if monitoring_results['alerts']:
        print(f"🚨 Alerts:")
        for alert in monitoring_results['alerts']:
            print(f"   - {alert['model_name']} v{alert['version']}: "
                  f"{alert['alert_type']} - {alert['metric']} = {alert['current_value']} "
                  f"(threshold: {alert['threshold']})")

    return monitoring_results


def demonstrate_fractional_attention():
    """Demonstrate fractional attention mechanism"""
    print("\n🧠 Demonstrating Fractional Attention...")

    # Create sample data
    batch_size, seq_len, d_model = 2, 10, 16
    x = torch.randn(batch_size, seq_len, d_model)

    # Create fractional attention layer
    attention = FractionalAttention(
        d_model=d_model,
        n_heads=4,
        fractional_order=0.5,
        dropout=0.1
    )

    # Apply fractional attention
    output = attention(x, method="RL")

    print(f"✅ Fractional Attention Applied:")
    print(f"   Input Shape: {x.shape}")
    print(f"   Output Shape: {output.shape}")
    print(f"   Fractional Order: {attention.fractional_order}")

    return attention


def demonstrate_fractional_layers():
    """Demonstrate fractional neural network layers"""
    print("\n🔧 Demonstrating Fractional Layers...")

    from hpfracc.ml.layers import LayerConfig

    # Create configuration for fractional layers
    from hpfracc.core.definitions import FractionalOrder
    config = LayerConfig(
        fractional_order=FractionalOrder(0.5),
        method="RL",
        use_fractional=True
    )

    # Demonstrate 1D Convolution
    conv1d = FractionalConv1D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        config=config
    )

    # Demonstrate 2D Convolution
    conv2d = FractionalConv2D(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        config=config
    )

    # Demonstrate LSTM
    lstm = FractionalLSTM(
        input_size=10,
        hidden_size=32,
        config=config
    )

    # Demonstrate Transformer
    transformer = FractionalTransformer(
        d_model=64,
        nhead=8,
        config=config
    )

    # Demonstrate Pooling
    pooling = FractionalPooling(
        kernel_size=2,
        config=config
    )

    # Demonstrate Batch Normalization
    batchnorm = FractionalBatchNorm1d(
        num_features=64,
        config=config
    )

    print("✅ Fractional Layers Created Successfully!")
    print(f"   Conv1D: {conv1d}")
    print(f"   Conv2D: {conv2d}")
    print(f"   LSTM: {lstm}")
    print(f"   Transformer: {transformer}")
    print(f"   Pooling: {pooling}")
    print(f"   BatchNorm: {batchnorm}")

    # Test forward pass with sample data
    try:
        # Test Conv1D
        x1d = torch.randn(1, 3, 10)
        out1d = conv1d(x1d)
        print(f"   Conv1D Input: {x1d.shape} -> Output: {out1d.shape}")

        # Test Conv2D
        x2d = torch.randn(1, 3, 8, 8)
        out2d = conv2d(x2d)
        print(f"   Conv2D Input: {x2d.shape} -> Output: {out2d.shape}")

        # Test LSTM
        x_lstm = torch.randn(5, 1, 10)  # (seq_len, batch, input_size)
        out_lstm, (h, c) = lstm(x_lstm)
        print(f"   LSTM Input: {x_lstm.shape} -> Output: {out_lstm.shape}")

        # Test BatchNorm
        x_bn = torch.randn(1, 64, 10)
        out_bn = batchnorm(x_bn)
        print(f"   BatchNorm Input: {x_bn.shape} -> Output: {out_bn.shape}")

        print("✅ All fractional layers working correctly!")

    except Exception as e:
        print(f"❌ Error testing layers: {e}")


def main():
    """Main demonstration function"""
    print("🚀 Starting hpfracc ML Integration Demo")
    print("=" * 50)
    print("NOTE: The MLOps features are under development. This demo is a placeholder.")

    # # Setup logging
    # setup_logging()

    # # Set random seeds for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    # try:
    #     # 1. Create sample data
    #     train_data, val_data = create_sample_data()

    #     # 2. Create and train fractional model
    #     model = create_fractional_model()
    #     train_losses, val_losses = train_model(model, train_data, val_data)

    #     # 3. Evaluate model
    #     performance_metrics = evaluate_model(model, val_data)

    #     # 4. Demonstrate model registry
    #     registry, model_id = demonstrate_model_registry(
    #         model, performance_metrics)

    #     # 5. Demonstrate development workflow
    #     demonstrate_development_workflow(registry, model_id, val_data)

    #     # 6. Demonstrate production workflow
    #     prod_workflow = ProductionWorkflow(registry, ModelValidator())
    #     promotion_results = demonstrate_production_workflow(
    #         registry, model_id, val_data)

    #     # 7. Demonstrate model monitoring
    #     if promotion_results['promoted']:
    #         demonstrate_model_monitoring(prod_workflow)

    #     # 8. Demonstrate fractional attention
    #     demonstrate_fractional_attention()

    #     # 9. Demonstrate fractional layers
    #     demonstrate_fractional_layers()

    #     print("\n" + "=" * 50)
    #     print("🎉 ML Integration Demo Completed Successfully!")
    #     print("\n📋 Summary of Features Demonstrated:")
    #     print("   ✅ Fractional Neural Networks")
    #     print("   ✅ Model Registry and Versioning")
    #     print("   ✅ Development vs. Production Workflow")
    #     print("   ✅ Quality Gates and Validation")
    #     print("   ✅ Production Monitoring and Alerts")
    #     print("   ✅ Fractional Attention Mechanisms")
    #     print("   ✅ Fractional Convolutional Layers")
    #     print("   ✅ Fractional LSTM Layers")
    #     print("   ✅ Fractional Loss Functions")
    #     print("   ✅ Fractional Optimizers")

    #     print(f"\n📁 Model files saved to: {Path('models/').absolute()}")
    #     print(f"\n📊 Registry database: {Path('models/registry.db').absolute()}")

    # except Exception as e:
    #     print(f"❌ Error during demo: {e}")
    #     logging.error(f"Demo failed: {e}", exc_info=True)
    #     raise


if __name__ == "__main__":
    main()
