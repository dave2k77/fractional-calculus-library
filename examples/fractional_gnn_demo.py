"""
Fractional Graph Neural Network Demo

This demo showcases the new fractional GNN architecture with:
- Multi-backend support (PyTorch, JAX, NUMBA)
- Various GNN architectures (GCN, GAT, GraphSAGE, U-Net)
- Fractional calculus integration
- Performance comparison across backends
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our fractional GNN components
from hpfracc.ml import (
    BackendType, get_backend_manager, switch_backend,
    FractionalGNNFactory, get_tensor_ops
)


def create_synthetic_graph_data(
    num_nodes: int = 100,
    num_features: int = 16,
    num_classes: int = 4,
    edge_probability: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic graph data for demonstration
    
    Args:
        num_nodes: Number of nodes in the graph
        num_features: Number of features per node
        num_classes: Number of classes for classification
        edge_probability: Probability of edge between any two nodes
    
    Returns:
        Tuple of (node_features, edge_index, node_labels)
    """
    # Create random node features
    node_features = np.random.randn(num_nodes, num_features)
    
    # Create random edge connections
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.random() < edge_probability:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph
    
    edge_index = np.array(edges).T if edges else np.array([[0], [0]])
    
    # Create random node labels
    node_labels = np.random.randint(0, num_classes, num_nodes)
    
    return node_features, edge_index, node_labels


def benchmark_backend_performance(
    backend: BackendType,
    model_type: str,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Benchmark performance of a specific backend and model type
    
    Args:
        backend: Backend to test
        model_type: Type of GNN model
        node_features: Node feature matrix
        edge_index: Graph connectivity
        num_runs: Number of runs for averaging
    
    Returns:
        Dictionary with performance metrics
    """
    print(f"🔄 Testing {backend.value.upper()} backend with {model_type.upper()} model...")
    
    try:
        # Switch to the specified backend
        switch_backend(backend)
        
        # Create model
        model = FractionalGNNFactory.create_model(
            model_type=model_type,
            input_dim=node_features.shape[1],
            hidden_dim=32,
            output_dim=4,
            fractional_order=0.5,
            num_layers=3
        )
        
        # Get tensor operations for the current backend
        tensor_ops = get_tensor_ops(backend)
        
        # Convert data to backend tensors
        x = tensor_ops.create_tensor(node_features)
        edge_idx = tensor_ops.create_tensor(edge_index)
        
        # Warm-up run
        _ = model.forward(x, edge_idx)
        
        # Benchmark runs
        forward_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = model.forward(x, edge_idx)
            forward_times.append(time.time() - start_time)
        
        avg_time = np.mean(forward_times)
        std_time = np.std(forward_times)
        
        print(f"   ✅ {backend.value.upper()} + {model_type.upper()}: "
              f"{avg_time:.4f}s ± {std_time:.4f}s")
        
        return {
            'backend': backend.value,
            'model_type': model_type,
            'avg_time': avg_time,
            'std_time': std_time,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ {backend.value.upper()} + {model_type.upper()}: Failed - {str(e)}")
        return {
            'backend': backend.value,
            'model_type': model_type,
            'avg_time': float('inf'),
            'std_time': 0.0,
            'success': False,
            'error': str(e)
        }


def compare_fractional_orders(
    model_type: str,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    fractional_orders: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
) -> Dict[str, List[float]]:
    """
    Compare performance across different fractional orders
    
    Args:
        model_type: Type of GNN model
        node_features: Node feature matrix
        edge_index: Graph connectivity
        fractional_orders: List of fractional orders to test
    
    Returns:
        Dictionary with timing results for each order
    """
    print(f"\n🔬 Comparing fractional orders for {model_type.upper()} model...")
    
    results = {'orders': fractional_orders, 'times': []}
    
    for alpha in fractional_orders:
        try:
            # Create model with specific fractional order
            model = FractionalGNNFactory.create_model(
                model_type=model_type,
                input_dim=node_features.shape[1],
                hidden_dim=32,
                output_dim=4,
                fractional_order=alpha,
                num_layers=3
            )
            
            # Get tensor operations
            tensor_ops = get_tensor_ops()
            
            # Convert data
            x = tensor_ops.create_tensor(node_features)
            edge_idx = tensor_ops.create_tensor(edge_index)
            
            # Time the forward pass
            start_time = time.time()
            _ = model.forward(x, edge_idx)
            forward_time = time.time() - start_time
            
            results['times'].append(forward_time)
            print(f"   α={alpha}: {forward_time:.4f}s")
            
        except Exception as e:
            print(f"   α={alpha}: Failed - {str(e)}")
            results['times'].append(float('inf'))
    
    return results


def visualize_performance_comparison(results: List[Dict[str, float]]) -> None:
    """
    Visualize performance comparison across backends and models
    
    Args:
        results: List of benchmark results
    """
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful results to visualize")
        return
    
    # Prepare data for plotting
    backends = list(set(r['backend'] for r in successful_results))
    model_types = list(set(r['model_type'] for r in successful_results))
    
    # Create performance matrix
    performance_matrix = np.zeros((len(backends), len(model_types)))
    
    for result in successful_results:
        i = backends.index(result['backend'])
        j = model_types.index(result['model_type'])
        performance_matrix[i, j] = result['avg_time']
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    im = ax1.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(model_types)))
    ax1.set_yticks(range(len(backends)))
    ax1.set_xticklabels([m.upper() for m in model_types])
    ax1.set_yticklabels([b.upper() for b in backends])
    ax1.set_title('Performance Heatmap (Lower is Better)')
    
    # Add text annotations
    for i in range(len(backends)):
        for j in range(len(model_types)):
            text = ax1.text(j, i, f'{performance_matrix[i, j]:.3f}s',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Bar chart
    x = np.arange(len(successful_results))
    times = [r['avg_time'] for r in successful_results]
    labels = [f"{r['backend'].upper()}-{r['model_type'].upper()}" for r in successful_results]
    
    bars = ax2.bar(x, times, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Backend-Model Combinations')
    ax2.set_ylabel('Average Forward Pass Time (s)')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fractional_gnn_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_fractional_effects(
    model_type: str,
    node_features: np.ndarray,
    edge_index: np.ndarray
) -> None:
    """
    Demonstrate the effects of different fractional orders
    
    Args:
        model_type: Type of GNN model
        node_features: Node feature matrix
        edge_index: Graph connectivity
    """
    print(f"\n🎯 Demonstrating fractional calculus effects with {model_type.upper()}...")
    
    fractional_orders = [0.0, 0.25, 0.5, 0.75, 1.0]
    output_features = []
    
    for alpha in fractional_orders:
        try:
            # Create model
            model = FractionalGNNFactory.create_model(
                model_type=model_type,
                input_dim=node_features.shape[1],
                hidden_dim=32,
                output_dim=4,
                fractional_order=alpha,
                num_layers=3
            )
            
            # Get tensor operations
            tensor_ops = get_tensor_ops()
            
            # Convert data
            x = tensor_ops.create_tensor(node_features)
            edge_idx = tensor_ops.create_tensor(edge_index)
            
            # Forward pass
            output = model.forward(x, edge_idx)
            
            # Convert to numpy for analysis
            if hasattr(output, 'numpy'):
                output_np = output.numpy()
            elif hasattr(output, 'cpu'):
                output_np = output.cpu().numpy()
            else:
                output_np = np.array(output)
            
            output_features.append(output_np)
            
            # Calculate statistics
            mean_val = np.mean(output_np)
            std_val = np.std(output_np)
            max_val = np.max(output_np)
            min_val = np.min(output_np)
            
            print(f"   α={alpha}: mean={mean_val:.4f}, std={std_val:.4f}, "
                  f"range=[{min_val:.4f}, {max_val:.4f}]")
            
        except Exception as e:
            print(f"   α={alpha}: Failed - {str(e)}")
            output_features.append(None)
    
    # Visualize the effects
    if all(f is not None for f in output_features):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (alpha, features) in enumerate(zip(fractional_orders, output_features)):
            if i < len(axes):
                ax = axes[i]
                
                # Histogram of output features
                ax.hist(features.flatten(), bins=30, alpha=0.7, color='skyblue')
                ax.set_title(f'α = {alpha}')
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # Remove extra subplot
        if len(axes) > len(fractional_orders):
            axes[-1].remove()
        
        plt.suptitle(f'Fractional Calculus Effects on {model_type.upper()} Outputs', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'fractional_effects_{model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main demonstration function"""
    print("🚀 Fractional Graph Neural Network Demo")
    print("=" * 50)
    
    # Create synthetic data
    print("\n📊 Creating synthetic graph data...")
    node_features, edge_index, node_labels = create_synthetic_graph_data(
        num_nodes=200,
        num_features=16,
        num_classes=4,
        edge_probability=0.15
    )
    
    print(f"   Graph: {node_features.shape[0]} nodes, {edge_index.shape[1]} edges")
    print(f"   Features: {node_features.shape[1]} dimensions")
    print(f"   Classes: {len(np.unique(node_labels))}")
    
    # Get available backends
    backend_manager = get_backend_manager()
    available_backends = backend_manager.available_backends
    print(f"\n🔧 Available backends: {[b.value for b in available_backends]}")
    
    # Get available models
    available_models = FractionalGNNFactory.get_available_models()
    print(f"🏗️  Available models: {available_models}")
    
    # Benchmark performance across backends and models
    print("\n⚡ Performance Benchmarking")
    print("-" * 30)
    
    results = []
    for backend in available_backends:
        for model_type in available_models:
            result = benchmark_backend_performance(backend, model_type, node_features, edge_index)
            results.append(result)
    
    # Visualize performance comparison
    visualize_performance_comparison(results)
    
    # Demonstrate fractional calculus effects
    demonstrate_fractional_effects('gcn', node_features, edge_index)
    
    # Show model information
    print("\n📚 Model Information")
    print("-" * 20)
    
    for model_type in available_models:
        info = FractionalGNNFactory.get_model_info(model_type)
        print(f"\n{info['name']} ({model_type.upper()})")
        print(f"   Description: {info['description']}")
        print(f"   Best for: {', '.join(info['best_for'])}")
        print(f"   Complexity: {info['complexity']}")
        print(f"   Memory efficient: {info['memory_efficient']}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Multi-backend support (PyTorch, JAX, NUMBA)")
    print("   • Fractional calculus integration")
    print("   • Multiple GNN architectures")
    print("   • Performance benchmarking")
    print("   • Fractional order effects")
    
    print("\n📁 Generated files:")
    print("   • fractional_gnn_performance_comparison.png")
    print("   • fractional_effects_gcn.png")


if __name__ == "__main__":
    main()
