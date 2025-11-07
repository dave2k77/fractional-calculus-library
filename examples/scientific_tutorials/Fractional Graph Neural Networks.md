# Fractional Graph Neural Networks

This tutorial explains how to build and evaluate graph neural networks that leverage fractional calculus operators using hpfracc. The walkthrough is based on `examples/ml_examples/fractional_gnn_demo.py` and related factory utilities.

## 1. Concepts and Toolkit

Fractional GNNs introduce long-range memory into message passing by applying fractional derivatives to node features or attention weights. hpfracc exposes:

- `FractionalGNNFactory.create_model(...)` to instantiate backbone architectures (`gcn`, `gat`, `graphsage`, `gunet`).
- `switch_backend` and `get_backend_manager()` for runtime backend selection.
- Tensor helpers via `get_tensor_ops(backend)` to convert NumPy arrays into backend tensors.

All implementations live under `hpfracc/ml` and mirror classic GNN APIs while injecting fractional operators at layer boundaries.

## 2. Environment and Backend Selection

Install dependencies:

```
pip install hpfracc torch jax jaxlib torch-geometric  # adjust for your hardware
```

Select a backend before constructing models:

```
from hpfracc.ml import BackendType, switch_backend

switch_backend(BackendType.TORCH)  # or BackendType.JAX / BackendType.NUMBA
```

The active backend determines tensor types, dropout behavior, and fractional derivative implementations. The demo script benchmarks all three backends for the same synthetic batch.

## 3. Graph Data Preparation

Create synthetic data mirroring the demo:

```
import numpy as np

def create_synthetic_graph(num_nodes=100, num_features=16, num_classes=4, edge_prob=0.1):
    node_features = np.random.randn(num_nodes, num_features)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                edges.append([i, j])
                edges.append([j, i])
    edge_index = np.array(edges).T if edges else np.array([[0], [0]])
    node_labels = np.random.randint(0, num_classes, size=num_nodes)
    return node_features, edge_index, node_labels
```

Convert to tensors using the backend helper:

```
from hpfracc.ml import get_tensor_ops

tensor_ops = get_tensor_ops()
x = tensor_ops.create_tensor(node_features, dtype='float32')
edge_idx = tensor_ops.create_tensor(edge_index)
labels = tensor_ops.create_tensor(node_labels)
```

## 4. Building Fractional GNN Models

Instantiate a fractional GNN with target fractional order `α`:

```
from hpfracc.ml import FractionalGNNFactory

model = FractionalGNNFactory.create_model(
    model_type='gat',
    input_dim=x.shape[1],
    hidden_dim=32,
    output_dim=4,
    fractional_order=0.5,
    num_layers=3,
)
```

Each architecture integrates fractional derivatives differently:

- **GCN / GraphSAGE**: apply fractional smoothing to aggregated neighborhoods.
- **GAT**: modulate attention coefficients with fractional operators; requires a PRNG key under JAX (`model.forward(x, edge_idx, key=subkey)`).
- **G-UNet**: includes spectral pooling layers benefitting from fractional diffusion.

## 5. Training Loop

For node classification on PyTorch:

```
import torch
import torch.nn.functional as F

def train_node_classifier(model, x, edge_idx, labels, epochs=100, lr=5e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model.forward(x, edge_idx)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            print(f"epoch={epoch:03d} loss={loss.item():.4f} acc={acc.item():.4f}")
```

The forward signature matches standard GNN APIs, so you can plug the model into PyTorch Geometric loaders or custom batching strategies. For batched graphs, ensure the fractional derivative operates on the correct dimension (use the helper utilities to flatten or reshape before calling `fractional_forward`).

## 6. Hyperparameter and Fractional Order Sweeps

Borrow the sweep pattern from `compare_fractional_orders` in the demo:

```
orders = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []
for alpha in orders:
    model = FractionalGNNFactory.create_model(
        model_type='gcn',
        input_dim=x.shape[1],
        hidden_dim=32,
        output_dim=4,
        fractional_order=alpha,
        num_layers=3,
    )
    logits = model.forward(x, edge_idx)
    loss = F.cross_entropy(logits, labels)
    results.append((alpha, float(loss.detach())))
print(results)
```

Key observations:

- Lower `α` values mimic heavier smoothing, improving noisy graphs but potentially hurting sharp community boundaries.
- Higher `α` values increase sensitivity to long-range dependencies, which benefits graphs with hierarchical structure.
- Benchmark across backends to identify the best runtime profile for production (`benchmark_backend_performance` helper).

## 7. Validation and Metrics

Use held-out nodes or entire graphs depending on the task:

- **Node classification**: track accuracy, F1 score, and calibration error on validation nodes.
- **Link prediction**: compute ROC-AUC using fractional embeddings.
- **Graph regression**: measure RMSE/MAE per graph; fractional derivatives often reduce long-term drift.

For lightning-style evaluation, wrap your model in `torch.no_grad()` and align dropout/attention randomness when using fractional operators.

## 8. Deployment and Integration Tips

- Serialize the model with `torch.save(model.state_dict(), path)` (PyTorch) or the backend-equivalent. Store the fractional hyperparameters alongside the checkpoint.
- During inference, call `model.eval()` and, if needed, disable fractional augmentation by passing `use_fractional=False` for baseline comparisons.
- For streaming graph updates, maintain incremental edge batches and re-run fractional smoothing on affected neighborhoods only.
- When using JAX backends, jit-compile the forward pass after selecting `α` to amortize derivative cost.

## 9. Troubleshooting

- **Diverging losses**: check that edge indices are symmetric for undirected graphs; fractional derivatives can accentuate asymmetries.
- **Slow JAX runs**: warm-up with a dummy forward; the demo uses a dedicated key split per iteration.
- **NaN attention weights**: reduce learning rate or clamp fractional outputs before softmax in custom layers.

## 10. Next Steps

- Extend the pipeline with fractional attention from `hpfracc/ml/core.py` for sequence-aware graphs.
- Combine GNN embeddings with fractional recurrent layers to model dynamic graphs.
- Compare fractional vs integer-order baselines using the benchmarking utilities to quantify gains.
