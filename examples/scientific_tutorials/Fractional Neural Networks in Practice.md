# Fractional Neural Networks in Practice

Fractional neural networks extend standard deep learning models with fractional calculus operators that capture long-memory behavior. This walkthrough shows how to build, train, validate, and deploy fractional feedforward models using the hpfracc toolkit.

## 1. Prerequisites and Environment Setup

- Install core dependencies: `pip install hpfracc torch numpy matplotlib`
- Optional: enable GPU-backed PyTorch by installing the CUDA-enabled wheel.
- Verify hpfracc imports:

```
python - <<'PY'
from hpfracc.ml import FractionalNeuralNetwork, MLConfig
print('hpfracc ready:', MLConfig())
PY
```

The ML APIs dynamically dispatch to `BackendType` (PyTorch, JAX, or NUMBA). For tutorials we use the torch backend because the training utilities in `examples/ml_examples/advanced_unified_training.py` assume PyTorch tensors.

## 2. Data Preparation

For regression/classification, you can generate synthetic data or wrap existing datasets as tensors. Borrowing from `advanced_unified_training.py`:

```
import torch

def make_regression(n=6000, d=64, noise=0.05):
    x = torch.randn(n, d)
    w = torch.linspace(0.1, 1.0, d)
    y = (x.sin() * w).sum(dim=1, keepdim=True)
    y += noise * torch.randn_like(y)
    return x, y
```

Split data into training and validation loaders:

```
from torch.utils.data import TensorDataset, DataLoader

def make_loaders(x, y, batch_size=256, val_frac=0.2):
    n = x.size(0)
    n_val = int(n * val_frac)
    idx = torch.randperm(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    ds_tr = TensorDataset(x[train_idx], y[train_idx])
    ds_va = TensorDataset(x[val_idx], y[val_idx])
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False),
    )
```

## 3. Building a Fractional Neural Network

The low-level API lives in `hpfracc/ml/core.py`. Instantiate `FractionalNeuralNetwork` with fractional order `α` and desired backend:

```
from hpfracc.ml import FractionalNeuralNetwork, MLConfig, BackendType

config = MLConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    fractional_order=0.6,
    learning_rate=1e-3,
    max_epochs=50,
)
model = FractionalNeuralNetwork(
    input_size=64,
    hidden_sizes=[128, 128],
    output_size=1,
    fractional_order=config.fractional_order,
    activation='relu',
    dropout=0.1,
    config=config,
    backend=BackendType.TORCH,
)
```

`FractionalNeuralNetwork.forward` applies a fractional derivative (default Riemann–Liouville) before the dense stack. You can switch to Caputo or Grünwald–Letnikov by setting `method='Caputo'` or `'GL'` during the forward pass.

### Option: SpectralFractionalNetwork

For spectral layers with learnable fractional orders, reuse the high-level `SpectralFractionalNetwork` from `hpfracc/ml/spectral_autograd.py` as demonstrated in `advanced_unified_training.py`:

```
from hpfracc.ml.spectral_autograd import SpectralFractionalNetwork

spectral_model = SpectralFractionalNetwork(
    input_dim=64,
    hidden_dims=[128, 128],
    output_dim=1,
    alpha=0.6,
    mode='unified',  # couples time and spatial fractional ops
)
```

## 4. Training and Validation Loop

The `SpectralFractionalNetwork` behaves like a standard PyTorch module. Combine with an optimizer and loss function:

```
import torch.nn as nn

def train_fractional(model, train_loader, val_loader, epochs=10, lr=1e-3, ckpt='./fnn_best.pt'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    device = next(model.parameters()).device

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * xb.size(0)
            seen += xb.size(0)
        train_mse = running / max(seen, 1)

        val_mse = evaluate(model, val_loader, loss_fn)  # reuse helper from example
        print(f"epoch={epoch:02d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), ckpt)
    return best_val
```

The `evaluate` helper mirrors the implementation in the example script and runs the model in evaluation mode without gradient tracking.

## 5. Inspecting Fractional Effects

To understand the influence of `α`:

1. Sweep fractional orders using a validation hold-out.
2. Compare training curves with `use_fractional=False` to recover the integer-order baseline.
3. Examine weight tensors and the `fractional_forward` output to ensure values remain finite.

```
with torch.no_grad():
    sample = next(iter(train_loader))[0][:8]
    baseline = spectral_model(sample, use_fractional=False)
    fractional = spectral_model(sample, use_fractional=True)
    print('baseline mean', baseline.mean().item(), 'fractional mean', fractional.mean().item())
```

## 6. Saving, Loading, and Deployment

- Call `torch.save(model.state_dict(), path)` for PyTorch-based models. The `FractionalNeuralNetwork.save_model` method serializes weights and a JSON config, enabling backend-agnostic reloads with `FractionalNeuralNetwork.load_model(path)`.
- During deployment, wrap inference in a function that reloads the model and precomputes the fractional derivative on the target inputs:

```
def load_for_inference(path, backend=BackendType.TORCH):
    model = FractionalNeuralNetwork.load_model(path)
    model.backend = backend
    return model
```

- For real-time scoring, keep the fractional buffers on the same device as your streaming inputs and disable dropout by calling `model.eval()`.

## 7. Troubleshooting Checklist

- **Gradient explosions**: lower `fractional_order` or learning rate; the spectral operators can amplify high-frequency noise.
- **Slow training**: switch to the NUMBA backend for CPU-only workloads, or reduce fractional derivative precision by subsampling the input sequence before the derivative.
- **NaNs**: ensure input tensors are contiguous and finite before the fractional transform.
- **Deployment drift**: monitor outputs from `fractional_forward` to confirm the order `α` still matches trained values.

## 8. Next Steps

- Explore attention-based architectures via `FractionalAttention` in `hpfracc/ml/core.py` for sequence modeling.
- Integrate AutoML sweeps over `fractional_order` using the hpfracc configuration objects.
- Continue with the graph and continuous-time tutorials to extend fractional insights beyond feedforward networks.

