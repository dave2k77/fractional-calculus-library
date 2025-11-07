# Advanced Fractional Neural Networks: Probabilistic Orders, Stochastic Memory, and Variance-Aware Training

This walkthrough combines hpfracc's advanced capabilities—probabilistic fractional orders, stochastic memory sampling, and variance-aware training—into a single end-to-end workflow. The goal is to design, train, and deploy a fractional neural network that explicitly models uncertainty in the fractional order, reduces computational cost via stochastic memory, and stabilizes training with variance monitoring.

## 1. Prerequisites and Environment Setup

1. Install core dependencies:
   - `pip install hpfracc torch numpy matplotlib`
   - Optional probabilistic backend: `pip install numpyro jax jaxlib` (required if you plan to run NumPyro-backed probabilistic orders).
2. Inspect the reference implementations:
   - Probabilistic fractional orders: `hpfracc/ml/probabilistic_fractional_orders.py`
   - Stochastic memory sampling: `hpfracc/ml/stochastic_memory_sampling.py`
   - Variance-aware training utilities: `hpfracc/ml/variance_aware_training.py`
3. Run the standalone demo to confirm the toolchain works:
   - `python examples/ml_examples/variance_aware_training_example.py`

Throughout this tutorial we assume PyTorch is the active backend. Adjust to JAX as needed for NumPyro SVI pipelines.

## 2. Conceptual Building Blocks

**Probabilistic Fractional Orders** (`ProbabilisticFractionalOrder`, `create_normal_alpha_layer`)
- Treat the fractional order `α` as a random variable with learnable distribution parameters.
- Supports torch distributions (Normal, Uniform, Beta) with reparameterized sampling (`rsample`) for gradient flow.
- Optional NumPyro backend enables full Bayesian inference via SVI (guide/model pattern in the module).

**Stochastic Memory Sampling** (`StochasticFractionalLayer`, `ImportanceSampler`, `StratifiedSampler`, `ControlVariateSampler`)
- Approximates fractional derivatives by sampling from the history kernel instead of aggregating the entire past.
- Importance sampling uses power-law weighting; stratified and control-variate samplers reduce variance further.
- Parameter `k` controls the number of sampled memory points per step; adaptive strategies can adjust `k` during training.

**Variance-Aware Training** (`VarianceAwareTrainer`, `VarianceMonitor`, `AdaptiveSamplingManager`)
- Registers hooks on layers containing `stochastic`, `probabilistic`, or `fractional` in their name to track output and gradient variance.
- Provides callbacks for logging, seed management, adaptive sampling, and variance-driven warnings.
- `create_variance_aware_trainer` builds a trainer that integrates all components into a standard PyTorch loop.

## 3. Architecture Blueprint

We combine deterministic dense layers with stochastic and probabilistic fractional modules. Start from the reference in `examples/ml_examples/variance_aware_training_example.py` and adapt it for a production-grade use case:

```
import torch
import torch.nn as nn
from hpfracc.ml.stochastic_memory_sampling import StochasticFractionalLayer
from hpfracc.ml.probabilistic_fractional_orders import create_normal_alpha_layer

class AdvancedFractionalNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, alpha_prior=(0.6, 0.08),
                 stochastic_k=16, sampler="importance"):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.stochastic_layer = StochasticFractionalLayer(
            alpha=alpha_prior[0],
            k=stochastic_k,
            method=sampler,
            tau=0.2,
        )
        self.prob_layer = create_normal_alpha_layer(
            mean=alpha_prior[0],
            std=alpha_prior[1],
            learnable=True,
        )
        # Combine original features with two fractional channels
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        stoch = self.stochastic_layer(features)
        if stoch.dim() == 2:
            stoch = stoch.mean(dim=1, keepdim=True)
        elif stoch.dim() == 1:
            stoch = stoch.unsqueeze(-1)

        prob = self.prob_layer(features)
        if prob.dim() == 2:
            prob = prob.mean(dim=1, keepdim=True)
        elif prob.dim() == 1:
            prob = prob.unsqueeze(-1)

        combined = torch.cat([features, stoch, prob], dim=-1)
        return self.head(combined)
```

Key practices:
- Average fractional outputs across feature dimensions to keep shapes consistent.
- Maintain differentiability by ensuring `rsample` or small residual connections (see `probabilistic_fractional_orders.py` lines 198–205).
- Use descriptive module names (`stochastic_layer`, `prob_layer`) so variance hooks auto-register.

## 4. Data and Loader Setup

Construct synthetic regression data that benefits from memory effects:

```
def generate_sequence_batch(n_samples=1024, input_dim=32, noise=0.05):
    x = torch.randn(n_samples, input_dim)
    memory_signal = torch.cumsum(x[..., :4], dim=-1)
    y = (
        0.7 * memory_signal.mean(dim=-1, keepdim=True)
        + 0.3 * torch.sin(x[:, 4:5])
        + noise * torch.randn(n_samples, 1)
    )
    return x, y

from torch.utils.data import TensorDataset, DataLoader

def build_loaders(batch=64):
    x_train, y_train = generate_sequence_batch()
    x_val, y_val = generate_sequence_batch(n_samples=256)
    ds_tr = TensorDataset(x_train, y_train)
    ds_va = TensorDataset(x_val, y_val)
    return (
        DataLoader(ds_tr, batch_size=batch, shuffle=True),
        DataLoader(ds_va, batch_size=batch, shuffle=False),
    )
```

## 5. Variance-Aware Training Loop

The central utility is `create_variance_aware_trainer` from `hpfracc/ml/variance_aware_training.py`, which wraps `VarianceAwareTrainer` with sensible defaults:

```
from hpfracc.ml.variance_aware_training import create_variance_aware_trainer

model = AdvancedFractionalNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
train_loader, val_loader = build_loaders()

trainer = create_variance_aware_trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    base_seed=123,
    variance_threshold=0.15,
    log_interval=5,
    adaptive_k_bounds=(8, 128),
)

results = trainer.train(train_loader, num_epochs=30)
```

What happens internally:
- `VarianceMonitor.update` tracks output, loss, and gradient variance per component; warnings trigger if coefficient of variation (CV) exceeds thresholds (see lines 104–141 of `variance_aware_training.py`).
- `StochasticSeedManager` rotates seeds per batch to decorrelate stochastic draws.
- `AdaptiveSamplingManager.update_k` adjusts the number of memory samples `k` when variance remains high.
- Optional callbacks (logging, early stopping) can be appended to `trainer.callbacks`.

After each epoch the trainer returns a summary containing losses, variance history, and adaptive sampling statistics. You can reuse `plot_training_results` from the example script to visualize stability over time.

## 6. Monitoring and Diagnostics

### 6.1 Variance Reports

```
variance_summary = results['variance_history'][-1]
for name, metrics in variance_summary.items():
    print(name, metrics['cv'], metrics['variance'])
```

High CV for `prob_layer_output` suggests the alpha distribution is too wide; reduce `std` or apply regularization. Persistent warnings from `grad_linear.weight` indicate exploding gradients—tune the optimizer or clipping.

### 6.2 Adaptive Sampling Log

```
adaptive_history = trainer.adaptive_sampling.k_history
print('K schedule:', adaptive_history[:10])
```

Check that `k` increases during volatile epochs and shrinks when variance stabilizes. If `k` oscillates strongly, increase `variance_threshold` or widen `(min_k, max_k)` bounds.

### 6.3 Probabilistic Alpha Statistics

```
stats = model.prob_layer.probabilistic_order.get_alpha_statistics()
print('alpha mean', stats['mean'].item(), 'alpha std', stats['std'].item())
```

This helps determine whether the learned distribution stays within physical constraints (e.g., `α ∈ (0, 2)`).

## 7. Validation and Calibration

1. **Hold-out evaluation:**
   ```
   model.eval()
   with torch.no_grad():
       val_loss = 0.0
       for xb, yb in val_loader:
           pred = model(xb)
           val_loss += loss_fn(pred, yb).item()
   val_loss /= len(val_loader)
   print('Validation MSE:', val_loss)
   ```
2. **Uncertainty calibration:** sample multiple stochastic forward passes:
   ```
   def mc_predict(model, x, n=30):
       preds = []
       model.train()  # enable stochasticity
       for _ in range(n):
           preds.append(model(x).detach())
       return torch.stack(preds)
   preds = mc_predict(model, next(iter(val_loader))[0][:16])
   mean_pred = preds.mean(dim=0)
   pred_std = preds.std(dim=0)
   ```
3. **Alpha posterior check:** ensure samples from `prob_layer.sample_alpha()` remain within expected range. Filter out-of-bounds draws by tightening the prior or clipping in deployment.

## 8. Deployment Playbook

- **Checkpointing:** save both the model weights and the probabilistic parameters. Torch layers created via `create_normal_alpha_layer` expose learnable `loc`/`scale` parameters automatically.
- **Deterministic inference:** switch to evaluation mode and optionally freeze stochastic effects:
  ```
  model.eval()
  with torch.no_grad():
      deterministic = model.prob_layer.probabilistic_order.sample(k=1).mean()
      model.prob_layer.probabilistic_order._torch_dist = \
          torch.distributions.Delta(deterministic)
  ```
- **Variance hooks in production:** reuse `VarianceMonitor` to flag drift; integrate with logging/alerting to detect shifts in fractional behavior.
- **Adaptive sampling for latency budgeting:** persist the final `k` from `AdaptiveSamplingManager` and clamp it during real-time inference to guarantee runtime bounds.

## 9. Extensions and Experiments

- **Alternative distributions:** swap in `create_uniform_alpha_layer` or `create_beta_alpha_layer` to enforce bounded support; compare learning dynamics.
- **Hybrid samplers:** combine `ControlVariateSampler` for long histories with stratified sampling on recent windows to balance bias and variance.
- **Bayesian SVI:** if NumPyro is available, initialize `ProbabilisticFractionalLayer` with the SVI backend to obtain full posterior samples of `α`.
- **Graph or sequence models:** inject the same fractional modules into GNN or Transformer architectures; reuse `VarianceAwareTrainer` for stability.
- **Variance-aware regularization:** augment the loss with CV penalties using metrics from `VarianceMonitor` to discourage high-variance components.

## 10. Checklist for Advanced Fractional Projects

- [ ] Environment validated with variance-aware example.
- [ ] Architecture integrates stochastic memory and probabilistic alpha.
- [ ] Variance monitoring logs show controlled CV (< 0.3 for key components).
- [ ] Adaptive sampling converges to efficient `k` without oscillation.
- [ ] Probabilistic alpha statistics remain within domain-specific bounds.
- [ ] Monte Carlo predictions exhibit calibrated uncertainty on validation data.
- [ ] Deployment script captures learned distribution parameters and variance guardrails.

This integrative pipeline demonstrates how hpfracc handles uncertainty modeling, memory compression, and training robustness in one workflow. Adapt these patterns to your domain-specific datasets to unlock advanced fractional machine learning capabilities.
