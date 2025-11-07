# Fractional Neural ODE and SDE Models

This tutorial walks through continuous-time modeling with fractional calculus using hpfracc. You will learn how to construct neural ODE/SDE systems, integrate fractional derivatives, and validate stochastic simulations. The examples draw from `examples/physics_examples/fractional_physics_demo.py` and `examples/advanced_applications/fractional_pde_solver.py`.

## 1. Why Fractional Dynamics?

Fractional neural ODEs/SDEs extend classic continuous-time models by incorporating memory kernels. Instead of relying on integer-order derivatives, hpfracc provides spectral fractional operators (`spectral_fractional_derivative`, `SpectralFractionalLayer`) and solver utilities (`PredictorCorrectorSolver`, `FractionalDiffusionSolver`). These components capture long-range dependence and heavy-tailed phenomena present in physical, biological, and financial systems.

## 2. Environment Setup

- Install hpfracc and PyTorch: `pip install hpfracc torch matplotlib`
- Optional: install `torchdiffeq` or `diffrax` if you need advanced ODE solvers.
- Ensure GPU support if your workloads are large: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

```
python - <<'PY'
import torch
from hpfracc.ml.spectral_autograd import spectral_fractional_derivative
print('hpfracc spectral derivative ready:', spectral_fractional_derivative(torch.randn(8), alpha=0.5))
PY
```

## 3. Fractional Neural ODE Workflow

### 3.1 Define State Dynamics

Use a neural network to parameterize the drift term of a fractional-order ODE:

```
import torch
import torch.nn as nn
from hpfracc.ml.spectral_autograd import SpectralFractionalLayer

class FractionalDrift(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, alpha=0.8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.frac = SpectralFractionalLayer(alpha=alpha, learnable_alpha=False)

    def forward(self, t, y):
        # Apply fractional derivative to capture memory in state representation
        frac_state = self.frac(y.unsqueeze(0)).squeeze(0)
        return self.net(frac_state)
```

`SpectralFractionalLayer` applies a spectral derivative along the last dimension. In `fractional_physics_demo.py`, similar layers operate on flattened spatiotemporal grids before reshaping back to the original domain.

### 3.2 Integrate with an ODE Solver

You can embed the fractional drift inside a standard ODE solver such as `torchdiffeq.odeint`:

```
from torchdiffeq import odeint

def solve_fractional_ode(model, y0, t_span):
    with torch.no_grad():
        sol = odeint(model, y0, t_span)
    return sol

state_dim = 4
y0 = torch.randn(state_dim)
t_span = torch.linspace(0, 5, steps=200)
model = FractionalDrift(state_dim, alpha=0.6)
solution = solve_fractional_ode(model, y0, t_span)
```

Even though the solver treats the system as first-order, the fractional layer injects history dependence into each evaluation by convolving the state with a fractional kernel.

### 3.3 Training via Continuous-Time Losses

To fit parameters, minimize a loss between simulated trajectories and observations:

```
def ct_loss(model, t_obs, y_obs):
    pred = odeint(model, y_obs[0], t_obs)
    return torch.mean((pred - y_obs) ** 2)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(1000):
    optimizer.zero_grad()
    loss = ct_loss(model, t_obs, y_obs)
    loss.backward()
    optimizer.step()
```

When training, keep the fractional order `α` fixed or learnable (`learnable_alpha=True`). If the order is learnable, monitor it for stability; clip to `[0, 1.9]` using custom constraints if necessary.

## 4. Fractional Neural SDE Workflow

### 4.1 Stochastic Drift and Diffusion

Combine deterministic drift with fractional noise using Brownian increments and a fractional diffusion term:

```
class FractionalSDE(nn.Module):
    def __init__(self, state_dim, alpha=0.7, sigma=0.1):
        super().__init__()
        self.drift = FractionalDrift(state_dim, hidden_dim=128, alpha=alpha)
        self.sigma = sigma

    def f(self, t, y):  # drift
        return self.drift(t, y)

    def g(self, t, y):  # diffusion
        return self.sigma * torch.ones_like(y)
```

To integrate, use an SDE solver (e.g., `torchsde.sdeint`) and add a fractional perturbation by filtering Brownian increments through `spectral_fractional_derivative` before applying them. Pseudocode:

```
import torchsde

class FractionalBrownianMotion(torchsde.BrownianInterval):
    def evaluate(self, t):
        raw = super().evaluate(t)
        return spectral_fractional_derivative(raw, alpha=0.6)

brownian = FractionalBrownianMotion(t0=0.0, t1=1.0, size=y0.shape)
sol = torchsde.sdeint(FractionalSDE(state_dim=4), y0, ts=t_span, bm=brownian)
```

This setup injects memory into the noise path, approximating a fractional Brownian motion. Validate that the increments maintain the correct variance structure after transformation.

## 5. PDE-Informed Training

The PDE solver examples show how to supervise neural ODE/SDE models with physics constraints.

- `FractionalDiffusionSolver.solve` returns spatial-temporal grids for diffusion equations. Use these grids as supervision for data-driven fractional ODEs by sampling trajectories along spatial slices.
- The anomalous diffusion demo reshapes fields and applies `spectral_fractional_derivative` along time before computing losses. Replicate that pattern when coupling neural ODEs with PDE data.

Example: fit a neural drift to replicate solver output at location `x₀`.

```
x, t, u = FractionalDiffusionSolver().solve(...)
obs = torch.tensor(u[:, idx_x0], dtype=torch.float32)
t_obs = torch.tensor(t, dtype=torch.float32)
loss = ct_loss(model, t_obs, obs)
```

## 6. Validation Strategies

- **Trajectory reconstruction**: compute MSE/MAE between simulated and observed trajectories; consider weighting early/late times differently to account for memory fade.
- **Spectral diagnostics**: apply `spectral_fractional_derivative` to predictions and compare with analytical derivatives from solvers.
- **Stochastic checks**: for SDEs, verify that empirical Hurst exponents match the target fractional order using rescaled range analysis.

## 7. Deployment Checklist

1. Export trained weights with `torch.save(model.state_dict(), path)` and persist fractional hyperparameters (`alpha`, solver tolerances).
2. For inference, wrap the ODE solver call inside a function that reuses a cached fractional layer to avoid recomputing weight initializations.
3. Profiling tips:
   - Precompile JAX versions of spectral operators if using that backend.
   - On GPU, keep fractional kernels in contiguous memory to prevent slow reshapes.
4. For probabilistic forecasts, sample multiple stochastic trajectories and summarize quantiles; fractional noise often widens uncertainty bands.

## 8. Troubleshooting

- **Instability in ODE solvers**: reduce step size or use adaptive solvers with tight tolerances; fractional layers can introduce stiffness.
- **NaNs in spectral derivatives**: ensure inputs are normalized and avoid zeros when taking logarithms (see handling in `fractional_physics_demo.py`).
- **Slow SDE simulations**: downsample fractional Brownian increments or precompute kernel matrices for reuse.

## 9. Further Exploration

- Combine fractional neural ODEs with attention (`FractionalAttention` from `hpfracc/ml/core.py`) to model externally forced systems.
- Investigate predictor-corrector integrators (`PredictorCorrectorSolver`) for training stability under high orders.
- Integrate PDE constraints directly into the loss via physics-informed neural networks, using solver outputs as soft penalties.
