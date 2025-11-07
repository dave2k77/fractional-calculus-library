# Fractional ML Learning Path

Use this roadmap to master fractional machine learning with hpfracc. The pathway moves from foundational feedforward models to graph networks and continuous-time systems, concluding with deployment guidance.

## Stage 0 – Foundations

1. Review hpfracc core APIs in `hpfracc/ml/core.py` and explore `MLConfig`, `FractionalNeuralNetwork`, and backend utilities.
2. Run the bundled demos to verify your environment:
   - `python examples/ml_examples/advanced_unified_training.py`
   - `python examples/ml_examples/fractional_gnn_demo.py`
   - `python examples/physics_examples/fractional_physics_demo.py`
3. Skim `examples/advanced_applications/fractional_pde_solver.py` for solver context.

## Stage 1 – Fractional Feedforward Networks

Follow the tutorial `Fractional Neural Networks in Practice.md`:
- Set up PyTorch and hpfracc.
- Build `FractionalNeuralNetwork` and `SpectralFractionalNetwork` models.
- Implement training/validation loops and save best checkpoints.
- Experiment with fractional orders and backend switches.

Outcome: you can train fractional regression/classification models and interpret fractional features.

## Stage 2 – Graph Learning with Fractional Memory

Study `Fractional Graph Neural Networks.md`:
- Create synthetic or real graph datasets.
- Instantiate fractional GNNs via `FractionalGNNFactory` and manage backends.
- Train node classification models and benchmark fractional orders.
- Validate metrics (accuracy, F1, ROC-AUC) and handle deployment specifics.

Outcome: you can engineer fractional message passing architectures for graph tasks.

## Stage 3 – Continuous-Time Dynamics (ODE/SDE)

Work through `Fractional Neural ODE and SDE Models.md`:
- Construct neural drift/diffusion terms with spectral fractional layers.
- Integrate with ODE/SDE solvers (`torchdiffeq`, `torchsde`, hpfracc predictors).
- Supervise with PDE solver outputs and evaluate trajectory fidelity.
- Implement fractional noise and ensure stochastic stability.

Outcome: you can model long-memory temporal systems with fractional neural differential equations.

## Stage 4 – Experimentation and Deployment

- Combine insights from earlier stages to build end-to-end pipelines.
- Use hpfracc serialization utilities (`FractionalNeuralNetwork.save_model`) to manage checkpoints.
- Design hyperparameter sweeps over fractional orders and backends.
- Integrate physics-informed losses with `PredictorCorrectorSolver` outputs for hybrid models.

## Suggested Progression Checklist

- [ ] Feedforward tutorial completed; baseline and fractional runs compared.
- [ ] Graph tutorial executed with at least two architectures and fractional orders.
- [ ] ODE/SDE tutorial executed with deterministic and stochastic scenarios.
- [ ] Deployment pipeline scripted (load → predict → monitor fractional effects).
- [ ] Documentation updated with observations on best-performing fractional orders.

## Additional Resources

- hpfracc API reference (run `pydoc hpfracc.ml` or inspect source under `hpfracc/ml`).
- Research background stored in `examples/scientific_tutorials/Fractional State Space and Long-Range Dependence i.md`.
- External reading: fractional calculus in machine learning, fractional Brownian motion analytics, and spectral methods for long-memory systems.

Advance through stages sequentially or revisit earlier modules to iterate on fractional order choices as new data arrives.
