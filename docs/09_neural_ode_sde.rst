Neural Fractional ODEs and SDEs
=================================

HPFRACC provides comprehensive frameworks for learning-based solution of fractional ODEs and SDEs with neural networks. This chapter consolidates both Neural fODE and Neural fSDE frameworks.

For complete API documentation, see :doc:`api/neural_ode_sde_api`.

Neural Fractional ODEs
-----------------------

The Neural fODE framework provides learning-based solution of fractional differential equations.

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import hpfracc.ml.neural_ode as nfode
   import torch

   # Create a neural ODE model
   model = nfode.NeuralODE(
       input_dim=2,
       hidden_dim=32,
       output_dim=1,
       num_layers=3,
       activation="tanh"
   )

   # Forward pass
   x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Batch of initial conditions
   t = torch.linspace(0, 1, 100)                # Time points
   solution = model(x0, t)
   print(f"Solution shape: {solution.shape}")  # (batch_size, time_steps, output_dim)

Fractional Neural ODE
~~~~~~~~~~~~~~~~~~~~~~

Extend to fractional calculus with configurable fractional order:

.. code-block:: python

   # Fractional neural ODE
   fode_model = nfode.NeuralFODE(
       input_dim=2,
       hidden_dim=32,
       output_dim=1,
       fractional_order=0.5,  # Fractional order α
       num_layers=3,
       activation="tanh"
   )

   solution = fode_model(x0, t)

For detailed documentation, see :doc:`neural_fode_guide`.

Neural Fractional SDEs
----------------------

Neural Fractional Stochastic Differential Equations (Neural fSDEs) combine neural networks with fractional calculus and stochastic dynamics.

Introduction
~~~~~~~~~~~~

Neural fSDEs extend neural ODEs by incorporating:
- **Stochasticity**: Random noise terms for modeling uncertainty
- **Memory effects**: Fractional derivatives capture long-range temporal dependencies
- **Learnable dynamics**: Neural networks parameterize drift and diffusion functions

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

A neural fractional SDE takes the form:

.. math::

   D_t^\alpha X(t) = f_\theta(t, X(t)) dt + g_\theta(t, X(t)) dW(t)

where:
- :math:`\alpha \in (0, 2)` is the fractional order
- :math:`D_t^\alpha` is the Caputo or Riemann-Liouville fractional derivative
- :math:`f_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d}` is the learnable drift function (neural network)
- :math:`g_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d \times m}` is the learnable diffusion function
- :math:`W(t)` is a Wiener process (or more general noise)

**Note on Diffusion Functions**: Currently supported diffusion types:
- **Scalar diffusion** (additive noise): :math:`g_\theta(t, X(t)) = \sigma \in \mathbb{R}`
- **Vector diffusion** (diagonal multiplicative noise): :math:`g_\theta(t, X(t)) = \sigma(t, X(t)) \in \mathbb{R}^{d}`
- **Matrix diffusion** (full multiplicative noise): :math:`g_\theta: \mathbb{R}^{d} \to \mathbb{R}^{d \times d}` - *Not yet implemented*

For matrix diffusion, consider using diagonal approximation or standard (non-fractional) SDE solvers first.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import torch
   from hpfracc.ml.neural_fsde import create_neural_fsde

   # Create a simple neural fSDE
   model = create_neural_fsde(
       input_dim=2,
       output_dim=2,
       hidden_dim=64,
       fractional_order=0.5,
       noise_type="additive"
   )

   # Forward pass
   x0 = torch.randn(32, 2)
   t = torch.linspace(0, 1, 50)
   trajectory = model(x0, t, method="euler_maruyama", num_steps=50)

   print(f"Trajectory shape: {trajectory.shape}")  # (32, 2)

Training Example
~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch.nn as nn

   model = create_neural_fsde(input_dim=2, output_dim=2, fractional_order=0.5)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   loss_fn = nn.MSELoss()

   # Training loop
   for epoch in range(100):
       optimizer.zero_grad()
       pred = model(x0, t)
       loss = loss_fn(pred, target)
       loss.backward()
       optimizer.step()
       
       if epoch % 10 == 0:
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Fractional Orders in SDEs
~~~~~~~~~~~~~~~~~~~~~~~~~

The fractional order :math:`\alpha` controls memory effects:

- :math:`\alpha \to 0`: Nearly instantaneous response (no memory)
- :math:`\alpha = 0.5`: Subdiffusion (slower than normal diffusion)
- :math:`\alpha = 1`: Standard first-order dynamics
- :math:`\alpha = 1.5`: Superdiffusion (faster than normal)
- :math:`\alpha \to 2`: Wave-like behavior

Drift and Diffusion Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Drift** :math:`f_\theta`: Determines deterministic dynamics

.. code-block:: python

   # Custom drift network
   drift_net = nn.Sequential(
       nn.Linear(3, 64),  # 2 features + time
       nn.Tanh(),
       nn.Linear(64, 2)
   )

   model = create_neural_fsde(
       input_dim=2,
       output_dim=2,
       drift_net=drift_net  # Use custom network
   )

**Diffusion** :math:`g_\theta`: Controls stochastic noise magnitude

Stochastic Noise Modeling
-------------------------

Choose noise type based on problem:

.. code-block:: python

   from hpfracc.solvers import BrownianMotion, FractionalBrownianMotion

   # Standard Brownian motion (independent increments)
   brownian = BrownianMotion(scale=1.0)

   # Fractional Brownian motion (correlated increments, Hurst H)
   fbm = FractionalBrownianMotion(hurst=0.7, scale=1.0)

Adjoint Training Methods
-------------------------

Efficient gradient computation using adjoint methods:

.. code-block:: python

   model = create_neural_fsde(
       input_dim=2,
       output_dim=2,
       fractional_order=0.5,
       use_adjoint=True  # Enable adjoint method
   )

See :doc:`neural_fsde_guide` for comprehensive documentation.

Summary
-------

Neural Fractional ODEs and SDEs provide:

✅ **Learning-based Solving**: Neural networks learn dynamics from data  
✅ **Fractional Memory**: Long-range dependencies through fractional derivatives  
✅ **Stochasticity**: Uncertainty modeling with noise terms  
✅ **Adjoint Training**: Efficient gradient computation  
✅ **Multiple Solvers**: Euler-Maruyama, Milstein, and more  

Next Steps
----------

- **API Reference**: See :doc:`api/neural_ode_sde_api` for complete API documentation
- **Neural fSDE Guide**: See :doc:`neural_fsde_guide` for comprehensive fSDE documentation
- **Neural fODE Guide**: See :doc:`neural_fode_guide` for neural ODE framework
- **Examples**: Check :doc:`sde_examples` for code examples and tutorials

