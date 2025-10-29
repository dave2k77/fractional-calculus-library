Theoretical Foundations
======================

This chapter provides comprehensive mathematical theory and foundations for fractional calculus, model theory, and the mathematical underpinnings of HPFRACC.

Mathematical Theory
-------------------

Fractional Calculus Fundamentals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractional calculus extends integer-order derivatives and integrals to arbitrary orders.

**Basic Definitions**:

Riemann-Liouville fractional derivative:

.. math::

   D^\alpha_{RL} f(t) = \frac{1}{\Gamma(n-\alpha)} \frac{d^n}{dt^n} \int_0^t (t-\tau)^{n-\alpha-1} f(\tau) d\tau

Caputo fractional derivative:

.. math::

   D^\alpha_C f(t) = \frac{1}{\Gamma(n-\alpha)} \int_0^t (t-\tau)^{n-\alpha-1} f^{(n)}(\tau) d\tau

Grünwald-Letnikov derivative:

.. math::

   D^\alpha_{GL} f(t) = \lim_{h \to 0} h^{-\alpha} \sum_{k=0}^\infty (-1)^k \binom{\alpha}{k} f(t-kh)

Key Properties
~~~~~~~~~~~~~~

1. **Linearity**: :math:`D^\alpha(af + bg) = aD^\alpha f + bD^\alpha g`
2. **Leibniz Rule**: :math:`D^\alpha(fg) = \sum_{k=0}^\infty \binom{\alpha}{k} D^{\alpha-k}f D^k g`
3. **Chain Rule**: :math:`D^\alpha(f \circ g) = \sum_{k=1}^\infty \binom{\alpha}{k} (D^k f \circ g) (D^\alpha g)^k`
4. **Semigroup Property**: :math:`D^\alpha(D^\beta f) = D^{\alpha+\beta} f`

For comprehensive mathematical theory, see :doc:`mathematical_theory`.

Model Theory
------------

Neural Network Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fractional neural networks extend standard neural networks by incorporating fractional calculus operations:

**Forward Pass**:

.. math::

   y = \sigma(W D^\alpha x + b)

where :math:`D^\alpha` is a fractional derivative operator.

**Backward Pass**:

Gradients flow through fractional derivatives using spectral methods:

.. math::

   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial D^\alpha x}{\partial x}

Autograd Implementation
~~~~~~~~~~~~~~~~~~~~~~~

The spectral autograd framework enables gradient computation through fractional operators using:

- **FFT-based methods**: O(N log N) complexity
- **Mellin transforms**: Scale-invariant operations
- **Fractional Laplacian**: Spatial operators

See :doc:`model_theory` for detailed model theory.

Fractional Operator Theory
--------------------------

Special Functions
~~~~~~~~~~~~~~~~~

**Mittag-Leffler Function**:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^\infty \frac{z^k}{\Gamma(\alpha k + \beta)}

**Gamma Function**:

.. math::

   \Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt

**Beta Function**:

.. math::

   B(a,b) = \int_0^1 t^{a-1}(1-t)^{b-1} dt = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}

Convergence and Stability
-------------------------

Numerical Stability
~~~~~~~~~~~~~~~~~~~

- **Riemann-Liouville**: Stable for 0 < α < 1, may have boundary effects
- **Caputo**: Better initial value behavior, stable for all α > 0
- **Grünwald-Letnikov**: Numerical stability depends on step size
- **Novel Methods**: Enhanced stability with non-singular kernels

Convergence Analysis
~~~~~~~~~~~~~~~~~~~~

Error estimates and convergence rates:

.. math::

   \|D^\alpha_{h} f - D^\alpha f\| \leq C h^p

where :math:`h` is the step size and :math:`p` is the convergence order.

Neural fSDE Theory
------------------

Stochastic Differential Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural fractional SDE:

.. math::

   D_t^\alpha X(t) = f_\theta(t, X(t)) dt + g_\theta(t, X(t)) dW(t)

**Drift Function**: :math:`f_\theta` determines deterministic dynamics  
**Diffusion Function**: :math:`g_\theta` controls stochastic noise  
**Fractional Order**: :math:`\alpha` controls memory effects

Solution Methods
~~~~~~~~~~~~~~~

- **Euler-Maruyama**: Basic stochastic integration
- **Milstein**: Higher-order accuracy for SDEs
- **Adjoint Methods**: Efficient gradient computation

See :doc:`neural_fsde_guide` for detailed SDE theory.

Complete Mathematical Reference
--------------------------------

For complete mathematical foundations including:

- Deep mathematical theory for ML models
- Neural fODE mathematical foundations
- GNN theoretical background
- Autograd kernel mathematics
- Fractional chain rule derivations

See :doc:`mathematical_theory` for comprehensive mathematical documentation.

Summary
-------

Theoretical Foundations provide:

✅ **Fractional Calculus**: Complete mathematical definitions and properties  
✅ **Model Theory**: Neural network integration and autograd theory  
✅ **Special Functions**: Mittag-Leffler, Gamma, Beta functions  
✅ **Convergence**: Stability analysis and error estimates  
✅ **Neural fSDE**: Stochastic differential equation theory  

Next Steps
----------

- **Mathematical Theory**: See :doc:`mathematical_theory` for deep theory
- **Model Theory**: See :doc:`model_theory` for model foundations
- **Operators Guide**: See :doc:`06_derivatives_integrals` for operator details

