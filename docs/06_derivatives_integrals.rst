Integrals and Derivatives
=======================

The HPFRACC library provides a comprehensive collection of fractional calculus operators, from classical definitions to cutting-edge advanced methods. This guide covers all available operators, their mathematical foundations, and practical usage.

For complete API documentation, see :doc:`api/derivatives_integrals_api`.

Classical Fractional Derivatives
--------------------------------

Riemann-Liouville Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**: 

.. math::

   D^\alpha_{RL} f(t) = \frac{1}{\Gamma(n-\alpha)} \frac{d^n}{dt^n} \int_0^t (t-\tau)^{n-\alpha-1} f(\tau) d\tau

**Usage**:

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative

   # Create Riemann-Liouville derivative with α = 0.5
   rl_derivative = create_fractional_derivative('riemann_liouville', 0.5)

   # Compute derivative of f(x) = x^2 at x = 2.0
   def f(x): return x**2
   result = rl_derivative.compute(f, 2.0)

**Characteristics**:
- Most fundamental fractional derivative definition
- Well-suited for initial value problems
- Computationally efficient with optimized algorithms

Caputo Derivative
~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_C f(t) = \frac{1}{\Gamma(n-\alpha)} \int_0^t (t-\tau)^{n-\alpha-1} f^{(n)}(\tau) d\tau

**Usage**:

.. code-block:: python

   caputo_derivative = create_fractional_derivative('caputo', 0.5)
   result = caputo_derivative.compute(f, 2.0)

**Characteristics**:
- Better behavior for initial value problems
- Preserves classical derivative properties
- Widely used in physics and engineering

Grünwald-Letnikov Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_{GL} f(t) = \lim_{h \to 0} h^{-\alpha} \sum_{k=0}^\infty (-1)^k \binom{\alpha}{k} f(t-kh)

**Usage**:

.. code-block:: python

   gl_derivative = create_fractional_derivative('grunwald_letnikov', 0.5)
   result = gl_derivative.compute(f, 2.0)

**Characteristics**:
- Discrete approximation approach
- Good for numerical computations
- Memory-efficient implementation

Novel Fractional Derivatives
----------------------------

Caputo-Fabrizio Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   ^{CF}D^\alpha f(t) = \frac{M(\alpha)}{1-\alpha} \int_0^t f'(\tau) \exp\left(-\frac{\alpha(t-\tau)}{1-\alpha}\right) d\tau

**Usage**:

.. code-block:: python

   cf_derivative = create_fractional_derivative('caputo_fabrizio', 0.5)
   result = cf_derivative.compute(f, 2.0)

**Characteristics**:
- Non-singular exponential kernel
- Better numerical stability
- Ideal for biological systems and viscoelasticity

Atangana-Baleanu Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   ^{AB}D^\alpha f(t) = \frac{B(\alpha)}{1-\alpha} \int_0^t f'(\tau) E_\alpha\left(-\frac{\alpha(t-\tau)^\alpha}{1-\alpha}\right) d\tau

**Usage**:

.. code-block:: python

   ab_derivative = create_fractional_derivative('atangana_baleanu', 0.5)
   result = ab_derivative.compute(f, 2.0)

**Characteristics**:
- Mittag-Leffler kernel
- Enhanced memory effects
- Advanced applications in complex systems

Advanced Methods
---------------

Weyl Derivative
~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_W f(x) = \frac{1}{\Gamma(n-\alpha)} \left(\frac{d}{dx}\right)^n \int_x^\infty (\tau-x)^{n-\alpha-1} f(\tau) d\tau

**Usage**:

.. code-block:: python

   weyl_derivative = create_fractional_derivative('weyl', 0.5)
   result = weyl_derivative.compute(f, 2.0)

**Characteristics**:
- FFT convolution implementation
- Parallel processing optimization
- Suitable for functions on entire real line

Marchaud Derivative
~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_M f(x) = \frac{\alpha}{\Gamma(1-\alpha)} \int_0^\infty \frac{f(x) - f(x-\tau)}{\tau^{\alpha+1}} d\tau

**Usage**:

.. code-block:: python

   marchaud_derivative = create_fractional_derivative('marchaud', 0.5)
   result = marchaud_derivative.compute(f, 2.0)

**Characteristics**:
- Difference quotient convolution
- Memory optimization
- General kernel support

Hadamard Derivative
~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_H f(x) = \frac{1}{\Gamma(n-\alpha)} \left(x\frac{d}{dx}\right)^n \int_1^x \left(\ln\frac{x}{t}\right)^{n-\alpha-1} \frac{f(t)}{t} dt

**Usage**:

.. code-block:: python

   hadamard_derivative = create_fractional_derivative('hadamard', 0.5)
   result = hadamard_derivative.compute(f, 2.0)

**Characteristics**:
- Logarithmic kernels
- Geometric interpretation
- Applications in geometric analysis

Riesz-Feller Derivative
~~~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   D^\alpha_{RF} f(x) = \frac{1}{2\pi} \int_{\mathbb{R}} |\xi|^\alpha \mathcal{F}[f](\xi) e^{i\xi x} d\xi

**Usage**:

.. code-block:: python

   rf_derivative = create_fractional_derivative('reiz_feller', 0.5)
   result = rf_derivative.compute(f, 2.0)

**Characteristics**:
- Spectral method implementation
- Fourier domain computation
- High accuracy for smooth functions

Special Operators
-----------------

Fractional Laplacian
~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   (-\Delta)^{\alpha/2} f(x) = \left(\frac{1}{2\pi}\right)^n \int_{\mathbb{R}^n} |\xi|^\alpha \mathcal{F}[f](\xi) e^{i\xi \cdot x} d\xi

**Usage**:

.. code-block:: python

   laplacian = create_fractional_derivative('fractional_laplacian', 0.5)
   result = laplacian.compute(f, x_array)

**Characteristics**:
- Spatial fractional derivatives
- Multi-dimensional support
- Applications in PDEs and image processing

Fractional Fourier Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   \mathcal{F}^\alpha[f](u) = \int_{\mathbb{R}} f(x) K_\alpha(x,u) dx

**Usage**:

.. code-block:: python

   fft = create_fractional_derivative('fractional_fourier_transform', 0.5)
   result = fft.compute(f, x_array)

**Characteristics**:
- Generalized Fourier transform
- Signal processing applications
- Time-frequency analysis

Riesz-Fisher Operator
~~~~~~~~~~~~~~~~~~~~~

**Definition**:

.. math::

   R^\alpha f(x) = \frac{1}{2} \left[D^\alpha_+ f(x) + D^\alpha_- f(x)\right]

**Usage**:

.. code-block:: python

   from hpfracc.core.fractional_implementations import create_riesz_fisher_operator

   # For derivative behavior (α > 0)
   rf_derivative = create_riesz_fisher_operator(0.5)
   result = rf_derivative.compute(f, x)

   # For integral behavior (α < 0)
   rf_integral = create_riesz_fisher_operator(-0.5)
   result = rf_integral.compute(f, x)

   # For identity behavior (α = 0)
   rf_identity = create_riesz_fisher_operator(0.0)
   result = rf_identity.compute(f, x)

**Characteristics**:
- Unified derivative/integral operator
- Smooth transition between operations
- Perfect for signal processing and image analysis

Fractional Integrals
-------------------

Available Integral Types
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Riemann-Liouville Integral** (`"RL"`)
2. **Caputo Integral** (`"Caputo"`)
3. **Weyl Integral** (`"Weyl"`)
4. **Hadamard Integral** (`"Hadamard"`)
5. **Miller-Ross Integral** (`"MillerRoss"`)
6. **Marchaud Integral** (`"Marchaud"`)

**Usage**:

.. code-block:: python

   from hpfracc.core.integrals import create_fractional_integral

   # Create Riemann-Liouville integral
   rl_integral = create_fractional_integral("RL", 0.5)
   result = rl_integral(f, x)

   # Create Weyl integral
   weyl_integral = create_fractional_integral("Weyl", 0.5)
   result = weyl_integral(f, x)

Usage Examples
--------------

Basic Usage Pattern
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hpfracc.core.derivatives import create_fractional_derivative
   import numpy as np

   # Define function
   def f(x): return x**2

   # Create derivative
   derivative = create_fractional_derivative('riemann_liouville', 0.5)

   # Single point computation
   result = derivative.compute(f, 2.0)

   # Array computation
   x_array = np.linspace(0, 5, 100)
   result_array = derivative.compute(f, x_array)

   # Numerical computation from function values
   f_values = f(x_array)
   result_numerical = derivative.compute_numerical(f_values, x_array)

Advanced Usage with Parallel Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use parallel-optimized methods for large computations
   parallel_derivative = create_fractional_derivative('parallel_riemann_liouville', 0.5)

   # Large array computation
   x_large = np.linspace(0, 100, 10000)
   result_large = parallel_derivative.compute(f, x_large)

Autograd Fractional Derivatives (ML)
------------------------------------

The ML module provides autograd-friendly fractional derivatives that preserve the computation graph.

.. code-block:: python

   import torch
   from hpfracc.ml.fractional_autograd import fractional_derivative, FractionalDerivativeLayer

   x = torch.randn(2, 64, 128, requires_grad=True)  # (batch, channels, time)

   # RL/GL
   y_rl = fractional_derivative(x, alpha=0.5, method="RL")

   # Caputo
   y_caputo = fractional_derivative(x, alpha=0.5, method="Caputo")

   # Caputo-Fabrizio (exponential kernel)
   y_cf = fractional_derivative(x, alpha=0.5, method="CF")

   # Atangana-Baleanu (blended kernel)
   y_ab = fractional_derivative(x, alpha=0.5, method="AB")

   # Layer wrapper
   layer = FractionalDerivativeLayer(alpha=0.5, method="RL")
   out = layer(torch.randn(4, 16, 256, requires_grad=True))

Performance Considerations
--------------------------

Method Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **For small computations (< 1000 points)**: Use classical methods
2. **For medium computations (1000-10000 points)**: Use advanced methods
3. **For large computations (> 10000 points)**: Use parallel-optimized methods
4. **For real-time applications**: Use optimized methods with JAX/Numba
5. **For memory-constrained systems**: Use memory-optimized methods

Optimization Tips
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable JAX acceleration when available
   derivative = create_fractional_derivative('riemann_liouville', 0.5, use_jax=True)

   # Enable Numba optimization
   derivative = create_fractional_derivative('riemann_liouville', 0.5, use_numba=True)

   # Use parallel processing for large arrays
   parallel_derivative = create_fractional_derivative('parallel_riemann_liouville', 0.5)

Mathematical Properties
-----------------------

Key Properties
~~~~~~~~~~~~~~

1. **Linearity**: :math:`D^\alpha(af + bg) = aD^\alpha f + bD^\alpha g`
2. **Leibniz Rule**: :math:`D^\alpha(fg) = \sum_{k=0}^\infty \binom{\alpha}{k} D^{\alpha-k}f D^k g`
3. **Chain Rule**: :math:`D^\alpha(f \circ g) = \sum_{k=1}^\infty \binom{\alpha}{k} (D^k f \circ g) (D^\alpha g)^k`
4. **Semigroup Property**: :math:`D^\alpha(D^\beta f) = D^{\alpha+\beta} f`

Convergence and Stability
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Riemann-Liouville**: Stable for 0 < α < 1, may have boundary effects
- **Caputo**: Better initial value behavior, stable for all α > 0
- **Grünwald-Letnikov**: Numerical stability depends on step size
- **Novel Methods**: Enhanced stability with non-singular kernels

Summary
-------

The HPFRACC library provides a comprehensive suite of fractional calculus operators suitable for a wide range of applications. From classical definitions to cutting-edge advanced methods, users can choose the most appropriate operator for their specific needs.

**Available Operators**:
- ✅ **Classical**: Riemann-Liouville, Caputo, Grünwald-Letnikov
- ✅ **Novel**: Caputo-Fabrizio, Atangana-Baleanu
- ✅ **Advanced**: Weyl, Marchaud, Hadamard, Riesz-Feller
- ✅ **Special**: Fractional Laplacian, Fractional Fourier Transform
- ✅ **Integrals**: RL, Caputo, Weyl, Hadamard, Miller-Ross, Marchaud

For complete API reference, see :doc:`api/derivatives_integrals_api`.

Next Steps
----------

- **Theory**: See :doc:`12_theoretical_foundations` for mathematical foundations
- **Neural Networks**: Explore :doc:`07_fractional_neural_networks` for ML integration
- **Examples**: Check :doc:`04_basic_examples` for practical examples

