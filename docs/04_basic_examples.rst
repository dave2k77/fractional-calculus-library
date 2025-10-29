Basic Examples
==============

This section provides fundamental examples demonstrating basic fractional calculus operations with HPFRACC. All examples include complete, runnable code that you can execute directly.

Basic Fractional Derivatives
----------------------------

Computing Fractional Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare different fractional derivative definitions for a simple function:

.. code-block:: python

   from hpfracc.algorithms.optimized_methods import (
       OptimizedCaputo,
       OptimizedRiemannLiouville,
       OptimizedGrunwaldLetnikov
   )
   import numpy as np
   import matplotlib.pyplot as plt

   # Create time grid (avoid t=0 to prevent interpolation issues)
   t = np.linspace(0.01, 5, 100)
   h = t[1] - t[0]

   # Test function: f(t) = t^2
   f = t**2

   # Compute derivatives for different orders
   alpha_values = [0.25, 0.5, 0.75, 0.95]

   plt.figure(figsize=(15, 10))

   for i, alpha in enumerate(alpha_values):
       # Initialize derivative calculators for this alpha
       caputo = OptimizedCaputo(order=alpha)
       riemann = OptimizedRiemannLiouville(order=alpha)
       grunwald = OptimizedGrunwaldLetnikov(order=alpha)

       # Compute derivatives
       caputo_result = caputo.compute(f, t, h)
       riemann_result = riemann.compute(f, t, h)
       grunwald_result = grunwald.compute(f, t, h)

       # Plot results
       plt.subplot(2, 2, i + 1)
       plt.plot(t, f, "k-", label="Original: f(t) = t²", linewidth=2)
       plt.plot(t, caputo_result, "r--", label=f"Caputo (α={alpha})", linewidth=2)
       plt.plot(t, riemann_result, "b:", label=f"Riemann-Liouville (α={alpha})", linewidth=2)
       plt.plot(t, grunwald_result, "g-.", label=f"Grünwald-Letnikov (α={alpha})", linewidth=2)

       plt.xlabel("Time t")
       plt.ylabel("Function Value")
       plt.title(f"Fractional Derivatives of f(t) = t² (α = {alpha})")
       plt.legend()
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   print("✅ Basic fractional derivatives computed and plotted!")

Using Simplified API
~~~~~~~~~~~~~~~~~~~~

The library also provides simplified functions for quick computation:

.. code-block:: python

   import numpy as np
   from hpfracc import FractionalOrder, optimized_riemann_liouville, optimized_caputo

   # Define test function
   def test_function(x):
       return np.sin(x)

   # Create different fractional derivatives
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 2*np.pi, 100)

   # Riemann-Liouville
   result_rl = optimized_riemann_liouville(x, test_function(x), alpha)

   # Caputo
   result_caputo = optimized_caputo(x, test_function(x), alpha)

   # Plot results
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12, 8))
   plt.plot(x, test_function(x), label='Original: sin(x)', linewidth=2)
   plt.plot(x, result_rl, label='Riemann-Liouville (α=0.5)', linewidth=2)
   plt.plot(x, result_caputo, label='Caputo (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Fractional Derivatives of sin(x)')
   plt.legend()
   plt.grid(True)
   plt.show()

Fractional Integrals
--------------------

Computing Fractional Integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute fractional integrals for different functions:

.. code-block:: python

   from hpfracc import FractionalOrder, riemann_liouville_integral, caputo_integral
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.special import gamma

   # Create time grid
   t = np.linspace(0.01, 3, 100)
   h = t[1] - t[0]

   # Test function: f(t) = sin(t)
   f = np.sin(t)

   # Compute integrals for different orders
   alpha_values = [0.25, 0.5, 0.75, 0.95]

   plt.figure(figsize=(12, 8))

   plt.subplot(2, 2, 1)
   plt.plot(t, f, "k-", label="Original: f(t) = sin(t)", linewidth=2)
   plt.xlabel("Time t")
   plt.ylabel("Function Value")
   plt.title("Original Function")
   plt.legend()
   plt.grid(True, alpha=0.3)

   for i, alpha in enumerate(alpha_values[1:], 2):
       # Compute fractional integral using analytical solution
       # For f(t) = sin(t), the fractional integral is approximately t^alpha * sin(t)
       integral_result = (t**alpha / gamma(alpha + 1)) * np.sin(t)

       plt.subplot(2, 2, i)
       plt.plot(t, f, "k-", label="Original: f(t) = sin(t)", linewidth=1, alpha=0.5)
       plt.plot(t, integral_result, "r-", label=f"Fractional Integral (α={alpha})", linewidth=2)

       plt.xlabel("Time t")
       plt.ylabel("Function Value")
       plt.title(f"Fractional Integral of f(t) = sin(t) (α = {alpha})")
       plt.legend()
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   print("✅ Fractional integrals computed and plotted!")

Using API Functions
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc import FractionalOrder, riemann_liouville_integral, caputo_integral

   # Define test function
   def test_function(x):
       return x**2

   # Create different fractional integrals
   alpha = FractionalOrder(0.5)
   x = np.linspace(0, 5, 100)

   # Riemann-Liouville
   result_rl = riemann_liouville_integral(x, test_function(x), alpha)

   # Caputo
   result_caputo = caputo_integral(x, test_function(x), alpha)

   # Plot results
   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 2, 1)
   plt.plot(x, test_function(x), label='Original: x²', linewidth=2)
   plt.plot(x, result_rl, label='Riemann-Liouville (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Riemann-Liouville Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.subplot(1, 2, 2)
   plt.plot(x, test_function(x), label='Original: x²', linewidth=2)
   plt.plot(x, result_caputo, label='Caputo (α=0.5)', linewidth=2)
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Caputo Fractional Integral')
   plt.legend()
   plt.grid(True)
   
   plt.tight_layout()
   plt.show()

Comparison with Analytical Solutions
------------------------------------

Validating Numerical Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare numerical results with known analytical solutions:

.. code-block:: python

   from hpfracc.algorithms.optimized_methods import OptimizedCaputo
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.special import gamma

   # Create time grid
   t = np.linspace(0.01, 2, 50)
   h = t[1] - t[0]

   # Test function: f(t) = t (linear function)
   f = t

   # Analytical Caputo derivative of f(t) = t is: t^(1-α) / Γ(2-α)
   def analytical_caputo(t, alpha):
       """Analytical Caputo derivative of f(t) = t."""
       return t ** (1 - alpha) / gamma(2 - alpha)

   # Compare for different orders
   alpha_values = [0.25, 0.5, 0.75]

   plt.figure(figsize=(15, 5))

   for i, alpha in enumerate(alpha_values):
       # Initialize derivative calculator
       caputo = OptimizedCaputo(order=alpha)

       # Numerical result
       numerical_result = caputo.compute(f, t, h)

       # Analytical result
       analytical_result = analytical_caputo(t, alpha)

       # Plot comparison
       plt.subplot(1, 3, i + 1)
       plt.plot(t, numerical_result, "ro-", label="Numerical", markersize=4)
       plt.plot(t, analytical_result, "b-", label="Analytical", linewidth=2)

       # Calculate error
       error = np.abs(numerical_result - analytical_result)
       max_error = np.max(error)

       plt.xlabel("Time t")
       plt.ylabel("Derivative Value")
       plt.title(f"Caputo Derivative (α = {alpha})\nMax Error: {max_error:.2e}")
       plt.legend()
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   print("✅ Comparison with analytical solutions completed!")

Error Analysis and Convergence
------------------------------

Convergence Study
~~~~~~~~~~~~~~~~~

Analyze numerical error and convergence rates:

.. code-block:: python

   from hpfracc.algorithms.optimized_methods import OptimizedCaputo
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.special import gamma

   # Test function: f(t) = t^2
   def f_analytical(t):
       return t**2

   def caputo_analytical(t, alpha):
       """Analytical Caputo derivative of f(t) = t^2."""
       return 2 * t ** (2 - alpha) / gamma(3 - alpha)

   # Test different grid sizes
   grid_sizes = [20, 40, 80, 160, 320]
   alpha = 0.5

   errors = []

   for N in grid_sizes:
       t = np.linspace(0.01, 2, N)
       h = t[1] - t[0]
       f = f_analytical(t)

       # Numerical result
       caputo = OptimizedCaputo(order=alpha)
       numerical_result = caputo.compute(f, t, h)

       # Analytical result
       analytical_result = caputo_analytical(t, alpha)

       # Calculate error
       error = np.max(np.abs(numerical_result - analytical_result))
       errors.append(error)

   # Plot convergence
   plt.figure(figsize=(10, 6))
   plt.loglog(grid_sizes, errors, "bo-", markersize=8, linewidth=2, label="Numerical Error")

   # Reference line for first-order convergence
   ref_errors = [errors[0] * (grid_sizes[0] / N) for N in grid_sizes]
   plt.loglog(grid_sizes, ref_errors, "r--", label="First-order convergence", alpha=0.7)

   plt.xlabel("Grid Size N")
   plt.ylabel("Maximum Error")
   plt.title(f"Convergence Analysis: Caputo Derivative (α = {alpha})")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

   print("✅ Error analysis and convergence study completed!")

Special Functions
-----------------

Working with Special Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use special functions in fractional calculus:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hpfracc.special import (
       gamma_function, beta_function, binomial_coefficient,
       mittag_leffler_function
   )

   # Gamma function
   x = np.linspace(0.1, 5, 100)
   gamma_vals = [gamma_function(xi) for xi in x]

   # Beta function
   a, b = 2.0, 3.0
   beta_val = beta_function(a, b)
   print(f"B({a}, {b}) = {beta_val}")

   # Binomial coefficient
   n, k = 5, 2
   binomial_val = binomial_coefficient(n, k)
   print(f"({n} choose {k}) = {binomial_val}")

   # Mittag-Leffler function
   alpha, z = 0.5, 1.0
   ml_val = mittag_leffler_function(alpha, z)
   print(f"E_{alpha}({z}) = {ml_val}")

   # Plot gamma function
   plt.figure(figsize=(10, 6))
   plt.plot(x, gamma_vals, linewidth=2)
   plt.xlabel('x')
   plt.ylabel('Γ(x)')
   plt.title('Gamma Function')
   plt.grid(True)
   plt.show()

Mathematical Utilities
----------------------

Using Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Leverage validation and mathematical utility functions:

.. code-block:: python

   from hpfracc.core.utilities import (
       validate_fractional_order, validate_function,
       factorial_fractional, binomial_coefficient
   )
   import numpy as np

   # Validate fractional order
   is_valid = validate_fractional_order(0.5)  # True
   print(f"Fractional order 0.5 is valid: {is_valid}")
   
   is_valid = validate_fractional_order(-1.0)  # False
   print(f"Fractional order -1.0 is valid: {is_valid}")

   # Validate function
   def test_func(x):
       return x**2
   
   is_valid = validate_function(test_func)  # True
   print(f"Function is valid: {is_valid}")

   # Fractional factorial
   x = 2.5
   factorial_val = factorial_fractional(x)
   print(f"Factorial of {x}: {factorial_val}")

   # Binomial coefficient
   n, k = 5, 2
   binomial_val = binomial_coefficient(n, k)
   print(f"({n} choose {k}) = {binomial_val}")

Summary
-------

These basic examples demonstrate:

✅ **Fractional Derivatives**: Caputo, Riemann-Liouville, Grünwald-Letnikov  
✅ **Fractional Integrals**: RL and Caputo integrals  
✅ **Validation**: Comparison with analytical solutions  
✅ **Error Analysis**: Convergence studies and numerical accuracy  
✅ **Special Functions**: Gamma, Beta, Mittag-Leffler, Binomial coefficients  
✅ **Utilities**: Validation and mathematical helper functions  

Next Steps
----------

- **Learn more**: See :doc:`06_derivatives_integrals` for comprehensive operator guide
- **Advanced examples**: Check :doc:`05_advanced_examples` for signal/image processing
- **ML integration**: Explore :doc:`07_fractional_neural_networks` for neural networks

