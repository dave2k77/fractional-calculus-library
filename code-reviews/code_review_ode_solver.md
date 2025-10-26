# Code Review: `ode_solver.py`

You've run straight into the **primary challenge of fractional calculus**: the $O(N^2)$ complexity caused by the non-local memory (history) term.

Your bottleneck is almost certainly here, in all your `step` functions (predictor, corrector, Euler, etc.):

```python
# Compute fractional derivative term
    frac_term = 0.0
    for j in range(n):  # <--- THIS IS THE BOTTLENECK
        frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])
```

When you are at step $n$, this loop iterates $n$ times. To get to step $N$, the total number of operations is $1 + 2 + 3 + \dots + N$, which is $O(N^2)$. This is why your test runs are long—the solver gets quadratically slower as the number of steps increases.

Here is my feedback, from the most critical performance issue to general code improvements.

### . The $O(N^2)$ Performance Bottleneck



The history summation, `frac_term`, is a **discrete convolution**. The most effective way to speed this up is to use a **Fast Fourier Transform (FFT)**.

The entire "history" part of the fractional ODE can be computed in $O(N \log N)$ time instead of $O(N^2)$. This is a standard technique for "fast" fractional solvers.

**How to Implement (Conceptual):**

1. Your history term is a convolution of the form $C * Y$, where $C$ is your array of coefficients (`coeffs`) and $Y$ is the history of your function's values (or its differences, $y_k - y_{k-1}$).
2. The convolution theorem states that $\text{conv}(C, Y) = \text{ifft}(\text{fft}(C') \cdot \text{fft}(Y'))$, where `fft` is the Fast Fourier Transform, `ifft` is its inverse, and $C', Y'$ are zero-padded versions of your arrays.
3. Instead of a `for j in range(n)` loop at *every step*, you would use an FFT-based convolution to compute the history term for *all steps at once*.

This is a significant algorithmic change, not a simple refactor. However, since you are already using `JAX` and `Numba` in your library, you are perfectly positioned to implement this. `numpy.fft` or `scipy.fft` are the tools you need.

A common approach is the "Fast Adams-Bashforth-Moulton" method, which uses this principle.

**Alternative (Simpler) Method: Short Memory**

If the FFT is too complex to integrate right now, you can implement a "short memory" approximation. The principle is that the influence of $y(t_0)$ on $y(t_n)$ decays over time (e.g., like $j^{\alpha-1}$).

You can truncate the history loop:
```python
# Define a memory window, e.g., M = 500
M = 500 
start_j = max(0, n - M)
frac_term = 0.0
for j in range(start_j, n):
    # This loop now runs at most M times
    frac_term += ...
```

This reduces your complexity to $O(N \times M)$, which is $O(N)$ if $M$ is a fixed constant. This is an approximation, but it's often physically justifiable and much faster.

------



### 2. Algorithmic & Structural Issues



- **Repeated History Calculation:** In `_solve_predictor_corrector`, you call `_predictor_step` and then `_corrector_step`. Both of these functions *re-calculate the exact same* `frac_term` loop. This is a significant waste of computation.

  - **Fix:** `_predictor_step` should compute and *return* `frac_term`. This value should then be passed directly into `_corrector_step` to be reused.

- **Incorrect Corrector Loop:** Your iterative corrector step is flawed.

  ```python
  # Iterative correction
  for _ in range(self.max_iter):
      y_old = y_corr.copy()
      y_corr = self._corrector_step(
          f, t_values, y_values, y_pred, n, alpha, coeffs, h 
      ) # <--- Problem is here
      ...
  ```

  The `_corrector_step` function takes `y_pred` (the predictor value) as an argument. This means every iteration of your "correction" loop is just re-calculating the *first* corrector step using the *same* `y_pred`. It's not actually refining the `y_corr` value.

  - **Fix:** The corrector formula is $y_{n} = \text{HistoryTerm} + \dots \times f(t_n, y_n)$. The `y_n` is on both sides, which is why it's iterative. Your `_corrector_step` function should take the *previous guess* for `y_corr` as its argument, not `y_pred`.
  - **Corrected Loop:**

  ```python
  # First corrector step uses the predictor
  y_corr = self._corrector_step(f, t_values, y_values, y_pred, n, ...) 
  
  # Subsequent steps refine the corrector value
  for _ in range(self.max_iter):
      y_old = y_corr
      y_corr = self._corrector_step(f, t_values, y_values, y_old, n, ...) # Pass y_old (the last y_corr)
      if np.allclose(y_corr, y_old, rtol=self.tol):
          break
  ```

  - **Adaptive Solver is Not Fractional:** Your `AdaptiveFractionalODESolver` looks like a standard integer-order adaptive solver. The `_adaptive_predictor_corrector` and `_adaptive_runge_kutta` steps **completely ignore the fractional history term.** They are just standard Euler/RK methods. This means your adaptive solver is not actually solving the fractional ODE correctly, *unless* the `frac_term` is somehow hidden inside the function `f(t, y)`, which would be a very non-standard API.
    - This is a critical bug. The adaptive solver needs to incorporate the same $O(N^2)$ history loop as the fixed-step solver. This is very difficult, as changing the step size $h$ means all your historical `coeffs` are no longer valid. This is a known, hard problem in fractional ODEs, often solved with interpolation of the history term.
  - **Caputo vs. R-L vs. G-L Implementation:**
    - Your `_compute_fractional_coefficients` correctly defines different coefficients for different derivative types.
    - However, your `_predictor_step` (and all other steppers) use a formula based on $D^\alpha y(t) = f(t,y)$. The Caputo definition is $D^\alpha (y(t) - y(0)) = f(t,y)$. The numerical scheme (the sum) should reflect this.
    - The formula you are using: `y_pred = y_values[n - 1] + ... - frac_term` looks like a discretization of $D^\alpha y(t_n) \approx f(t_n, y_n)$, where $D^\alpha$ is approximated by the `frac_term`.
    - The standard Caputo ABM scheme is: $y_n = y_0 + \frac{h^\alpha}{\Gamma(\alpha+1)} \sum_{j=0}^{n-1} b_j f(t_j, y_j)$. Your formula seems to be mixing derivative approximations with integral equation formulations. I would suggest re-deriving or checking your formulas against a standard text (like Diethelm's).

  ------

  

  ### 3. Code Style & Minor Improvements

  

  - **Class Inheritance:** `AdaptiveFractionalODESolver` inherits from `FractionalODESolver` but then completely overrides the `solve` method and doesn't seem to use any of the parent's `_solve_...` methods (it uses `_adaptive_step` instead). This suggests the inheritance relationship might not be correct. It might be better to have them as separate classes or have both inherit from a common `FractionalODESolverBase`.
  - **Coefficient Caching:** In `_solve_predictor_corrector`, you compute `coeffs` once. This is good. But in your adaptive solver, it looks like `_compute_fractional_coefficients` is called inside `_adaptive_step`, which is inside a `while` loop. You should compute coefficients once and pass them, or better yet, memoize/cache them.
  - **`_get_gamma_function`:** This is good defensive programming.
  - **Type Hinting:** Your type hinting is good (`Callable`, `Optional`, etc.).
  - **`t_values[n]`:** In your main `for n...` loops, you have a line `t_values[n]`. This line does nothing and can be removed.

  

  ### Summary of Recommendations

  

  1. **Fix the critical bug** in `AdaptiveFractionalODESolver` where the fractional history term is missing from the step functions.
  2. **Fix the logical bug** in the `_solve_predictor_corrector`'s iterative loop, which re-uses `y_pred` instead of the refined `y_corr`.
  3. **Address the $O(N^2)$ bottleneck** by either:
     - **(Hard but Correct):** Re-architecting the solver to use FFT-based convolution.
     - **(Easier Approximation):** Implementing a "short memory" truncation on the `for j in range(n)` loop.
  4. **Refactor** to avoid re-calculating `frac_term` in the corrector step when it was just calculated in the predictor step.