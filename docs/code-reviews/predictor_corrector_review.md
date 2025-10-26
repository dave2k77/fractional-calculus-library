# Code Review: `predictor_corrector.py`



This module review follows up on our analysis of `ode_solver.py`. This file has a much better Object-Oriented structure, with `AdamsBashforthMoultonSolver` and `VariableStepPredictorCorrector` correctly inheriting from `PredictorCorrectorSolver`.

However, this module suffers from the **exact same critical flaws** as the previous `ode_solver.py` file:

1. The **adaptive solver is non-functional** for fractional calculus.
2. The **corrector iteration logic is flawed** and does not converge.
3. The **fixed-step solver has an $O(N^2)$ bottleneck**.

Here is a detailed breakdown.



## 1. Critical Issue: Adaptive Solver is Non-Functional 🛑



This is the most significant problem. The adaptive solver (`_solve_adaptive`) does not solve the fractional ODE. It solves a simple, *integer-order* ODE.

Your adaptive step functions, `_adaptive_predictor_step` and `_adaptive_corrector_step`, **completely ignore the fractional history term (`frac_term`)**.

Here is your `_adaptive_predictor_step`:

```python
def _adaptive_predictor_step(...):
    # ...
    # Simple predictor (THIS IS JUST FORWARD EULER)
    y_pred = y_current + (h**alpha_val / gamma(alpha_val + 1)) * f(
        t_current, y_current
    )
    return y_pred
```

This is the formula for the integer-order Forward Euler method, $y_{n+1} = y_n + h \cdot f(t_n, y_n)$, just with a fractional $h$ scaling. It does not include the history sum $\sum b_j f_j$, which is the entire point of the fractional derivative. The `_adaptive_corrector_step` has the same flaw (it's just the integer-order Trapezoidal rule).

**Why this is a hard problem:** An adaptive step size $h$ is notoriously difficult for fractional ODEs. All history coefficients/weights ($a_j, b_j$) depend on a fixed $h$. If $h$ changes, the entire history becomes invalid. Solving this requires highly complex methods, like history interpolation, which are not implemented here.

**Recommendation:** This adaptive solver is fundamentally incorrect and should be **removed or completely redesigned** using a proper variable-stepsize FDE algorithm.

## 2. Critical Issue: Incorrect Corrector Iteration Logic 🐛



In your `_solve_fixed_step` function, the iterative correction loop is logically flawed and will not converge as intended.

```python
# Iterative correction
for _ in range(self.max_iter):
    y_old = y_corr.copy()
    y_corr = self._corrector_step(
        f, t_values, y_values, y_pred, n, alpha, coeffs, h0
    ) # <--- BUG HERE
    # ...
```

The corrector step is implicit, meaning you are trying to solve for $y_n$ in the equation $y_n = \text{History} + \dots \times f(t_n, \mathbf{y_n})$. The iteration should refine the *guess* for $y_n$.

Your loop always passes `y_pred` (the initial predictor guess) to `_corrector_step`. This means you are not refining the solution; you are just re-calculating the very first corrector value `max_iter` times.

**The Fix:** You must pass the *previous corrector guess* (`y_old`) back into the function.

```python
# First corrector step uses the predictor's value
y_corr = self._corrector_step(
    f, t_values, y_values, y_pred, n, alpha, coeffs, h0
)

# Subsequent steps refine the corrector value
for _ in range(self.max_iter):
    y_old = y_corr.copy()
    # Pass y_old (the last y_corr) as the new guess for y_n
    y_corr = self._corrector_step(
        f, t_values, y_values, y_old, n, alpha, coeffs, h0
    )

    if np.allclose(y_corr, y_old, rtol=self.tol):
        break
```

This same logic bug exists in `_adaptive_corrector_step` as well.

## 3. Performance Bottleneck: $O(N^2)$ Complexity 🐢



Your fixed-step solvers (`PredictorCorrectorSolver` and `AdamsBashforthMoultonSolver`) both contain $O(N^2)$ bottlenecks, which is why your tests are slow.

- In `PredictorCorrectorSolver._predictor_step` (and `_corrector_step`):



```python
frac_term = 0.0
for j in range(n): # <--- O(N) loop
    frac_term += coeffs[j] * (y_values[n - j] - y_values[n - j - 1])
```

- In AdamsBashforthMoultonSolver._predictor_step:

```python
for j in range(n): # <--- O(N) loop
    y_pred += (
        h**alpha_val / gamma(alpha_val + 1) *
        weights[j] * f(t_j, y_values[j])
    )
```

Since this $O(N)$ loop is inside the main `for n in range(1, N)` loop, the total complexity is $O(N^2)$.

**Recommendations:**

1. **Short Memory:** Truncate the `j` loop to only sum the last `M` (e.g., `M=500`) steps. This approximates the solution in $O(N \times M)$ or $O(N)$ time.
2. **FFT Convolution:** The "correct" way to speed this up is to recognize the sum as a convolution and use an FFT to compute it in $O(N \log N)$ time. This is the basis of all "Fast" FDE solvers.

------



## 4. Structural & Formulaic Concerns





### Confusing Class Implementations



This is a major source of confusion. The base `PredictorCorrectorSolver` and the subclass `AdamsBashforthMoultonSolver` implement **two completely different numerical methods.**

- `PredictorCorrectorSolver` implements some form of $y_n = y_{n-1} + h^\alpha (\dots) - \text{HistoryTerm}$.
- `AdamsBashforthMoultonSolver` implements $y_n = y_{n-1} + h^\alpha (\dots) \sum w_j f(t_j, y_j)$.

The base class is *not* a generic ABM solver; it's a different, specific (and likely incorrect) method. This makes the inheritance misleading.



### Incorrect ABM Formulation



The standard Fractional Adams-Bashforth-Moulton method is derived from the Volterra integral equation:

$y(t) = \mathbf{y_0} + \frac{1}{\Gamma(\alpha)} \int_{0}^{t} (t - \tau)^{\alpha-1} f(\tau, y(\tau)) d\tau$

The resulting numerical scheme is always based on the initial condition $\mathbf{y_0}$:

$y_n = \mathbf{y_0} + \frac{h^\alpha}{\Gamma(\alpha+1)} \sum_{j=0}^{n-1} b_{j,n} f_j$  (Predictor)

$y_n = \mathbf{y_0} + \frac{h^\alpha}{\Gamma(\alpha+2)} \sum_{j=0}^{n} a_{j,n} f_j$  (Corrector)

Your code implements formulas based on $\mathbf{y_{n-1}}$:

y_pred = y_values[n - 1] + ...

y_corr = y_values[n - 1] + ...

This suggests a formulation like $y_n = y_{n-1} + \int_{t_{n-1}}^{t_n} \dots$, which is incorrect for a non-local operator. The entire history from $t_0$ must be included at every step. I strongly recommend you **check your formulas against a trusted academic source** like Diethelm's *The Analysis of Fractional Differential Equations*.