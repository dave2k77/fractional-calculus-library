# What’s already good

-   Clear statement of the problem and the spectral strategy (Fourier and Mellin). You frame the non-locality vs. AD issue well. 
-   Correct use of “apply the adjoint in the backward pass” as the chain-rule core. Your adjoint forms are (mostly) right in spirit. 
-   Practical sections on stability/convergence, testing with power/exponential functions, and a skeleton `torch.autograd.Function`.    

# Critical issues & concrete fixes

## 1) Forward–backward inconsistency (K vs K*)

You state Theorem 1 with the conjugated kernel in the gradient,

$\frac{\partial L}{\partial f}=\mathcal F^{-1}\!\left[K_\alpha^{*}(\xi)\,\mathcal F\!\left[\frac{\partial L}{\partial (D^\alpha f)}\right]\right],$

but the proof that follows drops the conjugate and ends with $K_\alpha$ instead of $K_\alpha^{*}$. That’s internally inconsistent.  

**Fix.** Choose a single Fourier convention (I recommend the *unitary* one so Parseval is exact), declare the $L^2$ inner product you differentiate under, and then derive the adjoint once: with $(\widehat{D^\alpha f})(\xi)=K_\alpha(\xi)\hat f(\xi)$, the $L^2$ gradient is

$\frac{\partial L}{\partial f}= \mathcal F^{-1}\!\left[\overline{K_\alpha(\xi)}\,\mathcal F\!\left(\frac{\partial L}{\partial (D^\alpha f)}\right)\right].$

Then ensure your `backward` uses `conj(kernel)` for FFT and the appropriate Mellin adjoint (your text does say this; the derivation should match it). 

## 2) Branch cuts and “realness”: $(i\xi)^\alpha$ is not innocent

Using $(i\xi)^\alpha$ needs a branch choice, otherwise you’ll get arbitrary phases and spurious imaginary parts for real inputs. Returning `result.real` in code silently discards information.  

**Fix.**

-   State the branch: $(i\xi)^\alpha=|\xi|^\alpha e^{i\,\mathrm{sign}(\xi)\,\pi\alpha/2}$ (principal branch).
-   If you need a *real* operator on real signals, use the **Riesz / fractional Laplacian** symbol $|\xi|^\alpha$ (or $-|\xi|^\alpha$ depending on convention). Then Hermitian symmetry is automatic and you avoid `.real` hacks.
-   If you specifically want the (left/right) Weyl derivative with $(i\xi)^\alpha$, enforce Hermitian symmetry of the kernel explicitly: build the spectrum for positive $\xi$ and mirror with conjugation.

## 3) Discretisation scaling errors (Δx and 2π)

Your discrete formulae and code omit the sampling interval and angular-frequency factor. `torch.fft.fftfreq(N)` returns *cycles per sample*. Your continuous formula uses $\xi$ as *angular* frequency; you must multiply by $2\pi$ and divide by the step $\Delta x$: $\omega_k=\tfrac{2\pi}{N\Delta x}\,k$, then use $(i\omega_k)^\alpha$.  

**Fix (sketch).**

```python
dx = spacing  # user-provided
N  = x.size(-1)
freq = torch.fft.fftfreq(N, d=dx)         # cycles / unit-x
omega = 2*torch.pi*freq                    # rad / unit-x
kernel = (1j*omega).pow(alpha)
```

Also define behaviour at $\omega=0$: set $0^\alpha=0$ for $\alpha>0$.

## 4) Convergence and aliasing claims need correction

You state “FFT method converges with order $O(N^{-\alpha})$” and “aliasing prevention: $\alpha<1$ or use anti-aliasing filters”. Neither is generally true.  

**Fix.**

-   **Convergence.** For periodic, sufficiently smooth $f$, spectral differentiation is *spectrally accurate* (error decays faster than any power of $N$), but the *fractional* operator amplifies high-$k$ modes by $|\xi|^\alpha$, so for functions with only $p$ derivatives, the error becomes algebraic $O(N^{-(p-\alpha)})$. State convergence in terms of Sobolev/Gevery regularity, not just $\alpha$.
-   **Aliasing.** Aliasing is about nonlinearity and truncation, not the order being $<1$. Recommend the standard **3/2-dealiasing rule** (zero-pad to $3N/2$, apply operator, truncate), or explicit low-pass filtering after applying $|\xi|^\alpha$. If you keep the “sinc” filter suggestion, label it as a *windowing* heuristic, not a guarantee. 

## 5) Function spaces and boundary terms

Your adjoint identities assume periodic boundary conditions or vanishing tails; otherwise Caputo/RL adjoints pick up boundary terms. Make the domain explicit (e.g., $f\in H^\alpha_\mathrm{per}$ on $[0,L]$, or tempered $L^2(\mathbb R)$) so the adjoint statements hold without caveats. 

**Fix.** Add a paragraph specifying domains and testing inner products:

$\langle D^\alpha f,g\rangle_{L^2}=\langle f,(D^\alpha)^*g\rangle_{L^2}+B(f,g),$

and state when $B\equiv 0$.

## 6) Mellin: discrete story is underspecified

“$\tilde f_{k-\alpha}$” suggests a continuous shift in the Mellin variable, but you haven’t specified the sampling (log-grid), interpolation, or how you compute the discrete Mellin transform in $O(N\log N)$.  

**Fix.**

-   Work on a **log-grid** $x_n=x_0 e^{n\Delta}$; the Mellin transform is a Fourier transform in $\log x$. Implement with an FFT in log-space (“FFTLog”-style).
-   The gamma-ratio kernel $\Gamma(s)/\Gamma(s-\alpha)$ then multiplies $\hat f(\cdot)$ *with a frequency shift*: you’ll need fractional-bin interpolation (Fourier-domain phase ramp) rather than indexing at $k-\alpha$.
-   State the vertical strip $\Re(s)\in(\alpha+\epsilon,\,\sigma_{\max})$ where the inversion contour lives to control the gamma ratio. 

## 7) Learnable $\alpha$: you’re leaving gradients on the table

Your `backward` returns gradients only w.r.t. $x$. For many models $\alpha$ is a parameter to learn. Add $\partial L/\partial \alpha$:

$\frac{\partial D^\alpha f}{\partial \alpha} =\mathcal F^{-1}\!\Big[ (i\xi)^\alpha \log(i\xi)\,\hat f(\xi)\Big],$

then use the inner product with $\partial L/\partial (D^\alpha f)$ to accumulate `grad_alpha`. (For Riesz, replace $(i\xi)^\alpha$ with $|\xi|^\alpha$ and $\log|\xi|$.)

**Fix (sketch).**

```python
if ctx.method == "fft":
    g = torch.fft.fft(grad_output)
    fhat = torch.fft.fft(ctx.saved_input)      # save input in forward
    dD_dalpha = torch.fft.ifft(kernel * (torch.log(1j*omega)) * fhat)
    grad_alpha = (grad_output.conj()*dD_dalpha.real).sum().real
    return grad_x.real, grad_alpha, None
```

## 8) API & numerical hygiene in the code

-   **Save what you need.** In `forward` you save only `kernel`, but your Mellin `backward` references `frequencies` without saving it; that will error. Save `frequencies`, `x` (or $\hat f$) if you’ll need $\partial/\partial\alpha$. 
-   **Use rFFT for real signals.** Saves memory and locks Hermitian symmetry.
-   **Device/dtype-safe.** Create `kernel` with `x.dtype`/`x.device`; avoid Python complex in mixed precision.
-   **Batching/ND.** Generalise to `fftn`/`rfftn` with an `axes` argument and broadcasted kernels.
-   **Tempered/regularised operator.** Your regulariser divides by $1+\varepsilon|\xi|^\alpha$, which *changes the operator*. If your intent is stability, prefer (i) *tempered* fractional derivatives (replace $|\xi|^\alpha\mapsto(\lambda^2+|\xi|^2)^{\alpha/2}$), or (ii) Tikhonov on the *loss*, not the operator. Clarify the rationale. 

## 9) “Method selection” guidance needs sharper guard-rails

“FFT for periodic, Mellin for non-periodic” is too coarse. The Fourier route also works on non-periodic data with padding/windowing; Mellin excels for multiplicative scalings. Expand to: (a) periodic / compact-support with padding → FFT (Riesz); (b) scale-invariant kernels or power-laws → Mellin/log-FFT; (c) bounded intervals with initial-condition semantics → consider **Caputo/RL via GL** (time-domain) or Fourier with domain extension. 

------

# Suggested “tightened” statements to drop in

-   **Operator and adjoint (Fourier, unitary convention).**
     Let $D^\alpha$ be the (Riesz) fractional derivative on $\mathbb T_L$ with symbol $|\xi|^\alpha$. Then for $f,g\in L^2(\mathbb T_L)$,

    Dαf^(ξ)=∣ξ∣αf^(ξ),⟨Dαf,g⟩=⟨f,Dαg⟩.\widehat{D^\alpha f}(\xi)=|\xi|^\alpha\hat f(\xi),\quad \langle D^\alpha f,g\rangle=\langle f,D^\alpha g\rangle.

    In the *Weyl* case with $(i\xi)^\alpha$, the adjoint symbol is $\overline{(i\xi)^\alpha}=(-i\xi)^\alpha$. Therefore

    ∂L∂f=F−1 ⁣[Kα‾ F ⁣(∂L∂(Dαf))].\frac{\partial L}{\partial f}=\mathcal F^{-1}\!\left[\overline{K_\alpha}\,\mathcal F\!\left(\frac{\partial L}{\partial (D^\alpha f)}\right)\right].

    (This aligns the theorem with your code path.)  

-   **Discrete implementation (with spacing).**
     For $x_n=f(n\Delta x)$,

    Dαxn=IFFT[(iωk)α x^k],ωk=2πNΔxk.D^\alpha x_n=\mathrm{IFFT}\Big[(i\omega_k)^\alpha\,\widehat x_k\Big],\quad \omega_k=\tfrac{2\pi}{N\Delta x}k.

    (Include zero-mode and branch handling.) 

-   **Mellin/log-FFT implementation.**
     On a log grid $x_n=x_0e^{n\Delta}$ with $g(n)=f(x_n)\,x_n^{1/2}$, compute $G(\nu)=\mathrm{FFT}[g]$. The Mellin derivative acts as
     $\widetilde{D^\alpha f}(\nu) = \tfrac{\Gamma(c+i\nu)}{\Gamma(c+i\nu-\alpha)}\,\tilde f(\nu)$.
     Implement the *fractional shift* in $\nu$ via a complex phase $e^{-i\alpha\,\varphi(\nu)}$ rather than index arithmetic. (Spell this out in section 3.2.)

------

# Validation & ablation you should add (quick wins)

1.  **Adjoint test.** Numerically verify $\langle D^\alpha f,g\rangle=\langle f,(D^\alpha)^* g\rangle$ for random $f,g$ (with windowing) across $\alpha\in\{0.2,0.5,1.2\}$. This catches branch and conjugation bugs immediately.
2.  **Grid-scaling test.** Fix a smooth periodic $f$, vary $\Delta x$ and $N$, confirm first-order invariance under consistent $L$.
3.  **Riesz vs Weyl comparison.** Show realness and error vs. analytic $e^{ikx}$ benchmarks.
4.  **Learnable $\alpha$.** Finite-difference $\partial L/\partial \alpha$ vs. analytic gradient using $\log(i\xi)$ (complex-step differentiation is excellent here).
5.  **Mellin vs Fourier on power-laws.** Use $f(x)=x^\beta$ and show the Mellin route is superior in dynamic range, matching your Section 8 aims. 

------

# Documentation tweaks

-   Replace the “first practical implementation” claim with something like “a practical, spectrally-based AD framework…”, unless you add a brief related-work note.
-   In “Stability & Convergence”, replace blanket rates with regularity-dependent statements and move the “sinc” window to a “windowing/dealiasing” subsection.  
-   Clarify the domain assumptions (periodic vs. $\mathbb R$) next to each adjoint statement. 

------

# Minimal code diffs (directional)

-   **FFT kernel (Riesz) with proper scaling & rFFT.**

```python
def riesz_kernel(alpha, shape, dx, device, dtype):
    N = shape[-1]
    freq = torch.fft.rfftfreq(N, d=dx).to(device=device, dtype=dtype)
    omega = 2*torch.pi*freq
    return omega.abs().pow(alpha)  # real, Hermitian

def forward_fft_riesz(x, alpha, dx):
    X = torch.fft.rfft(x, dim=-1)
    K = riesz_kernel(alpha, x.shape, dx, x.device, x.dtype)
    Y = X * K
    return torch.fft.irfft(Y, n=x.shape[-1], dim=-1)
```

-   **Backward (adjoint is itself for Riesz).**

```python
def backward_fft_riesz(grad_out, alpha, dx):
    return forward_fft_riesz(grad_out, alpha, dx)  # self-adjoint
```

-   **Gradient w.r.t. $\alpha$.**

```python
def d_forward_dalpha_fft_riesz(x, alpha, dx):
    X = torch.fft.rfft(x, dim=-1)
    freq = torch.fft.rfftfreq(x.size(-1), d=dx).to(x)
    omega = 2*torch.pi*freq
    K_alpha = omega.abs().pow(alpha)
    dK_dalpha = K_alpha * torch.log(omega.abs().clamp_min(torch.finfo(x.dtype).tiny))
    Y = X * dK_dalpha
    return torch.fft.irfft(Y, n=x.size(-1), dim=-1)
```

------

# Bottom line

You’re close. Make the adjoint usage consistent (use $K^*$ in the gradient), resolve the branch/realness and Δx/2π issues, state function spaces to silence boundary terms, and beef up the Mellin discretisation with a log-FFT implementation and fractional shifts. Add learnable-$\alpha$ gradients and a small validation battery. With those, this becomes a clean, general “spectral-AD for fractional operators” that you can confidently ship and cite in your fPINN/fSNN work.