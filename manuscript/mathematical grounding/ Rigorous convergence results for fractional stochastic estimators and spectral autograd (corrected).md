# Rigorous convergence results for fractional stochastic estimators and spectral autograd (corrected)

## Notation and setting

-   Let $D^\alpha$ denote a (linear) fractional derivative of order $\alpha>0$ acting on a suitable Sobolev space.
-   For Monte Carlo statements, expectations are under the target $p(\alpha)$ or proposal $q(\alpha)$ as indicated.
-   For spectral statements, we work on a periodic domain (Fourier basis) or any orthonormal eigenbasis $\{e_k\}$ with eigenvalues $\{s_k\}$ growing like $|k|$ (explicitly noted where used).

------

## Theorem 1 (Fractional importance sampling; unbiasedness, variance, tails)

**Estimator.** For i.i.d. $\alpha_i\sim q$, importance weights $w_i=p(\alpha_i)/q(\alpha_i)$, and an integrable target $h(\alpha)\equiv f(D^{\alpha}\phi)$,

$\widehat\mu_n \;=\; \frac1n\sum_{i=1}^n w_i\,h(\alpha_i),  \qquad  \mu=\mathbb E_{p}[h(\alpha)].$

This matches your original set-up. 

**Assumptions.**
 A1. $\mathbb E_q[w^2]=\rho<\infty$. 
 A2′. $h\in L^2(p)$ (square-integrable target; replaces uniform boundedness).
 A3. Fractional derivatives $D^\alpha\phi$ exist (e.g. in the spectral sense) on the support of $q$. 

**Unbiasedness.** $\mathbb E[\widehat\mu_n]=\mu$ (standard IS argument). 

**Variance and Chebyshev tail.** With independence,

$\mathrm{Var}(\widehat\mu_n)=\frac1n\mathrm{Var}(w h) \;\le\;\frac1n\mathbb E_q[w^2h^2] \;\le\;\frac{\rho}{n}\,\mathbb E_p[h^2].$

Hence, for any $\varepsilon>0$,

$\mathbb P\!\left(|\widehat\mu_n-\mu|>\varepsilon\right) \;\le\; \frac{\mathrm{Var}(\widehat\mu_n)}{\varepsilon^2} \;\le\;\frac{\rho\,\mathbb E_p[h^2]}{n\,\varepsilon^2}.$

This **replaces** the spurious factor “4” that was introduced after a correct Chebyshev step.  

**Effective sample size (ESS).** For normalised weights the classical identity gives

$\mathbb E[n_{\mathrm{eff}}]\;\approx\; \frac{n}{1+\mathrm{cv}^2(w)} \;=\;\frac{n}{\rho},$

since $\mathrm{cv}^2(w)=\rho-1$ with $\mathbb E_q[w]=1$. Thus $n_{\mathrm{eff}}\le n$ and scales like $n/\rho$. This **replaces** the incorrect $n_{\mathrm{eff}}\ge n^2/(1+\rho)$ bound. 

>   **Summary of changes.**
>    • Keep unbiasedness and the base variance derivation. 
>    • Remove the factor “4” in the probability/MSE bounds. 
>    • Replace the ESS corollary by the $n/\rho$ scaling. 

------

## Theorem 2 (REINFORCE with stochastic fractional order)

**Estimator.** For $\alpha\sim \pi(\alpha\mid\theta)$,

$\widehat{\nabla}_\theta  \;=\; \frac1n\sum_{i=1}^n f(D^{\alpha_i}\phi)\,\nabla_\theta\log \pi(\alpha_i\mid\theta).$

Matches your statement. 

**Assumptions.**
 B1. $\pi(\cdot\mid\theta)$ differentiable in $\theta$; B2. $\mathbb E[f^2(D^\alpha\phi)]<\infty$;
 B3. $\mathbb E\|\nabla_\theta\log\pi(\alpha\mid\theta)\|^2<\infty$;
 B4. Dominated convergence to swap $\nabla_\theta$ and $\int$. 

>   For **uniform** convergence over a compact $\Theta$, additionally assume:
>    (i) $\Theta$ compact; (ii) $\nabla_\theta\log\pi$ is Lipschitz in $\theta$ with an $L^2$ envelope;
>    (iii) $f(D^\alpha\phi)$ admits an integrable envelope uniform in $\theta$.
>    This strengthens your uniformity corollary, which was previously under-specified. 

**Unbiasedness.** $\mathbb E[\widehat{\nabla}_\theta]=\nabla_\theta \mathbb E[f(D^\alpha\phi)]$ (score-function identity + B4).  

**Variance and CLT.**

$\mathrm{Var}(\widehat{\nabla}_\theta)=\frac1n\,\mathrm{Var}\!\big(f(D^\alpha\phi)\nabla_\theta\log\pi\big) \;\le\;\frac1n\,\mathbb E[f^2(D^\alpha\phi)]\,\mathbb E\|\nabla_\theta\log\pi\|^2.$

Thus $\mathrm{Var}(\widehat{\nabla}_\theta)=O(1/n)$. A standard CLT applies under B2–B3.  

**Error statement (fixed).** Replace the contradictory bias bound $|\mathbb E[\widehat{\nabla}_\theta]-\nabla_\theta \cdot|\le C/\sqrt n$ (LHS $=0$) with either:
 (i) **RMSE:** $\sqrt{\mathbb E\|\widehat{\nabla}_\theta-\nabla_\theta\|^2}\le C/\sqrt n$; or
 (ii) a high-probability deviation inequality. This corrects the original “error bound”. 

------

## Theorem 3 (Spectral fractional autograd; consistent rates in Sobolev scales)

We make explicit the standard function-space setting behind your spectral construction.

**Setting.** Let $\phi\in H^{s}(\mathbb T^d)$ with $s>\alpha>0$, and let $P_n$ be the orthogonal projector onto $\{\,|k|\le n\,\}$ Fourier modes. The fractional operator $D^\alpha: H^{\alpha}\to L^2$ is bounded. (This reverses the earlier, incorrect mapping $T:L^2\to H^\alpha$.) 

**Forward spectral truncation rate (corrected).**
 Define $\widehat{D}^\alpha\phi := D^\alpha P_n \phi$. Then

$\|\,\widehat{D}^\alpha\phi - D^\alpha\phi\,\|_{L^2} \;=\;\|\,D^\alpha(\phi-P_n\phi)\,\|_{L^2} \;\lesssim\; n^{-(s-\alpha)}\|\phi\|_{H^{s}}, \quad s>\alpha.$

This **replaces** the non-standard $n^{-\alpha/2}$ rate and the intermediate tail bound $\sum_{k>n}s_k^{2\alpha}|\langle\phi,e_k\rangle|^2\lesssim n^{-\alpha}\|\phi\|_{H^\alpha}^2$, which do not hold in general without extra structure.  

**Adjoint (backward) error.** If back-prop uses the adjoint $(D^\alpha)^*$ and the backward operator $\mathcal B$ is bounded on $L^2$ (your adjoint consistency assumption), then

$\|\widehat{\nabla}f-\nabla f\|_{L^2} \;=\;\|\mathcal B\big((\widehat{D}^\alpha)^*-(D^\alpha)^*\big)\,\nabla L\|_{L^2} \;\lesssim\; n^{-(s-\alpha)}\|\nabla L\|_{H^{s-\alpha}}.$

This **replaces** the unmotivated $\mathbb E\|\widehat{\nabla}f-\nabla f\|^2\le C_\alpha(\log n)/n$ claim. If you later prove a $(\log n)/n$ inequality in a specific model, it can be added as a **model-specific** refinement.  

>   **What changed here.**
>    • Replaced $n^{-\alpha/2}$ by the classical spectral rate $n^{-(s-\alpha)}$ for $s>\alpha$. 
>    • Corrected operator mapping and kept your adjoint consistency premise. 
>    • Removed the unproved $(\log n)/n$ gradient error unless separately established. 

**Remark (computational context).** The FFT-based forward/backward realises these operators with $O(n\log n)$ time and $O(n)$ memory, as in your earlier spectral framework. 

------

## Practical corollaries (corrected)

1.  **Sample complexity (to accuracy $\varepsilon$).**
     • Importance sampling: $n = O\!\left(\rho\,\mathbb E_p[h^2]/\varepsilon^2\right)$. (Chebyshev.)
     • REINFORCE: $n = O(\sigma^2/\varepsilon^2)$ with $\sigma^2=\mathbb E[f^2(D^\alpha\phi)]\,\mathbb E\|\nabla_\theta\log\pi\|^2$. 
     • Spectral autograd truncation: choose $n$ such that $n^{-(s-\alpha)}\lesssim \varepsilon$, i.e.

    n  ≳  ε−1/(s−α).n \;\gtrsim\; \varepsilon^{-1/(s-\alpha)}.

    This **replaces** the previously stated $n=O(1/(\varepsilon^2\log(1/\varepsilon)))$, which depended on an unproved $(\log n)/n$ bound. 

2.  **Optimal proposal for IS.** For *unnormalised* IS, the variance-minimising heuristic takes $q^\star(\alpha)\propto |h(\alpha)|\,p(\alpha)$. If you use **self-normalised IS**, the exact optimum differs; state explicitly which estimator is used. (Clarifies your previous corollary.) 

3.  **Control variates.** Replace the “up to 50%” claim by the standard formula: optimal control-variate reduction factor $1-\rho^2$, where $\rho$ is the correlation with the target. (Magnitude depends on the actual $\rho$.) 

------

## Short appendix: algorithms and numerical cautions

-   **Adjoint implementation.** In Fourier: forward multiplies by $(i\omega)^\alpha$; backward multiplies by $(-\!i\omega)^\alpha$ (the conjugate symbol), then inverse transform. Ensure the inverse operator matches the forward transform when using Mellin/Laplace. (This mirrors your earlier pseudocode.) 
-   **Numerical stability.** Address spectral leakage (periodicity), complex powers/branch cuts, and low-frequency regularisation $s^\alpha\mapsto (s^2+\varepsilon^2)^{\alpha/2}$; choose the transform (Fourier/Mellin/Laplace) to fit domain physics; chunk long signals. (As per your complexity/stability notes.) 

------

### What this rewrite fixes relative to your original

-   Chebyshev/MSE **without** the ad-hoc factor 4. 
-   ESS corollary corrected to $n/\rho$ scaling (no $n^2$ behaviour). 
-   REINFORCE **error statement** changed to RMSE/deviation; unbiasedness retained. 
-   Spectral autograd **rate** set to the standard $n^{-(s-\alpha)}$ (with $s>\alpha$); mapping $D^\alpha:H^\alpha\to L^2$ clarified; removed $(\log n)/n$ unless proven under extra hypotheses. 

------

**Complexity reminder.** The spectral realisation still delivers $O(n\log n)$ forward/backward with $O(n)$ memory—your key engineering win remains intact. 

If you want this as a standalone Markdown file with your original theorem numbering preserved, I can drop this directly into your repo and annotate the diff.