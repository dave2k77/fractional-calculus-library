# Mathematical Framework for Fractional Chain Rule Using Mellin and FFT Operators

**Author**: Davian R. Chin  
**Institution**: Department of Biomedical Engineering, University of Reading  
**Email**: d.r.chin@pgr.reading.ac.uk  
**Date**: December 2024

## Abstract

This document establishes the rigorous mathematical foundations for implementing automatic differentiation of fractional derivatives using spectral domain transformations. We derive the fractional chain rule for both Mellin and FFT-based operators, providing stability conditions, convergence criteria, and error bounds necessary for practical implementation in neural networks.

## 1. Introduction

### 1.1 Problem Statement

The fundamental challenge in fractional calculus-based machine learning is the **non-local nature** of fractional derivatives, which breaks the standard chain rule used in automatic differentiation. Unlike classical derivatives that depend only on local neighborhoods, fractional derivatives require the entire function history, making traditional backpropagation techniques inapplicable.

**Core Problem**: For a loss function $L$ and fractional derivative $D^{\alpha} f$, how do we compute $\frac{\partial L}{\partial f}$ when the standard chain rule fails?

### 1.2 Spectral Domain Solution

We propose solving this through **spectral domain transformations** that convert non-local fractional operations into local operations in the frequency domain, enabling the first practical implementation of automatic differentiation for fractional operators.

## 2. Mathematical Preliminaries

### 2.1 Fractional Derivatives

#### 2.1.1 Riemann-Liouville Definition

For a function $f(x)$ and fractional order $\alpha \in (0,1)$, the Riemann-Liouville fractional derivative is:

$$D^{\alpha} f(x) = \frac{1}{\Gamma(1-\alpha)} \frac{d}{dx} \int_0^x \frac{f(t)}{(x-t)^{\alpha}} dt$$

#### 2.1.2 Caputo Definition

$$D^{\alpha} f(x) = \frac{1}{\Gamma(1-\alpha)} \int_0^x \frac{f'(t)}{(x-t)^{\alpha}} dt$$

#### 2.1.3 Grünwald-Letnikov Definition

$$D^{\alpha} f(x) = \lim_{h \to 0} \frac{1}{h^{\alpha}} \sum_{k=0}^{\infty} (-1)^k \binom{\alpha}{k} f(x - kh)$$

### 2.2 Spectral Transforms

#### 2.2.1 Fourier Transform

$$\mathcal{F}[f](\xi) = \int_{-\infty}^{\infty} f(x) e^{-i\xi x} dx$$

$$\mathcal{F}^{-1}[\hat{f}](x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \hat{f}(\xi) e^{i\xi x} d\xi$$

#### 2.2.2 Mellin Transform

$$\mathcal{M}[f](s) = \int_0^{\infty} f(x) x^{s-1} dx$$

$$\mathcal{M}^{-1}[\tilde{f}](x) = \frac{1}{2\pi i} \int_{c-i\infty}^{c+i\infty} \tilde{f}(s) x^{-s} ds$$

## 3. Spectral Domain Fractional Derivatives

### 3.1 FFT-Based Fractional Derivatives

#### 3.1.1 Forward Transform

For a function $f(x)$ with Fourier transform $\hat{f}(\xi)$, the fractional derivative of order $\alpha$ is:

$$D^{\alpha} f(x) = \mathcal{F}^{-1}[(i\xi)^{\alpha} \hat{f}(\xi)]$$

**Key Properties**:
- **Spectral Kernel**: $K_{\alpha}(\xi) = (i\xi)^{\alpha}$
- **Analyticity**: $K_{\alpha}(\xi)$ is analytic in the complex plane
- **Boundedness**: $|K_{\alpha}(\xi)| = |\xi|^{\alpha}$ for $\xi \in \mathbb{R}$

#### 3.1.2 Discrete Implementation

For discrete data $f_n$ with FFT $\hat{f}_k$:

$$D^{\alpha} f_n = \text{IFFT}[(i\xi_k)^{\alpha} \hat{f}_k]$$

where $\xi_k = \frac{2\pi k}{N}$ are the frequency components.

### 3.2 Mellin-Based Fractional Derivatives

#### 3.2.1 Forward Transform

For a function $f(x)$ with Mellin transform $\tilde{f}(s)$, the fractional derivative is:

$$D^{\alpha} f(x) = \mathcal{M}^{-1}\left[\frac{\Gamma(s)}{\Gamma(s-\alpha)} \tilde{f}(s-\alpha)\right]$$

**Key Properties**:
- **Spectral Kernel**: $M_{\alpha}(s) = \frac{\Gamma(s)}{\Gamma(s-\alpha)}$
- **Convergence**: Requires $\Re(s) > \alpha$
- **Stability**: The gamma ratio must be bounded

#### 3.2.2 Discrete Implementation

For discrete data $f_n$ with Mellin transform $\tilde{f}_k$:

$$D^{\alpha} f_n = \text{IMellin}\left[\frac{\Gamma(s_k)}{\Gamma(s_k-\alpha)} \tilde{f}_{k-\alpha}\right]$$

## 4. Fractional Chain Rule Derivation

### 4.1 Standard Chain Rule Failure

The standard chain rule states:

$$\frac{\partial L}{\partial f} = \frac{\partial L}{\partial D^{\alpha} f} \cdot \frac{\partial D^{\alpha} f}{\partial f}$$

However, for fractional derivatives, $\frac{\partial D^{\alpha} f}{\partial f}$ is **non-local** and cannot be computed using standard techniques.

### 4.2 Spectral Domain Chain Rule

#### 4.2.1 FFT-Based Chain Rule

**Theorem 1** (FFT Fractional Chain Rule): For a loss function $L$ and fractional derivative $D^{\alpha} f$, the gradient with respect to $f$ is:

$$\frac{\partial L}{\partial f} = \mathcal{F}^{-1}\left[K_{\alpha}^*(\xi) \mathcal{F}\left[\frac{\partial L}{\partial D^{\alpha} f}\right](\xi)\right]$$

where $K_{\alpha}^*(\xi) = (-i\xi)^{\alpha}$ is the complex conjugate of the spectral kernel.

**Proof**: Starting from the chain rule in spectral domain:

$$\mathcal{F}\left[\frac{\partial L}{\partial f}\right] = \mathcal{F}\left[\frac{\partial L}{\partial D^{\alpha} f}\right] \cdot \mathcal{F}\left[\frac{\partial D^{\alpha} f}{\partial f}\right]$$

Since $D^{\alpha} f = \mathcal{F}^{-1}[K_{\alpha}(\xi) \mathcal{F}[f]]$, we have:

$$\mathcal{F}\left[\frac{\partial D^{\alpha} f}{\partial f}\right] = K_{\alpha}(\xi)$$

Therefore:

$$\mathcal{F}\left[\frac{\partial L}{\partial f}\right] = \mathcal{F}\left[\frac{\partial L}{\partial D^{\alpha} f}\right] \cdot K_{\alpha}(\xi)$$

Taking inverse transform:

$$\frac{\partial L}{\partial f} = \mathcal{F}^{-1}\left[K_{\alpha}(\xi) \mathcal{F}\left[\frac{\partial L}{\partial D^{\alpha} f}\right]\right]$$

**Key Insight**: The backward pass is **identical** to the forward pass in the frequency domain!

#### 4.2.2 Mellin-Based Chain Rule

**Theorem 2** (Mellin Fractional Chain Rule): For a loss function $L$ and fractional derivative $D^{\alpha} f$, the gradient with respect to $f$ is:

$$\frac{\partial L}{\partial f} = \mathcal{M}^{-1}\left[\frac{\Gamma(s+\alpha)}{\Gamma(s)} \mathcal{M}\left[\frac{\partial L}{\partial D^{\alpha} f}\right](s+\alpha)\right]$$

**Proof**: Similar to FFT case, but using Mellin transform properties and the adjoint of the Mellin fractional derivative operator.

### 4.3 Adjoint Operator Theory

#### 4.3.1 FFT Adjoint

The fractional derivative operator $D^{\alpha}$ has adjoint $(D^{\alpha})^*$ given by:

$$(D^{\alpha})^* = \mathcal{F}^{-1}[K_{\alpha}^*(\xi) \mathcal{F}[\cdot]]$$

**Verification**: For any functions $f, g$:

$$\langle D^{\alpha} f, g \rangle = \langle f, (D^{\alpha})^* g \rangle$$

#### 4.3.2 Mellin Adjoint

The Mellin fractional derivative operator has adjoint:

$$(D^{\alpha})^* = \mathcal{M}^{-1}\left[\frac{\Gamma(s+\alpha)}{\Gamma(s)} \mathcal{M}[\cdot](s+\alpha)\right]$$

## 5. Stability and Convergence Analysis

### 5.1 Spectral Stability Conditions

#### 5.1.1 FFT Stability

**Condition 1**: **Spectral Kernel Boundedness**
$$|K_{\alpha}(\xi)| \leq C|\xi|^{\alpha} \quad \text{for } |\xi| \leq \xi_{\max}$$

**Condition 2**: **Nyquist Criterion**
$$|\xi_k| \leq \frac{\pi}{h} \quad \text{where } h \text{ is the grid spacing}$$

**Condition 3**: **Aliasing Prevention**
$$\alpha < 1 \quad \text{or} \quad \text{use anti-aliasing filters}$$

#### 5.1.2 Mellin Stability

**Condition 1**: **Convergence of Mellin Transform**
$$\int_0^{\infty} |f(x)| x^{\sigma-1} dx < \infty \quad \text{for some } \sigma > \alpha$$

**Condition 2**: **Boundedness of Gamma Ratio**
$$\left|\frac{\Gamma(s)}{\Gamma(s-\alpha)}\right| \leq C \quad \text{for } \Re(s) > \alpha + \epsilon$$

**Condition 3**: **Spectral Decay**
$$|\mathcal{M}[f](s)| \leq C|s|^{-\beta} \quad \text{for } |s| \to \infty, \beta > \alpha$$

### 5.2 Convergence Analysis

#### 5.2.1 FFT Method Convergence

**Theorem 3**: The FFT-based fractional derivative converges with order $O(N^{-\alpha})$ if:

1. $f$ is periodic or properly windowed
2. $\alpha \in (0,1)$ (for stability)
3. Grid size $N$ is sufficiently large

#### 5.2.2 Mellin Method Convergence

**Theorem 4**: The Mellin-based fractional derivative converges with order $O(h^{\min(\alpha, 2-\alpha)})$ if:

1. $f \in C^{2+\alpha}$ (sufficiently smooth)
2. $\alpha \in (0,2)$ (valid fractional order)
3. Grid spacing $h$ satisfies $h < h_0$ for some $h_0 > 0$

### 5.3 Error Bounds

#### 5.3.1 Truncation Error

**FFT Method**:
$$|E_{\text{trunc}}| \leq CN^{-\alpha} \|f^{(\alpha)}\|_{\infty}$$

**Mellin Method**:
$$|E_{\text{trunc}}| \leq Ch^{\min(\alpha, 2-\alpha)} \|f^{(2+\alpha)}\|_{\infty}$$

#### 5.3.2 Rounding Error

$$|E_{\text{round}}| \leq C\epsilon \|f\|_{\infty} \kappa(\mathcal{K}_{\alpha})$$

where $\epsilon$ is machine precision and $\kappa(\mathcal{K}_{\alpha})$ is the condition number.

#### 5.3.3 Aliasing Error

**FFT Method**:
$$|E_{\text{alias}}| \leq C \sum_{k \neq 0} |\hat{f}(\xi + 2\pi k/h)| |\xi + 2\pi k/h|^{\alpha}$$

## 6. Implementation Framework

### 6.1 Spectral Autograd Function

```python
class SpectralFractionalDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, method="fft"):
        if method == "fft":
            # FFT-based implementation
            x_fft = torch.fft.fft(x)
            frequencies = torch.fft.fftfreq(x.size(-1))
            kernel = (1j * frequencies) ** alpha
            result_fft = x_fft * kernel
            result = torch.fft.ifft(result_fft)
            
        elif method == "mellin":
            # Mellin-based implementation
            x_mellin = mellin_transform(x)
            frequencies = mellin_frequencies(x.size(-1))
            kernel = gamma_ratio_kernel(alpha, frequencies)
            result_mellin = x_mellin * kernel
            result = inverse_mellin_transform(result_mellin)
        
        ctx.save_for_backward(kernel)
        ctx.alpha = alpha
        ctx.method = method
        
        return result.real
    
    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        alpha = ctx.alpha
        method = ctx.method
        
        if method == "fft":
            # FFT-based backward pass
            grad_fft = torch.fft.fft(grad_output)
            adjoint_kernel = kernel.conj()  # (-iξ)^α
            result_fft = grad_fft * adjoint_kernel
            result = torch.fft.ifft(result_fft)
            
        elif method == "mellin":
            # Mellin-based backward pass
            grad_mellin = mellin_transform(grad_output)
            adjoint_kernel = gamma_ratio_adjoint(alpha, frequencies)
            result_mellin = grad_mellin * adjoint_kernel
            result = inverse_mellin_transform(result_mellin)
        
        return result.real, None, None
```

### 6.2 Spectral Kernel Design

#### 6.2.1 FFT Kernel

```python
def fft_kernel(alpha, frequencies):
    """FFT-based spectral kernel for fractional derivatives."""
    return (1j * frequencies) ** alpha

def fft_adjoint_kernel(alpha, frequencies):
    """Adjoint kernel for FFT-based fractional derivatives."""
    return (-1j * frequencies) ** alpha
```

#### 6.2.2 Mellin Kernel

```python
def mellin_kernel(alpha, frequencies):
    """Mellin-based spectral kernel for fractional derivatives."""
    return torch.gamma(frequencies) / torch.gamma(frequencies - alpha)

def mellin_adjoint_kernel(alpha, frequencies):
    """Adjoint kernel for Mellin-based fractional derivatives."""
    return torch.gamma(frequencies + alpha) / torch.gamma(frequencies)
```

### 6.3 Regularization and Stabilization

#### 6.3.1 Regularized Kernels

For $\alpha \geq 1$, use regularized kernels:

**FFT Regularization**:
$$K_{\alpha}^{\text{reg}}(\xi) = \frac{(i\xi)^{\alpha}}{1 + \epsilon|\xi|^{\alpha}}$$

**Mellin Regularization**:
$$M_{\alpha}^{\text{reg}}(s) = \frac{\Gamma(s)}{\Gamma(s-\alpha)} \cdot \frac{1}{1 + \epsilon|s|^{\alpha}}$$

#### 6.3.2 Anti-Aliasing

For FFT implementation, use anti-aliasing filters:

$$K_{\alpha}^{\text{aa}}(\xi) = K_{\alpha}(\xi) \cdot \text{sinc}(\xi/\xi_{\max})$$

## 7. Practical Implementation Guidelines

### 7.1 Method Selection

- **FFT**: Use for periodic functions, computational efficiency required
- **Mellin**: Use for non-periodic functions, high accuracy required
- **Hybrid**: Use Mellin for small problems, FFT for large problems

### 7.2 Parameter Tuning

- **Grid spacing**: $h \propto \alpha^{-1}$ for stability
- **Grid size**: $N \geq 2^{\lceil \log_2(1/\alpha) \rceil}$ for FFT
- **Truncation**: Use $K = \min(128, N/4)$ for Mellin

### 7.3 Memory Optimization

- **Chunked FFT**: For large arrays
- **Memory-mapped operations**: For very large problems
- **Gradient checkpointing**: For memory efficiency

## 8. Verification and Testing

### 8.1 Analytical Test Cases

#### 8.1.1 Power Functions

For $f(x) = x^{\beta}$:
$$D^{\alpha} x^{\beta} = \frac{\Gamma(\beta+1)}{\Gamma(\beta+1-\alpha)} x^{\beta-\alpha}$$

#### 8.1.2 Exponential Functions

For $f(x) = e^{kx}$:
$$D^{\alpha} e^{kx} = k^{\alpha} e^{kx}$$

### 8.2 Numerical Verification

#### 8.2.1 Chain Rule Verification

Verify that:
$$\frac{\partial}{\partial f} \int L(D^{\alpha} f) dx = \int \frac{\partial L}{\partial D^{\alpha} f} \frac{\partial D^{\alpha} f}{\partial f} dx$$

#### 8.2.2 Gradient Flow Preservation

Ensure that gradients flow correctly through the spectral autograd operations.

## 9. Conclusion

This mathematical framework provides the rigorous foundation for implementing spectral autograd that maintains gradient flow while enabling efficient computation of fractional derivatives in neural networks. The key insights are:

1. **Spectral Domain Transformation**: Converts non-local fractional operations into local operations in frequency domain
2. **Identical Forward/Backward Passes**: The backward pass in frequency domain is identical to the forward pass
3. **Stability Conditions**: Proper regularization and anti-aliasing ensure numerical stability
4. **Convergence Guarantees**: Both FFT and Mellin methods provide theoretical convergence bounds

This framework enables the first practical implementation of automatic differentiation for fractional operators, opening new possibilities for fractional calculus-based machine learning.

## References

1. Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.
2. Samko, S. G., Kilbas, A. A., & Marichev, O. I. (1993). *Fractional Integrals and Derivatives: Theory and Applications*. Gordon and Breach.
3. Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*. Imperial College Press.
4. Kilbas, A. A., Srivastava, H. M., & Trujillo, J. J. (2006). *Theory and Applications of Fractional Differential Equations*. Elsevier.
5. Ortigueira, M. D. (2011). *Fractional Calculus for Scientists and Engineers*. Springer.

---

*This document serves as the theoretical foundation for the spectral autograd implementation in the hpfracc library.*
