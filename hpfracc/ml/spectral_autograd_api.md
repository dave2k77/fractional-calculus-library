# Spectral Fractional Autograd API

## Goals
- Unified, backend-agnostic fractional autograd for Mellin, FFT, and Laplacian methods
- Drop-in `torch.autograd.Function` and `nn.Module` interfaces
- Hooks for stochastic memory sampling and probabilistic fractional orders

## Namespaces
- `hpfracc.ml.spectral_autograd`

## Core Classes
- `SpectralFractionalEngine(method: Literal["mellin","fft","laplacian"], backend: Literal["pytorch","jax","numba"])`
  - `forward(x: Tensor, alpha: float, **kwargs) -> Tuple[Tensor, Tuple]`
  - `backward(grad_output: Tensor, saved: Tuple) -> Tensor`
  - Internal hooks: `_to_spectral`, `_apply_fractional_operator`, `_from_spectral`

- `MellinEngine(SpectralFractionalEngine)`
  - `_to_spectral`: Mellin transform
  - `_apply_fractional_operator`: multiply by `s**alpha`
  - `_from_spectral`: inverse Mellin

- `FFTEngine(SpectralFractionalEngine)`
  - `_to_spectral`: FFT
  - `_apply_fractional_operator`: multiply by `(i*omega)**alpha`
  - `_from_spectral`: IFFT

- `LaplacianEngine(SpectralFractionalEngine)`
  - `_to_spectral`: FFT
  - `_apply_fractional_operator`: multiply by `|xi|**alpha`
  - `_from_spectral`: IFFT

## Unified Autograd
- `class FractionalAutogradFunction(torch.autograd.Function)`
  - `forward(ctx, x, alpha: float, method: str = "auto", engine: str = "fft", **kwargs)`
  - `backward(ctx, grad_output)`

- `class FractionalAutogradLayer(nn.Module)`
  - `__init__(alpha: float, method: str = "auto", engine: str = "fft")`
  - `forward(x)`

## Stochastic Extensions (optional kwargs)
- `stochastic_memory: bool = False`
- `memory_sampler: Optional[Callable] = None` (importance sampling / stratified)
- `probabilistic_alpha: bool = False`
- `alpha_dist: Optional[Distribution] = None` (reparameterized when possible)
- `variance_budget: Optional[float] = None` (adaptive K)

## Error Handling & Validation
- Validate `alpha` ranges per-operator
- Deterministic fallback path when stochastic flags disabled

## Minimal Example
```python
layer = FractionalAutogradLayer(alpha=0.6, method="derivative", engine="fft")
y = layer(x)
```
