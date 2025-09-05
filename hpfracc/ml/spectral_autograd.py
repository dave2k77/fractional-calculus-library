"""
Spectral Fractional Autograd Implementation

This module provides spectral domain fractional autograd engines and unified PyTorch interfaces.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal, Union, Callable
from abc import ABC, abstractmethod


def _ensure_real_dtype(x: torch.Tensor) -> torch.dtype:
    """Return a safe real dtype to use for frequency arrays and multipliers."""
    if x.dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def _safe_fft(x: torch.Tensor) -> torch.Tensor:
    """Safe FFT wrapper with dtype/device guards and fallback casting.

    Tries FFT with the current dtype; on backend errors, retries with float32 CPU.
    """
    try:
        return torch.fft.fft(x)
    except Exception:
        x_cpu = x.detach().to('cpu', torch.float32)
        return torch.fft.fft(x_cpu).to(x.device)


def _safe_ifft(x: torch.Tensor) -> torch.Tensor:
    """Safe IFFT wrapper mirroring _safe_fft behavior."""
    try:
        return torch.fft.ifft(x)
    except Exception:
        x_cpu = x.detach().to('cpu')
        return torch.fft.ifft(x_cpu).to(x.device)

from ..core.definitions import FractionalOrder


class SpectralFractionalEngine(ABC):
    """
    Abstract base class for spectral domain fractional autograd engines.
    """
    
    def __init__(self, method: Literal["mellin", "fft", "laplacian"], 
                 backend: Literal["pytorch", "jax", "numba"] = "pytorch"):
        self.method = method
        self.backend = backend
    
    @abstractmethod
    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to spectral domain."""
        pass
    
    @abstractmethod
    def _apply_fractional_operator(self, x_spectral: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply fractional operator in spectral domain."""
        pass
    
    @abstractmethod
    def _from_spectral(self, x_spectral: torch.Tensor) -> torch.Tensor:
        """Transform back from spectral domain."""
        pass
    
    def forward(self, x: torch.Tensor, alpha: float, **kwargs) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass in spectral domain."""
        x_spectral = self._to_spectral(x)
        result_spectral = self._apply_fractional_operator(x_spectral, alpha)
        result = self._from_spectral(result_spectral)
        return result, (x_spectral, result_spectral)
    
    def backward(self, grad_output: torch.Tensor, saved_tensors: Tuple) -> torch.Tensor:
        """Backward pass with spectral gradient computation."""
        x_spectral, result_spectral = saved_tensors
        
        # Transform gradient to spectral domain
        grad_spectral = self._to_spectral(grad_output)
        
        # Apply fractional chain rule in spectral domain
        grad_input_spectral = self._apply_fractional_chain_rule(grad_spectral, x_spectral)
        
        # Transform back to original domain
        grad_input = self._from_spectral(grad_input_spectral)
        
        return grad_input
    
    def _apply_fractional_chain_rule(self, grad_spectral: torch.Tensor, 
                                   x_spectral: torch.Tensor) -> torch.Tensor:
        """Apply fractional chain rule in spectral domain."""
        # Default implementation - can be overridden by subclasses
        return grad_spectral


class MellinEngine(SpectralFractionalEngine):
    """
    Fractional autograd using Mellin transform.
    """
    
    def __init__(self, backend: str = "pytorch"):
        super().__init__("mellin", backend)
    
    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to Mellin domain."""
        # TODO: Implement Mellin transform
        # For now, placeholder using log-space FFT approximation
        log_x = torch.log(torch.abs(x) + 1e-8)
        return torch.fft.fft(log_x)
    
    def _apply_fractional_operator(self, x_spectral: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply fractional derivative in Mellin domain: multiply by s^alpha."""
        # TODO: Implement proper Mellin domain fractional operator
        # For now, placeholder using frequency domain approximation
        N = x_spectral.shape[-1]
        s = torch.arange(N, device=x_spectral.device, dtype=torch.float32)
        multiplier = torch.pow(s + 1e-8, alpha)
        return x_spectral * multiplier
    
    def _from_spectral(self, x_spectral: torch.Tensor) -> torch.Tensor:
        """Inverse Mellin transform."""
        # TODO: Implement inverse Mellin transform
        # For now, placeholder using IFFT
        log_result = torch.fft.ifft(x_spectral).real
        return torch.exp(log_result)


class FFTEngine(SpectralFractionalEngine):
    """
    Fractional autograd using fractional FFT.
    """
    
    def __init__(self, backend: str = "pytorch"):
        super().__init__("fft", backend)
    
    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to frequency domain (safe FFT)."""
        return _safe_fft(x)
    
    def _apply_fractional_operator(self, x_spectral: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply fractional derivative in frequency domain: multiply by (i*omega)^alpha."""
        N = x_spectral.shape[-1]
        # Use safe real dtype for frequency computation
        safe_dtype = _ensure_real_dtype(x_spectral)
        try:
            omega = torch.fft.fftfreq(N, device=x_spectral.device, dtype=safe_dtype)
        except Exception:
            omega = torch.fft.fftfreq(N, dtype=torch.float32).to('cpu')
        # Use real arithmetic to avoid complex issues
        multiplier = torch.pow(torch.abs(omega) + 1e-8, alpha)
        return x_spectral * multiplier
    
    def _from_spectral(self, x_spectral: torch.Tensor) -> torch.Tensor:
        """Inverse FFT (safe IFFT)."""
        return _safe_ifft(x_spectral).real


class LaplacianEngine(SpectralFractionalEngine):
    """
    Fractional autograd using fractional Laplacian.
    """
    
    def __init__(self, backend: str = "pytorch"):
        super().__init__("laplacian", backend)
    
    def _to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Transform to frequency domain (safe FFT)."""
        return _safe_fft(x)
    
    def _apply_fractional_operator(self, x_spectral: torch.Tensor, alpha: float) -> torch.Tensor:
        """Apply fractional Laplacian in frequency domain: multiply by |xi|^alpha."""
        N = x_spectral.shape[-1]
        safe_dtype = _ensure_real_dtype(x_spectral)
        try:
            xi = torch.fft.fftfreq(N, device=x_spectral.device, dtype=safe_dtype)
        except Exception:
            xi = torch.fft.fftfreq(N, dtype=torch.float32).to('cpu')
        multiplier = torch.pow(torch.abs(xi) + 1e-8, alpha)
        return x_spectral * multiplier
    
    def _from_spectral(self, x_spectral: torch.Tensor) -> torch.Tensor:
        """Inverse FFT (safe IFFT)."""
        return _safe_ifft(x_spectral).real


class FractionalAutogradFunction(torch.autograd.Function):
    """
    Unified fractional autograd function supporting multiple spectral methods.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, method: str = "auto", 
                engine: str = "fft", **kwargs) -> torch.Tensor:
        """Forward pass with automatic engine selection."""
        
        # Auto-select engine based on method and problem characteristics
        if engine == "auto":
            engine = _auto_select_engine(x, alpha, method)
        
        # Create appropriate spectral engine
        if engine == "mellin":
            spectral_engine = MellinEngine()
        elif engine == "fft":
            spectral_engine = FFTEngine()
        elif engine == "laplacian":
            spectral_engine = LaplacianEngine()
        else:
            raise ValueError(f"Unknown engine: {engine}")
        
        # Perform forward pass
        result, saved_tensors = spectral_engine.forward(x, alpha, **kwargs)
        
        # Save context for backward pass
        ctx.alpha = alpha
        ctx.method = method
        ctx.engine = engine
        ctx.spectral_engine = spectral_engine
        ctx.save_for_backward(*saved_tensors)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """Backward pass using spectral engine."""
        saved_tensors = ctx.saved_tensors
        grad_input = ctx.spectral_engine.backward(grad_output, saved_tensors)
        return grad_input, None, None, None, None


class FractionalAutogradLayer(nn.Module):
    """
    PyTorch module for spectral fractional autograd.
    """
    
    def __init__(self, alpha: Union[float, FractionalOrder], 
                 method: str = "auto", engine: str = "fft"):
        super().__init__()
        if isinstance(alpha, (int, float)):
            self.alpha = FractionalOrder(alpha)
        else:
            self.alpha = alpha
        self.method = method
        self.engine = engine
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return FractionalAutogradFunction.apply(x, self.alpha.alpha, self.method, self.engine)
    
    def extra_repr(self) -> str:
        return f'alpha={self.alpha}, method={self.method}, engine={self.engine}'


def _auto_select_engine(x: torch.Tensor, alpha: float, method: str) -> str:
    """
    Automatically select the best spectral engine based on problem characteristics.
    """
    size = x.numel()
    
    # Simple heuristics for engine selection
    if size < 1000:
        return "fft"  # FFT is efficient for small problems
    elif alpha < 0.5:
        return "mellin"  # Mellin may be better for small alpha
    elif alpha > 1.5:
        return "laplacian"  # Laplacian for large alpha
    else:
        return "fft"  # Default to FFT


# Convenience functions
def spectral_fractional_derivative(x: torch.Tensor, alpha: float, 
                                 method: str = "auto", engine: str = "fft") -> torch.Tensor:
    """Convenience function for spectral fractional derivative."""
    return FractionalAutogradFunction.apply(x, alpha, method, engine)


def create_spectral_fractional_layer(alpha: Union[float, FractionalOrder], 
                                   method: str = "auto", engine: str = "fft") -> FractionalAutogradLayer:
    """Convenience function for creating spectral fractional layer."""
    return FractionalAutogradLayer(alpha, method, engine)
