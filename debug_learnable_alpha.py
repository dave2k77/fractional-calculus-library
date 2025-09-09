"""
Debug learnable α issue
"""

import torch
import torch.nn as nn
import numpy as np


class BoundedAlphaParameter(nn.Parameter):
    """Debug bounded α parameter."""
    
    def __init__(self, alpha_init: float = 0.5, alpha_min: float = 0.01, alpha_max: float = 1.99):
        # Transform to unbounded space
        rho_init = torch.logit((alpha_init - alpha_min) / (alpha_max - alpha_min))
        super().__init__(rho_init)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
    
    def get_alpha(self) -> torch.Tensor:
        """Get bounded α value."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self)


def debug_learnable_alpha():
    """Debug the learnable α issue."""
    print("Debugging Learnable α...")
    
    # Create parameter
    alpha_param = BoundedAlphaParameter(alpha_init=0.3)
    print(f"Initial α: {alpha_param.get_alpha()}")
    print(f"α type: {type(alpha_param.get_alpha())}")
    print(f"α requires grad: {alpha_param.get_alpha().requires_grad}")
    
    # Test forward pass
    x = torch.linspace(0, 2*np.pi, 32, dtype=torch.float32)
    f = torch.sin(x)
    
    # Get alpha value
    alpha = alpha_param.get_alpha()
    print(f"Alpha value: {alpha}")
    print(f"Alpha type: {type(alpha)}")
    
    # Test if alpha is a tensor
    if isinstance(alpha, torch.Tensor):
        print("Alpha is a tensor - good!")
        print(f"Alpha shape: {alpha.shape}")
        print(f"Alpha device: {alpha.device}")
        print(f"Alpha dtype: {alpha.dtype}")
    else:
        print("Alpha is not a tensor - this is the problem!")
        print(f"Alpha value: {alpha}")
        print(f"Alpha type: {type(alpha)}")
    
    # Test gradient flow
    try:
        # Create a simple function that uses alpha
        result = f * alpha
        loss = result.sum()
        loss.backward()
        print(f"Gradient: {alpha_param.grad}")
    except Exception as e:
        print(f"Error in gradient flow: {e}")
    
    return True


if __name__ == "__main__":
    debug_learnable_alpha()
