"""
Minimal training loop using spectral, stochastic, and probabilistic fractional modules.
"""
import torch
import torch.nn as nn
import numpy as np

from hpfracc.ml import (
    SpectralFractionalLayer,
    StochasticFractionalLayer,
    create_normal_alpha_layer,
)

# Check if NumPyro is available for probabilistic layers
try:
    from hpfracc.ml.probabilistic_fractional_orders import NUMPYRO_AVAILABLE
    NUMPYRO_AVAILABLE = NUMPYRO_AVAILABLE
except (ImportError, AttributeError):
    NUMPYRO_AVAILABLE = False

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frac_spec = SpectralFractionalLayer(input_size=128, alpha_init=0.6)
        self.frac_stoch = StochasticFractionalLayer(alpha=0.6, k=32, method="importance")
        
        # Use probabilistic layer if NumPyro is available, otherwise use a simple spectral layer
        if NUMPYRO_AVAILABLE:
            try:
                self.frac_prob = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=False)
            except (ImportError, RuntimeError):
                # Fallback to spectral layer if probabilistic layer fails
                self.frac_prob = SpectralFractionalLayer(input_size=128, alpha_init=0.5)
        else:
            # Use spectral layer as fallback when NumPyro is not available
            self.frac_prob = SpectralFractionalLayer(input_size=128, alpha_init=0.5)
        
        # Three fractional features -> linear expects 3 inputs
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        # x: [B, N]
        y1 = self.frac_spec(x)      # [B, N]
        y2 = self.frac_stoch(x)     # [B, N]
        y3 = self.frac_prob(x)      # [B, N]
        
        # Reduce all outputs to [B, 1]
        y1 = y1.mean(dim=1, keepdim=True)
        y2 = y2.mean(dim=1, keepdim=True)
        y3 = y3.mean(dim=1, keepdim=True)
        
        feats = torch.cat([y1, y2, y3], dim=-1)  # [B, 3]
        return self.linear(feats)


def main():
    torch.manual_seed(0)
    # Create synthetic data
    t = torch.linspace(0, 2*np.pi, 128)
    x = torch.sin(t).unsqueeze(0)  # [1, N]
    y = (0.3 * torch.cos(t) + 0.1).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]

    net = TinyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    net.train()
    for step in range(10):
        opt.zero_grad()
        out = net(x)
        # Broadcast out over N to match target shape
        out_broadcast = out.unsqueeze(1).expand(-1, x.shape[1], -1)
        loss = criterion(out_broadcast, y)
        loss.backward()
        opt.step()
        if step % 2 == 0:
            print(f"step {step} loss {loss.item():.6f}")

    print("Training completed.")

if __name__ == "__main__":
    main()
