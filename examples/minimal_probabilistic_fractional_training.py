"""
Minimal training loop using spectral, stochastic, and probabilistic fractional modules.
"""
import torch
import torch.nn as nn
import numpy as np

from hpfracc.ml import (
    FractionalAutogradLayer,
    StochasticFractionalLayer,
    create_normal_alpha_layer,
)

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frac_spec = FractionalAutogradLayer(alpha=0.6, engine="fft")
        self.frac_stoch = StochasticFractionalLayer(alpha=0.6, k=32, method="importance")
        self.frac_prob = create_normal_alpha_layer(mean=0.5, std=0.1, learnable=False)
        # Three fractional features -> linear expects 3 inputs
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        # x: [B, N]
        y1 = self.frac_spec(x)      # [B, N] or scalar
        y2 = self.frac_stoch(x)     # scalar
        y3 = self.frac_prob(x)      # scalar
        
        # Ensure all outputs are [B, 1]
        if y1.dim() == 2:
            y1 = y1.mean(dim=-1, keepdim=True)
        elif y1.dim() == 1:
            y1 = y1.unsqueeze(-1)
        elif y1.dim() == 0:
            y1 = y1.unsqueeze(0).unsqueeze(-1)
        
        if y2.dim() == 1:
            y2 = y2.unsqueeze(-1)
        elif y2.dim() == 0:
            y2 = y2.unsqueeze(0).unsqueeze(-1)
            
        if y3.dim() == 1:
            y3 = y3.unsqueeze(-1)
        elif y3.dim() == 0:
            y3 = y3.unsqueeze(0).unsqueeze(-1)
        
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
