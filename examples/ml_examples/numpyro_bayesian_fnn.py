"""
Bayesian Fractional Neural Network with NumPyro
==============================================

This example demonstrates how to build a Bayesian Fractional Neural Network
using the NumPyro backend. We define a simple model where the fractional
order `alpha` is a random variable, and then use the No-U-Turn Sampler (NUTS)
to sample from the posterior distribution of `alpha`.
"""

import torch
import torch.nn as nn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from hpfracc.ml.layers import FractionalLinear
from hpfracc.ml.probabilistic_fractional_orders import ProbabilisticFractionalOrder

class BayesianFNN(nn.Module):
    """
    A simple Bayesian Fractional Neural Network.
    """
    def __init__(self, alpha_prior):
        super().__init__()
        self.alpha_prior = alpha_prior
        self.fc1 = nn.Linear(1, 10)
        self.frac_fc = FractionalLinear(10, 10, alpha=0.5)  # Placeholder alpha
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x, alpha):
        self.frac_fc.alpha = alpha
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.frac_fc(x))
        x = self.fc2(x)
        return x

def numpyro_model(X, y=None):
    """
    Defines the NumPyro model.
    """
    alpha_prior = dist.Uniform(0, 1)
    alpha = numpyro.sample("alpha", alpha_prior)
    
    # Define the neural network within the model
    # Note: This is a simplified example. In a real application, you would
    # need to handle the conversion between PyTorch and JAX tensors.
    # For simplicity, we will re-implement a simple JAX FNN here.
    
    key = numpyro.prng_key()
    w1_key, b1_key, w2_key, b2_key, w3_key, b3_key = random.split(key, 6)

    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((1, 10)), jnp.ones((1, 10))))
    b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(10), jnp.ones(10)))
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((10, 10)), jnp.ones((10, 10))))
    b2 = numpyro.sample("b2", dist.Normal(jnp.zeros(10), jnp.ones(10)))
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((10, 1)), jnp.ones((10, 1))))
    b3 = numpyro.sample("b3", dist.Normal(jnp.zeros(1), jnp.ones(1)))

    def forward(x, alpha, w1, b1, w2, b2, w3, b3):
        x = jax.nn.relu(x @ w1 + b1)
        # Apply fractional derivative (simplified for this example)
        x = x**alpha
        x = jax.nn.relu(x @ w2 + b2)
        x = x @ w3 + b3
        return x.squeeze()

    mu = forward(X, alpha, w1, b1, w2, b2, w3, b3)
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

def main():
    # Generate some synthetic data
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(X).squeeze() + 0.1 * np.random.randn(100)
    
    # Convert to JAX arrays
    X_jax = jax.device_put(X)
    y_jax = jax.device_put(y)

    # Run MCMC
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
    mcmc.run(random.PRNGKey(0), X_jax, y_jax)

    # Print results
    mcmc.print_summary()
    samples = mcmc.get_samples()
    alpha_samples = samples["alpha"]
    
    print("\nPosterior distribution of alpha:")
    print(f"  Mean: {alpha_samples.mean():.3f}")
    print(f"  Std: {alpha_samples.std():.3f}")
    print(f"  90% CI: ({np.percentile(alpha_samples, 5):.3f}, {np.percentile(alpha_samples, 95):.3f})")

if __name__ == "__main__":
    main()
