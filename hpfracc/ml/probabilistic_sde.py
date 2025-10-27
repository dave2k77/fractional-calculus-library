"""
Probabilistic SDE Models with NumPyro

This module provides Bayesian neural fractional SDEs with uncertainty quantification
using NumPyro for probabilistic programming.
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
import warnings

try:
    import numpyro
    import numpyro.distributions as dist
    from jax import random as jax_random
    import jax.numpy as jnp
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.optim import Adam
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    numpyro = None
    dist = None
    jnp = None


def numpyro_fsde_model(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    fractional_order_prior=None,
    drift_prior=None,
    diffusion_prior=None
):
    """
    NumPyro model for Bayesian inference in Neural fSDEs.
    
    Args:
        X: Input data
        y: Target data (optional, for supervised learning)
        fractional_order_prior: Prior distribution for fractional order
        drift_prior: Prior distribution for drift parameters
        diffusion_prior: Prior distribution for diffusion parameters
        
    Returns:
        Pyro computation
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro is required for probabilistic SDE models")
    
    # Default priors
    if fractional_order_prior is None:
        fractional_order_prior = dist.Uniform(0.1, 1.9)
    
    if drift_prior is None:
        drift_prior = dist.Normal(0.0, 1.0)
    
    if diffusion_prior is None:
        diffusion_prior = dist.Exponential(1.0)
    
    # Sample fractional order
    alpha = numpyro.sample("fractional_order", fractional_order_prior)
    
    # Sample drift parameters
    drift_params = numpyro.sample("drift_params", drift_prior.expand([X.shape[-1]]))
    
    # Sample diffusion parameters
    diffusion_params = numpyro.sample("diffusion_params", diffusion_prior.expand([X.shape[-1]]))
    
    # Placeholder for SDE dynamics
    # In full implementation, would solve SDE here
    
    # Likelihood
    if y is not None:
        # Assume Gaussian likelihood
        prediction_scale = 0.1
        prediction = X @ drift_params  # Simplified
        numpyro.sample("obs", dist.Normal(prediction, prediction_scale), obs=y)
    
    return alpha, drift_params, diffusion_params


def numpyro_guide_fsde(X):
    """
    Guide (variational posterior) for Bayesian fSDE.
    
    Args:
        X: Input data
        
    Returns:
        Guide computation
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro is required")
    
    # Learnable parameters for variational distribution
    alpha_mean = numpyro.param("alpha_mean", 0.5, constraint=dist.constraints.interval(0.1, 1.9))
    alpha_std = numpyro.param("alpha_std", 0.1, constraint=dist.constraints.positive)
    
    # Sample from variational distribution
    numpyro.sample("fractional_order", dist.Normal(alpha_mean, alpha_std))
    
    # Drift and diffusion parameters
    drift_shape = (X.shape[-1],)
    diffusion_shape = (X.shape[-1],)
    
    drift_mean = numpyro.param("drift_mean", jnp.zeros(drift_shape))
    drift_std = numpyro.param("drift_std", jnp.ones(drift_shape), constraint=dist.constraints.positive)
    numpyro.sample("drift_params", dist.Normal(drift_mean, drift_std))
    
    diffusion_mean = numpyro.param("diffusion_mean", jnp.ones(diffusion_shape), constraint=dist.constraints.positive)
    diffusion_std = numpyro.param("diffusion_std", jnp.ones(diffusion_shape), constraint=dist.constraints.positive)
    numpyro.sample("diffusion_params", dist.Normal(diffusion_mean, diffusion_std))


class BayesianNeuralFractionalSDE:
    """
    Bayesian neural fractional SDE with NumPyro for uncertainty quantification.
    
    Provides:
    - Prior distributions over drift/diffusion parameters
    - Variational inference for parameter learning
    - Posterior predictive distributions for uncertainty quantification
    """
    
    def __init__(
        self,
        model_fn=numpyro_fsde_model,
        guide_fn=numpyro_guide_fsde,
        num_samples: int = 1000
    ):
        """
        Initialize Bayesian neural fSDE.
        
        Args:
            model_fn: NumPyro model function
            guide_fn: NumPyro guide (variational posterior) function
            num_samples: Number of posterior samples for inference
        """
        if not NUMPYRO_AVAILABLE:
            raise ImportError("NumPyro is required for Bayesian fSDEs")
        
        self.model_fn = model_fn
        self.guide_fn = guide_fn
        self.num_samples = num_samples
        self.svi = None
        self.svi_state = None
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000):
        """
        Fit the Bayesian model using variational inference.
        
        Args:
            X: Input features
            y: Target values
            num_epochs: Number of training epochs
        """
        # Setup SVI
        optimizer = Adam(step_size=1e-3)
        self.svi = SVI(self.model_fn, self.guide_fn, optimizer, loss=Trace_ELBO())
        
        # Initialize SVI state
        rng_key = jax_random.PRNGKey(0)
        self.svi_state = self.svi.init(rng_key, X, y)
        
        # Training loop
        for epoch in range(num_epochs):
            rng_key, rng_key_step = jax_random.split(rng_key)
            self.svi_state, loss = self.svi.update(self.svi_state, rng_key_step, X, y)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        
        self.trained = True
        print("Training complete")
    
    def predict(self, X: np.ndarray, num_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        
        Args:
            X: Input features
            num_samples: Number of samples from posterior (default: self.num_samples)
            
        Returns:
            Dictionary with predictions, uncertainty, etc.
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before prediction")
        
        if num_samples is None:
            num_samples = self.num_samples
        
        # Get posterior samples
        predictive = Predictive(self.model_fn, self.svi.sample(self.svi_state, num_samples), return_sites=["obs"])
        rng_key = jax_random.PRNGKey(1)
        samples = predictive(rng_key, X)
        
        # Compute statistics
        predictions = np.array(samples["obs"])
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'samples': predictions,
            'fractional_order': samples.get("fractional_order", None)
        }
    
    def get_parameter_posterior(self) -> Dict[str, Any]:
        """Get posterior distributions of parameters."""
        if not self.trained:
            raise RuntimeError("Model must be trained first")
        
        params = self.svi.get_params(self.svi_state)
        return params


def create_bayesian_fsde(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    num_epochs: int = 1000
) -> BayesianNeuralFractionalSDE:
    """
    Factory function to create and fit a Bayesian neural fractional SDE.
    
    Args:
        X: Input features
        y: Target values (optional)
        num_epochs: Number of training epochs
        
    Returns:
        Trained BayesianNeuralFractionalSDE instance
    """
    model = BayesianNeuralFractionalSDE()
    
    if y is not None:
        model.fit(X, y, num_epochs)
    
    return model
