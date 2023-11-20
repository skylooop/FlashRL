import equinox as eqx
import equinox.nn as eqxnn
import jax
import distrax

from jaxtyping import *
from typing import *

class GaussianPolicy(eqx.Module):
    net: eqx.Module
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims:list, key: int):
        net_key, key = jax.random.split(jax.random.PRNGKey(key), 2)
        
        self.net = eqxnn.MLP(in_size=obs_dim, out_size=action_dim * 2, width_size=hidden_dims[0],
                             depth=len(hidden_dims), activation=jax.nn.gelu, key=net_key)
    
    def __call__(self, obs):
        mean, log_std = jax.numpy.split(self.net(obs), 2, axis=-1)
        log_stds = jax.numpy.clip(log_std, self.log_std_min, self.log_std_max)
        dist = FixedDistrax(distrax.MultivariateNormalDiag, loc=mean,
                            scale_diag=jax.numpy.exp(log_stds))
        return dist

class CategoricalPolicy(eqx.Module):
    net: eqx.Module
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims:list, key: jax.Array):
        net_key, key = jax.random.split(key, 2)
        
        self.net = eqxnn.MLP(in_size=obs_dim, out_size=action_dim, width_size=hidden_dims[0],
                             depth=len(hidden_dims), activation=jax.nn.gelu, key=net_key)
    
    def __call__(self, obs):
        x = self.net(obs)
        dist = FixedDistrax(distrax.Categorical, logits=x)
        return dist

class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def sample_and_log_prob(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(seed=seed)

    def sample(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample(seed=seed)

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def probs(self):
        return self.cls(*self.args, **self.kwargs).probs
    
    def mean(self):
        return self.cls(*self.args, **self.kwargs).mean()