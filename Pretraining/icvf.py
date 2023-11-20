import jax
import jax.numpy as jnp
import optax

import equinox.nn as eqxnn
from typing import *

import functools
import equinox as eqx
import dataclasses

class TrainStateEQX(eqx.Module):
    model: eqx.Module
    optim: optax.GradientTransformation
    optim_state: optax.OptState

    @classmethod
    def create(cls, *, model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state,
                   **kwargs)
    
    @eqx.filter_jit
    def apply_updates(self, grads):
        updates, new_optim_state = self.optim.update(grads, self.optim_state, self.model)
        new_model = eqx.apply_updates(self.model, updates)
        return dataclasses.replace(
            self,
            model=new_model,
            optim_state=new_optim_state
        )

class TrainTargetStateEQX(TrainStateEQX):
    target_model: eqx.Module

    @classmethod
    def create(cls, *, model, target_model, optim, **kwargs):
        optim_state = optim.init(eqx.filter(model, eqx.is_array))
        return cls(model=model, optim=optim, optim_state=optim_state, target_model=target_model,
                   **kwargs)

    def soft_update(self, tau: float = 0.005):
        model_params = eqx.filter(self.model, eqx.is_array)
        target_model_params, target_model_static = eqx.partition(self.target_model, eqx.is_array)

        new_target_params = optax.incremental_update(model_params, target_model_params, tau)
        return dataclasses.replace(
            self,
            model=self.model,
            target_model=eqx.combine(new_target_params, target_model_static)
        )
        
        
def expectile_loss(adv, diff, expectile=0.8):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * diff ** 2
    
def icvf_loss(value_fn, target_value_fn, batch, config):
    if config['no_intent']:
        batch['icvf_desired_goals'] = jax.tree_map(jnp.ones_like, batch['icvf_desired_goals'])
    ###
    # Compute TD error for outcome s_+
    # 1(s == s_+) + V(s', s_+, z) - V(s, s_+, z)
    ###

    (next_v1_gz, next_v2_gz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['icvf_goals'], batch['icvf_desired_goals'])
    q1_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v1_gz
    q2_gz = batch['icvf_rewards'] + config['discount'] * batch['icvf_masks'] * next_v2_gz
    q1_gz, q2_gz = jax.lax.stop_gradient(q1_gz), jax.lax.stop_gradient(q2_gz)

    (v1_gz, v2_gz) = eval_ensemble(value_fn, batch['observations'], batch['icvf_goals'], batch['icvf_desired_goals'])

    ###
    # Compute the advantage of s -> s' under z
    # r(s, z) + V(s', z, z) - V(s, z, z)
    ###
    (next_v1_zz, next_v2_zz) = eval_ensemble(target_value_fn, batch['next_observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    if config['min_q']:
        next_v_zz = jnp.minimum(next_v1_zz, next_v2_zz)
    else:
        next_v_zz = (next_v1_zz + next_v2_zz) / 2.
    
    q_zz = batch['icvf_desired_rewards'] + config['discount'] * batch['icvf_desired_masks'] * next_v_zz
    (v1_zz, v2_zz) = eval_ensemble(target_value_fn, batch['observations'], batch['icvf_desired_goals'], batch['icvf_desired_goals'])
    v_zz = (v1_zz + v2_zz) / 2.
    
    adv = q_zz - v_zz
        
    value_loss1 = expectile_loss(adv, q1_gz-v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz-v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2

    def masked_mean(x, mask):
        return (x * mask).sum() / (1e-5 + mask.sum())

    advantage = adv
    return value_loss, {
        'value_loss': value_loss,
        'v_gz max': v1_gz.max(),
        'v_gz min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        'abs adv mean': jnp.abs(advantage).mean(),
        'adv mean': advantage.mean(),
        'adv max': advantage.max(),
        'adv min': advantage.min(),
        'accept prob': (advantage >= 0).mean(),
        'reward mean': batch['rewards'].mean(),
        'mask mean': batch['masks'].mean(),
        'q_gz max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz-v1_gz)**2, batch['masks']), # Loss on s \neq s_+
        'value_loss2': masked_mean((q1_gz-v1_gz)**2, 1.0 - batch['masks']), # Loss on s = s_+
    }

class ICVF_EQX_Agent(eqx.Module):
    value_learner: TrainTargetStateEQX
    config: dict
 
@eqx.filter_vmap(in_axes=dict(ensemble=eqx.if_array(0), s=None, g=None, z=None), out_axes=0)
def eval_ensemble(ensemble, s, g, z):
    return eqx.filter_vmap(ensemble)(s, g, z)

@eqx.filter_jit
def update(agent, batch):
    (val, value_aux), v_grads = eqx.filter_value_and_grad(icvf_loss, has_aux=True)(agent.value_learner.model, agent.value_learner.target_model, batch, agent.config)
    
    updated_v_learner = agent.value_learner.apply_updates(v_grads).soft_update()
    return dataclasses.replace(agent, value_learner=updated_v_learner, config=agent.config), value_aux

class MultilinearVF_EQX(eqx.Module):
    phi_net: eqx.Module
    psi_net: eqx.Module
    T_net: eqx.Module
    matrix_a: eqx.Module
    matrix_b: eqx.Module
    
    def __init__(self, key, state_dim, hidden_dims, pretrained_phi=None):
        key, phi_key, psi_key, t_key, matrix_a_key, matrix_b_key = jax.random.split(key, 6)
        
        network_cls = functools.partial(eqxnn.MLP, in_size=state_dim, out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu)
            
        if pretrained_phi is None:
            self.phi_net = network_cls(key=phi_key)
        else:
            self.phi_net = pretrained_phi
            
        self.psi_net = network_cls(key=psi_key)
        self.T_net = eqxnn.MLP(in_size=hidden_dims[-1], out_size=hidden_dims[-1], width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu, key=t_key)
        self.matrix_a = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_a_key)
        self.matrix_b = eqxnn.Linear(in_features=hidden_dims[-1], out_features=hidden_dims[-1], key=matrix_b_key)
        
    def __call__(self, observations, outcomes, intents):
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)
        
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        v = (phi_z * psi_z).sum(axis=-1)
        return v


def create_eqx_learner(seed: int,
                       observations: jnp.array,
                       hidden_dims: list,
                       optim_kwargs: dict = {
                            'learning_rate': 0.00005,
                            'eps': 0.0003125
                        },
                        load_pretrained_phi: bool=False,
                        discount: float = 0.99,
                        target_update_rate: float = 0.005,
                        expectile: float = 0.9,
                        no_intent: bool = False,
                        min_q: bool = True,
                        periodic_target_update: bool = False,
                        **kwargs):
    
        rng = jax.random.PRNGKey(seed)
        
        if load_pretrained_phi:
            network_cls = functools.partial(eqxnn.MLP, in_size=observations.shape[-1], out_size=hidden_dims[-1],
                                        width_size=hidden_dims[0], depth=len(hidden_dims),
                                        final_activation=jax.nn.relu)
            phi_net = network_cls(key=rng)
            loaded_phi_net = eqx.tree_deserialise_leaves("/home/m_bobrin/GOTIL/icvf_phi_300k_umaze.eqx", phi_net)
        else:
            loaded_phi_net = None
            
        @eqx.filter_vmap
        def ensemblize(keys):
            return MultilinearVF_EQX(key=keys, state_dim=observations.shape[-1], hidden_dims=hidden_dims,
                                     pretrained_phi=loaded_phi_net)
        
        value_learner = TrainTargetStateEQX.create(
            model=ensemblize(jax.random.split(rng, 2)),
            target_model=ensemblize(jax.random.split(rng, 2)),
            optim=optax.adam(**optim_kwargs)
        )
        config = dict(
            discount=discount,
            target_update_rate=target_update_rate,
            expectile=expectile,
            no_intent=no_intent, 
            min_q=min_q,
            periodic_target_update=periodic_target_update,
        )
        return ICVF_EQX_Agent(value_learner=value_learner, config=config)
