import rootutils
import logging

ROOT = rootutils.setup_root(search_from=__file__, indicator=[".git"], pythonpath=True, cwd=True)
logger = logging.getLogger(__name__)

import hydra    
import gymnax
import equinox as eqx

import jax
import optax

from networks.nets import CategoricalPolicy
from tqdm.auto import tqdm

import wandb
import functools
from omegaconf import OmegaConf, DictConfig
import logging

# Some custom utilities & Networks
from networks.base_eqx import TrainState
from utils.wandb_logger import setup_wandb
from utils.viz import TqdmExtraFormat

@functools.partial(eqx.filter_pmap, in_axes=(None, 0, 0))
def per_epoch_update(key, scan_out, train_state):
    obs, action, reward, next_obs, done = scan_out
    
    def discounted_returns(reward, done): # take each episode
        def cumsum(carry, xs):
            total_discount = xs[0] + xs[1] * (0.99 * carry)
            return total_discount, total_discount
        
        carry, returns = jax.lax.scan(cumsum, 0, [reward, (~done).astype(jax.numpy.float32)], reverse=True)    
        return returns
    
    def f_loss(model):
        distr = eqx.filter_vmap(eqx.filter_vmap(model))(obs)
        log_probs = distr.log_prob(action)
        returns = jax.vmap(discounted_returns, in_axes=1)(reward, done)
        loss = -jax.numpy.mean(jax.numpy.sum(log_probs * returns.T, axis=0))
        return loss
    
    val_loss, grads = eqx.filter_value_and_grad(f_loss, has_aux=False)(train_state.model)
    train_state = train_state.apply_updates(grads)
    return train_state, val_loss

@functools.partial(eqx.filter_pmap, in_axes=(0, None, 0, None, None))
def rollout(key, env, model, episode_steps, cfg):
    env, env_params = env
    key, sample_key, step_key, reset_key = jax.random.split(key, 4)
    
    vmap_step_keys = jax.random.split(step_key, cfg.num_environments)
    vmap_reset_keys = jax.random.split(reset_key, cfg.num_environments)

    def policy_step(model, input, keys):
        return model(input).sample(seed=keys)
    
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0))
    vmap_action = eqx.filter_vmap(policy_step, in_axes=(None, 0, 0))
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(vmap_reset_keys, env_params)
    init_policy_params, init_policy_static = eqx.partition(model, eqx.is_array)
    
    def rollout(carry, xs):
        policy_params, obs, state, sample_key = carry
        sample_key, old_sample_key = jax.random.split(sample_key, 2)
        vmap_sample_keys = jax.random.split(sample_key, cfg.num_environments)
        action = vmap_action(eqx.combine(policy_params, init_policy_static), obs, vmap_sample_keys)
        next_obs, n_state, reward, done, _ = vmap_step(vmap_step_keys, state, action)
        carry = policy_params, next_obs, n_state, sample_key
        
        return carry, [obs, action, reward, next_obs, done]
    
    init_carry = (init_policy_params, obs, state, sample_key)
    _, scan_out = jax.lax.scan(rollout, init_carry, None, episode_steps)
    
    return scan_out


def train(env: tuple, train_state: TrainState, cfg: DictConfig, rng: jax.Array):
    epochs, episode_steps = cfg.epochs, cfg.episode_steps
    
    for epoch in tqdm(range(epochs), leave=True, desc="Epoch"):
        rng, old_key = jax.random.split(rng, 2)
        key = jax.random.split(rng, 4)
        scan_out = rollout(key, env, train_state.model, episode_steps, cfg) # note that shape is (NUM_DEVICES, NUM_EPISODES, NUM_ENV)
        train_state, info = per_epoch_update(key, scan_out, train_state)
        reward_info = ((scan_out[2].sum(axis=1)) / scan_out[-1].sum(axis=1)).mean(-1).mean()
        
        wandb.log({f"Reward across {cfg.num_environments} envs": reward_info})

        if epoch == cfg.epochs-1:            
            import gym
            from gym.utils import save_video
            env = gym.make(cfg.env.name, render_mode="rgb_array")
            observation, info = env.reset()
            done = False
            frames = []
            single_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, train_state.model)
            
            for i in range(500):
                rng, sample_key = jax.random.split(rng, 2)
                action = single_model(observation).sample(seed=rng)
                observation, reward, terminated, truncated, info = env.step(jax.device_get(action))
                frames.append(env.render())
                done = terminated or truncated
                if done:
                    observation, info = env.reset()
            save_video.save_video(frames, video_folder="videos", fps=env.metadata["render_fps"])
            
@hydra.main(version_base="1.4", config_path=str(ROOT) + "/OnlineRL/configs", config_name="base.yaml")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    setup_wandb(config=cfg)
    
    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.random_key), 2)
    env = tuple(gymnax.make(cfg.env.name)) # returns env, env_params
    
    @eqx.filter_vmap
    def ensemblize_model(keys):
        return TrainState.create(model=CategoricalPolicy(obs_dim=4, action_dim=2, hidden_dims=[256, 256], key=keys),
                                 optim=optax.adam(learning_rate=3e-4))
    
    train_state = ensemblize_model(jax.random.split(key, cfg.num_devices))
    # model=hydra.utils.instantiate(cfg.algo.policy)
    # train_state = TrainState.create(model=ensemblize_model(jax.random.split(key, cfg.num_devices)),
    #                                 optim=optax.adam(learning_rate=3e-4))
    
    train(env, train_state, cfg, init_key)
        
    
if __name__ == "__main__":
    main()