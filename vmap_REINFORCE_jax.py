import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import distrax
import optax
from flax.training.train_state import TrainState
from functools import partial

import gymnax

class mlp(nn.Module):
	action_dim: int
	layer_num: int
	layer_size: int

	@nn.compact
	def __call__(self, inputs):
		x = inputs 
		x = nn.relu(nn.Dense(self.layer_size)(x))
		for i in range(self.layer_num):
			x = nn.relu(nn.Dense(self.layer_size)(x))
		x = nn.Dense(self.action_dim)(x)
		return distrax.Categorical(logits=x)

def init_train_state(rng, action_dim, obs_shape, layer_num, layer_size):

	model = mlp(action_dim, layer_num, layer_size)
	key, rng = random.split(rng)
	model_params = model.init(key, jnp.zeros(obs_shape))

	schedule_fn = optax.linear_schedule(
        init_value=0.001,
        end_value=0.0005,
        transition_steps=3000
    )

	tx = optax.adam(schedule_fn)

	# not necessary but makes it easier and prettier
	# seems like the opt_state is built-in to the wrapper by default
	train_state = TrainState.create(
		apply_fn=model.apply,
		params=model_params,
		tx=tx
	)

	return train_state

@jax.jit
def discounted_returns(rewards, done):
	# quite proud of this one, came up with it all by myself
	def cumsum_with_discount(carry, xs):
		new_total_discount =  xs[1] * (0.99 * carry) + xs[0]
		return new_total_discount, new_total_discount 

	carry, returns = jax.lax.scan(cumsum_with_discount, 
									0, 
									[rewards,  (~done).astype(jnp.float32)], 
									reverse=True)
	return returns

@jax.jit
def f_loss(model_params, obs, actions, rewards, done):

	pi = train_state.apply_fn(model_params, obs)
	log_probs = pi.log_prob(actions)
	# CHANGED HERE TO ADD VMAP - in order to do discounted_returns on batches
	returns = jax.vmap(discounted_returns, in_axes=1)(rewards, done)
	loss = -jnp.mean(jnp.sum(log_probs * returns.T, axis=0))
	return loss

@jax.jit
def step(train_state: TrainState, obs, actions, rewards, done):
	loss_value, grads = jax.value_and_grad(f_loss,allow_int=True)(
		train_state.params, obs, actions, rewards, done
	)
	train_state = train_state.apply_gradients(grads=grads)
	return loss_value, train_state

@partial(jax.jit, static_argnums=[3,4])
def rollout(rng_input, train_state, env_params, steps_in_episode, num_p=32):
    """Rollout a jitted gymnax episode with lax.scan."""
    # from the gymnax getting started guide, with some changes
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    reset_rng = jax.random.split(rng_reset, num_p)
    # CHANGED HERE TO ADD VMAP - batch the environment reset
    obs, state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, train_state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        logit = train_state.apply_fn(train_state.params, obs)
        action = logit.sample(seed=rng_net)
        rng_step = jax.random.split(rng_step, num_p)
        # CHANGED HERE TO ADD VMAP - batch the environment step, returns batched data
        next_obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, train_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, train_state, rng_episode],
        (),
        steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done

def train_model(rng, train_state, env_params, EPOCHS, steps_per_episode=500):

	reported_reward, avg_num_Tries = 0, 0 

	reporting_interval = 100

	for e in range(EPOCHS + 1):

		key, rng = random.split(rng)
		obs, actions, rewards, next_obs, done = rollout(
			key, train_state, env_params, steps_per_episode, num_p=64
		)
		loss_value, train_state = step(train_state, obs, actions, rewards, done)

		reported_reward += jnp.sum(rewards) / jnp.sum(done.astype(jnp.float32)) 

		if e % reporting_interval == 0: 
			print(f"Epoch: {e}, Average reward = {reported_reward / reporting_interval}")
			reported_reward, avg_num_Tries = 0, 0 

rng = random.PRNGKey(491495711086)
key, rng = random.split(rng)

env, env_params = gymnax.make("CartPole-v1")

num_actions = env.num_actions
num_obs = env.observation_space(env_params).shape

train_state = init_train_state(key, action_dim=num_actions, obs_shape=num_obs, layer_num=1, layer_size=16)

train_model(rng, train_state, env_params, EPOCHS=1000, steps_per_episode=500)


