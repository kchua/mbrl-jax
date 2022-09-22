from functools import partial
from typing import Callable, Dict, Optional

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import trange

from mbrl.agents import DeepModelBasedAgent
from mbrl.misc import Dataset, NeuralNetDynamicsModel, QFunction
from mbrl.policies import TanhGaussianPolicy
from mbrl._src.utils import Array


class ModelBasedPolicyOptimization(DeepModelBasedAgent):
    def __init__(
        self,
        env: MujocoEnv,
        reward_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        dynamics_model: NeuralNetDynamicsModel,
        ensemble_size: int,
        dynamics_optimizer: optax.GradientTransformation,
        n_model_train_steps: int,
        model_train_batch_size: int,
        rng_key: jax.random.KeyArray,
        actor: TanhGaussianPolicy,
        critic: QFunction,
        discount_factor: float,
        max_replay_buffer_length: int,
        include_augmentations: bool,
        augmentation_batch_size: bool,
        augmentation_keep_epochs: int,
        augmentation_horizon: int,
        real_ratio: float,
        initial_temperature: float,
        target_smoothing_coefficient: float,
        sac_batch_size: int,
        sac_optimizer: optax.GradientTransformation,
        n_steps_per_batch: int,
        target_entropy_multiplier: float = 1.0,
        target_entropy: Optional[float] = None
    ):
        """Creates an RL agent which uses a neural network model together with a policy.
        The policy is trained via policy gradients, using the model to predict returns.

        Args:
            env: Environment within which agent will be acting, used for inferring shapes.
            reward_fn: Reward function defined on (observation, action, next_observation).
            dynamics_model: Dynamics model to be used by the agent.
            ensemble_size: Number of models to train for ensemble.
            dynamics_optimizer: Optimizer to use for training the dynamics model.
            n_model_train_steps: Number of parameter updates to perform for the model.
            model_train_batch_size: Size of batches to use for each parameter update for the model.
            rng_key: JAX RNG key to be used by this agent internally. Do not reuse.
        """
        super().__init__(
            env,
            reward_fn,
            dynamics_model,
            ensemble_size,
            dynamics_optimizer,
            n_model_train_steps,
            model_train_batch_size,
            rng_key
        )

        self._actor = actor
        self._critic = critic

        self._discount_factor = discount_factor
        self._include_augmentations = include_augmentations
        self._augmentation_batch_size = augmentation_batch_size
        self._augmentation_keep_epochs = augmentation_keep_epochs
        self._augmentation_horizon = augmentation_horizon
        self._real_ratio = real_ratio if self._include_augmentations else 1.0
        self._initial_temperature = initial_temperature
        self._target_smoothing_coefficient = target_smoothing_coefficient
        self._sac_batch_size = sac_batch_size
        self._sac_optimizer = sac_optimizer
        self._n_steps_per_batch = n_steps_per_batch

        if target_entropy is None:
            self._target_entropy = target_entropy_multiplier * (-env.action_space.shape[0])
        else:
            self._target_entropy = target_entropy

        self._sac_dataset = Dataset(
            observation=env.observation_space.shape,
            action=env.action_space.shape,
            next_observation=env.observation_space.shape,
            max_length=max_replay_buffer_length
        )
        self._model_augmentation_dataset = Dataset(
            observation=env.observation_space.shape,
            action=env.action_space.shape,
            next_observation=env.observation_space.shape,
            max_length=augmentation_keep_epochs * augmentation_batch_size * augmentation_horizon
        )
        self._n_env_steps = 0

        self._current_params, self._state = {}, {}
        self._rng_key, subkey = jax.random.split(self._rng_key)
        self._current_params["actor"], self._state["actor"] = self._actor.init({}, {}, subkey)
        self._rng_key, subkey = jax.random.split(self._rng_key)
        critic_initializer = partial(self._critic.init, {}, {})
        self._current_params["critic"], self._state["critic"] = jax.vmap(critic_initializer)(jax.random.split(subkey))
        self._current_params["temperature"] = jnp.log(jnp.exp(initial_temperature) - 1)

        self._target_params = {
            "critic": jax.tree_map(lambda x: jnp.copy(x), self._current_params["critic"])
        }

        self._optimizer_states = {
            "actor": self._sac_optimizer.init(self._current_params["actor"]),
            "critic": self._sac_optimizer.init(self._current_params["critic"]),
            "temperature": self._sac_optimizer.init(self._current_params["temperature"])
        }

        self._ops = {
            "augment": jax.jit(self._augment_batch),
            "update": jax.jit(self._sac_update)
        }

    def add_interaction_data(
        self,
        obs_seq: onp.array,
        action_seq: onp.array
    ):
        super().add_interaction_data(obs_seq, action_seq)
        self._sac_dataset.add(
            observation=obs_seq[:-1],
            action=action_seq,
            next_observation=obs_seq[1:]
        )
        self._n_env_steps += len(action_seq)

    def get_logging_statistics(self, *_args, **_kwargs) -> Dict[str, float]:
        base_statistics = super().get_logging_statistics()

        base_statistics["Actor temperature"] = jax.nn.softplus(self._current_params["temperature"])
        base_statistics["Target entropy"] = self._target_entropy

        return base_statistics

    def act(
        self,
        obs: Array,
        evaluation: bool = False
    ):
        if evaluation:
            mean_action = self._actor.get_mean_action(self._current_params["actor"], self._state["actor"], obs)
            return onp.array(mean_action)
        else:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            action = self._actor.act(self._current_params["actor"], self._state["actor"], obs, subkey)
            return onp.array(action)

    def train(self):
        super().train()

        n_real_points_per_batch = int(self._real_ratio * self._sac_batch_size)
        n_augmentations_per_batch = self._sac_batch_size - n_real_points_per_batch

        if n_augmentations_per_batch > 0:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            augmentation_base = self._sac_dataset.sample(self._augmentation_batch_size, subkey)
            self._rng_key, subkey = jax.random.split(self._rng_key)
            new_augmentations = self._ops["augment"](
                self._current_params,
                self._state,
                self._dynamics_params,
                self._dynamics_state,
                augmentation_base,
                subkey
            )
            self._model_augmentation_dataset.add(**new_augmentations)

        real_data_subset, augmentations = None, None
        for _ in trange(self._n_env_steps * self._n_steps_per_batch, desc="MBPO training", unit="batches", ncols=150):
            if n_real_points_per_batch > 0:
                self._rng_key, subkey = jax.random.split(self._rng_key)
                real_data_subset = self._sac_dataset.sample(n_real_points_per_batch, subkey)

            if n_augmentations_per_batch > 0:
                self._rng_key, subkey = jax.random.split(self._rng_key)
                augmentations = self._model_augmentation_dataset.sample(n_augmentations_per_batch, subkey)

            if n_real_points_per_batch > 0 and n_augmentations_per_batch > 0:
                train_batch = jax.tree_map(
                    lambda x, y: jnp.concatenate([x, y[:n_augmentations_per_batch]]),
                    real_data_subset,
                    augmentations
                )
            elif n_real_points_per_batch == 0:
                train_batch = augmentations
            else:
                train_batch = real_data_subset

            assert len(train_batch["action"]) == self._sac_batch_size

            self._rng_key, subkey = jax.random.split(self._rng_key)
            self._current_params, self._target_params, self._optimizer_states = self._ops["update"](
                self._current_params,
                self._target_params,
                self._state,
                self._optimizer_states,
                train_batch,
                subkey
            )

        self._n_env_steps = 0

    def _augment_batch(
        self,
        current_params,
        state,
        dynamics_params,
        dynamics_state,
        batch_timesteps,
        rng_key
    ):
        def get_augmentation(start_obs, key):
            key, temp_key = jax.random.split(key)
            rollout = self._rollout(
                current_params["actor"], state["actor"], dynamics_params, dynamics_state, start_obs, temp_key
            )
            key, temp_key = jax.random.split(key)
            timestep = jax.random.randint(temp_key, (), 0, self._augmentation_horizon, dtype=int)

            return {
                "observation": rollout["observation_sequence"][timestep],
                "action": rollout["action_sequence"][timestep],
                "next_observation": rollout["observation_sequence"][timestep + 1]
            }

        rng_key, subkey = jax.random.split(rng_key)
        augmentations = jax.vmap(get_augmentation)(
            batch_timesteps["observation"], jax.random.split(subkey, num=len(batch_timesteps["action"]))
        )
        return augmentations

    def _sac_update(
        self,
        current_params,
        target_params,
        state,
        optimizer_states,
        batch_timesteps,
        rng_key
    ):
        def batch_mean_loss(fn, params, batch, key):
            per_timestep_loss_computation = partial(fn, params, target_params, state)
            batch_losses = jax.vmap(per_timestep_loss_computation)(
                batch, jax.random.split(key, num=len(batch["action"]))
            )
            return jnp.mean(batch_losses)

        batch_critic_loss = partial(batch_mean_loss, self._compute_critic_loss)
        batch_critic_grad = jax.grad(batch_critic_loss)(current_params, batch_timesteps, rng_key)["critic"]
        critic_updates, optimizer_states["critic"] = self._sac_optimizer.update(
            batch_critic_grad, optimizer_states["critic"], current_params["critic"]
        )
        current_params["critic"] = optax.apply_updates(current_params["critic"], critic_updates)

        batch_actor_loss = partial(batch_mean_loss, self._compute_actor_loss)
        batch_actor_grad = jax.grad(batch_actor_loss)(current_params, batch_timesteps, rng_key)["actor"]
        actor_updates, optimizer_states["actor"] = self._sac_optimizer.update(
            batch_actor_grad, optimizer_states["actor"], current_params["actor"]
        )
        current_params["actor"] = optax.apply_updates(current_params["actor"], actor_updates)

        batch_temperature_loss = partial(batch_mean_loss, self._compute_temperature_loss)
        batch_temperature_grad = \
            jax.grad(batch_temperature_loss)(current_params, batch_timesteps, rng_key)["temperature"]
        temperature_update, optimizer_states["temperature"] = self._sac_optimizer.update(
            batch_temperature_grad, optimizer_states["temperature"], current_params["temperature"]
        )
        current_params["temperature"] = optax.apply_updates(current_params["temperature"], temperature_update)

        target_params = {
            name: optax.incremental_update(
                current_params[name], target_params[name], self._target_smoothing_coefficient
            )
            for name in target_params
        }
        return current_params, target_params, optimizer_states

    def _compute_actor_loss(
        self,
        current_params,
        _target_params,
        state,
        timestep,
        rng_key
    ):
        rng_key, subkey = jax.random.split(rng_key)
        actor_action, actor_entropy = self._actor.sample_with_entropy_estimate(
            current_params["actor"], state["actor"], timestep["observation"], subkey
        )
        actor_value = self._aggregate_critic(
            jax.lax.stop_gradient(current_params["critic"]), state["critic"], timestep["observation"], actor_action
        )
        current_temperature = self._current_temperature(current_params)
        actor_return = actor_value + jax.lax.stop_gradient(current_temperature) * actor_entropy
        return -actor_return

    def _compute_critic_loss(
        self,
        current_params,
        target_params,
        state,
        timestep,
        rng_key
    ):
        timestep_reward = self._reward_fn(timestep["observation"], timestep["action"], timestep["next_observation"])
        current_temperature = self._current_temperature(current_params)
        predicted_q_values = jax.vmap(self._critic.value, (0, 0, None, None))(
            current_params["critic"], state["critic"], timestep["observation"], timestep["action"]
        )
        rng_key, subkey = jax.random.split(rng_key)
        next_step_action, next_step_entropy = self._actor.sample_with_entropy_estimate(
            current_params["actor"], state["actor"], timestep["next_observation"], subkey
        )
        terminal_value = self._aggregate_critic(
            target_params["critic"], state["critic"], timestep["next_observation"], next_step_action
        )
        target_q_value = timestep_reward + self._discount_factor * (
            terminal_value + current_temperature * next_step_entropy
        )
        return 0.5 * jnp.sum(jnp.square(predicted_q_values - jax.lax.stop_gradient(target_q_value)))

    def _compute_temperature_loss(
        self,
        current_params,
        _target_params,
        state,
        timestep,
        rng_key
    ):
        # Actor loss
        current_temperature = self._current_temperature(current_params)
        rng_key, subkey = jax.random.split(rng_key)
        actor_entropy = self._actor.entropy(
            current_params["actor"], state["actor"], timestep["observation"], subkey
        )
        delta_entropy = jax.lax.stop_gradient(actor_entropy - self._target_entropy)
        return current_temperature * delta_entropy

    def _rollout(
        self,
        actor_params,
        actor_state,
        dynamics_params,
        dynamics_state,
        start_obs,
        rng_key
    ):
        def rollout_policy(rollout_policy_params, obs, _timestep, key):
            return self._actor.act(rollout_policy_params["params"], rollout_policy_params["state"], obs, key)

        rollout_evaluator = self._create_rollout_evaluator(
            rollout_policy=rollout_policy,
            rollout_horizon=self._augmentation_horizon,
            fn_to_accumulate=lambda *args: 0.,
            accumulator_init=0.
        )
        return rollout_evaluator(
            {
                "params": actor_params,
                "state": actor_state,
            },
            dynamics_params,
            dynamics_state,
            start_obs,
            rng_key
        )

    def _current_temperature(self, current_params):
        return jax.nn.softplus(current_params["temperature"])

    def _aggregate_critic(self, critic_params, critic_state, observation, action):
        all_critic_predictions = jax.vmap(self._critic.value, (0, 0, None, None))(
            critic_params, critic_state, observation, action
        )
        return jnp.min(all_critic_predictions)
