from abc import ABC, abstractmethod
from typing import Callable

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import trange

from mbrl.misc import NeuralNetDynamicsModel, NeuralNetPolicy
from mbrl._src.utils import Array


class DeepModelBasedAgent(ABC):
    def __init__(
        self,
        env: MujocoEnv,
        dynamics_model: NeuralNetDynamicsModel,
        ensemble_size: int,
        dynamics_optimizer: optax.GradientTransformation,
        plan_horizon: int,
        n_particles: int,
        obs_reward_fn: Callable[[jnp.ndarray], jnp.ndarray],
        action_reward_fn: Callable[[jnp.ndarray], jnp.ndarray],
        rng_key: jax.random.KeyArray,
        *_args, **_kwargs
    ):
        """Creates an RL agent which uses a neural network dynamics model to solve tasks via planning.

        Args:
            env: Environment within which agent will be acting, used for inferring shapes.
            dynamics_model: Dynamics model to be used by the agent.
            ensemble_size: Number of models to train for ensemble.
            dynamics_optimizer: Optimizer to use for training the dynamics model.
            plan_horizon: Planning horizon to use.
            n_particles: Number of independent particles to use for evaluating each action sequence.
            obs_reward_fn: Reward function defined on observations.
            action_reward_fn: Reward function defined on actions.
            rng_key: JAX RNG key to be used by this agent internally. Do not reuse.
        """
        self._dynamics_model = dynamics_model
        self._ensemble_size = ensemble_size
        self._dynamics_optimizer = dynamics_optimizer
        self._plan_horizon = plan_horizon
        self._n_particles = n_particles
        self._obs_reward_fn = obs_reward_fn
        self._action_reward_fn = action_reward_fn
        self._rng_key = rng_key

        self._dynamics_dataset = {
            "observations": jnp.zeros([0, env.observation_space.shape[0]]),
            "actions": jnp.zeros([0, env.action_space.shape[0]]),
            "next_observations": jnp.zeros([0, env.observation_space.shape[0]])
        }

        params_per_member = [{} for _ in range(self._ensemble_size)]
        for idx in range(self._ensemble_size):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            params_per_member[idx] = self._dynamics_model.init(params_per_member[idx], subkey)
        self._dynamics_params = jax.tree_map(lambda *a: jnp.stack(a), *params_per_member)

        self._dynamics_optimizer_state = self._dynamics_optimizer.init(self._dynamics_params)
        self._model_update_op = self._create_model_update_op()

    def reset(self) -> None:
        """Resets the agent after an environment rollout. Does nothing by default. """
        pass

    def add_interaction_data(
        self,
        obs_seq: Array,
        action_seq: Array
    ) -> None:
        """Provides the agent with new trajectory data for future training.
        This method expects that len(obs_seq) == len(action_seq) + 1 (the additional observation being
        the initial one).

        Args:
            obs_seq: Sequence of new observations to provide to the learner.
            action_seq: Sequence of new actions to provide to the learner.
        """
        if len(obs_seq) != len(action_seq) + 1:
            raise RuntimeError("Invalid trajectory data passed to agent. "
                               "Ensure that len(obs_seq) == len(action_seq) + 1.")

        self._dynamics_dataset["observations"] = jnp.concatenate([
            self._dynamics_dataset["observations"], obs_seq[:-1],
        ])
        self._dynamics_dataset["actions"] = jnp.concatenate([
            self._dynamics_dataset["actions"], action_seq,
        ])
        self._dynamics_dataset["next_observations"] = jnp.concatenate([
            self._dynamics_dataset["next_observations"], obs_seq[1:],
        ])

    def train(
        self,
        n_model_train_steps: int,
        model_train_batch_size: int,
        *_args, **_kwargs
    ) -> None:
        """Trains the internal dynamics model of this agent with all the provided interaction data so far.

        Args:
            n_model_train_steps: Number of parameter updates to perform for the model.
            model_train_batch_size: Size of batches to use for each parameter update for the model.
        """
        dataset_size = self._dynamics_dataset["observations"].shape[0]

        if self._ensemble_size > 1:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            bootstrap_subidxs = jax.random.randint(
                subkey, minval=0, maxval=dataset_size, shape=[self._ensemble_size, dataset_size]
            )
        else:
            bootstrap_subidxs = jnp.arange(dataset_size)[None]

        epoch_steps = 0
        for _ in trange(n_model_train_steps, desc="Dynamics model training", unit="batches", ncols=150):
            if model_train_batch_size * (epoch_steps + 1) > dataset_size:
                epoch_steps = 0     # Not enough points left for full batch, reset.
            if epoch_steps == 0:    # Start of new run through dataset, shuffle bootstrap datasets
                self._rng_key, subkey = jax.random.split(self._rng_key)
                bootstrap_subidxs = jax.random.permutation(
                    subkey, bootstrap_subidxs, axis=-1, independent=True
                )

            batch_start = epoch_steps * model_train_batch_size
            batch_end = batch_start + model_train_batch_size
            batch_idxs = bootstrap_subidxs[:, batch_start:batch_end]
            batches_per_member = {
                "observations": self._dynamics_dataset["observations"][batch_idxs],
                "actions": self._dynamics_dataset["actions"][batch_idxs],
                "next_observations": self._dynamics_dataset["next_observations"][batch_idxs]
            }
            self._dynamics_params, self._dynamics_optimizer_state = self._model_update_op(
                self._dynamics_params, self._dynamics_optimizer_state, batches_per_member
            )
            epoch_steps += 1

    @abstractmethod
    def act(
        self,
        obs: Array
    ) -> onp.ndarray:
        """Queries the agent for its action on the given observation.

        Args:
            obs: The current observation.

        Returns:
            Action chosen by the agent
        """
        pass

    def _create_model_update_op(self):
        @jax.jit
        def model_update(dynamics_params, dynamics_optimizer_state, all_member_batches):
            def batch_mean_loss(single_net_params, batch):
                return jnp.mean(jax.vmap(self._dynamics_model.prediction_loss, (None, 0, 0, 0))(
                    single_net_params, batch["observations"], batch["actions"], batch["next_observations"]
                ))

            def sum_ensemble_losses(ensemble_params, ensemble_batches):
                return jnp.sum(jax.vmap(batch_mean_loss)(ensemble_params, ensemble_batches))

            ensemble_grads = jax.grad(sum_ensemble_losses)(dynamics_params, all_member_batches)
            updates, dynamics_optimizer_state = self._dynamics_optimizer.update(
                ensemble_grads, dynamics_optimizer_state, dynamics_params
            )
            return optax.apply_updates(dynamics_params, updates), dynamics_optimizer_state

        return model_update

    def _create_rollout_evaluator(
        self,
        rollout_policy,
        rollout_horizon
    ):
        """Helper method that creates evaluators using the model to roll out policies.
        Created so that the rollout_policy parameters are JAX vmap-able.

        Args:
            rollout_policy: A policy mapping of the form (params, obs, i) -> action.
                See examples below.
            rollout_horizon: Rollout horizon

        Returns:
            Policy evaluator.

        >>> # Evaluator for length-10 action sequences (e.g. MPC/random shooting methods).
        >>> evaluator_mpc = self._create_rollout_evaluator(lambda action_seq, _obs, i: action_seq[i], 10)
        >>>
        >>> # Evaluator for time-independent policy over a length-50 horizon
        >>> policy: NeuralNetPolicy
        >>> evaluator_policy = self._create_rollout_evaluator(
        ...     lambda policy_params, obs, _i: policy.act(policy_params, obs), 50
        ... )
        """
        def rollout_and_evaluate(rollout_policy_params, dynamics_params, start_obs, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            particle_to_member = jax.random.randint(
                subkey, minval=0, maxval=self._ensemble_size, shape=[self._n_particles]
            )
            params_per_particle = jax.tree_map(
                lambda all_params: all_params[particle_to_member], dynamics_params
            )

            start_obs = jnp.repeat(start_obs[None], self._n_particles, axis=0)
            running_return = jnp.zeros(self._n_particles)

            def simulate_single_timestep(i, args):
                cur_obs, cur_return, r_key = args
                actions = jax.vmap(rollout_policy, (None, 0, None))(rollout_policy_params, cur_obs, i)

                r_key, s_key = jax.random.split(r_key)
                predicted_next_obs = jax.vmap(self._dynamics_model.predict)(
                    params_per_particle, cur_obs, actions, jax.random.split(s_key, num=self._n_particles)
                )

                obs_reward = jax.vmap(self._obs_reward_fn)(predicted_next_obs)
                action_reward = jax.vmap(self._action_reward_fn)(actions)
                cur_return += obs_reward + action_reward

                return predicted_next_obs, cur_return, r_key

            _, rollout_returns, _ = jax.lax.fori_loop(
                0, rollout_horizon, simulate_single_timestep,
                (start_obs, running_return, rng_key)
            )
            return jnp.mean(rollout_returns)

        return rollout_and_evaluate
