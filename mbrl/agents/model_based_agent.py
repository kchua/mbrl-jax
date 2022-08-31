from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import trange

from mbrl.misc import Dataset, NeuralNetDynamicsModel
from mbrl.policies import DeterministicPolicy
from mbrl._src.utils import Array


class DeepModelBasedAgent(ABC):
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
        *_args, **_kwargs
    ):
        """Creates an RL agent which uses a neural network dynamics model to solve tasks via planning.

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
        self._reward_fn = reward_fn
        self._dynamics_model = dynamics_model
        self._ensemble_size = ensemble_size
        self._dynamics_optimizer = dynamics_optimizer
        self._n_model_train_steps = n_model_train_steps
        self._model_train_batch_size = model_train_batch_size
        self._rng_key = rng_key

        self._dynamics_dataset = Dataset(
            observation=env.observation_space.shape,
            action=env.action_space.shape,
            next_observation=env.observation_space.shape
        )

        self._rng_key, subkey = jax.random.split(self._rng_key)
        initializer = partial(self._dynamics_model.init, {}, {})
        self._dynamics_params, self._dynamics_state = \
            jax.vmap(initializer)(jax.random.split(subkey, num=self._ensemble_size))

        self._dynamics_optimizer_state = self._dynamics_optimizer.init(self._dynamics_params)
        self._model_update_op = jax.jit(self._update_model)

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

        self._dynamics_dataset.add(
            observation=obs_seq[:-1],
            action=action_seq,
            next_observation=obs_seq[1:]
        )

    def train(self) -> None:
        """Trains the internal dynamics model of this agent with all the provided interaction data so far.
        """
        self._rng_key, subkey = jax.random.split(self._rng_key)
        bootstrapped_dataset = self._dynamics_dataset.bootstrap(self._ensemble_size, subkey)

        self._dynamics_params, self._dynamics_state = jax.vmap(self._dynamics_model.fit_normalizer)(
            self._dynamics_params,
            self._dynamics_state,
            bootstrapped_dataset["observation"],
            bootstrapped_dataset["action"],
            bootstrapped_dataset["next_observation"]
        )

        self._rng_key, subkey = jax.random.split(self._rng_key)
        epoch_iterator = bootstrapped_dataset.epoch(self._model_train_batch_size, subkey)
        for _ in trange(self._n_model_train_steps, desc="Dynamics model training", unit="batches", ncols=150):
            while True:
                try:
                    batches_per_member = next(epoch_iterator)
                    break
                except StopIteration:
                    self._rng_key, subkey = jax.random.split(self._rng_key)
                    epoch_iterator = bootstrapped_dataset.epoch(self._model_train_batch_size, subkey)

            self._dynamics_params, self._dynamics_state, self._dynamics_optimizer_state = self._model_update_op(
                self._dynamics_params, self._dynamics_state, self._dynamics_optimizer_state, batches_per_member
            )

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

    def get_logging_statistics(self, *_args, **_kwargs) -> Dict[str, float]:
        """Get statistics for logging from this agent.

        Args:
            *_args: UNUSED
            **_kwargs: UNUSED

        Returns:
            Dictionary of named statistics to be logged.
        """
        self._rng_key, subkey = jax.random.split(self._rng_key)

        full_dataset_evaluations = jax.vmap(self._batch_mean_log_likelihood, (0, 0, None))(
            self._dynamics_params,
            self._dynamics_state,
            {name: self._dynamics_dataset[name] for name in self._dynamics_dataset}
        )

        return {
            "Dynamics model log-likelihood": jnp.mean(full_dataset_evaluations)
        }

    def _update_model(self, dynamics_params, dynamics_state, dynamics_optimizer_state, all_member_batches):
        def sum_ensemble_losses(ensemble_params, ensemble_state, ensemble_batches):
            return -jnp.sum(jax.vmap(self._batch_mean_log_likelihood)(
                ensemble_params, ensemble_state, ensemble_batches
            ))

        ensemble_grads = jax.grad(sum_ensemble_losses)(dynamics_params, dynamics_state, all_member_batches)
        updates, dynamics_optimizer_state = self._dynamics_optimizer.update(
            ensemble_grads, dynamics_optimizer_state, dynamics_params
        )
        updated_params = optax.apply_updates(dynamics_params, updates)
        return updated_params, dynamics_state, dynamics_optimizer_state

    def _create_rollout_evaluator(
        self,
        rollout_policy: Callable[[Dict, jnp.ndarray, int, jax.random.KeyArray], jnp.ndarray],
        rollout_horizon: int,
        fn_to_accumulate: Callable[[Dict, Dict, Dict, Dict, jax.random.KeyArray], Any],
        accumulator_init: Any
    ):
        """Helper method that creates evaluators using the model to roll out policies.
        Created so that the rollout_policy parameters are JAX vmap-able.

        Args:
            rollout_policy: A policy mapping of the form (rollout_policy_params, obs, timestep, rng_key) -> action.
                See examples below.
            rollout_horizon: Rollout horizon
            fn_to_accumulate: A pytree-valued function on
                (rollout_policy_params, dynamics_params, dynamics_state, timestep_information, rng_key)
                that will be summed over the rollout for trajectory evaluation.
                rng_key is provided to allow for random functions (e.g. entropy bonus estimated from samples)
            accumulator_init: Initial value for the accumulator, must be of the same type as the output of
                fn_to_accumulate.

        Returns:
            Policy evaluator for the set horizon which, given
            (rollout policy params, dynamics params, dynamics state, start_obs, JAX RNG key),
            outputs a predicted final state, the sum of fn_to_accumulate over the rollout, and predicted observation
            and action sequences.

        >>> # Evaluator for length-10 action sequences (e.g. MPC/random shooting methods).
        >>> evaluator_mpc = self._create_rollout_evaluator(
        ...     lambda action_seq, _, i, __: action_seq[i],
        ...     10,
        ...     self._wrap_basic_reward(self._reward_fn),
        ...     0.
        ... )
        >>>
        >>> # Evaluator for time-independent policy over a length-50 horizon
        >>> policy: DeterministicPolicy
        >>> evaluator_policy = self._create_rollout_evaluator(
        ...     lambda policy_params, obs, _, __: policy.act(policy_params, obs),
        ...     50,
        ...     self._wrap_basic_reward(self._reward_fn),
        ...     0.
        ... )
        """
        def rollout_and_evaluate(rollout_policy_params, dynamics_params, dynamics_state, start_obs, rng_key):
            # Randomly select an ensemble member for this rollout
            rng_key, subkey = jax.random.split(rng_key)
            member = jax.random.randint(subkey, minval=0, maxval=self._ensemble_size, shape=())
            member_params = jax.tree_map(lambda all_params: all_params[member], dynamics_params)
            member_state = jax.tree_map(lambda all_states: all_states[member], dynamics_state)

            def simulate_single_timestep(prev_timestep_carry, h):
                cur_obs, cur_return, r_key = prev_timestep_carry

                r_key, s_key = jax.random.split(r_key)
                action = rollout_policy(rollout_policy_params, cur_obs, h, s_key)

                r_key, s_key = jax.random.split(r_key)
                next_obs = self._dynamics_model.predict(member_params, member_state, cur_obs, action, s_key)

                timestep_info = {
                    "timestep": h,
                    "observation": cur_obs,
                    "action": action,
                    "next_observation": next_obs
                }

                r_key, s_key = jax.random.split(r_key)
                summand = fn_to_accumulate(rollout_policy_params, member_params, member_state, timestep_info, s_key)
                cur_return = jax.tree_map(lambda x, y: x + y, cur_return, summand)

                return (next_obs, cur_return, r_key), (cur_obs, action)

            (final_obs, accumulation, _), (obs_seq, action_seq) = jax.lax.scan(
                simulate_single_timestep,
                (start_obs, accumulator_init, rng_key),
                jnp.arange(rollout_horizon, dtype=int)
            )
            return {
                "final_observation": final_obs,
                "accumulation": accumulation,
                "observation_sequence": jnp.append(obs_seq, final_obs[None], axis=0),
                "action_sequence": action_seq
            }

        return rollout_and_evaluate

    def _batch_mean_log_likelihood(self, net_params, net_state, batch):
        log_likelihood_evaluator = partial(self._dynamics_model.log_likelihood, net_params, net_state)
        log_likelihoods = jax.vmap(log_likelihood_evaluator)(
            batch["observation"], batch["action"], batch["next_observation"]
        )
        return jnp.mean(log_likelihoods)

    @staticmethod
    def _wrap_basic_reward(reward_fn):
        """Convenience function which wraps a basic reward function in another function
        that is compatible with create_rollout_evaluator.
        """
        return lambda _, __, ___, timestep_info, ____: \
            reward_fn(timestep_info["observation"], timestep_info["action"], timestep_info["next_observation"])
