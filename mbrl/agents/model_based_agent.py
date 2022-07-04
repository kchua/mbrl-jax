from abc import ABC, abstractmethod
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
        n_model_eval_points: int,
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
            n_model_eval_points: Number of points to evaluate the trained dynamics model on for logging statistics.
            rng_key: JAX RNG key to be used by this agent internally. Do not reuse.
        """
        self._reward_fn = reward_fn
        self._dynamics_model = dynamics_model
        self._ensemble_size = ensemble_size
        self._dynamics_optimizer = dynamics_optimizer
        self._n_model_train_steps = n_model_train_steps
        self._model_train_batch_size = model_train_batch_size
        self._n_model_eval_points = n_model_eval_points
        self._rng_key = rng_key

        self._dynamics_dataset = Dataset(
            observation=env.observation_space.shape,
            action=env.action_space.shape,
            next_observation=env.observation_space.shape
        )

        params_per_member = [{} for _ in range(self._ensemble_size)]
        for idx in range(self._ensemble_size):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            params_per_member[idx] = self._dynamics_model.init(params_per_member[idx], subkey)
        self._dynamics_params = jax.tree_map(lambda *a: jnp.stack(a), *params_per_member)

        self._model_eval_op = self._create_model_eval_op()

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

        self._dynamics_dataset.add(
            observation=obs_seq[:-1],
            action=action_seq,
            next_observation=obs_seq[1:]
        )

    def train(self) -> None:
        """Trains the internal dynamics model of this agent with all the provided interaction data so far.
        """
        self._dynamics_model.fit_normalizer(
            self._dynamics_dataset["observation"],
            self._dynamics_dataset["action"],
            self._dynamics_dataset["next_observation"]
        )

        self._rng_key, subkey = jax.random.split(self._rng_key)
        bootstrapped_dataset = self._dynamics_dataset.bootstrap(self._ensemble_size, subkey)

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

            self._dynamics_params, self._dynamics_optimizer_state = self._model_update_op(
                self._dynamics_params, self._dynamics_optimizer_state, batches_per_member
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
        iterator = self._dynamics_dataset.epoch(self._n_model_eval_points, subkey, full_batch_required=False)
        return {
            "Dynamics model log-likelihood": self._model_eval_op(self._dynamics_params, next(iterator))
        }

    def _create_model_eval_op(self):
        @jax.jit
        def ensemble_eval(dynamics_params, eval_batch):
            def batch_mean_loss(single_net_params, batch):
                return jnp.mean(jax.vmap(self._dynamics_model.prediction_loss, (None, 0, 0, 0))(
                    single_net_params, batch["observation"], batch["action"], batch["next_observation"]
                ))
            return jnp.mean(jax.vmap(batch_mean_loss, (0, None))(dynamics_params, eval_batch))

        return ensemble_eval

    def _create_model_update_op(self):
        @jax.jit
        def model_update(dynamics_params, dynamics_optimizer_state, all_member_batches):
            def batch_mean_loss(single_net_params, batch):
                return jnp.mean(jax.vmap(self._dynamics_model.prediction_loss, (None, 0, 0, 0))(
                    single_net_params, batch["observation"], batch["action"], batch["next_observation"]
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
        rollout_policy: Callable[[Any, jnp.ndarray, int, jax.random.KeyArray], jnp.ndarray],
        rollout_horizon: int,
        fn_to_accumulate: Callable[[int, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict, Dict, jax.random.KeyArray],
                                   jnp.ndarray],
        discount_factor: float = 1.0
    ):
        """Helper method that creates evaluators using the model to roll out policies.
        Created so that the rollout_policy parameters are JAX vmap-able.

        Args:
            rollout_policy: A policy mapping of the form (params, obs, i, rng_key) -> action.
                See examples below.
            rollout_horizon: Rollout horizon
            fn_to_accumulate: A scalar-valued function on
                (observation, action, next_observation, rollout_policy_params, dynamics_params, rng_key)
                that will be accumulated over the rollout for trajectory evaluation.
                For example, setting this to self.reward_fn will return an evaluator that computes the rollout return.
                rng_key is provided to allow for random functions (e.g. entropy bonus estimated from samples)
            discount_factor: Discount factor used for computing returns.
                Defaults to 1.0 (i.e. no discount).

        Returns:
            Policy evaluator which, given (rollout policy params, dynamics parameters, start_obs, JAX RNG key),
            outputs a predicted final state and the sum of fn_to_accumulate over the rollout (truncated to the given
            horizon).

        >>> # Evaluator for length-10 action sequences (e.g. MPC/random shooting methods).
        >>> evaluator_mpc = self._create_rollout_evaluator(
        ...     lambda action_seq, _obs, i: action_seq[i],
        ...     10,
        ...     self._wrap_basic_reward(self._reward_fn)
        ... )
        >>>
        >>> # Evaluator for time-independent policy over a length-50 horizon
        >>> policy: DeterministicPolicy
        >>> evaluator_policy = self._create_rollout_evaluator(
        ...     lambda policy_params, obs, _i: policy.act(policy_params, obs),
        ...     50,
        ...     self._wrap_basic_reward(self._reward_fn)
        ... )
        """
        def rollout_and_evaluate(rollout_policy_params, dynamics_params, start_obs, rng_key):
            # Assign an ensemble member to each particle at random
            rng_key, subkey = jax.random.split(rng_key)
            member = jax.random.randint(
                subkey, minval=0, maxval=self._ensemble_size, shape=()
            )
            member_params = jax.tree_map(
                lambda all_params: all_params[member], dynamics_params
            )

            def simulate_single_timestep(prev_timestep_info, h):
                cur_obs, cur_return, r_key = prev_timestep_info

                r_key, s_key = jax.random.split(r_key)
                action = rollout_policy(rollout_policy_params, cur_obs, h, s_key)

                r_key, s_key = jax.random.split(r_key)
                next_obs = self._dynamics_model.predict(member_params, cur_obs, action, s_key)

                r_key, s_key = jax.random.split(r_key)
                cur_return += (discount_factor ** h) * fn_to_accumulate(
                    h, cur_obs, action, next_obs, rollout_policy_params, member_params, s_key
                )

                return (next_obs, cur_return, r_key), cur_obs

            (final_obs, rollout_return, _), obs_seq = jax.lax.scan(
                simulate_single_timestep, (start_obs, 0., rng_key), jnp.arange(rollout_horizon, dtype=int)
            )
            return {
                "final_observation": final_obs,
                "rollout_return": rollout_return,
                "observation_sequence": jnp.append(obs_seq, final_obs[None], axis=0)
            }

        return rollout_and_evaluate

    @staticmethod
    def _wrap_basic_reward(reward_fn):
        """Convenience function which wraps a basic reward function in another function
        that is compatible with create_rollout_evaluator.
        """
        return lambda _, observation, action, next_observation, __, ___, ____: \
            reward_fn(observation, action, next_observation)
