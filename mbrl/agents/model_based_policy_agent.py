from typing import Callable

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import trange

from mbrl.agents import DeepModelBasedAgent
from mbrl.misc import Dataset, NeuralNetDynamicsModel
from mbrl.policies import DeterministicPolicy
from mbrl._src.utils import Array


class ModelBasedPolicyAgent(DeepModelBasedAgent):
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
        policy: DeterministicPolicy,
        plan_horizon: int,
        n_particles: int,
        policy_optimizer: optax.GradientTransformation,
        n_policy_train_steps: int,
        policy_train_batch_size: int,
        *_args, **_kwargs
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
            n_model_eval_points: Number of points to evaluate the trained dynamics model on for logging statistics.
            rng_key: JAX RNG key to be used by this agent internally. Do not reuse.
            policy: The neural network policy that will be used by this agent
            plan_horizon: Planning horizon to use.
            n_particles: Number of independent particles to use for evaluating each action sequence.
            policy_optimizer: The optimizer that will be used to train the policy.
            n_policy_train_steps: Number of parameter updates to perform for the policy.
            policy_train_batch_size: Size of batches to use for each parameter update for the policy.
        """
        super().__init__(
            env,
            reward_fn,
            dynamics_model,
            ensemble_size,
            dynamics_optimizer,
            n_model_train_steps,
            model_train_batch_size,
            n_model_eval_points,
            rng_key
        )

        self._policy = policy
        self._plan_horizon = plan_horizon
        self._n_particles = n_particles
        self._policy_optimizer = policy_optimizer
        self._n_policy_train_steps = n_policy_train_steps
        self._policy_train_batch_size = policy_train_batch_size

        self._policy_dataset = Dataset(observation=env.observation_space.shape)

        self._rng_key, subkey = jax.random.split(self._rng_key)
        self._policy_params = self._policy.init({}, subkey)

        self._policy_optimizer_state = self._policy_optimizer.init(self._policy_params)
        self._policy_update_op = self._create_policy_update_op()

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
        super().add_interaction_data(obs_seq, action_seq)

        self._policy_dataset.add(observation=obs_seq)

    def train(self) -> None:
        """Trains the internal dynamics model and policy of this agent with all interaction data so far.
        """
        super().train()

        self._rng_key, subkey = jax.random.split(self._rng_key)
        epoch_iterator = self._policy_dataset.epoch(self._policy_train_batch_size, subkey)
        for _ in trange(self._n_policy_train_steps, desc="Policy training", unit="batches", ncols=150):
            while True:
                try:
                    batch_obs = next(epoch_iterator)["observation"]
                    break
                except StopIteration:
                    self._rng_key, subkey = jax.random.split(self._rng_key)
                    epoch_iterator = self._policy_dataset.epoch(self._policy_train_batch_size, subkey)

            self._rng_key, subkey = jax.random.split(self._rng_key)
            self._policy_params, self._policy_optimizer_state = self._policy_update_op(
                self._policy_params, self._dynamics_params, self._policy_optimizer_state,
                batch_obs,
                subkey
            )

    def act(
        self,
        obs: Array,
    ) -> onp.ndarray:
        """Queries the agent for its action on the given observation.

        Args:
            obs: The current observation.

        Returns:
            Action chosen by the agent
        """
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return onp.array(self._policy.act(self._policy_params, obs, subkey))

    def _create_policy_update_op(self):
        rollout_and_evaluate = self._create_rollout_evaluator(
            rollout_policy=lambda policy_params, obs, _i, rng_key: self._policy.act(policy_params, obs, rng_key),
            rollout_horizon=self._plan_horizon,
            fn_to_accumulate=self._wrap_basic_reward(self._reward_fn)
        )

        def multiple_particle_evaluator(policy_params, dynamics_params, start_obs, rng_key):
            particle_returns = jax.vmap(rollout_and_evaluate, (None, None, None, 0))(
                policy_params, dynamics_params, start_obs, jax.random.split(rng_key, num=self._n_particles)
            )["rollout_return"]
            return jnp.mean(particle_returns)

        @jax.jit
        def perform_policy_update(policy_params, dynamics_params, policy_optimizer_state, batch_start, rng_key):
            def mean_batch_cost_to_go(*args):
                return -jnp.mean(jax.vmap(multiple_particle_evaluator, (None, None, 0, 0))(*args)[1])

            batch_grad = jax.grad(mean_batch_cost_to_go)(
                policy_params, dynamics_params, batch_start, jax.random.split(rng_key, num=batch_start.shape[0])
            )
            updates, policy_optimizer_state = \
                self._policy_optimizer.update(batch_grad, policy_optimizer_state, policy_params)
            return optax.apply_updates(policy_params, updates), policy_optimizer_state

        return perform_policy_update
