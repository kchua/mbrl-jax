from functools import partial
from typing import Callable

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp
import numpy as onp
import optax

from mbrl.agents import DeepModelBasedAgent
from mbrl.misc import NeuralNetDynamicsModel
from mbrl._src.utils import Array


class ModelPredictiveControlAgent(DeepModelBasedAgent):
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
        n_candidates: int,
        n_elites: int,
        plan_horizon: int,
        n_particles: int,
        cem_epsilon: float,
        max_cem_iters: int,
        *_args, **_kwargs
    ):
        """Creates an agent implementing PETS, i.e. a model predictive control-based agent
        which uses an ensemble of neural networks for its internal dynamics model, and CEM
        as its action sequence optimizer.

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
            n_candidates: Number of action sequence candidates for every iteration of CEM.
            n_elites: Number of elite action sequences used to update the CEM proposal distribution.
            plan_horizon: Planning horizon to use.
            n_particles: Number of independent particles to use for evaluating each action sequence.
            cem_epsilon: If the proposal distribution stddev drops below this, terminates CEM optimization.
            max_cem_iters: Maximum number of CEM iterations before forced termination.
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

        self._action_bounds = (env.action_space.low, env.action_space.high)
        self._action_dim = env.action_space.shape[0]
        self._n_candidates = n_candidates
        self._n_elites = n_elites
        self._plan_horizon = plan_horizon
        self._n_particles = n_particles
        self._cem_epsilon = cem_epsilon
        self._max_cem_iters = max_cem_iters

        self._proposal_dist = {
            "mean": jnp.zeros([self._plan_horizon, self._action_dim]),
            "stddev": jnp.ones([self._plan_horizon, self._action_dim])
        }

        self._cem_op = self._create_cross_entropy_method_op()

    def reset(self) -> None:
        """Resets the agent after an environment rollout."""
        self._proposal_dist = {
            "mean": jnp.zeros([self._plan_horizon, self._action_dim]),
            "stddev": jnp.ones([self._plan_horizon, self._action_dim])
        }

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
        optimized_action_seq_dist = self._cem_op(
            self._dynamics_params, self._dynamics_state, obs, self._proposal_dist, subkey
        )

        next_proposal_dist_mean = jnp.concatenate([
            optimized_action_seq_dist["mean"][1:],
            jnp.zeros([1, self._action_dim])]
        )
        next_proposal_dist_stddev = jnp.concatenate([
            0.5 * jnp.ones([self._plan_horizon - 1, self._action_dim]),
            jnp.ones([1, self._action_dim])   # Adding more noise since new timestep is unoptimized
        ])
        self._proposal_dist = {"mean": next_proposal_dist_mean, "stddev": next_proposal_dist_stddev}

        return onp.array(self._proposal_space_to_actions(optimized_action_seq_dist["mean"][0]))

    def _create_cross_entropy_method_op(self):
        rollout_and_evaluate = self._create_rollout_evaluator(
            rollout_policy=lambda action_seq, _obs, i, _rng_key: action_seq[i],
            rollout_horizon=self._plan_horizon,
            fn_to_accumulate=self._wrap_basic_reward(self._reward_fn),
            accumulator_init=0.
        )

        def multiple_particle_evaluator(action_seq, dynamics_params, dynamics_state, start_obs, rng_key):
            single_particle_evaluator = \
                partial(rollout_and_evaluate, action_seq, dynamics_params, dynamics_state, start_obs)
            particle_returns = \
                jax.vmap(single_particle_evaluator)(jax.random.split(rng_key, num=self._n_particles))["accumulation"]
            return jnp.mean(particle_returns)

        @jax.jit
        def cross_entropy_method_update(dynamics_params, dynamics_state, cur_obs, proposal_dist, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            pre_bounding_candidates = proposal_dist["mean"][None] + proposal_dist["stddev"][None] * \
                jax.random.normal(subkey, shape=[self._n_candidates] + list(proposal_dist["mean"].shape))

            action_seq_candidates = self._proposal_space_to_actions(pre_bounding_candidates)

            rng_key, subkey = jax.random.split(rng_key)
            candidate_evals = jax.vmap(multiple_particle_evaluator, (0, None, None, None, 0))(
                action_seq_candidates,
                dynamics_params,
                dynamics_state,
                cur_obs,
                jax.random.split(subkey, num=self._n_candidates)
            )

            sorting_idxs = jnp.argsort(candidate_evals)
            elite_candidates = pre_bounding_candidates[sorting_idxs[-self._n_elites:]]
            updated_proposal_dist = {
                "mean": jnp.mean(elite_candidates, axis=0),
                "stddev": jnp.std(elite_candidates, axis=0)
            }

            return updated_proposal_dist

        def cross_entropy_method(dynamics_params, dynamics_state, cur_obs, proposal_dist, rng_key):
            n_iters = 0
            while n_iters < self._max_cem_iters and jnp.max(proposal_dist["stddev"]) > self._cem_epsilon:
                rng_key, subkey = jax.random.split(rng_key)
                proposal_dist = cross_entropy_method_update(
                    dynamics_params, dynamics_state, cur_obs, proposal_dist, subkey
                )
                n_iters += 1
            return proposal_dist

        return cross_entropy_method

    def _proposal_space_to_actions(self, proposals):
        box_center = (self._action_bounds[0] + self._action_bounds[1]) / 2
        box_width = (self._action_bounds[1] - self._action_bounds[0]) / 2
        return box_center + box_width * jnp.tanh(proposals)
