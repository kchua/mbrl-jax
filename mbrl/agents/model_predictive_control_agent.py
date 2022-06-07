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
        dynamics_model: NeuralNetDynamicsModel,
        ensemble_size: int,
        dynamics_optimizer: optax.GradientTransformation,
        plan_horizon: int,
        n_particles: int,
        obs_reward_fn: Callable[[jnp.ndarray], jnp.ndarray],
        action_reward_fn: Callable[[jnp.ndarray], jnp.ndarray],
        rng_key: jax.random.KeyArray,
        n_candidates: int,
        n_elites: int,
        cem_epsilon: float,
        max_cem_iters: int,
        *_args, **_kwargs
    ):
        """Creates an agent implementing PETS, i.e. a model predictive control-based agent
        which uses an ensemble of neural networks for its internal dynamics model, and CEM
        as its action sequence optimizer.

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
            n_candidates: Number of action sequence candidates for every iteration of CEM.
            n_elites: Number of elite action sequences used to update the CEM proposal distribution.
            cem_epsilon: If the proposal distribution stddev drops below this, terminates CEM optimization.
            max_cem_iters: Maximum number of CEM iterations before forced termination.
        """
        super().__init__(
            env,
            dynamics_model, ensemble_size, dynamics_optimizer, plan_horizon, n_particles,
            obs_reward_fn, action_reward_fn, rng_key,
        )

        self._action_bounds = (env.action_space.low, env.action_space.high)
        self._action_dim = env.action_space.shape[0]
        self._n_candidates = n_candidates
        self._n_elites = n_elites
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
        optimized_action_seq_dist = self._cem_op(self._dynamics_params, obs, self._proposal_dist, subkey)

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
            rollout_policy=lambda action_seq, _obs, i: action_seq[i],
            rollout_horizon=self._plan_horizon
        )

        @jax.jit
        def cross_entropy_method_update(params, cur_obs, proposal_dist, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            pre_bounding_candidates = proposal_dist["mean"][None] + proposal_dist["stddev"][None] * \
                jax.random.normal(subkey, shape=[self._n_candidates] + list(proposal_dist["mean"].shape))

            action_seq_candidates = self._proposal_space_to_actions(pre_bounding_candidates)

            rng_key, subkey = jax.random.split(rng_key)
            candidate_evals = jax.vmap(rollout_and_evaluate, (0, None, None, 0))(
                action_seq_candidates, params, cur_obs, jax.random.split(subkey, num=self._n_candidates)
            )

            sorting_idxs = jnp.argsort(candidate_evals)
            elite_candidates = pre_bounding_candidates[sorting_idxs[-self._n_elites:]]
            updated_proposal_dist = {
                "mean": jnp.mean(elite_candidates, axis=0),
                "stddev": jnp.std(elite_candidates, axis=0)
            }

            return updated_proposal_dist

        def cross_entropy_method(dynamics_params, cur_obs, proposal_dist, rng_key):
            n_iters = 0
            while n_iters < self._max_cem_iters and jnp.max(proposal_dist["stddev"]) > self._cem_epsilon:
                rng_key, subkey = jax.random.split(rng_key)
                proposal_dist = cross_entropy_method_update(
                    dynamics_params, cur_obs, proposal_dist, subkey
                )
                n_iters += 1
            return proposal_dist

        return cross_entropy_method

    def _proposal_space_to_actions(self, proposals):
        box_center = (self._action_bounds[0] + self._action_bounds[1]) / 2
        box_width = (self._action_bounds[1] - self._action_bounds[0]) / 2
        return box_center + box_width * jnp.tanh(proposals)
