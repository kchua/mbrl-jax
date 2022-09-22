from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from gym.envs.mujoco import MujocoEnv

from mbrl.misc import FullyConnectedNeuralNet
from mbrl._src.gaussian_utils import create_bounded_gaussianizer, gaussian_entropy, reparameterized_gaussian_sampler
from mbrl._src.utils import Activation, Array


class TanhGaussianPolicy:
    def __init__(
        self,
        env: MujocoEnv,
        dummy_obs: Array,
        hidden_dims: List[int],
        hidden_activations: Optional[Union[Activation, List[Optional[Activation]]]],
        obs_preproc: Callable[[Array], Array] = lambda obs: obs,
        min_stddev: float = 1e-5,
        max_stddev: float = 100
    ):
        self._action_space_center = (env.action_space.low + env.action_space.high) / 2
        self._action_space_half_width = (env.action_space.high - env.action_space.low) / 2
        self._obs_preproc = obs_preproc

        preproc_obs_dim = self._obs_preproc(dummy_obs).shape[0]
        action_dim = env.action_space.shape[0]

        self._pre_tanh_network = FullyConnectedNeuralNet(
            input_dim=preproc_obs_dim,
            output_dim=2*action_dim,
            hidden_dims=hidden_dims,
            hidden_activations=hidden_activations,
            output_activation=create_bounded_gaussianizer(min_stddev, max_stddev)
        )

    @property
    def is_stochastic(self) -> bool:
        return True

    def init(
        self,
        params: Dict,
        state: Dict,
        rng_key: jax.random.KeyArray
    ) -> (Dict, Dict):
        return self._pre_tanh_network.init(params, state, rng_key)

    def act(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        rng_key: jax.random.KeyArray = None
    ):
        pre_tanh_distribution = self._compute_forward_pass(params, state, obs)
        pre_tanh_sample = reparameterized_gaussian_sampler(pre_tanh_distribution, rng_key)
        action = self._convert_to_actions(pre_tanh_sample)
        return action

    def entropy(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        rng_key: jax.random.KeyArray
    ):
        pre_tanh_distribution = self._compute_forward_pass(params, state, obs)
        pre_tanh_sample = reparameterized_gaussian_sampler(pre_tanh_distribution, rng_key)
        entropy = self._compute_sample_entropy(pre_tanh_distribution, pre_tanh_sample)
        return entropy

    def sample_with_entropy_estimate(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        rng_key: jax.random.KeyArray
    ):
        pre_tanh_distribution = self._compute_forward_pass(params, state, obs)
        pre_tanh_sample = reparameterized_gaussian_sampler(pre_tanh_distribution, rng_key)
        action = self._convert_to_actions(pre_tanh_sample)
        entropy = self._compute_sample_entropy(pre_tanh_distribution, pre_tanh_sample)
        return action, entropy

    def get_mean_action(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
    ):
        pre_tanh_distribution = self._compute_forward_pass(params, state, obs)
        return self._convert_to_actions(pre_tanh_distribution["mean"])

    def _compute_forward_pass(self, params, state, obs):
        return self._pre_tanh_network.forward(params, state, self._obs_preproc(obs))

    def _convert_to_actions(self, query):
        return self._action_space_center + self._action_space_half_width * jnp.tanh(query)

    def _compute_sample_entropy(self, pre_tanh_params, pre_tanh_sample):
        base_gaussian_entropy = gaussian_entropy(pre_tanh_params)
        pushforward_entropy_contribution = jnp.sum(jnp.log(4 * self._action_space_half_width)) \
            - jnp.sum(jax.nn.softplus(2 * pre_tanh_sample) + jax.nn.softplus(-2 * pre_tanh_sample))
        return base_gaussian_entropy + pushforward_entropy_contribution
