from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

from mbrl.misc import FullyConnectedNeuralNet
from mbrl._src.utils import Activation, Array


class QFunction:
    def __init__(
        self,
        dummy_obs: Array,
        dummy_action: Array,
        hidden_dims: List[int],
        hidden_activations: Optional[Union[Activation, List[Optional[Activation]]]],
        obs_preproc: Callable[[Array], Array] = lambda obs: obs
    ):
        self._obs_preproc = obs_preproc

        preproc_obs_dim = self._obs_preproc(dummy_obs).shape[0]
        action_dim = dummy_action.shape[0]

        self._internal_net = FullyConnectedNeuralNet(
            input_dim=preproc_obs_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            hidden_activations=hidden_activations,
            output_activation=lambda x: x[0]    # Converts one-dim output to scalar.
        )

    def init(
        self,
        params: Dict,
        state: Dict,
        rng_key: jax.random.KeyArray
    ):
        return self._internal_net.init(params, state, rng_key)

    def value(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        action: Array
    ):
        return self._internal_net.forward(params, state, self._compute_net_input(obs, action))

    def _compute_net_input(
        self,
        obs: Array,
        action: Array
    ):
        return jnp.concatenate([self._obs_preproc(obs), action])
