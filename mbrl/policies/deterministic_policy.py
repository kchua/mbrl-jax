from typing import Callable, Dict, List, Union

from gym.envs.mujoco import MujocoEnv
import jax
import jax.numpy as jnp

from mbrl.misc import FullyConnectedNeuralNet
from mbrl._src.utils import Array


class DeterministicPolicy:
    def __init__(
        self,
        env: MujocoEnv,
        dummy_obs: Array,
        hidden_dims: List[int],
        hidden_activations: Union[None, Callable[[Array], Array], List[Union[None, Callable[[Array], Array]]]],
        obs_preproc: Callable[[Array], Array] = lambda obs: obs,
    ):
        """Creates a policy network.

        Args:
            env: Environment. Used for inferring shapes and action bounds.
            dummy_obs: Observation from the environment, used for inferring shapes.
            hidden_dims: List of hidden dimensions for internal network.
            hidden_activations: A single activation type to use for every hidden layer,
                OR a list of activations to use for each hidden layer specified in hidden_dims.
            obs_preproc: Preprocesses observations before feeding into the network.
                Defaults to identity operation.
        """
        self._action_bounds = (env.action_space.low, env.action_space.high)
        self._obs_preproc = obs_preproc

        preproc_obs_dim = self._obs_preproc(dummy_obs).shape[0]
        action_dim = env.action_space.shape[0]

        def convert_to_actions(query):
            box_center = (self._action_bounds[0] + self._action_bounds[1]) / 2
            box_width = (self._action_bounds[1] - self._action_bounds[0]) / 2
            return box_center + box_width * jnp.tanh(query)

        self._internal_net = FullyConnectedNeuralNet(
            input_dim=preproc_obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_activations=hidden_activations,
            output_activation=convert_to_actions
        )

    def init(
        self,
        params: Dict,
        state: Dict,
        rng_key: jax.random.KeyArray
    ) -> Dict:
        """Randomly initializes policy parameters, places them within the given parameter dictionary, and returns it.

        Args:
            params: Dictionary where initialization parameters will be placed.
            state: Dictionary where policy network state will be placed.
            rng_key: JAX RNG key, which should not be reused outside this function.

        Returns:
            params
        """
        return self._internal_net.init(params, state, rng_key)

    def act(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        _rng_key=None
    ) -> jnp.ndarray:
        """Returns the action of the policy on a SINGLE observation.

        Args:
            params: Dictionary of parameters.
            obs: Environment observation.
            _rng_key: Unused.

        Returns:
            The action taken by the policy on the given observation.
        """
        return self._internal_net.forward(params, state, self._obs_preproc(obs))
