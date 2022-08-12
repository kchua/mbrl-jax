from typing import Any, Callable, Dict, List, Union

import jax.numpy as jnp
import jax.random

from mbrl._src.utils import Array


class FullyConnectedNeuralNet:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        hidden_activations: Union[None, Callable[[Array], Array], List[Union[None, Callable[[Array], Array]]]],
        output_activation: Union[None, Callable[[Array], Any]] = None
    ):
        """Creates a fully-connected network.

        Args:
            input_dim: Net input dimension.
            output_dim: Net output dimension.
            hidden_dims: List of hidden layer dimensions.
            hidden_activations: A single activation type to use for every hidden layer,
                OR a list of activations to use for each hidden layer specified in hidden dims.
            output_activation: Activation to use for the final layer.
                Defaults to the identity operation.

        >>> net = FullyConnectedNeuralNet(5, 10, [100], jax.nn.relu)   # Creates a network R^5 -> R^10 with a single
                                                                       #   hidden layer of 100 ReLU neurons.
        """
        if not isinstance(hidden_activations, list):
            hidden_activations = len(hidden_dims) * [hidden_activations]
        if len(hidden_dims) != len(hidden_activations):
            raise ValueError("Number of hidden layers does not match number of hidden activations provided.")

        self._input_dim = input_dim
        self._intermediate_dims = hidden_dims + [output_dim]
        self._intermediate_activations = hidden_activations + [output_activation]

    def init(
        self,
        params: Dict,
        state: Dict,
        rng_key: jax.random.KeyArray
    ) -> (Dict, Dict):
        """Places randomly initialized net parameters and state within the given dictionaries, which are returned.

        Args:
            params: Dictionary where the network will place its own parameters.
            state: Dictionary where the network will place its internal state.
            rng_key: JAX RNG key, which should not be reused outside this function.

        Returns:
            params
        """
        cur_dim = self._input_dim
        for depth, out_dim in enumerate(self._intermediate_dims):
            layer_params = params["transform{}".format(depth)] = {}
            rng_key, subkey = jax.random.split(rng_key)
            layer_params["weights"] = jnp.sqrt(2 / cur_dim) * jax.random.normal(subkey, shape=[out_dim, cur_dim])
            layer_params["biases"] = jnp.zeros(out_dim)
            cur_dim = out_dim
        return params, state

    def forward(
        self,
        params: Dict,
        _state: Dict,
        query: Array
    ) -> Any:
        """Returns the output of the network on a single query point, using the given parameter dictionary.

        Args:
            params: Dictionary of network parameters.
            _state: Dictionary containing network state.
            query: Point on which network will be queried. query.shape = [input_dim].

        Returns:
            Network output evaluated at query. return.shape = [output_dim]
        """
        try:
            hidden = query
            for depth, activation in enumerate(self._intermediate_activations):
                layer_params = params["transform{}".format(depth)]
                pre_activation = jnp.dot(layer_params["weights"], hidden) + layer_params["biases"]
                hidden = pre_activation if activation is None else activation(pre_activation)
            return hidden
        except KeyError:
            raise RuntimeError("Missing parameters in provided parameter dictionary. Did you call init()?")

