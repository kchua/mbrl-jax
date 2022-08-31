from typing import Callable, Dict, List, Optional, Union

import jax
import jax.numpy as jnp

from mbrl.misc.fully_connected_neural_net import FullyConnectedNeuralNet
from mbrl._src.utils import Array, denormalize, normalize
from mbrl._src.gaussian_utils import create_bounded_gaussianizer, gaussian_log_prob, reparameterized_gaussian_sampler


class NeuralNetDynamicsModel:
    def __init__(
        self,
        dummy_obs: Array,
        dummy_action: Array,
        hidden_dims: List[int],
        hidden_activations: Optional[Union[Callable[[Array], Array], List[Optional[Callable[[Array], Array]]]]],
        is_probabilistic: bool,
        min_stddev: float = 1e-5,
        max_stddev: float = 100.,
        obs_preproc: Callable[[Array], Array] = lambda obs: obs,
        targ_comp: Callable[[Array, Array], Array] = lambda obs, next_obs: next_obs,
        next_obs_comp: Callable[[Array, Array], Array] = lambda obs, pred: pred,
        normalize_inputs: bool = True,
        normalize_outputs: bool = True
    ):
        """Creates a dynamics model which internally uses a neural network with a deterministic output.

        Args:
            dummy_obs: Single dummy observation used to infer shapes.
            dummy_action: Single dummy action used to infer shapes.
            hidden_dims: List of hidden dimensions for internal network.
            hidden_activations: A single activation type to use for every hidden layer,
                OR a list of activations to use for each hidden layer specified in hidden dims.
            is_probabilistic: Indicates whether the internal network will output a distribution.
            min_stddev: Minimum stddev that can be output by a probabilistic model, for training stability.
            max_stddev: Maximum stddev that can be output by a probabilistic model, for training stability.
            obs_preproc: Preprocesses observations before feeding into the network.
                Defaults to identity operation.
            targ_comp: Computes network targets from (obs, next_obs).
                Defaults to (obs, next_obs) -> next_obs (i.e. network directly predicts next observation).
            next_obs_comp: Computes next_obs prediction from (obs, net_output).
                Defaults to (obs, net_output) -> net_output (i.e. network directly predicts next observation).
            normalize_inputs: Indicates if input normalization will be used.
            normalize_outputs: Indicates if output normalization will be used.
        """
        self._obs_preproc = obs_preproc
        self._targ_comp = targ_comp
        self._next_obs_comp = next_obs_comp
        self._is_probabilistic = is_probabilistic
        self._is_normalizing_inputs = normalize_inputs
        self._is_normalizing_outputs = normalize_outputs

        self._preproc_obs_dim = self._obs_preproc(dummy_obs).size
        self._action_dim = dummy_action.size
        self._pred_dim = self._targ_comp(dummy_obs, dummy_obs).size

        self._internal_net = FullyConnectedNeuralNet(
            input_dim=self._preproc_obs_dim + self._action_dim,
            output_dim=2*self._pred_dim if self._is_probabilistic else self._pred_dim,
            hidden_dims=hidden_dims,
            hidden_activations=hidden_activations,
            output_activation=create_bounded_gaussianizer(min_stddev, max_stddev) if self._is_probabilistic else None
        )

    @property
    def is_probabilistic(self) -> bool:
        """Does the model return stochastic predictions?"""
        return self._is_probabilistic

    def init(
        self,
        params: Dict,
        state: Dict,
        rng_key: jax.random.KeyArray
    ) -> (Dict, Dict):
        """Places randomly initialized net parameters and state within the given dictionary, which is returned.

        Args:
            params: Dictionary where initialized model parameters will be placed.
            state: Dictionary where model state will be placed.
            rng_key: JAX RNG key, which should not be reused outside this function.

        Returns:
            params
        """
        params["internal_net"], state["internal_net"] = self._internal_net.init({}, {}, rng_key)
        state["normalizer"] = {
            "input": {
                "center": jnp.zeros(shape=[self._preproc_obs_dim + self._action_dim]),
                "scale": jnp.ones(shape=[self._preproc_obs_dim + self._action_dim]),
            },
            "output": {
                "center": jnp.zeros(shape=[self._pred_dim]),
                "scale": jnp.ones(shape=[self._pred_dim])
            }
        }
        return params, state

    def fit_normalizer(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        actions: Array,
        next_obs: Array
    ) -> (Dict, Dict):
        """Fits the normalizer of this network to the given points.

        Args:
            params: Dictionary of model parameters.
            state: Dictionary of model state.
            obs: Array of observations.
            actions: Array of actions.
            next_obs: Array of next observations.

        Returns:
            Updated params and state reflecting fitted normalizer.
        """
        if obs.shape[0] != actions.shape[0] or actions.shape[0] != next_obs.shape[0]:
            raise RuntimeError("Arrays for fitting normalizer do not have matching lengths.")

        if self._is_normalizing_inputs:
            inputs = jax.vmap(self._compute_unnormalized_net_input)(obs, actions)
            input_mean = jnp.mean(inputs, axis=0)
            input_stddev = jnp.std(inputs, axis=0)
            state["normalizer"]["input"] = {
                "center": input_mean,
                "scale": jnp.where(input_stddev > 1e-5, input_stddev, jnp.ones_like(input_stddev))
            }

        if self._is_normalizing_outputs:
            outputs = jax.vmap(self._targ_comp)(obs, next_obs)
            output_mean = jnp.mean(outputs, axis=0)
            output_stddev = jnp.std(outputs, axis=0)
            state["normalizer"]["output"] = {
                "center": output_mean,
                "scale": jnp.where(output_stddev > 1e-5, output_stddev, jnp.ones_like(output_stddev))
            }

        return params, state

    def predict(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        action: Array,
        rng_key=None
    ) -> jnp.ndarray:
        """Returns the prediction of the model on a SINGLE observation and action.

        Args:
            params: Dictionary of model parameters.
            state: Dictionary of model state.
            obs: Environment observation.
            action: Action.
            rng_key: JAX RNG key, only needed for probabilistic model. Do not reuse outside function.

        Returns:
            The predicted next observation according to the model.
        """
        raw_output = self._compute_net_output(params, state, obs, action)

        if self._is_probabilistic:
            if rng_key is None:
                raise RuntimeError("Probabilistic dynamics models require a JAX RNG key for prediction.")

            raw_prediction = reparameterized_gaussian_sampler(raw_output, rng_key)
        else:
            raw_prediction = raw_output

        raw_prediction = denormalize(state["normalizer"]["output"], raw_prediction)
        return self._next_obs_comp(obs, raw_prediction)

    def log_likelihood(
        self,
        params: Dict,
        state: Dict,
        obs: Array,
        action: Array,
        next_obs: Array
    ) -> jnp.ndarray:
        """Computes the log-likelihood of the target induced by (obs, next_obs) with respect to the model,
        conditioned on (obs, action).

        Note: For deterministic models, the log-likelihood is computed as if the network output is the mean of a
        multivariate Gaussian with identity covariance.

        Args:
            params: Dictionary of model parameters.
            state: Dictionary of model state.
            obs: Environment observation.
            action: Action.
            next_obs: Next environment observation.

        Returns:
            Log-likelihood.
        """
        raw_output = self._compute_net_output(params, state, obs, action)
        targ = normalize(state["normalizer"]["output"], self._targ_comp(obs, next_obs))

        if self.is_probabilistic:
            gaussian_params = raw_output
        else:
            gaussian_params = {"mean": raw_output, "stddev": 1.}

        return gaussian_log_prob(gaussian_params, targ)

    def _compute_net_output(self, params, state, obs, action):
        unnormalized_net_input = self._compute_unnormalized_net_input(obs, action)
        return self._internal_net.forward(
            params["internal_net"],
            state["internal_net"],
            normalize(state["normalizer"]["input"], unnormalized_net_input)
        )

    def _compute_unnormalized_net_input(self, obs, action):
        preproc_obs = self._obs_preproc(obs)
        return jnp.concatenate([preproc_obs, action])
