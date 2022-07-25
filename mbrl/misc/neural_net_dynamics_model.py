from typing import Callable, Dict, List, Union

import jax
import jax.numpy as jnp

from mbrl.misc.fully_connected_neural_net import FullyConnectedNeuralNet
from mbrl._src.utils import Array, create_gaussianizer, gaussian_log_prob, normalize, reparameterized_gaussian_sampler


class NeuralNetDynamicsModel:
    def __init__(
        self,
        name: str,
        dummy_obs: Array,
        dummy_action: Array,
        hidden_dims: List[int],
        hidden_activations: Union[None, Callable[[Array], Array], List[Union[None, Callable[[Array], Array]]]],
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
            name: Model name. Used for referencing parameters, so must be unique.
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
        self._name = name
        self._obs_preproc = obs_preproc
        self._targ_comp = targ_comp
        self._next_obs_comp = next_obs_comp
        self._is_probabilistic = is_probabilistic
        self._is_normalizing_inputs = normalize_inputs
        self._is_normalizing_outputs = normalize_outputs

        preproc_obs_dim = self._obs_preproc(dummy_obs).shape[0]
        action_dim = dummy_action.shape[0]
        pred_dim = self._targ_comp(dummy_obs, dummy_obs).shape[0]

        self._normalizer_params = {
            name: {
                "center": jnp.zeros(shape=[dim]),
                "scale": jnp.ones(shape=[dim])
            } for (name, dim) in [("input", preproc_obs_dim + action_dim), ("output", pred_dim)]
        }

        self._internal_net = FullyConnectedNeuralNet(
            name="{}_internal".format(self._name),
            input_dim=preproc_obs_dim + action_dim,
            output_dim=2*pred_dim if self._is_probabilistic else pred_dim,
            hidden_dims=hidden_dims,
            hidden_activations=hidden_activations,
            output_activation=create_gaussianizer(
                stddev_nonlinearity=jax.nn.softplus,
                min_stddev=min_stddev,
                max_stddev=max_stddev
            ) if self._is_probabilistic else None
        )

    @property
    def is_probabilistic(self) -> bool:
        """Does the model return stochastic predictions?"""
        return self._is_probabilistic

    def init(
        self,
        params: Dict,
        rng_key: jax.random.KeyArray
    ) -> Dict:
        """Places randomly initialized net parameters within the given dictionary, which is returned.

        Args:
            params: Dictionary of parameters where initialization will be placed.
            rng_key: JAX RNG key, which should not be reused outside this function.

        Returns:
            params
        """
        return self._internal_net.init(params, rng_key)

    def fit_normalizer(self, obs, actions, next_obs) -> None:
        """Fits the normalizer of this network to the given points.

        Args:
            obs: Array of observations.
            actions: Array of actions.
            next_obs: Array of next observations.
        """
        if obs.shape[0] != actions.shape[0] or actions.shape[0] != next_obs.shape[0]:
            raise RuntimeError("Arrays for fitting normalizer do not have matching lengths.")

        if self._is_normalizing_inputs:
            inputs = jnp.concatenate([jax.vmap(self._obs_preproc)(obs), actions], axis=-1)
            input_mean = jnp.mean(inputs, axis=0)
            input_stddev = jnp.std(inputs, axis=0)
            self._normalizer_params["input"] = {
                "center": input_mean,
                "scale": jnp.where(input_stddev > 1e-5, input_stddev, jnp.ones_like(input_stddev))
            }

        if self._is_normalizing_outputs:
            outputs = jax.vmap(self._targ_comp)(obs, next_obs)
            output_mean = jnp.mean(outputs, axis=0)
            output_stddev = jnp.std(outputs, axis=0)
            self._normalizer_params["output"] = {
                "center": output_mean,
                "scale": jnp.where(output_stddev > 1e-5, output_stddev, jnp.ones_like(output_stddev))
            }

    def predict(
        self,
        params: Dict,
        obs: Array,
        action: Array,
        rng_key=None
    ) -> jnp.ndarray:
        """Returns the prediction of the model on a SINGLE observation and action.

        Args:
            params: Dictionary of parameters.
            obs: Environment observation.
            action: Action.
            rng_key: JAX RNG key, only needed for probabilistic model. Do not reuse outside function.

        Returns:
            The predicted next observation according to the model.
        """
        raw_output = self._compute_net_output(params, obs, action)

        if self._is_probabilistic:
            if rng_key is None:
                raise RuntimeError("Probabilistic dynamics models require a JAX RNG key for prediction.")

            raw_prediction = reparameterized_gaussian_sampler(raw_output, rng_key)
        else:
            raw_prediction = raw_output

        if self._is_normalizing_outputs:
            raw_prediction = normalize(raw_prediction, self._normalizer_params["output"], invert=True)
        return self._next_obs_comp(obs, raw_prediction)

    def prediction_loss(
        self,
        params: Dict,
        obs: Array,
        action: Array,
        next_obs: Array
    ) -> jnp.ndarray:
        """Computes the negative log-likelihood of the target induced by (obs, next_obs) with respect to the model,
        conditioned on (obs, action), up to additive constants.

        Note: For deterministic models, the log-likelihood is computed as if the network output is the mean of a
        multivariate Gaussian with identity covariance.

        Args:
            params: Dictionary of parameters.
            obs: Environment observation.
            action: Action.
            next_obs: Next environment observation.

        Returns:
            Negative log-likelihood.
        """
        raw_output = self._compute_net_output(params, obs, action)

        if self._is_normalizing_outputs:
            targ = normalize(self._targ_comp(obs, next_obs), self._normalizer_params["output"])
        else:
            targ = self._targ_comp(obs, next_obs)

        if self.is_probabilistic:
            gaussian_params = raw_output
        else:
            gaussian_params = {"mean": raw_output, "stddev": jnp.ones_like(raw_output)}

        return -gaussian_log_prob(gaussian_params, targ)

    def _compute_net_output(self, params, obs, action):
        preproc_obs = self._obs_preproc(obs)
        if self._is_normalizing_inputs:
            return self._internal_net.forward(
                params,
                normalize(jnp.concatenate([preproc_obs, action]), self._normalizer_params["input"])
            )
        else:
            return self._internal_net.forward(params, jnp.concatenate([preproc_obs, action]))
