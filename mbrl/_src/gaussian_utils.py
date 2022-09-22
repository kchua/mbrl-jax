from typing import Callable

import jax
import jax.numpy as jnp


def gaussian_log_prob(gaussian_params, query):
    """Evaluates the log-likelihood of the query point under the multivariate Gaussian with the given
    parameters.

    Args:
        gaussian_params: A dictionary containing "mean", "stddev", and "log_stddev"
        query: Query point.

    Returns:
        Log-likelihood
    """
    query, mean, stddev, log_stddev = jnp.broadcast_arrays(
        query, gaussian_params["mean"], gaussian_params["stddev"], gaussian_params["log_stddev"]
    )
    dim = mean.size
    weighted_mse = 0.5 * jnp.sum(jnp.square((query - mean) / stddev))
    log_det_cov = jnp.sum(log_stddev)
    return -(weighted_mse + log_det_cov + (dim / 2) * jnp.log(2 * jnp.pi))


def gaussian_entropy(gaussian_params):
    log_cov_term = jnp.sum(gaussian_params["log_stddev"])
    constant = 0.5 * gaussian_params["mean"].size * (1 + jnp.log(2 * jnp.pi))
    return log_cov_term + constant


def create_bounded_gaussianizer(
    min_stddev: float,
    max_stddev: float,
    mean_activation: Callable = lambda x: x
):
    """Creates an output activation which maps from an input tensor of shape [..., dim] to the
    parameters of a Gaussian with mean and stddev both of shape [..., dim // 2].

    Args:
        min_stddev: Minimum standard deviation output.
        max_stddev: Maximum standard deviation output.
        mean_activation: Activation applied to obtain the mean, defaults to the identity operation.

    Returns:
        Activation function.
    """
    delta = max_stddev - min_stddev

    def gaussianize(query):
        out_dim = query.shape[-1] // 2
        stddev_logit = query[..., out_dim:]
        stddev = min_stddev + delta * jax.nn.sigmoid(4 * (stddev_logit / delta))
        return {
            "mean": mean_activation(query[..., :out_dim]),
            "stddev": stddev,
            "log_stddev": jnp.log(stddev)
        }

    return gaussianize


def reparameterized_gaussian_sampler(gaussian_params, rng_key):
    """Samples from the given Gaussian using the reparameterization trick (to allow gradients to pass into the
    Gaussian parameters).

    Args:
        gaussian_params: Parameters of the Gaussian that will be sampled.
        rng_key: JAX RNG key for sampling, do not reuse outside function.

    Returns:
        sample from Gaussian.
    """
    mean, stddev = jnp.broadcast_arrays(gaussian_params["mean"], gaussian_params["stddev"])
    standard_normal_sample = jax.random.normal(rng_key, shape=mean.shape)
    return mean + stddev * standard_normal_sample
