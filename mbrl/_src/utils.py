from typing import Union

import jax
import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]


def create_gaussianizer(
    stddev_nonlinearity=None,
    log_stddev_nonlinearity=None,
    min_stddev: float = -onp.inf,
    max_stddev: float = onp.inf,
):
    if stddev_nonlinearity is None and log_stddev_nonlinearity is None:
        raise RuntimeError("Must specify nonlinearity used to obtain (log) standard deviation.")
    if stddev_nonlinearity is not None and log_stddev_nonlinearity is not None:
        raise RuntimeError("Only specify either standard deviation or its log, the other is inferred.")

    def gaussianize(query):
        out_dim = query.shape[-1] // 2
        if stddev_nonlinearity is not None:
            stddev = jnp.clip(
                stddev_nonlinearity(query[..., out_dim:]),
                a_min=min_stddev,
                a_max=max_stddev
            )
            return {
                "mean": query[..., :out_dim],
                "log_stddev": jnp.log(stddev),
                "stddev": stddev
            }
        else:
            log_stddev = jnp.clip(
                log_stddev_nonlinearity(query[..., out_dim:]),
                a_min=jnp.log(min_stddev),
                a_max=jnp.log(max_stddev)
            )
            return {
                "mean": query[..., :out_dim],
                "log_stddev": log_stddev,
                "stddev": jnp.exp(log_stddev)
            }

    return gaussianize


def reparameterized_gaussian_sampler(gaussian_params, rng_key):
    # Use reparameterization trick to allow gradients to pass through sampling step.
    return gaussian_params["mean"] + \
        gaussian_params["stddev"] * jax.random.normal(rng_key, shape=gaussian_params["mean"].shape)
