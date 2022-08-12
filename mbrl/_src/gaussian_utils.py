import jax
import jax.numpy as jnp


def gaussian_log_prob(gaussian_params, query):
    weighted_mse = 0.5 * jnp.sum(jnp.square((query - gaussian_params["mean"]) / gaussian_params["stddev"]))
    log_det_cov = jnp.sum(gaussian_params["log_stddev"])
    dim = gaussian_params["mean"].shape[0]
    return -(weighted_mse + log_det_cov + (dim / 2) * jnp.log(2 * jnp.pi))


def create_bounded_gaussianizer(
    min_stddev: float,
    max_stddev: float
):
    delta = max_stddev - min_stddev

    def gaussianize(query):
        out_dim = query.shape[-1] // 2
        stddev_logit = query[..., out_dim:]
        stddev = min_stddev + delta * jax.nn.sigmoid(4 * (stddev_logit / delta))
        return {
            "mean": query[..., :out_dim],
            "stddev": stddev,
            "log_stddev": jnp.log(stddev)
        }

    return gaussianize


def reparameterized_gaussian_sampler(gaussian_params, rng_key):
    # Use reparameterization trick to allow gradients to pass through sampling step.
    return gaussian_params["mean"] + \
           gaussian_params["stddev"] * jax.random.normal(rng_key, shape=gaussian_params["mean"].shape)
