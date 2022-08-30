from typing import Union

import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]


def normalize(normalizer_params, query):
    return (query - normalizer_params["center"]) / normalizer_params["scale"]

def denormalize(normalizer_params, query):
    return query * normalizer_params["scale"] + normalizer_params["center"]
