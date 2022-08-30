from typing import Union

import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]


def normalize(normalizer_params, query, invert=False):
    if not invert:
        return (query - normalizer_params["center"]) / normalizer_params["scale"]
    else:
        return query * normalizer_params["scale"] + normalizer_params["center"]
