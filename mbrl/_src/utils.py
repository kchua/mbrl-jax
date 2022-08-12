from typing import Union

import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]


def normalize(input, normalizer_params, invert=False):
    if not invert:
        return (input - normalizer_params["center"]) / normalizer_params["scale"]
    else:
        return input * normalizer_params["scale"] + normalizer_params["center"]
