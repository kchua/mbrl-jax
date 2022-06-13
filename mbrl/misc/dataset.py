from typing import Iterable

import jax
import jax.numpy as jnp


class Dataset:
    def __init__(self, **kwargs: Iterable[int]):
        """Creates a dataset with named components.

        Args:
            **kwargs: Dictionary mapping from a named component to expected shape.

        >>> # Creates a dataset for a regression task with inputs in R^10 and outputs in R
        >>> Dataset(inputs=(10,), outputs=())
        >>> # Creates a dataset for an image classification task with 256x256 images and binary labels
        >>> Dataset(inputs=(256, 256), outputs=())
        """
        self._dataset = {}
        self._names = []
        self._length = 0

        for name, shape in kwargs.items():
            self._names.append(name)
            self._dataset[name] = jnp.zeros([0] + list(shape))

    def __len__(self):
        return self._length

    def add(self, **kwargs: jnp.ndarray) -> None:
        """Adds a list of new datapoints to the existing dataset.

        Args:
            **kwargs: Dictionary mapping from named components to list of new points.
                List of named components provided must match that provided at construction time,
                and the lists associated with each named component must all be of equal length.
        """
        if len(kwargs) != len(self._names):
            raise RuntimeError("Number of named components during add() does not match "
                               "dataset at construction time.")

        n_elems_to_add = None
        for name in self._names:
            if name not in kwargs:
                raise RuntimeError("Cannot add to dataset without specifying "
                                   "all named components provided during dataset construction.")

            if n_elems_to_add is None:
                n_elems_to_add = kwargs[name].shape[0]
            elif kwargs[name].shape[0] != n_elems_to_add:
                raise RuntimeError(
                    "Named components must be equal-length lists of items when adding to dataset."
                )

            if kwargs[name].shape[1:] != self._dataset[name].shape[1:]:
                raise RuntimeError("Invalid shape for {} provided when adding to dataset.".format(name))

        self._length += n_elems_to_add
        for name in self._names:
            self._dataset[name] = jnp.concatenate([self._dataset[name], kwargs[name]])

    def bootstrap(
        self,
        ensemble_size: int,
        rng_key: jax.random.KeyArray
    ):
        """Creates a bootstrapped version of this dataset on which the user can call epoch().
        Allows bootstraps to persist between separate epochs.

        Args:
            ensemble_size: The number of bootstrap datasets to generate
            rng_key: JAX RNG key used to sampled bootstraps, do not reuse outside this function.
        """
        return Dataset._BootstrappedDataset(self, ensemble_size, rng_key)

    def epoch(
        self,
        batch_size: int,
        rng_key: jax.random.KeyArray
    ):
        """Creates an iterator to iterate through the dataset with batches of the given size.

        Args:
            batch_size: Size of batches returned by the iterator.
            rng_key: JAX RNG key used to shuffle data for the epoch, do not reuse outside this function.
        """
        return Dataset._DatasetIterator(self, batch_size, rng_key)

    class _BootstrappedDataset:
        def __init__(self, dataset, ensemble_size, rng_key):
            self._dataset = dataset

            if ensemble_size > 1:
                rng_key, subkey = jax.random.split(rng_key)
                # Sample bootstraps
                self._bootstrap_idxs = jax.random.randint(
                    subkey, minval=0, maxval=len(dataset), shape=[ensemble_size, len(dataset)]
                )
            else:
                self._bootstrap_idxs = jnp.arange(len(dataset))[None]

        def epoch(self, batch_size, rng_key):
            return Dataset._DatasetIterator(
                self._dataset, batch_size, rng_key, idxs=self._bootstrap_idxs
            )

    class _DatasetIterator:
        def __init__(self, dataset, batch_size, rng_key, idxs=None):
            self._dataset: Dataset = dataset
            self._dataset_length = len(dataset)
            self._batch_size = batch_size
            self._epoch_steps = 0
            self._rng_key = rng_key

            if idxs is not None:
                self._idxs = idxs
            else:
                self._idxs = jnp.arange(len(dataset))

            if self._idxs.shape[-1] < batch_size:
                raise RuntimeError("Not enough data for full batch.")

            # Shuffle indices for epoch
            self._rng_key, subkey = jax.random.split(self._rng_key)
            self._idxs = jax.random.permutation(subkey, self._idxs, axis=-1, independent=True)

        def __iter__(self):
            return self

        def __next__(self):
            if self._batch_size * (self._epoch_steps + 1) > self._dataset_length:
                raise StopIteration("Not enough data for a full batch.")

            batch_start = self._epoch_steps * self._batch_size
            batch_end = batch_start + self._batch_size
            batch_idxs = self._idxs[..., batch_start:batch_end]
            self._epoch_steps += 1

            return jax.tree_map(lambda x: x[batch_idxs], self._dataset._dataset)
