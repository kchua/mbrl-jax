from typing import Iterable, Optional

import jax
import jax.numpy as jnp


class Dataset:
    def __init__(
        self,
        max_length: Optional[int] = None,
        **kwargs: Iterable[int]
    ):
        """Creates a dataset with named components.

        Args:
            max_length: Optional argument specifying the largest size allowed for this dataset. Points are dropped
                in order to enforce this constraint, with the oldest points being dropped first.
            **kwargs: Dictionary mapping from a named component to expected shape.

        >>> # Creates a dataset for a regression task with inputs in R^10 and outputs in R
        >>> Dataset(inputs=(10,), outputs=())
        >>> # Creates a dataset for an image classification task with 256x256 images and binary labels
        >>> Dataset(inputs=(256, 256), outputs=())
        """
        self._dataset = {}
        self._length = 0
        self._max_length = max_length

        for name, shape in kwargs.items():
            self._dataset[name] = jnp.zeros([0] + list(shape))

    def __len__(self):
        """Returns number of points added to this dataset so far."""
        return self._length

    def __iter__(self):
        """Returns an iterator over the named elements of this dataset."""
        return self._dataset.__iter__()

    def __getitem__(self, key) -> jnp.ndarray:
        """Returns a named component in the dataset, if it exists."""
        return self._dataset[key]

    def add(self, **kwargs: jnp.ndarray) -> None:
        """Adds a list of new datapoints to the existing dataset.

        Args:
            **kwargs: Dictionary mapping from named components to list of new points.
                List of named components provided must match that provided at construction time,
                and the lists associated with each named component must all be of equal length.
        """
        if len(kwargs) != len(self._dataset.keys()):
            raise RuntimeError("Number of named components during add() does not match "
                               "dataset at construction time.")

        n_elems_to_add = None
        for name in self._dataset:
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

        if self._max_length is None:
            self._length += n_elems_to_add
        else:
            self._length = min(self._length + n_elems_to_add, self._max_length)

        for name in self._dataset:
            self._dataset[name] = jnp.concatenate([self._dataset[name], kwargs[name]])[-self._length:]

    def sample(
        self,
        sample_size,
        rng_key: jax.random.KeyArray
    ):
        """Samples a set of size sample_size uniformly with replacement from this dataset.

        Args:
            sample_size: Size of set to sample from this dataset.
            rng_key: JAX RNG key used to sample from this dataset, do not reuse.

        Returns:
            A set of samples of size sample_size sampled uniformly with replacement.
        """
        idxs = jax.random.randint(rng_key, (sample_size,), 0, len(self), dtype=int)
        return jax.tree_map(lambda x: x[idxs], self._dataset)

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
        rng_key: jax.random.KeyArray,
        full_batch_required: bool = True
    ):
        """Creates an iterator to iterate through the dataset with batches of the given size.

        Args:
            batch_size: Size of batches returned by the iterator.
            rng_key: JAX RNG key used to shuffle data for the epoch, do not reuse outside this function.
            full_batch_required: If True, the epoch terminates once there is not enough data left to form a full
                batch.
        """
        return Dataset._DatasetIterator(
            self, batch_size, rng_key,
            full_batch_required=full_batch_required
        )

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

        def __len__(self):
            """Returns the number of points in each bootstrap."""
            return len(self._dataset)

        def __iter__(self):
            """Returns an iterator over the named elements of this dataset."""
            return self._dataset.__iter__()

        def __getitem__(self, key) -> jnp.ndarray:
            """Returns a named component in the dataset, if it exists."""
            return self._dataset[key][self._bootstrap_idxs]

        def epoch(self, batch_size, rng_key, full_batch_required=True):
            return Dataset._DatasetIterator(
                self._dataset, batch_size, rng_key,
                idxs=self._bootstrap_idxs,
                full_batch_required=full_batch_required
            )

    class _DatasetIterator:
        def __init__(self, dataset, batch_size, rng_key, idxs=None, full_batch_required=True):
            self._dataset: Dataset = dataset
            self._batch_size = batch_size
            self._epoch_steps = 0
            self._rng_key = rng_key
            self._full_batch_required = full_batch_required

            if idxs is not None:
                self._idxs = idxs
            else:
                self._idxs = jnp.arange(len(dataset))

            if self._idxs.shape[-1] < batch_size and full_batch_required:
                raise RuntimeError("Not enough data for full batch.")

            # Shuffle indices for epoch
            self._rng_key, subkey = jax.random.split(self._rng_key)
            self._idxs = jax.random.permutation(subkey, self._idxs, axis=-1, independent=True)

        def __iter__(self):
            return self

        def __next__(self):
            if self._full_batch_required and self._batch_size * (self._epoch_steps + 1) > self._idxs.shape[-1]:
                raise StopIteration("Not enough data left for a full batch.")
            if self._batch_size * self._epoch_steps >= self._idxs.shape[-1]:
                raise StopIteration("Reached end of dataset.")

            batch_start = self._epoch_steps * self._batch_size
            batch_end = min(batch_start + self._batch_size, self._idxs.shape[-1])
            batch_idxs = self._idxs[..., batch_start:batch_end]
            self._epoch_steps += 1

            return jax.tree_map(lambda x: x[batch_idxs], self._dataset._dataset)
