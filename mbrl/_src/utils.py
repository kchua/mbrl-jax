import logging
from typing import Callable, Union

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import jax.numpy as jnp
import numpy as onp

Array = Union[jnp.ndarray, onp.ndarray]
Activation = Callable[[Array], Array]


def normalize(normalizer_params, query):
    return (query - normalizer_params["center"]) / normalizer_params["scale"]


def denormalize(normalizer_params, query):
    return query * normalizer_params["scale"] + normalizer_params["center"]


def rollout(env, discount_factor, agent=None, recording_path=None, evaluation=False):
    observations, actions = [], []
    observations.append(env.reset())

    recorder = VideoRecorder(env, base_path=recording_path, enabled=(recording_path is not None))
    recorder.capture_frame()

    rollout_return, rollout_discounted_return, cur_discount_multiplier = 0., 0., 1.
    done = False
    while not done:
        if agent is None:
            ac = env.action_space.sample()
        else:
            ac = agent.act(observations[-1], evaluation=evaluation)

        ob, reward, done, _ = env.step(ac)
        recorder.capture_frame()

        rollout_return += reward
        rollout_discounted_return += cur_discount_multiplier * reward
        cur_discount_multiplier *= discount_factor

        observations.append(ob)
        actions.append(ac)

    rollout_statistics = {
        "Return": rollout_return,
        "Discounted return": rollout_discounted_return
    }

    recorder.close()
    return observations, actions, rollout_statistics


def print_logging_statistics(iteration, logging_statistics, n_after_decimal=6):
    max_len = len(max(logging_statistics, key=lambda x: len(x)))
    number_length = 3 + n_after_decimal + 4
    writeable_length = max(max_len + 1 + 1 + 1 + number_length, 30)
    label_length = writeable_length - 3 - number_length

    logging.info("#" * (1 + 1 + label_length + 1 + 1 + 1 + number_length + 1 + 1))
    logging.info(("# {:^%d} #" % writeable_length).format("Iteration {} Statistics".format(iteration)))
    logging.info("# " + (" " * writeable_length) + " #")
    for stat, value in logging_statistics.items():
        if value >= 0:
            logging.info(("# {:>%d} :  {:.%de} #" % (label_length, n_after_decimal)).format(stat, value))
        else:
            logging.info(("# {:>%d} : {:.%de} #" % (label_length, n_after_decimal)).format(stat, value))
    logging.info("#" * (1 + 1 + max_len + 1 + 1 + 1 + 3 + n_after_decimal + 4 + 1 + 1))
