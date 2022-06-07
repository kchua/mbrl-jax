import os

from gym import utils
from gym.envs.mujoco import mujoco_env, InvertedPendulumEnv
import numpy as np


class MujocoCartpoleEnv(InvertedPendulumEnv):
    """A modified version of the InvertedPendulum-v2 environment in OpenAI Gym.
    Allows the pendulum to fall all the way, requiring the agent to learn how to swing the pole back up
    when this happens.
    """
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.getcwd(), "mbrl/envs/assets/cartpole.xml"), 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos_reward = np.exp(-np.sum(np.square(
            np.array([ob[0] + 0.6 * np.sin(ob[1]), 0.6 * np.cos(ob[1])]) - np.array([0, 0.6])
        )))
        action_cost = 0.01 * np.sum(np.square(a))
        done = not np.isfinite(ob).all()
        return ob, pos_reward - action_cost, done, {}
