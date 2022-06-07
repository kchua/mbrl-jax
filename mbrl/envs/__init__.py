from gym import register

register(
    id="MujocoCartpole-v0",
    entry_point="mbrl.envs.mujoco_cartpole:MujocoCartpoleEnv",
    max_episode_steps=200,
)
