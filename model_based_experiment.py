import argparse
import os

import dill
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from tqdm import trange

from mbrl.agents import ModelPredictiveControlAgent, ModelBasedPolicyAgent
from mbrl.misc import NeuralNetDynamicsModel, NeuralNetPolicy
import mbrl.envs


def cartpole_reward(obs, action, _next_obs):
    obs_reward = jnp.exp(-jnp.sum(jnp.square(
        jnp.array([obs[0] + 0.6 * jnp.sin(obs[1]), 0.6 * jnp.cos(obs[1])]) - jnp.array([0, 0.6])
    )))
    action_reward = -0.01 * jnp.sum(jnp.square(action))
    return obs_reward + action_reward


def halfcheetah_reward(obs, action, next_obs):
    obs_reward = (next_obs[0] - obs[0]) / 0.05     # TODO: 0.05 is the dt of the environment, should set automatically
    action_reward = -0.1 * jnp.sum(jnp.square(action))
    return obs_reward + action_reward


CONFIG = {
    "MujocoCartpole-v0": {
        "env_args": {},
        "dynamics": {
            "hidden_dims": [50, 50, 50],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic": True
        },
        "policy": {
            "hidden_dims": [50, 50, 50],
            "hidden_activations": jax.nn.swish,
        },
        "preprocessing_functions": {
            "obs_preproc": lambda obs: jnp.concatenate([obs[:1], jnp.sin(obs[1:2]), jnp.cos(obs[1:2]), obs[2:]]),
            "targ_comp": lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "model_training": {
            "n_model_train_steps": 2000,
            "model_train_batch_size": 32
        },
        "prediction": {
            "ensemble_size": 10,
            "dynamics_optimizer": optax.adamw(1e-3),
            "plan_horizon": 20,
            "n_particles": 30,
        },
        "reward_function": cartpole_reward,
        "cem": {
            "n_candidates": 400,
            "n_elites": 40,
            "cem_epsilon": 0.05,
            "max_cem_iters": 10,
        },
        "policy_training": {
            "n_policy_train_steps": 2000,
            "policy_train_batch_size": 32
        }
    },
    "HalfCheetah-v3": {
        "env_args": {
            "exclude_current_positions_from_observation": False,    # Need access to x_pos to compute velocity
        },
        "dynamics": {
            "hidden_dims": [200, 200, 200],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic": True
        },
        "policy": {},
        "preprocessing_functions": {
            "obs_preproc": lambda obs: jnp.concatenate([obs[1:2], jnp.sin(obs[2:3]), jnp.cos(obs[2:3]), obs[3:]]),
            "targ_comp": lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "model_training": {
            "n_model_train_steps": 2000,
            "model_train_batch_size": 32
        },
        "prediction": {
            "ensemble_size": 5,
            "dynamics_optimizer": optax.adamw(1e-3),
            "plan_horizon": 30,
            "n_particles": 15,
        },
        "reward_function": halfcheetah_reward,
        "cem": {
            "n_candidates": 500,
            "n_elites": 50,
            "cem_epsilon": 0.05,
            "max_cem_iters": 5,
        },
        "policy_training": {}
    }
}


def rollout(env, agent=None, recording_path=None):
    observations, actions = [], []
    observations.append(env.reset())

    recorder = VideoRecorder(env, base_path=recording_path, enabled=(recording_path is not None))
    recorder.capture_frame()
    done = False
    while not done:
        if agent is None:
            ac = env.action_space.sample()
        else:
            ac = agent.act(observations[-1])

        ob, reward, done, _ = env.step(ac)
        recorder.capture_frame()

        observations.append(ob)
        actions.append(ac)

    recorder.close()
    return observations, actions


def main(
    env_name,
    agent_type,
    logdir=None,
    save_every=1,
    keep_all_checkpoints=False,
    seed=0
):
    env = gym.make(env_name, **CONFIG[env_name]["env_args"])
    env.seed(seed)

    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, "agent_rollouts"), exist_ok=True)
        os.makedirs(os.path.join(logdir, "checkpoints"), exist_ok=True)

    dynamics_model = NeuralNetDynamicsModel(
        name="dynamics_model",
        dummy_obs=env.reset(),
        dummy_action=env.action_space.sample(),
        **CONFIG[env_name]["dynamics"],
        **CONFIG[env_name]["preprocessing_functions"]
    )

    if agent_type == "PETS":
        agent = ModelPredictiveControlAgent(
            env=env,
            dynamics_model=dynamics_model,
            **CONFIG[env_name]["prediction"],
            reward_fn=CONFIG[env_name]["reward_function"],
            **CONFIG[env_name]["cem"],
            rng_key=jax.random.PRNGKey(seed),
        )
    elif agent_type == "Policy":
        policy = NeuralNetPolicy(
            name="policy",
            env=env,
            dummy_obs=env.reset(),
            **CONFIG[env_name]["policy"],
            obs_preproc=CONFIG[env_name]["preprocessing_functions"]["obs_preproc"]
        )
        agent = ModelBasedPolicyAgent(
            env=env,
            dynamics_model=dynamics_model,
            **CONFIG[env_name]["prediction"],
            reward_fn=CONFIG[env_name]["reward_function"],
            rng_key=jax.random.PRNGKey(seed),
            policy=policy,
            policy_optimizer=optax.adamw(1e-3)
        )
    else:
        raise RuntimeError("Invalid agent type.")

    for _ in trange(1, ncols=150, desc="Collecting initial trajectories"):
        observations, actions = rollout(env)
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

    for i in range(100):
        agent.train(
            **CONFIG[env_name]["model_training"],
            **(CONFIG[env_name]["policy_training"] if agent_type == "Policy" else {})
        )
        agent.reset()

        if logdir is not None and (i + 1) % save_every == 0:
            recording_path = os.path.join(logdir, "agent_rollouts/iter_{}".format(i + 1))
        else:
            recording_path = None

        observations, actions = rollout(env, agent=agent, recording_path=recording_path)
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

        if logdir is not None and (i + 1) % save_every == 0:
            if keep_all_checkpoints:
                checkpoint_path = os.path.join(logdir, "checkpoints/iter_{}.pkl".format(i + 1))
            else:
                checkpoint_path = os.path.join(logdir, "checkpoints/checkpoint.pkl")

            with open(checkpoint_path, "wb") as f:
                dill.dump({
                    "agent": agent,
                    "env": env,
                    "iteration": i + 1,
                    "seed": seed
                }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None,
                        help="Path to folder for saving all logs.")
    parser.add_argument("--save-every", type=int, default=1,
                        help="How often agent checkpoints and videos will be saved.")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="If provided, keeps all checkpoints (rather than only the most recent one).")
    parser.add_argument("-s", type=int, default=-1,
                        help="Random seed.")
    parser.add_argument("env", choices=["MujocoCartpole-v0", "HalfCheetah-v3"],
                        help="Environment [MujocoCartpole-v0, HalfCheetah-v3]")
    parser.add_argument("agent_type", choices=["PETS", "Policy"],
                        help="Agent type [PETS/Policy]")
    args = parser.parse_args()

    main(
        args.env,
        args.agent_type,
        logdir=args.logdir,
        save_every=args.save_every,
        keep_all_checkpoints=args.keep_all_checkpoints,
        seed=args.s if args.s != -1 else onp.random.randint(10000)
    )
