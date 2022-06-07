import argparse
import os

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


CONFIG = {
    "MujocoCartpole-v0": {
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
        "reward_functions": {
            "obs_reward_fn": lambda obs: jnp.exp(-jnp.sum(jnp.square(
                jnp.array([obs[0] + 0.6 * jnp.sin(obs[1]), 0.6 * jnp.cos(obs[1])]) - jnp.array([0, 0.6])
            ))),
            "action_reward_fn": lambda action: -0.01 * jnp.sum(jnp.square(action))
        },
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
    }
}


def rollout(env, agent=None, record_path=""):
    observations, actions = [], []
    observations.append(env.reset())
    if record_path != "":
        recorder = VideoRecorder(env, base_path=record_path)
        recorder.capture_frame()

    done = False
    while not done:
        if agent is None:
            ac = env.action_space.sample()
        else:
            ac = agent.act(observations[-1])

        ob, reward, done, _ = env.step(ac)
        if record_path != "":
            recorder.capture_frame()

        observations.append(ob)
        actions.append(ac)

    if record_path != "":
        recorder.close()
    return observations, actions


def main(env_name, agent_type, record_dir="", seed=0):
    env = gym.make(env_name)

    if args.record_dir != "":
        os.makedirs(args.record_dir, exist_ok=True)
        os.makedirs(os.path.join(args.record_dir, "init_trajs"), exist_ok=True)
        os.makedirs(os.path.join(args.record_dir, "agent_rollouts"), exist_ok=True)

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
            **CONFIG[env_name]["reward_functions"],
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
            **CONFIG[env_name]["reward_functions"],
            rng_key=jax.random.PRNGKey(seed),
            policy=policy,
            policy_optimizer=optax.adamw(1e-3)
        )
    else:
        raise RuntimeError("Invalid agent type.")

    for i in trange(1, ncols=150, desc="Collecting initial trajectories"):
        record_path = record_dir if record_dir == "" else os.path.join(record_dir, "init_trajs", str(i))
        observations, actions = rollout(env, record_path=record_path)
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

    for i in range(100):
        agent.train(
            **CONFIG[env_name]["model_training"],
            **(CONFIG[env_name]["policy_training"] if agent_type == "Policy" else {})
        )
        agent.reset()
        record_path = record_dir if record_dir == "" else os.path.join(record_dir, "agent_rollouts", str(i))
        observations, actions = rollout(env, agent=agent, record_path=record_path)
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=str, default="",
                        help="Path to folder for saving rollouts.")
    parser.add_argument("-s", type=int, default=-1,
                        help="Random seed.")
    parser.add_argument("env", choices=["MujocoCartpole-v0"],
                        help="Environment [MujocoCartpole-v0]")
    parser.add_argument("agent_type", choices=["PETS", "Policy"],
                        help="Agent type [PETS/Policy]")
    args = parser.parse_args()

    main(
        args.env,
        args.agent_type,
        record_dir=args.record_dir,
        seed=args.s if args.s != -1 else onp.random.randint(10000)
    )
