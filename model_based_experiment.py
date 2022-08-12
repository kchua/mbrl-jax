import argparse
import logging
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
from mbrl.misc import NeuralNetDynamicsModel
from mbrl.policies import DeterministicPolicy
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
        "discount_factor": 0.99,
        "env_args": {},
        "preprocessing_functions": {
            "obs_preproc": lambda obs: jnp.concatenate([obs[:1], jnp.sin(obs[1:2]), jnp.cos(obs[1:2]), obs[2:]]),
            "targ_comp": lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "reward_function": cartpole_reward,

        "dynamics": {
            "hidden_dims": [50, 50, 50],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic": True
        },
        "model_training_and_evaluation": {
            "ensemble_size": 10,
            "dynamics_optimizer": optax.adamw(1e-3),
            "n_model_train_steps": 2000,
            "model_train_batch_size": 32,
            "n_model_eval_points": 1000
        },

        "cem": {
            "n_candidates": 400,
            "n_elites": 40,
            "plan_horizon": 20,
            "n_particles": 30,
            "cem_epsilon": 0.05,
            "max_cem_iters": 10,
        },

        "policy": {
            "hidden_dims": [50, 50, 50],
            "hidden_activations": jax.nn.swish,
        },
        "policy_training": {
            "plan_horizon": 20,
            "n_particles": 30,
            "policy_optimizer": optax.adamw(1e-4),
            "n_policy_train_steps": 2000,
            "policy_train_batch_size": 32
        }
    },
    "HalfCheetah-v3": {
        "discount_factor": 0.99,
        "env_args": {
            "exclude_current_positions_from_observation": False,    # Need access to x_pos to compute velocity
        },
        "preprocessing_functions": {
            "obs_preproc": lambda obs: jnp.concatenate([obs[1:2], jnp.sin(obs[2:3]), jnp.cos(obs[2:3]), obs[3:]]),
            "targ_comp": lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "reward_function": halfcheetah_reward,

        "dynamics": {
            "hidden_dims": [200, 200, 200],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic": True
        },
        "model_training_and_evaluation": {
            "n_model_train_steps": 2000,
            "model_train_batch_size": 32,
            "ensemble_size": 5,
            "dynamics_optimizer": optax.adamw(1e-3),
            "n_model_eval_points": 1000,
        },

        "cem": {
            "n_candidates": 500,
            "n_elites": 50,
            "plan_horizon": 30,
            "n_particles": 15,
            "cem_epsilon": 0.05,
            "max_cem_iters": 5,
        },

        "policy": {},
        "policy_training": {}
    }
}


def rollout(env, discount_factor, agent=None, recording_path=None):
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
            ac = agent.act(observations[-1])

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


def main(
    env_name,
    agent_type,
    logdir=None,
    save_every=1,
    keep_all_checkpoints=False,
    n_init_trajs=1,
    n_iters=100,
    seed=0
):
    env = gym.make(env_name, **CONFIG[env_name]["env_args"])
    env.seed(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    if logdir is not None:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, "agent_rollouts"), exist_ok=True)
        os.makedirs(os.path.join(logdir, "checkpoints"), exist_ok=True)

        logging.getLogger().addHandler(
            logging.FileHandler(os.path.join(logdir, "experiment.log"), mode="w")
        )

    dynamics_model = NeuralNetDynamicsModel(
        dummy_obs=env.reset(),
        dummy_action=env.action_space.sample(),
        **CONFIG[env_name]["dynamics"],
        **CONFIG[env_name]["preprocessing_functions"]
    )

    if agent_type == "PETS":
        agent = ModelPredictiveControlAgent(
            env=env,
            dynamics_model=dynamics_model,
            reward_fn=CONFIG[env_name]["reward_function"],
            **CONFIG[env_name]["model_training_and_evaluation"],
            rng_key=jax.random.PRNGKey(seed),
            **CONFIG[env_name]["cem"],
        )
    elif agent_type == "Policy":
        policy = DeterministicPolicy(
            env=env,
            dummy_obs=env.reset(),
            **CONFIG[env_name]["policy"],
            obs_preproc=CONFIG[env_name]["preprocessing_functions"]["obs_preproc"]
        )
        agent = ModelBasedPolicyAgent(
            env=env,
            dynamics_model=dynamics_model,
            reward_fn=CONFIG[env_name]["reward_function"],
            **CONFIG[env_name]["model_training_and_evaluation"],
            rng_key=jax.random.PRNGKey(seed),
            policy=policy,
            **CONFIG[env_name]["policy_training"]
        )
    else:
        raise RuntimeError("Invalid agent type.")

    for _ in trange(n_init_trajs, ncols=150, desc="Collecting initial trajectories"):
        observations, actions, _ = rollout(env, CONFIG[env_name]["discount_factor"])
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

    all_logging_statistics = None
    for iteration in range(n_iters):
        agent.train()

        if logdir is not None and (iteration + 1) % save_every == 0:
            recording_path = os.path.join(logdir, "agent_rollouts/iter_{}".format(iteration + 1))
        else:
            recording_path = None

        agent.reset()
        observations, actions, rollout_statistics = rollout(
            env,
            CONFIG[env_name]["discount_factor"],
            agent=agent,
            recording_path=recording_path
        )
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

        current_logging_statistics = {**agent.get_logging_statistics(), **rollout_statistics}
        print_logging_statistics(iteration + 1, current_logging_statistics)
        if all_logging_statistics is None:
            all_logging_statistics = {
                name: onp.array([stat])
                for (name, stat) in current_logging_statistics.items()
            }
        else:
            all_logging_statistics = {
                name: onp.append(all_logging_statistics[name], current_logging_statistics[name])
                for (name, stat) in current_logging_statistics.items()
            }

        if logdir is not None and (iteration + 1) % save_every == 0:
            if keep_all_checkpoints:
                checkpoint_path = os.path.join(logdir, "checkpoints/iter_{}.pkl".format(iteration + 1))
            else:
                checkpoint_path = os.path.join(logdir, "checkpoints/checkpoint.pkl")

            with open(checkpoint_path, "wb") as f:
                dill.dump({
                    "agent": agent,
                    "env": env,
                    "iteration": iteration + 1,
                    "seed": seed,
                    "logged_statistics": all_logging_statistics
                }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default=None,
                        help="Path to folder for saving all logs.")
    parser.add_argument("--save-every", type=int, default=1,
                        help="How often agent checkpoints and videos will be saved.")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="If provided, keeps all checkpoints (rather than only the most recent one).")
    parser.add_argument("--n-init-trajs", type=int, default=1,
                        help="Number of initial trajectories collected with random actions.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Number of training iterations.")
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
        n_init_trajs=args.n_init_trajs,
        n_iters=args.iters,
        seed=args.s if args.s != -1 else onp.random.randint(10000)
    )
