import argparse
import importlib
import logging
import os

import dill
import gym
import jax
import jax.numpy as jnp
import numpy as onp
from tqdm import trange

from mbrl.agents import ModelPredictiveControlAgent, ModelBasedPolicyAgent
from mbrl.misc import NeuralNetDynamicsModel
from mbrl.policies import DeterministicPolicy
from mbrl._src.utils import rollout, print_logging_statistics
import mbrl.envs


env_to_config = {
    "MujocoCartpole-v0": "cartpole",
    "HalfCheetah-v3": "halfcheetah"
}


def main(
    env_name,
    agent_type,
    logdir=None,
    save_every=1,
    keep_all_checkpoints=False,
    n_init_trajs=1,
    n_eval_runs=1,
    n_iters=100,
    seed=0
):
    config = importlib.import_module("." + env_to_config[env_name], "mbrl.config").create_config()

    env = gym.make(env_name, **config["env_args"])
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
        **config["dynamics"],
        **config["preprocessing_functions"]
    )

    if agent_type == "PETS":
        agent = ModelPredictiveControlAgent(
            env=env,
            dynamics_model=dynamics_model,
            reward_fn=config["reward_function"],
            **config["model_training_and_evaluation"],
            rng_key=jax.random.PRNGKey(seed),
            **config["cem"],
        )
    elif agent_type == "Policy":
        policy = DeterministicPolicy(
            env=env,
            dummy_obs=env.reset(),
            **config["policy"],
            obs_preproc=config["preprocessing_functions"]["obs_preproc"]
        )
        agent = ModelBasedPolicyAgent(
            env=env,
            dynamics_model=dynamics_model,
            reward_fn=config["reward_function"],
            **config["model_training_and_evaluation"],
            rng_key=jax.random.PRNGKey(seed),
            policy=policy,
            **config["policy_training"]
        )
    else:
        raise RuntimeError("Invalid agent type.")

    for _ in trange(n_init_trajs, ncols=150, desc="Collecting initial trajectories"):
        observations, actions, _ = rollout(env, config["discount_factor"])
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

    all_logging_statistics, all_miscellaneous_data = None, []
    for iteration in range(n_iters):
        agent.train()

        if logdir is not None and (iteration + 1) % save_every == 0:
            recording_path = os.path.join(logdir, "agent_rollouts/iter_{}".format(iteration + 1))
        else:
            recording_path = None

        aggregate_statistics = None
        for _ in range(n_eval_runs):
            agent.reset()
            _, _, rollout_statistics = rollout(
                env,
                config["discount_factor"],
                agent=agent,
                recording_path=recording_path,
                evaluation=True
            )

            if aggregate_statistics is None:
                aggregate_statistics = jax.tree_map(lambda x: jnp.array([x]), rollout_statistics)
            else:
                aggregate_statistics = jax.tree_map(
                    lambda x, y: jnp.concatenate([x, jnp.array([y])]),
                    aggregate_statistics,
                    rollout_statistics
                )
        mean_eval_statistics = jax.tree_map(lambda x: jnp.mean(x), aggregate_statistics)

        agent.reset()
        observations, actions, rollout_statistics = rollout(
            env,
            config["discount_factor"],
            agent=agent,
            recording_path=recording_path
        )
        agent.add_interaction_data(jnp.array(observations), jnp.array(actions))

        current_logging_statistics = {**agent.get_logging_statistics(), **mean_eval_statistics}
        all_miscellaneous_data.append(agent.get_miscellaneous_data())
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
                    "logged_statistics": all_logging_statistics,
                    "miscellaneous_data": all_miscellaneous_data
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
    parser.add_argument("--n-eval-runs", type=int, default=1,
                        help="Number of rollouts used to evaluate the agent after every iteration.")
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
        n_eval_runs=args.n_eval_runs,
        seed=args.s if args.s != -1 else onp.random.randint(10000)
    )
