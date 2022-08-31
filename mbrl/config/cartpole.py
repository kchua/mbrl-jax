import jax
import jax.numpy as jnp
import optax


def create_config():
    def cartpole_reward(obs, action, _next_obs):
        obs_reward = jnp.exp(-jnp.sum(jnp.square(
            jnp.array([obs[0] + 0.6 * jnp.sin(obs[1]), 0.6 * jnp.cos(obs[1])]) - jnp.array([0, 0.6])
        )))
        action_reward = -0.01 * jnp.sum(jnp.square(action))
        return obs_reward + action_reward


    return {
        "discount_factor"              : 0.99,
        "env_args"                     : {},
        "preprocessing_functions"      : {
            "obs_preproc"  : lambda obs: jnp.concatenate([obs[:1], jnp.sin(obs[1:2]), jnp.cos(obs[1:2]), obs[2:]]),
            "targ_comp"    : lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "reward_function"              : cartpole_reward,

        "dynamics"                     : {
            "hidden_dims"       : [50, 50, 50],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic"  : True
        },
        "model_training_and_evaluation": {
            "ensemble_size"         : 10,
            "dynamics_optimizer"    : optax.adamw(1e-3),
            "n_model_train_steps"   : 2000,
            "model_train_batch_size": 32,
        },

        "cem"                          : {
            "n_candidates" : 400,
            "n_elites"     : 40,
            "plan_horizon" : 20,
            "n_particles"  : 30,
            "cem_epsilon"  : 0.05,
            "max_cem_iters": 10,
        },

        "policy"                       : {
            "hidden_dims"       : [50, 50, 50],
            "hidden_activations": jax.nn.swish,
        },
        "policy_training"              : {
            "plan_horizon"           : 30,
            "n_particles"            : 30,
            "policy_optimizer"       : optax.adamw(1e-4),
            "n_policy_train_steps"   : 2000,
            "policy_train_batch_size": 32
        }
    }