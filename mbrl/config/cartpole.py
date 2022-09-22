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
            "hidden_dims"       : [200, 200, 200],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic"  : True
        },
        "model_training_and_evaluation": {
            "ensemble_size"         : 7,
            "dynamics_optimizer"    : optax.adamw(1e-3, weight_decay=1e-5, eps=1e-7),
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
        },

        "MBPO_policy"                  : {
            "hidden_dims"       : [256, 256],
            "hidden_activations": jax.nn.swish,
            "min_stddev"        : 1e-5,
            "max_stddev"        : 4.
        },
        "MBPO_critic"                  : {
            "hidden_dims"       : [256, 256],
            "hidden_activations": jax.nn.swish,
        },
        "MBPO_training"                : {
            "discount_factor"             : 0.99,
            "max_replay_buffer_length"    : 100 * 200,
            "include_augmentations"       : True,
            "augmentation_horizon"        : 1,
            "augmentation_batch_size"     : int(1e5),
            "augmentation_keep_epochs"    : 2,
            "real_ratio"                  : 0.05,
            "initial_temperature"         : 1e-2,
            "target_smoothing_coefficient": 5e-3,
            "sac_batch_size"              : 256,
            "sac_optimizer"               : optax.adam(3e-4, eps=1e-7),   # optax.adamw(3e-4)
            "n_steps_per_batch"           : 20,
            "target_entropy"              : -1.
        }
    }