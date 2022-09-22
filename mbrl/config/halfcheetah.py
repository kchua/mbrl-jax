import jax
import jax.numpy as jnp
import optax


def create_config():
    def halfcheetah_reward(obs, action, next_obs):
        obs_reward = (next_obs[0] - obs[0]) / 0.05     # TODO: 0.05 is the dt of the environment, should set automatically
        action_reward = -0.1 * jnp.sum(jnp.square(action))
        return obs_reward + action_reward

    return {
        "discount_factor"              : 0.99,
        "env_args"                     : {
            "exclude_current_positions_from_observation": False,  # Need access to x_pos to compute velocity
        },
        "preprocessing_functions"      : {
            "obs_preproc"  : lambda obs: jnp.concatenate([obs[1:2], jnp.sin(obs[2:3]), jnp.cos(obs[2:3]), obs[3:]]),
            "targ_comp"    : lambda obs, next_obs: next_obs - obs,
            "next_obs_comp": lambda obs, pred: obs + pred
        },
        "reward_function"              : halfcheetah_reward,

        "dynamics"                     : {
            "hidden_dims"       : [200, 200, 200, 200],
            "hidden_activations": jax.nn.swish,
            "is_probabilistic"  : True
        },
        "model_training_and_evaluation": {
            "n_model_train_steps"   : 2000,
            "model_train_batch_size": 32,
            "ensemble_size"         : 7,
            "dynamics_optimizer"    : optax.adamw(1e-3, weight_decay=1e-5, eps=1e-7),
        },

        "cem"                          : {
            "n_candidates" : 500,
            "n_elites"     : 50,
            "plan_horizon" : 30,
            "n_particles"  : 15,
            "cem_epsilon"  : 0.05,
            "max_cem_iters": 5,
        },

        "policy"                       : {},
        "policy_training"              : {},

        "MBPO_policy"  : {
            "hidden_dims"       : [256, 256],
            "hidden_activations": jax.nn.swish,
            "min_stddev"        : 1e-5,
            "max_stddev"        : 4.
        },
        "MBPO_critic"  : {
            "hidden_dims"       : [256, 256],
            "hidden_activations": jax.nn.swish,
        },
        "MBPO_training": {
            "discount_factor"             : 0.99,
            "max_replay_buffer_length"    : 100 * 1000,
            "include_augmentations"       : True,
            "augmentation_horizon"        : 1,
            "augmentation_batch_size"     : int(1e5),
            "augmentation_keep_epochs"    : 10,
            "real_ratio"                  : 0.05,
            "initial_temperature"         : 1e-2,
            "target_smoothing_coefficient": 5e-3,
            "sac_batch_size"              : 256,
            "sac_optimizer"               : optax.adam(3e-4, eps=1e-7),   # optax.adamw(3e-4)
            "n_steps_per_batch"           : 40
        }
    }
