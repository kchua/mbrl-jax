# `mbrl-jax`

A library intended for running model-based RL experiments, written with JAX.
Currently only includes a reimplementation of [PETS](https://arxiv.org/abs/1805.12114).
Other algorithm implementations are planned soon!

**Warning**: This is a work-in-progress, and has not been evaluated on harder environments! Please let me know if you find any bugs.

### TODOs:
1. Evaluate on harder environments (e.g. HalfCheetah, Ant).
2. Implement other more recent model-based algorithms.

## Installing Dependencies

A `Dockerfile` with all required dependencies is provided in the `/docker/` folder, together with an accompanying `docker-compose.yml` file.
Remember to include the appropriate mounts in the docker-compose file as necessary for your needs!

## Running Experiments

A starter script for running an example experiment on cartpole is provided in `model_based_experiment.py`.
This script can be run via

```
  python3 model_based_experiment.py
      --record-dir   DIR       (optional)   Directory for saving rollouts. If not provided,
                                            recordings will not be saved.
      -s             SEED      (optional)   Experiment random seed. If not provided,
                                            randomly chosen in [0, 10000).
      env            ENV       (required)   Experiment environment. Currently only supports
                                            `MujocoCartpole-v0`.
      agent_type     AGENT     (required)   Agent type. Choices: [PETS, Policy].
```

For example, to run PETS and save recordings of rollouts to `/external/`:

```
python3 model_based_experiment.py --record-dir /external/ MujocoCartpole-v0 PETS
```

Note: The policy agent is a naive implementation of a policy-based learner, and is only provided to illustrate the ways in which one can extend the `DeepModelBasedAgent` class in `/mbrl/agents/model_based_agent.py` (in addition to the random shooting-based PETS learner.)
