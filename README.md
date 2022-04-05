# Preproject

My preproject for my M.Sc. degree in Engineering Cybernetics at NTNU.

The preproject work comprises multiple repositories:
- [`krisbrud/gym-auv`](https://github.com/krisbrud/gym-auv):
    - A forked and refactored version of [simentha's](https://github.com/simentha/gym-auv) custom OpenAI gym environment [`gym-auv`](https://github.com/simentha/gym-auv)
- [`krisbrud/assisted-baselines`](https://github.com/krisbrud/assisted-baselines):
    - An extension of [`DLR-RM/stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) that supports for training of reinforcement learning agents through the help of an assistant.
- [`krisbrud/preproject`](https://github.com/krisbrud/preproject):
    - This repository. Contains configuration and training scripts for running experiments. Experiments may be run locally or in Azure ML, while experiments, their results and artifacts may be tracked using [MLFlow Tracking](https://www.mlflow.org/docs/latest/tracking.html). The repository depends on the afforementioned repositories, but `krisbrud/gym-auv` and `krisbrud/assisted-baselines` are independent of eachother, although both use [`openai/gym`](https://github.com/openai/gym) as a common interface for environments.

TODO Write about symlinks when developing