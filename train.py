# import matplotlib  # Try to fix absurd bug due to this not being imported
# matplotlib.use("TKAgg")
import dataclasses

# import matplotlib as plt
import torch
from assisted_baselines.common.assistant import AssistantWrapper
from assisted_baselines.common.mask import ActiveActionsMask, BaseMaskSchedule
from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule

# matplotlib import fixes absurd bug where stable_baselines3 imports
# matplotlib in a way that causes an error


from config import get_config, Config

import os
import time

import gym
import gym_auv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils.assisted_env_factory import get_assisted_envs, get_eval_env, get_normal_envs

# from utils.azuremlutils import get_mlflow_uri_from_aml_workspace_config

from utils.parsing import ExperimentConfig
from utils.trackers.mlflow_tracker import MLFlowTracker
import utils.utils
from yacs.config import CfgNode

from assisted_baselines.PPO.assisted_ppo import AssistedPPO

# from assisted_baselines.common.assistants.pid import (
#     PIDController,
#     PIDAssistant,
#     PIDGains,
# )
from utils.callbacks import SaveAndTrackCallback, SaveCallback
from utils.auv_masks import mask_assistant_only


# def get_mask_schedule(cfg) -> BaseMaskSchedule:
#     if cfg.assistance.checkpoints:
#         # Make dictionary from list of tuples
#         n_actions = cfg.env.n_actions

#         checkpoints = dict()
#         for timestep, mask in cfg.assistance.checkpoints:
#             checkpoints[timestep] = ActiveActionsMask(
#                 n_actions=n_actions, mask=torch.BoolTensor(mask)
#             )

#         schedule = CheckpointSchedule(
#             checkpoints=checkpoints, total_timesteps=cfg.train.total_timesteps
#         )
#         return schedule


def baseline_env_agent(cfg: Config):
    # Train a normal PPO agent from `stable_baselines3`, in order to compare the
    # overhead of the AssistedPPO algorithm
    if cfg.train.num_envs > 1:
        env = SubprocVecEnv(
            [
                lambda: Monitor(
                    gym.make(cfg.env.name),
                    cfg.system.output_path,
                    allow_early_resets=True,
                )
                for i in range(cfg.train.num_envs)
            ]
        )
    else:
        # Only one env
        env = DummyVecEnv(
            [
                lambda: Monitor(
                    gym.make(cfg.env.name),
                    cfg.system.output_path,
                    allow_early_resets=True,
                )
            ]
        )

    hyperparams = dataclasses.asdict(cfg.hyperparam)
    agent = PPO("MlpPolicy", env, **hyperparams)

    return env, agent


def train():
    cfg = get_config()

    agents_dir = cfg.system.output_path  # os.path.join(cfg.output_path, "agents")
    tensorboard_dir = cfg.system.tensorboard_dir
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    hyperparams = dataclasses.asdict(cfg.hyperparam)
    hyperparams["tensorboard_log"] = tensorboard_dir
    if hyperparams["policy_kwargs"] is None:
        # Don't pass optional parameter as None
        del hyperparams["policy_kwargs"]

    # if cfg.mlflow_tracking_uri is None:
    #     # Try to look for AzureML workspace configuration in working directory and deduce MLFLow URI from there
    #     cfg.mlflow_tracking_uri = get_mlflow_uri_from_aml_workspace_config()

    num_envs = cfg.train.num_envs
    print("Num envs:", num_envs)

    # Initialize a new agent from scratch
    # agent = PPO("MlpPolicy", env, **hyperparams)

    if cfg.train.algorithm == "AssistedPPO":
        env = get_assisted_envs(cfg)
        mask_schedule = cfg.assistance.mask_schedule
    else:
        env = get_normal_envs(cfg)

    # print(f"env action space {env.action_space}")

    # env, agent = baseline_env_agent(cfg)

    # Save agent periodically
    # Number of environment steps summed across multiple envs
    save_freq_target = int(100e3)
    save_freq = max(int(save_freq_target) // num_envs, 1)

    # Instantiate a MLFlow Tracker if URI is provided
    if cfg.train.mlflow_tracking_uri is not None:
        tracker = MLFlowTracker(
            mlflow_tracking_uri=cfg.train.mlflow_tracking_uri,
            experiment_name=cfg.experiment.name,
        )

        # Log hyperparameters and cfg
        tracker.log_params(hyperparams, prefix="hyperparams")

        # tracker.log_params(dataclasses.asdict(cfg.assistance), cfg.assistance.auv_pid)
        tracker.log_params(dataclasses.asdict(cfg.env), prefix="env")
        tracker.log_params(dataclasses.asdict(cfg.experiment), prefix="experiment")
        tracker.log_params(dataclasses.asdict(cfg.system), prefix="system")
        train_dict = dataclasses.asdict(cfg.train)
        train_dict.pop(
            "mlflow_tracking_uri"
        )  # Too long to log in mlflow?. Also logged implicitly?
        tracker.log_params(train_dict, prefix="train")
        # Log PID gains:
        tracker.log_params(
            cfg.assistance.auv_pid.elevator._asdict(),
            prefix="cfg.assistance.auv_pid.elevator",
        )
        tracker.log_params(
            cfg.assistance.auv_pid.rudder._asdict(),
            prefix="cfg.assistance.auv_pid.rudder",
        )
        tracker.log_params(
            cfg.assistance.auv_pid.surge._asdict(),
            prefix="cfg.assistance.auv_pid.surge",
        )

        # tracker.log_param("cfg", pprint.pformat(dataclasses.asdict(cfg)))
        # tracker.log_param("environment_config", pprint.pformat(gym_auv.pid_auv3d_config))
    else:
        tracker = None

    print(
        "cfg assistant available probability",
        cfg.assistance.assistant_available_probability,
    )

    if cfg.train.algorithm == "AssistedPPO":
        print("Using AssistedPPO")
        agent = AssistedPPO(
            "AssistedPolicy",
            env,
            mask_schedule,
            tracker=tracker,
            assistant_available_probability=cfg.assistance.assistant_available_probability,
            learn_from_assistant_actions=cfg.train.learn_from_assistant_actions,
            assistant_action_noise_std=cfg.assistance.assistant_action_noise_std,
            **hyperparams,
        )
    elif cfg.train.algorithm == "PPO":
        print("Using normal PPO")
        agent = PPO("MlpPolicy", env, **hyperparams)

    save_and_track_callback = SaveAndTrackCallback(
        cfg.system.output_path, tracker=tracker, save_freq=save_freq
    )
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=save_freq, save_path=cfg.system.output_path, name_prefix="model"
    # )

    print("Total timesteps:", cfg.train.total_timesteps)
    tic = time.perf_counter()
    agent.learn(
        total_timesteps=cfg.train.total_timesteps,
        tb_log_name=cfg.experiment.name,  # "PPO",
        callback=save_and_track_callback,
    )
    toc = time.perf_counter()

    save_path = os.path.join(cfg.system.output_path, "last_model.pkl")
    agent.save(save_path)

    # Evaluate policy and log metrics
    print("Evaluating performance!")
    mean_eval_reward, mean_eval_reward_std = evaluate_policy(
        agent, env, n_eval_episodes=cfg.train.n_eval_episodes
    )

    # Find the tensorboard log paths, as the filename is somewhat unpredictable
    tb_logs = utils.utils.find_tb_logs(cfg.system.log_path)
    if len(tb_logs) > 1:
        print(f"Warning: {len(tb_logs)} tb_logs, not just one!")

    if tracker:
        tracker.log_metric("mean_eval_reward", mean_eval_reward)
        tracker.log_metric("mean_eval_reward_std", mean_eval_reward_std)

        for tb_log in tb_logs:
            tracker.log_artifact(tb_log)

    print(f"Using {num_envs} environments")
    timesteps = cfg.train.total_timesteps
    print(f"Average total fps {timesteps / (toc - tic):0.2f})")
    print(f"Trained {timesteps} in {toc - tic:0.2f} seconds")
    print(
        f"Average: {timesteps / ((toc - tic) * num_envs):0.1f} timesteps per second per environment"
    )


if __name__ == "__main__":
    train()
