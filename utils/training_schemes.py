import dataclasses
import os
import time
import pprint

import gym
import stable_baselines3
# import gym_auv

import decoupled_ppo

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils.azuremlutils import get_mlflow_uri_from_aml_workspace_config

from utils.parsing import ExperimentConfig
from utils.trackers.mlflow_tracker import MLFlowTracker
import utils.utils

from .callbacks import SaveAndTrackCallback, SaveCallback

scenarios = ["beginner", "intermediate", "proficient", "advanced", "expert"]


def train_end_to_end_path_follow(hyperparams,
                                 experiment_config: ExperimentConfig,
                                 env_name="PathFollowAuv3D-v0"):

    agents_dir = experiment_config.output_path  # os.path.join(experiment_config.output_path, "agents")
    tensorboard_dir = os.path.join(experiment_config.log_path, "tensorboard")
    os.makedirs(agents_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    hyperparams["tensorboard_log"] = tensorboard_dir

    if experiment_config.mlflow_tracking_uri is None:
        # Try to look for AzureML workspace configuration in working directory and deduce MLFLow URI from there
        experiment_config.mlflow_tracking_uri = get_mlflow_uri_from_aml_workspace_config()

    num_envs = experiment_config.num_envs
    print("Num envs:", num_envs)

    if num_envs > 1:
        env = SubprocVecEnv([
            lambda: Monitor(gym.make(env_name),
                            agents_dir,
                            allow_early_resets=True) for i in range(num_envs)
        ])
    else:
        # Only one env
        env = DummyVecEnv([
            lambda: Monitor(gym.make(env_name),
                            agents_dir,
                            allow_early_resets=True)
        ])

    print(f"env action space {env.action_space}")

    # Initialize a new agent from scratch
    agent = PPO("MlpPolicy", env, **hyperparams)

    timesteps = experiment_config.timesteps

    # Save agent periodically
    # Number of environment steps summed across multiple envs
    save_freq_target = int(100e3)
    save_freq = max(int(save_freq_target) // num_envs, 1)

    # Instantiate a MLFlow Tracker if URI is provided
    if experiment_config.mlflow_tracking_uri is not None:
        tracker = MLFlowTracker(
            mlflow_tracking_uri=experiment_config.mlflow_tracking_uri,
            experiment_name=experiment_config.exp_id)
        
        # Log hyperparameters and experiment_config
        tracker.log_param("hyperparams", pprint.pformat(hyperparams))
        tracker.log_param("experiment_config", pprint.pformat(dataclasses.asdict(experiment_config)))
        tracker.log_param("environment_config", pprint.pformat(gym_auv.pid_auv3d_config))
    else:
        tracker = None

    save_and_track_callback = SaveAndTrackCallback(experiment_config.output_path, tracker=tracker, save_freq=save_freq)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=experiment_config.output_path,
        name_prefix="model")

    tic = time.perf_counter()
    agent.learn(total_timesteps=timesteps,
                tb_log_name="PPO",
                callback=save_and_track_callback)
    toc = time.perf_counter()

    save_path = os.path.join(experiment_config.output_path, "last_model.pkl")
    agent.save(save_path)


    # Evaluate policy and log metrics
    mean_eval_reward, mean_eval_reward_std = evaluate_policy(agent, env, n_eval_episodes=experiment_config.n_eval_episodes)


    # Find the tensorboard log paths, as the filename is somewhat unpredictable
    tb_logs = utils.utils.find_tb_logs(experiment_config.log_path)
    if len(tb_logs) > 1:
        print(f"Warning: {len(tb_logs)} tb_logs, not just one!")
    
    if tracker:
        tracker.log_metric("mean_eval_reward", mean_eval_reward)
        tracker.log_metric("mean_eval_reward_std", mean_eval_reward_std)
        
        for tb_log in tb_logs:
            tracker.log_artifact(tb_log)


    print(f"Using {num_envs} environments")
    print(f"Average total fps {timesteps / (toc - tic):0.2f})")
    print(f"Trained {timesteps} in {toc - tic:0.2f} seconds")
    print(
        f"Average: {timesteps / ((toc - tic) * num_envs):0.1f} timesteps per second per environment"
    )
