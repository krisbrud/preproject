import cProfile
import matplotlib as plt
import torch
from assisted_baselines.common.assistant import AssistantWrapper
from assisted_baselines.common.mask import ActiveActionsMask, BaseMaskSchedule
from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule
# matplotlib import fixes absurd bug where stable_baselines3 imports
# matplotlib in a way that causes an error


from config import get_cfg_defaults

import os
import time
import pprint


import gym
import gym_auv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from utils.azuremlutils import get_mlflow_uri_from_aml_workspace_config

from utils.parsing import ExperimentConfig
from utils.trackers.mlflow_tracker import MLFlowTracker
import utils.utils
from yacs.config import CfgNode

from assisted_baselines.PPO.assisted_ppo import AssistedPPO
from assisted_baselines.common.assistants.pid import PIDController, PIDAssistant, PIDGains
from utils.callbacks import SaveAndTrackCallback, SaveCallback

def get_config(verbose=True):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join("configs", "first-experiment.yaml")) # Needs to be updated here if using different experiment config
    cfg.freeze()

    if verbose:
        print("Config:\n", cfg)
    
    return cfg

def get_assisted_envs(cfg):
    def get_gains(cfg_node: CfgNode):
        pid_gains = PIDGains(
            Kp=cfg_node.Kp,
            Ki=cfg_node.Ki,
            Kd=cfg_node.Kd
        )

        return pid_gains

    def make_pid_controller(gains: PIDGains, timestep: float):
        pid_controller = PIDController(
            Kp=gains.Kp,
            Ki=gains.Ki,
            Kd=gains.Kd, 
            timestep=timestep)

        return pid_controller


    def make_pid_assistant(cfg):
        timestep = 0.1 # cfg.timestep
        
        surge_gains = get_gains(cfg.assistance.pid.surge)
        surge_pid = make_pid_controller(surge_gains, timestep)

        rudder_gains = get_gains(cfg.assistance.pid.rudder)
        rudder_pid = make_pid_controller(rudder_gains, timestep)
        
        elevator_gains = get_gains(cfg.assistance.pid.elevator)
        elevator_pid = make_pid_controller(elevator_gains, timestep)

        pid_assistant = PIDAssistant(pid_controllers=[surge_pid, rudder_pid, elevator_pid], pid_error_indices=[9, 10, 11], n_actions=3)
        return pid_assistant

    if cfg.train.num_envs > 1:
        env = SubprocVecEnv([
            lambda: Monitor(AssistantWrapper(gym.make(cfg.env.name), make_pid_assistant(cfg)),
                            cfg.system.output_path,
                            allow_early_resets=True) for i in range(cfg.train.num_envs)
        ])
    else:
        # Only one env
        env = DummyVecEnv([
            lambda: Monitor(AssistantWrapper(gym.make(cfg.env.name), make_pid_assistant(cfg)),
                            cfg.system.output_path,
                            allow_early_resets=True)
        ])
    
    return env

def get_mask_schedule(cfg) -> BaseMaskSchedule:
    if cfg.assistance.checkpoints:
        # Make dictionary from list of tuples
        n_actions = cfg.env.n_actions

        checkpoints = dict()
        for timestep, mask in cfg.assistance.checkpoints:
            checkpoints[timestep] = ActiveActionsMask(n_actions=n_actions, mask =torch.BoolTensor(mask))

        schedule = CheckpointSchedule(
            checkpoints=checkpoints,
            total_timesteps=cfg.train.total_timesteps
        )

        return schedule

def baseline_env_agent(cfg):
    # Train a normal PPO agent from `stable_baselines3`, in order to compare the
    # overhead of the AssistedPPO algorithm
    if cfg.train.num_envs > 1:
        env = SubprocVecEnv([
            lambda: Monitor(gym.make(cfg.env.name),
                            cfg.system.output_path,
                            allow_early_resets=True) for i in range(cfg.train.num_envs)
        ])
    else:
        # Only one env
        env = DummyVecEnv([
            lambda: Monitor(gym.make(cfg.env.name),
                            cfg.system.output_path,
                            allow_early_resets=True)
        ])
    
    hyperparams = cfg.hyperparam
    agent = PPO("MlpPolicy", env, **hyperparams)

    return env, agent

def train():
    cfg = get_config()

    agents_dir = cfg.system.output_path  # os.path.join(cfg.output_path, "agents")
    os.makedirs(agents_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(cfg.system.log_path, "tensorboard")
    hyperparams = cfg.hyperparam
    os.makedirs(tensorboard_dir, exist_ok=True)
    hyperparams["tensorboard_log"] = tensorboard_dir


    # if cfg.mlflow_tracking_uri is None:
    #     # Try to look for AzureML workspace configuration in working directory and deduce MLFLow URI from there
    #     cfg.mlflow_tracking_uri = get_mlflow_uri_from_aml_workspace_config()

    num_envs = cfg.train.num_envs
    print("Num envs:", num_envs)

    env = get_assisted_envs(cfg)

    # if num_envs > 1:
    #     env = SubprocVecEnv([
    #         lambda: Monitor(gym.make(cfg.env.name),
    #                         agents_dir,
    #                         allow_early_resets=True) for i in range(num_envs)
    #     ])
    # else:
    #     # Only one env
    #     env = DummyVecEnv([
    #         lambda: Monitor(gym.make(cfg.env.name),
    #                         agents_dir,
    #                         allow_early_resets=True)
    #     ])

    print(f"env action space {env.action_space}")

    # Initialize a new agent from scratch
    # agent = PPO("MlpPolicy", env, **hyperparams)

    mask_schedule = get_mask_schedule(cfg)
    
    # agent = AssistedPPO("AssistedPolicy", env, mask_schedule, **hyperparams)

    env, agent = baseline_env_agent(cfg)

    timesteps = cfg.train.total_timesteps

    # Save agent periodically
    # Number of environment steps summed across multiple envs
    save_freq_target = int(100e3)
    save_freq = max(int(save_freq_target) // num_envs, 1)

    # Instantiate a MLFlow Tracker if URI is provided
    if cfg.train.mlflow_tracking_uri is not None:
        tracker = MLFlowTracker(
            mlflow_tracking_uri=cfg.train.mlflow_tracking_uri,
            experiment_name=cfg.experiment.name)
        
        # Log hyperparameters and cfg
        tracker.log_param("hyperparams", pprint.pformat(hyperparams))
        # tracker.log_param("cfg", pprint.pformat(dataclasses.asdict(cfg)))
        # tracker.log_param("environment_config", pprint.pformat(gym_auv.pid_auv3d_config))
    else:
        tracker = None

    save_and_track_callback = SaveAndTrackCallback(cfg.system.output_path, tracker=tracker, save_freq=save_freq)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=cfg.system.output_path,
        name_prefix="model")

    tic = time.perf_counter()
    agent.learn(total_timesteps=timesteps,
                tb_log_name="PPO",
                callback=save_and_track_callback)
    toc = time.perf_counter()

    # save_path = os.path.join(cfg.system.output_path, "last_model.pkl")
    # agent.save(save_path)


    # Evaluate policy and log metrics
    mean_eval_reward, mean_eval_reward_std = evaluate_policy(agent, env, n_eval_episodes=cfg.n_eval_episodes)


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
    print(f"Average total fps {timesteps / (toc - tic):0.2f})")
    print(f"Trained {timesteps} in {toc - tic:0.2f} seconds")
    print(
        f"Average: {timesteps / ((toc - tic) * num_envs):0.1f} timesteps per second per environment"
    )

if __name__ == '__main__':
    train()