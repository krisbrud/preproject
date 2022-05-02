# my_project/config.py
import argparse
import dataclasses
import os

from dataclasses import dataclass

from pytablewriter import Bool

from assisted_baselines.common.assistants.pid import PIDGains
from assisted_baselines.common.mask import BaseMaskSchedule
from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule
from utils.auv_masks import (
    mask_rudder_and_elevator,
    mask_rudder_only,
    mask_elevator_only,
)


@dataclass
class ExperimentConfig:
    name: str = "example-experiment-name"


@dataclass
class SystemConfig:
    # Log path. Required to be "./logs" by AzureML
    log_path: str = os.path.join(os.curdir, "logs")
    # Path for outputs. Required to be "./outputs" by AzureML for saving of artifacts
    output_path: str = os.path.join(os.curdir, "outputs")

    def __post_init__(self):
        self.tensorboard_dir = os.path.join(self.output_path, "tensorboard")


@dataclass
class HyperparamConfig:
    # Also see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
    # for more details
    n_steps: int = 1024
    learning_rate: float = 5e-4  # 5e-5
    batch_size: int = 1024
    gae_lambda: float = 0.95
    gamma: float = 0.999  # Discount factor
    n_epochs = 4  # Number of epochs per rollout
    clip_range: float = 0.2  # Clip range for PPO objective function
    ent_coef: float = 0.01  # Coefficient for entropy loss
    verbose: int = 2


@dataclass
class TrainConfig:
    # Number of parallel environments to use with SubProcVecEnv
    num_envs: int = 10  # Test if this makes a difference on total timesteps
    # Total timesteps to run training
    total_timesteps: int = int(30e6)  # int(100e3)
    # How many timesteps between each time agent is saved to disk and MLFlow
    save_freq: int = int(100e3)  # int(100e3)
    # MLFlow Tracking URI for logging metrics and artifacts.
    # Set to None if it's not going to be used.
    mlflow_tracking_uri: str = "azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/3165a1c1-fd45-4c8d-938e-0058c823f960/resourceGroups/aml-playground/providers/Microsoft.MachineLearningServices/workspaces/aml-playground"
    # RL Algorithm to use. Currently supports "AssistedPPO", which is implmented in `krisbrud/assisted-baselines`.
    algorithm: str = "AssistedPPO"
    # How many evaluation episodes to run when evaluating the environment
    n_eval_episodes: int = 100
    # Whether actions taken by the assistant should be used when optimizing the policy during training
    learn_from_assistant_actions: bool = False


@dataclass
class AuvPidConfig:
    # Gains for PID controllers for assistant
    # surge = PIDGains(Kp=2, Ki=1.5, Kd=0)
    # rudder = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)
    # elevator = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)
    surge = PIDGains(Kp=1, Ki=0.5, Kd=0)
    rudder = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)
    elevator = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)


@dataclass
class AssistanceConfig:
    mask_schedule: BaseMaskSchedule
    # Define parameters for PID controllers
    auv_pid: AuvPidConfig = AuvPidConfig()
    assistant_available_probability = 0.2
    assistant_action_noise_std = 0.1


@dataclass
class EnvConfig:
    # Class for configuring the OpenAI gym env
    # Name of gym environment to look up
    name: str = "PathFollowAuv3D-v0"
    # Number of actuators in action space
    n_actions: int = 3


@dataclass
class Config:
    experiment: ExperimentConfig
    system: SystemConfig
    hyperparam: HyperparamConfig
    train: TrainConfig
    assistance: AssistanceConfig
    env: EnvConfig


def _get_default_config() -> Config:
    # Since some hyperparameters and instantiations may depend on others, we do it
    # this way
    experiment = ExperimentConfig()
    system = SystemConfig()
    hyperparam = HyperparamConfig()
    train = TrainConfig()
    env = EnvConfig()

    # AssistanceConfig depends on the others, instantiate last
    assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_and_elevator}, total_timesteps=train.total_timesteps
        )
    )

    config = Config(
        experiment=experiment,
        system=system,
        hyperparam=hyperparam,
        train=train,
        assistance=assistance,
        env=env,
    )
    return config


def train_rudder_and_elevator_config():
    cfg = _get_default_config()
    cfg.experiment.name = "train-rudder-elevator"
    cfg.train.total_timesteps = int(30e6)
    return cfg


def train_rudder_config():
    cfg = _get_default_config()
    # Train only rudder while assisting the others according to
    # simentha's paper draft
    cfg.experiment.name = "train-rudder"
    cfg.train.total_timesteps = int(30e6)
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def train_rudder_then_elevator_config():
    cfg = _get_default_config()
    # Train rudder then switch to elevator after 1.5 timesteps
    cfg.experiment.name = "train-rudder-then-elevator-3M"
    cfg.train.total_timesteps = int(3e6)
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {
                0: mask_rudder_only,
                1500000: mask_elevator_only,  # Switch after 1.5M to show problems with value function
            },
            total_timesteps=cfg.train.total_timesteps,
        )
    )

    return cfg


def train_elevator_config():
    # Train only elevator according to Simentha's paper draft
    cfg = _get_default_config()
    cfg.experiment.name = "train-elevator"
    cfg.train.total_timesteps = int(30e6)
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def train_all_1m_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "train-all-1m"
    cfg.train.total_timesteps = int(1e6)
    cfg.train.num_envs = (
        10  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def higher_assistant_prob_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "higher-assistant-prob-1m"
    cfg.assistance.assistant_available_probability = 0.5
    cfg.train.total_timesteps = int(1e6)
    cfg.train.num_envs = (
        10  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def learn_from_assistant_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "learn-from-assistant"
    cfg.train.total_timesteps = int(1e6)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.train.learn_from_assistant_actions = True

    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def train_all_5m_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "train-all-5m"
    cfg.train.total_timesteps = int(5e6)
    cfg.train.num_envs = (
        10  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def debug_config() -> Config:
    # Train only elevator according to Simentha's paper draft
    cfg = _get_default_config()
    cfg.experiment.name = "debug-config"
    cfg.train.total_timesteps = int(10e3)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.mlflow_tracking_uri = None
    cfg.train.n_eval_episodes = (
        1  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def debug_with_mlflow_config() -> Config:
    # Train only elevator according to Simentha's paper draft
    cfg = _get_default_config()
    cfg.experiment.name = "debug-with-mlflow"
    cfg.train.total_timesteps = int(1e3)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        1  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def debug_colav_config() -> Config:
    # Train only elevator according to Simentha's paper draft
    cfg = _get_default_config()
    cfg.env.name = "PathColavAuv3D-v0"
    cfg.experiment.name = "debug-colav"
    cfg.train.total_timesteps = int(10e3)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    # cfg.train.mlflow_tracking_uri = None
    cfg.train.n_eval_episodes = (
        1  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def debug_colav2_config() -> Config:
    # Train only elevator according to Simentha's paper draft
    cfg = _get_default_config()
    cfg.env.name = "PathColavAuv3D-v0"
    cfg.experiment.name = "debug-colav2"
    cfg.train.total_timesteps = int(50e3)
    cfg.train.num_envs = 10
    cfg.train.n_eval_episodes = (
        1  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    cfg.assistance.assistant_available_probability = 0.5
    return cfg


def colav_10m_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "colav-10m"
    cfg.env.name = "PathColavAuv3D-v0"
    cfg.train.total_timesteps = int(10e6)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def normal_ppo_config():
    # Train all actuators for 10 million timesteps with PPO
    cfg = _get_default_config()
    cfg.experiment.name = "normal-ppo-10m"
    cfg.env.name = "PathFollowAuv3D-v0"
    cfg.train.algorithm = "PPO"
    cfg.train.total_timesteps = int(10e6)
    cfg.train.num_envs = (
        10  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def normal_ppo_lower_lr_config():
    # Train all actuators for 10 million timesteps with PPO
    cfg = _get_default_config()
    cfg.experiment.name = "normal-ppo-10m-lower-lr"
    cfg.env.name = "PathFollowAuv3D-v0"

    # Lower the learning rate
    cfg.hyperparam.learning_rate = 5e-5

    cfg.train.algorithm = "PPO"
    cfg.train.total_timesteps = int(10e6)
    cfg.train.num_envs = (
        10  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    return cfg


def colav_high_assistance_config():
    # Train all actuators for 1 million timesteps
    cfg = _get_default_config()
    cfg.experiment.name = "colav-high-assistance"
    cfg.env.name = "PathColavAuv3D-v0"
    cfg.train.total_timesteps = int(10e6)
    cfg.train.num_envs = (
        4  # More than one, so we use multiprocessing, but still easy to find
    )
    cfg.train.n_eval_episodes = (
        100  # Just check that it doesn't crash, we don't care about it being many
    )
    cfg.assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_rudder_only}, total_timesteps=cfg.train.total_timesteps
        )
    )
    cfg.assistance.assistant_available_probability = 0.8
    return cfg


def get_config() -> Config:
    """
    Parse the command line argument, pick the chosen config
    or none if no config is given.
    """
    available_configs = {
        "train-rudder": train_rudder_config,
        "train-elevator": train_elevator_config,
        "train-rudder-and-elevator": train_rudder_and_elevator_config,
        "train-rudder-then-elevator": train_rudder_then_elevator_config,
        "train-all-1m": train_all_1m_config,
        "train-all-5m": train_all_5m_config,
        "debug": debug_config,
        "debug-with-mlflow": debug_with_mlflow_config,
        "learn-from-assistant": learn_from_assistant_config,
        "debug-colav": debug_colav_config,
        "colav-10m": colav_10m_config,
        "colav-high-assistance": colav_high_assistance_config,
        "normal-ppo-10m": normal_ppo_config,
        "normal-ppo-10m-lower-lr": normal_ppo_lower_lr_config,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=list(available_configs.keys()),
        default="train-rudder-and-elevator",
        type=str,
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg: Config = available_configs[args.config]()
    if args.timesteps is not None:
        # Overwrite number of timesteps only if given as argument
        cfg.train.total_timesteps = int(args.timesteps)

    if args.no_mlflow:
        print("Not tracking with MLFlow!")
        cfg.train.mlflow_tracking_uri = None

    if args.num_envs is not None:
        cfg.train.num_envs = int(args.num_envs)

    print("get config train", dataclasses.asdict(cfg.train))

    return cfg
