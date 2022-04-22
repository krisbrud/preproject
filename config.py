# my_project/config.py
import argparse
import dataclasses
import os

from dataclasses import dataclass

from assisted_baselines.common.assistants.pid import PIDGains
from assisted_baselines.common.mask import BaseMaskSchedule
from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule
from utils.auv_masks import mask_rudder_and_elevator, mask_rudder_only


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
    learning_rate: float = 1e-3
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
    num_envs: int =  10  # Test if this makes a difference on total timesteps
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


@dataclass
class AuvPidConfig:
    # Gains for PID controllers for assistant
    surge = PIDGains(Kp=2, Ki=1.5, Kd=0)
    rudder = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)
    elevator = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)


@dataclass
class AssistanceConfig:
    mask_schedule: BaseMaskSchedule
    # Define parameters for PID controllers
    auv_pid: AuvPidConfig = AuvPidConfig()


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


def get_config() -> Config:
    """
    Parse the command line argument, pick the chosen config
    or none if no config is given.
    """
    available_configs = {
        "train-rudder": train_rudder_config,
        "train-elevator": train_elevator_config,
        "train-rudder-elevator": train_rudder_and_elevator_config,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=list(available_configs.keys()),
        default="train-rudder-elevator",
        type=str,
    )
    parser.add_argument("--timesteps", default=int(30e6), type=int)
    args = parser.parse_args()

    cfg: Config = available_configs[args.config]()
    cfg.train.total_timesteps = int(args.timesteps)

    print("get config train", dataclasses.asdict(cfg.train))

    return cfg
